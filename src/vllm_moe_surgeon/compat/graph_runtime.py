# SPDX-License-Identifier: Apache-2.0
"""Piecewise CUDA-graph support for the tier, pulled in out of tree.

The last capability the in-tree cache had that the plugin lacked: it runs under
**piecewise** CUDA graphs. The MoE op is a splitting point -- attention, norms and the
rest are graph-captured, the MoE op runs eager with the cache -- and the MoE op's output
is copied to a **capture-stable address** so the next graph piece reads a fixed
location. Two fork changes made that work, and neither is a method/class substitution,
so neither came for free out of tree:

1. A ``VllmConfig`` post-init that adds ``vllm::moe_forward`` to
   ``compilation_config.splitting_ops`` (and downgrades a full-graph mode to piecewise)
   when the cache is on -- so the MoE op is *carved out* of the captured region. This
   has to happen at config time, before the piecewise backend reads ``splitting_ops``;
   the fork does it inline in ``VllmConfig.__post_init__`` gated on its own
   ``offload_config.moe_expert_cache_size``. A plugin has no such field and cannot edit
   the config class, so this module **wraps** ``VllmConfig.__post_init__`` and does the
   same append gated on ``additional_config['surgeon']`` instead. The wrap is installed
   from the plugin entry point, which vLLM runs in ``EngineArgs.__post_init__`` --
   before any ``VllmConfig`` is built -- so the wrap is in place when the config that
   matters is post-initialised. (An earlier attempt injected from the runner's
   ``__init__``; that is too late -- the split points are fixed before the runner is
   constructed, and the MoE op ends up captured, which aborts with
   ``operation not permitted when stream is capturing``.)
2. ``MoERunner._maybe_stabilize_output`` -- routes the MoE output through a per-runner
   persistent buffer on PIECEWISE passes (the kernels return workspace views whose
   address moves between capture and replay). This is a genuine method substitution, so
   it goes through ``register_oot`` on the ``MoERunner`` (a ``PluggableLayer``). The
   stock ``_moe_forward`` custom op calls ``layer._forward_impl`` with no stabilisation
   wrap, so the wrap lives in an overridden ``_forward_impl``, not in the op.

Both are optional and tolerant: if either import fails, :func:`install_graph` declines
and :func:`validate` keeps requiring ``--enforce-eager``.
"""

from __future__ import annotations

from typing import Any

from .._logging import init_logger

logger = init_logger(__name__)

_MOE_SPLITTING_OPS = ("vllm::moe_forward", "vllm::moe_forward_shared")


def _inject_moe_splitting(vllm_config: Any) -> None:
    """Carve the MoE op out of the captured region, at config time.

    Mirrors the fork's ``VllmConfig.__post_init__`` block, gated on the surgeon tier
    rather than the fork-only offload config. Runs after the stock
    ``set_splitting_ops_for_v1`` has defaulted ``splitting_ops`` to the attention ops,
    so it only appends. A no-op unless the tier is on, the model is not eager, and
    piecewise compilation is actually in play.
    """
    from vllm.config import CompilationMode, CUDAGraphMode

    from .runtime import read_config_from

    config = read_config_from(vllm_config)
    if not config.enabled:
        return
    mc = getattr(vllm_config, "model_config", None)
    if mc is None or mc.enforce_eager:
        return
    cc = vllm_config.compilation_config
    if cc.cudagraph_mode == CUDAGraphMode.NONE:
        return
    if cc.mode != CompilationMode.VLLM_COMPILE:
        # Non-eager but not piecewise-compiled: the MoE op cannot be carved out, so
        # capture would swallow the cache's host code. validate() refuses this rather
        # than let it become a silent capture-address bug.
        return
    if not cc.splitting_ops:
        # No stock split points at all, so attention is inside one full graph. Reached
        # by passing an empty splitting_ops, or by pass_config.fuse_attn_quant, both of
        # which leave it empty with cudagraph_mode FULL. Injecting here would produce
        # PIECEWISE carrying only the MoE ops, and vLLM's CudagraphDispatcher asserts
        # on that at worker init -- with a message blaming its own compilation settings
        # rather than the tier that caused it. The MoE op cannot be carved out while
        # attention stays in a full graph, so decline; moe_split_active() then reports
        # False and validate() refuses with --enforce-eager and an accurate reason.
        return
    for op in _MOE_SPLITTING_OPS:
        if op not in cc.splitting_ops:
            cc.splitting_ops.append(op)
    if cc.cudagraph_mode.has_full_cudagraphs():
        logger.info(
            "out-of-tree expert cache: downgrading cudagraph_mode %s -> PIECEWISE; "
            "the MoE op must run eager between graph segments.",
            cc.cudagraph_mode,
        )
        cc.cudagraph_mode = CUDAGraphMode.PIECEWISE


def _install_splitting_patch() -> bool:
    """Wrap ``VllmConfig.__post_init__`` to inject the MoE splitting ops. Idempotent."""
    from vllm.config import VllmConfig

    if getattr(VllmConfig, "_surgeon_splitting_patched", False):
        return False
    original = VllmConfig.__post_init__

    def __post_init__(self, *args, **kwargs):  # noqa: N807
        original(self, *args, **kwargs)
        try:
            _inject_moe_splitting(self)
        except Exception as exc:  # pragma: no cover - never break config init
            logger.warning(
                "out-of-tree MoE splitting-op injection failed (%s: %s); the tier "
                "will need --enforce-eager",
                type(exc).__name__,
                exc,
            )

    VllmConfig.__post_init__ = __post_init__
    VllmConfig._surgeon_splitting_patched = True
    return True


def make_surgeon_moe_runner(base_cls: Any) -> Any:
    """Build ``SurgeonMoERunner`` subclassing the installed ``MoERunner``."""

    class SurgeonMoERunner(base_cls):  # type: ignore[valid-type,misc]
        """MoERunner that copies its output to a capture-stable address on piecewise
        passes, so the tier runs under CUDA graphs without enforce_eager. The splitting
        that makes the MoE op eager is done at config time by _inject_moe_splitting."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            from .runtime import read_config

            if not read_config().enabled:
                # The tier is off, so this class has nothing to say -- and it must be
                # careful to say nothing. Registration is unconditional (at plugin
                # time no VllmConfig exists yet, so it cannot be gated on the tier),
                # which means this subclass also wraps runs that are not ours. On the
                # disk-tier fork the base MoERunner computes exactly these attributes
                # for its OWN in-tree cache, gated on splitting_ops rather than on
                # which cache is enabled; overwriting them here would leave that cache
                # running under piecewise graphs with its stabilisation silently off,
                # which is the configuration the control run showed producing garbage
                # (one token id repeated). Supply defaults only where the base set
                # none -- stock vLLM -- so the overrides below never hit an
                # AttributeError, and otherwise leave the base's state alone.
                if not hasattr(self, "_stable_moe_output"):
                    self._stable_moe_output = False
                    self._stable_output_rows = 0
                    self._stable_output_bufs = {}
                return

            self._stable_moe_output = False
            self._stable_output_rows = 0
            self._stable_output_bufs: dict[tuple, Any] = {}

            from vllm.config import CUDAGraphMode, get_current_vllm_config

            vllm_config = get_current_vllm_config()
            mc = vllm_config.model_config
            cc = vllm_config.compilation_config
            if mc is None or mc.enforce_eager:
                return
            if cc.cudagraph_mode == CUDAGraphMode.NONE:
                return
            # The split point is injected at config time; stabilisation is needed
            # exactly when it took, i.e. the MoE op runs eager between graph pieces.
            self._stable_moe_output = "vllm::moe_forward" in (cc.splitting_ops or ())
            if self._stable_moe_output:
                self._stable_output_rows = cc.max_cudagraph_capture_size or 0
                logger.info(
                    "out-of-tree MoE output stabilization active (piecewise CUDA "
                    "graphs); the tier no longer requires --enforce-eager"
                )

        def _forward_impl(self, *args, **kwargs):
            # Stock _moe_forward calls layer._forward_impl(...) with no stabilization
            # wrap (that wrap is fork-only in the custom op), so it lives here.
            return self._maybe_stabilize_output(super()._forward_impl(*args, **kwargs))

        def _maybe_stabilize_output(self, result):
            if not self._stable_moe_output:
                return result
            from vllm.config import CUDAGraphMode
            from vllm.forward_context import (
                get_forward_context,
                is_forward_context_available,
            )

            if not is_forward_context_available():
                return result
            if get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE:
                return result
            if isinstance(result, tuple):
                return (
                    self._stable_copy(result[0], 0),
                    self._stable_copy(result[1], 1),
                )
            return self._stable_copy(result, 0)

        def _stable_copy(self, t, role: int):
            import torch

            if t.dim() != 2 or t.size(0) > self._stable_output_rows:
                logger.warning_once(
                    "moe output stabilization skipped for shape %s on a PIECEWISE "
                    "pass; downstream pieces may replay against stale addresses",
                    tuple(t.shape),
                )
                return t
            key = (role, t.size(-1), t.dtype)
            buf = self._stable_output_bufs.get(key)
            if buf is None:
                buf = torch.empty(
                    self._stable_output_rows, t.size(-1), dtype=t.dtype, device=t.device
                )
                self._stable_output_bufs[key] = buf
            out = buf[: t.size(0)]
            out.copy_(t)
            return out

    return SurgeonMoERunner


def graph_runtime_installed() -> bool:
    """Whether the OOT MoERunner is registered."""
    try:
        from vllm.model_executor.custom_op import op_registry_oot
    except ImportError:
        return False
    return "MoERunner" in op_registry_oot


def moe_split_active(vllm_config: Any) -> bool:
    """Whether the MoE op is actually carved out of the captured region *for this run*.

    Registration is not the invariant that makes non-eager safe -- the **split** is.
    ``_inject_moe_splitting`` declines silently in cases the runner cannot detect from
    the registry, the load-bearing one being a non-eager run whose compilation mode is
    not ``VLLM_COMPILE``: there is no piecewise compilation to split, so the MoE op ends
    up inside a captured graph with the cache's dynamic host code, which is exactly the
    configuration that produced garbage output (one token id repeated) in the control
    run. The fork refuses that case outright; out of tree the injection just does not
    happen, so the refusal has to be re-derived here.

    This is also precisely the condition ``SurgeonMoERunner.__init__`` uses to decide
    whether to stabilise its output, which is the point: validate must demand the same
    invariant the runtime relies on, or the two can disagree silently.
    """
    if not graph_runtime_installed():
        return False
    cc = getattr(vllm_config, "compilation_config", None)
    return bool(cc is not None and "vllm::moe_forward" in (cc.splitting_ops or ()))


def install_graph() -> bool:
    """Register the OOT MoERunner and the config-time splitting-op injection.

    Optional and tolerant: on any import failure (vLLM absent, MoERunner moved) it
    declines, and the tier falls back to requiring --enforce-eager.
    """
    try:
        from vllm.model_executor.custom_op import op_registry_oot
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    except ImportError as exc:  # pragma: no cover - vLLM absent or MoERunner moved
        logger.debug("MoERunner not importable; not installing graph runtime: %s", exc)
        return False

    _install_splitting_patch()

    if "MoERunner" in op_registry_oot:
        return False
    runner_cls = make_surgeon_moe_runner(MoERunner)
    MoERunner.register_oot(runner_cls)
    logger.info(
        "registered the out-of-tree MoERunner for piecewise CUDA-graph support"
    )
    return True
