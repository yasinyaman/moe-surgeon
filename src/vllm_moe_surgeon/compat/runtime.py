# SPDX-License-Identifier: Apache-2.0
"""The disk tier, served from outside the vLLM tree.

This is the claim the whole package exists to make good on. The prototype reaches
the same behaviour with ~800 lines of hooks threaded through upstream files; here it
is one substituted quantisation method, registered from a plugin, touching no vLLM
source.

The simplification that makes it small: the method already receives ``layer``, so
the provider can be built and stashed from inside it. No ``RoutedExperts`` subclass
is needed, which drops the most churn-prone seam in the table.

Scope, stated rather than implied. This covers the **unquantized** path with the
weights loaded normally:

- fp8 checkpoints still need ``Fp8MoEMethod`` substituted, and its cache must be
  installed before ``get_fused_moe_quant_config`` captures the scale tensors --
  getting that order wrong scales every expert by whichever one occupies its slot,
  which is wrong output from the first token and no exception.
- streaming the checkpoint straight into the store IS ported and is on by default
  whenever the disk tier is on (see :mod:`.stream_load`), so the full
  ``[num_experts, ...]`` tensors are never materialised; ``"stream_load": false``
  forces the old full-materialisation path.
- ``enforce_eager`` is no longer required: :mod:`.graph_runtime` substitutes the
  ``MoERunner`` so the MoE op is a piecewise-graph splitting point that runs eager
  and copies its output to a capture-stable address. If that substitution does not
  install (MoERunner moved), :func:`validate` falls back to requiring
  ``--enforce-eager`` -- refusing is still better than a silent capture-address bug.
- tensor parallelism is refused: the store identity carries no rank component.

Activation is deliberately *not* ``--moe-expert-cache-size``: that flag drives the
in-tree implementation. This reads ``--additional-config '{"surgeon": {...}}'``
instead, so the two can run side by side and be compared -- which is how this file
is verified at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .. import env as envs
from .._logging import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class RuntimeConfig:
    """What the out-of-tree tier was asked to do."""

    expert_cache_size: int = 0
    split: str = "token"
    store_dir: str | None = None
    ram_cache: int = 0
    fp8_store: bool = False
    hot_experts: str | None = None
    #: Write experts into the store as the checkpoint arrives, so the full
    #: ``[num_experts, ...]`` tensor is never materialised. See :mod:`.stream_load`.
    #: ``None`` means "on whenever the disk tier is on", which is the useful default:
    #: the non-streaming path pays 12.0 GiB of page-locked host memory (measured on
    #: OLMoE) and buys nothing, so nobody should land on it by accident. Set it
    #: explicitly to False to force the old path.
    stream_load: bool | None = None
    #: Compute cold experts on the host instead of fetching them over H2D.
    #: A per-machine opt-in, never a default: measured 3.7x on a discrete
    #: card (the PCIe H2D and the CPU's DRAM reads are separate pools) and a
    #: LOSS on unified memory, where they are the same pool
    #: (docs/benchmarks.md, "CPU co-execution"). Not bit-exact -- the host
    #: GEMM's reduction
    #: order differs from the fused kernel's.
    cpu_experts: bool = False
    #: Experts below this routed-token count stay on the GPU path. The T=1
    #: GEMV cliff is handled by pad-to-2 inside the CPU kernel, so 1 is the
    #: measured default; excluding T=1 empties the candidate set entirely.
    cpu_expert_min_tokens: int = 1
    #: torch intra-op threads for the host GEMMs. 0 leaves the pool alone.
    #: Applied at weight-load time, not on the first co-exec forward: torch
    #: documents set_num_threads as needing to run before parallel work
    #: starts, so a lazy call mid-serve can be ignored outright -- silently
    #: forfeiting the measured tuning (12 on the i7-12700H).
    cpu_expert_threads: int = 0

    @property
    def enabled(self) -> bool:
        return self.expert_cache_size > 0

    @property
    def use_disk(self) -> bool:
        return bool(self.store_dir) and self.ram_cache > 0

    @property
    def streams(self) -> bool:
        """Whether to intercept the loader. Unset means yes, if there is a store."""
        if self.stream_load is None:
            return self.use_disk
        return bool(self.stream_load)


def read_config() -> RuntimeConfig:
    """Read from ``additional_config['surgeon']``, falling back to the environment.

    ``additional_config`` is the channel vLLM's own comment nominates for
    out-of-tree configuration, and it reaches every process. The env fallback keeps
    the recipes in the project notes working unchanged.
    """
    try:
        from vllm.config import get_current_vllm_config

        config: Any = get_current_vllm_config()
    except Exception:  # pragma: no cover - no config in scope
        config = None
    return read_config_from(config)


def read_config_from(vllm_config: Any) -> RuntimeConfig:
    """Build the config from an explicit ``VllmConfig`` (or ``None``), env-fallback.

    Separate from :func:`read_config` so a caller holding the config directly can pass
    it in rather than relying on the ambient one. The graph path needs this: it injects
    splitting ops from inside ``VllmConfig.__post_init__``, which runs *before*
    ``get_current_vllm_config()`` is set, so the ambient lookup would miss the tier.
    """
    payload: dict[str, Any] = {}
    extra = getattr(vllm_config, "additional_config", None) or {}
    if isinstance(extra, dict):
        payload = dict(extra.get("surgeon") or {})

    def env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        return int(raw) if raw not in (None, "") else default

    return RuntimeConfig(
        expert_cache_size=int(
            payload.get(
                "expert_cache_size", env_int("VLLM_MOE_EXPERT_CACHE_SIZE", 0)
            )
        ),
        split=str(
            payload.get("split", os.environ.get("VLLM_MOE_EXPERT_CACHE_SPLIT", "token"))
        ),
        store_dir=payload.get("store_dir", os.environ.get("VLLM_MOE_DISK_STORE_DIR")),
        ram_cache=int(payload.get("ram_cache", env_int("VLLM_MOE_RAM_CACHE", 0))),
        fp8_store=bool(
            payload.get(
                "fp8_store",
                os.environ.get("VLLM_MOE_DISK_STORE_FP8", "") in ("1", "true"),
            )
        ),
        hot_experts=payload.get("hot_experts", os.environ.get("VLLM_MOE_HOT_EXPERTS")),
        stream_load=_tri_state(
            payload.get("stream_load", os.environ.get("VLLM_MOE_STREAM_LOAD")),
        ),
        # Through env.py so the documented spellings (yes/on/True) mean here
        # what its _SPEC says they mean; reading os.environ directly made
        # VLLM_MOE_CPU_EXPERTS=True parse as *off*, with no warning and no
        # mode tag -- the feature simply never engaged.
        cpu_experts=bool(payload.get("cpu_experts", envs.VLLM_MOE_CPU_EXPERTS)),
        cpu_expert_min_tokens=int(
            payload.get(
                "cpu_expert_min_tokens", envs.VLLM_MOE_CPU_EXPERT_MIN_TOKENS
            )
        ),
        cpu_expert_threads=int(
            payload.get("cpu_expert_threads", envs.VLLM_MOE_CPU_EXPERT_THREADS)
        ),
    )


def _tri_state(value: Any) -> bool | None:
    """``None`` when unset, so "unset" and "explicitly off" stay distinguishable."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes")


def check_config(config: RuntimeConfig) -> None:
    """Refuse contradictory configurations. No vLLM, no layer, no device.

    Split out from :func:`validate` deliberately: these are questions about the
    request alone, so they can be answered on a host with no vLLM installed and
    *before* an engine boots -- the same reason the job server rejects an unrunnable
    request instead of starting a pipeline that will fail three stages in.
    """
    if not config.enabled:
        return

    if config.store_dir and config.ram_cache <= 0:
        # This combination silently produced a mislabelled measurement. Three
        # boot-floor arms were recorded as "tiered, ram_cache 0" when `use_disk` had
        # turned the disk tier off entirely, leaving the provider in full-DRAM mode
        # with every expert page-locked for the process lifetime -- 31.10 GiB against
        # the genuine tiered arm's 23.40 GiB. Both modes logged the same "expert cache
        # active" line, so nothing gave it away. A store_dir asks for a disk tier and
        # ram_cache=0 refuses to build one: say so rather than silently picking the
        # more expensive of the two.
        raise ValueError(
            f"store_dir={config.store_dir!r} asks for the disk tier, but "
            "ram_cache=0 disables it -- the provider would hold every expert in "
            "pinned host memory instead, which costs more, not less. Set "
            "ram_cache >= expert_cache_size, or drop store_dir to ask for "
            "full-DRAM mode deliberately."
        )

    if config.use_disk and config.ram_cache < config.expert_cache_size:
        # The warm tier feeds the GPU slots. A pool smaller than the resident set
        # cannot hold what the GPU already has, so every eviction becomes a disk read.
        raise ValueError(
            f"ram_cache={config.ram_cache} is below expert_cache_size="
            f"{config.expert_cache_size}: the warm tier would be smaller than the "
            "resident set it feeds, so every eviction would go to disk."
        )

    if config.stream_load and not config.use_disk:
        raise ValueError(
            "stream_load writes experts into the disk store as they arrive, and no "
            "store is being built. Set store_dir and ram_cache, or drop stream_load."
        )

    if config.fp8_store and not config.use_disk:
        raise ValueError(
            "fp8_store only describes how the disk store encodes records, and no "
            "store is being built. Set store_dir and ram_cache, or drop fp8_store."
        )

    if config.cpu_experts and config.fp8_store:
        raise ValueError(
            "cpu_experts computes offloaded experts from the host records, and an "
            "fp8_store's records hold fp8 bytes plus scales -- the CPU path would "
            "have to dequantize every forward, which is not implemented. Drop "
            "fp8_store, or drop cpu_experts."
        )

    if config.cpu_experts:
        if envs.VLLM_MOE_ZERO_COPY:
            raise ValueError(
                "VLLM_MOE_ZERO_COPY makes the pinned pool the set the kernel "
                "reads directly, so there is no cold host tier left to compute "
                "from and an eviction is kernel-visible. Drop one of the two."
            )

    if config.cpu_expert_min_tokens < 1:
        raise ValueError(
            f"cpu_expert_min_tokens={config.cpu_expert_min_tokens} is below 1. "
            "Single-token experts are handled by pad-to-2 inside the CPU kernel "
            "(measured: 1000-1500 us raw vs ~300 us padded); the knob only "
            "raises the bar, it cannot go below one token."
        )


def validate(config: RuntimeConfig, layer: Any) -> None:
    """Refuse the combinations this port does not implement, by name.

    Each of these is a case where proceeding would produce wrong output rather than
    an error, so they are checked before a single weight is allocated. Pure-config
    contradictions are delegated to :func:`check_config`.
    """
    if not config.enabled:
        return

    check_config(config)

    moe = layer.moe_config
    top_k = moe.experts_per_token
    capacity = min(config.expert_cache_size, layer.local_num_experts)
    if capacity < top_k and config.split != "expert":
        raise ValueError(
            f"expert_cache_size={config.expert_cache_size} gives {capacity} slots, "
            f"fewer than the {top_k} experts one token routes to. Raise it, or set "
            'split="expert", which runs a token across several launches.'
        )
    if moe.has_bias:
        raise ValueError(
            "the expert cache does not support MoE layers with bias terms: the "
            "kernel receives w1/w2 only. Layer "
            f"{getattr(layer, 'layer_name', '?')}."
        )

    parallel = moe.moe_parallel_config
    if parallel.use_ep:
        raise ValueError(
            f"the expert cache is incompatible with expert parallelism "
            f"(ep_size={parallel.ep_size}): expert ids would be rank-local."
        )
    if parallel.dp_size > 1 or parallel.is_sequence_parallel:
        raise ValueError(
            "the expert cache is incompatible with data or sequence parallelism."
        )
    if getattr(parallel, "tp_size", 1) > 1:
        raise ValueError(
            f"the expert cache does not support tensor parallelism "
            f"(tp_size={parallel.tp_size}): each rank holds a different shard of "
            "every expert at identical shapes, but the store identity is "
            "{model, revision, layer} with no rank component, so ranks would share "
            "one store file and rank 1 would serve rank 0's shard. Refused rather "
            "than silently served wrong."
        )

    if config.cpu_experts:
        # Layer-only checks, before anything touches vLLM: testable anywhere.
        activation = getattr(layer, "activation", None)
        # A string in older vLLMs, a MoEActivation enum in newer ones -- the
        # enum's value is the string. Caught live: the first laptop boot was
        # refused for using <MoEActivation.SILU: 'silu'>.
        name = getattr(activation, "value", activation)
        if name != "silu":
            raise ValueError(
                "cpu_experts re-implements the expert activation on the host, "
                f"and only 'silu' has a verified CPU twin; this layer uses "
                f"{activation!r}. Drop cpu_experts for this model."
            )
        if getattr(layer, "apply_router_weight_on_input", False):
            raise ValueError(
                "cpu_experts applies the gate weights after the expert, and "
                "this model folds them into the input (top_k==1 style) -- the "
                "CPU path would double-apply them. Refused rather than "
                "silently wrong."
            )

    from vllm.config import CUDAGraphMode, get_current_vllm_config

    from .graph_runtime import moe_split_active

    vllm_config = get_current_vllm_config()
    graphs_on = (
        not vllm_config.model_config.enforce_eager
        and vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
    )
    if graphs_on and not moe_split_active(vllm_config):
        raise ValueError(
            "the out-of-tree expert cache requires --enforce-eager unless the MoE op "
            "is carved out of the captured region. The out-of-tree MoERunner does "
            "that by adding vllm::moe_forward to compilation_config.splitting_ops at "
            "config time, which needs piecewise compilation (compilation mode "
            "VLLM_COMPILE); it is not in splitting_ops for this run, so the MoE op "
            "would be captured together with the cache's dynamic host code. That "
            "fails as wrong output rather than as an error -- a captured graph "
            "replays against a stale workspace address, which reproduces as the same "
            "token repeated -- so it is refused. Pass --enforce-eager, or drop the -O "
            "override that disabled piecewise compilation."
        )
    if getattr(layer, "_moe_expert_cache_size", 0) > 0:
        raise ValueError(
            "both the in-tree expert cache (--moe-expert-cache-size) and the "
            "out-of-tree one (additional_config.surgeon) are enabled. Pick one; "
            "they would each try to own the same expert weights."
        )

    if config.cpu_experts and graphs_on:
        raise ValueError(
            "cpu_experts joins a host-computed partial into the MoE output "
            "on every forward; a captured graph replays without the host, "
            "and the piecewise split copies the MoE output to a "
            "capture-stable address before the join could land. Pass "
            "--enforce-eager, or drop cpu_experts."
        )


def _identity(layer: Any) -> dict[str, str]:
    from vllm.config import get_current_vllm_config

    model_config = get_current_vllm_config().model_config
    return {
        "model": model_config.model,
        "revision": str(model_config.revision),
        "layer": layer.layer_name,
    }


def build_provider(layer: Any, config: RuntimeConfig) -> Any:
    """Construct the provider for one layer and release the full weight tensors.

    The order matters: the provider must exist before the quant config is built,
    because the quant config captures scale tensors, and the full ``w13``/``w2``
    parameters must be released only after the provider has copied what it needs.
    """
    import torch

    from ..store import CachedWeightProvider, DiskExpertStore

    capacity = min(config.expert_cache_size, layer.local_num_experts)
    disk_store = None
    streamer = getattr(layer, "_surgeon_streamer", None)
    if streamer is not None:
        # Already written, record by record, while the checkpoint was arriving. There
        # are no full weight tensors left to build a store *from* -- that is the point.
        disk_store = streamer.finalize()
        layer._surgeon_streamer = None
    elif config.use_disk:
        assert config.store_dir is not None
        os.makedirs(config.store_dir, exist_ok=True)
        from ..surgery.tiered import store_path

        disk_store = DiskExpertStore.build(
            store_path(config.store_dir, layer.layer_name),
            layer.w13_weight.data,
            layer.w2_weight.data,
            identity=_identity(layer),
            quantize_fp8=config.fp8_store,
        )

    # Every argument by keyword: capacity is the *first* positional parameter, not
    # the weights, so a positional call silently binds the wrong things.
    provider = CachedWeightProvider(
        capacity=capacity,
        w13_weight=layer.w13_weight.data,
        w2_weight=layer.w2_weight.data,
        w13_scale=None,
        w2_scale=None,
        split=config.split,
        ram_capacity=config.ram_cache if disk_store is not None else 0,
        disk_store=disk_store,
        layer_name=layer.layer_name,
    )

    if config.hot_experts:
        from .. import hot_experts as hot

        try:
            hints = hot.load(config.hot_experts)
            priors = hints.for_layer_name(layer.layer_name)
            for policy in (provider._gpu_policy, provider._ram_policy):
                if policy is not None:
                    hot.seed_policy(policy, priors)
        except Exception as exc:
            logger.warning("could not seed residency hints: %s", exc)

    # The full tensors are the point of the exercise: release them so the layer
    # holds only the cache's slot-sized buffers.
    from vllm.model_executor.utils import replace_parameter

    replace_parameter(layer, "w13_weight", torch.empty(0))
    replace_parameter(layer, "w2_weight", torch.empty(0))
    return provider


#: Row count above which a forward is treated as a prefill and left alone. See
#: :func:`_warn_if_oversubscribed` for why the distinction decides whether the warning
#: is worth having. Set well above any plausible decode batch rather than just above the
#: one that was benchmarked: at 32 this silenced exactly the wide-batch serving case
#: where oversubscription costs the most, which is the opposite of the intent. A prefill
#: forward carries hundreds to thousands of token rows, so the two populations stay
#: separated at this value.
_DECODE_ROWS = 512

#: How many decode forwards a layer is inspected for before the check retires itself.
#: A diagnostic that never stops looking is a tax on the configuration it is trying to
#: praise: ``topk_ids.unique()`` synchronises with the device, and on a correctly sized
#: cache the check would otherwise run on every decode forward of every MoE layer for
#: the process lifetime -- recomputing, on the fast path, a value the provider does not
#: even need there. A handful of probes is enough to catch a batch that oversubscribes
#: while bounding the cost to a constant.
_WARN_PROBES = 8


def _warn_if_oversubscribed(layer: Any, provider: Any, topk_ids: Any) -> None:
    """Say so, once, when a forward routes to more experts than there are slots.

    A cache smaller than the batch's per-layer *union* is the most expensive
    misconfiguration this system has, and until now it was completely silent. Measured
    on OLMoE at batch 8, where the decode union is 35.3 mean / 46 max: 24 slots decode
    at 55.3 tok/s and 48 slots at 143.0 tok/s -- 2.6x apart, for one integer, with
    nothing in the logs to distinguish them.

    Where that 2.6x comes from, measured rather than assumed, because the obvious guess
    is wrong: **~76-89% of it is simply fetching more bytes.** Loads fall from 26.4 to
    4.7 per layer per step, and a load on the bf16 path is 217 us of host->device DMA
    (12.58 MB at 57.9 GB/s) with no other term hiding in it. Only ~11-24% is the chunk
    split itself, and the split's real cost is *not* re-fetching experts over the bus --
    that was measured at 2.5% of misses. It is the GPU re-reading resident weights out
    of its own DRAM once per chunk (the sum of the chunk unions is ~50 records per layer
    against a union of ~37), plus extra kernel launches. Python planning is ~32 us per
    ``prepare()``, and removing the blocking mapping upload entirely moved throughput by
    -1.4%. So the actionable advice is about bytes and residency, not about syncs.

    So this is a diagnostic, not a policy: it changes nothing about what runs. It is
    deliberately a warning rather than a refusal, because a too-small cache is exactly
    what someone serving on a device that cannot hold the working set has to do -- the
    split is the feature that makes that possible. What is not acceptable is paying for
    it without being told.

    Only **decode-shaped** forwards are considered, and that restriction is the whole
    difference between a useful warning and a useless one. A prefill routes hundreds of
    tokens at once, so its union approaches the full expert set -- measured 56-62 of 64
    on OLMoE -- and a check that fired on it would demand ``capacity == num_experts``
    from every configuration, including the one that measured fastest. It would also be
    advice about the wrong cost: a prefill is paid once per request, while a decode step
    repeats for every token generated, which is where the 2.6x was measured. The row
    count is the heuristic (a decode step contributes one row per sequence); it is
    deliberately generous, because a decode batch wide enough to exceed it genuinely
    does want a cache near the full expert set.
    """
    if getattr(layer, "_surgeon_capacity_warned", False):
        return
    capacity = getattr(provider, "capacity", 0)
    if not capacity:
        return
    if getattr(layer, "_surgeon_config", None) is not None and (
        layer._surgeon_config.cpu_experts
    ):
        # Under co-execution both halves of this message are false: the cold
        # experts were computed on the host with no H2D at all, and taking
        # every miss removes the chunk split. Worse, its remedy -- raise
        # expert_cache_size toward the union -- shrinks the co-exec headroom
        # (ram_cache - expert_cache_size), i.e. it advises disabling most of
        # the mode. The oversubscription it describes is the mode's premise.
        layer._surgeon_capacity_warned = True
        return
    if getattr(provider, "split", None) == "expert":
        # An explicit expert split IS the acknowledgement that the device cannot
        # hold the working set -- the exact case the message's last sentence
        # excuses. Warning anyway printed sixteen layers of advice to raise a
        # setting the configuration had already, deliberately, declined to
        # raise. Observed live on the small-device config this split exists for.
        # The provider carries the split it was built with; read_config() here
        # would miss it, because at forward time there is no ambient vLLM config
        # (get_current_vllm_config raises) and the env fallback cannot see
        # additional_config -- the documented channel.
        layer._surgeon_capacity_warned = True
        return
    try:
        rows = int(topk_ids.shape[0])
        if rows > _DECODE_ROWS:
            return
        union = int(topk_ids.unique().numel())
    except Exception:  # pragma: no cover - never break a forward to warn
        return
    if union <= capacity:
        # Retire after a bounded number of clean probes. Without this the check would
        # keep synchronising with the device on every decode forward, forever, on
        # precisely the configuration it has nothing to complain about.
        probes = getattr(layer, "_surgeon_capacity_probes", 0) + 1
        layer._surgeon_capacity_probes = probes
        if probes >= _WARN_PROBES:
            layer._surgeon_capacity_warned = True
        return
    layer._surgeon_capacity_warned = True
    logger.warning(
        "expert_cache_size=%d is below the expert union of a decode step (%d) on %s: "
        "most of these experts are re-fetched over the host->device link every step, "
        "and each step is additionally split into chunks that re-read resident weights "
        "and relaunch the expert kernel. Raising the cache toward the union cuts the "
        "bytes moved per token (measured 2.6x on OLMoE at batch 8, same tokens; ~80%% "
        "of it fewer fetches, the rest the split). Keep it smaller only if the device "
        "cannot hold that many slots -- which is the case the tier exists for.",
        capacity,
        union,
        getattr(layer, "layer_name", "?"),
    )


def install() -> bool:
    """Register the out-of-tree method. Returns whether it took.

    Called from the plugin, which vLLM may load more than once, so this is
    idempotent -- ``register_oot`` asserts on a duplicate name and that assertion
    would take the engine down.
    """
    try:
        from vllm.model_executor.custom_op import op_registry_oot
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )
    except ImportError as exc:  # pragma: no cover - vLLM absent
        logger.debug("vLLM not importable, not installing the runtime: %s", exc)
        return False

    # The registry is keyed by the *class* name, because CustomOp.__new__ looks up
    # cls.__name__ -- not by the op name passed to @CustomOp.register. Registering
    # under "unquantized_fused_moe" put the class somewhere nothing ever looks, so
    # the substitution silently no-opped and every layer ran the stock path. Tokens
    # still matched, which is exactly why that was not evidence of anything.
    OOT_KEY = UnquantizedFusedMoEMethod.__name__
    if OOT_KEY in op_registry_oot:
        return False

    class SurgeonUnquantizedMoEMethod(UnquantizedFusedMoEMethod):
        """Unquantized MoE with expert weights served from the tier."""

        def create_weights(self, layer, num_experts, hidden_size,
                           intermediate_size_per_partition, params_dtype,
                           **extra_weight_attrs):
            config = read_config()
            if not config.enabled:
                return super().create_weights(
                    layer, num_experts, hidden_size,
                    intermediate_size_per_partition, params_dtype,
                    **extra_weight_attrs,
                )

            validate(config, layer)
            layer._surgeon_config = config
            # Allocate in pinned host memory so loading never needs device
            # capacity for the full expert set. device="cpu" is explicit because
            # vLLM loads under a torch.device("cuda") context, which pin_memory()
            # alone would not override.
            import torch
            from vllm.model_executor.utils import set_weight_attrs

            up_dim = (
                2 * intermediate_size_per_partition
                if self.moe.is_act_and_mul
                else intermediate_size_per_partition
            )

            streamer = None
            if config.streams and not self.moe.is_act_and_mul:
                # The record layout is w13 = [2 * inter, hidden] with gate and up in
                # known halves. A layer without act-and-mul has no w3 shard to place,
                # so the halves would be undefined rather than merely wrong-sized.
                # Asked for explicitly that is an error; reached by the default it is
                # a reason to take the other path.
                if config.stream_load:
                    raise ValueError(
                        "stream_load assumes the gate/up record layout, and this "
                        f"layer ({layer.layer_name}) is not act-and-mul."
                    )
                logger.warning(
                    "%s is not act-and-mul, so the loader cannot be streamed; "
                    "falling back to materialising the full expert set, which costs "
                    "page-locked host memory for every expert",
                    layer.layer_name,
                )
            elif config.streams:
                from .stream_load import StreamLoader

                identity = _identity(layer)
                streamer = StreamLoader.open(
                    store_dir=config.store_dir,
                    layer_name=layer.layer_name,
                    num_experts=num_experts,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size_per_partition,
                    params_dtype=params_dtype,
                    fp8=config.fp8_store,
                    model=identity["model"],
                    revision=identity["revision"],
                    map_expert_id=getattr(
                        layer, "_map_global_expert_id_to_local_expert_id", None
                    ),
                )
                layer._surgeon_streamer = streamer

            def empty(*shape):
                if streamer is not None:
                    # Zero experts: nothing to fill, nothing for
                    # device_loading_context to hoist, nothing to release. The
                    # trailing dims survive because the provider sizes its device
                    # buffers from shape[1:].
                    return torch.empty(0, *shape[1:], dtype=params_dtype, device="cpu")
                host = torch.empty(*shape, dtype=params_dtype, device="cpu")
                return host.pin_memory()

            attrs = dict(extra_weight_attrs)
            if streamer is not None:
                inner = attrs.get("weight_loader")
                if inner is None:
                    raise ValueError(
                        "stream_load needs the loader vLLM would have used, and "
                        "create_weights received no weight_loader to wrap"
                    )
                attrs["weight_loader"] = streamer.wrap(inner)

            w13 = torch.nn.Parameter(
                empty(num_experts, up_dim, hidden_size), requires_grad=False
            )
            layer.register_parameter("w13_weight", w13)
            set_weight_attrs(w13, attrs)
            w2 = torch.nn.Parameter(
                empty(num_experts, hidden_size, intermediate_size_per_partition),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight", w2)
            set_weight_attrs(w2, attrs)

        def process_weights_after_loading(self, layer) -> None:
            config = getattr(layer, "_surgeon_config", None)
            if config is None or not config.enabled:
                return super().process_weights_after_loading(layer)

            from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (  # noqa: E501
                make_unquantized_moe_kernel,
            )

            # _setup_kernel is skipped on purpose: it shuffles the full weights
            # into runtime format on device, which is exactly what the tier exists
            # to avoid. The kernel is built by hand instead.
            layer._surgeon_provider = build_provider(layer, config)
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            if getattr(self, "moe_kernel", None) is None:
                assert self.experts_cls is not None
                self.moe_kernel = make_unquantized_moe_kernel(
                    quant_config=self.moe_quant_config,
                    moe_config=self.moe,
                    backend=self.unquantized_backend,
                    experts_cls=self.experts_cls,
                    routing_tables=layer._expert_routing_tables(),
                )
            # Naming the mode is not cosmetic: full-DRAM and disk-backed differ by
            # whether the full expert set stays page-locked for the process lifetime,
            # and an earlier measurement was misattributed because one line covered
            # both. The prototype logged the distinction; the port had dropped it.
            logger.info(
                "out-of-tree expert cache active on %s: %d slots of %d experts, "
                "%s",
                layer.layer_name,
                min(config.expert_cache_size, layer.local_num_experts),
                layer.local_num_experts,
                (
                    f"disk-backed (ram_cache={config.ram_cache}"
                    f"{', fp8' if config.fp8_store else ''}"
                    f"{', streamed' if config.streams else ''}"
                    f"{', cpu-coexec' if config.cpu_experts else ''})"
                    if config.use_disk
                    else "full-DRAM: every expert stays pinned in host memory"
                    + (", cpu-coexec" if config.cpu_experts else "")
                ),
            )
            if config.cpu_experts:
                # Loud on purpose: this mode is not bit-exact, and it is a
                # per-machine decision -- measured 3.7x on a discrete card
                # and a loss on unified memory (docs/benchmarks.md,
                # "CPU co-execution").
                if config.cpu_expert_threads > 0:
                    # Here, not on the first co-exec forward: torch documents
                    # set_num_threads as needing to run before parallel work
                    # begins, so a lazy call mid-serve can be ignored with
                    # only a warning -- silently forfeiting the tuning the
                    # gates measured while the gates, which set it in a fresh
                    # process, keep showing the win.
                    import torch

                    torch.set_num_threads(config.cpu_expert_threads)
                logger.warning(
                    "cpu_experts is ON for %s: cold experts are computed on "
                    "the host and joined in fp32 -- NOT bit-exact with the "
                    "fused kernel, and only a win on machines where CPU DRAM "
                    "and H2D are separate bandwidth pools (discrete GPUs).",
                    layer.layer_name,
                )

        def apply(self, layer, x, topk_weights, topk_ids, shared_experts,
                  shared_experts_input):
            provider = getattr(layer, "_surgeon_provider", None)
            if provider is None:
                return super().apply(
                    layer, x, topk_weights, topk_ids, shared_experts,
                    shared_experts_input,
                )

            from ..store import run_with_cpu_coexec, run_with_expert_cache

            _warn_if_oversubscribed(layer, provider, topk_ids)

            def run(result, rows, include_shared):
                assert self.moe_kernel is not None
                return self.moe_kernel.apply(
                    hidden_states=x[rows],
                    w1=result.w1,
                    w2=result.w2,
                    topk_weights=topk_weights[rows],
                    topk_ids=topk_ids[rows],
                    activation=layer.activation,
                    apply_router_weight_on_input=layer.apply_router_weight_on_input,
                    global_num_experts=layer.global_num_experts,
                    expert_map=result.expert_map,
                    # Shared experts belong to the forward, not to one chunk.
                    shared_experts=shared_experts if include_shared else None,
                    shared_experts_input=(
                        shared_experts_input[rows]
                        if include_shared and shared_experts_input is not None
                        else None
                    ),
                )

            cfg = getattr(layer, "_surgeon_config", None)
            if cfg is not None and cfg.cpu_experts:
                out = run_with_cpu_coexec(
                    provider, x, topk_ids, topk_weights, run,
                    min_tokens=cfg.cpu_expert_min_tokens,
                )
                if out is not None:
                    return out
            return run_with_expert_cache(provider, topk_ids, run)

    # Called on the class being replaced, so register_oot's default reg_name is
    # that class's name -- the key __new__ will look up.
    UnquantizedFusedMoEMethod.register_oot(SurgeonUnquantizedMoEMethod)
    assert op_registry_oot.get(OOT_KEY) is SurgeonUnquantizedMoEMethod, (
        f"registered under the wrong key: {sorted(op_registry_oot)}"
    )
    logger.info("registered the out-of-tree unquantized MoE method as %s", OOT_KEY)

    # The fp8 path is separate: Fp8MoEMethod is not a CustomOp, so it goes through a
    # shadowing quantization config instead of register_oot. Optional -- an fp8
    # checkpoint may never be served, and its internals are the churn-prone ones.
    try:
        from .fp8_runtime import install_fp8

        install_fp8()
    except Exception as exc:  # pragma: no cover - fp8 internals moved
        logger.debug("fp8 runtime not installed: %s", exc)

    # The graph path is separate again: MoERunner IS a PluggableLayer, so it goes
    # through register_oot like the unquantized method, but keyed under "MoERunner".
    # Optional -- if it does not install, validate() keeps requiring --enforce-eager.
    try:
        from .graph_runtime import install_graph

        install_graph()
    except Exception as exc:  # pragma: no cover - MoERunner internals moved
        logger.debug("graph runtime not installed: %s", exc)
    return True
