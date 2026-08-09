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
- streaming the checkpoint straight into the store is not ported, so the full
  ``[num_experts, ...]`` tensors are materialised during load.
- ``enforce_eager`` is required. Running the MoE op inside a CUDA graph needs the
  output-address stabilisation the prototype added to ``MoERunner``, and refusing
  is better than a silent capture-address bug.

Activation is deliberately *not* ``--moe-expert-cache-size``: that flag drives the
in-tree implementation. This reads ``--additional-config '{"surgeon": {...}}'``
instead, so the two can run side by side and be compared -- which is how this file
is verified at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

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
    stream_load: bool = False

    @property
    def enabled(self) -> bool:
        return self.expert_cache_size > 0

    @property
    def use_disk(self) -> bool:
        return bool(self.store_dir) and self.ram_cache > 0


def read_config() -> RuntimeConfig:
    """Read from ``additional_config['surgeon']``, falling back to the environment.

    ``additional_config`` is the channel vLLM's own comment nominates for
    out-of-tree configuration, and it reaches every process. The env fallback keeps
    the recipes in the project notes working unchanged.
    """
    payload: dict[str, Any] = {}
    try:
        from vllm.config import get_current_vllm_config

        config = get_current_vllm_config()
        extra = getattr(config, "additional_config", None) or {}
        if isinstance(extra, dict):
            payload = dict(extra.get("surgeon") or {})
    except Exception:  # pragma: no cover - no config in scope
        payload = {}

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
        stream_load=bool(
            payload.get(
                "stream_load",
                os.environ.get("VLLM_MOE_STREAM_LOAD", "") in ("1", "true"),
            )
        ),
    )


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

    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    if not vllm_config.model_config.enforce_eager:
        raise ValueError(
            "the out-of-tree expert cache requires --enforce-eager. Running the "
            "MoE op inside a CUDA graph needs output-address stabilisation that "
            "this port does not implement, and the failure mode is a silent "
            "capture-address bug rather than an error."
        )
    if getattr(layer, "_moe_expert_cache_size", 0) > 0:
        raise ValueError(
            "both the in-tree expert cache (--moe-expert-cache-size) and the "
            "out-of-tree one (additional_config.surgeon) are enabled. Pick one; "
            "they would each try to own the same expert weights."
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
            if config.stream_load:
                if not self.moe.is_act_and_mul:
                    # The record layout is w13 = [2 * inter, hidden] with gate and up
                    # in known halves. A layer without act-and-mul has no w3 shard to
                    # place, so the halves would be undefined rather than merely
                    # wrong-sized.
                    raise ValueError(
                        "stream_load assumes the gate/up record layout, and this "
                        f"layer ({layer.layer_name}) is not act-and-mul."
                    )
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
                    f"{', streamed' if config.stream_load else ''})"
                    if config.use_disk
                    else "full-DRAM: every expert stays pinned in host memory"
                ),
            )

        def apply(self, layer, x, topk_weights, topk_ids, shared_experts,
                  shared_experts_input):
            provider = getattr(layer, "_surgeon_provider", None)
            if provider is None:
                return super().apply(
                    layer, x, topk_weights, topk_ids, shared_experts,
                    shared_experts_input,
                )

            from ..store import run_with_expert_cache

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

            return run_with_expert_cache(provider, topk_ids, run)

    # Called on the class being replaced, so register_oot's default reg_name is
    # that class's name -- the key __new__ will look up.
    UnquantizedFusedMoEMethod.register_oot(SurgeonUnquantizedMoEMethod)
    assert op_registry_oot.get(OOT_KEY) is SurgeonUnquantizedMoEMethod, (
        f"registered under the wrong key: {sorted(op_registry_oot)}"
    )
    logger.info("registered the out-of-tree unquantized MoE method as %s", OOT_KEY)
    return True
