# SPDX-License-Identifier: Apache-2.0
"""The fp8 disk tier, served from outside the vLLM tree.

The unquantized path is one substituted method (``runtime.py``). fp8 cannot be that
clean, and this module is the price. ``Fp8MoEMethod`` is not a ``CustomOp``, so
``register_oot`` cannot reach it; the route is a ``Fp8Config`` subclass registered
under ``"fp8"`` via ``register_quantization_config``, whose ``get_quant_method``
returns our method for MoE layers. And the cache-install ordering lives inside
``Fp8MoEMethod._setup_kernel`` -- the cache MUST take over the weights and repoint the
scale parameters *before* ``get_fused_moe_quant_config`` captures them, or the kernel
holds expert-indexed scales the cache indexes by slot, scaling every expert by
whichever one occupies its slot, with no exception. So the override copies that method's
body (from vLLM) and swaps exactly one line -- the fork-only
``layer._maybe_init_expert_lru_cache`` -- for :func:`build_fp8_provider`.

Everything else is stock vLLM: the shuffle-to-runtime-format, the quant-config build and
the kernel build are the same helpers the in-tree method uses.
"""

from __future__ import annotations

import os
from typing import Any

from .._logging import init_logger

logger = init_logger(__name__)


def refuse_unverified_fp8_scheme(method: Any) -> None:
    """Refuse fp8 schemes the OOT tier has not been verified against.

    fp8's failure mode is silent -- a mishandled per-expert scale scales every expert
    by whichever one occupies its slot, with no exception -- so an unverified scheme is
    refused loudly rather than served wrong. Verified: per-tensor / per-channel
    **static** fp8 (``weight_scale``). Refused: **block-quantized** fp8
    (``weight_block_size`` set,
    ``weight_scale_inv`` block scales), whose scale is a per-expert *block tensor* the
    slot-indexed cache cannot keep in sync. ``dynamic`` activation is warned, not
    refused: it rescales inputs, not the per-expert weight scales the cache remaps, so
    it is lower risk -- but still unverified, so the output should be checked.
    """
    if getattr(method, "block_quant", False) or getattr(
        method, "weight_block_size", None
    ):
        raise ValueError(
            "the out-of-tree fp8 tier does not support block-quantized fp8 "
            "(weight_block_size set / weight_scale_inv): its per-expert block scales "
            "cannot be slot-indexed by the cache, and the mishandling would be silent. "
            "Verified only for per-tensor/per-channel static fp8. Refusing."
        )
    scheme = getattr(getattr(method, "quant_config", None), "activation_scheme", None)
    if scheme not in (None, "static"):
        logger.warning(
            "fp8 activation_scheme=%r is unverified with the out-of-tree tier (only "
            "'static' is verified); proceeding, but check the output against the "
            "untiered baseline",
            scheme,
        )


def build_fp8_provider(method: Any, layer: Any, config: Any) -> Any:
    """Install the cache for an already-fp8 layer and release its full weights.

    Reproduces the fp8 arm of the in-tree ``_maybe_init_expert_lru_cache``: build the
    store from the fp8 weights and their per-expert scales (no requantization -- the
    checkpoint already ships fp8), construct the provider, **repoint the per-expert
    scale parameters at the cache's slot-indexed buffers before the quant config reads
    them**, and release the full weight tensors.
    """
    import torch
    from vllm.model_executor.utils import replace_parameter

    from ..store import CachedWeightProvider, DiskExpertStore
    from .runtime import _identity

    scale_suffix = method.weight_scale_name  # "weight_scale" or "weight_scale_inv"
    w13_scale_name = f"w13_{scale_suffix}"
    w2_scale_name = f"w2_{scale_suffix}"
    w13_scale = getattr(layer, w13_scale_name, None)
    w2_scale = getattr(layer, w2_scale_name, None)

    # Only expert-indexed scales need to travel with the cache. A per-tensor or global
    # scale is slot-agnostic and is left on the layer, exactly as in-tree.
    n = layer.local_num_experts
    per_expert = (
        w13_scale is not None
        and w2_scale is not None
        and w13_scale.size(0) == n
        and w2_scale.size(0) == n
    )
    if not per_expert:
        w13_scale = w2_scale = None
    elif w13_scale.ndim > 2 or w2_scale.ndim > 2:
        # Defence in depth behind refuse_unverified_fp8_scheme: a per-expert scale
        # with a block dimension (ndim>2) is a block-quant scale the cache cannot
        # slot-index. Refuse rather than mis-scale silently.
        raise ValueError(
            f"per-expert fp8 weight scales are block-shaped "
            f"(w13 ndim {w13_scale.ndim}); the slot-indexed cache cannot keep block "
            "scales in sync. Refusing."
        )

    capacity = min(config.expert_cache_size, n)
    disk_store = None
    if config.use_disk:
        assert config.store_dir is not None
        os.makedirs(config.store_dir, exist_ok=True)
        from ..surgery.tiered import store_path

        disk_store = DiskExpertStore.build(
            store_path(config.store_dir, layer.layer_name),
            layer.w13_weight.data,
            layer.w2_weight.data,
            w13_scale,
            w2_scale,
            identity=_identity(layer),
            quantize_fp8=False,  # the checkpoint's weights are already fp8
        )

    provider = CachedWeightProvider(
        capacity=capacity,
        w13_weight=layer.w13_weight.data,
        w2_weight=layer.w2_weight.data,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
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
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not seed residency hints: %s", exc)

    # Repoint the scale parameters at the cache's slot-indexed buffers, BEFORE the quant
    # config captures them. The assertion is the guard the whole file exists for.
    if getattr(provider, "buf_w13_scale", None) is not None:
        assert provider.buf_w2_scale is not None
        assert method.moe_quant_config is None, (
            "expert cache installed after the fp8 quant config was built; the kernel "
            "would hold expert-indexed scales the cache cannot keep in sync"
        )
        replace_parameter(layer, w13_scale_name, provider.buf_w13_scale)
        replace_parameter(layer, w2_scale_name, provider.buf_w2_scale)

    replace_parameter(layer, "w13_weight", torch.empty(0))
    replace_parameter(layer, "w2_weight", torch.empty(0))
    return provider


def make_surgeon_fp8_method(base_cls: Any) -> Any:
    """Build ``SurgeonFp8MoEMethod`` subclassing the installed ``Fp8MoEMethod``."""

    class SurgeonFp8MoEMethod(base_cls):  # type: ignore[valid-type,misc]
        """Fp8 MoE method whose experts are served from the tier."""

        def _setup_kernel(self, layer, w13, w2, w13_scale, w2_scale,
                          w13_input_scale, w2_input_scale):
            from .runtime import read_config, validate

            config = read_config()
            if not config.enabled:
                return super()._setup_kernel(
                    layer, w13, w2, w13_scale, w2_scale,
                    w13_input_scale, w2_input_scale,
                )

            validate(config, layer)
            refuse_unverified_fp8_scheme(self)

            # --- copied from Fp8MoEMethod._setup_kernel, one line swapped -----------
            from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
                Fp8MoeBackend,
                convert_to_fp8_moe_kernel_format,
                make_fp8_moe_kernel,
            )
            from vllm.model_executor.utils import replace_parameter

            w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
                fp8_backend=self.fp8_backend,
                layer=layer,
                w13=w13,
                w2=w2,
                w13_scale=w13_scale,
                w2_scale=w2_scale,
                w13_input_scale=w13_input_scale,
                w2_input_scale=w2_input_scale,
            )
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, f"w13_{self.weight_scale_name}", w13_scale)
            replace_parameter(layer, f"w2_{self.weight_scale_name}", w2_scale)
            if self.fp8_backend == Fp8MoeBackend.AITER:
                layer.w13_weight.is_shuffled = True
                layer.w2_weight.is_shuffled = True

            # The one swapped line: our provider build in place of the fork-only
            # layer._maybe_init_expert_lru_cache(self.weight_scale_name).
            layer._surgeon_provider = build_fp8_provider(self, layer, config)

            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.moe_quant_config is not None
            assert self.experts_cls is not None
            self.moe_kernel = make_fp8_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                fp8_backend=self.fp8_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
                layer=layer,
            )
            logger.info(
                "out-of-tree fp8 expert cache active on %s: %d slots of %d experts, "
                "%s",
                layer.layer_name,
                min(config.expert_cache_size, layer.local_num_experts),
                layer.local_num_experts,
                (
                    f"disk-backed (ram_cache={config.ram_cache})"
                    if config.use_disk
                    else "full-DRAM: fp8 experts stay pinned in host memory"
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
            from .runtime import _warn_if_oversubscribed

            _warn_if_oversubscribed(layer, provider, topk_ids)

            def run(result, rows, include_shared):
                assert self.moe_kernel is not None
                return self.moe_kernel.apply(
                    x[rows],
                    result.w1,
                    result.w2,
                    topk_weights[rows],
                    topk_ids[rows],
                    activation=layer.activation,
                    global_num_experts=layer.global_num_experts,
                    expert_map=result.expert_map,
                    apply_router_weight_on_input=layer.apply_router_weight_on_input,
                    shared_experts=shared_experts if include_shared else None,
                    shared_experts_input=(
                        shared_experts_input[rows]
                        if include_shared and shared_experts_input is not None
                        else None
                    ),
                )

            return run_with_expert_cache(provider, topk_ids, run)

    return SurgeonFp8MoEMethod


def make_surgeon_fp8_config(config_cls: Any, method_cls: Any) -> Any:
    """Build ``SurgeonFp8Config`` subclassing the installed ``Fp8Config``."""

    class SurgeonFp8Config(config_cls):  # type: ignore[valid-type,misc]
        """Fp8 config that hands MoE layers our cache-aware method."""

        def get_quant_method(self, layer, prefix):
            from vllm.model_executor.layers.fused_moe.routed_experts import (
                RoutedExperts,
            )

            from .runtime import read_config

            if isinstance(layer, RoutedExperts) and read_config().enabled:
                # Mirror the base's skip check by consulting it first; only when it
                # would have returned an Fp8MoEMethod do we substitute ours.
                base = super().get_quant_method(layer, prefix)
                if type(base).__name__ == "Fp8MoEMethod":
                    return method_cls(self, layer)
                return base
            return super().get_quant_method(layer, prefix)

    return SurgeonFp8Config


def install_fp8() -> bool:
    """Register the out-of-tree fp8 config under ``"fp8"``, shadowing the builtin.

    Returns whether it took. Idempotent and tolerant: on any import failure (vLLM
    absent, or the fp8 internals moved) it declines rather than raising, since fp8 is
    an optional path.
    """
    try:
        # Presence check: the copied _setup_kernel body needs these.
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import (  # noqa: F401
            Fp8MoeBackend,
            convert_to_fp8_moe_kernel_format,
            make_fp8_moe_kernel,
        )
        from vllm.model_executor.layers.quantization import (
            register_quantization_config,
        )
        from vllm.model_executor.layers.quantization.fp8 import (
            Fp8Config,
            Fp8MoEMethod,
        )
    except ImportError as exc:  # pragma: no cover - vLLM absent or fp8 moved
        logger.debug("fp8 internals not importable; not installing fp8: %s", exc)
        return False

    method_cls = make_surgeon_fp8_method(Fp8MoEMethod)
    config_cls = make_surgeon_fp8_config(Fp8Config, method_cls)
    try:
        register_quantization_config("fp8")(config_cls)
    except Exception as exc:  # pragma: no cover - already registered, or refused
        logger.debug("could not register the fp8 config: %s", exc)
        return False
    logger.info("moe-surgeon fp8 runtime installed (shadowing the 'fp8' config)")
    return True
