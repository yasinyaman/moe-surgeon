# SPDX-License-Identifier: Apache-2.0
"""Which decoder layers actually have experts, and how many.

Pure functions over an HF config mapping. No vLLM, no torch.

This exists because of a specific hazard in the capture path. vLLM's
``RoutedExpertsCapturer`` allocates its buffer with one row per
``num_hidden_layers`` and zeroes it at the start of every step; only layers that
actually route ever write into it. In a model where some decoder layers are
dense, those rows stay zero -- and zero is a *valid expert id*. Aggregate the
capture naively and expert 0 looks like the hottest expert in the model,
including in layers that have no experts at all. A pruning decision built on that
would preserve expert 0 everywhere and drop genuinely hot experts to stay in
budget.

So the layer set is never guessed when the config can tell us.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Order matters: the first key present wins, matching vLLM's own resolver in
# routed_experts_capturer.get_num_experts.
_NUM_EXPERTS_KEYS = ("num_experts", "n_routed_experts", "num_local_experts")
_TOP_K_KEYS = ("num_experts_per_tok", "top_k_experts")


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Descend into the text sub-config of a multimodal config, if present."""
    for key in ("text_config", "llm_config"):
        inner = config.get(key)
        if isinstance(inner, Mapping):
            return inner
    return config


def get_num_experts(config: Mapping[str, Any]) -> int:
    """Routed expert count, across HF naming conventions."""
    cfg = _text_config(config)
    for key in _NUM_EXPERTS_KEYS:
        value = cfg.get(key)
        if value is not None:
            return int(value)
    raise ValueError(
        f"could not resolve num_experts; tried {list(_NUM_EXPERTS_KEYS)}"
    )


def get_top_k(config: Mapping[str, Any]) -> int:
    """Experts selected per token."""
    cfg = _text_config(config)
    for key in _TOP_K_KEYS:
        value = cfg.get(key)
        if value is not None:
            return int(value)
    raise ValueError(f"could not resolve top_k; tried {list(_TOP_K_KEYS)}")


def resolve_moe_layers(config: Mapping[str, Any]) -> list[int]:
    """Global decoder-layer indices that carry routed experts.

    Mirrors the placement rules the model definitions use:

    - ``mlp_only_layers`` + ``decoder_sparse_step`` -- Qwen2/Qwen3-MoE
      (``qwen3_moe.py``: sparse when ``idx not in mlp_only_layers`` and
      ``(idx + 1) % decoder_sparse_step == 0``).
    - ``first_k_dense_replace`` + ``moe_layer_freq`` -- DeepSeek-V2/V3
      (``deepseek_v2.py``: sparse when ``idx >= first_k_dense_replace`` and
      ``idx % moe_layer_freq == 0``).
    - otherwise every layer is MoE (Mixtral, OLMoE).

    A model family with a placement rule not covered here will produce a
    *superset*, which is why :class:`~.stats.ExpertStats` cross-checks the
    result against the data instead of trusting it blindly.
    """
    cfg = _text_config(config)
    num_layers = cfg.get("num_hidden_layers")
    if num_layers is None:
        raise ValueError("config has no num_hidden_layers")
    num_layers = int(num_layers)

    # Go through the shared resolver rather than probing keys again here. A
    # config.json that says "num_experts" can reach us as "num_local_experts"
    # after transformers normalises it (Qwen3-MoE does exactly that), so any
    # second, narrower copy of this lookup silently reports a real MoE model as
    # dense.
    try:
        if get_num_experts(config) == 0:
            return []
    except ValueError:
        # No expert count under any known name: not an MoE config.
        return []

    mlp_only = set(cfg.get("mlp_only_layers") or ())
    sparse_step = int(cfg.get("decoder_sparse_step") or 1)
    first_dense = int(cfg.get("first_k_dense_replace") or 0)
    layer_freq = int(cfg.get("moe_layer_freq") or 1)

    layers = []
    for idx in range(num_layers):
        if idx in mlp_only:
            continue
        if idx < first_dense:
            continue
        # Qwen convention is 1-based on the step, DeepSeek's is 0-based on the
        # frequency. Both reduce to "every layer" when the knob is absent.
        if sparse_step > 1 and (idx + 1) % sparse_step != 0:
            continue
        if layer_freq > 1 and idx % layer_freq != 0:
            continue
        layers.append(idx)
    return layers
