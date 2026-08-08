# SPDX-License-Identifier: Apache-2.0
"""Drive an offline profiling run through vLLM's public capture API.

This is the only vLLM-dependent part of Faz 1, and it is thin on purpose: boot
an ``LLM`` with ``enable_return_routed_experts=True``, generate, and hand each
sequence's ``routed_experts`` array to :mod:`vllm_moe_surgeon.telemetry`. No
layer is hooked and no internal is patched.

Kept in ``compat/`` because it imports vLLM, so the layering test stays true and
the aggregation half remains runnable on a laptop.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .._logging import init_logger
from ..telemetry.layers import get_num_experts, get_top_k, resolve_moe_layers
from ..telemetry.stats import ExpertStats

logger = init_logger(__name__)


def load_hf_config(model: str, revision: str | None = None) -> dict[str, Any]:
    """The model's HF config as a plain dict, for the pure layer resolvers."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model, revision=revision)
    return config.to_dict()


def profile_offline(
    model: str,
    prompts: Sequence[str],
    *,
    max_tokens: int = 32,
    revision: str | None = None,
    with_cooc: bool = False,
    llm_kwargs: dict[str, Any] | None = None,
) -> ExpertStats:
    """Run ``prompts`` through ``model`` and aggregate per-expert usage.

    Raises if the run produced no routing at all, rather than returning an empty
    profile: a profile that silently covers nothing looks identical to a model
    whose experts are all cold, and the second reading would authorise pruning
    everything.
    """
    from vllm import LLM, SamplingParams

    config = load_hf_config(model, revision)
    num_experts = get_num_experts(config)
    top_k = get_top_k(config)
    moe_layers = resolve_moe_layers(config)
    if not moe_layers:
        raise ValueError(f"{model} has no MoE layers to profile")

    logger.info(
        "profiling %s: %d experts, top_k=%d, MoE layers %s",
        model,
        num_experts,
        top_k,
        moe_layers,
    )

    kwargs: dict[str, Any] = {"enable_return_routed_experts": True}
    if revision is not None:
        kwargs["revision"] = revision
    kwargs.update(llm_kwargs or {})

    llm = LLM(model=model, **kwargs)
    outputs = llm.generate(
        list(prompts), SamplingParams(max_tokens=max_tokens, temperature=0.0)
    )

    stats = ExpertStats.empty(
        num_experts, moe_layers, with_cooc=with_cooc, top_k=top_k
    )
    missing = 0
    for request in outputs:
        for completion in request.outputs:
            capture = getattr(completion, "routed_experts", None)
            if capture is None:
                missing += 1
                continue
            stats.add(capture)
    stats.finalize()

    if stats.tokens.sum() == 0:
        raise RuntimeError(
            f"{model}: profiling captured no routing at all "
            f"({missing} completions had no routed_experts). Refusing to return "
            "an empty profile -- it is indistinguishable from a model whose "
            "experts are all cold."
        )
    if missing:
        logger.warning(
            "%d completions carried no routed_experts and were skipped", missing
        )
    return stats


def summarize(stats: ExpertStats) -> str:
    """A human-checkable report. Numbers first, caveats not buried."""
    import numpy as np

    share = stats.layer_share()
    lines = [
        f"sequences:        {stats.n_sequences}",
        f"MoE layers:       {len(stats.moe_layers)} {stats.moe_layers}",
        f"experts:          {stats.num_experts}",
        f"token slots:      {int(stats.tokens.sum())}",
        f"mean experts/tok: {stats.top_k_hint:.3f}",
        f"dropped rows:     {stats.dropped_degenerate_rows} degenerate, "
        f"{stats.dropped_sentinels} sentinels",
        f"silent layers:    {sorted(stats.silent_layers) or 'none'}",
        f"position order:   {stats.position_order_correlation():+.3f}  "
        f"({'ranked' if stats.positions_look_ordered else 'NOT ordered'})",
        "",
        "per-layer expert concentration:",
        f"  {'layer':>5}  {'rows':>7}  {'used':>5}  {'top1%':>6}  {'top25%':>7}",
    ]
    for slot, layer in enumerate(stats.moe_layers):
        row = share[slot]
        used = int((stats.tokens[slot] > 0).sum())
        ordered = np.sort(row)[::-1]
        quarter = max(1, stats.num_experts // 4)
        lines.append(
            f"  {layer:>5}  {int(stats.layer_token_rows[slot]):>7}  "
            f"{used:>5}  {ordered[0] * 100:>5.1f}%  "
            f"{ordered[:quarter].sum() * 100:>6.1f}%"
        )
    return "\n".join(lines)


def iter_prompts_from_jsonl(path: str, field: str = "text") -> Iterable[str]:
    """Read a calibration corpus: one JSON object per line."""
    import json

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, str):
                yield record
            elif field in record:
                yield record[field]
            elif "prompt" in record:
                yield record["prompt"]
            else:
                raise ValueError(
                    f"{path}: record has neither {field!r} nor 'prompt': "
                    f"{sorted(record)[:5]}"
                )
