# SPDX-License-Identifier: Apache-2.0
"""Pick a strategy for a target, from properties measured rather than assumed.

``DECISIONS.md`` carries two tables: what each method wins and loses on, and which
subset a given target and model want. This is those tables as code, so the choice is
made from the checkpoint's actual geometry and the profile's actual distribution
instead of from a habit.

The rule it encodes, which is the point of the register: **a method that loses on one
axis is still chosen when it wins on another.** Deletion costs accuracy and is
recommended anyway when it buys feasibility the target does not otherwise have --
and refused when it would only trade accuracy for nothing.

Everything here is decided from ``surgeon budget`` and ``surgeon inspect``-grade
inputs, so a recommendation costs no GPU and no engine. What it cannot decide it
says instead of guessing: the quality of a deletion is measured by ``surgeon gate``,
and this only ever proposes the plan that gate should judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._logging import init_logger
from .budget import CheckpointGeometry, floor_plan, plan_for_vram

logger = init_logger(__name__)

#: Below this share of a layer's load, an expert is cold enough that deleting it is
#: worth measuring. Above it, deletion is trading accuracy for very little.
COLD_SHARE = 0.5
#: A dead tail exists when the coldest experts sit far under a uniform share.
#: Measured on OLMoE the coldest was 0.5% against 1.56% uniform -- only 3x under,
#: which is why deletion cost 1.25x perplexity there rather than nothing.
DEAD_TAIL_RATIO = 0.2


@dataclass
class Recommendation:
    """What to do, why, and what was not decidable here."""

    use_tier: bool = False
    tier_capacity: int | None = None
    fp8_store: bool = True
    delete_experts: int = 0
    merge_enabled: bool = False
    seed_prior: bool = False
    cache_policy: str = "ewma"
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Things this cannot decide without a measurement.
    must_measure: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = ["recommendation:"]
        if self.use_tier:
            lines.append(
                f"  disk tier          capacity {self.tier_capacity}"
                f", fp8_store={self.fp8_store}"
            )
        else:
            lines.append("  disk tier          not needed")
        lines.append(
            f"  delete             {self.delete_experts} experts per layer"
            if self.delete_experts
            else "  delete             nothing"
        )
        merge = "enabled" if self.merge_enabled else "no"
        lines.append(f"  merge              {merge}")
        lines.append(
            f"  residency prior    {'seed it' if self.seed_prior else 'skip'}"
            f"   (policy {self.cache_policy})"
        )
        lines += ["", "because:"]
        lines += [f"  - {r}" for r in self.reasons]
        if self.warnings:
            lines += ["", "warnings:"]
            lines += [f"  ! {w}" for w in self.warnings]
        if self.must_measure:
            lines += ["", "not decidable here -- measure it:"]
            lines += [f"  ? {m}" for m in self.must_measure]
        return "\n".join(lines)


def dead_tail_size(share: np.ndarray, num_experts: int) -> int:
    """How many experts per layer sit far enough under uniform to be worth cutting.

    Counted on the *worst* layer rather than the mean: the config carries one expert
    count for the whole model, so a plan can only delete as many as the least
    prunable layer tolerates.
    """
    uniform = 1.0 / num_experts
    threshold = uniform * DEAD_TAIL_RATIO
    per_layer = [(row < threshold).sum() for row in share]
    return int(min(per_layer)) if per_layer else 0


def recommend(
    geometry: CheckpointGeometry,
    *,
    vram_gib: float | None = None,
    kv_cache_gib: float = 1.0,
    stats: Any | None = None,
    max_similarity: float | None = None,
    merge_threshold: float = 0.85,
    restarts_often: bool = False,
    latency_sensitive: bool = False,
) -> Recommendation:
    """Choose the subset of methods this target and model want."""
    rec = Recommendation()
    floor = floor_plan(geometry)
    floor_gib = floor.gpu_bytes / 1024**3
    full_gib = geometry.total_bytes / 1024**3

    # --- feasibility decides whether the tier is optional at all -------------
    if vram_gib is None:
        rec.must_measure.append(
            "no target VRAM given, so feasibility could not be checked; pass "
            "--vram to find out whether the tier is optional or mandatory"
        )
    elif vram_gib < full_gib:
        rec.use_tier = True
        plan = plan_for_vram(
            geometry,
            int(vram_gib * 1024**3),
            kv_cache_bytes=int(kv_cache_gib * 1024**3),
        )
        rec.tier_capacity = plan.capacity if plan else geometry.top_k
        rec.reasons.append(
            f"the model needs {full_gib:.1f} GiB resident but the target has "
            f"{vram_gib:.1f} GiB, so the tier is mandatory, not a choice"
        )
        if plan is None:
            rec.warnings.append(
                f"even one expert slot does not fit in {vram_gib:.1f} GiB after the "
                f"{kv_cache_gib} GiB KV reserve; reduce the reserve or use zero-copy "
                "on unified memory"
            )
        elif plan.needs_expert_split:
            rec.warnings.append(
                f"capacity {plan.capacity} is below top_k {geometry.top_k}, so the "
                "expert split is required and output is no longer bit-exact"
            )
    else:
        rec.reasons.append(
            f"the whole model ({full_gib:.1f} GiB) fits in {vram_gib:.1f} GiB, so the "
            f"tier is optional; its floor would be {floor_gib:.1f} GiB"
        )
        if latency_sensitive:
            rec.reasons.append(
                "target is latency-sensitive and VRAM-rich, so the tier's throughput "
                "cost (measured 0.38x decode on OLMoE) is not worth paying"
            )
        else:
            rec.use_tier = True
            rec.tier_capacity = max(geometry.top_k, geometry.num_experts // 3)
            rec.reasons.append(
                "the tier is still worth enabling: it halves load time and frees VRAM "
                "for a larger KV cache at ~1.003x perplexity"
            )

    # --- deletion is judged on what it buys, not on accuracy alone -----------
    if stats is None:
        rec.must_measure.append(
            "no profile given, so no expert can be called cold; run `surgeon profile`"
        )
    else:
        share = stats.layer_share()
        tail = dead_tail_size(share, stats.num_experts)
        if tail == 0:
            rec.reasons.append(
                f"no dead tail: even the coldest expert of the worst layer carries "
                f"more than {DEAD_TAIL_RATIO:.0%} of a uniform share, so deletion "
                "would trade accuracy for very little"
            )
        elif rec.use_tier:
            rec.delete_experts = tail
            rec.reasons.append(
                f"{tail} experts per layer are far enough under uniform to delete, "
                "and with the tier enabled that pays twice: a smaller store and a "
                "smaller candidate set, measured 1.32x decode over the tier alone"
            )
        else:
            rec.delete_experts = tail
            rec.reasons.append(
                f"{tail} experts per layer sit under the dead-tail threshold; "
                "deleting them shrinks the artifact at no throughput cost"
            )
        if rec.delete_experts:
            rec.must_measure.append(
                "what the deletions cost -- `surgeon gate` decides, and `apply` "
                "refuses an unmeasured plan"
            )

    # --- merging is available only if the model actually has redundancy ------
    if max_similarity is None:
        rec.must_measure.append(
            "no similarity supplied, so merging could not be considered; pass a "
            "checkpoint to `surgeon plan` to compute it (~80 s per layer)"
        )
    elif max_similarity >= merge_threshold:
        rec.merge_enabled = True
        rec.reasons.append(
            f"a pair reaches similarity {max_similarity:.3f} >= {merge_threshold}, so "
            "merging is available and loses no capacity where deletion would"
        )
    else:
        rec.reasons.append(
            f"the most similar pair is {max_similarity:.3f}, under the "
            f"{merge_threshold} threshold, so there is nothing to merge -- measured "
            "0.37 on OLMoE and 0.401 on Qwen3-30B, so this is the common case"
        )

    # --- the prior wins on one axis only, and that is enough to keep it -----
    if rec.use_tier:
        rec.seed_prior = True
        rec.reasons.append(
            "seed the residency prior: +10 points of hit rate over the first 50 "
            "accesses, decaying to nothing by 2000 -- narrow, but free, since the "
            "manifest is a byproduct of the plan"
            if restarts_often
            else "seed the residency prior anyway; it is free and helps only cold "
            "start, which is the axis it is for"
        )
        rec.reasons.append(
            "use the EWMA policy: it beats LFRU by 2.3-4.0 points on the cold-start "
            "window and 1.2 sustained, for one env var"
        )
    return rec
