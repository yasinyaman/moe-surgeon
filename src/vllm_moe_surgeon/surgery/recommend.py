# SPDX-License-Identifier: Apache-2.0
"""Pick a strategy for a target, from properties measured rather than assumed.

Which method a target and model want, decided from the checkpoint's actual
geometry and the profile's actual distribution instead of from a habit. The
measurements behind each rule are in ``docs/benchmarks.md``.

The rule it encodes, updated once task accuracy was measured: **the tier is the
primary mechanism; deletion is a last resort.** The tier keeps every expert (cold
ones stream from NVMe), so it costs no accuracy; deletion permanently removes
experts and, measured on downstream tasks, costs real accuracy that the perplexity
gate does not catch and the amplitude fix does not recover (arc_challenge −25%
relative on OLMoE rank-1 pruned-40). So deletion is recommended **only when the tier
alone cannot fit the target** — when even the tier's own floor is above the target's
VRAM, a smaller model is the only way to run at all. Where the tier fits, deletion is
declined even against a cold tail: shrinking the store is not worth measured task
accuracy unless a hard resource constraint forces it.

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

#: Attached to every deletion recommendation. Deletion looked cheap on perplexity and
#: is not on downstream tasks; say so wherever deletion is proposed.
_DELETION_COST_WARNING = (
    "deletion permanently removes experts and costs measured task accuracy the "
    "perplexity gate does not catch: OLMoE rank-1 pruned-40 lost 25% relative "
    "arc_challenge accuracy (paired McNemar p=5e-08), and the amplitude fix does not "
    "recover it. Do this only because the resource constraint forces it, and prefer "
    "the tier wherever it fits."
)

#: Also attached to every deletion recommendation, as a *prerequisite* rather than a
#: caveat. Deletion buys footprint at a measured quality cost; measured once against
#: an off-the-shelf 800M-active checkpoint on a narrow domain, the same footprint came
#: free and quality improved. That check costs minutes and can retire the whole
#: pipeline, so it belongs before the plan, not after the regret.
_HEADROOM_PREREQUISITE = (
    "whether a smaller existing checkpoint already serves this domain better -- "
    "`surgeon headroom --corpus heldout.jsonl --model A --model B` ranks candidates "
    "in minutes. Measured once: an 800M-active checkpoint beat the unpruned teacher "
    "by 4.8% bits/byte with no significant arc_challenge acc_norm loss (p=0.51), "
    "where deleting cost 11.6 points -- though on raw acc the same run measured "
    "-0.068 at p=0.0008, so the two arc metrics disagree. Deletion is worth its "
    "damage only if nothing off the shelf wins"
)


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
        if self.delete_experts == -1:
            delete_line = "  delete             last resort (size with budget)"
        elif self.delete_experts:
            delete_line = (
                f"  delete             {self.delete_experts} experts per layer"
            )
        else:
            delete_line = "  delete             nothing"
        lines.append(delete_line)
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
    # Resident bytes, not checkpoint bytes: vLLM downcasts an fp32 model, so a 12.57
    # GiB fp32 checkpoint is 6.29 GiB resident and would otherwise be called
    # tier-mandatory on hardware that fits it fine.
    full_gib = geometry.resident_scale(geometry.total_bytes) / 1024**3

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

    # --- deletion is a LAST RESORT: only when the tier alone cannot fit -------
    # The tier keeps every expert, so it costs no accuracy; deletion is permanent and
    # measured to cost real task accuracy the gate does not catch. So it is proposed
    # only when even the tier's own floor is above the target's VRAM -- a smaller model
    # is then the only way to run at all -- and declined otherwise, cold tail or not.
    tier_cannot_fit = vram_gib is not None and vram_gib < floor_gib
    if stats is None and tier_cannot_fit:
        rec.delete_experts = -1  # count is a sizing question; defer to `surgeon budget`
        rec.reasons.append(
            f"the tier's own floor is {floor_gib:.1f} GiB but the target has only "
            f"{vram_gib:.1f} GiB, so the tier alone cannot run this model; deletion is "
            "the last resort that makes it fit. Size it with `surgeon budget --vram`, "
            "then measure with `surgeon gate`"
        )
        rec.warnings.append(_DELETION_COST_WARNING)
        rec.must_measure.append(_HEADROOM_PREREQUISITE)
    elif stats is None and vram_gib is not None:
        rec.reasons.append(
            "the tier fits the target, so deletion is not needed; no profile is "
            "required to reach that conclusion"
        )
    elif stats is None:
        rec.must_measure.append(
            "no profile given, so coldness cannot be judged; run `surgeon profile` "
            "(and pass --vram, since deletion is only worth considering when the tier "
            "cannot fit)"
        )
    else:
        share = stats.layer_share()
        tail = dead_tail_size(share, stats.num_experts)
        if tier_cannot_fit:
            rec.delete_experts = tail or -1
            rec.reasons.append(
                f"the tier's floor is {floor_gib:.1f} GiB but the target has "
                f"{vram_gib:.1f} GiB, so the tier alone cannot run this model; "
                "deletion is the last resort that makes it fit. Delete the cold tail "
                "first"
                + (f" ({tail} per layer)" if tail else "")
                + ", size the rest with `surgeon budget --vram`, and measure the cost"
            )
            rec.warnings.append(_DELETION_COST_WARNING)
            rec.must_measure.append(_HEADROOM_PREREQUISITE)
            rec.must_measure.append(
                "what the deletions cost -- `surgeon gate` decides, and `apply` "
                "refuses an unmeasured plan"
            )
        elif tail == 0:
            rec.reasons.append(
                "no dead tail and the tier fits, so nothing is deleted -- deletion "
                "would only trade measured task accuracy for a smaller store"
            )
        else:
            rec.reasons.append(
                f"there is a {tail}-expert cold tail, but the tier already fits the "
                f"target (floor {floor_gib:.1f} GiB), so deletion is NOT recommended: "
                "keeping those experts on the tier costs no accuracy, while deleting "
                "them is a measured task-accuracy loss worth paying only under a hard "
                "resource constraint. Delete only the genuinely worthless, under a gate"
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
            "merging is available; it keeps the capacity deletion would remove, but "
            "it is not free -- below the threshold it measured WORSE than deleting "
            "the same experts (1.416x against 1.226x on OLMoE), and no gate verdict "
            "covers a merge, since zeroing a donor does not model folding it in"
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
