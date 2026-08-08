# SPDX-License-Identifier: Apache-2.0
"""The decision engine: a profile plus a budget becomes a per-expert placement.

Pure -- numpy in, dataclasses out, no vLLM and no checkpoint I/O. The plan it
emits is the reviewable artifact of the whole pipeline: JSON, one line of
rationale per decision, and editable by hand before anything touches weights.

Three placements, so pruning is a placement problem rather than a deletion:

``merge_into_core``
    Stays resident. Either hot enough on its own, or the target another expert
    folds into.
``keep_on_disk``
    Survives in the checkpoint but is served from the NVMe tier on demand. The
    cheap answer for the long tail -- no quality lost, only latency when hit.
``drop``
    Deleted, router row removed. Only ever chosen when explicitly budgeted.

Two rules the engine will not be argued out of:

**A thin profile cannot authorise pruning.** With few tokens per expert, the
ranking is noise, and the cost of acting on noise is asymmetric: a dropped expert
cannot be recovered from the artifact. So coverage is checked first and refused
loudly.

**A silent layer is never touched.** If a layer produced no routing rows the
profile says nothing about it, which is not the same as saying its experts are
cold.

**Merging wants similar experts that are rarely used together.** High similarity
alone is not enough. Under top-k routing, two experts that frequently serve the
*same token* are complementary -- the token asked for both -- and merging them
takes real capacity away from it. Co-occurrence therefore vetoes a merge that
similarity alone would have allowed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from .._logging import init_logger
from ..telemetry.stats import ExpertStats

logger = init_logger(__name__)

Action = Literal["merge_into_core", "keep_on_disk", "drop"]

PLAN_VERSION = 1


class ProfileTooThin(RuntimeError):
    """The profile does not carry enough evidence to justify a decision."""


@dataclass
class Budget:
    """What the plan is allowed to do."""

    #: Experts kept resident per layer. The core.
    core_experts: int
    #: Extra experts retained on the disk tier. ``None`` means "all survivors",
    #: which is the safe default: nothing is deleted unless asked.
    disk_experts: int | None = None
    #: Similarity at or above which two experts may be merged.
    merge_threshold: float = 0.85
    #: Veto a merge when this fraction of the donor's tokens also used the
    #: target -- they are complementary, not redundant.
    max_cooccurrence: float = 0.10
    #: Drop, rather than keeping on disk, any expert below this per-layer share.
    #: ``None`` disables dropping entirely.
    drop_share_below: float | None = None
    #: Minimum mean token slots per expert before any pruning is allowed.
    min_slots_per_expert: float = 200.0

    def __post_init__(self) -> None:
        if self.core_experts < 1:
            raise ValueError("core_experts must be >= 1")
        if not 0.0 <= self.merge_threshold <= 1.0:
            raise ValueError("merge_threshold must be in [0, 1]")
        if not 0.0 <= self.max_cooccurrence <= 1.0:
            raise ValueError("max_cooccurrence must be in [0, 1]")


@dataclass
class ExpertPlacement:
    layer: int
    expert: int
    action: Action
    tokens: int
    share: float
    #: Per-layer share of the *importance* the ranking used (rank-1 frequency by
    #: default), which is what residency should follow. ``share`` remains the raw
    #: token share, so both are auditable and the tier does not have to re-derive
    #: the ranking from counts it was told not to trust.
    importance_share: float = 0.0
    #: For a merge, the surviving expert this one folds into.
    merge_target: int | None = None
    similarity: float | None = None
    reason: str = ""


@dataclass
class Plan:
    model: str | None
    revision: str | None
    budget: dict[str, Any]
    placements: list[ExpertPlacement]
    #: Layers left completely untouched, and why.
    untouched_layers: dict[int, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    #: Ablation verdict, written by the quality gate. ``None`` means the plan has
    #: never been measured, which :func:`~.apply.apply_plan` treats as a refusal
    #: when the plan deletes anything.
    gate: dict[str, Any] | None = None
    plan_version: int = PLAN_VERSION

    def by_layer(self, layer: int) -> list[ExpertPlacement]:
        return [p for p in self.placements if p.layer == layer]

    def counts(self) -> dict[str, int]:
        out = {"merge_into_core": 0, "keep_on_disk": 0, "drop": 0}
        for placement in self.placements:
            out[placement.action] += 1
        return out

    def to_json(self, indent: int = 2) -> str:
        payload = {
            "plan_version": self.plan_version,
            "model": self.model,
            "revision": self.revision,
            "budget": self.budget,
            "provenance": self.provenance,
            "gate": self.gate,
            "warnings": self.warnings,
            "untouched_layers": {str(k): v for k, v in self.untouched_layers.items()},
            "placements": [asdict(p) for p in self.placements],
        }
        return json.dumps(payload, indent=indent)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())
            f.write("\n")


def load_plan(path: str) -> Plan:
    """Read a plan back, validating it -- plans are meant to be hand-edited."""
    with open(path) as f:
        payload = json.load(f)
    if payload.get("plan_version") != PLAN_VERSION:
        raise ValueError(
            f"{path}: plan_version {payload.get('plan_version')} != {PLAN_VERSION}"
        )
    placements = [ExpertPlacement(**p) for p in payload["placements"]]
    plan = Plan(
        model=payload.get("model"),
        revision=payload.get("revision"),
        budget=payload.get("budget", {}),
        placements=placements,
        untouched_layers={
            int(k): v for k, v in (payload.get("untouched_layers") or {}).items()
        },
        warnings=list(payload.get("warnings") or ()),
        provenance=payload.get("provenance") or {},
        gate=payload.get("gate"),
    )
    validate_plan(plan)
    return plan


def validate_plan(plan: Plan) -> None:
    """Reject a plan that cannot be applied, before any weight is touched."""
    seen: set[tuple[int, int]] = set()
    # Every layer gets an entry even when nothing survives -- otherwise a
    # fully-dropped layer simply has no key and the emptiness check below never
    # runs on it, which is the one case it exists to catch.
    survivors: dict[int, set[int]] = {}
    for placement in plan.placements:
        key = (placement.layer, placement.expert)
        if key in seen:
            raise ValueError(f"duplicate placement for layer {key[0]} expert {key[1]}")
        seen.add(key)
        if placement.action not in ("merge_into_core", "keep_on_disk", "drop"):
            raise ValueError(f"unknown action {placement.action!r} at {key}")
        kept = survivors.setdefault(placement.layer, set())
        if placement.action != "drop":
            kept.add(placement.expert)

    for placement in plan.placements:
        if placement.merge_target is None:
            continue
        # Self-merge first: it is the more specific diagnosis, and a self-merging
        # donor also fails the "target survives" test, which would otherwise
        # mask it behind a confusing message.
        if placement.merge_target == placement.expert:
            raise ValueError(
                f"layer {placement.layer} expert {placement.expert} merges into itself"
            )
        if placement.action != "drop":
            raise ValueError(
                f"layer {placement.layer} expert {placement.expert}: a merge donor "
                f"must be action 'drop' (its weights fold into "
                f"{placement.merge_target}), not {placement.action!r}"
            )
        if placement.merge_target not in survivors.get(placement.layer, ()):
            raise ValueError(
                f"layer {placement.layer} expert {placement.expert} merges into "
                f"{placement.merge_target}, which does not survive"
            )

    for layer, keep in sorted(survivors.items()):
        if not keep:
            raise ValueError(f"layer {layer} would keep no experts")


def deletes_anything(plan: Plan) -> bool:
    """Whether this plan removes expert capacity at all.

    A plan that only re-places experts (core vs disk) loses nothing, so it needs
    no quality gate. Only deletions are irreversible.
    """
    return any(p.action == "drop" for p in plan.placements)


def gate_passed(plan: Plan) -> bool:
    return bool(plan.gate and plan.gate.get("passed"))


def coverage(stats: ExpertStats) -> float:
    """Mean token slots per (layer, expert) over layers that produced data.

    The quantity that decides whether the ranking is signal or noise.
    """
    live = stats.layer_token_rows > 0
    if not live.any():
        return 0.0
    return float(stats.tokens[live].sum() / (live.sum() * stats.num_experts))


#: Rank weights that measured best on OLMoE. Concentrating importance on the
#: first top-k slot beat plain selection count by a wide margin: it nominated a
#: keep-set that discarded *more* raw routing load (21.6% vs 14.6%) while costing
#: *less* perplexity (1.30x vs 1.74x), which is the clearest evidence that count
#: is the wrong quantity for pruning. A top-k output is a gate-weighted sum, and
#: an expert selected constantly at the last slot barely moves it.
def rank1_weights(top_k: int) -> np.ndarray:
    weights = np.zeros(top_k, dtype=np.float64)
    weights[0] = 1.0
    return weights


def default_importance(stats: ExpertStats) -> tuple[np.ndarray, str]:
    """The ranking to prune by, and the name of what was used.

    Prefers rank-1 selection frequency, falling back to plain counts when the
    profile carries no position histogram or the router's ids are not
    score-ordered -- in which case rank weighting would be noise, and saying so is
    better than silently using a worse ranking that looks the same.
    """
    if stats.positions is None:
        return stats.tokens.astype(np.float64), "token count (no position data)"
    if not stats.positions_look_ordered:
        return (
            stats.tokens.astype(np.float64),
            "token count (router ids not score-ordered)",
        )
    top_k = stats.positions.shape[2]
    return stats.importance(rank1_weights(top_k)), "rank-1 selection frequency"


def build_plan(
    stats: ExpertStats,
    budget: Budget,
    *,
    importance: np.ndarray | None = None,
    similarity: dict[int, np.ndarray] | None = None,
    model: str | None = None,
    revision: str | None = None,
    provenance: dict[str, Any] | None = None,
    force: bool = False,
) -> Plan:
    """Decide a placement for every expert.

    ``similarity[layer]`` is an ``[E, E]`` permutation-invariant similarity
    matrix (see :mod:`.descriptors`). Without it no merge is proposed, and the
    tail goes to disk or is dropped per the budget.
    """
    slots = coverage(stats)
    warnings: list[str] = []
    if importance is None:
        importance, importance_name = default_importance(stats)
    else:
        importance_name = "caller-supplied"
    if importance.shape != stats.tokens.shape:
        raise ValueError(
            f"importance has shape {importance.shape}, expected {stats.tokens.shape}"
        )
    if slots < budget.min_slots_per_expert:
        message = (
            f"profile carries {slots:.1f} token slots per expert, below the "
            f"{budget.min_slots_per_expert:.0f} required. At this density the "
            "expert ranking is noise, and a dropped expert cannot be recovered "
            "from the artifact. Profile on more data, or pass force=True to "
            "accept a plan built on insufficient evidence."
        )
        if not force:
            raise ProfileTooThin(message)
        warnings.append("FORCED: " + message)
        logger.warning(message)

    if budget.core_experts >= stats.num_experts:
        warnings.append(
            f"core_experts={budget.core_experts} >= num_experts="
            f"{stats.num_experts}: every expert stays resident, so this plan is "
            "a no-op"
        )

    placements: list[ExpertPlacement] = []
    untouched: dict[int, str] = {}
    share = stats.layer_share()

    for slot, layer in enumerate(stats.moe_layers):
        counts = stats.tokens[slot]
        rank_by = importance[slot]
        rank_total = float(rank_by.sum())
        rank_share = (
            rank_by / rank_total if rank_total > 0 else np.zeros_like(rank_by)
        )

        if stats.layer_token_rows[slot] == 0:
            untouched[layer] = (
                "no routing rows in the profile; silence is not evidence of cold "
                "experts"
            )
            for expert in range(stats.num_experts):
                placements.append(
                    ExpertPlacement(
                        layer=layer,
                        expert=expert,
                        action="merge_into_core",
                        tokens=int(counts[expert]),
                        share=float(share[slot, expert]),
                        importance_share=float(share[slot, expert]),
                        reason="layer untouched: no profile data",
                    )
                )
            continue

        # Ties broken by expert id so a plan is reproducible.
        # Ranked by importance, not by raw count -- see default_importance.
        order = sorted(
            range(stats.num_experts), key=lambda e: (-float(rank_by[e]), e)
        )
        core = order[: budget.core_experts]
        core_set = set(core)
        tail = order[budget.core_experts :]

        for expert in core:
            placements.append(
                ExpertPlacement(
                    layer=layer,
                    expert=expert,
                    action="merge_into_core",
                    tokens=int(counts[expert]),
                    share=float(share[slot, expert]),
                    importance_share=float(rank_share[expert]),
                    reason=f"rank {order.index(expert) + 1} by importance",
                )
            )

        layer_similarity = (similarity or {}).get(layer)
        disk_budget = budget.disk_experts
        disk_used = 0

        for expert in tail:
            expert_share = float(share[slot, expert])
            target, sim = _best_merge_target(
                stats, slot, expert, core, layer_similarity, budget
            )

            if target is not None:
                placements.append(
                    ExpertPlacement(
                        layer=layer,
                        expert=expert,
                        action="drop",
                        tokens=int(counts[expert]),
                        share=expert_share,
                        importance_share=float(rank_share[expert]),
                        merge_target=target,
                        similarity=sim,
                        reason=(
                            f"merged into {target}: subspace similarity "
                            f"{sim:.3f} >= {budget.merge_threshold}, "
                            "co-occurrence below veto"
                        ),
                    )
                )
                continue

            if (
                budget.drop_share_below is not None
                and expert_share < budget.drop_share_below
            ):
                placements.append(
                    ExpertPlacement(
                        layer=layer,
                        expert=expert,
                        action="drop",
                        tokens=int(counts[expert]),
                        share=expert_share,
                        importance_share=float(rank_share[expert]),
                        similarity=sim,
                        reason=(
                            f"share {expert_share:.5f} < "
                            f"{budget.drop_share_below} and no merge target"
                        ),
                    )
                )
                continue

            if disk_budget is not None and disk_used >= disk_budget:
                placements.append(
                    ExpertPlacement(
                        layer=layer,
                        expert=expert,
                        action="drop",
                        tokens=int(counts[expert]),
                        share=expert_share,
                        importance_share=float(rank_share[expert]),
                        similarity=sim,
                        reason="disk budget exhausted and no merge target",
                    )
                )
                continue

            disk_used += 1
            placements.append(
                ExpertPlacement(
                    layer=layer,
                    expert=expert,
                    action="keep_on_disk",
                    tokens=int(counts[expert]),
                    share=expert_share,
                    importance_share=float(rank_share[expert]),
                    similarity=sim,
                    reason="cold tail retained on the disk tier",
                )
            )

        del core_set

    if similarity is None:
        warnings.append(
            "no similarity matrix supplied: no merges proposed, the tail is "
            "placed on disk or dropped by budget alone"
        )
    else:
        merged = [p for p in placements if p.merge_target is not None]
        observed = [
            p.similarity for p in placements if p.similarity is not None
        ]
        if not merged and observed:
            # Silence here would read as a bug. It usually is not: measured on
            # OLMoE-1B-7B, no pair of the 64 experts in any layer reaches 0.4,
            # and the ceiling *falls* as descriptor rank grows -- the signature
            # of experts that are genuinely distinct rather than redundant.
            warnings.append(
                f"similarity was supplied but no merge cleared the bar: the "
                f"highest similarity seen between a tail expert and any core "
                f"expert was {max(observed):.3f}, against a threshold of "
                f"{budget.merge_threshold}. Either these experts are genuinely "
                f"distinct in weight space, or merging needs activation-based "
                f"similarity measured on the target domain instead."
            )

    plan = Plan(
        model=model,
        revision=revision,
        budget=asdict(budget),
        placements=placements,
        untouched_layers=untouched,
        warnings=warnings,
        provenance={
            "slots_per_expert": slots,
            "n_sequences": stats.n_sequences,
            "num_experts": stats.num_experts,
            "moe_layers": stats.moe_layers,
            "silent_layers": sorted(stats.silent_layers),
            "forced": bool(force and slots < budget.min_slots_per_expert),
            "ranked_by": importance_name,
            **(provenance or {}),
        },
    )
    validate_plan(plan)
    return plan


def _best_merge_target(
    stats: ExpertStats,
    slot: int,
    donor: int,
    core: list[int],
    layer_similarity: np.ndarray | None,
    budget: Budget,
) -> tuple[int | None, float | None]:
    """Pick the core expert ``donor`` should fold into, if any."""
    if layer_similarity is None:
        return None, None
    if layer_similarity.shape != (stats.num_experts, stats.num_experts):
        raise ValueError(
            f"similarity matrix is {layer_similarity.shape}, expected "
            f"({stats.num_experts}, {stats.num_experts})"
        )

    best: int | None = None
    best_sim = -1.0
    observed = -1.0
    for candidate in core:
        sim = float(layer_similarity[donor, candidate])
        observed = max(observed, sim)
        if sim < budget.merge_threshold:
            continue
        overlap = _cooccurrence_fraction(stats, slot, donor, candidate)
        if overlap > budget.max_cooccurrence:
            # Complementary, not redundant: this pair serves the same tokens
            # together, so folding one into the other removes capacity the
            # router was actually using.
            continue
        if sim > best_sim:
            best, best_sim = candidate, sim
    if best is None:
        return None, (observed if observed >= 0 else None)
    return best, best_sim


def _cooccurrence_fraction(
    stats: ExpertStats, slot: int, donor: int, target: int
) -> float:
    """Fraction of ``donor``'s token slots whose token also chose ``target``."""
    if stats.cooc is None:
        return 0.0
    donor_tokens = int(stats.tokens[slot, donor])
    if donor_tokens == 0:
        return 0.0
    return float(stats.cooc[slot, donor, target]) / donor_tokens


def summarize_plan(plan: Plan) -> str:
    """A report a human can check before anything is written."""
    counts = plan.counts()
    layers = sorted({p.layer for p in plan.placements})
    lines = [
        f"model:      {plan.model or '(unset)'}",
        f"core/layer: {plan.budget.get('core_experts')}",
        f"placements: {counts['merge_into_core']} core, "
        f"{counts['keep_on_disk']} on disk, {counts['drop']} dropped",
        f"slots/expert in profile: "
        f"{plan.provenance.get('slots_per_expert', float('nan')):.1f}",
        f"ranked by:  {plan.provenance.get('ranked_by', 'unknown')}",
    ]
    merged = [p for p in plan.placements if p.merge_target is not None]
    deleted = [
        p for p in plan.placements if p.action == "drop" and p.merge_target is None
    ]
    lines.append(f"of the dropped: {len(merged)} merged away, {len(deleted)} deleted")
    if plan.untouched_layers:
        lines.append(f"untouched layers: {sorted(plan.untouched_layers)}")
    for warning in plan.warnings:
        lines.append(f"WARNING: {warning}")
    lines.append("")
    lines.append(
        f"  {'layer':>5}  {'core':>4}  {'disk':>4}  {'merged':>6}  {'deleted':>7}"
    )
    for layer in layers:
        rows = plan.by_layer(layer)
        lines.append(
            f"  {layer:>5}  "
            f"{sum(1 for p in rows if p.action == 'merge_into_core'):>4}  "
            f"{sum(1 for p in rows if p.action == 'keep_on_disk'):>4}  "
            f"{sum(1 for p in rows if p.merge_target is not None):>6}  "
            f"{sum(1 for p in rows if p.action == 'drop' and not p.merge_target):>7}"
        )
    return "\n".join(lines)
