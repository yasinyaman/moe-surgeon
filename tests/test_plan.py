# SPDX-License-Identifier: Apache-2.0
"""The decision engine, with emphasis on what it refuses to do."""

from __future__ import annotations

import numpy as np
import pytest

from vllm_moe_surgeon.surgery import (
    Budget,
    Plan,
    ProfileTooThin,
    build_plan,
    coverage,
    load_plan,
    summarize_plan,
    validate_plan,
)
from vllm_moe_surgeon.telemetry.stats import ExpertStats

E = 8


def _stats(
    counts_by_layer: dict[int, list[int]],
    *,
    cooc: dict[int, np.ndarray] | None = None,
    rows: int | None = None,
) -> ExpertStats:
    layers = sorted(counts_by_layer)
    stats = ExpertStats.empty(E, layers, with_cooc=cooc is not None)
    for slot, layer in enumerate(layers):
        counts = np.asarray(counts_by_layer[layer], dtype=np.int64)
        stats.tokens[slot] = counts
        stats.layer_token_rows[slot] = (
            rows if rows is not None else max(1, int(counts.sum() // 2))
        )
        if counts.sum() == 0:
            stats.layer_token_rows[slot] = 0
        if cooc is not None and layer in cooc:
            stats.cooc[slot] = cooc[layer]
    stats.n_sequences = 10
    stats.finalize()
    return stats


def _hot_profile() -> ExpertStats:
    """A profile dense enough to authorise pruning: 8 experts, plenty of slots."""
    return _stats({0: [4000, 3000, 2000, 1000, 500, 200, 100, 50]})


def _identity_similarity(value: float = 0.0) -> np.ndarray:
    matrix = np.full((E, E), value, dtype=np.float32)
    np.fill_diagonal(matrix, 1.0)
    return matrix


# ------------------------------------------------------------------ refusals


def test_thin_profile_is_refused():
    """A ranking built on a handful of tokens is noise, and drops are final."""
    thin = _stats({0: [10, 8, 6, 4, 3, 2, 1, 1]})
    assert coverage(thin) < 200
    with pytest.raises(ProfileTooThin, match="token slots per expert"):
        build_plan(thin, Budget(core_experts=4))


def test_thin_profile_can_be_forced_but_says_so():
    thin = _stats({0: [10, 8, 6, 4, 3, 2, 1, 1]})
    plan = build_plan(thin, Budget(core_experts=4), force=True)
    assert any("FORCED" in w for w in plan.warnings)
    assert plan.provenance["forced"] is True


def test_silent_layer_is_never_pruned():
    """No routing rows means no evidence -- which is not evidence of cold."""
    stats = _stats({0: [4000, 3000, 2000, 1000, 500, 200, 100, 50], 1: [0] * 8})
    plan = build_plan(stats, Budget(core_experts=2))

    assert 1 in plan.untouched_layers
    layer1 = plan.by_layer(1)
    assert len(layer1) == E
    assert all(p.action == "merge_into_core" for p in layer1)
    # Layer 0 still gets pruned normally.
    assert any(p.action != "merge_into_core" for p in plan.by_layer(0))


# ------------------------------------------------------------- core selection


def test_core_is_the_hottest_experts():
    plan = build_plan(_hot_profile(), Budget(core_experts=3))
    core = {p.expert for p in plan.by_layer(0) if p.action == "merge_into_core"}
    assert core == {0, 1, 2}


def test_ties_break_by_expert_id_for_reproducibility():
    stats = _stats({0: [1000] * 8})
    a = build_plan(stats, Budget(core_experts=3))
    b = build_plan(stats, Budget(core_experts=3))
    core_a = sorted(p.expert for p in a.by_layer(0) if p.action == "merge_into_core")
    core_b = sorted(p.expert for p in b.by_layer(0) if p.action == "merge_into_core")
    assert core_a == core_b == [0, 1, 2]


def test_tail_goes_to_disk_by_default_nothing_deleted():
    """The safe default: pruning is placement, deletion must be asked for."""
    plan = build_plan(_hot_profile(), Budget(core_experts=3))
    counts = plan.counts()
    assert counts["drop"] == 0
    assert counts["merge_into_core"] == 3
    assert counts["keep_on_disk"] == 5


def test_no_similarity_means_no_merges_and_says_so():
    plan = build_plan(_hot_profile(), Budget(core_experts=3))
    assert all(p.merge_target is None for p in plan.placements)
    assert any("no similarity matrix" in w for w in plan.warnings)


# -------------------------------------------------------------------- merging


def test_similar_expert_merges_into_core():
    similarity = _identity_similarity(0.1)
    similarity[7, 0] = similarity[0, 7] = 0.95  # cold expert 7 ~ hot expert 0
    plan = build_plan(
        _hot_profile(), Budget(core_experts=3), similarity={0: similarity}
    )
    seven = next(p for p in plan.by_layer(0) if p.expert == 7)
    assert seven.action == "drop"
    assert seven.merge_target == 0
    assert seven.similarity == pytest.approx(0.95)
    assert "merged into 0" in seven.reason


def test_cooccurrence_vetoes_a_merge_similarity_would_have_allowed():
    """Complementary experts must not be merged.

    Experts that serve the *same token* together are not redundant -- the router
    asked for both. Folding one into the other silently reduces the capacity
    that token was getting, and similarity alone cannot see the difference.
    """
    similarity = _identity_similarity(0.1)
    similarity[7, 0] = similarity[0, 7] = 0.95

    cooc = np.zeros((E, E), dtype=np.int64)
    # Expert 7 has 50 slots; 40 of its tokens also chose expert 0 -> 0.8.
    cooc[7, 0] = cooc[0, 7] = 40

    plan = build_plan(
        _stats({0: [4000, 3000, 2000, 1000, 500, 200, 100, 50]}, cooc={0: cooc}),
        Budget(core_experts=3, max_cooccurrence=0.10),
        similarity={0: similarity},
    )
    seven = next(p for p in plan.by_layer(0) if p.expert == 7)
    assert seven.merge_target is None, "high co-occurrence must veto the merge"
    assert seven.action == "keep_on_disk"
    # The similarity it *would* have used is still reported, for auditing.
    assert seven.similarity == pytest.approx(0.95)


def test_raising_the_cooccurrence_ceiling_permits_the_merge():
    """The veto is a threshold, not a hard rule -- and it is in the plan."""
    similarity = _identity_similarity(0.1)
    similarity[7, 0] = similarity[0, 7] = 0.95
    cooc = np.zeros((E, E), dtype=np.int64)
    cooc[7, 0] = cooc[0, 7] = 40

    plan = build_plan(
        _stats({0: [4000, 3000, 2000, 1000, 500, 200, 100, 50]}, cooc={0: cooc}),
        Budget(core_experts=3, max_cooccurrence=0.9),
        similarity={0: similarity},
    )
    seven = next(p for p in plan.by_layer(0) if p.expert == 7)
    assert seven.merge_target == 0


def test_merge_picks_the_most_similar_core_target():
    similarity = _identity_similarity(0.1)
    similarity[7, 0] = similarity[0, 7] = 0.88
    similarity[7, 2] = similarity[2, 7] = 0.97
    plan = build_plan(
        _hot_profile(), Budget(core_experts=3), similarity={0: similarity}
    )
    seven = next(p for p in plan.by_layer(0) if p.expert == 7)
    assert seven.merge_target == 2


def test_below_threshold_similarity_does_not_merge():
    similarity = _identity_similarity(0.1)
    similarity[7, 0] = similarity[0, 7] = 0.5
    plan = build_plan(
        _hot_profile(),
        Budget(core_experts=3, merge_threshold=0.85),
        similarity={0: similarity},
    )
    seven = next(p for p in plan.by_layer(0) if p.expert == 7)
    assert seven.merge_target is None
    assert seven.similarity == pytest.approx(0.5)  # reported anyway


def test_wrong_similarity_shape_raises():
    with pytest.raises(ValueError, match="similarity matrix is"):
        build_plan(
            _hot_profile(),
            Budget(core_experts=3),
            similarity={0: np.eye(3, dtype=np.float32)},
        )


# --------------------------------------------------------------------- budget


def test_drop_share_below_deletes_the_coldest():
    plan = build_plan(
        _hot_profile(), Budget(core_experts=3, drop_share_below=0.02)
    )
    dropped = {p.expert for p in plan.by_layer(0) if p.action == "drop"}
    # Total is 10850, so shares are .369 .277 .184 .092 .046 .0184 .0092 .0046
    # -- experts 5, 6 and 7 fall under 2%.
    assert dropped == {5, 6, 7}
    assert all(p.merge_target is None for p in plan.by_layer(0) if p.action == "drop")


def test_disk_budget_exhaustion_drops_the_remainder():
    plan = build_plan(_hot_profile(), Budget(core_experts=3, disk_experts=2))
    rows = plan.by_layer(0)
    assert sum(1 for p in rows if p.action == "keep_on_disk") == 2
    assert sum(1 for p in rows if p.action == "drop") == 3
    # Disk slots go to the warmest of the tail.
    on_disk = {p.expert for p in rows if p.action == "keep_on_disk"}
    assert on_disk == {3, 4}


def test_core_larger_than_num_experts_is_a_flagged_noop():
    plan = build_plan(_hot_profile(), Budget(core_experts=E + 4))
    assert plan.counts()["merge_into_core"] == E
    assert any("no-op" in w for w in plan.warnings)


def test_invalid_budget_rejected():
    with pytest.raises(ValueError, match="core_experts must be"):
        Budget(core_experts=0)
    with pytest.raises(ValueError, match="merge_threshold"):
        Budget(core_experts=2, merge_threshold=1.5)


# ----------------------------------------------------------------- validation


def test_validate_rejects_merge_into_a_non_survivor():
    plan = build_plan(_hot_profile(), Budget(core_experts=3))
    victim = next(p for p in plan.placements if p.expert == 7)
    victim.action = "drop"
    victim.merge_target = 99  # not in the layer
    with pytest.raises(ValueError, match="does not survive"):
        validate_plan(plan)


def test_validate_rejects_self_merge():
    plan = build_plan(_hot_profile(), Budget(core_experts=3))
    victim = next(p for p in plan.placements if p.expert == 7)
    victim.action = "drop"
    victim.merge_target = 7
    with pytest.raises(ValueError, match="merges into itself"):
        validate_plan(plan)


def test_validate_rejects_a_donor_that_is_not_dropped():
    """A merge donor's weights fold into the target, so it cannot also survive."""
    plan = build_plan(_hot_profile(), Budget(core_experts=3))
    victim = next(p for p in plan.placements if p.expert == 7)
    victim.merge_target = 0  # still keep_on_disk
    with pytest.raises(ValueError, match="must be action 'drop'"):
        validate_plan(plan)


def test_validate_rejects_duplicate_placements():
    plan = build_plan(_hot_profile(), Budget(core_experts=3))
    plan.placements.append(plan.placements[0])
    with pytest.raises(ValueError, match="duplicate placement"):
        validate_plan(plan)


def test_validate_rejects_a_layer_with_no_survivors():
    plan = Plan(model=None, revision=None, budget={}, placements=[])
    for expert in range(E):
        from vllm_moe_surgeon.surgery import ExpertPlacement

        plan.placements.append(
            ExpertPlacement(layer=0, expert=expert, action="drop", tokens=0, share=0.0)
        )
    with pytest.raises(ValueError, match="keep no experts"):
        validate_plan(plan)


# ------------------------------------------------------------------ round trip


def test_save_load_roundtrip_and_revalidate(tmp_path):
    similarity = _identity_similarity(0.1)
    similarity[7, 0] = similarity[0, 7] = 0.95
    plan = build_plan(
        _hot_profile(),
        Budget(core_experts=3),
        similarity={0: similarity},
        model="test/tiny",
        revision="abc",
    )
    path = str(tmp_path / "plan.json")
    plan.save(path)

    back = load_plan(path)
    assert back.model == "test/tiny"
    assert back.revision == "abc"
    assert back.counts() == plan.counts()
    assert next(p for p in back.by_layer(0) if p.expert == 7).merge_target == 0
    assert back.provenance["slots_per_expert"] == plan.provenance["slots_per_expert"]


def test_load_rejects_a_hand_edit_that_breaks_the_plan(tmp_path):
    """Plans are meant to be edited, so loading has to re-validate."""
    plan = build_plan(_hot_profile(), Budget(core_experts=3))
    path = str(tmp_path / "plan.json")
    plan.save(path)

    import json

    payload = json.loads(open(path).read())
    for placement in payload["placements"]:
        placement["action"] = "drop"
        placement["merge_target"] = None
    with open(path, "w") as f:
        json.dump(payload, f)

    with pytest.raises(ValueError, match="keep no experts"):
        load_plan(path)


def test_summary_reports_merged_versus_deleted_separately():
    """"Dropped" conflates two very different things; the report must not."""
    similarity = _identity_similarity(0.1)
    similarity[7, 0] = similarity[0, 7] = 0.95
    plan = build_plan(
        _hot_profile(),
        Budget(core_experts=3, drop_share_below=0.01),
        similarity={0: similarity},
    )
    text = summarize_plan(plan)
    assert "merged away" in text
    assert "deleted" in text
    assert "layer" in text


def test_no_merge_despite_similarity_is_explained_not_silent():
    """Measured reality on OLMoE: nothing comes close to the merge threshold.

    A plan that proposes no merges after being handed a similarity matrix looks
    like a bug. It usually is not, so the plan reports the ceiling it actually
    saw and names the alternative.
    """
    similarity = _identity_similarity(0.12)  # realistic: OLMoE median ~0.08
    plan = build_plan(
        _hot_profile(), Budget(core_experts=3), similarity={0: similarity}
    )
    assert all(p.merge_target is None for p in plan.placements)
    explanation = next(w for w in plan.warnings if "no merge cleared the bar" in w)
    assert "0.120" in explanation
    assert "activation-based" in explanation


def test_successful_merges_produce_no_such_warning():
    similarity = _identity_similarity(0.1)
    similarity[7, 0] = similarity[0, 7] = 0.95
    plan = build_plan(
        _hot_profile(), Budget(core_experts=3), similarity={0: similarity}
    )
    assert not any("no merge cleared the bar" in w for w in plan.warnings)


# --------------------------------------------------- rank-weighted importance


def _ranked_stats(counts, positions):
    """counts: [E]; positions: [E, top_k] histogram."""
    import numpy as np

    top_k = np.asarray(positions).shape[1]
    stats = ExpertStats.empty(E, [0], top_k=top_k)
    stats.tokens[0] = np.asarray(counts, dtype=np.int64)
    stats.positions[0] = np.asarray(positions, dtype=np.int64)
    stats.layer_token_rows[0] = int(np.sum(counts) // top_k) or 1
    stats.finalize()
    return stats


def _ordered_positions(counts, top_k=4, promote=None):
    """A position histogram that looks ordered at the population level.

    Hotter experts get more of their mass at slot 0, which is the regularity
    ``position_order_correlation`` keys on. ``promote`` names one mid-ranked expert
    whose mass is concentrated at slot 0 anyway, so rank-1 importance can disagree
    with total count without making the whole capture look unordered.
    """
    import numpy as np

    counts = np.asarray(counts, dtype=np.int64)
    order = np.argsort(counts)[::-1]
    positions = np.zeros((len(counts), top_k), dtype=np.int64)
    for rank, expert in enumerate(order):
        # Hottest expert concentrates at slot 0, coldest at the last slot.
        favourite = min(rank * top_k // max(len(counts), 1), top_k - 1)
        positions[expert, favourite] = counts[expert]
    if promote is not None:
        positions[promote] = 0
        positions[promote, 0] = counts[promote]
    return positions


def test_rank_one_ranking_can_disagree_with_count():
    """The finding this default exists for.

    Measured on OLMoE: ranking by rank-1 frequency instead of total count picked a
    keep-set that discarded MORE raw routing load (21.6% vs 14.6%) and cost LESS
    perplexity (1.30x vs 1.74x). So the two rankings must be able to differ.
    """
    import numpy as np

    counts = [9000, 6000, 3000, 2500, 2000, 1500, 1000, 500]
    # Expert 5 is mid-count but always the top choice when chosen.
    positions = _ordered_positions(counts, promote=5)
    stats = _ranked_stats(counts, positions)
    assert stats.positions_look_ordered, "fixture must look ordered, or the gate fires"

    by_count = list(np.argsort(stats.importance()[0])[::-1])
    by_rank1 = list(np.argsort(stats.importance(np.eye(4)[0])[0])[::-1])
    assert by_count != by_rank1, "the two definitions must be able to disagree"
    assert by_rank1.index(5) < by_count.index(5), "rank-1 must promote expert 5"


def test_build_plan_defaults_to_rank_one_and_records_it():
    counts = [9000, 6000, 3000, 2500, 2000, 1500, 1000, 500]
    stats = _ranked_stats(counts, _ordered_positions(counts, promote=5))

    plan = build_plan(stats, Budget(core_experts=3))
    assert plan.provenance["ranked_by"] == "rank-1 selection frequency"
    core = {p.expert for p in plan.by_layer(0) if p.action == "merge_into_core"}
    # Expert 5 is only 6th by count but always chosen first, so rank-1 keeps it.
    assert 5 in core


def test_unordered_capture_falls_back_to_counts_and_says_so():
    """Rank weighting on arbitrary-order ids would be noise, so it is refused."""
    import numpy as np

    # Every expert spread evenly over all slots -> no ordering signal.
    positions = np.full((E, 4), 100, dtype=np.int64)
    stats = _ranked_stats(positions.sum(axis=1), positions)
    assert not stats.positions_look_ordered

    plan = build_plan(stats, Budget(core_experts=2))
    assert "not score-ordered" in plan.provenance["ranked_by"]

    with pytest.raises(ValueError, match="not look score-ordered"):
        stats.importance(np.eye(4)[0])


def test_profile_without_positions_still_works():
    stats = _hot_profile()
    assert stats.positions is None
    plan = build_plan(stats, Budget(core_experts=3))
    assert "no position data" in plan.provenance["ranked_by"]


def test_importance_shape_is_validated():
    import numpy as np

    with pytest.raises(ValueError, match="importance has shape"):
        build_plan(_hot_profile(), Budget(core_experts=3), importance=np.zeros((2, 2)))
