# SPDX-License-Identifier: Apache-2.0
"""Strategy selection, and the rule that makes it worth having.

The register's point is that a method losing on one axis is still chosen when it wins
on another. These tests pin that behaviour rather than the wording: deletion is
recommended when it buys feasibility, refused when it would only cost accuracy, and
the tier flips from optional to mandatory purely on the target's VRAM.
"""

from __future__ import annotations

import numpy as np

from vllm_moe_surgeon.surgery.budget import CheckpointGeometry
from vllm_moe_surgeon.surgery.recommend import dead_tail_size, recommend
from vllm_moe_surgeon.telemetry.stats import ExpertStats

E = 16


def _geometry(**over):
    base = dict(
        num_experts=E,
        top_k=4,
        moe_layers=list(range(4)),
        hidden=512,
        intermediate=256,
        expert_bytes=0,
        routed_expert_bytes=4 << 30,
        shared_expert_bytes=0,
        router_bytes=1 << 20,
        other_bytes=512 << 20,
        checkpoint_dtype_bytes=2,
    )
    base.update(over)
    return CheckpointGeometry(**base)


def _stats(per_layer_counts):
    stats = ExpertStats.empty(E, list(range(len(per_layer_counts))))
    for slot, counts in enumerate(per_layer_counts):
        stats.tokens[slot] = np.asarray(counts, dtype=np.int64)
        stats.layer_token_rows[slot] = int(np.sum(counts))
    stats.finalize()
    return stats


def _flat():
    return _stats([[1000] * E] * 4)


def _with_dead_tail():
    counts = [4000, 3000, 2000, 1500] + [1000] * 8 + [1, 1, 1, 1]
    return _stats([counts] * 4)


# ------------------------------------------------------------- the tier switch


def test_tier_is_mandatory_when_the_model_does_not_fit():
    rec = recommend(_geometry(), vram_gib=1.0, stats=_flat(), max_similarity=0.1)
    assert rec.use_tier
    assert any("mandatory" in r for r in rec.reasons)


def test_tier_is_optional_when_the_model_fits():
    rec = recommend(_geometry(), vram_gib=64.0, stats=_flat(), max_similarity=0.1)
    assert any("optional" in r for r in rec.reasons)


def test_latency_sensitive_and_vram_rich_declines_the_tier():
    """Losing 0.38x decode is not worth paying when nothing forces it."""
    rec = recommend(
        _geometry(),
        vram_gib=64.0,
        stats=_flat(),
        max_similarity=0.1,
        latency_sensitive=True,
    )
    assert not rec.use_tier
    assert any("latency-sensitive" in r for r in rec.reasons)


def test_an_impossible_target_warns_rather_than_recommending_a_fantasy():
    rec = recommend(
        _geometry(other_bytes=32 << 30), vram_gib=1.0, kv_cache_gib=0.5,
        stats=_flat(), max_similarity=0.1,
    )
    assert any("does not fit" in w for w in rec.warnings)


# ---------------------------------------------------------------- deletion


def test_a_flat_distribution_is_not_pruned():
    """No dead tail means deletion trades accuracy for very little."""
    rec = recommend(_geometry(), vram_gib=1.0, stats=_flat(), max_similarity=0.1)
    assert rec.delete_experts == 0
    assert any("no dead tail" in r for r in rec.reasons)


def test_a_dead_tail_is_not_pruned_when_the_tier_fits():
    """New policy: the tier keeps every expert at no accuracy cost, so a cold tail is
    left on the tier rather than deleted, even though the model does not fit
    resident. Deletion is a measured task-accuracy loss, not a free win."""
    rec = recommend(
        _geometry(), vram_gib=1.0, stats=_with_dead_tail(), max_similarity=0.1
    )
    assert rec.use_tier
    assert rec.delete_experts == 0
    assert any("NOT recommended" in r for r in rec.reasons)


def test_deletion_is_the_last_resort_only_when_the_tier_cannot_fit():
    """Below the tier's own floor, a smaller model is the only way to run; deletion is
    then recommended -- with the task-accuracy warning and a deferral to the gate."""
    # A tiny target below the tier floor forces deletion.
    rec = recommend(
        _geometry(), vram_gib=0.2, stats=_with_dead_tail(), max_similarity=0.1
    )
    assert rec.delete_experts != 0
    assert any("last resort" in r for r in rec.reasons)
    assert any("task accuracy" in w for w in rec.warnings)
    assert any("surgeon gate" in m for m in rec.must_measure)


def test_dead_tail_is_counted_on_the_worst_layer():
    """One num_experts for the model means the least prunable layer sets the budget."""
    prunable = [4000, 3000] + [1000] * 10 + [1, 1, 1, 1]
    flat = [1000] * E
    stats = _stats([prunable, flat])
    assert dead_tail_size(stats.layer_share(), E) == 0


# ------------------------------------------------------------------ merging


def test_merging_is_offered_only_when_a_pair_clears_the_threshold():
    hot = recommend(_geometry(), vram_gib=1.0, stats=_flat(), max_similarity=0.92)
    assert hot.merge_enabled
    cold = recommend(_geometry(), vram_gib=1.0, stats=_flat(), max_similarity=0.37)
    assert not cold.merge_enabled
    assert any("nothing to merge" in r for r in cold.reasons)


# ----------------------------------------------------- what it will not guess


def test_missing_inputs_are_reported_not_assumed():
    rec = recommend(_geometry())
    assert any("no target VRAM" in m for m in rec.must_measure)
    assert any("no profile" in m for m in rec.must_measure)
    assert any("no similarity" in m for m in rec.must_measure)
    assert rec.delete_experts == 0
    assert not rec.merge_enabled


def test_the_tier_fitting_makes_deletion_unneeded_without_a_profile():
    """When VRAM is known and the tier fits, deletion is off with no profile needed."""
    rec = recommend(_geometry(), vram_gib=64.0)
    assert rec.delete_experts == 0
    assert any("deletion is not needed" in r for r in rec.reasons)


def test_the_prior_is_recommended_whenever_the_tier_is_used():
    """It wins on cold start only, and that is enough to keep it."""
    rec = recommend(_geometry(), vram_gib=1.0, stats=_flat(), max_similarity=0.1)
    assert rec.seed_prior
    assert rec.cache_policy == "ewma"
    assert any("EWMA" in r for r in rec.reasons)


def test_report_is_readable_and_carries_the_reasons():
    # vram=None leaves feasibility (and therefore deletion) undecided, so the report
    # carries a "not decidable here" section.
    rec = recommend(_geometry(), stats=_with_dead_tail(), max_similarity=0.1)
    text = rec.report()
    assert "recommendation:" in text
    assert "because:" in text
    assert "not decidable here" in text
    assert "delete" in text


def test_last_resort_deletion_reports_as_such():
    rec = recommend(_geometry(), vram_gib=0.2, max_similarity=0.1)
    assert rec.delete_experts == -1
    assert "last resort" in rec.report()


def test_every_deletion_recommendation_asks_for_the_headroom_check_first():
    """Deletion buys footprint at a measured quality cost; measured once, a
    smaller off-the-shelf checkpoint bought the same footprint for free. So the
    cheap check is a prerequisite of recommending deletion, not a footnote."""
    from vllm_moe_surgeon.surgery.recommend import recommend

    # Both branches that recommend deletion: with a profile (a sized cut) and
    # without one (the unsized last resort). Neither may propose deletion
    # without first asking whether an existing checkpoint makes it unnecessary.
    sized = recommend(
        _geometry(), vram_gib=0.2, stats=_with_dead_tail(), max_similarity=0.1
    )
    unsized = recommend(_geometry(), vram_gib=0.2, max_similarity=0.1)

    for rec in (sized, unsized):
        assert rec.delete_experts != 0
        assert any("surgeon headroom" in m for m in rec.must_measure)
        assert any("bits/byte" in m for m in rec.must_measure)
    # And it is a prerequisite, not a caveat: it sits with what must be
    # measured, not with the warnings.
    assert not any("surgeon headroom" in w for w in sized.warnings)
