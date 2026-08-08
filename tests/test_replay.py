# SPDX-License-Identifier: Apache-2.0
"""The cache replay, checked on traces whose right answer is known by hand.

A simulator that silently miscounts would make the prior look good or bad for the
wrong reason, so the accounting is pinned on tiny traces first: a token needing
top_k experts costs top_k accesses, and misses inside one token are charged
individually.
"""

from __future__ import annotations

import numpy as np
import pytest

from vllm_moe_surgeon.store.expert_policy import EWMAPolicy, LFRUPolicy
from vllm_moe_surgeon.store.replay import (
    ReplayResult,
    replay_captures,
    replay_layer,
    summarize,
)


def _lfru(n):
    return LFRUPolicy(n)


def _ewma(n):
    return EWMAPolicy(n)


# ------------------------------------------------------------------ accounting


def test_every_expert_of_a_token_is_charged():
    """A token needing 2 experts is 2 accesses, not 1."""
    result = replay_layer(
        [[0, 1], [0, 1]], num_experts=4, capacity=4, policy_factory=_lfru
    )
    assert result.accesses == 4
    assert result.misses == 2  # first token cold
    assert result.hits == 2  # second token both resident


def test_a_cache_that_holds_everything_misses_only_once_per_expert():
    accesses = [[0, 1], [2, 3], [0, 1], [2, 3]]
    result = replay_layer(
        accesses, num_experts=4, capacity=4, policy_factory=_lfru
    )
    assert result.misses == 4
    assert result.hits == 4
    assert result.hit_rate == 0.5


def test_a_tight_cache_thrashes():
    """Alternating disjoint pairs with room for two must miss every time."""
    accesses = [[0, 1], [2, 3]] * 5
    result = replay_layer(
        accesses, num_experts=4, capacity=2, policy_factory=_lfru
    )
    assert result.hits == 0
    assert result.misses == 20


def test_sentinels_are_not_charged():
    result = replay_layer(
        [[0, -1], [0, -1]], num_experts=4, capacity=4, policy_factory=_lfru
    )
    assert result.accesses == 2


def test_capacity_below_a_token_width_is_refused():
    """The tier would have to split the forward pass; silence would mislead."""
    with pytest.raises(RuntimeError, match="cannot hold one token"):
        replay_layer(
            [[0, 1, 2]], num_experts=8, capacity=2, policy_factory=_lfru
        )


def test_zero_capacity_rejected():
    with pytest.raises(ValueError, match="capacity must be"):
        replay_layer([[0]], num_experts=2, capacity=0, policy_factory=_lfru)


def test_an_expert_needed_by_the_current_token_is_never_evicted():
    """The prototype's silent-corruption bug, kept out by construction.

    A freshly loaded expert has the lowest score, so a naive victim scan would
    evict it mid-token and reuse its slot while the mapping still pointed there.
    """
    # Token needs 3 of a 3-slot cache, then a 4th expert arrives next token.
    accesses = [[0, 1, 2], [3, 0, 1]]
    result = replay_layer(
        accesses, num_experts=8, capacity=3, policy_factory=_lfru
    )
    # 3 cold misses, then expert 3 misses and 0/1 hit -- no corruption, no raise.
    assert result.misses == 4
    assert result.hits == 2


# ---------------------------------------------------------------- the prior


def test_prewarming_removes_the_cold_start_misses():
    """The whole point of a residency prior, on a trace where it must show."""
    accesses = [[0, 1]] * 10
    cold = replay_layer(
        accesses, num_experts=8, capacity=2, policy_factory=_ewma
    )
    warm = replay_layer(
        accesses,
        num_experts=8,
        capacity=2,
        policy_factory=_ewma,
        prior={0: 8.0, 1: 8.0},
        prewarm=[0, 1],
    )
    assert cold.misses == 2
    assert warm.misses == 0
    assert warm.hit_rate > cold.hit_rate


def test_prewarm_cannot_exceed_capacity():
    warm = replay_layer(
        [[0, 1]] * 4,
        num_experts=8,
        capacity=2,
        policy_factory=_ewma,
        prewarm=[0, 1, 2, 3, 4],
    )
    assert warm.accesses == 8


def test_a_wrong_prior_costs_only_latency_not_correctness():
    """Warming the wrong experts must not break anything, just miss."""
    accesses = [[0, 1]] * 5
    warm_wrong = replay_layer(
        accesses,
        num_experts=8,
        capacity=2,
        policy_factory=_ewma,
        prior={6: 8.0, 7: 8.0},
        prewarm=[6, 7],
    )
    assert warm_wrong.hits + warm_wrong.misses == 10
    assert warm_wrong.misses >= 2


def test_prior_on_a_policy_without_one_degrades_to_cold():
    """LFRU has no prior table, so the warm arm must not silently claim a win."""
    accesses = [[0, 1], [2, 3]] * 4
    cold = replay_layer(accesses, num_experts=8, capacity=2, policy_factory=_lfru)
    warm = replay_layer(
        accesses,
        num_experts=8,
        capacity=2,
        policy_factory=_lfru,
        prior={0: 100.0},
    )
    assert warm.hit_rate == cold.hit_rate


# ------------------------------------------------------------- prefix window


def test_prefix_hit_rate_scores_only_the_window():
    result = ReplayResult(hits=1, misses=3, outcomes=[False, False, False, True])
    assert result.prefix_hit_rate(2) == 0.0
    assert result.prefix_hit_rate(4) == 0.25
    assert result.prefix_hit_rate(100) == 0.25
    assert result.prefix_hit_rate(0) == 0.0


# ----------------------------------------------------------------- captures


def _capture(rows_by_layer, num_layers, top_k=2):
    seq = len(next(iter(rows_by_layer.values())))
    arr = np.zeros((seq, num_layers, top_k), dtype=np.int32)
    for layer, rows in rows_by_layer.items():
        arr[:, layer, :] = np.asarray(rows, dtype=np.int32)
    return arr


def test_each_layer_gets_its_own_cache():
    """Layer 0 thrashing must not affect layer 1's hit rate."""
    capture = _capture(
        {0: [[0, 1], [2, 3], [0, 1], [2, 3]], 1: [[0, 1]] * 4},
        num_layers=2,
    )
    results = replay_captures(
        [capture],
        moe_layers=[0, 1],
        num_experts=4,
        capacity=2,
        policy_factory=_lfru,
    )
    assert results[0].hits == 0  # thrashes
    assert results[1].hits == 6  # 1 cold token, then 3 hits x 2 experts


def test_uncaptured_rows_are_dropped_before_replay():
    """All-identical rows are dense-layer padding, not routing."""
    capture = _capture({0: [[0, 0], [1, 2]]}, num_layers=1)
    results = replay_captures(
        [capture],
        moe_layers=[0],
        num_experts=4,
        capacity=4,
        policy_factory=_lfru,
    )
    assert results[0].accesses == 2  # only the genuine row


def test_summarize_sums_layers():
    capture = _capture({0: [[0, 1]] * 3, 1: [[0, 1]] * 3}, num_layers=2)
    results = replay_captures(
        [capture],
        moe_layers=[0, 1],
        num_experts=4,
        capacity=2,
        policy_factory=_lfru,
    )
    total = summarize(results)
    assert total.accesses == results[0].accesses + results[1].accesses
    assert total.hits == results[0].hits + results[1].hits
