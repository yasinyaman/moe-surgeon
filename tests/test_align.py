# SPDX-License-Identifier: Apache-2.0
"""Neuron alignment, and the exactness property that makes merging meaningful.

The decisive test: merge an expert with a *permuted copy of itself* and the
result must be the original, bit-for-bit. That only holds if the alignment
recovers the shuffle exactly. Without alignment, the same merge produces
something resembling neither input -- and that failure is silent, which is why it
is asserted here rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from vllm_moe_surgeon.surgery.align import (
    align_donor_to_target,
    apply_permutation,
    cost_matrix,
    merge_experts,
    neuron_features,
    solve_assignment,
)

H = 16
INTER = 12


def _expert(seed: int):
    rng = np.random.default_rng(seed)
    gate = rng.standard_normal((INTER, H)).astype(np.float32)
    up = rng.standard_normal((INTER, H)).astype(np.float32)
    down = rng.standard_normal((H, INTER)).astype(np.float32)
    return gate, up, down


def _ffn(expert, x):
    """SwiGLU forward, to check a permutation really is function-preserving."""
    gate, up, down = expert
    hidden = (x @ gate.T)
    activated = hidden / (1.0 + np.exp(-hidden)) * (x @ up.T)  # silu(gate) * up
    return activated @ down.T


# ---------------------------------------------------------------- the symmetry


def test_permuting_neurons_preserves_the_function():
    """The premise. If this fails, the whole approach is misconceived."""
    expert = _expert(1)
    rng = np.random.default_rng(2)
    perm = rng.permutation(INTER)
    permuted = apply_permutation(*expert, perm)

    x = rng.standard_normal((5, H)).astype(np.float32)
    np.testing.assert_allclose(_ffn(expert, x), _ffn(permuted, x), rtol=1e-5, atol=1e-5)


def test_permuting_only_some_matrices_breaks_the_function():
    """Why apply_permutation touches all three together.

    Reordering gate/up rows without reordering down's columns rewires the expert
    into a different function -- and nothing raises.
    """
    gate, up, down = _expert(3)
    rng = np.random.default_rng(4)
    perm = rng.permutation(INTER)
    half_done = (gate[perm], up[perm], down)  # down left alone: wrong

    x = rng.standard_normal((5, H)).astype(np.float32)
    assert not np.allclose(
        _ffn((gate, up, down), x), _ffn(half_done, x), rtol=1e-3, atol=1e-3
    )


# ---------------------------------------------------------------- assignment


def test_alignment_recovers_an_exact_shuffle():
    expert = _expert(5)
    rng = np.random.default_rng(6)
    perm = rng.permutation(INTER)
    donor = apply_permutation(*expert, perm)

    recovered = align_donor_to_target(expert, donor)
    # donor neuron recovered[i] must be target neuron i. donor row k came from
    # target row perm[k], so the inverse of perm is what we expect.
    np.testing.assert_array_equal(recovered, np.argsort(perm))


def test_realigned_donor_equals_the_target_exactly():
    expert = _expert(7)
    rng = np.random.default_rng(8)
    donor = apply_permutation(*expert, rng.permutation(INTER))

    perm = align_donor_to_target(expert, donor)
    restored = apply_permutation(*donor, perm)
    for original, back in zip(expert, restored, strict=True):
        np.testing.assert_allclose(original, back, rtol=0, atol=0)


def test_cost_matrix_diagonal_is_one_for_self_comparison():
    expert = _expert(9)
    features = neuron_features(*expert)
    scores = cost_matrix(features, features)
    np.testing.assert_allclose(np.diag(scores), 1.0, atol=1e-5)
    assert scores.argmax(axis=1).tolist() == list(range(INTER))


def test_assignment_is_a_permutation():
    rng = np.random.default_rng(10)
    scores = rng.random((INTER, INTER)).astype(np.float32)
    perm = solve_assignment(scores)
    assert sorted(perm.tolist()) == list(range(INTER))


def test_non_square_cost_matrix_rejected():
    with pytest.raises(ValueError, match="square cost matrix"):
        solve_assignment(np.zeros((3, 4), dtype=np.float32))


def test_greedy_matches_the_solver_on_an_exact_permutation():
    """The SciPy-free fallback must be correct in the case that matters."""
    from vllm_moe_surgeon.surgery.align import _greedy_assignment

    expert = _expert(11)
    rng = np.random.default_rng(12)
    donor = apply_permutation(*expert, rng.permutation(INTER))
    scores = cost_matrix(neuron_features(*expert), neuron_features(*donor))

    np.testing.assert_array_equal(_greedy_assignment(scores), solve_assignment(scores))


def test_dead_neuron_does_not_produce_nan():
    """An all-zero neuron has no direction; it must not poison the cost matrix."""
    gate, up, down = _expert(13)
    gate[3] = 0.0
    up[3] = 0.0
    down[:, 3] = 0.0
    features = neuron_features(gate, up, down)
    scores = cost_matrix(features, features)
    assert not np.isnan(scores).any()


def test_mismatched_shapes_rejected():
    gate, up, down = _expert(14)
    with pytest.raises(ValueError, match="must match"):
        neuron_features(gate, up[:-1], down)
    with pytest.raises(ValueError, match="neuron columns"):
        neuron_features(gate, up, down[:, :-1])


# -------------------------------------------------------------------- merging


def test_merging_an_expert_with_its_permuted_copy_returns_the_original():
    """The property the whole merge path stands on.

    Two functionally identical experts must merge back to that same function.
    Any residual here is misalignment, not rounding.
    """
    expert = _expert(15)
    rng = np.random.default_rng(16)
    donor = apply_permutation(*expert, rng.permutation(INTER))

    merged = merge_experts(expert, [(donor, 1.0)], target_weight=1.0)
    for original, result in zip(expert, merged, strict=True):
        np.testing.assert_allclose(original, result, rtol=1e-6, atol=1e-6)


def test_naive_averaging_without_alignment_destroys_the_expert():
    """The contrast that justifies the alignment step's cost.

    Averaging the same two functionally-identical experts elementwise, in their
    given neuron order, produces a function unlike either one.
    """
    expert = _expert(17)
    rng = np.random.default_rng(18)
    donor = apply_permutation(*expert, rng.permutation(INTER))

    naive = tuple((a + b) / 2.0 for a, b in zip(expert, donor, strict=True))
    x = rng.standard_normal((8, H)).astype(np.float32)

    reference = _ffn(expert, x)
    aligned = _ffn(merge_experts(expert, [(donor, 1.0)], 1.0), x)
    unaligned = _ffn(naive, x)

    aligned_error = np.linalg.norm(aligned - reference)
    naive_error = np.linalg.norm(unaligned - reference)
    assert aligned_error < 1e-3
    assert naive_error > 10 * max(aligned_error, 1e-6), (
        f"naive error {naive_error:.4f} should dwarf aligned {aligned_error:.2e}"
    )


def test_usage_weighting_biases_toward_the_heavier_expert():
    target = _expert(19)
    donor = _expert(20)
    mostly_target = merge_experts(target, [(donor, 1.0)], target_weight=999.0)
    for original, result in zip(target, mostly_target, strict=True):
        np.testing.assert_allclose(original, result, rtol=0.02, atol=0.02)


def test_multiple_donors_are_each_aligned_to_the_target():
    """Donors align to the target, not to each other -- one common frame."""
    expert = _expert(21)
    rng = np.random.default_rng(22)
    donors = [
        (apply_permutation(*expert, rng.permutation(INTER)), 1.0) for _ in range(3)
    ]
    merged = merge_experts(expert, donors, target_weight=1.0)
    for original, result in zip(expert, merged, strict=True):
        np.testing.assert_allclose(original, result, rtol=1e-6, atol=1e-6)


def test_zero_total_weight_is_refused():
    expert = _expert(23)
    with pytest.raises(ValueError, match="sum to zero"):
        merge_experts(expert, [(_expert(24), 0.0)], target_weight=0.0)
