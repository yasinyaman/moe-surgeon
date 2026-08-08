# SPDX-License-Identifier: Apache-2.0
"""Aligning two experts' intermediate neurons before averaging them.

:mod:`.descriptors` establishes *whether* two experts are similar in a way that
ignores the FFN's neuron-permutation symmetry. This module does the other half:
once a merge is decided, the donor's neurons have to be put into the target's
order before any averaging happens.

Skipping this is not a small error. Neuron ``i`` of the donor and neuron ``i`` of
the target are unrelated -- their numbering is an accident of training -- so a
straight elementwise mean of two functionally identical experts produces
something that resembles neither. Averaging is only meaningful along a matched
permutation.

A neuron is identified by everything attached to it: row ``i`` of ``gate_proj``,
row ``i`` of ``up_proj``, and column ``i`` of ``down_proj``. Concatenating those
gives a feature vector in :math:`\\mathbb{R}^{3H}`, and the alignment is the
assignment between donor and target neurons that maximises total cosine
similarity -- a linear assignment problem.

The exactness test is the one that matters: align an expert against a permuted
copy of itself and the recovered permutation must invert the shuffle exactly, so
merging the two returns the original bit-for-bit.
"""

from __future__ import annotations

import numpy as np

from .._logging import init_logger

logger = init_logger(__name__)


def neuron_features(
    gate: np.ndarray, up: np.ndarray, down: np.ndarray
) -> np.ndarray:
    """``[I, 3H]`` -- one row per intermediate neuron.

    ``gate`` and ``up`` are ``[I, H]``; ``down`` is ``[H, I]``. The neuron's
    share of ``down`` is a *column*, which is why it is transposed in.
    """
    if gate.shape != up.shape:
        raise ValueError(f"gate {gate.shape} and up {up.shape} must match")
    if down.shape[1] != gate.shape[0]:
        raise ValueError(
            f"down {down.shape} does not have {gate.shape[0]} neuron columns"
        )
    return np.concatenate(
        [gate, up, down.T], axis=1, dtype=np.float32, casting="unsafe"
    )


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A dead neuron (all-zero row) has no direction; leave it at zero so it
    # matches everything equally badly rather than producing NaNs.
    np.maximum(norms, 1e-12, out=norms)
    return matrix / norms


def cost_matrix(target: np.ndarray, donor: np.ndarray) -> np.ndarray:
    """``C[i, j]`` = cosine similarity of target neuron ``i`` to donor ``j``."""
    return _l2_normalize(target) @ _l2_normalize(donor).T


def _greedy_assignment(scores: np.ndarray) -> np.ndarray:
    """Highest-scoring pair first, without replacement.

    The fallback when SciPy is absent. Not optimal, but for the case that
    matters -- a genuine permutation of the same neurons, where the true match
    scores 1.0 and everything else is far below -- greedy and optimal agree.
    """
    n = scores.shape[0]
    order = np.argsort(scores, axis=None)[::-1]
    rows, cols = np.unravel_index(order, scores.shape)
    assignment = np.full(n, -1, dtype=np.int64)
    taken = np.zeros(scores.shape[1], dtype=bool)
    filled = 0
    for row, col in zip(rows, cols, strict=True):
        if assignment[row] != -1 or taken[col]:
            continue
        assignment[row] = col
        taken[col] = True
        filled += 1
        if filled == n:
            break
    return assignment


def solve_assignment(scores: np.ndarray) -> np.ndarray:
    """``perm`` such that donor neuron ``perm[i]`` corresponds to target ``i``."""
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"expected a square cost matrix, got {scores.shape}")
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        logger.warning_once(
            "SciPy is not installed; falling back to greedy neuron alignment. "
            "For an exact permutation the two agree, but for a partial match "
            "greedy is not optimal -- install scipy for the Hungarian solver."
        )
        return _greedy_assignment(scores)
    rows, cols = linear_sum_assignment(scores, maximize=True)
    perm = np.empty(scores.shape[0], dtype=np.int64)
    perm[rows] = cols
    return perm


def align_donor_to_target(
    target: tuple[np.ndarray, np.ndarray, np.ndarray],
    donor: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Neuron permutation putting ``donor`` into ``target``'s order.

    Each argument is ``(gate, up, down)``.
    """
    return solve_assignment(
        cost_matrix(neuron_features(*target), neuron_features(*donor))
    )


def apply_permutation(
    gate: np.ndarray, up: np.ndarray, down: np.ndarray, perm: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reorder one expert's neurons. The function it computes is unchanged.

    The permutation must hit ``gate``/``up`` rows and ``down`` columns with the
    *same* indices -- applying it to only some of the three silently rewires the
    expert into a different function.
    """
    return gate[perm], up[perm], down[:, perm]


def merge_experts(
    target: tuple[np.ndarray, np.ndarray, np.ndarray],
    donors: list[tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float]],
    target_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Usage-weighted average of ``target`` and each aligned donor.

    ``donors`` is ``[((gate, up, down), weight), ...]``; weights are token
    counts, so a rarely-used donor perturbs the target only slightly.

    Every donor is aligned to the *target's* neuron order, not to each other, so
    the result stays in a single consistent frame.
    """
    weights = [target_weight] + [w for _, w in donors]
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("merge weights sum to zero; cannot form an average")

    gate = np.asarray(target[0], dtype=np.float32) * (target_weight / total)
    up = np.asarray(target[1], dtype=np.float32) * (target_weight / total)
    down = np.asarray(target[2], dtype=np.float32) * (target_weight / total)

    for donor, weight in donors:
        perm = align_donor_to_target(target, donor)
        d_gate, d_up, d_down = apply_permutation(*donor, perm)
        scale = weight / total
        gate += np.asarray(d_gate, dtype=np.float32) * scale
        up += np.asarray(d_up, dtype=np.float32) * scale
        down += np.asarray(d_down, dtype=np.float32) * scale

    return gate, up, down
