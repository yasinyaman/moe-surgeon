# SPDX-License-Identifier: Apache-2.0
"""The router refit machinery, tested exactly on synthetic clusters.

No model has a mergeable pair to exercise this on (three families screened, max
subspace similarity 0.33-0.40), so these tests pin the math directly: the refit
reproduces a single expert exactly, beats the usage-weighted mean it replaces against
the log-sum-exp target, and refuses malformed input.
"""

from __future__ import annotations

import numpy as np
import pytest

from vllm_moe_surgeon.surgery.refit import (
    cluster_target_logits,
    refit_cluster_router,
    refit_residual,
)

H = 16
N = 512


def _rng():
    return np.random.default_rng(0)


def test_a_single_expert_cluster_returns_its_own_row():
    """log-sum-exp of one member is that member's logit, so with no donors the fit is
    the survivor's own row -- merging nothing changes nothing."""
    rng = _rng()
    hidden = rng.standard_normal((N, H))
    row = rng.standard_normal((1, H))
    refit = refit_cluster_router(hidden, row)
    np.testing.assert_allclose(refit, row[0], atol=1e-8)


def test_the_target_is_the_log_sum_exp_of_member_logits():
    rng = _rng()
    hidden = rng.standard_normal((8, H))
    rows = rng.standard_normal((3, H))
    logits = hidden @ rows.T
    want = np.log(np.exp(logits).sum(axis=1))  # naive, fine at this scale
    np.testing.assert_allclose(cluster_target_logits(hidden, rows), want, atol=1e-9)


def test_the_refit_beats_the_usage_weighted_mean():
    """Least squares is optimal against the log-sum-exp target, so its residual is
    never worse than the mean the writer would otherwise use -- and is strictly better
    when the donors carry real mass."""
    rng = _rng()
    hidden = rng.standard_normal((N, H))
    rows = rng.standard_normal((3, H))
    weights = np.array([0.5, 0.3, 0.2])

    mean_row = (rows * weights[:, None]).sum(axis=0) / weights.sum()
    refit = refit_cluster_router(hidden, rows)

    r_mean = refit_residual(hidden, rows, mean_row)
    r_refit = refit_residual(hidden, rows, refit)
    assert r_refit <= r_mean + 1e-9
    assert r_refit < r_mean  # donors carry mass, so the fit is strictly better


def test_the_missing_bias_makes_multi_expert_clusters_irreducibly_lossy():
    """A softmax gate has no bias, so the log-sum-exp's ~log(k) constant offset cannot
    be fit by any row through the origin -- the reason merging is lossy in the first
    place. The refit is the *best* such row, but for a real cluster its residual stays
    bounded away from zero, unlike the exact single-expert case."""
    rng = _rng()
    hidden = rng.standard_normal((N, H))
    one = rng.standard_normal((1, H))
    three = rng.standard_normal((3, H))
    r_one = refit_residual(hidden, one, refit_cluster_router(hidden, one))
    r_three = refit_residual(hidden, three, refit_cluster_router(hidden, three))
    assert r_one < 1e-6  # single expert: linear target, exact fit
    assert r_three > 0.1  # cluster: the summed mass has no bias to live in


def test_malformed_input_is_refused():
    rng = _rng()
    with pytest.raises(ValueError, match="hidden must be"):
        refit_cluster_router(rng.standard_normal(H), rng.standard_normal((1, H)))
    with pytest.raises(ValueError, match="!="):
        refit_cluster_router(
            rng.standard_normal((N, H)), rng.standard_normal((1, H + 1))
        )
