# SPDX-License-Identifier: Apache-2.0
"""Least-squares router refit for merged clusters.

Merging folds a cluster of experts ``{survivor, donors...}`` into the survivor. The
donors' router rows go away, and the survivor's row must now route to it for the tokens
that used to go to *any* cluster member. The usage-weighted mean the writer uses
(:func:`~.apply._blend_expert_rows`) cannot express that: a softmax gate has no bias
term, so the cluster's **combined selection mass** -- which is additive across members,
not an average of their rows -- has nowhere to live in one averaged row. That is the
gap in merging's router rewrite: it "cannot represent the
cluster's summed selection mass."

The principled target is the cluster's log-sum-exp. For a softmax, the probability mass
on a cluster ``C`` is ``exp(logsumexp_{e in C} l_e) / Z``, so the single row that
reproduces that marginal -- against unchanged other-survivor rows -- is the one whose
logit equals the log-sum-exp of the members' logits. Fitting it is ordinary linear
least squares on a batch of the layer's hidden states:

    w_new = argmin_w || X @ w - logsumexp(X @ W_C.T, axis=1) ||^2

solved by :func:`numpy.linalg.lstsq`. No labels, no engine -- only hidden states, which
``surgeon calibrate`` already collects. This module is the machinery; it is kept and
tested exactly (like the merge alignment in :mod:`.align`) even though the models
measured here have no mergeable pair to exercise it on.
"""

from __future__ import annotations

import numpy as np


def cluster_target_logits(hidden: np.ndarray, cluster_rows: np.ndarray) -> np.ndarray:
    """``logsumexp`` of the cluster members' logits per token: the combined mass.

    ``hidden`` is ``[N, H]``, ``cluster_rows`` is ``[k, H]`` (survivor first, then
    donors). Returns ``[N]``. Computed in the numerically stable shifted form so a
    large logit does not overflow.
    """
    logits = np.asarray(hidden, dtype=np.float64) @ np.asarray(
        cluster_rows, dtype=np.float64
    ).T
    m = logits.max(axis=1, keepdims=True)
    return m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))


def refit_cluster_router(hidden: np.ndarray, cluster_rows: np.ndarray) -> np.ndarray:
    """The survivor router row that best reproduces the cluster's selection mass.

    ``cluster_rows[0]`` is the survivor's original row; the rest are donors. Returns a
    single row ``[H]``. With no donors it reduces to the survivor's own row: the
    log-sum-exp of one member is that member's logit, and the fit is exact on a
    full-rank batch. By construction the fit's residual is never worse than the
    usage-weighted mean's -- least squares is optimal against this target.
    """
    hidden = np.asarray(hidden, dtype=np.float64)
    cluster_rows = np.asarray(cluster_rows, dtype=np.float64)
    if cluster_rows.ndim != 2 or hidden.ndim != 2:
        raise ValueError("hidden must be [N, H] and cluster_rows [k, H]")
    if hidden.shape[1] != cluster_rows.shape[1]:
        raise ValueError(
            f"hidden dim {hidden.shape[1]} != router row dim {cluster_rows.shape[1]}"
        )
    target = cluster_target_logits(hidden, cluster_rows)
    w, *_ = np.linalg.lstsq(hidden, target, rcond=None)
    return w


def refit_residual(hidden: np.ndarray, cluster_rows: np.ndarray,
                   row: np.ndarray) -> float:
    """RMS error of ``row`` against the cluster's log-sum-exp target, for scoring."""
    target = cluster_target_logits(hidden, cluster_rows)
    pred = np.asarray(hidden, dtype=np.float64) @ np.asarray(row, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - target) ** 2)))
