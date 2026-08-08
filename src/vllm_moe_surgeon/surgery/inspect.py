# SPDX-License-Identifier: Apache-2.0
"""What routing signals does this checkpoint actually carry -- and do they work?

Static analysis of a router is appealing because it is nearly free: no engine, no
GPU, no calibration data. The signals people reach for are the gate matrix
geometry (row norms as a popularity proxy, row cosine as a co-selection proxy),
the learned load-balancing bias where a model has one, and the per-expert weight
spectrum as a quantization-tolerance proxy.

The trouble is that whether any of them carries information is a property of the
*model*, not of the idea. So this module does two things, and the second is the
point:

1. report which signals a checkpoint contains at all, and
2. when a measured profile is available, **score each signal against it** --
   rank correlation with real per-expert load and real co-occurrence.

That inverts the usual order. Instead of adopting a static heuristic and hoping,
you get a number saying whether it predicts anything on the model in front of you.

Measured on OLMoE-1B-7B against a 400-prompt profile, both gate-geometry signals
came back at rank correlation ~0.00 (see the project README), while real load
spread 18.5x across experts -- strong structure that the static signals captured
none of. Centering the gate rows to remove their shared component did not help.
That is why this package derives residency priors from profiles rather than from
weights, and why this tool exists to check rather than assume.

The one static signal with a principled basis is the aux-loss-free balancing bias
(``e_score_correction_bias``, DeepSeek-V3 style): it is a *learned* correction that
the optimizer pushed down for over-selected experts, so a large negative value is
a direct popularity readout rather than a proxy. Neither OLMoE nor Qwen3-MoE has
one, so it is detected and reported but has not been validated here.

Where static signals stay safe regardless: placement and prefetch, where a wrong
guess costs a cache miss. Never pruning or merging, where a wrong guess costs
quality and cannot be undone from the artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._logging import init_logger
from .descriptors import CheckpointIndex, _as_float32

logger = init_logger(__name__)

#: Candidate names for the aux-loss-free balancing bias, across model families.
BIAS_SUFFIXES = (
    "mlp.gate.e_score_correction_bias",
    "mlp.gate.bias",
)


def _midranks(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged.

    The naive ``argsort(argsort(x))`` is wrong here in a way that matters: it
    hands tied values *sequential* ranks, so a completely flat signal comes out
    perfectly rank-correlated with anything. A checkpoint that ships a
    zero-initialised bias tensor -- entirely plausible -- would then be reported
    as a perfect popularity predictor. Averaged ranks give a constant input zero
    variance instead, which :func:`spearman` turns into NaN.
    """
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, without pulling in SciPy for one function.

    Returns NaN when either input carries no ranking information, rather than a
    number that would read as a strong result.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 2:
        return float("nan")
    ra = _midranks(a)
    rb = _midranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denom) if denom > 0 else float("nan")


def _cosines(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    unit = rows / np.maximum(norms, 1e-12)
    return unit @ unit.T


def effective_rank(singular_values: np.ndarray) -> float:
    """Participation ratio of the spectrum: 1 for rank-1, n for a flat spectrum.

    A proxy for how concentrated an expert's weight energy is, and therefore a
    *candidate* signal for quantization tolerance. Reported, not acted on: the
    store format carries one dtype per layer, so per-expert mixed precision would
    need a format change before this could be used.
    """
    s2 = np.asarray(singular_values, dtype=np.float64) ** 2
    total = s2.sum()
    if total <= 0:
        return 0.0
    return float(total**2 / (s2**2).sum())


@dataclass
class LayerSignals:
    """Every static routing signal one layer carries."""

    layer: int
    num_experts: int
    hidden: int
    #: L2 norm of each expert's gate row.
    gate_norm: np.ndarray
    #: Same, after removing the mean row.
    gate_norm_centered: np.ndarray
    #: Cosine between gate rows, as stored.
    gate_cosine: np.ndarray
    #: Cosine after removing the mean row -- the discriminative part, since a
    #: component shared by every row adds the same score to all and is cancelled
    #: by top-k.
    gate_cosine_centered: np.ndarray
    #: Share of row energy sitting in the mean row. High means raw cosine is
    #: mostly measuring a direction that cannot discriminate.
    shared_direction_fraction: float
    #: The learned balancing bias, when the model has one.
    router_bias: np.ndarray | None = None
    #: Per-expert participation ratio, only when spectra were requested.
    effective_rank: np.ndarray | None = None
    #: Per-expert Frobenius norm over gate/up/down, when spectra were requested.
    weight_norm: np.ndarray | None = None

    @property
    def has_bias(self) -> bool:
        return self.router_bias is not None


def find_router_bias(index: CheckpointIndex, layer: int) -> np.ndarray | None:
    """The aux-loss-free balancing bias for this layer, if the model has one."""
    for suffix in BIAS_SUFFIXES:
        name = f"model.layers.{layer}.{suffix}"
        if name in index.weight_map:
            return _as_float32(index.read(name))
    return None


def inspect_layer(
    index: CheckpointIndex,
    layer: int,
    *,
    spectra: bool = False,
    spectra_rank: int | None = None,
) -> LayerSignals:
    """Collect one layer's static signals.

    ``spectra`` adds a full SVD per expert. That is the expensive part -- tens of
    seconds per layer on a real model -- so it is off by default and the gate
    geometry, which is instant, is always reported.
    """
    gate = _as_float32(index.read(index.router_name(layer))).astype(np.float64)
    if gate.ndim != 2:
        raise ValueError(f"router weight has shape {gate.shape}, expected 2-D")
    # Stored as [num_experts, hidden]: each *row* is an expert's direction. The
    # maths convention writes W_g as [hidden, num_experts] with experts in
    # columns, and conflating the two silently transposes the whole analysis.
    num_experts, hidden = gate.shape

    mean_row = gate.mean(axis=0)
    shared = float(
        np.linalg.norm(mean_row) ** 2
        / max((np.linalg.norm(gate, axis=1) ** 2).mean(), 1e-30)
    )

    ranks: np.ndarray | None = None
    norms: np.ndarray | None = None
    if spectra:
        ranks_list, norms_list = [], []
        for expert in index.expert_ids(layer):
            total_sq = 0.0
            spectrum: list[np.ndarray] = []
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = index.tensor_name(layer, expert, projection)
                w = _as_float32(index.read(name))
                total_sq += float(np.sum(w.astype(np.float64) ** 2))
                sv = np.linalg.svd(w, compute_uv=False)
                if spectra_rank:
                    sv = sv[:spectra_rank]
                spectrum.append(sv)
            ranks_list.append(effective_rank(np.concatenate(spectrum)))
            norms_list.append(float(np.sqrt(total_sq)))
        ranks = np.asarray(ranks_list)
        norms = np.asarray(norms_list)

    return LayerSignals(
        layer=layer,
        num_experts=num_experts,
        hidden=hidden,
        gate_norm=np.linalg.norm(gate, axis=1),
        gate_norm_centered=np.linalg.norm(gate - mean_row, axis=1),
        gate_cosine=_cosines(gate),
        gate_cosine_centered=_cosines(gate - mean_row),
        shared_direction_fraction=shared,
        router_bias=find_router_bias(index, layer),
        effective_rank=ranks,
        weight_norm=norms,
    )


def score_against_profile(
    signals: LayerSignals,
    load_share: np.ndarray,
    cooccurrence: np.ndarray | None = None,
) -> dict[str, float]:
    """Rank-correlate each static signal with measured behaviour.

    ``load_share`` is this layer's per-expert share; ``cooccurrence`` its
    ``[E, E]`` within-token pair counts. The returned correlations are the whole
    justification for using -- or not using -- a signal on this model.
    """
    if load_share.shape[0] != signals.num_experts:
        raise ValueError(
            f"profile has {load_share.shape[0]} experts, checkpoint layer "
            f"{signals.layer} has {signals.num_experts}"
        )

    out: dict[str, float] = {
        "gate_norm_vs_load": spearman(signals.gate_norm, load_share),
        "gate_norm_centered_vs_load": spearman(
            signals.gate_norm_centered, load_share
        ),
    }

    if signals.router_bias is not None:
        # Negative bias means the balancer pushed a popular expert down, so the
        # expected relationship is *inverse*.
        out["neg_bias_vs_load"] = spearman(-signals.router_bias, load_share)

    if cooccurrence is not None:
        off = ~np.eye(signals.num_experts, dtype=bool)
        out["gate_cosine_vs_cooccurrence"] = spearman(
            signals.gate_cosine[off], cooccurrence[off]
        )
        out["gate_cosine_centered_vs_cooccurrence"] = spearman(
            signals.gate_cosine_centered[off], cooccurrence[off]
        )

    if signals.effective_rank is not None:
        out["effective_rank_vs_load"] = spearman(signals.effective_rank, load_share)
    if signals.weight_norm is not None:
        out["weight_norm_vs_load"] = spearman(signals.weight_norm, load_share)
    return out


#: Below this absolute rank correlation a signal is not worth acting on, even for
#: placement. Chosen so the measured OLMoE gate signals (|rho| <= 0.05) fall well
#: under it and a genuinely useful signal would have to clear a visible bar.
USABLE_RHO = 0.2


def verdict(scores: dict[str, float]) -> dict[str, str]:
    """Turn correlations into a usable/unusable call, with the threshold stated."""
    out = {}
    for name, rho in scores.items():
        if not np.isfinite(rho):
            out[name] = "not measurable"
        elif abs(rho) >= USABLE_RHO:
            out[name] = f"usable for placement (|rho|={abs(rho):.2f})"
        else:
            out[name] = f"no usable signal (|rho|={abs(rho):.2f})"
    return out


def report(
    per_layer: list[tuple[LayerSignals, dict[str, float]]],
    *,
    model: str | None = None,
) -> str:
    """A human-readable summary. Numbers first, verdict last."""
    lines = [f"model: {model or '(unnamed)'}"]
    if not per_layer:
        return "\n".join(lines + ["no layers inspected"])

    first = per_layer[0][0]
    lines += [
        f"experts/layer: {first.num_experts}   hidden: {first.hidden}",
        f"router bias present: {'yes' if first.has_bias else 'no'}"
        + ("" if first.has_bias else "  (aux-loss balanced, no learned prior)"),
        "",
        "gate geometry per layer:",
        f"  {'layer':>5}  {'norm spread':>11}  {'|cos| med':>9}  {'|cos| max':>9}"
        f"  {'shared dir':>10}",
    ]
    for signals, _ in per_layer:
        off = ~np.eye(signals.num_experts, dtype=bool)
        absc = np.abs(signals.gate_cosine[off])
        spread = signals.gate_norm.max() / max(signals.gate_norm.min(), 1e-30)
        lines.append(
            f"  {signals.layer:>5}  {spread:>10.2f}x  {np.median(absc):>9.3f}"
            f"  {absc.max():>9.3f}  {signals.shared_direction_fraction:>10.3f}"
        )

    scored = [s for _, s in per_layer if s]
    if scored:
        keys = sorted({k for s in scored for k in s})
        lines += ["", "validated against the measured profile (mean rank correlation):"]
        means = {}
        for key in keys:
            values = [s[key] for s in scored if key in s and np.isfinite(s[key])]
            if values:
                means[key] = float(np.mean(values))
        for key, rho in means.items():
            lines.append(f"  {key:<42} {rho:+.3f}")
        lines += ["", "verdict:"]
        for key, call in verdict(means).items():
            lines.append(f"  {key:<42} {call}")
        lines += [
            "",
            "Reminder: a usable static signal belongs in placement and prefetch,",
            "where a wrong guess costs a cache miss. Not in pruning or merging,",
            "where a wrong guess costs quality and the artifact cannot recover it.",
        ]
    else:
        lines += [
            "",
            "no profile supplied, so nothing was validated -- these are candidate",
            "signals only. Pass a profile to find out whether they predict",
            "anything on this model.",
        ]
    return "\n".join(lines)


def inspect_checkpoint(
    checkpoint: str,
    layers: list[int],
    *,
    profile: Any = None,
    spectra: bool = False,
) -> tuple[list[tuple[LayerSignals, dict[str, float]]], str | None]:
    """Inspect the named layers, scoring against ``profile`` when given."""
    index = CheckpointIndex.open(checkpoint)
    share = profile.layer_share() if profile is not None else None

    results = []
    for layer in layers:
        signals = inspect_layer(index, layer, spectra=spectra)
        scores: dict[str, float] = {}
        if profile is not None and layer in profile.moe_layers:
            slot = profile.moe_layers.index(layer)
            cooc = profile.cooc[slot] if profile.cooc is not None else None
            scores = score_against_profile(signals, share[slot], cooc)
        results.append((signals, scores))
    return results, getattr(profile, "model", None)
