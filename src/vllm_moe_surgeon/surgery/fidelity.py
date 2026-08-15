# SPDX-License-Identifier: Apache-2.0
"""How far did the output distribution move? -- the instrument this project lacked.

Two measurements already on record say why nothing else here will do.

**Perplexity does not certify task accuracy.** A rank-1 pruned OLMoE kept
perplexity at 1.25x and lost 25% of arc_challenge relative (p=4.8e-08), and the
amplitude fix that removed ~60% of the perplexity damage recovered *no* task
accuracy (p = 1.0 / 0.83 / 0.53). So a 4-decimal perplexity is not evidence.

**Task accuracy does not certify token fidelity either, and at our sample sizes
it cannot.** Measured 2026-08-15 on the laptop: llama.cpp's Q4_K_M against its
own bf16, same engine, same corpus -- **top-1 agreement 92.01 +/- 0.43%**, i.e.
the sampled token changes one time in twelve. gsm8k@200 saw 0.095 vs 0.105
(+/- 0.021) and HellaSwag@400 saw 62.25% vs 61.75% with overlapping intervals.
Both tasks were blind to it.

The axis between the two is the token distribution itself, and this module is
its instrument: **top-1 agreement** (the same statistic llama.cpp's KL tool
reports, so external numbers are comparable in kind), the **delta-p** table on
the reference's own top token, and a **truncated KL divergence**.

Deliberately vLLM-free: the arms are captured by ``compat/fidelity.py`` in one
child process each, and everything below is numpy over two ``.npz`` files. That
keeps the maths testable on a machine with no GPU, and it means a capture taken
on GB10 can be compared on the Mac.

**What "truncated" costs, and what raising K does not fix.** A capture holds the
top ``K`` tokens per position, not the vocabulary, so KL is computed between the
two distributions *restricted and renormalised to the reference's top-K support*.
When a reference token is missing from the test arm's top-K all we know is that
its probability is below that arm's K-th; substituting the K-th is the most
favourable value for the test arm, so the figure is a **lower bound on that
restricted KL** -- and, because the restriction renormalises, not a bound on the
full-vocabulary KL at all.

The first instinct was to refuse the figure and tell the caller to raise K. That
advice was measured and it is wrong for this model: on OLMoE / gsm8k held-out,
**K 32 -> 128 (four times the tokens) moved coverage 89.29% -> 94.50% and the
substitution rate 3.22% -> 2.97%.** The tail is structural, so refusing means
never reporting KL at all. It is therefore reported, labelled ``>=``, next to
the coverage and substitution numbers that say how much slack it has.

**The headline is top-1 agreement, and that one is exact** -- an argmax is in
its own top-K by construction, so no truncation assumption touches it. It is
also the statistic llama.cpp's KL tool reports, which is what makes an external
number ("Q4_K_M: 92.01%") comparable in kind.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Positions are compared in blocks of this many, so the [N, K, K] match tensor
#: the id lookup needs stays bounded no matter how long the corpus is.
_BLOCK = 4096

#: Above this fraction of substituted reference tokens the KL figure stops being
#: an estimate and is printed as a lower bound. Not a refusal: raising K was
#: measured not to fix it (module docstring), so refusing would mean never
#: reporting KL for a model with a heavy tail.
MAX_SUBSTITUTION = 0.02


@dataclass
class Capture:
    """One arm's top-K next-token distributions over a fixed corpus.

    ``ids`` and ``logprobs`` are ``[positions, K]``, each row sorted by
    descending logprob, so column 0 is the arm's argmax at that position.
    ``doc`` records which prompt each position came from, which is what lets a
    comparison prove the two arms scored the *same* text in the same order.
    """

    ids: np.ndarray
    logprobs: np.ndarray
    doc: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def positions(self) -> int:
        return int(self.ids.shape[0])

    @property
    def top_k(self) -> int:
        return int(self.ids.shape[1])

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            ids=self.ids.astype(np.int32),
            logprobs=self.logprobs.astype(np.float32),
            doc=self.doc.astype(np.int32),
            meta=np.frombuffer(
                json.dumps(self.meta).encode("utf-8"), dtype=np.uint8
            ),
        )

    @staticmethod
    def load(path: str) -> Capture:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(bytes(data["meta"]).decode("utf-8"))
            return Capture(
                ids=data["ids"],
                logprobs=data["logprobs"],
                doc=data["doc"],
                meta=meta,
            )


@dataclass
class Fidelity:
    """One arm measured against the reference capture."""

    arm: str
    positions: int
    top_k: int
    #: Fraction of positions where both arms' argmax token is the same. Exact --
    #: it needs no support assumption, which is why it is the headline.
    top1_agreement: float
    #: Percentage points, on the *reference's* argmax token: 100 * (p_test - p_ref).
    dp_mean: float
    dp_rms: float
    dp_percentiles: dict[str, float]
    #: KL(reference || test), nats, over the reference's renormalised top-K.
    kld_mean: float
    kld_percentiles: dict[str, float]
    #: Mean probability mass the reference's own top-K covers. Below ~0.99 the
    #: truncation itself, not the arm, dominates the KL figure.
    coverage: float
    #: Fraction of reference top-K entries absent from the test arm's top-K.
    substitution_rate: float

    @property
    def kld_is_exact(self) -> bool:
        """No reference token needed substituting, so the restricted KL is exact."""
        return self.substitution_rate <= MAX_SUBSTITUTION

    @property
    def identical(self) -> bool:
        """Nothing moved: the arms agree everywhere and KL is exactly zero."""
        return self.top1_agreement >= 1.0 and self.kld_mean == 0.0


_PERCENTILES = (50.0, 75.0, 90.0, 95.0, 99.0, 99.9)


def compare(reference: Capture, test: Capture, *, arm: str = "test") -> Fidelity:
    """Measure *test* against *reference*. Both must cover the same positions.

    Refuses mismatched captures rather than silently comparing position *i* of
    one corpus against position *i* of another -- the failure that would produce
    a large, entirely fictitious divergence.
    """
    if reference.positions != test.positions:
        raise ValueError(
            f"captures cover different corpora: reference has "
            f"{reference.positions} positions, {arm} has {test.positions}. "
            "Both arms must be captured from the same staged corpus."
        )
    if reference.positions == 0:
        raise ValueError("the reference capture is empty; nothing to compare")
    if not np.array_equal(reference.doc, test.doc):
        raise ValueError(
            f"captures disagree about which prompt each position belongs to; "
            f"{arm} did not score the same text in the same order"
        )

    ref_ids, ref_lp = reference.ids, reference.logprobs.astype(np.float64)
    test_ids, test_lp = test.ids, test.logprobs.astype(np.float64)

    top1 = float(np.mean(ref_ids[:, 0] == test_ids[:, 0]))

    ref_p = np.exp(ref_lp)
    coverage = float(np.mean(ref_p.sum(axis=1)))

    kld = np.empty(reference.positions, dtype=np.float64)
    dp = np.empty(reference.positions, dtype=np.float64)
    substituted = 0
    total = 0

    for start in range(0, reference.positions, _BLOCK):
        stop = min(start + _BLOCK, reference.positions)
        r_ids = ref_ids[start:stop]
        r_lp = ref_lp[start:stop]
        t_ids = test_ids[start:stop]
        t_lp = test_lp[start:stop]

        # For every reference token, that token's logprob under the test arm.
        # Absent -> the test arm's K-th, which is the largest value it could
        # have had; see the module docstring on why that biases KL downwards.
        match = r_ids[:, :, None] == t_ids[:, None, :]
        found = match.any(axis=2)
        index = match.argmax(axis=2)
        gathered = np.take_along_axis(t_lp, index, axis=1)
        floor = t_lp[:, -1][:, None]
        q_lp = np.where(found, gathered, floor)

        substituted += int((~found).sum())
        total += found.size

        # Renormalise both restrictions to the reference's support so the KL is
        # between two distributions rather than two truncations.
        p = np.exp(r_lp)
        p_sum = p.sum(axis=1, keepdims=True)
        p_hat = p / p_sum
        q = np.exp(q_lp)
        q_sum = q.sum(axis=1, keepdims=True)
        q_hat = q / np.where(q_sum > 0, q_sum, 1.0)

        log_p = np.log(np.maximum(p_hat, 1e-30))
        log_q = np.log(np.maximum(q_hat, 1e-30))
        kld[start:stop] = np.sum(p_hat * (log_p - log_q), axis=1)

        # delta-p is quoted on the reference's own top token: the quantity a
        # sampler at temperature 0 actually acts on.
        dp[start:stop] = 100.0 * (np.exp(q_lp[:, 0]) - np.exp(r_lp[:, 0]))

    kld = np.maximum(kld, 0.0)  # renormalisation can leave -1e-17
    return Fidelity(
        arm=arm,
        positions=reference.positions,
        top_k=reference.top_k,
        top1_agreement=top1,
        dp_mean=float(dp.mean()),
        dp_rms=float(np.sqrt(np.mean(dp**2))),
        dp_percentiles={
            "min": float(dp.min()),
            "p05": float(np.percentile(dp, 5)),
            "p50": float(np.percentile(dp, 50)),
            "p95": float(np.percentile(dp, 95)),
            "max": float(dp.max()),
        },
        kld_mean=float(kld.mean()),
        kld_percentiles={
            f"p{p:g}": float(np.percentile(kld, p)) for p in _PERCENTILES
        },
        coverage=coverage,
        substitution_rate=substituted / total if total else 0.0,
    )


def report(reference_arm: str, results: list[Fidelity]) -> str:
    """The table, and the one sentence that keeps it from being over-read."""
    lines = [
        f"reference: {reference_arm}",
        f"positions: {results[0].positions if results else 0} "
        f"(top-{results[0].top_k if results else 0} per position)",
        "",
        f"  {'arm':<28}{'top-1 agree':>13}{'KL(ref||arm)':>14}{'RMS dp':>10}",
    ]
    for r in results:
        kld = f"{r.kld_mean:.5f}" if r.kld_is_exact else f">={r.kld_mean:.5f}"
        lines.append(
            f"  {r.arm[:28]:<28}{100 * r.top1_agreement:>12.3f}%"
            f"{kld:>14}{r.dp_rms:>9.3f}%"
        )
    lines.append("")

    for r in results:
        if r.identical:
            # Deliberately NOT "bit-identical": a capture holds K tokens per
            # position, so what this establishes is that nothing moved *within
            # the capture*. The token-hash control is the stronger test and this
            # command does not replace it.
            lines.append(
                f"  {r.arm}: no captured difference -- every argmax agrees and "
                f"KL over the top-{r.top_k} is exactly zero. For a bit-exactness "
                f"claim, run the token-hash control."
            )
        else:
            lines.append(
                f"  {r.arm}: dp median {r.dp_percentiles['p50']:+.3f}%, "
                f"p95 {r.dp_percentiles['p95']:+.3f}%, "
                f"max {r.dp_percentiles['max']:+.3f}%"
                f"; KL p95 {r.kld_percentiles['p95']:.5f}, "
                f"p99.9 {r.kld_percentiles['p99.9']:.5f}"
            )
            if not r.kld_is_exact:
                lines.append(
                    f"      KL is a LOWER BOUND: {100 * r.substitution_rate:.2f}% "
                    f"of reference tokens fell outside this arm's top-{r.top_k}, "
                    f"and were scored at the most favourable value they could "
                    f"have had. Raising K may not fix it -- on OLMoE/gsm8k, 32 to "
                    f"128 moved this by 0.25 points."
                )
        # Runs for every arm, the agreeing ones included: a capture too thin to
        # hold the mass cannot establish agreement either, and a caveat that
        # skips exactly the reassuring rows is worse than no caveat.
        if r.coverage < 0.99:
            lines.append(
                f"      top-{r.top_k} covers only {100 * r.coverage:.2f}% of the "
                f"reference's mass; the truncation is competing with the effect."
            )

    lines.append("")
    lines.append(
        "  top-1 agreement is what a temperature-0 sampler acts on; KL is the "
        "distribution behind it. Neither is task accuracy -- measured here, a "
        "change moving the top token 8% of the time was invisible to both "
        "gsm8k@200 and HellaSwag@400 (docs/benchmarks.md)."
    )
    return "\n".join(lines)


def to_dict(result: Fidelity) -> dict[str, Any]:
    row = dict(vars(result))
    return {
        k: (None if isinstance(v, float) and not math.isfinite(v) else v)
        for k, v in row.items()
    }
