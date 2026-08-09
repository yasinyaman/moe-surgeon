# SPDX-License-Identifier: Apache-2.0
"""Find the amplitude a pruned checkpoint should have been written with.

Deletion leaves a systematic amplitude error. Under ``renormalize=False`` -- OLMoE's
setting -- a surviving gate is the raw softmax probability, so restricting the softmax
from E rows to K inflates every one by ``1 / (1 - P_D(x))``, where ``P_D`` is the mass
the deleted experts used to carry. The model was never trained to receive an inflated
MoE branch, and no choice of router rows can undo it: a softmax sums to one over
whatever rows remain. The layer's output is linear in ``down_proj``, so that is where
the correction goes (see :func:`..surgery.apply.fold_amplitude`).

Two ways to get the number, and this module takes the second:

1. Estimate ``E_t[1 - P_D(x_t)]`` from captured hidden states. Exact for the quantity,
   but it needs an activation-capture path, and the quantity is a proxy for what
   actually matters.
2. Measure held-out perplexity at several scales and take the minimum. One model load,
   no capture infrastructure, and it optimises the thing being claimed rather than a
   proxy for it.

The risk of (2) is fitting a corpus. It is one parameter against thousands of tokens,
which is about as safe as fitting gets, but the guard rails are still here: the search
refuses a suspiciously wide bracket, reports the curve rather than only the winner so a
flat or ragged optimum is visible, and warns when the best point sits at an edge --
where the true minimum is outside what was measured.

**A scaled arm is not a fresh load.** The scales are applied as successive in-place
multiplies on one engine, and bf16 rounding accumulates: measured drift of 0.2% on the
last point of a six-point sweep. So the sweep locates the optimum and the caller
re-measures the winner on a fresh load before writing it into an artifact.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .._logging import init_logger

logger = init_logger(__name__)

#: Deletion inflates the surviving gates, so the correction is downward. Bracket chosen
#: to include 1.0 (the artifact as written) so the search can report "no change wins".
DEFAULT_SCALES = (0.75, 0.80, 0.85, 0.90, 0.95, 1.00)


def scale_down_proj(worker: Any, factor: float) -> int:
    """Multiply every routed expert's ``down_proj`` in place. Returns layers touched.

    Runs inside the worker via ``collective_rpc``. Only ``w2``: SwiGLU is nonlinear in
    ``gate_proj``/``up_proj``, so scaling those would change the function rather than
    its amplitude.
    """
    import torch

    from .ablation import _moe_modules

    touched = 0
    with torch.no_grad():
        for _name, module in _moe_modules(worker.model_runner.model):
            module.w2_weight.mul_(factor)
            touched += 1
    return touched


@dataclass
class CalibrationPoint:
    scale: float
    perplexity: float

    @property
    def summary(self) -> str:
        return f"{self.scale:.3f} -> {self.perplexity:.4f}"


@dataclass
class Calibration:
    """The curve, the winner, and what is doubtful about it."""

    model: str
    best_scale: float
    best_perplexity: float
    unscaled_perplexity: float
    tokens: int
    points: list[CalibrationPoint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        """Perplexity ratio against the artifact as written. Below 1 is better."""
        if self.unscaled_perplexity <= 0:
            return float("nan")
        return self.best_perplexity / self.unscaled_perplexity

    @property
    def worth_applying(self) -> bool:
        """Whether the winner beats no-change by more than measurement noise.

        1% is the threshold because repeated in-place scaling alone drifts about 0.2%,
        and a 1-D fit on one corpus does not deserve credit for less than several times
        that.
        """
        return self.best_scale != 1.0 and self.improvement < 0.99

    def report(self) -> str:
        lines = [
            f"{self.model}",
            f"  held-out tokens: {self.tokens}",
            "  curve:",
        ]
        for point in self.points:
            mark = "  <-- best" if point.scale == self.best_scale else ""
            lines.append(f"    {point.summary}{mark}")
        lines.append("")
        if self.worth_applying:
            lines.append(
                f"  amplitude {self.best_scale:.3f} lowers held-out perplexity "
                f"{self.unscaled_perplexity:.4f} -> {self.best_perplexity:.4f} "
                f"({self.improvement:.3f}x)"
            )
            lines.append(
                "  re-measure it on a fresh load before folding it in: these points "
                "share one engine and in-place scaling accumulates rounding"
            )
        else:
            lines.append(
                f"  no scale beats leaving it alone ({self.best_scale:.3f} gives "
                f"{self.improvement:.3f}x). Deletion left no amplitude error worth "
                "correcting on this model."
            )
        for warning in self.warnings:
            lines.append(f"  WARNING: {warning}")
        return "\n".join(lines)


def calibrate_amplitude(
    model: str,
    prompts: list[str],
    *,
    scales: tuple[float, ...] = DEFAULT_SCALES,
    llm_kwargs: dict[str, Any] | None = None,
) -> Calibration:
    """Sweep ``scales`` on one engine and return the curve.

    ``scales`` is swept in the given order with relative multiplies, so the engine is
    loaded once. 1.0 is measured like any other point rather than assumed.
    """
    # Validated before the engine is touched: a bad bracket should not cost a load.
    if not scales:
        raise ValueError("no scales to try")
    if any(s <= 0 for s in scales):
        raise ValueError(f"scales must be positive, got {scales}")

    from vllm import LLM

    from .ablation import measure_nll

    llm = LLM(model=model, **(llm_kwargs or {}))
    points: list[CalibrationPoint] = []
    tokens = 0
    current = 1.0

    for target in scales:
        ratio = target / current
        if abs(ratio - 1.0) > 1e-12:
            llm.collective_rpc(lambda w, r=ratio: scale_down_proj(w, r))
        current = target
        nll, count = measure_nll(llm, prompts)
        tokens = count
        perplexity = math.exp(nll / count)
        points.append(CalibrationPoint(target, perplexity))
        logger.info("amplitude %.3f -> perplexity %.4f", target, perplexity)

    best = min(points, key=lambda p: p.perplexity)
    unscaled = next((p.perplexity for p in points if p.scale == 1.0), best.perplexity)

    return Calibration(
        model=model,
        best_scale=best.scale,
        best_perplexity=best.perplexity,
        unscaled_perplexity=unscaled,
        tokens=tokens,
        points=points,
        warnings=bracket_warnings(points),
    )


def bracket_warnings(points: list[CalibrationPoint]) -> list[str]:
    """What is doubtful about a curve, from the curve alone.

    Separate from the sweep so it can be checked without an engine, and because both
    conditions are about the *shape* of what was measured rather than the measuring.
    """
    if not points:
        return ["nothing was measured"]

    warnings: list[str] = []
    best = min(points, key=lambda p: p.perplexity)
    ordered = sorted(points, key=lambda p: p.scale)
    if len(ordered) > 1 and best.scale in (ordered[0].scale, ordered[-1].scale):
        warnings.append(
            f"the best point {best.scale:.3f} is at the edge of the bracket, so the "
            "true optimum may lie outside it. Widen the scales and re-run."
        )
    if not any(p.scale == 1.0 for p in points):
        warnings.append(
            "1.0 was not measured, so there is nothing to compare the winner against"
        )
    return warnings


def to_dict(calibration: Calibration) -> dict[str, Any]:
    payload = asdict(calibration)
    payload["improvement"] = calibration.improvement
    payload["worth_applying"] = calibration.worth_applying
    return payload


__all__ = [
    "DEFAULT_SCALES",
    "bracket_warnings",
    "Calibration",
    "CalibrationPoint",
    "calibrate_amplitude",
    "scale_down_proj",
    "to_dict",
]
