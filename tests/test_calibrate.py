# SPDX-License-Identifier: Apache-2.0
"""Amplitude correction: the folding, and the search that finds the value.

The folding is arithmetic and runs anywhere. The search needs an engine, so only its
bookkeeping is tested here -- the curve, the edge warnings, and the refusal to credit a
change smaller than the noise it was measured against.
"""

from __future__ import annotations

import numpy as np
import pytest

from vllm_moe_surgeon.compat.calibrate import (
    Calibration,
    CalibrationPoint,
    bracket_warnings,
    calibrate_amplitude,
)
from vllm_moe_surgeon.surgery.apply import fold_amplitude


def _curve(pairs, model="m", tokens=2863):
    points = [CalibrationPoint(s, p) for s, p in pairs]
    best = min(points, key=lambda q: q.perplexity)
    unscaled = next((q.perplexity for q in points if q.scale == 1.0), best.perplexity)
    return Calibration(
        model=model,
        best_scale=best.scale,
        best_perplexity=best.perplexity,
        unscaled_perplexity=unscaled,
        tokens=tokens,
        points=points,
        warnings=bracket_warnings(points),
    )


# ------------------------------------------------------------------- the folding


def test_folding_scales_down_proj_and_nothing_else():
    down = np.arange(12, dtype=np.float32).reshape(3, 4)
    scaled = fold_amplitude(down, 0.85)
    np.testing.assert_allclose(scaled, down * 0.85, rtol=0, atol=0)
    # Not in place: apply_plan still needs the original for anything downstream.
    np.testing.assert_allclose(down, np.arange(12).reshape(3, 4), rtol=0, atol=0)


def test_a_scale_of_one_is_the_identity():
    down = np.random.default_rng(0).normal(size=(4, 5)).astype(np.float32)
    np.testing.assert_allclose(fold_amplitude(down, 1.0), down, rtol=0, atol=0)


def test_a_nonpositive_scale_is_refused():
    down = np.ones((2, 2), dtype=np.float32)
    for bad in (0.0, -0.5):
        with pytest.raises(ValueError, match="must be positive"):
            fold_amplitude(down, bad)


def test_amplitude_reaches_the_written_checkpoint_and_only_down_proj(tmp_path):
    """The constraint the maths imposes: gate and up must not be scaled.

    The layer's output is linear in down_proj, so scaling it is exactly scaling the
    gate. SwiGLU is *nonlinear* in gate_proj and up_proj, so scaling those changes the
    function rather than its amplitude -- and nothing downstream would report it.
    """
    import json

    import torch
    from safetensors.torch import save_file

    from vllm_moe_surgeon.surgery import Plan
    from vllm_moe_surgeon.surgery.apply import apply_plan
    from vllm_moe_surgeon.surgery.descriptors import CheckpointIndex
    from vllm_moe_surgeon.surgery.plan import ExpertPlacement

    inter, hidden, experts = 6, 8, 3
    generator = torch.Generator().manual_seed(5)
    tensors = {
        "model.layers.0.mlp.gate.weight": torch.randn(
            experts, hidden, generator=generator
        )
    }
    original = {}
    for expert in range(experts):
        base = f"model.layers.0.mlp.experts.{expert}"
        gate = torch.randn(inter, hidden, generator=generator)
        up = torch.randn(inter, hidden, generator=generator)
        down = torch.randn(hidden, inter, generator=generator)
        tensors[f"{base}.gate_proj.weight"] = gate
        tensors[f"{base}.up_proj.weight"] = up
        tensors[f"{base}.down_proj.weight"] = down
        original[expert] = (gate, up, down)

    src = tmp_path / "src"
    src.mkdir()
    save_file(tensors, str(src / "model.safetensors"), metadata={"format": "pt"})
    with open(src / "config.json", "w") as f:
        json.dump(
            {
                "num_hidden_layers": 1,
                "num_experts": experts,
                "num_experts_per_tok": 2,
            },
            f,
        )

    plan = Plan(
        model="m",
        revision=None,
        budget={},
        placements=[
            ExpertPlacement(0, 0, "merge_into_core", tokens=10, share=0.6),
            ExpertPlacement(0, 1, "merge_into_core", tokens=10, share=0.4),
            ExpertPlacement(0, 2, "drop", tokens=1, share=0.0),
        ],
        gate={"passed": True, "reason": "test"},
    )
    manifest = apply_plan(
        plan, str(src), str(tmp_path / "out"), amplitude=0.85, copy_extra_files=False
    )

    assert manifest["amplitude"] == 0.85
    assert "do not apply it again" in manifest["amplitude_note"]

    out = CheckpointIndex.open(str(tmp_path / "out"))
    for new_id, old_id in enumerate([0, 1]):
        gate, up, down = original[old_id]
        np.testing.assert_allclose(
            out.read_expert(0, new_id, "down_proj"),
            down.numpy() * 0.85,
            rtol=1e-6,
            atol=0,
        )
        # The two that must be left alone.
        np.testing.assert_allclose(
            out.read_expert(0, new_id, "gate_proj"), gate.numpy(), rtol=1e-6, atol=0
        )
        np.testing.assert_allclose(
            out.read_expert(0, new_id, "up_proj"), up.numpy(), rtol=1e-6, atol=0
        )


def test_no_amplitude_leaves_the_manifest_saying_so(tmp_path):
    # Reuse the fixture above by calling it for its side effects would be obscure;
    # this only needs the manifest fields, so a minimal source is enough.
    import json

    import torch
    from safetensors.torch import save_file

    from vllm_moe_surgeon.surgery import Plan
    from vllm_moe_surgeon.surgery.apply import apply_plan
    from vllm_moe_surgeon.surgery.plan import ExpertPlacement

    tensors = {"model.layers.0.mlp.gate.weight": torch.zeros(2, 4)}
    for expert in range(2):
        base = f"model.layers.0.mlp.experts.{expert}"
        tensors[f"{base}.gate_proj.weight"] = torch.zeros(3, 4)
        tensors[f"{base}.up_proj.weight"] = torch.zeros(3, 4)
        tensors[f"{base}.down_proj.weight"] = torch.zeros(4, 3)
    src = tmp_path / "src"
    src.mkdir()
    save_file(tensors, str(src / "model.safetensors"), metadata={"format": "pt"})
    with open(src / "config.json", "w") as f:
        json.dump(
            {"num_hidden_layers": 1, "num_experts": 2, "num_experts_per_tok": 1}, f
        )

    plan = Plan(
        model="m",
        revision=None,
        budget={},
        placements=[
            ExpertPlacement(0, 0, "merge_into_core", tokens=10, share=1.0),
            ExpertPlacement(0, 1, "keep_on_disk", tokens=1, share=0.0),
        ],
    )
    manifest = apply_plan(
        plan, str(src), str(tmp_path / "out"), copy_extra_files=False
    )
    assert manifest["amplitude"] is None
    assert "surgeon calibrate" in manifest["amplitude_note"]


# -------------------------------------------------------------------- the search


def test_the_winner_is_reported_against_no_change():
    """Measured shape on OLMoE pruned to 40 experts."""
    curve = _curve([(0.80, 10.6736), (0.85, 10.5306), (0.90, 10.74), (1.00, 11.8754)])
    assert curve.best_scale == 0.85
    assert curve.worth_applying
    assert curve.improvement == pytest.approx(10.5306 / 11.8754, rel=1e-6)
    text = curve.report()
    assert "0.850" in text and "<-- best" in text
    assert "fresh load" in text


def test_a_change_smaller_than_the_noise_is_not_credited():
    """In-place scaling alone drifts ~0.2%, so sub-1% wins are not wins."""
    curve = _curve([(0.95, 11.87), (1.00, 11.90)])
    assert curve.best_scale == 0.95
    assert not curve.worth_applying
    assert "no scale beats leaving it alone" in curve.report()


def test_no_change_winning_is_a_clean_answer():
    curve = _curve([(0.85, 12.9), (1.00, 11.9)])
    assert curve.best_scale == 1.0
    assert not curve.worth_applying
    assert curve.improvement == 1.0


def test_an_edge_optimum_warns_that_the_bracket_is_too_narrow():
    """A minimum at the edge means the real one may lie outside what was measured."""
    curve = _curve([(0.75, 10.0), (0.85, 10.5), (1.00, 11.9)])
    assert curve.best_scale == 0.75
    assert any("edge of the bracket" in w for w in curve.warnings)
    assert "WARNING" in curve.report()

    # An interior optimum is bracketed on both sides, so it needs no warning.
    interior = _curve([(0.75, 10.9), (0.85, 10.5), (1.00, 11.9)])
    assert interior.best_scale == 0.85
    assert not any("edge of the bracket" in w for w in interior.warnings)


def test_a_curve_without_one_point_of_reference_says_so():
    curve = _curve([(0.80, 10.6), (0.85, 10.5)])
    assert any("1.0 was not measured" in w for w in curve.warnings)


def test_an_empty_curve_is_not_silently_fine():
    assert bracket_warnings([]) == ["nothing was measured"]


def test_an_empty_or_negative_bracket_is_refused():
    with pytest.raises(ValueError, match="no scales"):
        calibrate_amplitude("m", ["p"], scales=())
    with pytest.raises(ValueError, match="must be positive"):
        calibrate_amplitude("m", ["p"], scales=(0.5, -1.0))
