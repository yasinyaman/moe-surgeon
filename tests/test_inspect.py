# SPDX-License-Identifier: Apache-2.0
"""Static signal inspection, and the validation that keeps it honest.

The tool's value is not that it computes gate geometry -- that is a few lines --
but that it *scores* each signal against measured behaviour instead of assuming.
So the tests check the scoring machinery on constructed cases where the right
answer is known: a signal built to correlate must score high, and one built to be
noise must score near zero.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from vllm_moe_surgeon.surgery.descriptors import CheckpointIndex
from vllm_moe_surgeon.surgery.inspect import (
    USABLE_RHO,
    effective_rank,
    find_router_bias,
    inspect_layer,
    report,
    score_against_profile,
    spearman,
    verdict,
)
from vllm_moe_surgeon.telemetry.stats import ExpertStats

H = 16
INTER = 8
E = 6


def _write(root, *, bias=None, gate=None, num_experts=E):
    tensors = {"model.norm.weight": torch.ones(H)}
    rng = torch.Generator().manual_seed(3)
    for expert in range(num_experts):
        base = f"model.layers.0.mlp.experts.{expert}"
        tensors[f"{base}.gate_proj.weight"] = torch.randn(INTER, H, generator=rng)
        tensors[f"{base}.up_proj.weight"] = torch.randn(INTER, H, generator=rng)
        tensors[f"{base}.down_proj.weight"] = torch.randn(H, INTER, generator=rng)
    tensors["model.layers.0.mlp.gate.weight"] = (
        gate if gate is not None else torch.randn(num_experts, H, generator=rng)
    )
    if bias is not None:
        tensors["model.layers.0.mlp.gate.e_score_correction_bias"] = bias
    root.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    with open(root / "config.json", "w") as f:
        json.dump({"num_hidden_layers": 1, "num_experts": num_experts}, f)


def _profile(share_by_expert, cooc=None):
    stats = ExpertStats.empty(E, [0], with_cooc=cooc is not None)
    stats.tokens[0] = (np.asarray(share_by_expert) * 10_000).astype(np.int64)
    stats.layer_token_rows[0] = 10_000
    if cooc is not None:
        stats.cooc[0] = cooc
    stats.finalize()
    return stats


# ------------------------------------------------------------------ primitives


def test_spearman_endpoints():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1], [1]))
    assert np.isnan(spearman([1, 2], [1, 2, 3]))


def test_spearman_on_a_constant_is_not_a_number():
    """A flat signal has no ranking, so a correlation would be meaningless."""
    assert np.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4]))


def test_effective_rank_spans_one_to_n():
    assert effective_rank([7, 0, 0, 0]) == pytest.approx(1.0)
    assert effective_rank([1, 1, 1, 1]) == pytest.approx(4.0)
    assert effective_rank([0, 0]) == 0.0
    # A decaying spectrum lands in between.
    assert 1.0 < effective_rank([4, 2, 1, 0.5]) < 4.0


# ------------------------------------------------------------- signal presence


def test_bias_is_detected_when_present(tmp_path):
    bias = torch.tensor([-0.5, -0.1, 0.0, 0.1, 0.2, 0.3])
    _write(tmp_path, bias=bias)
    index = CheckpointIndex.open(str(tmp_path))
    found = find_router_bias(index, 0)
    assert found is not None
    np.testing.assert_allclose(found, bias.numpy(), rtol=0, atol=0)
    assert inspect_layer(index, 0).has_bias


def test_bias_absence_is_reported_not_faked(tmp_path):
    """OLMoE and Qwen3-MoE have no such tensor; that must read as absent."""
    _write(tmp_path)
    index = CheckpointIndex.open(str(tmp_path))
    assert find_router_bias(index, 0) is None
    assert inspect_layer(index, 0).has_bias is False


def test_gate_rows_are_read_as_experts_not_transposed(tmp_path):
    """[num_experts, hidden] -- transposing this silently inverts the analysis."""
    _write(tmp_path)
    signals = inspect_layer(CheckpointIndex.open(str(tmp_path)), 0)
    assert signals.num_experts == E
    assert signals.hidden == H
    assert signals.gate_norm.shape == (E,)
    assert signals.gate_cosine.shape == (E, E)


def test_shared_direction_fraction_detects_a_common_component(tmp_path):
    """A component every row shares inflates raw cosine without discriminating.

    Top-k cancels it, so the diagnostic has to surface it or raw cosine gets
    read as structure that cannot affect routing.
    """
    rng = torch.Generator().manual_seed(9)
    distinct = torch.randn(E, H, generator=rng)
    shared = distinct + 10.0 * torch.ones(E, H)

    _write(tmp_path / "a", gate=distinct)
    _write(tmp_path / "b", gate=shared)
    a = inspect_layer(CheckpointIndex.open(str(tmp_path / "a")), 0)
    b = inspect_layer(CheckpointIndex.open(str(tmp_path / "b")), 0)

    assert b.shared_direction_fraction > a.shared_direction_fraction
    off = ~np.eye(E, dtype=bool)
    # Raw cosine is dominated by the shared part...
    assert np.median(b.gate_cosine[off]) > 0.9
    # ...while centering recovers the distinct structure.
    assert abs(np.median(b.gate_cosine_centered[off])) < 0.5


def test_diagonal_of_cosine_is_one(tmp_path):
    _write(tmp_path)
    signals = inspect_layer(CheckpointIndex.open(str(tmp_path)), 0)
    np.testing.assert_allclose(np.diag(signals.gate_cosine), 1.0, atol=1e-6)


def test_spectra_are_optional_and_populate_when_asked(tmp_path):
    _write(tmp_path)
    index = CheckpointIndex.open(str(tmp_path))
    assert inspect_layer(index, 0).effective_rank is None

    with_spectra = inspect_layer(index, 0, spectra=True)
    assert with_spectra.effective_rank is not None
    assert with_spectra.effective_rank.shape == (E,)
    assert with_spectra.weight_norm.shape == (E,)
    assert (with_spectra.effective_rank > 0).all()


# ---------------------------------------------------------------- the scoring


def test_a_signal_built_to_correlate_scores_high(tmp_path):
    """Sanity on the scorer itself: a gate whose norms track load must score."""
    load = np.array([0.30, 0.25, 0.20, 0.13, 0.08, 0.04])
    rng = torch.Generator().manual_seed(11)
    gate = torch.randn(E, H, generator=rng)
    gate = gate / gate.norm(dim=1, keepdim=True)
    gate = gate * torch.tensor(load * 10).float().unsqueeze(1)  # norm := load
    _write(tmp_path, gate=gate)

    signals = inspect_layer(CheckpointIndex.open(str(tmp_path)), 0)
    scores = score_against_profile(signals, load)
    assert scores["gate_norm_vs_load"] > 0.9


def test_a_noise_signal_scores_near_zero(tmp_path):
    """And the converse, so a high score cannot be an artefact of the method."""
    _write(tmp_path)
    signals = inspect_layer(CheckpointIndex.open(str(tmp_path)), 0)
    load = np.array([0.30, 0.25, 0.20, 0.13, 0.08, 0.04])
    scores = score_against_profile(signals, load)
    assert abs(scores["gate_norm_vs_load"]) < 0.9


def test_bias_is_scored_with_the_inverse_sign(tmp_path):
    """Aux-loss-free balancing pushes popular experts *down*.

    So a large negative bias means popular, and the scorer must correlate
    ``-bias`` with load or it would report a real signal as anti-correlated.
    """
    load = np.array([0.30, 0.25, 0.20, 0.13, 0.08, 0.04])
    bias = torch.tensor(-load * 10.0)  # hottest expert pushed down hardest
    _write(tmp_path, bias=bias)

    signals = inspect_layer(CheckpointIndex.open(str(tmp_path)), 0)
    scores = score_against_profile(signals, load)
    assert scores["neg_bias_vs_load"] > 0.9


def test_cooccurrence_signals_are_scored_when_available(tmp_path):
    _write(tmp_path)
    signals = inspect_layer(CheckpointIndex.open(str(tmp_path)), 0)
    load = np.full(E, 1.0 / E)
    cooc = np.zeros((E, E), dtype=np.int64)
    cooc[0, 1] = cooc[1, 0] = 100
    scores = score_against_profile(signals, load, cooc)
    assert "gate_cosine_vs_cooccurrence" in scores
    assert "gate_cosine_centered_vs_cooccurrence" in scores


def test_expert_count_mismatch_is_refused(tmp_path):
    _write(tmp_path)
    signals = inspect_layer(CheckpointIndex.open(str(tmp_path)), 0)
    with pytest.raises(ValueError, match="profile has"):
        score_against_profile(signals, np.ones(E + 1) / (E + 1))


# ----------------------------------------------------------------- the verdict


def test_verdict_threshold_separates_usable_from_noise():
    calls = verdict({"weak": 0.05, "strong": 0.6, "missing": float("nan")})
    assert "no usable signal" in calls["weak"]
    assert "usable for placement" in calls["strong"]
    assert calls["missing"] == "not measurable"
    # The measured OLMoE gate signals sit near 0.00-0.05, well under the bar.
    assert USABLE_RHO > 0.05


def test_verdict_language_confines_static_signals_to_placement():
    """The report must not invite using these for pruning."""
    assert "placement" in verdict({"x": 0.9})["x"]


def test_report_without_a_profile_says_nothing_was_validated(tmp_path):
    _write(tmp_path)
    signals = inspect_layer(CheckpointIndex.open(str(tmp_path)), 0)
    text = report([(signals, {})], model="test/m")
    assert "nothing was validated" in text
    assert "candidate" in text


def test_report_with_a_profile_shows_correlations_and_the_reminder(tmp_path):
    _write(tmp_path)
    signals = inspect_layer(CheckpointIndex.open(str(tmp_path)), 0)
    load = np.array([0.30, 0.25, 0.20, 0.13, 0.08, 0.04])
    text = report([(signals, score_against_profile(signals, load))], model="test/m")
    assert "gate_norm_vs_load" in text
    assert "verdict:" in text
    assert "Not in pruning or merging" in text


def test_report_handles_no_layers():
    assert "no layers inspected" in report([], model="test/m")


def test_bias_absent_is_stated_in_the_report(tmp_path):
    _write(tmp_path)
    signals = inspect_layer(CheckpointIndex.open(str(tmp_path)), 0)
    text = report([(signals, {})])
    assert "router bias present: no" in text
    assert "aux-loss balanced" in text


def test_zero_initialised_bias_is_not_mistaken_for_a_perfect_signal(tmp_path):
    """The realistic form of the tie bug.

    Some checkpoints ship e_score_correction_bias as zeros. With naive
    argsort-of-argsort ranking, a constant vector rank-correlates perfectly with
    anything, so the tool would have reported an untrained bias as a flawless
    popularity predictor -- and the verdict would have said "usable".
    """
    _write(tmp_path, bias=torch.zeros(E))
    signals = inspect_layer(CheckpointIndex.open(str(tmp_path)), 0)
    assert signals.has_bias

    load = np.array([0.30, 0.25, 0.20, 0.13, 0.08, 0.04])
    scores = score_against_profile(signals, load)
    assert np.isnan(scores["neg_bias_vs_load"])
    assert verdict(scores)["neg_bias_vs_load"] == "not measurable"


def test_ties_are_averaged_not_sequenced():
    from vllm_moe_surgeon.surgery.inspect import _midranks

    np.testing.assert_allclose(_midranks(np.array([5.0, 5.0, 5.0, 5.0])), [1.5] * 4)
    np.testing.assert_allclose(
        _midranks(np.array([1.0, 2.0, 2.0, 3.0])), [0, 1.5, 1.5, 3]
    )
    # A half-tied signal still correlates, just imperfectly.
    rho = spearman([1, 1, 2, 2], [1, 2, 3, 4])
    assert 0.7 < rho < 1.0
