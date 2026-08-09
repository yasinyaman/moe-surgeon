# SPDX-License-Identifier: Apache-2.0
"""The bisection's logic, exercised without a GPU.

Every test here replaces ``probe`` with a fake whose floor is known, so what is under
test is the search: does it find the boundary, does it bracket it, and does it tell a
model that boots nowhere apart from a model that boots cheaply. The real boots are
measured on the GPU boxes and recorded in the README.
"""

from __future__ import annotations

import json

import pytest

from vllm_moe_surgeon.compat import bisect_vram
from vllm_moe_surgeon.compat.bisect_vram import Attempt, bisect_floor


@pytest.fixture(autouse=True)
def _fixed_device(monkeypatch):
    monkeypatch.setattr(bisect_vram, "device_total_gib", lambda: 100.0)


def _fake_probe(true_floor, log=None):
    """Boots at or above ``true_floor``. Records what it was asked."""

    def probe(model, fraction, **kwargs):
        if log is not None:
            log.append(fraction)
        return Attempt(fraction, fraction >= true_floor, seconds=1.0)

    return probe


def test_it_finds_the_boundary_within_tolerance(monkeypatch):
    monkeypatch.setattr(bisect_vram, "probe", _fake_probe(0.42))
    result = bisect_floor("m", low=0.05, high=0.90, tolerance=0.01)

    assert result.floor_fraction is not None
    # The reported floor must actually boot, and be within tolerance of the truth.
    assert result.floor_fraction >= 0.42
    assert result.floor_fraction - 0.42 <= 0.01
    assert result.floor_gib == pytest.approx(result.floor_fraction * 100.0)


def test_the_reported_floor_is_a_fraction_that_booted(monkeypatch):
    """Never a probed failure. A floor that does not boot is worse than no floor."""
    monkeypatch.setattr(bisect_vram, "probe", _fake_probe(0.31))
    result = bisect_floor("m", low=0.05, high=0.90, tolerance=0.005)

    booted = {a.fraction for a in result.attempts if a.booted}
    assert result.floor_fraction in booted
    assert result.highest_failure is not None
    assert result.highest_failure < result.floor_fraction


def test_a_model_that_boots_nowhere_is_not_a_floor_of_low(monkeypatch):
    """The failure mode this ordering exists to prevent.

    Bisecting a broken configuration walks the bracket down and would report ``low``
    -- the smallest number in the range, which reads as a spectacular result and
    means the opposite. So the ceiling is probed first and a failure there stops the
    search.
    """
    log: list[float] = []
    monkeypatch.setattr(bisect_vram, "probe", _fake_probe(2.0, log))  # boots nowhere
    result = bisect_floor("m", low=0.05, high=0.90)

    assert result.floor_fraction is None
    assert result.floor_gib is None
    assert result.highest_failure == 0.90
    assert log == [0.90], "must not bisect after the ceiling fails"
    assert "broken configuration" in result.report()


def test_the_ceiling_is_probed_before_anything_else(monkeypatch):
    log: list[float] = []
    monkeypatch.setattr(bisect_vram, "probe", _fake_probe(0.20, log))
    bisect_floor("m", low=0.05, high=0.80, tolerance=0.05)
    assert log[0] == 0.80


def test_a_model_that_boots_everywhere_is_reported_as_an_upper_bound(monkeypatch):
    """The flaw a real measurement exposed.

    Two Granite arms 3.7 GiB apart in weights both came back "floor 0.059", because
    neither ever failed -- the true floor was below the range for both. Calling that
    a floor makes different arms look equal, which is exactly the comparison this
    tool exists to support. So an unbracketed result has to say so.
    """
    monkeypatch.setattr(bisect_vram, "probe", _fake_probe(0.0))
    result = bisect_floor("m", low=0.05, high=0.90, tolerance=0.01)

    # Nothing below `low` is ever probed, so `low` is the best answer available.
    assert result.floor_fraction == pytest.approx(0.05, abs=0.02)
    assert result.highest_failure is None
    assert result.bracketed is False
    text = result.report()
    assert "UNBRACKETED" in text
    assert "upper bound" in text
    assert "<=" in text


def test_a_bracketed_result_does_not_hedge(monkeypatch):
    monkeypatch.setattr(bisect_vram, "probe", _fake_probe(0.42))
    result = bisect_floor("m", low=0.05, high=0.90, tolerance=0.01)
    assert result.bracketed is True
    assert "UNBRACKETED" not in result.report()


def test_fewer_boots_than_a_linear_sweep(monkeypatch):
    """Each attempt is a full engine boot, so the count is the cost."""
    monkeypatch.setattr(bisect_vram, "probe", _fake_probe(0.42))
    result = bisect_floor("m", low=0.05, high=0.90, tolerance=0.01)
    # log2(0.85 / 0.01) ~= 6.4, plus the ceiling probe.
    assert len(result.attempts) <= 9


def test_a_bad_range_is_refused():
    for low, high in ((0.5, 0.5), (0.6, 0.4), (0.0, 0.9), (0.1, 1.5)):
        with pytest.raises(ValueError, match="low"):
            bisect_floor("m", low=low, high=high)


def test_the_report_shows_every_probe(monkeypatch):
    monkeypatch.setattr(bisect_vram, "probe", _fake_probe(0.42))
    result = bisect_floor("m", low=0.05, high=0.90, tolerance=0.02)
    text = result.report()
    assert "floor" in text
    assert "bracketed" in text
    assert result.bracketed
    for attempt in result.attempts:
        assert f"{attempt.fraction:.3f}" in text


def test_serialisation_carries_the_gib_figure(monkeypatch):
    monkeypatch.setattr(bisect_vram, "probe", _fake_probe(0.42))
    result = bisect_floor("m", low=0.05, high=0.90, tolerance=0.02)
    payload = bisect_vram.to_dict(result)
    assert payload["floor_gib"] == pytest.approx(result.floor_fraction * 100.0)
    assert len(payload["attempts"]) == len(result.attempts)


def test_a_timeout_counts_as_not_booting(monkeypatch):
    """A hang is a failure at that budget, not a reason to abandon the search."""
    seen = []

    def probe(model, fraction, **kwargs):
        seen.append(fraction)
        if fraction < 0.5:
            return Attempt(fraction, False, 900.0, error="timeout")
        return Attempt(fraction, True, 30.0)

    monkeypatch.setattr(bisect_vram, "probe", probe)
    result = bisect_floor("m", low=0.05, high=0.90, tolerance=0.02)
    assert result.floor_fraction is not None
    assert result.floor_fraction >= 0.5
    assert any(a.error == "timeout" for a in result.attempts)


# ------------------------------------------------- what a failed boot is blamed on


class _Completed:
    def __init__(self, stderr="", stdout="", returncode=1):
        self.stderr, self.stdout, self.returncode = stderr, stdout, returncode


def test_the_blamed_line_is_the_error_not_the_signoff():
    """A failed vLLM boot's *last* stderr line is a shutdown warning, not the cause.

    Measured: every failure came back attributed to NCCL's "destroy_process_group()
    was not called", which is emitted after the real error and says nothing about
    why the budget was too small.
    """
    from vllm_moe_surgeon.compat.bisect_vram import _diagnose

    stderr = (
        "INFO loading model\n"
        "ValueError: No available memory for the cache blocks. Try increasing "
        "`gpu_memory_utilization`.\n"
        "[rank0]:[W809] Warning: WARNING: destroy_process_group() was not called "
        "before program exit, which can leak resources.\n"
    )
    blamed = _diagnose(_Completed(stderr=stderr), 0.05)
    assert "cache blocks" in blamed
    assert "destroy_process_group" not in blamed


def test_an_oom_is_recognised_over_a_generic_error():
    from vllm_moe_surgeon.compat.bisect_vram import _diagnose

    stderr = "RuntimeError: something\ntorch.OutOfMemoryError: CUDA out of memory\n"
    assert "out of memory" in _diagnose(_Completed(stderr=stderr), 0.05).lower()


def test_a_silent_failure_still_says_something():
    from vllm_moe_surgeon.compat.bisect_vram import _diagnose

    assert _diagnose(_Completed(returncode=137), 0.05) == "exit 137"


def test_the_result_records_what_it_was_booted_with(monkeypatch):
    """A floor without its configuration is not interpretable.

    Three arms were once written up as "tiered, ram_cache 0" when that setting had
    turned the disk tier off entirely. Nothing in the saved result could have
    contradicted the label, so the label went unchallenged into the README.
    """
    monkeypatch.setattr(bisect_vram, "probe", _fake_probe(0.42))
    kwargs = {
        "enforce_eager": True,
        "additional_config": {"surgeon": {"expert_cache_size": 24, "ram_cache": 48}},
    }
    result = bisect_floor("m", low=0.05, high=0.90, tolerance=0.02, llm_kwargs=kwargs)

    assert result.llm_kwargs == kwargs
    assert "ram_cache" in result.report()
    assert "ram_cache" in json.dumps(bisect_vram.to_dict(result))


def test_an_unconfigured_run_does_not_invent_a_config_line(monkeypatch):
    monkeypatch.setattr(bisect_vram, "probe", _fake_probe(0.42))
    result = bisect_floor("m", low=0.05, high=0.90, tolerance=0.02)
    assert result.llm_kwargs == {}
    assert "booted with" not in result.report()
