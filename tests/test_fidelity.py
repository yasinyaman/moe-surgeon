# SPDX-License-Identifier: Apache-2.0
"""The fidelity maths, pinned against hand-computed KL divergences.

No engine and no GPU: the arms arrive as ``.npz`` captures, which is the whole
reason the maths lives outside ``compat/``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vllm_moe_surgeon.surgery.fidelity import (
    MAX_SUBSTITUTION,
    Capture,
    compare,
    report,
    to_dict,
)


def _capture(rows: list[tuple[list[int], list[float]]], doc: list[int] | None = None):
    """Build a capture from (ids, probabilities) rows; probabilities are logged."""
    ids = np.asarray([r[0] for r in rows], dtype=np.int32)
    logprobs = np.asarray(
        [[math.log(p) for p in r[1]] for r in rows], dtype=np.float32
    )
    return Capture(
        ids=ids,
        logprobs=logprobs,
        doc=np.asarray(doc if doc is not None else [0] * len(rows), dtype=np.int32),
    )


def test_identical_captures_measure_as_identical():
    cap = _capture([([1, 2], [0.6, 0.4]), ([3, 4], [0.9, 0.1])])
    result = compare(cap, cap, arm="same")

    assert result.top1_agreement == 1.0
    assert result.kld_mean == 0.0
    assert result.dp_rms == 0.0
    assert result.identical
    # Deliberately not the words "bit-identical": a top-K capture can only
    # establish that nothing moved within the capture, and the report says so.
    text = report("ref", [result])
    assert "no captured difference" in text
    assert "token-hash control" in text


def test_kld_matches_the_hand_computed_value():
    ref = _capture([([1, 2], [0.6, 0.4])])
    test = _capture([([1, 2], [0.5, 0.5])])
    expected = 0.6 * math.log(0.6 / 0.5) + 0.4 * math.log(0.4 / 0.5)

    result = compare(ref, test, arm="flatter")

    assert result.kld_mean == pytest.approx(expected, abs=1e-6)
    # Both arms still rank token 1 first, so the sampler is unaffected even
    # though the distribution moved -- the two statistics are not redundant.
    assert result.top1_agreement == 1.0
    assert result.dp_mean == pytest.approx(-10.0, abs=1e-4)


def test_a_reordered_top_1_is_caught_by_agreement_and_by_kld():
    ref = _capture([([1, 2], [0.6, 0.4])])
    test = _capture([([2, 1], [0.7, 0.3])])
    expected = 0.6 * math.log(0.6 / 0.3) + 0.4 * math.log(0.4 / 0.7)

    result = compare(ref, test, arm="swapped")

    assert result.top1_agreement == 0.0
    assert result.kld_mean == pytest.approx(expected, abs=1e-6)


def test_kld_is_labelled_a_lower_bound_when_tokens_had_to_be_substituted():
    # Refusing instead was the first design and it was measured wrong: raising K
    # does not clear a heavy tail (OLMoE/gsm8k, K 32->128 moved the substitution
    # rate 3.22% -> 2.97%), so refusing means never reporting KL at all.
    ref = _capture([([1, 2], [0.6, 0.4])])
    test = _capture([([3, 4], [0.6, 0.4])])

    result = compare(ref, test, arm="disjoint")

    assert result.substitution_rate == 1.0
    assert not result.kld_is_exact
    text = report("ref", [result])
    assert "LOWER BOUND" in text
    assert ">=" in text


def test_a_rare_substitution_stays_within_the_trust_threshold():
    # 1 of 100 reference tokens missing from the arm's top-K.
    rows_ref = [([1, 2], [0.6, 0.4])] * 50
    rows_test = [([1, 2], [0.6, 0.4])] * 50
    rows_ref[0] = ([1, 9], [0.6, 0.4])
    ref, test = _capture(rows_ref), _capture(rows_test)

    result = compare(ref, test, arm="mostly-covered")

    assert result.substitution_rate == pytest.approx(0.01)
    assert result.substitution_rate <= MAX_SUBSTITUTION
    assert result.kld_is_exact


def test_mismatched_corpora_are_refused_rather_than_compared():
    ref = _capture([([1, 2], [0.6, 0.4]), ([1, 2], [0.6, 0.4])])
    test = _capture([([1, 2], [0.6, 0.4])])

    with pytest.raises(ValueError, match="different corpora"):
        compare(ref, test)


def test_captures_that_scored_different_text_are_refused():
    ref = _capture([([1, 2], [0.6, 0.4]), ([1, 2], [0.6, 0.4])], doc=[0, 1])
    test = _capture([([1, 2], [0.6, 0.4]), ([1, 2], [0.6, 0.4])], doc=[0, 0])

    with pytest.raises(ValueError, match="same text in the same order"):
        compare(ref, test, arm="other-corpus")


def test_an_empty_reference_is_refused():
    empty = Capture(
        ids=np.zeros((0, 2), dtype=np.int32),
        logprobs=np.zeros((0, 2), dtype=np.float32),
        doc=np.zeros((0,), dtype=np.int32),
    )
    with pytest.raises(ValueError, match="empty"):
        compare(empty, empty)


def test_capture_round_trips_through_npz(tmp_path):
    cap = _capture([([1, 2], [0.6, 0.4])])
    cap.meta = {"arm": "tier", "top_k": 2}
    path = str(tmp_path / "cap.npz")
    cap.save(path)

    loaded = Capture.load(path)

    assert loaded.meta == {"arm": "tier", "top_k": 2}
    assert np.array_equal(loaded.ids, cap.ids)
    assert loaded.positions == 1
    assert loaded.top_k == 2


def test_low_coverage_is_called_out_in_the_report():
    # A top-2 holding only 30% of the mass: the truncation, not the arm, is what
    # the KL figure would mostly be measuring.
    ref = _capture([([1, 2], [0.2, 0.1])])
    test = _capture([([1, 2], [0.2, 0.1])])

    result = compare(ref, test, arm="thin")

    assert result.coverage == pytest.approx(0.3, abs=1e-6)
    assert "competing with the effect" in report("ref", [result])


def test_to_dict_is_json_safe():
    import json

    ref = _capture([([1, 2], [0.6, 0.4])])
    test = _capture([([1, 2], [0.5, 0.5])])

    payload = to_dict(compare(ref, test, arm="a"))

    assert json.loads(json.dumps(payload))["arm"] == "a"


def test_an_arm_that_asked_for_coexecution_is_detected():
    """A mechanism that never engaged must not be reported as 'no difference'.

    This project has already shipped a substitution that silently no-opped while
    the tokens matched perfectly, so a capture that produces identical numbers is
    exactly the case where the counters have to be checked.
    """
    from vllm_moe_surgeon.compat.fidelity import _asked_for_coexec

    on = {"additional_config": {"surgeon": {"cpu_experts": True}}}
    off = {"additional_config": {"surgeon": {"expert_cache_size": 4}}}

    assert _asked_for_coexec(on)
    assert not _asked_for_coexec(off)
    assert not _asked_for_coexec({})
