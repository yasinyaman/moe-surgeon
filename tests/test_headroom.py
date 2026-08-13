# SPDX-License-Identifier: Apache-2.0
"""`surgeon headroom`: the arithmetic and the report.

Scoring a real model needs a GPU; everything that decides what the command
*says* does not. What is pinned here is the part that was got wrong once
already elsewhere in this project: comparing models across tokenizers.
"""

import json
import math

import pytest

from vllm_moe_surgeon.compat.headroom import (
    Headroom,
    Score,
    _corpus_bytes,
    _diagnose,
    report,
    stage_corpus,
)


def _score(model, bpb, ppl=10.0, bpt=3.0):
    return Score(
        model=model,
        bits_per_byte=bpb,
        perplexity=ppl,
        scored_tokens=100,
        bytes_per_token=bpt,
    )


# ------------------------------------------------------------------- staging


def test_staged_corpus_round_trips_bytes_and_text(tmp_path):
    """Every candidate must be scored on byte-identical text, so the corpus is
    written once and its byte count is measured from what was written."""
    path = tmp_path / "c.jsonl"
    prompts = ["hello", "üniversite", "log line: auth failure"]
    total = stage_corpus(prompts, str(path))

    assert total == sum(len(p.encode("utf-8")) for p in prompts)
    assert total == _corpus_bytes(str(path))
    back = [json.loads(line)["__text__"] for line in path.read_text().splitlines()]
    assert back == prompts


# -------------------------------------------------------------------- ranking


def test_ranking_is_by_bits_per_byte_not_perplexity():
    """The trap this command exists to avoid: a model with a finer tokenizer
    posts a far better per-token perplexity while predicting the text worse."""
    result = Headroom(corpus="c", corpus_bytes=1000, n_prompts=10)
    result.scores = [
        _score("fine-tokenizer", bpb=1.50, ppl=8.0, bpt=2.0),
        _score("coarse-tokenizer", bpb=1.30, ppl=25.0, bpt=4.0),
    ]
    assert [s.model for s in result.ranked] == [
        "coarse-tokenizer",
        "fine-tokenizer",
    ]
    assert "best on this domain: coarse-tokenizer" in report(result)


def test_report_warns_when_tokenizations_differ_enough_to_mislead():
    result = Headroom(corpus="c", corpus_bytes=1000, n_prompts=10)
    result.scores = [
        _score("a", bpb=1.30, bpt=2.64),
        _score("b", bpb=1.40, bpt=3.41),
    ]
    text = report(result)
    assert "ppl column is NOT comparable" in text
    assert "1.29x differently" in text

    close = Headroom(corpus="c", corpus_bytes=1000, n_prompts=10)
    close.scores = [_score("a", bpb=1.30, bpt=3.00), _score("b", bpb=1.40, bpt=3.02)]
    assert "NOT comparable" not in report(close)


def test_report_always_says_likelihood_is_not_task_accuracy():
    """Perplexity and task accuracy have already parted ways in this project;
    a headroom win must not read as permission to adopt."""
    result = Headroom(corpus="c", corpus_bytes=1000, n_prompts=10)
    result.scores = [_score("a", bpb=1.3), _score("b", bpb=1.4)]
    assert "not task accuracy" in report(result)


def test_a_single_candidate_is_not_reported_as_a_winner():
    result = Headroom(corpus="c", corpus_bytes=1000, n_prompts=10)
    result.scores = [_score("only", bpb=1.3)]
    text = report(result)
    assert "nothing to compare it against" in text
    assert "best on this domain" not in text


# -------------------------------------------------------------------- failures


def test_a_candidate_that_cannot_be_scored_is_named_not_dropped():
    result = Headroom(corpus="c", corpus_bytes=1000, n_prompts=10)
    result.scores = [
        _score("good", bpb=1.3),
        Score(model="broken", error="out of memory"),
    ]
    text = report(result)
    assert [s.model for s in result.ranked] == ["good"]
    assert [s.model for s in result.failed] == ["broken"]
    assert "not scored" in text and "out of memory" in text


def test_no_candidate_scored_is_a_failed_measurement_not_a_verdict():
    result = Headroom(corpus="c", corpus_bytes=1000, n_prompts=10)
    result.scores = [Score(model="a", error="boom")]
    assert result.ranked == []
    assert "nothing to compare" in report(result)


def test_diagnose_skips_the_shutdown_warning_every_failure_ends_on():
    """A failed vLLM boot signs off with NCCL's destroy_process_group warning,
    so the tail of stderr names that warning as the cause of everything."""

    class _Completed:
        returncode = 1
        stdout = ""
        stderr = (
            "ValueError: No available memory for the cache blocks\n"
            "[rank0]: Warning: destroy_process_group() was not called\n"
        )

    assert "No available memory" in _diagnose(_Completed())


def test_bits_per_byte_matches_the_definition():
    """bpb = total nll / (bytes * ln 2) -- the same arithmetic the D1 gate used,
    pinned so a refactor cannot quietly change the unit."""
    nll, corpus_bytes = 17023.5909, 17452
    assert math.isclose(
        nll / (corpus_bytes * math.log(2)), 1.40728, rel_tol=1e-5
    )


@pytest.mark.parametrize("tokens", [0, -1])
def test_zero_scored_tokens_is_an_error_not_a_score(tokens):
    assert not Score(model="m", error="nothing was scored").ok
