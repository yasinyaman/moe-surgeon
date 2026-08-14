# SPDX-License-Identifier: Apache-2.0
"""`surgeon headroom`: the arithmetic and the report.

Scoring a real model needs a GPU; everything that decides what the command
*says* does not. What is pinned here is the part that was got wrong once
already elsewhere in this project: comparing models across tokenizers.
"""

import json
import math

import pytest

from vllm_moe_surgeon.compat import headroom
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
    # It still printed a bits/byte figure, so it still owes the caveat. This
    # path used to return before reaching it.
    assert "1.3000" in text
    assert "not task accuracy" in text


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


# ------------------------------------------------------- the arithmetic itself
#
# These drive `score_model` end to end against a stubbed child. The real child
# imports vLLM; the stub prints the same SCORE_JSON line, so every figure the
# command reports is computed by the shipped code path rather than restated in
# the test. Asserting the formula inline against literals -- which is what stood
# here before -- only proves that Python divides.


def _stub_probe(body: str) -> str:
    return "import json, sys, time\n" + body


def _corpus(tmp_path):
    path = tmp_path / "c.jsonl"
    stage_corpus(["irrelevant: the stub does not read it"], str(path))
    return str(path)


def test_score_model_reproduces_the_d1_measurement(tmp_path, monkeypatch):
    """The recorded D1 gate: OLMoE-full on the log corpus scored 1.4073
    bits/byte, 27.87 perplexity, 3.41 bytes/token. Feeding the child's own
    payload back through `score_model` must land on all three."""
    monkeypatch.setattr(
        headroom,
        "_PROBE",
        _stub_probe(
            'print("SCORE_JSON" + json.dumps('
            '{"nll": 17023.5909, "tokens": 5116, "load_seconds": 12.5}))\n'
        ),
    )
    score = headroom.score_model(
        "olmoe", _corpus(tmp_path), corpus_bytes=17452, timeout=60
    )

    assert score.ok
    assert math.isclose(score.bits_per_byte, 1.40728, rel_tol=1e-5)
    assert math.isclose(score.perplexity, 27.869, rel_tol=1e-4)
    assert math.isclose(score.bytes_per_token, 3.4113, rel_tol=1e-4)
    assert score.scored_tokens == 5116
    assert score.load_seconds == 12.5
    # bits/byte divides by the text, so it must not move when the tokenizer
    # does -- that is the entire reason this is the ranked column.
    monkeypatch.setattr(
        headroom,
        "_PROBE",
        _stub_probe(
            'print("SCORE_JSON" + json.dumps('
            '{"nll": 17023.5909, "tokens": 6612, "load_seconds": 1.0}))\n'
        ),
    )
    finer = headroom.score_model(
        "finer-tokenizer", _corpus(tmp_path), corpus_bytes=17452, timeout=60
    )
    assert math.isclose(finer.bits_per_byte, score.bits_per_byte, rel_tol=1e-9)
    assert finer.perplexity < score.perplexity


@pytest.mark.parametrize("tokens", [0, -1])
def test_zero_scored_tokens_is_an_error_not_a_score(tmp_path, monkeypatch, tokens):
    """A child that reports no tokens has measured nothing. Dividing by it would
    produce inf or a negative bits/byte that would then rank first."""
    monkeypatch.setattr(
        headroom,
        "_PROBE",
        _stub_probe(
            'print("SCORE_JSON" + json.dumps('
            f'{{"nll": 100.0, "tokens": {tokens}, "load_seconds": 1.0}}))\n'
        ),
    )
    score = headroom.score_model(
        "m", _corpus(tmp_path), corpus_bytes=1000, timeout=60
    )
    assert not score.ok
    assert score.error == "nothing was scored"


def test_a_hanging_candidate_costs_its_own_row_not_the_run(tmp_path, monkeypatch):
    """`score_model` promises never to raise for a failed model. A child that
    hangs is the one failure that used to break that promise, taking every
    already-scored candidate down with it."""
    monkeypatch.setattr(headroom, "_PROBE", _stub_probe("time.sleep(30)\n"))
    score = headroom.score_model(
        "hangs", _corpus(tmp_path), corpus_bytes=1000, timeout=0.5
    )
    assert not score.ok
    assert "timed out" in score.error


def test_a_child_that_dies_is_named_with_its_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(
        headroom,
        "_PROBE",
        _stub_probe(
            'sys.stderr.write("OSError: is not a local folder\\n"); sys.exit(1)\n'
        ),
    )
    score = headroom.score_model(
        "missing", _corpus(tmp_path), corpus_bytes=1000, timeout=60
    )
    assert not score.ok
    assert "is not a local folder" in score.error


def test_the_artifact_is_valid_json_when_a_candidate_fails(tmp_path, monkeypatch):
    """`vars(Score)` carries NaN for an unscored candidate, and bare NaN is a
    Python extension that a strict parser rejects -- taking the successful rows
    down with it.

    Driven through `main` rather than through the helper, because the defect
    was in the writer: asserting on `_jsonable` alone would stay green with
    `json.dump(vars(s))` put back.
    """
    import vllm_moe_surgeon.cli as cli

    def _fake(model, *_a, **_kw):
        if model == "good":
            return _score("good", 1.3)
        return Score(model=model, error="out of memory")

    monkeypatch.setattr(headroom, "score_model", _fake)
    corpus = tmp_path / "c.jsonl"
    corpus.write_text('{"text": "a log line"}\n')
    out = tmp_path / "headroom.json"
    rc = cli.main(
        ["headroom", "--corpus", str(corpus), "--model", "good",
         "--model", "broken", "--out", str(out)]
    )

    text = out.read_text()
    assert "NaN" not in text
    # A strict reader: json.loads accepts NaN by default, so the default is
    # exactly what would hide this.
    back = json.loads(
        text, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c!r}")
    )
    rows = {row["model"]: row for row in back["scores"]}
    assert rows["good"]["bits_per_byte"] == 1.3
    assert rows["broken"]["bits_per_byte"] is None
    assert rows["broken"]["error"] == "out of memory"
    # One of two candidates scored is a partial result, not a failed stage:
    # headroom runs first in the pipeline and is deliberately not a gate.
    assert rc == 0
