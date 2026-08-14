# SPDX-License-Identifier: Apache-2.0
"""Does a smaller checkpoint already serve this domain better?

The question this answers is the cheapest one in the whole pipeline and it was
found the expensive way. Pruning buys footprint at a measured quality cost --
1.25x perplexity and arc_challenge -25% relative on OLMoE. Measured against an
off-the-shelf 800M-active checkpoint on the same domain, footprint came *free*
and quality improved: 4.8% better bits per byte than the unpruned teacher, and
no significant arc_challenge *acc_norm* loss (-0.014, p=0.51) where pruning cost
11.6 points -- though on raw `acc` the same paired run measured -0.068 at
p=0.0008, so the two arc metrics disagree and the recorded protocol used
`acc_norm`. See docs/benchmarks.md, "Checkpoint selection". So: before tiering
or cutting, score the candidates. It costs minutes and it can retire the rest
of the pipeline.

**Bits per byte, not perplexity.** Per-token perplexity is a property of the
tokenizer as much as the model -- one model splitting the same text into more,
easier tokens scores better while predicting the text no better at all. On the
log corpus OLMoE reads 3.41 bytes/token against granite's 2.64, which alone
moves per-token perplexity by more than any effect worth reporting. Bits per
byte divides the same total loss by the text instead of by the tokenization, so
it is the only figure comparable across models. Per-token perplexity is still
reported, for comparison against numbers recorded on *that* model.

**One model per process**, for the reason `compat/bench.py` records: a vLLM
engine does not release its device memory when the object goes out of scope, so
a second model in the same process starts against the first one's allocation and
simply fails to boot. Each candidate is scored in a child.

What this does **not** answer: whether the winner does the *task*. The score is
held-out likelihood, i.e. how well the model compresses the domain's text, and
this project has already measured perplexity and task accuracy parting ways.
Treat a win here as permission to run the task evaluation, not as its result.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

from .._logging import init_logger

logger = init_logger(__name__)


@dataclass
class Score:
    """One candidate's likelihood on the corpus, or why it could not be scored."""

    model: str
    #: The comparable figure: total loss divided by the corpus's bytes.
    bits_per_byte: float = float("nan")
    #: Comparable only against numbers recorded on this same model.
    perplexity: float = float("nan")
    scored_tokens: int = 0
    bytes_per_token: float = float("nan")
    load_seconds: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and math.isfinite(self.bits_per_byte)


@dataclass
class Headroom:
    """Every candidate scored on one corpus, best first."""

    corpus: str
    corpus_bytes: int
    n_prompts: int
    scores: list[Score] = field(default_factory=list)

    @property
    def ranked(self) -> list[Score]:
        return sorted(
            (s for s in self.scores if s.ok), key=lambda s: s.bits_per_byte
        )

    @property
    def failed(self) -> list[Score]:
        return [s for s in self.scores if not s.ok]


# The child: one model, one corpus, one number. Kept as source rather than a
# module entry point so it cannot import anything the parent has already loaded.
_PROBE = """
import json, math, sys, time
model, corpus, extra = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
prompts = [json.loads(line)["__text__"] for line in open(corpus) if line.strip()]
start = time.perf_counter()
from vllm import LLM
from vllm_moe_surgeon.compat.ablation import measure_nll
llm = LLM(model=model, **extra)
load = time.perf_counter() - start
nll, tokens = measure_nll(llm, prompts)
print("SCORE_JSON" + json.dumps({
    "nll": nll, "tokens": tokens, "load_seconds": load,
}))
"""


def _kill_group(proc: subprocess.Popen) -> None:
    """Take down the child and everything it spawned, then reap it.

    A vLLM engine's workers are children of the child, so killing the direct
    process leaves them holding the device and the next candidate boots against
    a GPU that is not free.
    """
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError):
        # Already reaped, not ours, or no process groups on this platform.
        proc.kill()
    proc.communicate()


def _diagnose(completed: subprocess.CompletedProcess) -> str:
    """Why the child failed, avoiding the shutdown warning every failure ends on."""
    markers = (
        "No available memory for the cache blocks",
        "out of memory",
        "does not appear to have a file named",
        "is not a local folder",
        "Unrecognized model",
        "not supported",
    )
    text = f"{completed.stdout}\n{completed.stderr}"
    for marker in markers:
        for line in text.splitlines():
            if marker in line:
                return line.strip()[:200]
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped and "destroy_process_group" not in stripped:
            return stripped[:200]
    return f"child exited {completed.returncode} with no diagnosable output"


def score_model(
    model: str,
    corpus_path: str,
    *,
    llm_kwargs: dict[str, Any] | None = None,
    timeout: float = 1800.0,
    corpus_bytes: int | None = None,
) -> Score:
    """Score one candidate in a fresh process. Never raises for a failed model.

    ``corpus_path`` is a JSONL whose records carry the prompt under ``__text__``
    -- written by :func:`stage_corpus`, so the child needs no field convention.
    """
    if corpus_bytes is None:
        corpus_bytes = _corpus_bytes(corpus_path)
    argv = [
        sys.executable,
        "-c",
        _PROBE,
        model,
        corpus_path,
        json.dumps(llm_kwargs or {}),
    ]
    # Popen rather than `run(timeout=...)`: on timeout `run` kills only the
    # direct child, and a vLLM engine's workers would survive it still holding
    # the device -- so the next candidate boots against a GPU that is not free.
    # Its own session lets us signal the whole group.
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(os.environ),
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # The docstring's promise is the whole point of the child: a candidate
        # that hangs costs its own row, never the run.
        _kill_group(proc)
        return Score(model=model, error=f"timed out after {timeout:.0f}s")
    except BaseException:
        # Its own session means a terminal Ctrl-C no longer reaches the child,
        # and `subprocess.run`'s implicit kill-on-KeyboardInterrupt is gone with
        # it -- so every other unwind path has to take the group down too, or an
        # interrupted run leaks a booted engine still holding the device.
        _kill_group(proc)
        raise
    completed = subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)
    line = next(
        (
            ln
            for ln in completed.stdout.splitlines()
            if ln.startswith("SCORE_JSON")
        ),
        None,
    )
    if completed.returncode != 0 or line is None:
        return Score(model=model, error=_diagnose(completed))

    payload = json.loads(line[len("SCORE_JSON") :])
    nll, tokens = float(payload["nll"]), int(payload["tokens"])
    if tokens <= 0 or corpus_bytes <= 0:
        return Score(model=model, error="nothing was scored")
    return Score(
        model=model,
        bits_per_byte=nll / (corpus_bytes * math.log(2)),
        perplexity=math.exp(nll / tokens),
        scored_tokens=tokens,
        bytes_per_token=corpus_bytes / tokens,
        load_seconds=float(payload["load_seconds"]),
    )


def _corpus_bytes(path: str) -> int:
    total = 0
    with open(path) as handle:
        for line in handle:
            if line.strip():
                total += len(json.loads(line)["__text__"].encode("utf-8"))
    return total


def stage_corpus(prompts: list[str], path: str) -> int:
    """Write the prompts where the children read them; return the byte count.

    Staged once rather than re-read per child so every candidate is scored on
    byte-identical text -- the corpus is the one thing that must not vary
    between arms.
    """
    total = 0
    with open(path, "w") as handle:
        for prompt in prompts:
            handle.write(json.dumps({"__text__": prompt}) + "\n")
            total += len(prompt.encode("utf-8"))
    return total


def report(result: Headroom) -> str:
    """The table, the winner, and what the number does not certify."""
    lines = [
        f"corpus: {result.n_prompts} prompts, "
        f"{result.corpus_bytes} bytes ({result.corpus})",
        "",
        f"  {'model':<44}{'bits/byte':>11}{'ppl':>10}{'B/token':>9}",
    ]
    for score in result.ranked:
        lines.append(
            f"  {score.model[:44]:<44}{score.bits_per_byte:>11.4f}"
            f"{score.perplexity:>10.2f}{score.bytes_per_token:>9.2f}"
        )
    for score in result.failed:
        lines.append(f"  {score.model[:44]:<44}{'not scored':>30}")
        lines.append(f"      {score.error}")

    ranked = result.ranked
    if not ranked:
        lines.append("")
        lines.append("no candidate could be scored; nothing to compare")
        return "\n".join(lines)

    lines.append("")
    best = ranked[0]
    if len(ranked) == 1:
        lines.append(f"only {best.model} was scored: nothing to compare it against")
    else:
        runner_up = ranked[1]
        margin = 100 * (1 - best.bits_per_byte / runner_up.bits_per_byte)
        lines.append(
            f"best on this domain: {best.model} "
            f"({margin:.1f}% better bits/byte than {runner_up.model})"
        )
        # Per-token perplexity is a tokenizer artefact as much as a model
        # property; say so wherever the candidates disagree enough to mislead.
        spread = max(s.bytes_per_token for s in ranked) / min(
            s.bytes_per_token for s in ranked
        )
        if spread > 1.05:
            lines.append(
                f"  the ppl column is NOT comparable across these rows: they "
                f"tokenize this text {spread:.2f}x differently. Compare bits/byte."
            )
    # Every path that printed a bits/byte figure carries the caveat, including
    # the single-candidate one -- a number on screen without it is the failure
    # mode this whole command exists to prevent.
    lines.append(
        "  this is held-out likelihood, not task accuracy -- the two have "
        "already parted ways once here (docs/benchmarks.md: pruning kept "
        "perplexity and lost 25% of arc_challenge). Evaluate the winner on "
        "real tasks before adopting it."
    )
    return "\n".join(lines)
