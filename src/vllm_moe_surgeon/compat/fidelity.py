# SPDX-License-Identifier: Apache-2.0
"""Capture each arm's top-K next-token distributions, one child process per arm.

The maths lives in :mod:`vllm_moe_surgeon.surgery.fidelity`, which imports no
vLLM; this module is only the part that has to boot an engine. The split is
deliberate: a capture taken on the GPU box is a plain ``.npz`` that can be
compared anywhere, and the comparison keeps its tests on a laptop with no CUDA.

**One arm per process**, for the reason `compat/bench.py` and `compat/headroom.py`
both record: a vLLM engine does not release its device memory when the object
goes out of scope, so the second arm boots against the first arm's allocation and
simply fails. This also means each arm can carry its own ``additional_config``,
which is the whole point -- the tier, the expert split and ``fp8_store`` are
configurations, not models.

**Teacher forcing, not generation.** Every arm scores the same given text with
``prompt_logprobs``, so what is compared is the model's distribution at identical
positions. Comparing generations instead would confound the change under test
with the divergence its own earlier tokens caused.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

from .._logging import init_logger
from ..surgery.fidelity import Capture
from .headroom import _diagnose, _kill_group

logger = init_logger(__name__)

#: Enough to cover ~all of a peaked LM's mass without making the engine hold a
#: vocabulary-sized table per position. Raise it if `compare` refuses.
DEFAULT_TOP_K = 32

_PROBE = """
import json, sys, time
import numpy as np

model, corpus, out = sys.argv[1], sys.argv[2], sys.argv[3]
extra_s, k_s = sys.argv[4], sys.argv[5]
extra = json.loads(extra_s)
K = int(k_s)
prompts = [json.loads(line)["__text__"] for line in open(corpus) if line.strip()]

start = time.perf_counter()
from vllm import LLM, SamplingParams
llm = LLM(model=model, **extra)
load = time.perf_counter() - start

params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=K)
outputs = llm.generate(prompts, params)

ids, lps, docs = [], [], []
for doc, output in enumerate(outputs):
    for entry in (output.prompt_logprobs or []):
        if entry is None:
            continue  # the first token of a prompt has no prediction
        items = sorted(entry.items(), key=lambda kv: kv[1].logprob, reverse=True)
        if len(items) < K:
            raise RuntimeError(
                "position returned %d logprobs, fewer than the requested top-%d; "
                "lower --top-k" % (len(items), K)
            )
        ids.append([int(t) for t, _ in items[:K]])
        lps.append([float(v.logprob) for _, v in items[:K]])
        docs.append(doc)

if not ids:
    raise RuntimeError(
        "no prompt logprobs were returned; this engine cannot be measured this way"
    )

# What the tier actually did while producing this capture. Without it, an arm
# whose mechanism never engaged is indistinguishable from one that engaged and
# changed nothing -- and this project has already shipped a substitution that
# silently no-opped while the tokens matched perfectly.
counters = None
counter_error = None
try:
    from vllm_moe_surgeon.compat.repl import collect
    stats = collect(llm)
    if stats is None:
        counter_error = "collect() returned None"
    else:
        counters = {
            "layers": stats.layers,
            "cpu_execs": stats.cpu_execs,
            "gpu_hits": stats.hits,
            "gpu_misses": stats.misses,
        }
except Exception as exc:  # keep the reason: a silent None is what hid this once
    counter_error = f"{type(exc).__name__}: {exc}"
np.savez_compressed(
    out,
    ids=np.asarray(ids, dtype=np.int32),
    logprobs=np.asarray(lps, dtype=np.float32),
    doc=np.asarray(docs, dtype=np.int32),
    meta=np.frombuffer(json.dumps({}).encode("utf-8"), dtype=np.uint8),
)
print("CAPTURE_JSON" + json.dumps({
    "positions": len(ids), "load_seconds": load, "counters": counters,
    "counter_error": counter_error,
}))
"""


def capture_arm(
    model: str,
    corpus_path: str,
    out_path: str,
    *,
    arm: str,
    llm_kwargs: dict[str, Any] | None = None,
    top_k: int = DEFAULT_TOP_K,
    timeout: float = 3600.0,
) -> Capture:
    """Boot one arm in a fresh process and write its capture. Raises on failure.

    Unlike ``headroom.score_model`` this does *not* swallow a failed arm: a
    fidelity run compares arms against a reference, so a missing arm is a missing
    row in the comparison, not a rankable outcome.
    """
    kwargs = dict(llm_kwargs or {})
    # The engine caps how many logprobs it will return, and the default (20) is
    # below this module's own default K -- a silent truncation that would show up
    # as a spurious substitution rate rather than as an error.
    kwargs["max_logprobs"] = max(int(kwargs.get("max_logprobs", 0)), top_k)

    argv = [
        sys.executable,
        "-c",
        _PROBE,
        model,
        corpus_path,
        out_path,
        json.dumps(kwargs),
        str(top_k),
    ]
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
        _kill_group(proc)
        raise RuntimeError(f"arm {arm!r} timed out after {timeout:.0f}s") from None
    except BaseException:
        # start_new_session detaches the child from our process group, so every
        # unwind path has to take the group down or an interrupted run leaks a
        # booted engine still holding the device.
        _kill_group(proc)
        raise

    completed = subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)
    line = next(
        (ln for ln in completed.stdout.splitlines() if ln.startswith("CAPTURE_JSON")),
        None,
    )
    if completed.returncode != 0 or line is None:
        raise RuntimeError(f"arm {arm!r} failed: {_diagnose(completed)}")

    payload = json.loads(line[len("CAPTURE_JSON") :])
    capture = Capture.load(out_path if out_path.endswith(".npz") else out_path + ".npz")
    counters = payload.get("counters")
    if counters is None and payload.get("counter_error"):
        # Not fatal -- the capture is still valid -- but it must not be silent:
        # without counters an arm whose mechanism never engaged looks identical
        # to one that engaged and changed nothing.
        logger.warning(
            "arm %r: tier counters unavailable (%s); this capture cannot prove "
            "its mechanism ran",
            arm,
            payload["counter_error"],
        )
    capture.meta = {
        "arm": arm,
        "model": model,
        "llm_kwargs": kwargs,
        "top_k": top_k,
        "load_seconds": payload["load_seconds"],
        "counters": counters,
    }
    capture.save(out_path)
    logger.info(
        "captured %s: %d positions, %.1fs load%s",
        arm,
        capture.positions,
        payload["load_seconds"],
        f", {counters['cpu_execs']} host expert forwards" if counters else "",
    )
    # An arm that asked for co-execution and never performed one is measuring
    # the arm without it, and would report "no captured difference" -- which is
    # true and useless. Say so at capture time, not after the table is read.
    if _asked_for_coexec(kwargs) and counters is not None and not counters["cpu_execs"]:
        logger.warning(
            "arm %r requested cpu_experts but the host path performed ZERO "
            "expert forwards during this capture: whatever this arm measures, "
            "it is not co-execution.",
            arm,
        )
    return capture


def _asked_for_coexec(kwargs: dict[str, Any]) -> bool:
    surgeon = (kwargs.get("additional_config") or {}).get("surgeon") or {}
    return bool(surgeon.get("cpu_experts"))


def stage_corpus(prompts: list[str], path: str) -> int:
    """Write the prompts where the children read them; return the byte count.

    Re-exported from :mod:`compat.headroom` so a caller staging a corpus for
    fidelity does not have to know the two commands share a file convention.
    """
    from .headroom import stage_corpus as _stage

    return _stage(prompts, path)
