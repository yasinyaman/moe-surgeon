# SPDX-License-Identifier: Apache-2.0
"""Measure the gains on three axes at once: feasibility, speed, accuracy.

Each axis alone is misleading. Perplexity says the disk tier is nearly free, but
says nothing about the tokens per second it costs. Throughput says the tier is
slower, but says nothing about the fact that without it the model does not fit at
all. And a memory figure says nothing about either. So they are measured together,
per configuration, and reported side by side.

**feasibility** load time, plus a peak-VRAM reading that must be read with care.
    On a unified-memory device ``max_memory_allocated`` largely reports the
    preallocated ``gpu_memory_utilization`` pool rather than what the model needs,
    so it does **not** answer "how small a device could this run on". Use
    :mod:`..surgery.budget` for that -- its arithmetic is validated against a real
    store. Load time, by contrast, is a genuine and useful signal: the tier halves
    it, because expert weights stream to the store instead of being staged onto the
    device.
**speed** prefill and decode tokens per second
**accuracy** held-out perplexity, the same metric the quality gate uses

The comparison that matters for this project is the fourth arm: surgery *plus* tier.
Deletion shrinks the store and shrinks the candidate set the cache chooses residency
among, so at a fixed capacity a pruned model should hit more often than a full one.
That is the claim that surgery supports the tier rather than competing with it, and
it is either in the numbers or it is not.

Every arm runs the same prompts, the same token budget and the same engine
settings; only the configuration under test changes. Settings are recorded in the
result so a number can never be quoted without them.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .._logging import init_logger

logger = init_logger(__name__)


def peak_vram_bytes(worker: Any) -> int:
    """Peak allocated device memory in the worker, for the feasibility axis."""
    import torch

    return int(torch.cuda.max_memory_allocated())


def reset_vram_peak(worker: Any) -> None:
    import torch

    torch.cuda.reset_peak_memory_stats()


@dataclass
class BenchResult:
    name: str
    #: feasibility
    load_seconds: float = 0.0
    peak_vram_gib: float = 0.0
    #: speed
    prefill_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_tokens: int = 0
    decode_seconds: float = 0.0
    #: accuracy
    perplexity: float = float("nan")
    #: what produced these numbers
    settings: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def decode_tok_s(self) -> float:
        return self.decode_tokens / self.decode_seconds if self.decode_seconds else 0.0

    @property
    def prompt_share(self) -> float:
        """Prompt tokens as a fraction of all tokens processed.

        The contamination in ``decode_tok_s``: the timed run includes prefill, so a
        large share here means the decode figure is diluted. Keep the workload
        short-prompt/long-generation and this stays small.
        """
        total = self.prefill_tokens + self.decode_tokens
        return self.prefill_tokens / total if total else 0.0


def _generation_timing(llm: Any, prompts: Sequence[str], max_tokens: int):
    """Decode throughput, measured so prefill cannot contaminate it.

    An earlier version timed a full run and subtracted a prefill-only run. That is
    wrong twice over: the second call benefits from the first one's warmup, so the
    difference can go negative and produce an absurd rate, and any residual
    scheduling difference lands entirely in the decode figure.

    Instead the workload is chosen so prefill is negligible -- short prompts, long
    generations -- and one warmup run precedes the timed one. Decode rate is then
    output tokens over wall time, with the prefill share reported alongside so the
    contamination is visible rather than assumed away.
    """
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)

    # Warm up compilation, graphs and any cache the tier maintains, so the timed
    # run measures steady state rather than first-call costs.
    llm.generate(list(prompts), SamplingParams(max_tokens=8, temperature=0.0))

    start = time.perf_counter()
    outputs = llm.generate(list(prompts), params)
    elapsed = time.perf_counter() - start

    decode_tokens = sum(len(c.token_ids) for o in outputs for c in o.outputs)
    prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    return {
        "prefill_tokens": prompt_tokens,
        "prefill_seconds": 0.0,
        "decode_tokens": decode_tokens,
        "decode_seconds": max(elapsed, 1e-6),
    }


def run_arm(
    name: str,
    model: str,
    prompts: Sequence[str],
    *,
    max_tokens: int = 64,
    env: dict[str, str] | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    accuracy_prompts: Sequence[str] | None = None,
) -> BenchResult:
    """One configuration, all three axes.

    Run one arm per process. A vLLM engine does not release its device memory when
    the Python object falls out of scope, so a second arm in the same process
    starts against the first one's allocation and simply fails to boot -- which is
    how an earlier attempt at this lost two of four arms.

    ``env`` is applied around the engine's construction, which is where the disk
    tier reads its settings. It is restored afterwards so arms cannot leak into
    each other -- the failure that would make every later arm inherit the previous
    one's configuration.
    """
    from vllm import LLM

    from .ablation import measure_nll

    settings = dict(llm_kwargs or {})
    settings["env"] = dict(env or {})
    result = BenchResult(name=name, settings=settings)

    previous = {key: os.environ.get(key) for key in (env or {})}
    for key, value in (env or {}).items():
        os.environ[key] = value
    try:
        start = time.perf_counter()
        llm = LLM(model=model, **(llm_kwargs or {}))
        result.load_seconds = time.perf_counter() - start

        try:
            llm.collective_rpc(reset_vram_peak)
        except Exception as exc:  # pragma: no cover - platform dependent
            logger.warning("could not reset the VRAM peak: %s", exc)

        timing = _generation_timing(llm, prompts, max_tokens)
        for key, value in timing.items():
            setattr(result, key, value)

        try:
            peak = llm.collective_rpc(peak_vram_bytes)[0]
            result.peak_vram_gib = peak / 1024**3
        except Exception as exc:  # pragma: no cover
            logger.warning("could not read the VRAM peak: %s", exc)

        nll, tokens = measure_nll(llm, accuracy_prompts or prompts)
        import math

        result.perplexity = math.exp(nll / tokens)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        logger.warning("arm %s failed: %s", name, result.error)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return result


def report(results: Sequence[BenchResult], baseline: str | None = None) -> str:
    """Three axes side by side, with ratios against a named baseline."""
    base = next((r for r in results if r.name == baseline), None)

    lines = [
        "                              feasibility          speed        accuracy",
        f"  {'arm':<26} {'load s':>7} {'VRAM':>8} "
        f"{'prompt%':>8} {'decode':>10} {'ppl':>8}",
    ]
    for r in results:
        if r.error:
            lines.append(f"  {r.name:<26} FAILED  {r.error[:44]}")
            continue
        lines.append(
            f"  {r.name:<26} {r.load_seconds:>7.1f} {r.peak_vram_gib:>7.2f}G "
            f"{100 * r.prompt_share:>7.1f}% {r.decode_tok_s:>9.1f}/s "
            f"{r.perplexity:>8.3f}"
        )

    if base and not base.error:
        lines += ["", f"  relative to {base.name}:"]
        for r in results:
            if r.error or r is base:
                continue
            lines.append(
                f"  {r.name:<26} "
                f"VRAM {r.peak_vram_gib / base.peak_vram_gib:>5.2f}x  "
                f"decode {r.decode_tok_s / base.decode_tok_s:>5.2f}x  "
                f"ppl {r.perplexity / base.perplexity:>5.3f}x"
            )
    return "\n".join(lines)


def to_dict(result: BenchResult) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(result)


def from_dict(payload: dict[str, Any]) -> BenchResult:
    return BenchResult(**payload)
