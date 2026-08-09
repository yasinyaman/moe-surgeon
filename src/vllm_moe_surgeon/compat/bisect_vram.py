# SPDX-License-Identifier: Apache-2.0
"""Find the smallest memory budget a configuration actually boots and serves in.

The feasibility axis needs a number that means "this fits". Two cheaper things were
tried first and neither means that:

``max_memory_allocated`` measures the *preallocated* pool. vLLM claims
``gpu_memory_utilization`` of the device up front and allocates the KV cache inside
it, so on a unified-memory box the reading tracks the fraction it was told to take,
not what the model needs. Asking for 35% of 128 GiB reports ~45 GiB whether the model
is 7 B or 3 B.

``surgeon budget`` adds up header bytes. That arithmetic is right about weights and
silent about everything else -- activation peaks, the graph pool, fragmentation, the
loader's transient copies. It gives a lower bound, not a floor.

So this measures the floor the only way it exists: boot the engine at a budget,
generate a token, and see whether it survives. A configuration "fits" at *b* when
that succeeds, and the smallest such *b* is the answer. That makes the number
directly comparable across arms -- baseline, pruned, tiered -- because every arm is
being asked the same question.

Two properties of the search matter:

**One process per attempt.** An engine does not release device memory when its Python
object goes out of scope, so a failed attempt would poison every later one in the same
process. This is the same constraint that invalidated the first benchmark harness.

**A failure to boot is data, not an error.** Out-of-memory at a low budget is the
expected answer, and the search has to tell it apart from a broken configuration --
a model that cannot boot at *any* budget is a bug, not a floor. So the top of the
range is probed first: if the largest budget fails, the search reports that rather
than bisecting its way down to a meaningless zero.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from .._logging import init_logger

logger = init_logger(__name__)

#: Deliberately coarse. The quantity is "how much memory does this need", and
#: resolving it past a percent of the device costs a boot per bit for a precision the
#: answer does not have -- the floor moves with batch shape and fragmentation.
DEFAULT_TOLERANCE = 0.01


@dataclass
class Attempt:
    fraction: float
    booted: bool
    seconds: float = 0.0
    error: str | None = None

    @property
    def summary(self) -> str:
        mark = "ok" if self.booted else "FAIL"
        return f"{self.fraction:.3f} {mark} ({self.seconds:.1f}s)"


@dataclass
class BisectResult:
    """The floor, plus every probe that established it."""

    name: str
    model: str
    #: Smallest fraction that booted and generated, or ``None`` if none did.
    floor_fraction: float | None
    #: Largest fraction that failed, so the bracket is visible.
    highest_failure: float | None
    device_total_gib: float
    attempts: list[Attempt] = field(default_factory=list)
    tolerance: float = DEFAULT_TOLERANCE
    #: The kwargs every probe was booted with. Recorded because a result without them
    #: is not interpretable: three arms were once written up as "tiered, ram_cache 0"
    #: when that setting had turned the disk tier off entirely, and nothing in the
    #: saved result could have contradicted the label.
    llm_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def floor_gib(self) -> float | None:
        if self.floor_fraction is None:
            return None
        return self.floor_fraction * self.device_total_gib

    @property
    def bracketed(self) -> bool:
        """Whether a failure was ever observed.

        Without one the search never found a budget too small, so the reported
        number is the smallest *probed* success -- an upper bound on the floor, not
        the floor. Reporting it as a floor would understate what the model can do
        and, worse, make two arms look equal when the search simply never pushed
        either of them hard enough to separate them.
        """
        return self.highest_failure is not None

    def report(self) -> str:
        lines = [f"{self.name}: {self.model}"]
        if self.floor_fraction is None:
            lines.append(
                "  did not boot at any probed budget -- this is a broken "
                "configuration, not a measured floor"
            )
        elif self.bracketed:
            lines.append(
                f"  floor {self.floor_fraction:.3f} of device "
                f"= {self.floor_gib:.2f} GiB (+/- "
                f"{self.tolerance * self.device_total_gib:.2f} GiB)"
            )
            lines.append(
                f"  bracketed: failed at {self.highest_failure:.3f}, "
                f"booted at {self.floor_fraction:.3f}"
            )
        else:
            lines.append(
                f"  floor <= {self.floor_fraction:.3f} of device "
                f"= {self.floor_gib:.2f} GiB -- UNBRACKETED"
            )
            lowest = min(a.fraction for a in self.attempts)
            lines.append(
                "  nothing failed, so the floor is below the probed range and this "
                f"is an upper bound. Re-run with --low below {lowest:.3f}"
            )
        lines.append(f"  {len(self.attempts)} boots: ")
        lines.append("    " + "  ".join(a.summary for a in self.attempts))
        if self.llm_kwargs:
            # Printed, not just stored: an arm's label is a claim about its
            # configuration, and the configuration is the only thing that can check it.
            booted = json.dumps(self.llm_kwargs, sort_keys=True)
            lines.append(f"  booted with: {booted}")
        return "\n".join(lines)


def device_total_gib() -> float:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device; the VRAM floor cannot be measured")
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


# ------------------------------------------------------------------ one attempt

_PROBE = """
import json, sys, time
model, fraction, extra = sys.argv[1], float(sys.argv[2]), json.loads(sys.argv[3])
start = time.perf_counter()
from vllm import LLM, SamplingParams
llm = LLM(model=model, gpu_memory_utilization=fraction, **extra)
# Booting is not enough: the KV cache is sized inside the same budget, so an engine
# can come up and then fail on the first token. Generating one is part of "fits".
out = llm.generate(["probe"], SamplingParams(max_tokens=4, temperature=0.0))
assert out and out[0].outputs[0].text is not None
print("PROBE_OK", time.perf_counter() - start)
"""


def probe(
    model: str,
    fraction: float,
    *,
    llm_kwargs: dict[str, Any] | None = None,
    timeout: float = 900.0,
    env: dict[str, str] | None = None,
) -> Attempt:
    """Boot ``model`` at ``fraction`` in a fresh process and generate a token."""
    import time

    argv = [
        sys.executable,
        "-c",
        _PROBE,
        model,
        str(fraction),
        json.dumps(llm_kwargs or {}),
    ]
    started = time.perf_counter()
    child_env = dict(os.environ)
    child_env.update(env or {})
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=child_env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Attempt(fraction, False, time.perf_counter() - started, "timeout")

    seconds = time.perf_counter() - started
    if completed.returncode == 0 and "PROBE_OK" in completed.stdout:
        return Attempt(fraction, True, seconds)

    return Attempt(
        fraction, False, seconds, error=_diagnose(completed, fraction)
    )


#: Lines that explain *why* a boot failed, most specific first. Needed because the
#: last line of stderr is usually not the reason: a failed vLLM boot signs off with
#: NCCL's "destroy_process_group() was not called" warning, so taking the tail
#: reported that warning as the cause of every single failure.
_ERROR_MARKERS = (
    "No available memory for the cache blocks",
    "out of memory",
    "OutOfMemoryError",
    "Free memory on device",
    "ValueError:",
    "RuntimeError:",
    "Error:",
)


def _diagnose(completed: Any, fraction: float) -> str:
    haystack = f"{completed.stderr or ''}\n{completed.stdout or ''}"
    lines = [ln.strip() for ln in haystack.splitlines() if ln.strip()]
    for marker in _ERROR_MARKERS:
        for line in reversed(lines):
            if marker in line:
                return line[:300]
    return (lines[-1][:300] if lines else "") or f"exit {completed.returncode}"


# -------------------------------------------------------------------- the search


def bisect_floor(
    model: str,
    *,
    name: str | None = None,
    low: float = 0.05,
    high: float = 0.90,
    tolerance: float = DEFAULT_TOLERANCE,
    llm_kwargs: dict[str, Any] | None = None,
    timeout: float = 900.0,
    env: dict[str, str] | None = None,
) -> BisectResult:
    """Smallest ``gpu_memory_utilization`` that boots and serves.

    Probes ``high`` first. A configuration that cannot boot with the largest budget
    on offer is broken, and bisecting it would return a floor of ``low`` -- a number
    that looks like a great result and means the opposite.
    """
    if not 0.0 < low < high <= 1.0:
        raise ValueError(f"need 0 < low < high <= 1, got low={low} high={high}")

    result = BisectResult(
        name=name or model,
        model=model,
        floor_fraction=None,
        highest_failure=None,
        device_total_gib=device_total_gib(),
        tolerance=tolerance,
        llm_kwargs=dict(llm_kwargs or {}),
    )

    ceiling = probe(model, high, llm_kwargs=llm_kwargs, timeout=timeout, env=env)
    result.attempts.append(ceiling)
    logger.info("%s @ %.3f -> %s", result.name, high, ceiling.summary)
    if not ceiling.booted:
        result.highest_failure = high
        logger.warning(
            "%s did not boot at %.3f (%s); not bisecting",
            result.name,
            high,
            ceiling.error,
        )
        return result

    result.floor_fraction = high
    lo, hi = low, high
    while hi - lo > tolerance:
        mid = (lo + hi) / 2
        attempt = probe(model, mid, llm_kwargs=llm_kwargs, timeout=timeout, env=env)
        result.attempts.append(attempt)
        logger.info("%s @ %.3f -> %s", result.name, mid, attempt.summary)
        if attempt.booted:
            hi = mid
            result.floor_fraction = mid
        else:
            lo = mid
            if result.highest_failure is None or mid > result.highest_failure:
                result.highest_failure = mid
    return result


def to_dict(result: BisectResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["floor_gib"] = result.floor_gib
    return payload


__all__ = [
    "Attempt",
    "BisectResult",
    "bisect_floor",
    "device_total_gib",
    "probe",
    "to_dict",
]
