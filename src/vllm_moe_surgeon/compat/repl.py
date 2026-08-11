# SPDX-License-Identifier: Apache-2.0
"""An interactive prompt that shows what the tier is doing while you use it.

``ollama run`` is the shape people expect: name a model, get a prompt, type. The
part worth adding here is not the prompt -- it is that a tiered deployment is
otherwise completely opaque. The whole package is memory bookkeeping happening
underneath a normal-looking model, and the two failure modes that matter are
invisible from the outside: a cache too small for the batch re-fetches experts
every step, and a RAM tier too small turns every eviction into a disk read. Both
show up as "it feels slow", which is not a diagnosis.

So every turn prints what it cost: tokens per second, the GPU cache hit rate, and
how many bytes came off disk. Those come from the provider's own counters, read
per turn and differenced, so the numbers describe *that* turn rather than the
process lifetime -- a lifetime average hides the thing you are trying to see,
which is usually the first few turns being cold and the rest not.

The engine boot is deliberately the same one :mod:`.autoconfig` prints. If this
command chose its own configuration it would be a second implementation of the
sizing rules, and the two would drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._logging import init_logger

logger = init_logger(__name__)

HELP = """\
  /stats     what the tier has done since this session started
  /config    the configuration this session booted with
  /clear     forget the conversation so far
  /help      this
  /bye       quit
"""


@dataclass(frozen=True)
class TierStats:
    """The provider counters, summed across layers."""

    hits: int = 0
    misses: int = 0
    ram_hits: int = 0
    ram_misses: int = 0
    disk_bytes: int = 0
    layers: int = 0

    def __sub__(self, other: TierStats) -> TierStats:
        return TierStats(
            hits=self.hits - other.hits,
            misses=self.misses - other.misses,
            ram_hits=self.ram_hits - other.ram_hits,
            ram_misses=self.ram_misses - other.ram_misses,
            disk_bytes=self.disk_bytes - other.disk_bytes,
            layers=self.layers,
        )

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float | None:
        """``None`` rather than 0.0 when nothing was looked up.

        A turn that never reached the cache and a turn that missed everything are
        opposite situations, and printing 0.0% for both would say the wrong one.
        """
        return self.hits / self.lookups if self.lookups else None


def collect(llm: Any) -> TierStats:
    """Sum the provider counters across every MoE layer. Never raises."""

    def _read(worker):
        out = []
        for module in worker.model_runner.model.modules():
            provider = getattr(module, "_surgeon_provider", None)
            if provider is not None:
                out.append(
                    (
                        getattr(provider, "hits", 0),
                        getattr(provider, "misses", 0),
                        getattr(provider, "ram_hits", 0),
                        getattr(provider, "ram_misses", 0),
                        getattr(provider, "n_disk_bytes", 0),
                    )
                )
        return out

    try:
        rows = llm.collective_rpc(_read)[0]
    except Exception as exc:  # pragma: no cover - engine internals moved
        logger.debug("could not read the tier counters: %s", exc)
        return TierStats()
    if not rows:
        return TierStats()
    return TierStats(
        hits=sum(r[0] for r in rows),
        misses=sum(r[1] for r in rows),
        ram_hits=sum(r[2] for r in rows),
        ram_misses=sum(r[3] for r in rows),
        disk_bytes=sum(r[4] for r in rows),
        layers=len(rows),
    )


def format_turn(stats: TierStats, tokens: int, seconds: float) -> str:
    """The one line printed after a reply."""
    parts = [f"{tokens} tok"]
    if seconds > 0:
        parts.append(f"{tokens / seconds:.1f} tok/s")
    rate = stats.hit_rate
    if rate is not None:
        parts.append(f"cache {100 * rate:.0f}% ({stats.hits}/{stats.lookups})")
    if stats.disk_bytes > 0:
        parts.append(f"disk {stats.disk_bytes / 1024**2:.0f} MiB")
    return "  [" + ", ".join(parts) + "]"


def format_session(stats: TierStats) -> str:
    """The `/stats` block, which is where a misconfiguration becomes visible."""
    lines = [f"  {stats.layers} MoE layers served from the tier"]
    rate = stats.hit_rate
    if rate is None:
        lines.append("  no expert lookups yet")
        return "\n".join(lines)

    lines.append(
        f"  GPU cache : {stats.hits} hits / {stats.misses} misses "
        f"({100 * rate:.1f}% hit rate)"
    )
    ram_total = stats.ram_hits + stats.ram_misses
    if ram_total:
        lines.append(
            f"  RAM tier  : {stats.ram_hits} hits / {stats.ram_misses} misses "
            f"({100 * stats.ram_hits / ram_total:.1f}% hit rate)"
        )
    lines.append(f"  read off disk: {stats.disk_bytes / 1024**2:.0f} MiB")

    # The two misconfigurations this package measured, named when the counters
    # show them, because "it feels slow" is not a diagnosis.
    if rate < 0.5:
        lines.append(
            "  ! under half of expert lookups hit the GPU cache. Raising "
            "expert_cache_size toward the batch's per-layer union is the lever "
            "measured at 2.59x; `surgeon autoconfig` sizes it."
        )
    if stats.ram_misses and stats.ram_misses > 0.1 * max(1, ram_total):
        lines.append(
            "  ! evictions are reaching disk. ram_cache below the expert count "
            "was measured at 1.66x slower, and more variable; raise it if the "
            "host can hold every expert."
        )
    return "\n".join(lines)


def handle_command(line: str) -> str | None:
    """Map a slash line to an action name, or ``None`` if it is not a command."""
    if not line.startswith("/"):
        return None
    name = line[1:].strip().split()[0] if line[1:].strip() else ""
    known = {
        "bye": "quit",
        "exit": "quit",
        "quit": "quit",
        "help": "help",
        "?": "help",
        "stats": "stats",
        "config": "config",
        "clear": "clear",
    }
    return known.get(name, "unknown")


def run(
    model: str,
    *,
    surgeon: dict[str, Any] | None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    llm_kwargs: dict[str, Any] | None = None,
) -> int:
    """Boot the engine and read prompts until told to stop."""
    from vllm import LLM, SamplingParams

    kwargs = dict(llm_kwargs or {})
    if surgeon:
        kwargs["additional_config"] = {"surgeon": surgeon}

    print(f"loading {model} ...")
    llm = LLM(model=model, **kwargs)
    baseline = collect(llm)
    if baseline.layers:
        print(f"tier active on {baseline.layers} MoE layers. /help for commands.")
    else:
        print("tier NOT active -- no layer holds a provider. /help for commands.")

    sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    messages: list[dict[str, str]] = []
    session_start = baseline

    while True:
        try:
            line = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        action = handle_command(line)
        if action == "quit":
            break
        if action == "help":
            print(HELP, end="")
            continue
        if action == "stats":
            print(format_session(collect(llm) - session_start))
            continue
        if action == "config":
            print(f"  {surgeon or 'no tier: serving untiered'}")
            continue
        if action == "clear":
            messages = []
            print("  conversation cleared")
            continue
        if action == "unknown":
            print(f"  unknown command {line!r}. /help for the list.")
            continue

        messages.append({"role": "user", "content": line})
        before = collect(llm)
        import time

        started = time.perf_counter()
        try:
            outputs = llm.chat(messages, sampling)
        except Exception:
            # No chat template, or chat unavailable: fall back to the raw prompt so
            # a base model still works, and say so rather than failing the turn.
            print("  (no chat template; sending the prompt raw)")
            outputs = llm.generate([line], sampling)
        elapsed = time.perf_counter() - started

        completion = outputs[0].outputs[0]
        text = completion.text.strip()
        print(text)
        messages.append({"role": "assistant", "content": text})
        print(
            format_turn(collect(llm) - before, len(completion.token_ids), elapsed)
        )

    print(format_session(collect(llm) - session_start))
    return 0
