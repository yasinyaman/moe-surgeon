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

try:  # arrow keys and history for input(); not present on every platform
    import readline  # noqa: F401
except ImportError:
    pass

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
    #: Expert forwards computed on the host by CPU co-execution. Their RAM
    #: reads count as ram_hits/ram_misses (RAM-tier truth) but never as GPU
    #: hits/misses -- those keep meaning "an H2D load happened".
    cpu_execs: int = 0
    cpu_gemm_s: float = 0.0
    layers: int = 0

    def __sub__(self, other: TierStats) -> TierStats:
        return TierStats(
            hits=self.hits - other.hits,
            misses=self.misses - other.misses,
            ram_hits=self.ram_hits - other.ram_hits,
            ram_misses=self.ram_misses - other.ram_misses,
            disk_bytes=self.disk_bytes - other.disk_bytes,
            cpu_execs=self.cpu_execs - other.cpu_execs,
            cpu_gemm_s=self.cpu_gemm_s - other.cpu_gemm_s,
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


def collect(llm: Any) -> TierStats | None:
    """Sum the provider counters across every MoE layer. Never raises.

    ``None`` means the counters could not be read at all -- an engine whose
    internals moved, or a boot the RPC cannot reach. That is a different
    statement from "no layer holds a provider" (an untiered run), and the two
    must not print the same line.
    """

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
                        getattr(provider, "cpu_execs", 0),
                        getattr(provider, "t_cpu_gemm", 0.0),
                    )
                )
        return out

    try:
        rows = llm.collective_rpc(_read)[0]
    except Exception as exc:  # pragma: no cover - engine internals moved
        logger.warning("could not read the tier counters: %s", exc)
        return None
    if not rows:
        return TierStats()
    return TierStats(
        hits=sum(r[0] for r in rows),
        misses=sum(r[1] for r in rows),
        ram_hits=sum(r[2] for r in rows),
        ram_misses=sum(r[3] for r in rows),
        disk_bytes=sum(r[4] for r in rows),
        cpu_execs=sum(r[5] for r in rows),
        cpu_gemm_s=sum(r[6] for r in rows),
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
    if stats.cpu_execs:
        per = stats.cpu_gemm_s / stats.cpu_execs * 1e6
        lines.append(
            f"  CPU co-exec: {stats.cpu_execs} expert forwards on the host "
            f"({stats.cpu_gemm_s * 1e3:.0f} ms GEMM, {per:.0f} us/expert)"
        )

    # The two misconfigurations this package measured, named when the counters
    # show them, because "it feels slow" is not a diagnosis.
    # (Under cpu_experts, CPU-path disk reads count as ram_misses by design --
    # RAM-resident-first selection keeps them rare, but the cold-fill equality
    # below can be off by those reads on a co-exec run.)
    if rate < 0.5:
        lines.append(
            "  ! under half of expert lookups hit the GPU cache. Raising "
            "expert_cache_size toward the batch's per-layer union is the lever "
            "measured at 2.59x; `surgeon autoconfig` sizes it."
        )
    cold_fill = stats.ram_hits == 0 and stats.ram_misses == stats.misses
    if cold_fill and stats.ram_misses:
        # Every RAM miss was also a GPU miss and nothing ever hit the RAM tier:
        # that is the signature of the first fill -- each expert read from the
        # store once -- not of an undersized tier. Observed live: a correctly
        # sized 64/64 config showed 0/730 here on its first turn and the old
        # warning told its user to fix a setting that was already right.
        lines.append(
            "  (all of that is the first fill -- each expert read once. If disk "
            "reads continue in later turns, then raise ram_cache.)"
        )
    elif stats.ram_misses and stats.ram_misses > 0.1 * max(1, ram_total):
        lines.append(
            "  ! evictions are reaching disk. ram_cache below the expert count "
            "was measured at 1.66x slower; raise it if the "
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


def _no_chat_template() -> type[BaseException]:
    """The exception ``llm.chat`` raises when the model has no chat template.

    Imported per vLLM version: where the dedicated class exists it is caught by
    name, and where it does not, vLLM raises a ``ValueError`` for this case --
    which is still far narrower than the bare ``Exception`` this replaced, under
    which a CUDA OOM was "no chat template" and got silently re-run.
    """
    try:
        from vllm.entrypoints.chat_utils import (
            ChatTemplateResolutionError,
        )

        return ChatTemplateResolutionError
    except ImportError:
        return ValueError


def run(
    model: str,
    *,
    surgeon: dict[str, Any] | None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    llm_kwargs: dict[str, Any] | None = None,
) -> int:
    """Boot the engine and read prompts until told to stop.

    Piped-output durability (line-buffered stdout, so vLLM's exit teardown
    cannot eat the session's prints) is handled once in ``cli.main`` -- it
    applies to every subcommand that boots an engine, not just this one.
    """
    import os

    # The counters are read with a collective_rpc *callable*, which cannot cross
    # the default multiprocess engine's serialization boundary -- on a stock boot
    # every read would fail and the REPL's whole point would be silently dead.
    # In-process is the configuration every measurement in this project used.
    # setdefault, so an explicit user override still wins.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm import LLM, SamplingParams

    kwargs = dict(llm_kwargs or {})
    if surgeon:
        kwargs["additional_config"] = {"surgeon": surgeon}

    print(f"loading {model} ...")
    llm = LLM(model=model, **kwargs)
    baseline = collect(llm)
    if baseline is None:
        print("tier status unknown -- could not read the provider counters. "
              "/help for commands.")
        baseline = TierStats()
    elif baseline.layers:
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
            now = collect(llm)
            if now is None:
                print("  tier counters unreadable this session")
            else:
                print(format_session(now - session_start))
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
        chatted = True
        try:
            try:
                outputs = llm.chat(messages, sampling, use_tqdm=False)
            except _no_chat_template():
                # A base model without a chat template: fall back to the raw
                # prompt. ONLY that case -- an engine fault (OOM, a dead engine)
                # must propagate with its real message, not be misdiagnosed as a
                # template problem and silently re-run.
                print(
                    "  (no chat template; sending this prompt raw -- previous "
                    "turns are not included)"
                )
                outputs = llm.generate([line], sampling, use_tqdm=False)
                chatted = False
        except KeyboardInterrupt:
            # Ctrl-C mid-generation. The engine's state after an interrupted
            # forward is not something to keep typing into; end the session with
            # the summary rather than pretend the turn can resume.
            print("\n  interrupted; ending the session")
            messages.pop()
            break
        elapsed = time.perf_counter() - started

        completion = outputs[0].outputs[0]
        text = completion.text.strip()
        print(text)
        if chatted:
            messages.append({"role": "assistant", "content": text})
        else:
            # The raw path transmitted only this line: recording a fictional
            # exchange would make the NEXT chat() turn claim context the model
            # never saw.
            messages.pop()
        after = collect(llm)
        if after is not None and before is not None:
            print(
                format_turn(after - before, len(completion.token_ids), elapsed)
            )
        else:
            print(format_turn(TierStats(), len(completion.token_ids), elapsed))

    final = collect(llm)
    if final is not None:
        print(format_session(final - session_start))
    return 0
