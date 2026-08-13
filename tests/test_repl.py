# SPDX-License-Identifier: Apache-2.0
"""The parts of the interactive prompt that do not need an engine.

The loop itself needs a GPU and a model, so what is testable here is everything
that decides what the user is *told*: the per-turn arithmetic, the difference
between "nothing was looked up" and "everything missed", and whether the two
measured misconfigurations get named when the counters show them. Those are the
reason the command exists -- a prompt that printed nothing about the tier would
be a worse `ollama run`.
"""

from __future__ import annotations

from vllm_moe_surgeon.compat.repl import (
    TierStats,
    collect,
    format_session,
    format_turn,
    handle_command,
)


def test_stats_subtract_to_describe_one_turn():
    """Counters are cumulative; a turn is a difference. A lifetime average hides
    exactly what you look at a per-turn number to see -- the cold first turns."""
    before = TierStats(hits=100, misses=50, disk_bytes=1 << 20, layers=16)
    after = TierStats(hits=180, misses=55, disk_bytes=3 << 20, layers=16)
    turn = after - before

    assert turn.hits == 80
    assert turn.misses == 5
    assert turn.disk_bytes == 2 << 20
    assert turn.layers == 16, "layer count is a property of the model, not a delta"


def test_no_lookups_is_not_a_zero_hit_rate():
    """A turn that never reached the cache and a turn that missed everything are
    opposite situations; printing 0% for both would report the wrong one."""
    assert TierStats().hit_rate is None
    assert TierStats(hits=0, misses=4).hit_rate == 0.0
    assert TierStats(hits=3, misses=1).hit_rate == 0.75


def test_the_turn_line_reports_what_the_turn_cost():
    line = format_turn(
        TierStats(hits=90, misses=10, disk_bytes=4 << 20), tokens=128, seconds=2.0
    )
    assert "128 tok" in line
    assert "64.0 tok/s" in line
    assert "cache 90%" in line
    assert "disk 4 MiB" in line


def test_the_turn_line_omits_what_it_cannot_say():
    """No disk read and no lookups mean those clauses are absent rather than zero,
    so the line stays about what actually happened."""
    line = format_turn(TierStats(), tokens=10, seconds=0.0)
    assert "10 tok" in line
    assert "tok/s" not in line, "no elapsed time means no rate"
    assert "cache" not in line
    assert "disk" not in line


def test_a_cold_cache_is_named_not_just_printed():
    """The 2.59x finding, surfaced where someone would otherwise conclude only that
    it feels slow."""
    text = format_session(TierStats(hits=20, misses=80, layers=16))
    assert "20 hits / 80 misses" in text
    assert "expert_cache_size" in text
    assert "2.59x" in text


def test_disk_pressure_is_named_too():
    text = format_session(
        TierStats(hits=90, misses=10, ram_hits=50, ram_misses=50, layers=16)
    )
    assert "ram_cache" in text
    assert "1.66x" in text
    # A healthy GPU cache must not also trigger the capacity warning.
    assert "2.59x" not in text


def test_a_healthy_session_says_nothing_alarming():
    text = format_session(
        TierStats(hits=990, misses=10, ram_hits=1000, ram_misses=0, layers=16)
    )
    assert "hit rate" in text
    assert "!" not in text


def test_a_session_with_no_lookups_says_so():
    text = format_session(TierStats(layers=16))
    assert "no expert lookups yet" in text


def test_commands_are_recognised_and_typos_are_not_prompts():
    assert handle_command("/bye") == "quit"
    assert handle_command("/exit") == "quit"
    assert handle_command("/stats") == "stats"
    assert handle_command("/config") == "config"
    assert handle_command("/clear") == "clear"
    assert handle_command("/help") == "help"
    # A mistyped command must not be sent to the model as a prompt.
    assert handle_command("/statz") == "unknown"
    assert handle_command("/") == "unknown"
    # Ordinary text is not a command, including text that merely mentions one.
    assert handle_command("what does /stats do?") is None
    assert handle_command("hello") is None


def test_unreadable_counters_are_unknown_not_untiered():
    """The counters are a nicety; the conversation is not. But "could not read" and
    "no layer holds a provider" are different statements -- one is a broken probe,
    the other an untiered run -- and returning the same value for both made the
    boot line lie about an active tier."""

    class _Hostile:
        def collective_rpc(self, _fn):
            raise RuntimeError("model_runner moved")

    assert collect(_Hostile()) is None

    class _Untiered:
        def collective_rpc(self, _fn):
            return [[]]

    stats = collect(_Untiered())
    assert stats is not None and stats.layers == 0


def test_collect_reads_the_real_worker_shape():
    """The production closure walks worker.model_runner.model.modules() and reads
    five counter attributes off _surgeon_provider. Every other test replaces
    collective_rpc wholesale, so this is the one that fails if that closure body
    drifts from the provider's actual attribute names."""

    class _Provider:
        hits, misses, ram_hits, ram_misses, n_disk_bytes = 7, 3, 9, 1, 5 << 20

    class _MoEModule:
        _surgeon_provider = _Provider()

    class _Plain:
        pass

    class _Model:
        def modules(self):
            return [_Plain(), _MoEModule(), _Plain(), _MoEModule()]

    class _Runner:
        model = _Model()

    class _Worker:
        model_runner = _Runner()

    class _LLM:
        def collective_rpc(self, fn):
            # Deliver the closure to a worker-shaped object, as the engine would.
            return [fn(_Worker())]

    stats = collect(_LLM())
    assert stats is not None
    assert stats.layers == 2
    assert stats.hits == 14 and stats.misses == 6
    assert stats.ram_hits == 18 and stats.ram_misses == 2
    assert stats.disk_bytes == 10 << 20


def test_the_first_fill_is_not_misdiagnosed_as_an_undersized_ram_tier():
    """Observed live on a correctly sized 64/64 config: first turn showed RAM
    0 hits / 730 misses -- every expert read from the store once -- and the old
    warning told the user to raise a setting that was already right. The
    signature of cold fill is ram_hits == 0 with ram_misses equal to the GPU
    misses; that gets an informational line, not advice to resize."""
    cold = format_session(
        TierStats(hits=2793, misses=730, ram_hits=0, ram_misses=730, layers=16)
    )
    assert "first fill" in cold
    assert "! evictions" not in cold

    # Genuine eviction pressure -- RAM hits mixed with ongoing misses -- still warns.
    warm = format_session(
        TierStats(hits=900, misses=100, ram_hits=50, ram_misses=50, layers=16)
    )
    assert "! evictions" in warm


def test_tier_stats_carries_the_cpu_coexec_counters():
    from vllm_moe_surgeon.compat.repl import TierStats, format_session

    a = TierStats(hits=10, misses=2, cpu_execs=8, cpu_gemm_s=0.004, layers=16)
    b = TierStats(hits=4, misses=1, cpu_execs=3, cpu_gemm_s=0.001, layers=16)
    d = a - b
    assert d.cpu_execs == 5
    assert abs(d.cpu_gemm_s - 0.003) < 1e-9

    text = format_session(a)
    assert "CPU co-exec: 8 expert forwards" in text
    # ... and the line stays out of a run that never used the host path.
    assert "CPU co-exec" not in format_session(
        TierStats(hits=10, misses=2, layers=16)
    )
