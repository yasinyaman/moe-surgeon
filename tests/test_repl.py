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


def test_collecting_stats_never_takes_the_session_down():
    """The counters are a nicety; the conversation is not. An engine whose internals
    moved must cost the stats line, not the turn."""

    class _Hostile:
        def collective_rpc(self, _fn):
            raise RuntimeError("model_runner moved")

    assert collect(_Hostile()) == TierStats()

    class _Untiered:
        def collective_rpc(self, _fn):
            return [[]]

    assert collect(_Untiered()).layers == 0
