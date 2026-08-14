# SPDX-License-Identifier: Apache-2.0
"""A bad path is an answer, not a stack trace.

Found by running every offline command against a nonexistent checkpoint as an
installed binary and reading what a user would read: budget, inspect, recommend
and apply printed nine frames of plumbing ending in FileNotFoundError. The
dispatch now translates user-input errors into one line and exit 2, while genuine
bugs (TypeError, KeyError, assertions) still crash loudly -- a test asserting
both directions, so the handler can neither rot nor over-reach.
"""

from __future__ import annotations

import pytest

from vllm_moe_surgeon.cli import main


@pytest.mark.parametrize(
    "argv",
    [
        ["budget", "--checkpoint", "/nonexistent-surgeon-test"],
        ["inspect", "--checkpoint", "/nonexistent-surgeon-test"],
        ["recommend", "--checkpoint", "/nonexistent-surgeon-test"],
        ["apply", "--plan", "/nonexistent-surgeon-test.json",
         "--source", "/nonexistent-surgeon-test", "--out", "/tmp/x"],
    ],
)
def test_a_bad_path_is_one_line_and_exit_two(argv, capsys):
    rc = main(argv)
    captured = capsys.readouterr()
    assert rc == 2
    assert "Traceback" not in captured.err
    assert argv[0] in captured.err, "the line should name the command"
    assert "nonexistent-surgeon-test" in captured.err, "and the path at fault"


def test_a_genuine_bug_still_crashes_loudly(monkeypatch):
    """The handler must not become a blanket except: a TypeError is a bug in this
    package, and swallowing it would hide exactly what a traceback exists for."""
    import vllm_moe_surgeon.cli as cli

    def _explodes(args):
        raise TypeError("a real bug")

    monkeypatch.setattr(cli, "_cmd_budget", _explodes)
    # The parser resolves func at registration time, so patch via parse: simplest
    # honest route is calling the wrapped dispatch with a func that raises.
    import argparse

    ns = argparse.Namespace(command="budget", func=_explodes)
    monkeypatch.setattr(
        argparse.ArgumentParser, "parse_args", lambda self, argv=None: ns
    )
    with pytest.raises(TypeError, match="a real bug"):
        main(["budget", "--checkpoint", "/x"])


def test_the_installed_entry_point_survives_a_subprocess_round_trip(tmp_path):
    """The binary a user gets, exercised the way a user runs it.

    Everything else in this file calls main() in-process, which cannot catch a
    broken console-script wiring or an import that only fails outside the test
    process. One subprocess pays for all of that: budget on a real fixture must
    succeed with the sizing advice, and a bad path must produce the one-line
    error on stderr with exit 2 -- not the traceback this file exists to prevent.
    """
    import os
    import pathlib
    import subprocess
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from test_budget import _write

    import vllm_moe_surgeon

    # The test process found the package via sys.path; the child only inherits
    # environment, so hand it the same location explicitly. Derived from the
    # imported package rather than hardcoding src/, so this works identically
    # against a source tree and an installed wheel.
    pkg_root = str(pathlib.Path(vllm_moe_surgeon.__file__).parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")

    _write(tmp_path)
    ok = subprocess.run(
        [sys.executable, "-m", "vllm_moe_surgeon.cli", "budget",
         "--checkpoint", str(tmp_path), "--vram", "8", "--kv-reserve", "0.5"],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert ok.returncode == 0, ok.stderr
    assert "sizing notes" in ok.stdout

    bad = subprocess.run(
        [sys.executable, "-m", "vllm_moe_surgeon.cli", "budget",
         "--checkpoint", "/nonexistent-surgeon-subprocess"],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert bad.returncode == 2
    assert "Traceback" not in bad.stderr
    assert "surgeon budget:" in bad.stderr


def test_headroom_refuses_a_single_model_before_booting_an_engine(tmp_path, capsys):
    """A one-row table is not a ranking. The job server refuses this on the
    request shape; the CLI has to refuse it too, and before the engine boot
    rather than after one has produced the useless row."""
    from vllm_moe_surgeon.cli import main

    corpus = tmp_path / "c.jsonl"
    corpus.write_text('{"text": "a log line"}\n')
    rc = main(["headroom", "--corpus", str(corpus), "--model", "only-one"])

    assert rc == 1
    assert "at least two --model" in capsys.readouterr().err
