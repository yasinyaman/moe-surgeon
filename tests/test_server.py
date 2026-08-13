# SPDX-License-Identifier: Apache-2.0
"""The job server, tested without a GPU, a model or a web request.

Stages are real subprocesses, but pointed at trivial commands instead of the CLI, so
the orchestration is exercised end to end while the tests stay fast. What matters here
is the behaviour around failure: a pipeline must stop at the first bad stage, and a job
record must never advertise an artifact that is not on disk.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from vllm_moe_surgeon.server.jobs import (
    STAGE_ORDER,
    JobState,
    JobStore,
    Runner,
    StageResult,
    build_stages,
    collect_artifacts,
)


def _ok(name="profile"):
    return StageResult(name=name, argv=[sys.executable, "-c", "print('fine')"])


def _fail(name="plan", code=3):
    return StageResult(
        name=name, argv=[sys.executable, "-c", f"import sys; sys.exit({code})"]
    )


def _spec(**over):
    # `checkpoint` points at an existing directory because the tier stage resolves one;
    # these tests are about orchestration, and resolution has its own tests below.
    base = {
        "model": "test/model",
        "checkpoint": os.path.dirname(os.path.abspath(__file__)),
        "corpus": "/tmp/corpus.jsonl",
        "heldout": "/tmp/heldout.jsonl",
        "core_experts": 24,
        # Only the headroom stage reads these; every other stage ignores them.
        "candidates": ["small/candidate"],
    }
    base.update(over)
    return base


# ------------------------------------------------------------------ stage build


def test_stages_run_in_dependency_order_regardless_of_request_order():
    """plan needs profile and gate needs plan, so the request's order is ignored."""
    stages = build_stages(_spec(stages=["gate", "profile", "plan"]), "/w")
    assert [s.name for s in stages] == ["profile", "plan", "gate"]


def test_default_pipeline_ends_at_tier_not_apply():
    """Deletion is opt-in; the default pipeline must not write a checkpoint."""
    names = [s.name for s in build_stages(_spec(), "/w")]
    assert names == ["profile", "plan", "gate", "tier"]
    assert "apply" not in names


def test_every_stage_invokes_the_cli_with_no_shell():
    for stage in build_stages(_spec(stages=list(STAGE_ORDER)), "/w"):
        assert stage.argv[0] == sys.executable
        assert stage.argv[1:3] == ["-m", "vllm_moe_surgeon.cli"]
        assert stage.argv[3] == stage.name


def test_stage_paths_are_inside_the_workdir():
    stages = {s.name: s.argv for s in build_stages(_spec(), "/w/job1")}
    assert "/w/job1/profile.npz" in stages["profile"]
    assert "/w/job1/plan.json" in stages["plan"]


def test_llm_kwargs_reach_the_stages_that_boot_an_engine():
    stages = {s.name: s.argv for s in build_stages(
        _spec(llm_kwargs={"gpu_memory_utilization": 0.3}), "/w"
    )}
    assert "--llm-kwargs" in stages["profile"]
    assert "--llm-kwargs" in stages["gate"]
    # tier and plan never boot an engine, so passing engine kwargs would be noise.
    assert "--llm-kwargs" not in stages["tier"]


# ------------------------------------------------------------ request validation


def test_unrunnable_requests_are_refused_before_anything_starts():
    """A profiling stage costs minutes, so the cheap checks come first."""
    with pytest.raises(ValueError, match="needs a model"):
        build_stages({"corpus": "c", "core_experts": 4}, "/w")
    with pytest.raises(ValueError, match="needs a corpus"):
        build_stages({"model": "m", "core_experts": 4}, "/w")
    with pytest.raises(ValueError, match="needs a held-out corpus"):
        build_stages(_spec(heldout=None), "/w")
    with pytest.raises(ValueError, match="needs core_experts"):
        build_stages(_spec(core_experts=None), "/w")


def test_unknown_stage_names_are_rejected():
    with pytest.raises(ValueError, match="unknown stages"):
        build_stages(_spec(stages=["profile", "polish"]), "/w")


def test_skipping_profile_requires_one_to_already_exist(tmp_path):
    spec = _spec(stages=["plan", "gate", "tier"])
    with pytest.raises(ValueError, match="need a profile"):
        build_stages(spec, str(tmp_path))

    (tmp_path / "profile.npz").write_bytes(b"")
    stages = build_stages(spec, str(tmp_path))
    assert [s.name for s in stages] == ["plan", "gate", "tier"]


# ----------------------------------------------------------------- the runner


def test_a_successful_pipeline_records_every_stage(tmp_path):
    store = JobStore(str(tmp_path / "jobs"))
    job = store.create(_spec(), str(tmp_path / "w"), [_ok("profile"), _ok("plan")])
    done = Runner(store).run(job)

    assert done.state == JobState.succeeded.value
    assert [s.state for s in done.stages] == [JobState.succeeded.value] * 2
    assert all(s.returncode == 0 for s in done.stages)
    assert all("fine" in s.tail for s in done.stages)
    assert done.finished is not None


def test_a_failing_stage_stops_the_pipeline(tmp_path):
    """A plan built on a failed profile is worse than none: nobody knows it is bad."""
    store = JobStore(str(tmp_path / "jobs"))
    job = store.create(
        _spec(), str(tmp_path / "w"), [_ok("profile"), _fail("plan"), _ok("tier")]
    )
    done = Runner(store).run(job)

    assert done.state == JobState.failed.value
    assert done.stages[0].state == JobState.succeeded.value
    assert done.stages[1].state == JobState.failed.value
    assert done.stages[1].returncode == 3
    # The stage after the failure must never have run.
    assert done.stages[2].state == JobState.pending.value
    assert done.stages[2].returncode is None


def test_a_gate_failure_is_labelled_a_verdict_not_a_crash(tmp_path):
    """`gate` exits non-zero on a quality failure, which is an answer, not a fault."""
    store = JobStore(str(tmp_path / "jobs"))
    job = store.create(_spec(), str(tmp_path / "w"), [_fail("gate", code=1)])
    done = Runner(store).run(job)
    assert done.state == JobState.failed.value
    assert "verdict" in done.error


def test_a_stage_that_cannot_start_fails_the_job_not_the_server(tmp_path):
    store = JobStore(str(tmp_path / "jobs"))
    job = store.create(
        _spec(), str(tmp_path / "w"),
        [StageResult(name="profile", argv=["/nonexistent/binary"])],
    )
    done = Runner(store).run(job)
    assert done.state == JobState.failed.value
    assert "could not start" in done.error


def test_a_timeout_kills_the_stage_and_says_so(tmp_path):
    store = JobStore(str(tmp_path / "jobs"))
    job = store.create(
        _spec(), str(tmp_path / "w"),
        [StageResult(
            name="profile", argv=[sys.executable, "-c", "import time; time.sleep(30)"]
        )],
    )
    done = Runner(store, timeout=1.0).run(job)
    assert done.state == JobState.failed.value
    assert "exceeded" in done.error


def test_cancelling_before_a_stage_starts_marks_the_job_cancelled(tmp_path):
    store = JobStore(str(tmp_path / "jobs"))
    job = store.create(_spec(), str(tmp_path / "w"), [_ok("profile")])
    runner = Runner(store)
    runner.cancel(job.id)
    done = runner.run(job)
    assert done.state == JobState.cancelled.value
    assert done.stages[0].state == JobState.cancelled.value


def test_stage_output_is_captured_to_a_file_and_the_record(tmp_path):
    store = JobStore(str(tmp_path / "jobs"))
    job = store.create(
        _spec(), str(tmp_path / "w"),
        [StageResult(
            name="profile",
            argv=[
                sys.executable,
                "-c",
                "import sys; print('to stderr', file=sys.stderr)",
            ],
        )],
    )
    done = Runner(store).run(job)
    assert os.path.exists(done.stages[0].log_path)
    # stderr is merged, so a traceback lands in the same place as normal output.
    assert "to stderr" in done.stages[0].tail


# ------------------------------------------------------------------ artifacts


def test_artifacts_are_reported_only_when_they_exist(tmp_path):
    workdir = tmp_path / "w"
    workdir.mkdir()
    store = JobStore(str(tmp_path / "jobs"))
    job = store.create(_spec(), str(workdir), [_ok()])

    assert collect_artifacts(job) == {}
    (workdir / "profile.npz").write_bytes(b"x")
    (workdir / "store").mkdir()
    found = collect_artifacts(job)
    assert set(found) == {"profile", "store"}
    assert "plan" not in found, "a missing artifact must not be advertised"


def test_a_successful_job_lists_what_it_produced(tmp_path):
    workdir = tmp_path / "w"
    workdir.mkdir()
    (workdir / "plan.json").write_text("{}")
    store = JobStore(str(tmp_path / "jobs"))
    job = store.create(_spec(), str(workdir), [_ok()])
    done = Runner(store).run(job)
    assert "plan" in done.artifacts


# --------------------------------------------------------------- persistence


def test_jobs_survive_a_restart(tmp_path):
    store = JobStore(str(tmp_path / "jobs"))
    job = store.create(_spec(), str(tmp_path / "w"), [_ok()])
    Runner(store).run(job)

    reopened = JobStore(str(tmp_path / "jobs"))
    back = reopened.get(job.id)
    assert back is not None
    assert back.state == JobState.succeeded.value
    assert back.stages[0].name == "profile"
    assert [j.id for j in reopened.list()] == [job.id]


def test_an_unreadable_job_file_does_not_blank_the_listing(tmp_path):
    store = JobStore(str(tmp_path / "jobs"))
    good = store.create(_spec(), str(tmp_path / "w"), [_ok()])
    (tmp_path / "jobs" / "corrupt.json").write_text("{ not json")
    assert [j.id for j in store.list()] == [good.id]


def test_get_returns_none_for_an_unknown_job(tmp_path):
    assert JobStore(str(tmp_path)).get("nope") is None


def test_saves_are_atomic_so_a_reader_never_sees_half_a_file(tmp_path):
    store = JobStore(str(tmp_path / "jobs"))
    job = store.create(_spec(), str(tmp_path / "w"), [_ok()])
    for name in os.listdir(tmp_path / "jobs"):
        assert not name.endswith(".tmp"), "a temp file was left behind"
    with open(tmp_path / "jobs" / f"{job.id}.json") as f:
        json.load(f)  # parses, so it was written whole


# ------------------------------------------------------------------- the app


def test_the_http_layer_is_optional():
    """The offline half must import with no web dependency installed."""
    import vllm_moe_surgeon.server as pkg

    assert hasattr(pkg, "build_stages")


@pytest.mark.parametrize("bad", [{}, {"model": "m"}, {"model": "m", "corpus": "c"}])
def test_the_api_rejects_an_unrunnable_request_with_400(tmp_path, bad):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from vllm_moe_surgeon.server.app import create_app

    with TestClient(create_app(str(tmp_path))) as client:
        assert client.post("/jobs", json=bad).status_code in (400, 422)


def test_the_api_runs_a_pipeline_and_reports_it(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from vllm_moe_surgeon.server.app import create_app

    app = create_app(str(tmp_path))
    with TestClient(app) as client:
        assert client.get("/health").json()["ok"] is True

        # Only the recommend stage, and pointed at a checkpoint that does not exist,
        # so the CLI runs for real and fails for a real reason -- exercising the
        # failure path without needing a model.
        (tmp_path / "profile.npz").write_bytes(b"")
        response = client.post(
            "/jobs",
            json={
                "model": "test/model",
                "checkpoint": str(tmp_path / "missing"),
                "stages": ["recommend"],
            },
        )
        assert response.status_code == 202
        job_id = response.json()["id"]

        app.state.jobs._queue.join()

        job = client.get(f"/jobs/{job_id}").json()
        assert job["state"] == "failed", job
        assert job["stages"][0]["name"] == "recommend"
        assert job["stages"][0]["returncode"] not in (0, None)

        log = client.get(f"/jobs/{job_id}/log/recommend").json()
        assert log["stage"] == "recommend"
        assert log["log"], "the failure reason must be retrievable"

        assert any(j["id"] == job_id for j in client.get("/jobs").json())
        assert client.get("/jobs/nosuchjob").status_code == 404
        assert client.get(f"/jobs/{job_id}/log/tier").status_code == 404
        # Already terminal, so cancelling is a conflict rather than a no-op.
        assert client.post(f"/jobs/{job_id}/cancel").status_code == 409


def test_weight_reading_stages_need_a_resolvable_checkpoint(tmp_path):
    """apply and tier open safetensors themselves, so a bare repo id will not do.

    Learned the expensive way: a pipeline spent 114 seconds profiling and then failed
    in `tier` with "no safetensors checkpoint under allenai/OLMoE-1B-7B-0924". The
    whole reason build_stages validates up front is to make that a rejected request
    rather than a wasted run.
    """
    with pytest.raises(ValueError, match="neither a directory nor a cached"):
        build_stages(
            _spec(stages=["profile", "plan", "tier"], checkpoint="no/such-model-xyz"),
            str(tmp_path),
        )

    real = tmp_path / "ckpt"
    real.mkdir()
    stages = build_stages(
        _spec(stages=["profile", "plan", "tier"], checkpoint=str(real)), str(tmp_path)
    )
    assert [s.name for s in stages] == ["profile", "plan", "tier"]
    assert str(real) in stages[-1].argv


def test_stages_that_never_read_weights_accept_a_repo_id(tmp_path):
    """profile and gate go through vLLM's loader, which resolves a repo id fine."""
    spec = {
        "model": "org/only-a-repo-id",
        "corpus": "/tmp/c.jsonl",
        "heldout": "/tmp/h.jsonl",
        "core_experts": 8,
        "stages": ["profile", "plan", "gate"],
    }
    stages = build_stages(spec, str(tmp_path))
    assert [s.name for s in stages] == ["profile", "plan", "gate"]
    assert "org/only-a-repo-id" in stages[0].argv


def test_a_directory_is_passed_through_unresolved(tmp_path):
    from vllm_moe_surgeon.server.jobs import resolve_checkpoint

    assert resolve_checkpoint(str(tmp_path), {"tier"}) == str(tmp_path)


# ------------------------------------------------------------------ security


def test_trust_remote_code_in_llm_kwargs_is_refused():
    """A splat into vLLM's loader; trust_remote_code would run the repo's Python."""
    with pytest.raises(ValueError, match="trust_remote_code"):
        build_stages(_spec(llm_kwargs={"trust_remote_code": True}), "/w")


def test_only_allowlisted_llm_kwargs_pass():
    # A safe engine kwarg is fine.
    build_stages(_spec(llm_kwargs={"gpu_memory_utilization": 0.3}), "/w")
    # An unknown key is refused by name, allow-list not deny-list.
    with pytest.raises(ValueError, match="download_dir"):
        build_stages(_spec(llm_kwargs={"download_dir": "/etc"}), "/w")


def _echo_offline(name="profile"):
    """A stage that prints HF_HUB_OFFLINE so the runner's env reaches the tail."""
    return StageResult(
        name=name,
        argv=[
            sys.executable,
            "-c",
            "import os; print('OFFLINE=' + os.environ.get('HF_HUB_OFFLINE', 'unset'))",
        ],
    )


def test_runner_boots_stages_offline_unless_download_opted_in(tmp_path, monkeypatch):
    # Neutralise any offline flag inherited from the test host's own environment.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    store = JobStore(str(tmp_path / "jobs"))

    job = store.create(_spec(), str(tmp_path / "w"), [_echo_offline()])
    done = Runner(store).run(job)
    assert "OFFLINE=1" in done.stages[0].tail

    job2 = store.create(
        _spec(allow_download=True), str(tmp_path / "w2"), [_echo_offline()]
    )
    done2 = Runner(store).run(job2)
    assert "OFFLINE=unset" in done2.stages[0].tail


def test_a_token_gates_every_route_but_health(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from vllm_moe_surgeon.server.app import create_app

    with TestClient(create_app(str(tmp_path), token="s3cret")) as client:
        assert client.get("/health").status_code == 200  # health is open
        assert client.get("/jobs").status_code == 401  # no token
        assert client.get(
            "/jobs", headers={"X-Surgeon-Token": "wrong"}
        ).status_code == 401
        assert client.get(
            "/jobs", headers={"X-Surgeon-Token": "s3cret"}
        ).status_code == 200


# ----------------------------------------------------------------------
# headroom as the first stage: ask whether a smaller checkpoint already
# wins BEFORE spending a profile and a plan on this one.
# ----------------------------------------------------------------------


def test_headroom_runs_before_everything_it_could_retire():
    from vllm_moe_surgeon.server.jobs import STAGE_ORDER, build_stages

    assert STAGE_ORDER[0] == "headroom"
    stages = build_stages(
        {
            "model": "m",
            "heldout": "/h.jsonl",
            "candidates": ["small-a", "small-b"],
            "corpus": "/c.jsonl",
            "core_experts": 4,
            "stages": ["profile", "headroom", "plan"],
        },
        "/tmp/wd",
    )
    assert [s.name for s in stages] == ["headroom", "profile", "plan"]
    argv = stages[0].argv
    # This model plus every candidate, all ranked on the held-out corpus.
    assert argv.count("--model") == 3
    assert "small-a" in argv and "small-b" in argv
    assert "--corpus" in argv and "/h.jsonl" in argv


def test_headroom_needs_a_corpus_and_something_to_compare_against():
    import pytest as _pytest

    from vllm_moe_surgeon.server.jobs import build_stages

    with _pytest.raises(ValueError, match="needs a held-out corpus"):
        build_stages(
            {"model": "m", "candidates": ["x"], "stages": ["headroom"]}, "/tmp/wd"
        )
    # A ranking of one is not a ranking; refuse rather than boot an engine to
    # produce a table with a single row.
    with _pytest.raises(ValueError, match="needs candidates"):
        build_stages(
            {"model": "m", "heldout": "/h.jsonl", "stages": ["headroom"]}, "/tmp/wd"
        )
