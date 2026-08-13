# SPDX-License-Identifier: Apache-2.0
"""Job orchestration: run the pipeline's stages and keep a record of what happened.

No FastAPI here. The web layer is a thin wrapper in :mod:`.app`, and everything that
decides anything lives in this module so it can be tested without a server, a GPU or
a model.

Three decisions shape it:

**Stages run as subprocesses, never in-process.** A profiling or gate stage boots a
vLLM engine, and an engine does not release its device memory when the Python object
falls out of scope -- measured while benchmarking, where running four arms in one
process left two unable to boot. A stage that crashes must also not take the server
with it. So each stage is ``python -m vllm_moe_surgeon.cli <stage> ...``, argv as a
list with no shell.

**The pipeline is the CLI.** The server adds orchestration, persistence and a record;
it adds no capability. Anything it can do is reproducible by hand from the log, which
is the property that makes it debuggable.

**Which stages to run is declared, not inferred.** Whether deletion is worth doing
depends on the profile, which does not exist until the pipeline is under way, so the
server would have to guess. It asks instead -- and ``surgeon recommend`` is a stage
whose whole job is to answer that question for the operator.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .._logging import init_logger

logger = init_logger(__name__)

#: Stages in the order they must run. A request names a subset; the runner keeps this
#: order regardless of how the request listed them, because plan needs profile and
#: gate needs plan.
#: headroom runs first on purpose: it asks whether a smaller existing
#: checkpoint already serves the domain better, and a yes retires every
#: stage after it. Measuring that after a profile and a plan is measuring
#: it after the regret.
STAGE_ORDER = (
    "headroom", "profile", "recommend", "plan", "gate", "apply", "tier",
)


class JobState(str, Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (JobState.succeeded, JobState.failed, JobState.cancelled)


@dataclass
class StageResult:
    name: str
    argv: list[str]
    state: str = JobState.pending.value
    returncode: int | None = None
    seconds: float = 0.0
    log_path: str | None = None
    #: Tail of the stage's output, so a failure is diagnosable from the job record
    #: alone without opening a file that may have been cleaned up.
    tail: str = ""


@dataclass
class Job:
    id: str
    spec: dict[str, Any]
    workdir: str
    stages: list[StageResult] = field(default_factory=list)
    state: str = JobState.pending.value
    created: float = 0.0
    finished: float | None = None
    error: str | None = None
    #: Paths the job produced, by role.
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Job:
        stages = [StageResult(**s) for s in payload.get("stages", [])]
        return cls(**{**payload, "stages": stages})


class JobStore:
    """Jobs on disk, one JSON file each.

    Persisted rather than held in memory so a server restart does not lose the
    record of a two-hour pipeline. Written atomically, because a half-written job
    file read back as JSON would fail the whole listing.
    """

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, job_id: str) -> str:
        return os.path.join(self.root, f"{job_id}.json")

    def create(
        self,
        spec: dict[str, Any],
        workdir: str,
        stages: list[StageResult],
        job_id: str | None = None,
    ) -> Job:
        job = Job(
            id=job_id or uuid.uuid4().hex[:12],
            spec=spec,
            workdir=workdir,
            stages=stages,
            created=time.time(),
        )
        self.save(job)
        return job

    def save(self, job: Job) -> None:
        path = self._path(job.id)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(job.to_dict(), f, indent=2)
        os.replace(tmp, path)

    def get(self, job_id: str) -> Job | None:
        try:
            with open(self._path(job_id)) as f:
                return Job.from_dict(json.load(f))
        except FileNotFoundError:
            return None

    def list(self) -> list[Job]:
        jobs = []
        for name in os.listdir(self.root):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.root, name)) as f:
                    jobs.append(Job.from_dict(json.load(f)))
            except (json.JSONDecodeError, TypeError) as exc:
                # One unreadable file must not blank the whole listing.
                logger.warning("skipping unreadable job %s: %s", name, exc)
        return sorted(jobs, key=lambda j: j.created, reverse=True)


# What the weight-reading stages actually open: shards, the shard index, and the
# config that says how many experts there are. Nothing else -- no tokenizer, no
# README. The narrowness is the point; see resolve_checkpoint.
WEIGHT_FILE_PATTERNS = ("*.safetensors", "*.safetensors.index.json", "config.json")


# The only engine kwargs a request may pass through to `LLM(...)`. An allow-list, not
# a deny-list, because the request comes over HTTP from a caller we do not trust: the
# splat reaches vLLM's model loader, so `trust_remote_code` here is "run this
# stranger's Python as the server user", and keys like `download_dir` or `load_format`
# widen the surface further. Anything outside this set is a 400 naming the key.
ALLOWED_LLM_KWARGS = frozenset({
    "gpu_memory_utilization",
    "max_model_len",
    "dtype",
    "enforce_eager",
    "max_num_seqs",
    "seed",
    "kv_cache_dtype",
    "swap_space",
})


def validate_llm_kwargs(llm_kwargs: Any) -> None:
    """Refuse any engine kwarg not on :data:`ALLOWED_LLM_KWARGS`, by name.

    Raises :class:`ValueError` (which the HTTP layer turns into a 400). The message
    names the offending key rather than silently dropping it, so a caller who meant
    well learns why, and a caller who did not gets nothing.
    """
    if llm_kwargs is None:
        return
    if not isinstance(llm_kwargs, dict):
        raise ValueError("llm_kwargs must be an object")
    bad = sorted(k for k in llm_kwargs if k not in ALLOWED_LLM_KWARGS)
    if bad:
        raise ValueError(
            f"llm_kwargs contains keys that are not allowed over the API: {bad}. "
            f"Allowed keys are {sorted(ALLOWED_LLM_KWARGS)}. In particular "
            "`trust_remote_code` is refused: it would execute the checkpoint's own "
            "Python in the server process."
        )


def resolve_checkpoint(reference: str, needed_by: set[str]) -> str:
    """A local directory of safetensors for ``reference``.

    Passes a directory through. Otherwise treats it as a Hugging Face repo id and asks
    the local cache where the snapshot is -- ``local_files_only``, so this never
    downloads behind the operator's back. Resolving is a usability matter as much as a
    correctness one: the snapshot lives in a hash-named directory nobody wants to find
    and paste by hand.

    ``allow_patterns`` is what makes this work on a real cache. A bare
    ``local_files_only`` resolution demands the *entire* repo be present and raises
    ``IncompleteSnapshotError`` when it is not -- which is the normal state of a cache
    filled by vLLM, since it never fetches a repo's README or .gitattributes. Asking
    only for the files these stages read states the actual requirement.
    """
    if os.path.isdir(reference):
        return reference
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            reference, local_files_only=True, allow_patterns=list(WEIGHT_FILE_PATTERNS)
        )
    except Exception as exc:
        raise ValueError(
            f"stages {sorted(needed_by)} read weights directly and need a local "
            f"directory, but {reference!r} is neither a directory nor a cached "
            f"Hugging Face snapshot with {list(WEIGHT_FILE_PATTERNS)} "
            f"({type(exc).__name__}). Download it first, or pass `checkpoint`."
        ) from exc


def build_stages(spec: dict[str, Any], workdir: str) -> list[StageResult]:
    """Turn a request into the argv of each stage, in dependency order.

    Raises on a request that cannot run, rather than starting a pipeline that will
    fail three stages in -- a profiling stage costs minutes, so the cheap checks
    happen first.
    """
    requested = list(spec.get("stages") or ("profile", "plan", "gate", "tier"))
    unknown = [s for s in requested if s not in STAGE_ORDER]
    if unknown:
        raise ValueError(f"unknown stages {unknown}; known: {list(STAGE_ORDER)}")
    ordered = [s for s in STAGE_ORDER if s in requested]

    model = spec.get("model")
    if not model:
        raise ValueError("spec needs a model")
    # Reject a dangerous engine kwarg before any stage is built, not after a stage
    # boots with it.
    validate_llm_kwargs(spec.get("llm_kwargs"))
    profile = os.path.join(workdir, "profile.npz")
    plan = os.path.join(workdir, "plan.json")

    if "headroom" in ordered and not spec.get("heldout"):
        raise ValueError("the headroom stage needs a held-out corpus")
    if "headroom" in ordered and len(spec.get("candidates") or ()) < 1:
        raise ValueError(
            "the headroom stage needs candidates: a list of models to "
            "rank against this one"
        )
    if "profile" in ordered and not spec.get("corpus"):
        raise ValueError("the profile stage needs a corpus")
    if "gate" in ordered and not spec.get("heldout"):
        raise ValueError("the gate stage needs a held-out corpus")
    if "plan" in ordered and not spec.get("core_experts"):
        raise ValueError("the plan stage needs core_experts")
    if {"plan", "gate", "apply", "tier"} & set(ordered) and "profile" not in ordered:
        if not os.path.exists(profile):
            raise ValueError(
                f"stages {ordered} need a profile, but the profile stage was not "
                f"requested and {profile} does not exist"
            )
    # apply and tier open safetensors themselves, so they need a real directory. A
    # bare repo id is enough for the stages that go through vLLM's loader and useless
    # to these. Resolved up front rather than after the fact: it cost a 114-second
    # profiling stage to learn that once, which is what this validation is for.
    weight_readers = {"apply", "tier"} & set(ordered)
    checkpoint = spec.get("checkpoint") or model
    if weight_readers:
        checkpoint = resolve_checkpoint(checkpoint, weight_readers)

    base = [sys.executable, "-m", "vllm_moe_surgeon.cli"]
    # Stages that go through vLLM's loader can take the repo id as given.
    loader_model = spec.get("checkpoint") or model
    llm_kwargs = spec.get("llm_kwargs")
    out: list[StageResult] = []

    for stage in ordered:
        if stage == "headroom":
            argv = base + [
                "headroom", "--corpus", spec["heldout"], "--model", loader_model,
                "--out", os.path.join(workdir, "headroom.json"),
            ]
            for candidate in spec["candidates"]:
                argv += ["--model", candidate]
            if llm_kwargs:
                argv += ["--llm-kwargs", json.dumps(llm_kwargs)]
        elif stage == "profile":
            argv = base + [
                "profile", "--model", model, "--corpus", spec["corpus"],
                "--cooc", "--out", profile,
                "--max-tokens", str(spec.get("max_tokens", 64)),
            ]
            if llm_kwargs:
                argv += ["--llm-kwargs", json.dumps(llm_kwargs)]
        elif stage == "recommend":
            argv = base + [
                "recommend", "--checkpoint", checkpoint, "--profile", profile,
            ]
            if spec.get("vram_gib"):
                argv += ["--vram", str(spec["vram_gib"])]
        elif stage == "plan":
            argv = base + [
                "plan", "--profile", profile,
                "--core-experts", str(spec["core_experts"]), "--out", plan,
            ]
            if spec.get("disk_experts") is not None:
                argv += ["--disk-experts", str(spec["disk_experts"])]
            if spec.get("drop_share_below") is not None:
                argv += ["--drop-share-below", str(spec["drop_share_below"])]
            if spec.get("force"):
                argv += ["--force"]
        elif stage == "gate":
            argv = base + [
                "gate", "--plan", plan, "--corpus", spec["heldout"],
                "--max-ratio", str(spec.get("max_ratio", 1.3)),
            ]
            if spec.get("gate_limit"):
                argv += ["--limit", str(spec["gate_limit"])]
            if llm_kwargs:
                argv += ["--llm-kwargs", json.dumps(llm_kwargs)]
        elif stage == "apply":
            argv = base + [
                "apply", "--plan", plan,
                "--source", checkpoint,
                "--out", os.path.join(workdir, "checkpoint"),
            ]
        else:  # tier
            argv = base + [
                "tier", "--plan", plan,
                "--source", checkpoint,
                "--store", os.path.join(workdir, "store"),
                "--hot-experts", os.path.join(workdir, "hot_experts.json"),
                "--model-id", spec.get("model_id", loader_model),
            ]
        out.append(StageResult(name=stage, argv=argv))
    return out


class Runner:
    """Runs a job's stages in order, stopping at the first failure.

    Stopping matters: a plan built on a failed profile, or a checkpoint written from
    a plan that failed its gate, would be worse than no artifact -- it would be an
    artifact nobody knows is invalid.
    """

    def __init__(self, store: JobStore, *, timeout: float | None = None):
        self.store = store
        self.timeout = timeout
        self._processes: dict[str, subprocess.Popen] = {}
        self._cancelled: set[str] = set()

    def cancel(self, job_id: str) -> bool:
        """Ask a running job to stop. Returns whether there was one to stop."""
        self._cancelled.add(job_id)
        proc = self._processes.get(job_id)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            return True
        return False

    def run(self, job: Job, env: dict[str, str] | None = None) -> Job:
        os.makedirs(job.workdir, exist_ok=True)
        job.state = JobState.running.value
        self.store.save(job)

        merged = dict(os.environ)
        # Offline by default: a stage that boots vLLM's loader would otherwise fetch
        # any repo id the request named, straight from the network, as the server
        # user. Downloading is opt-in per job, not the default an unauthenticated
        # POST gets. This does not by itself stop `trust_remote_code` on an
        # already-cached repo -- that is what validate_llm_kwargs is for.
        if not job.spec.get("allow_download"):
            merged.setdefault("HF_HUB_OFFLINE", "1")
            merged.setdefault("TRANSFORMERS_OFFLINE", "1")
        merged.update(env or {})

        for stage in job.stages:
            if job.id in self._cancelled:
                stage.state = JobState.cancelled.value
                job.state = JobState.cancelled.value
                job.error = "cancelled before this stage started"
                break

            stage.log_path = os.path.join(job.workdir, f"{stage.name}.log")
            stage.state = JobState.running.value
            self.store.save(job)
            started = time.time()

            try:
                with open(stage.log_path, "w") as log:
                    proc = subprocess.Popen(
                        stage.argv, stdout=log, stderr=subprocess.STDOUT,
                        env=merged, cwd=os.getcwd(),
                    )
                    self._processes[job.id] = proc
                    stage.returncode = proc.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                stage.returncode = -1
                stage.state = JobState.failed.value
                job.state = JobState.failed.value
                job.error = f"stage {stage.name} exceeded {self.timeout}s"
            except OSError as exc:
                stage.returncode = -1
                stage.state = JobState.failed.value
                job.state = JobState.failed.value
                job.error = f"stage {stage.name} could not start: {exc}"
            finally:
                self._processes.pop(job.id, None)
                stage.seconds = time.time() - started
                stage.tail = _tail(stage.log_path)

            if job.state in (JobState.failed.value, JobState.cancelled.value):
                break
            if job.id in self._cancelled:
                stage.state = JobState.cancelled.value
                job.state = JobState.cancelled.value
                job.error = f"cancelled during {stage.name}"
                break
            if stage.returncode != 0:
                stage.state = JobState.failed.value
                job.state = JobState.failed.value
                # The gate exits non-zero on a genuine quality failure, which is a
                # verdict rather than a malfunction; say which it is.
                job.error = (
                    f"stage {stage.name} failed with exit {stage.returncode}"
                    + (
                        " -- for `gate` this usually means the plan exceeded the "
                        "perplexity ceiling, which is a verdict, not a crash"
                        if stage.name == "gate"
                        else ""
                    )
                )
                break
            stage.state = JobState.succeeded.value
            self.store.save(job)

        if job.state == JobState.running.value:
            job.state = JobState.succeeded.value
        job.finished = time.time()
        job.artifacts = collect_artifacts(job)
        self.store.save(job)
        return job


def collect_artifacts(job: Job) -> dict[str, str]:
    """Which of the expected outputs actually exist on disk.

    Existence is checked rather than assumed from a stage's exit code, so a job
    record never advertises an artifact that is not there.
    """
    candidates = {
        "headroom": os.path.join(job.workdir, "headroom.json"),
        "profile": os.path.join(job.workdir, "profile.npz"),
        "plan": os.path.join(job.workdir, "plan.json"),
        "checkpoint": os.path.join(job.workdir, "checkpoint"),
        "store": os.path.join(job.workdir, "store"),
        "hot_experts": os.path.join(job.workdir, "hot_experts.json"),
    }
    return {role: path for role, path in candidates.items() if os.path.exists(path)}


def _tail(path: str | None, limit: int = 2000) -> str:
    if not path:
        return ""
    try:
        with open(path) as f:
            return f.read()[-limit:]
    except OSError:
        return ""


def submit(
    store: JobStore,
    spec: dict[str, Any],
    *,
    root: str,
    runner: Runner | None = None,
    env: dict[str, str] | None = None,
) -> Job:
    """Create and run a job synchronously. Returns the finished job.

    The workdir is fixed before the stages are built, because a stage's argv embeds
    the paths it reads and writes -- building them against a placeholder and
    rewriting afterwards is how a validation ends up checking the wrong directory.
    """
    job_id = uuid.uuid4().hex[:12]
    workdir = os.path.join(root, "runs", job_id)
    stages = build_stages(spec, workdir)  # raises before anything is created
    os.makedirs(workdir, exist_ok=True)
    job = store.create(spec, workdir, stages, job_id=job_id)
    return (runner or Runner(store)).run(job, env=env)
