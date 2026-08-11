# SPDX-License-Identifier: Apache-2.0
"""HTTP over :mod:`.jobs`. Deliberately thin.

Nothing here decides anything -- it validates a request, hands it to the runner and
reports what the runner recorded. Keeping it that shallow is what lets the pipeline be
tested without a web server, and lets a failed job be reproduced by copying one argv
out of the job record.

Jobs run in a background thread with a **single worker**, not a pool. Two pipelines at
once would both try to claim the GPU, and the second would fail to boot with an
out-of-memory error that looks like a bug in the model rather than a scheduling
mistake. Concurrency here would be a way of manufacturing confusing failures.
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Any

from pydantic import BaseModel, Field

from .._logging import init_logger
from .jobs import JobState, JobStore, Runner, build_stages, submit

logger = init_logger(__name__)


# Module scope, not inside create_app, and that is load-bearing. This module uses
# `from __future__ import annotations`, so FastAPI resolves a handler's annotations by
# name against the *module* globals -- a model defined inside the factory is invisible
# there, and FastAPI silently falls back to treating the parameter as a query string.
# The symptom is a 422 "field required: query.spec" for a perfectly good JSON body.
class JobSpec(BaseModel):
    model: str
    corpus: str | None = None
    heldout: str | None = None
    checkpoint: str | None = None
    model_id: str | None = None
    core_experts: int | None = None
    disk_experts: int | None = None
    drop_share_below: float | None = None
    vram_gib: float | None = None
    max_ratio: float = 1.3
    max_tokens: int = 64
    gate_limit: int | None = None
    force: bool = False
    stages: list[str] | None = None
    llm_kwargs: dict[str, Any] | None = None
    #: Opt in to fetching an uncached model from the network. Off by default, so an
    #: unauthenticated request cannot make the server download an arbitrary repo id.
    allow_download: bool = False

    # `model` and `model_id` are ordinary fields here, not pydantic internals.
    model_config = {"protected_namespaces": ()}


class JobSummary(BaseModel):
    id: str
    state: str
    error: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    stages: list[dict[str, Any]] = Field(default_factory=list)


DEFAULT_ROOT = os.environ.get("MOE_SURGEON_STATE", "./surgeon-state")


class JobQueue:
    """One worker, FIFO. See the module docstring on why not a pool."""

    def __init__(self, store: JobStore, root: str, *, timeout: float | None = None):
        self.store = store
        self.root = root
        self.runner = Runner(store, timeout=timeout)
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                job = self.store.get(job_id)
                if job is None or job.state != JobState.pending.value:
                    continue
                self.runner.run(job)
            except Exception as exc:  # pragma: no cover - worker must not die
                logger.warning("job %s raised in the worker: %s", job_id, exc)
                job = self.store.get(job_id)
                if job is not None:
                    job.state = JobState.failed.value
                    job.error = f"worker error: {exc}"
                    self.store.save(job)
            finally:
                self._queue.task_done()

    def enqueue(self, spec: dict[str, Any]):
        import uuid

        job_id = uuid.uuid4().hex[:12]
        workdir = os.path.join(self.root, "runs", job_id)
        # Built before the job is recorded, so an unrunnable request is a 400 rather
        # than a job that exists only to fail.
        stages = build_stages(spec, workdir)
        os.makedirs(workdir, exist_ok=True)
        job = self.store.create(spec, workdir, stages, job_id=job_id)
        self._queue.put(job.id)
        return job

    def cancel(self, job_id: str) -> bool:
        return self.runner.cancel(job_id)


def create_app(
    root: str | None = None,
    *,
    timeout: float | None = None,
    token: str | None = None,
):
    """Build the FastAPI app. Import is local so the offline half needs no web deps.

    ``token``: when set, every request except ``/health`` must carry it in the
    ``X-Surgeon-Token`` header or gets a 401. The server submits subprocesses from
    request-supplied fields, so an open ``0.0.0.0`` bind with no token is a remote
    handle on the operator's machine. There is no default token: unset means the
    server trusts whoever can reach it, which is only safe on a loopback bind.
    """
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    state_root = root or DEFAULT_ROOT
    store = JobStore(os.path.join(state_root, "jobs"))
    jobs = JobQueue(store, state_root, timeout=timeout)

    app = FastAPI(title="moe-surgeon", version="0.1.0")

    if token:
        @app.middleware("http")
        async def _require_token(request: Request, call_next):
            if request.url.path != "/health":
                supplied = request.headers.get("x-surgeon-token")
                if supplied != token:
                    return JSONResponse(
                        {"detail": "missing or invalid X-Surgeon-Token"},
                        status_code=401,
                    )
            return await call_next(request)

    def summarize(job) -> JobSummary:
        return JobSummary(
            id=job.id,
            state=job.state,
            error=job.error,
            artifacts=job.artifacts,
            stages=[
                {
                    "name": s.name,
                    "state": s.state,
                    "returncode": s.returncode,
                    "seconds": round(s.seconds, 2),
                }
                for s in job.stages
            ],
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "state_root": state_root}

    @app.post("/jobs", response_model=JobSummary, status_code=202)
    def create_job(spec: JobSpec) -> JobSummary:
        try:
            job = jobs.enqueue(spec.model_dump(exclude_none=True))
        except ValueError as exc:
            # An unrunnable request is the client's problem, and saying which field is
            # wrong beats a job that exists only to fail three stages later.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return summarize(job)

    @app.get("/jobs", response_model=list[JobSummary])
    def list_jobs() -> list[JobSummary]:
        return [summarize(j) for j in store.list()]

    @app.get("/jobs/{job_id}", response_model=JobSummary)
    def get_job(job_id: str) -> JobSummary:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")
        return summarize(job)

    @app.get("/jobs/{job_id}/log/{stage}")
    def get_log(job_id: str, stage: str) -> dict[str, str]:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")
        for result in job.stages:
            if result.name == stage:
                return {"stage": stage, "log": result.tail}
        raise HTTPException(status_code=404, detail=f"job {job_id} has no {stage}")

    @app.post("/jobs/{job_id}/cancel", response_model=JobSummary)
    def cancel_job(job_id: str) -> JobSummary:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")
        if JobState(job.state).terminal:
            raise HTTPException(
                status_code=409, detail=f"job {job_id} already {job.state}"
            )
        jobs.cancel(job_id)
        return summarize(store.get(job_id) or job)

    app.state.store = store
    app.state.jobs = jobs
    return app


__all__ = ["JobQueue", "create_app", "submit"]
