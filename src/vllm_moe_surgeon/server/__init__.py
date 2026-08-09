# SPDX-License-Identifier: Apache-2.0
"""The job server: orchestration, persistence and a record of what ran.

It adds no capability. Every stage is a `surgeon` CLI invocation, so anything the
server does is reproducible by hand from the job's log -- which is what makes a
failed pipeline debuggable rather than mysterious.

:mod:`.jobs` holds everything that decides anything and imports no web framework, so
it is testable without a server. :mod:`.app` is the HTTP wrapper and needs the
``server`` extra.
"""

from .jobs import (
    STAGE_ORDER,
    Job,
    JobState,
    JobStore,
    Runner,
    StageResult,
    build_stages,
    collect_artifacts,
    submit,
)

__all__ = [
    "STAGE_ORDER",
    "Job",
    "JobState",
    "JobStore",
    "Runner",
    "StageResult",
    "build_stages",
    "collect_artifacts",
    "submit",
]
