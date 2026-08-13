# SPDX-License-Identifier: Apache-2.0
"""The three-tier expert weight store: NVMe -> pinned host RAM -> VRAM.

Lifted verbatim out of the in-tree prototype (branch ``disk-tier-proto``), with
only the imports rewritten: ``vllm.envs`` -> :mod:`vllm_moe_surgeon.env` and
``vllm.logger`` -> :mod:`vllm_moe_surgeon._logging`. Nothing here imports vLLM,
which is what makes the store reusable from the offline surgery pipeline -- the
same record format that serves a tiered artifact is what the pipeline writes.

The layer-facing glue that installs a provider onto a MoE layer does *not* live
here; it lives in :mod:`vllm_moe_surgeon.compat`, so that everything which knows
about vLLM internals sits behind one seam.
"""

from .expert_cpu_exec import run_with_cpu_coexec
from .expert_disk_store import ALIGN, DiskExpertStore, quantize_rowwise_fp8
from .expert_load_pipeline import (
    DiskLoadWorker,
    get_disk_load_worker,
    shutdown_disk_load_worker,
    worker_count,
)
from .expert_policy import EWMAPolicy, LFRUPolicy, make_policy
from .expert_weight_provider import (
    CachedWeightProvider,
    ExpertWeightResult,
    routing_trace_file,
    run_with_expert_cache,
)

__all__ = [
    "ALIGN",
    "CachedWeightProvider",
    "DiskExpertStore",
    "DiskLoadWorker",
    "EWMAPolicy",
    "ExpertWeightResult",
    "LFRUPolicy",
    "get_disk_load_worker",
    "make_policy",
    "quantize_rowwise_fp8",
    "routing_trace_file",
    "run_with_cpu_coexec",
    "run_with_expert_cache",
    "shutdown_disk_load_worker",
    "worker_count",
]
