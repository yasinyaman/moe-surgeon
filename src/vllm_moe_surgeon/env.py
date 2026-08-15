# SPDX-License-Identifier: Apache-2.0
"""Lazily-read settings for the expert cache and disk tier.

This module deliberately mimics the shape of ``vllm.envs``: every name is
resolved through the module-level ``__getattr__`` on *each* access, reading
``os.environ`` at that moment. Two things depend on that laziness:

- The ported unit suite drives behaviour with ``monkeypatch.setenv`` and then
  re-constructs a provider. Snapshotting into a dataclass at import time would
  silently ignore those writes.
- ``worker_count()`` in :mod:`.store.expert_load_pipeline` re-reads the thread
  count after a fork.

The ``VLLM_MOE_*`` names are kept verbatim from the in-tree prototype. That is a
compatibility decision, not inertia: ``bench/`` on both test machines and the
recipes in the project notes set these exact variables, and renaming them would
invalidate every recorded measurement's provenance.

Precedence, highest first:

1. :func:`set_overrides` -- populated by the vLLM plugin from
   ``--additional-config '{"surgeon": {...}}'``, so a served deployment can be
   configured without environment plumbing through the worker processes.
2. ``os.environ``
3. the default in :data:`_SPEC`
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

# Populated by vllm_moe_surgeon.plugin from --additional-config. Keys are the
# bare setting names (VLLM_MOE_RAM_CACHE, ...) and values are already-typed
# Python objects, so they bypass the parsers below.
_OVERRIDES: dict[str, Any] = {}


def set_overrides(values: dict[str, Any]) -> None:
    """Install config-file overrides. Replaces any previous set.

    Called once per process by the plugin. Re-entrant by construction: the
    plugin may run more than once and vLLM makes no promise about how often.
    """
    _OVERRIDES.clear()
    _OVERRIDES.update(values)


def _bool(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _opt_str(raw: str) -> str | None:
    return raw or None


def _choice(*allowed: str) -> Callable[[str], str]:
    def parse(raw: str) -> str:
        value = raw.strip().lower()
        if value not in allowed:
            raise ValueError(
                f"expected one of {list(allowed)}, got {raw!r}"
            )
        return value

    return parse


# name -> (parser, default)
_SPEC: dict[str, tuple[Callable[[str], Any], Any]] = {
    # --- disk tier -------------------------------------------------------
    # Directory holding one <layer>.experts record file per MoE layer.
    # Together with VLLM_MOE_RAM_CACHE > 0 this turns the disk tier on.
    "VLLM_MOE_DISK_STORE_DIR": (_opt_str, None),
    # Pinned host-RAM records per layer (the warm tier). Must be >= the GPU
    # capacity; 0 disables the disk tier entirely.
    "VLLM_MOE_RAM_CACHE": (int, 0),
    # Intercept the weight loader and write experts straight to the store, so
    # the [num_experts, ...] tensor is never materialised.
    "VLLM_MOE_STREAM_LOAD": (_bool, False),
    # Route disk reads through the bounded reader pool. 0 is serial on the
    # calling thread and bit-identical to the pre-pipeline implementation.
    "VLLM_MOE_DISK_PIPELINE": (_bool, True),
    # Reader threads, clamped to [1, 4] at use. Past 4 the measured p99 read
    # latency degrades badly (4 ms -> 128 ms) for no bandwidth gain.
    "VLLM_MOE_DISK_IO_THREADS": (int, 2),
    # Read the store through the page cache instead of O_DIRECT, making the
    # kernel the warm tier rather than the pinned RAM pool. Off by default:
    # O_DIRECT is what keeps the RAM tier the only RAM this path uses, and a
    # cached copy alongside a pinned copy is the same bytes twice. Selectable
    # because the alternative is worth measuring -- a page cache is reclaimable,
    # so it never trips the pinned-pool refusal that a large ram_cache does.
    # Records stay 4096-padded either way, so the same store file serves both.
    "VLLM_MOE_DISK_BUFFERED": (_bool, False),
    # Cross-group RAM prefetch under the current group's kernel. Silently
    # no-ops unless ram_capacity >= 2 * capacity; rejects zero copy.
    "VLLM_MOE_DISK_PREFETCH": (_bool, False),
    # CPU expert co-execution: compute cold experts on the host instead of
    # fetching them over H2D. Measured 3.7x on a discrete card (l6 gate) and
    # a LOSS on unified memory (l5 gate) -- per-machine opt-in, never a
    # default, and not bit-exact.
    "VLLM_MOE_CPU_EXPERTS": (_bool, False),
    # Experts below this routed-token count stay on the GPU path. The T=1
    # cliff is handled by pad-to-2 inside the CPU kernel, so 1 is the
    # measured default -- excluding T=1 empties the candidate set entirely.
    "VLLM_MOE_CPU_EXPERT_MIN_TOKENS": (int, 1),
    # torch intra-op threads for the host GEMMs. 0 leaves the pool alone;
    # 10-of-20 measured best on GB10-class hosts, 12 on the i7-12700H.
    "VLLM_MOE_CPU_EXPERT_THREADS": (int, 0),
    # Row-scaled fp8-e4m3 records. bf16/fp16 unquantized checkpoints only.
    # Must be set identically at store-build time and at serve time.
    "VLLM_MOE_DISK_STORE_FP8": (_bool, False),
    # --- cache policy ----------------------------------------------------
    "VLLM_MOE_CACHE_POLICY": (_choice("lfru", "ewma"), "lfru"),
    "VLLM_MOE_CACHE_DECAY": (float, 0.999),
    # --- zero copy -------------------------------------------------------
    # Let the kernel read the pinned RAM pool directly. Unified memory only;
    # collapses the GPU tier so capacity becomes VLLM_MOE_RAM_CACHE.
    "VLLM_MOE_ZERO_COPY": (_bool, False),
    # Retained fp8 cold rows behind the zero-copy pool.
    "VLLM_MOE_ZC_FP8_SLOTS": (int, 0),
    # --- observability ---------------------------------------------------
    # Append path for the routing trace: "<layer_name> <comma-separated ids>"
    # per prepare() call. Single-process only.
    "VLLM_MOE_ROUTING_TRACE": (_opt_str, None),
    # --- expert cache placement (was CLI in the in-tree prototype) -------
    "VLLM_MOE_EXPERT_CACHE_SIZE": (int, 0),
    "VLLM_MOE_EXPERT_CACHE_SPLIT": (_choice("token", "expert"), "token"),
    # Path to hot_experts.json, produced by the surgery pipeline. Seeds the
    # eviction policy's prior so a tiered artifact's cache does not boot cold.
    "VLLM_MOE_HOT_EXPERTS": (_opt_str, None),
    # --- telemetry -------------------------------------------------------
    # Accumulate per-(layer, expert) token counts and gate mass during the
    # forward pass. Cheap, GPU-side, no host sync.
    "VLLM_MOE_RECORD_STATS": (_bool, False),
    # Also accumulate the within-token expert co-occurrence matrix. O(E^2) per
    # layer, so it is separate from VLLM_MOE_RECORD_STATS.
    "VLLM_MOE_RECORD_COOC": (_bool, False),
}


def __getattr__(name: str) -> Any:
    if name not in _SPEC:
        raise AttributeError(f"unknown setting: {name}")
    if name in _OVERRIDES:
        return _OVERRIDES[name]
    parse, default = _SPEC[name]
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return parse(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is invalid: {exc}") from exc


def __dir__() -> list[str]:
    return sorted(_SPEC)


def snapshot() -> dict[str, Any]:
    """Every setting's current effective value. For logging and manifests."""
    return {name: __getattr__(name) for name in _SPEC}
