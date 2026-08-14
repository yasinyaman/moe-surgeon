# SPDX-License-Identifier: Apache-2.0
"""CPU expert co-execution: compute cold experts on the host, never H2D them.

Measured basis (bench/l5_cpu_coexec_gate.py on GB10, bench/l6_cpu_coexec_gate.py
on the laptop; docs/benchmarks.md, "CPU co-execution"). On a discrete card the
host GEMM streams weights at ~34 GB/s effective while PCIe H2D delivers
~11 GB/s: a cold expert costs 368.8 us on the CPU against 1139.7 us of H2D,
and the concurrent single-layer arm measured **3.7x at f=1.0** with contention
1.037. On unified memory (GB10) the same experiment measured contention 2.385
and the co-exec arm LOST (0.719x) -- there is no second bandwidth pool there.
Hence a loud per-machine opt-in, never a default. Not bit-exact: the host
GEMM's reduction order differs from the fused kernel's.

The T=1 trap, measured twice: a single-token expert forward takes the GEMV
path at 1000-1500 us where the T=2 GEMM costs ~280-370 us -- and the cold
tail is almost entirely T=1 (with a T>=2 *exclusion* the first gate run
selected zero experts in every rep). The rule is therefore pad, never
exclude: :func:`cpu_expert_forward` pads a single row to two and discards
the pad's output.

**The GPU cache stops adapting while this is on, and that is by design.**
Every CPU-served expert is masked out of the ids the planners see, so
``prepare()`` only ever encounters residents: no insertions, no evictions,
the resident set frozen at whatever the last uncovered forward left. This
is why the A/B's GPU misses collapse (95,405 -> 1,414) -- the misses became
invisible, the cache did not learn. Two consequences worth knowing before
reading any counter: the GPU tier's eviction policy is unreachable while
co-exec covers the whole miss set, and a low "GPU cache hit rate" reading
means the opposite of what it means without this mode.

This module is vLLM-free on purpose (tests/test_layering.py enforces it);
everything vLLM-specific stays in ``compat/``.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from .._logging import init_logger
from .expert_weight_provider import run_with_expert_cache

logger = init_logger(__name__)

#: Co-execution is measured for decode-shaped forwards. A prefill routes
#: hundreds of rows whose union approaches the full expert set, where
#: residency churn -- not fetch latency -- dominates. Deliberately NOT named
#: ``_DECODE_ROWS``: ``compat/runtime.py`` owns that name for the
#: prefill/decode split at 512, and this is a narrower policy cutoff (the
#: band the gates measured), not a second opinion on what a decode is.
_MAX_COEXEC_ROWS = 512


def select_cpu_experts(
    unique_ids: list[int],
    counts: dict[int, int],
    gpu_resident: set[int],
    ram_resident: set[int],
    capacity: int,
    max_cpu: int,
    min_tokens: int,
) -> list[int]:
    """The experts this forward should compute on the host.

    Empty when the union already fits the GPU cache. Otherwise: every
    GPU-miss with at least *min_tokens* routed tokens, RAM-resident ones
    first (no disk read), then coldest first (least host FLOPs), capped at
    *max_cpu* so the provider's RAM victim scan always has a victim. Taking
    every miss leaves only residents on the GPU path -- the f=1.0 policy the
    laptop gate measured as the winner -- and removes the chunk split as a
    side effect.
    """
    if len(unique_ids) <= capacity or max_cpu <= 0:
        return []
    candidates = [
        e
        for e in unique_ids
        if e not in gpu_resident and counts.get(e, 0) >= min_tokens
    ]
    candidates.sort(key=lambda e: (e not in ram_resident, counts.get(e, 0), e))
    return candidates[:max_cpu]


def masked_ids(topk_ids: torch.Tensor, cpu_ids: list[int]) -> torch.Tensor:
    """A copy of *topk_ids* with the CPU-served experts hidden as -1.

    The planners filter negative ids at the boundary, so the GPU path never
    plans -- and ``prepare()`` never reads GPU-tier state for -- an expert
    the host is serving. The kernel still receives the ORIGINAL ids, so its
    ``expert_map`` zeroes these pairs and the outputs stay additive.
    """
    if not cpu_ids:
        return topk_ids
    hide = torch.tensor(cpu_ids, dtype=topk_ids.dtype, device=topk_ids.device)
    return torch.where(torch.isin(topk_ids, hide), -1, topk_ids)


def cpu_expert_forward(
    x_host: torch.Tensor,
    ids_host: torch.Tensor,
    weights_host: torch.Tensor,
    cpu_ids: list[int],
    views: dict[int, tuple[torch.Tensor, torch.Tensor]],
    out: torch.Tensor,
) -> int:
    """Accumulate the CPU-served experts' gate-weighted outputs into *out*.

    *out* is fp32 ``[T, H]``, pre-zeroed; the return value is the number of
    (token, expert) pairs computed. Silu-and-mul only -- the caller has
    already refused anything else.
    """
    pairs = 0
    for eid in cpu_ids:
        rows, kslots = (ids_host == eid).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        w13, w2 = views[eid]
        xr = x_host[rows]
        padded = xr.shape[0] == 1
        if padded:
            # The T=1 GEMV cliff, measured 1000-1500 us vs ~300 padded.
            xr = torch.cat([xr, torch.zeros_like(xr)])
        h = xr @ w13.t()
        # chunk rather than slicing at a precomputed intermediate size: the
        # invariant is w13 = [2I, H] with gate first, and it is right here.
        gate, up = h.chunk(2, dim=1)
        act = torch.nn.functional.silu(gate.float()).to(xr.dtype) * up
        y = act @ w2.t()
        if padded:
            y = y[:1]
        gates = weights_host[rows, kslots].float().unsqueeze(1)
        # index_add_, not out[rows] += ...: the latter lowers to a
        # non-accumulating index_put_, so a token routed to this expert in
        # two k-slots would keep only the last term and silently lose the
        # other. Distinct top-k never produces that, but a grouped router or
        # a global->local map that collapses two ids does.
        out.index_add_(0, rows, gates * y.float())
        pairs += int(rows.numel())
    return pairs


@torch.compiler.disable
def run_with_cpu_coexec(
    provider,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    run: Callable,
    *,
    min_tokens: int = 1,
) -> torch.Tensor | None:
    """One MoE forward with the cold experts computed on the host.

    Returns ``None`` when co-execution is not applicable this forward
    (prefill-shaped, union fits the cache, nothing eligible) -- the caller
    falls through to the plain path.

    Ordering is what makes this lock-free: the provider is only ever touched
    from this host thread, the RAM rows the host reads are protected from
    eviction between :meth:`cpu_views_for` and :meth:`cpu_release`, and the
    fp32 join lands after both the in-flight kernel and the partial's H2D in
    plain stream order.
    """
    if x.shape[0] > _MAX_COEXEC_ROWS:
        return None
    capacity = int(getattr(provider, "capacity", 0))
    if not capacity:
        return None

    # One D2H staging pass; the host copies are also what the CPU GEMMs eat.
    ids_host = topk_ids.cpu()
    # Counted on the host in one pass, dropping the -1 skip markers rather
    # than clamping them: clamp_min(0) folded every padded slot into expert
    # 0's count, which then sorted it as the hottest candidate and pushed a
    # genuinely cold expert 0 back onto the H2D path this mode exists to
    # avoid.
    counts: dict[int, int] = {}
    for row in ids_host.tolist():
        for e in row:
            if e >= 0:
                counts[e] = counts.get(e, 0) + 1
    unique_ids = sorted(counts)
    if len(unique_ids) <= capacity:
        return None

    ram_capacity = int(getattr(provider, "ram_capacity", 0))
    max_cpu = (
        len(unique_ids)
        if getattr(provider, "_disk_store", None) is None
        else max(0, ram_capacity - capacity)
    )
    cpu_set = select_cpu_experts(
        unique_ids,
        counts,
        set(getattr(provider, "_lru", {})),
        set(getattr(provider, "_ram_lru", {})),
        capacity,
        max_cpu,
        min_tokens,
    )
    if not cpu_set:
        return None

    # The provider serves as many as its pool can hold without evicting a
    # row this forward still needs -- possibly fewer than asked. Its answer,
    # not the request, is what may be masked out of the GPU path.
    views = provider.cpu_views_for(cpu_set, protect=unique_ids)
    cpu_set = [e for e in cpu_set if e in views]
    if not cpu_set:
        provider.cpu_release(list(views))
        return None

    # Everything from here holds eviction protection on those rows, so every
    # exit -- including a raise inside the GPU path or the host GEMMs -- has
    # to release it. A leaked id is protected forever: both RAM victim scans
    # skip it, and the next forward that needs a victim dies on the scan's
    # bare assert with nothing naming co-execution.
    try:
        x_host = x.cpu()
        weights_host = topk_weights.cpu()

        # Reusable pinned fp32 accumulator: allocating pinned memory per
        # forward would cost more than the GEMMs it carries.
        acc = getattr(provider, "_cpu_coexec_acc", None)
        if acc is None or acc.shape[0] < x.shape[0] or acc.shape[1] != x.shape[1]:
            acc = torch.zeros(
                x.shape[0], x.shape[1], dtype=torch.float32
            ).pin_memory()
            provider._cpu_coexec_acc = acc
        buf = acc[: x.shape[0]]
        buf.zero_()

        # GPU path on the masked ids: kernels launch async and stay in
        # flight while the host computes its half below.
        gpu_out = run_with_expert_cache(
            provider, masked_ids(topk_ids, cpu_set), run
        )

        t0 = time.perf_counter()
        cpu_expert_forward(x_host, ids_host, weights_host, cpu_set, views, buf)
        provider.t_cpu_gemm += time.perf_counter() - t0
        provider.cpu_execs += len(cpu_set)
    finally:
        provider.cpu_release(cpu_set)

    partial = buf.to(gpu_out.device, non_blocking=True)
    return partial.add_(gpu_out).to(gpu_out.dtype)
