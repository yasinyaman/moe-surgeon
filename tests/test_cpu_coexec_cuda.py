# SPDX-License-Identifier: Apache-2.0
"""CPU expert co-execution: the CUDA half.

The offline suite (tests/test_cpu_exec.py) pins the selection policy, the
masking join and the host kernel against a dense reference. What needs a
device is the provider integration: host views over real RAM rows, the
protected set under eviction pressure, and the fp32 join with a live
slot-indexed GPU path.
"""

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

from test_expert_cache import (  # noqa: E402
    _make_disk_provider,
    _make_provider,
    _topk,
)

from vllm_moe_surgeon.store.expert_cpu_exec import (  # noqa: E402
    run_with_cpu_coexec,
)

HIDDEN, INTERMEDIATE = 16, 32


def test_cpu_views_read_protect_and_release():
    provider, store, w13, w2 = _make_disk_provider(
        num_experts=8, capacity=2, ram_capacity=4
    )

    views = provider.cpu_views_for([5, 6], protect=[])
    assert torch.equal(views[5][0], w13[5])
    assert torch.equal(views[5][1], w2[5])
    assert provider._cpu_protect == {5, 6}

    # Eviction pressure: fill the remaining RAM slots and demand more. The
    # protected rows must survive both victim scans.
    provider.prepare(_topk([0, 1]), [0, 1])
    provider.prepare(_topk([2, 3]), [2, 3])
    provider.prepare(_topk([7, 0]), [7, 0])
    assert 5 in provider._ram_lru and 6 in provider._ram_lru
    assert torch.equal(views[5][0], w13[5]), "protected row was overwritten"

    provider.cpu_release([5, 6])
    assert provider._cpu_protect == set()
    # Released rows are victims again: new traffic may now evict them.
    provider.prepare(_topk([1, 4]), [1, 4])
    provider.prepare(_topk([2, 7]), [2, 7])


def test_coexec_joins_the_host_partial_into_the_gpu_output():
    """Full-DRAM provider, capacity below the union: the co-exec output must
    equal the dense fp32 reference over every (token, expert) pair."""
    provider, w13, w2, _ = _make_provider(num_experts=8, capacity=3)
    # Warm two residents so the selection has a real GPU-resident set.
    provider.prepare(_topk([0, 1]))

    T, K = 4, 3
    torch.manual_seed(0)
    x = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor(
        [[0, 2, 5], [1, 3, 6], [0, 4, 7], [1, 2, 5]],
        dtype=torch.int32, device="cuda",
    )
    weights = torch.rand(T, K, dtype=torch.float32, device="cuda")

    def _expert(xr, e, w13_t, w2_t):
        h = xr @ w13_t[e].t()
        gate, up = h[:, :INTERMEDIATE], h[:, INTERMEDIATE:]
        act = torch.nn.functional.silu(gate.float()).to(xr.dtype) * up
        return act @ w2_t[e].t()

    w13_d, w2_d = w13.cuda(), w2.cuda()

    def reference(id_tensor):
        out = torch.zeros(T, HIDDEN, dtype=torch.float32, device="cuda")
        for t in range(T):
            for k in range(K):
                e = int(id_tensor[t, k])
                if e < 0:
                    continue
                y = _expert(x[t : t + 1], e, w13_d, w2_d)
                out[t] += weights[t, k] * y[0].float()
        return out

    def run(result, rows, include_shared):
        """A faithful stand-in for the fused kernel: slot-indexed weights,
        expert_map authority, gate-weighted fp32 sum, model-dtype output."""
        out = torch.zeros(T, HIDDEN, dtype=torch.float32, device="cuda")
        emap = result.expert_map
        for t in range(T):
            for k in range(K):
                e = int(ids[t, k])
                if e < 0 or int(emap[e]) < 0:
                    continue
                slot = int(emap[e])
                h = x[t : t + 1] @ result.w1[slot].t()
                gate, up = h[:, :INTERMEDIATE], h[:, INTERMEDIATE:]
                act = torch.nn.functional.silu(gate.float()).to(x.dtype) * up
                y = act @ result.w2[slot].t()
                out[t] += weights[t, k] * y[0].float()
        return out.to(torch.bfloat16)

    got = run_with_cpu_coexec(provider, x, ids, weights, run, min_tokens=1)
    assert got is not None, "union exceeds capacity; co-exec must engage"
    # This mode is declared not-bit-exact (host reduction order + one extra
    # bf16 rounding at the join), so the assertion is about the JOIN: a
    # missing or double-counted (token, expert) pair moves the output by a
    # whole expert contribution, orders of magnitude above rounding noise.
    ref = reference(ids)
    rel = (got.float() - ref).norm() / ref.norm()
    assert rel < 0.01, f"relative error {rel:.4f}: a pair went missing"
    assert provider.cpu_execs > 0
    assert provider._cpu_protect == set(), "views must be released"


def test_gpu_bookkeeping_is_untouched_by_the_host_path():
    provider, store, w13, w2 = _make_disk_provider(
        num_experts=8, capacity=2, ram_capacity=6
    )
    provider.prepare(_topk([0, 1]), [0, 1])  # residents
    lru_before = dict(provider._lru)
    misses_before = provider.misses

    T, K = 2, 4
    x = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor(
        [[0, 2, 3, 4], [1, 2, 5, 3]], dtype=torch.int32, device="cuda"
    )
    weights = torch.rand(T, K, dtype=torch.float32, device="cuda")

    def run(result, rows, include_shared):
        return torch.zeros(T, HIDDEN, dtype=torch.bfloat16, device="cuda")

    got = run_with_cpu_coexec(provider, x, ids, weights, run, min_tokens=1)
    assert got is not None

    # The host served every miss: GPU tier state and counters unmoved.
    assert dict(provider._lru) == lru_before
    assert provider.misses == misses_before
    assert provider.cpu_execs == 4  # experts 2, 3, 4, 5
    assert provider.t_cpu_gemm > 0
    # ... and the RAM tier recorded the use: the served ids are resident.
    for e in (2, 3, 4, 5):
        assert e in provider._ram_lru
