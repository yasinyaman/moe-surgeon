# SPDX-License-Identifier: Apache-2.0
"""CPU expert co-execution: the offline half.

Everything here runs with no CUDA and no vLLM: the selection policy, the
masking join, and the host kernel against a dense reference. The concurrent
speedup itself is CUDA-only and measured by the gates
(bench/l5_cpu_coexec_gate.py, bench/l6_cpu_coexec_gate.py) and the laptop A/B.
"""

import pytest

torch = pytest.importorskip("torch")

from vllm_moe_surgeon.store.expert_cpu_exec import (  # noqa: E402
    cpu_expert_forward,
    masked_ids,
    select_cpu_experts,
)

HID, INT = 64, 32


def _weights(n_experts, seed=0):
    g = torch.Generator().manual_seed(seed)
    w13 = torch.randn(n_experts, 2 * INT, HID, generator=g)
    w2 = torch.randn(n_experts, HID, INT, generator=g)
    return w13, w2


def _dense_moe(x, ids, weights, w13, w2):
    """The reference: every (token, expert) pair, gate-weighted fp32 sum."""
    out = torch.zeros(x.shape[0], HID, dtype=torch.float32)
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            if e < 0:
                continue
            h = x[t : t + 1] @ w13[e].t()
            gate, up = h[:, :INT], h[:, INT:]
            act = torch.nn.functional.silu(gate.float()).to(x.dtype) * up
            y = act @ w2[e].t()
            out[t] += float(weights[t, k]) * y[0].float()
    return out


# ------------------------------------------------------------- selection


def test_selection_is_a_noop_when_the_union_fits():
    assert (
        select_cpu_experts(
            [0, 1, 2], {0: 4, 1: 2, 2: 2}, set(), set(),
            capacity=4, max_cpu=8, min_tokens=1,
        )
        == []
    )


def test_selection_takes_every_miss_ram_first_then_coldest():
    """f=1.0 is the measured winner on the laptop: every GPU-miss goes to
    the CPU, RAM-resident ones first (no disk read), then coldest first."""
    unique = [0, 1, 2, 3, 4, 5]
    counts = {0: 5, 1: 4, 2: 3, 3: 2, 4: 1, 5: 1}
    picked = select_cpu_experts(
        unique, counts, gpu_resident={0, 1}, ram_resident={2, 4},
        capacity=2, max_cpu=8, min_tokens=1,
    )
    # All four misses; 4 (ram, T=1) before 2 (ram, T=3); disk-only after.
    assert picked == [4, 2, 5, 3]


def test_selection_never_excludes_single_token_experts_by_default():
    """The first L5 run selected ZERO experts in every rep because T>=2 was
    an exclusion and the cold tail is ~all T=1. Pad, never exclude."""
    unique = list(range(6))
    counts = dict.fromkeys(unique, 1)
    picked = select_cpu_experts(
        unique, counts, gpu_resident={0}, ram_resident=set(),
        capacity=1, max_cpu=8, min_tokens=1,
    )
    assert len(picked) == 5


def test_selection_respects_the_ram_headroom_cap():
    unique = list(range(10))
    counts = dict.fromkeys(unique, 2)
    picked = select_cpu_experts(
        unique, counts, gpu_resident=set(), ram_resident=set(),
        capacity=4, max_cpu=3, min_tokens=1,
    )
    assert len(picked) == 3


def test_a_min_tokens_knob_still_filters_when_asked():
    unique = [0, 1, 2]
    counts = {0: 1, 1: 2, 2: 3}
    picked = select_cpu_experts(
        unique, counts, gpu_resident=set(), ram_resident=set(),
        capacity=1, max_cpu=8, min_tokens=2,
    )
    assert picked == [1, 2]


# --------------------------------------------------------------- masking


def test_masked_ids_hides_only_the_cpu_set():
    ids = torch.tensor([[0, 1, 2], [3, -1, 1]])
    m = masked_ids(ids, [1, 3])
    assert m.tolist() == [[0, -1, 2], [-1, -1, -1]]
    # ... and the original is untouched.
    assert ids.tolist() == [[0, 1, 2], [3, -1, 1]]


def test_masking_nothing_returns_the_same_tensor():
    ids = torch.tensor([[0, 1]])
    assert masked_ids(ids, []) is ids


# ------------------------------------------------------------ the kernel


def test_cpu_forward_matches_a_dense_reference():
    torch.manual_seed(1)
    E, T, K = 6, 4, 3
    w13, w2 = _weights(E)
    x = torch.randn(T, HID)
    ids = torch.tensor([[0, 1, 2], [1, 3, 4], [2, 4, 5], [0, 3, 5]])
    weights = torch.rand(T, K)

    ref = _dense_moe(x, ids, weights, w13, w2)
    out = torch.zeros(T, HID, dtype=torch.float32)
    views = {e: (w13[e], w2[e]) for e in range(E)}
    pairs = cpu_expert_forward(x, ids, weights, list(range(E)), views, out)
    assert pairs == T * K
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_single_token_expert_is_padded_not_mismultiplied():
    """The pad row's output must contribute nothing."""
    torch.manual_seed(2)
    w13, w2 = _weights(1)
    x = torch.randn(1, HID)
    ids = torch.tensor([[0]])
    weights = torch.ones(1, 1)

    ref = _dense_moe(x, ids, weights, w13, w2)
    out = torch.zeros(1, HID, dtype=torch.float32)
    cpu_expert_forward(x, ids, weights, [0], {0: (w13[0], w2[0])}, out)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_gpu_plus_cpu_partition_covers_every_pair_once():
    """The additivity join itself: reference(all) == reference(masked) +
    cpu_partial. This is the property the runtime's fp32 join relies on."""
    torch.manual_seed(3)
    E, T, K = 8, 5, 4
    w13, w2 = _weights(E)
    x = torch.randn(T, HID)
    ids = torch.randint(0, E, (T, K))
    weights = torch.rand(T, K)
    cpu_set = [1, 4, 6]

    full = _dense_moe(x, ids, weights, w13, w2)
    gpu_part = _dense_moe(x, masked_ids(ids, cpu_set), weights, w13, w2)
    cpu_part = torch.zeros(T, HID, dtype=torch.float32)
    views = {e: (w13[e], w2[e]) for e in cpu_set}
    cpu_expert_forward(x, ids, weights, cpu_set, views, cpu_part)
    assert torch.allclose(gpu_part + cpu_part, full, atol=1e-4, rtol=1e-4)


def test_an_expert_with_no_rows_is_skipped():
    out = torch.zeros(2, HID, dtype=torch.float32)
    w13, w2 = _weights(1)
    pairs = cpu_expert_forward(
        torch.randn(2, HID), torch.tensor([[5], [5]]), torch.ones(2, 1),
        [0], {0: (w13[0], w2[0])}, out,
    )
    assert pairs == 0
    assert out.abs().sum() == 0


def test_a_token_routed_twice_to_one_expert_keeps_both_terms():
    """`out[rows] += ...` lowers to a NON-accumulating index_put_, so a
    duplicate row index kept only the last write and silently dropped a whole
    gate-weighted expert term. Distinct top-k never produces the duplicate; a
    grouped router or a global->local id collapse does."""
    torch.manual_seed(4)
    w13, w2 = _weights(1)
    x = torch.randn(1, HID)
    ids = torch.tensor([[0, 0]])       # one token, same expert twice
    weights = torch.tensor([[0.25, 0.75]])

    ref = _dense_moe(x, ids, weights, w13, w2)
    out = torch.zeros(1, HID, dtype=torch.float32)
    pairs = cpu_expert_forward(x, ids, weights, [0], {0: (w13[0], w2[0])}, out)
    assert pairs == 2
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
