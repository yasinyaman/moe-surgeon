# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for the lifted disk store and cache policies.

The ported suite in ``test_expert_cache.py`` is gated on CUDA end to end, so on
a machine without a device it validates nothing -- including whether the import
rewrite that moved these modules out of the vLLM tree actually worked. These
tests close that gap: the on-disk record format, the fingerprint reuse rule, the
fp8 row quantizer and the policy seam are all device-independent, so they can
guard the lift from a laptop.

They are also the tests the offline surgery pipeline depends on, since the
pipeline writes stores on a host that may have no GPU at all.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from vllm_moe_surgeon.store import (
    DiskExpertStore,
    make_policy,
    quantize_rowwise_fp8,
)

NUM_EXPERTS = 6
HIDDEN = 8
INTERMEDIATE = 12
IDENTITY = {"model": "test/tiny-moe", "revision": "abc123", "layer": "layers.0.mlp"}


def _weights(dtype=torch.bfloat16):
    torch.manual_seed(0)
    w13 = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=dtype)
    w2 = torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, dtype=dtype)
    return w13, w2


def _read_back(store: DiskExpertStore, expert_id: int) -> dict[str, torch.Tensor]:
    row = torch.empty(store.record_stride, dtype=torch.uint8)
    store.read_record(expert_id, row)
    return {name: store.field_view(row, name).clone() for name in store.fields}


def test_record_roundtrip_is_byte_exact(tmp_path):
    """Every expert reads back exactly what was written."""
    w13, w2 = _weights()
    path = str(tmp_path / "layer0.experts")
    store = DiskExpertStore.build(path, w13, w2, identity=IDENTITY)

    assert store.is_complete
    assert store.num_experts == NUM_EXPERTS
    # Records are ALIGN-padded so any pool row starts page-aligned.
    assert store.record_stride % 4096 == 0

    for e in range(NUM_EXPERTS):
        got = _read_back(store, e)
        torch.testing.assert_close(got["w13"], w13[e], rtol=0, atol=0)
        torch.testing.assert_close(got["w2"], w2[e], rtol=0, atol=0)


def test_record_offsets_do_not_overlap(tmp_path):
    """Field layout is contiguous and inside the stride."""
    w13, w2 = _weights()
    store = DiskExpertStore.build(
        str(tmp_path / "l.experts"), w13, w2, identity=IDENTITY
    )
    fields = sorted(store.fields.values(), key=lambda f: f.offset)
    cursor = 0
    for f in fields:
        assert f.offset >= cursor, f"{f.name} overlaps the previous field"
        cursor = f.offset + f.nbytes
    assert cursor <= store.record_stride


def test_matching_identity_reuses_and_mismatch_rebuilds(tmp_path):
    """The fingerprint is what stops two models sharing a store."""
    w13, w2 = _weights()
    path = str(tmp_path / "l.experts")

    DiskExpertStore.build(path, w13, w2, identity=IDENTITY)
    with open(path + ".json") as f:
        first = json.load(f)
    mtime = os.stat(path).st_mtime_ns

    # Same identity -> reuse, file untouched.
    DiskExpertStore.build(path, w13, w2, identity=IDENTITY)
    assert os.stat(path).st_mtime_ns == mtime

    # Different revision of the same architecture -> must NOT be reused.
    other = dict(IDENTITY, revision="def456")
    DiskExpertStore.build(path, w13, w2, identity=other)
    with open(path + ".json") as f:
        second = json.load(f)
    assert second != first
    assert second["identity"]["revision"] == "def456"


def test_fp8_store_reconstructs_within_e4m3_resolution(tmp_path):
    """An fp8 store round-trips to the original within quantizer error."""
    w13, w2 = _weights()
    path = str(tmp_path / "l.experts")
    store = DiskExpertStore.build(
        path, w13, w2, identity=IDENTITY, quantize_fp8=True
    )

    with open(path + ".json") as f:
        assert json.load(f)["quant"] == "fp8e4m3-row"
    assert {"w13", "w2", "w13_qs", "w2_qs"} <= set(store.fields)

    for e in range(NUM_EXPERTS):
        got = _read_back(store, e)
        recon = got["w13"].float() * got["w13_qs"].float().unsqueeze(-1)
        # e4m3 has 3 mantissa bits, so ~6% relative error is the honest bound.
        torch.testing.assert_close(recon, w13[e].float(), rtol=0.07, atol=0.02)


def test_fp8_zero_rows_get_unit_scale():
    """All-zero rows must not divide by zero; scale is 1 and recon is exact."""
    t = torch.zeros(3, 4)
    t[1] = torch.tensor([1.0, -2.0, 0.5, 0.25])
    q, scale = quantize_rowwise_fp8(t)
    assert scale[0] == 1.0 and scale[2] == 1.0
    recon = q.float() * scale.unsqueeze(-1)
    torch.testing.assert_close(recon[0], t[0], rtol=0, atol=0)
    torch.testing.assert_close(recon[2], t[2], rtol=0, atol=0)


def test_streaming_build_matches_materialized_build(tmp_path):
    """The load-path store and the offline store produce identical records."""
    w13, w2 = _weights()

    whole = DiskExpertStore.build(
        str(tmp_path / "whole.experts"), w13, w2, identity=IDENTITY
    )

    specs = [
        ("w13", tuple(w13.shape[1:]), w13.dtype),
        ("w2", tuple(w2.shape[1:]), w2.dtype),
    ]
    streamed = DiskExpertStore.create_for_streaming(
        str(tmp_path / "streamed.experts"), NUM_EXPERTS, specs, identity=IDENTITY
    )
    assert streamed.record_stride == whole.record_stride

    for e in range(NUM_EXPERTS):
        row = torch.zeros(streamed.record_stride, dtype=torch.uint8)
        streamed.field_view(row, "w13").copy_(w13[e])
        streamed.field_view(row, "w2").copy_(w2[e])
        streamed.write_record(e, row)
    streamed.finalize()
    assert streamed.is_complete

    for e in range(NUM_EXPERTS):
        torch.testing.assert_close(
            _read_back(streamed, e)["w13"], w13[e], rtol=0, atol=0
        )


def test_streaming_finalize_refuses_incomplete_store(tmp_path):
    """A partially-written store must fail loudly, not serve zeros."""
    w13, w2 = _weights()
    specs = [
        ("w13", tuple(w13.shape[1:]), w13.dtype),
        ("w2", tuple(w2.shape[1:]), w2.dtype),
    ]
    store = DiskExpertStore.create_for_streaming(
        str(tmp_path / "partial.experts"), NUM_EXPERTS, specs, identity=IDENTITY
    )
    row = torch.zeros(store.record_stride, dtype=torch.uint8)
    store.write_record(0, row)

    with pytest.raises(RuntimeError, match="streaming load ended with"):
        store.finalize()


def test_policy_env_selection_and_prior_seam(monkeypatch):
    """Policy choice is read lazily, and the prior seam starts empty."""
    monkeypatch.delenv("VLLM_MOE_CACHE_POLICY", raising=False)
    assert type(make_policy(8)).__name__ == "LFRUPolicy"

    monkeypatch.setenv("VLLM_MOE_CACHE_POLICY", "ewma")
    policy = make_policy(8)
    assert type(policy).__name__ == "EWMAPolicy"
    # Faz 4 seeds this from hot_experts.json; today nothing populates it, and a
    # test that asserts the emptiness is how we notice when that changes.
    assert policy.prior == {}

    monkeypatch.setenv("VLLM_MOE_CACHE_POLICY", "nonsense")
    with pytest.raises(ValueError, match="expected one of"):
        make_policy(8)


def test_env_overrides_beat_environment(monkeypatch):
    """--additional-config values must win over the process environment."""
    from vllm_moe_surgeon import env

    monkeypatch.setenv("VLLM_MOE_RAM_CACHE", "8")
    assert env.VLLM_MOE_RAM_CACHE == 8
    try:
        env.set_overrides({"VLLM_MOE_RAM_CACHE": 32})
        assert env.VLLM_MOE_RAM_CACHE == 32
    finally:
        env.set_overrides({})
    assert env.VLLM_MOE_RAM_CACHE == 8


def test_buffered_mode_reads_the_same_bytes_without_o_direct(tmp_path, monkeypatch):
    """The page-cache path must be selectable, and identical on the bytes.

    O_DIRECT is a residency/accounting choice, not a format one -- records stay
    4096-padded either way -- so the SAME store file has to serve both modes.
    That is what makes a buffered-vs-O_DIRECT A/B honest: no rebuild between
    arms, so nothing but the read path differs.
    """
    w13, w2 = _weights()
    path = str(tmp_path / "layer0.experts")
    store = DiskExpertStore.build(path, w13, w2, identity=IDENTITY)
    direct = [_read_back(store, e) for e in range(NUM_EXPERTS)]
    store.close()

    monkeypatch.setenv("VLLM_MOE_DISK_BUFFERED", "1")
    buffered_store = DiskExpertStore.build(path, w13, w2, identity=IDENTITY)
    for e in range(NUM_EXPERTS):
        got = _read_back(buffered_store, e)
        torch.testing.assert_close(got["w13"], direct[e]["w13"], rtol=0, atol=0)
        torch.testing.assert_close(got["w2"], direct[e]["w2"], rtol=0, atol=0)
    # The flag has to actually reach the descriptor; a knob that silently keeps
    # O_DIRECT would make every arm of the A/B measure the same thing. Only
    # checkable where O_DIRECT exists -- on macOS both modes are buffered
    # already, so the assertion would pass without testing anything.
    if hasattr(os, "O_DIRECT"):
        assert store._o_direct is True
        assert buffered_store._o_direct is False
