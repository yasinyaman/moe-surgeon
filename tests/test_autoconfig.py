# SPDX-License-Identifier: Apache-2.0
"""The probe-once-and-remember path.

Two things decide whether this is useful. The configuration has to follow the
measured rules rather than a plausible-sounding heuristic, and the cache has to be
keyed on what actually changes the answer -- coarsely enough that ordinary jitter
in a free-memory reading does not invalidate it, finely enough that a real change
does. Both are what these tests pin.
"""

from __future__ import annotations

import json

import pytest
from test_budget import _write  # noqa: F401  (fixture writer)

from vllm_moe_surgeon.surgery.autoconfig import (
    Environment,
    autoconfigure,
    decide,
    fingerprint,
)
from vllm_moe_surgeon.surgery.budget import analyze


@pytest.fixture
def checkpoint(tmp_path):
    _write(tmp_path)
    return str(tmp_path)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_MOE_SURGEON_CACHE", str(tmp_path / "cache"))


def _env(vram=8.0, ram=64.0, disk=500.0):
    return Environment(vram_gib=vram, host_ram_gib=ram, disk_free_gib=disk)


def _olmoe_geometry():
    """OLMoE-1B-7B's shape, built directly.

    The tiny fixture checkpoint cannot exercise the space branches: its records are
    small enough that ALIGN padding erases the fp8 saving entirely, so a test written
    against it would assert on a difference that does not exist at that scale. The
    measured numbers in this module are OLMoE's, so the geometry is too.
    """
    from vllm_moe_surgeon.surgery.budget import CheckpointGeometry

    hidden, inter, experts, layers = 2048, 1024, 64, 16
    expert_bytes = 2 * ((2 * inter * hidden) + (hidden * inter))
    return CheckpointGeometry(
        num_experts=experts,
        top_k=8,
        moe_layers=list(range(layers)),
        hidden=hidden,
        intermediate=inter,
        expert_bytes=expert_bytes,
        routed_expert_bytes=expert_bytes * experts * layers,
        shared_expert_bytes=0,
        router_bytes=2 * hidden * experts * layers,
        other_bytes=2 << 30,
        checkpoint_dtype_bytes=2,
        serving_dtype_bytes=2,
    )


def test_capacity_aims_at_what_the_batch_can_reach(checkpoint):
    """The lever, and the rule behind it: cover the per-layer union of the serving
    batch. A batch of N at top_k K cannot reach more than N*K experts in a layer, so
    that -- capped by the expert count -- is the target."""
    g = analyze(checkpoint)
    narrow = decide(g, _env(), store_dir="/s", max_num_seqs=1)
    wide = decide(g, _env(), store_dir="/s", max_num_seqs=64)

    assert narrow.surgeon["expert_cache_size"] <= g.top_k * 1
    assert wide.surgeon["expert_cache_size"] == g.num_experts
    assert any("route to at most" in line for line in narrow.why)


def test_ram_cache_covers_every_expert_when_the_host_can_hold_it(checkpoint):
    """`ram_cache` below the expert count turns every eviction into a disk read --
    measured at 1.66x. So it is only given away under pressure, never by default."""
    config = decide(analyze(checkpoint), _env(ram=512.0), store_dir="/s")
    assert config.surgeon["ram_cache"] >= analyze(checkpoint).num_experts
    assert not config.surgeon.get("fp8_store")


def test_fp8_is_reached_for_only_when_space_forces_it():
    """fp8_store is a space mechanism: 1.11x slower and not bit-exact on a non-fp8
    checkpoint. It must never be a default sweetener, only a way to fit."""
    g = _olmoe_geometry()
    layers, experts = g.num_moe_layers, g.num_experts
    bf16_gib = layers * experts * g.record_bytes(fp8=False) / 1024**3
    fp8_gib = layers * experts * g.record_bytes(fp8=True) / 1024**3
    assert fp8_gib < bf16_gib

    roomy = decide(g, _env(vram=40.0, ram=4 * bf16_gib, disk=500.0), store_dir="/s")
    assert "fp8_store" not in roomy.surgeon
    assert roomy.surgeon["ram_cache"] == experts

    # Host holds the store in fp8 but not at full width (the budget is half the host).
    cramped = decide(
        g, _env(vram=40.0, ram=bf16_gib + fp8_gib, disk=500.0), store_dir="/s"
    )
    assert cramped.surgeon.get("fp8_store") is True
    assert cramped.surgeon["ram_cache"] == experts
    assert any("only in fp8" in line for line in cramped.why)


def test_a_disk_too_small_for_the_full_store_reaches_for_fp8_too():
    g = _olmoe_geometry()
    bf16_gib = g.num_moe_layers * g.num_experts * g.record_bytes(fp8=False) / 1024**3
    config = decide(
        g,
        _env(vram=40.0, ram=4 * bf16_gib, disk=bf16_gib * 0.75),
        store_dir="/s",
    )
    assert config.surgeon.get("fp8_store") is True


def test_a_target_that_cannot_fit_says_so_with_numbers(checkpoint):
    with pytest.raises(ValueError, match="does not fit"):
        decide(analyze(checkpoint), _env(vram=0.001), store_dir="/s", kv_reserve_gib=8)


def test_the_fingerprint_ignores_jitter_but_not_a_real_change(checkpoint):
    """The whole point of caching. Free VRAM moves by a few MiB between reads; if
    that invalidated the answer the probe would run every boot, which is what this
    exists to avoid."""
    g = analyze(checkpoint)
    kw = dict(
        checkpoint=checkpoint, store_dir="/s", max_num_seqs=8, kv_reserve_gib=2.0
    )

    base = fingerprint(g, _env(vram=8.00), **kw)
    jitter = fingerprint(g, _env(vram=8.04), **kw)
    real = fingerprint(g, _env(vram=4.00), **kw)

    assert base == jitter, "a few MiB of drift must not invalidate the cache"
    assert base != real, "halving the free VRAM must invalidate it"


def test_the_fingerprint_tracks_the_serving_shape(checkpoint):
    """Batch size changes the union, which changes the capacity, so it has to be
    part of the identity or a cached answer would be served for the wrong shape."""
    g = analyze(checkpoint)
    kw = dict(checkpoint=checkpoint, store_dir="/s", kv_reserve_gib=2.0)
    assert fingerprint(g, _env(), max_num_seqs=8, **kw) != fingerprint(
        g, _env(), max_num_seqs=64, **kw
    )


def test_second_call_is_served_from_cache_and_refresh_bypasses_it(checkpoint):
    first = autoconfigure(checkpoint, store_dir="/s", vram_gib=8.0)
    assert first.cached is False

    second = autoconfigure(checkpoint, store_dir="/s", vram_gib=8.0)
    assert second.cached is True
    assert second.surgeon == first.surgeon
    assert second.fingerprint == first.fingerprint

    refreshed = autoconfigure(checkpoint, store_dir="/s", vram_gib=8.0, refresh=True)
    assert refreshed.cached is False
    assert refreshed.surgeon == first.surgeon


def test_the_emitted_config_is_what_the_runtime_reads(checkpoint):
    """A recipe, not a suggestion: the dict must survive the JSON round trip the
    command line imposes and come back out of `read_config_from` unchanged."""
    from vllm_moe_surgeon.compat.runtime import read_config_from

    config = autoconfigure(checkpoint, store_dir="/tmp/store", vram_gib=8.0)
    argv = config.serve_args("some-model")
    payload = json.loads(argv[argv.index("--additional-config") + 1])

    class _Cfg:
        additional_config = payload

    runtime_config = read_config_from(_Cfg())
    assert runtime_config.enabled
    assert runtime_config.expert_cache_size == config.surgeon["expert_cache_size"]
    assert runtime_config.ram_cache == config.surgeon["ram_cache"]
    assert runtime_config.store_dir == "/tmp/store"


def test_an_unmeasurable_resource_is_named_not_defaulted_to_zero(monkeypatch):
    """A probe that silently reports zero would size the tier for a machine with no
    memory. The caller has to be able to tell "measured small" from "unknown".

    Forced rather than assumed: the first version asserted "no CUDA here", which is
    true on the laptop and false on the GPU box, so the test failed on exactly the
    machine most worth running it on."""
    import torch

    from vllm_moe_surgeon.surgery import autoconfig

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    env = autoconfig.probe("/nonexistent-path-for-probe")
    assert env.vram_gib == 0.0
    assert any("vram" in u for u in env.unknown)


def test_an_existing_store_is_not_counted_against_free_disk(tmp_path, checkpoint):
    """The second-boot trap. The store's own bytes reduce statvfs free space, so a
    probe that reads free space naively concludes after every build that the store
    no longer fits -- and either refuses or silently flips to fp8, which rebuilds
    the store slower and no longer bit-exact. probe() must credit the store's bytes
    back, so "can the store live here" gets the same answer on every boot."""
    from vllm_moe_surgeon.surgery.autoconfig import probe

    store = tmp_path / "store"
    store.mkdir()
    empty = probe(str(store)).disk_free_gib

    (store / "layer0.experts").write_bytes(b"\0" * (64 << 20))
    populated = probe(str(store)).disk_free_gib

    # Within a rounding error, the 64 MiB inside the store came back as "free".
    assert abs(populated - empty) < 0.01


def test_the_recipe_carries_the_batch_bound_it_was_sized_for(checkpoint):
    """Capacity covers the union a batch of max_num_seqs can reach. A boot that
    does not enforce that bound can route to more experts than there are slots
    right after the tool printed "no step should have to re-fetch" -- so the bound
    is part of the recipe, not an annotation."""
    config = autoconfigure(checkpoint, store_dir="/s", vram_gib=8.0, max_num_seqs=4)
    argv = config.serve_args("m")
    assert "--max-num-seqs" in argv
    assert argv[argv.index("--max-num-seqs") + 1] == "4"


def test_fp8_is_never_reached_for_on_a_one_byte_checkpoint():
    """On an already-1-byte checkpoint the fp8 record is *larger* -- it adds row
    scales to the same payload -- so flipping to it under pressure approves a store
    that fits even less than the one just measured not to."""
    g = _olmoe_geometry()
    g.checkpoint_dtype_bytes = 1
    g.serving_dtype_bytes = 1
    assert g.record_bytes(fp8=True) >= g.record_bytes(fp8=False)

    store_gib = (
        g.num_moe_layers * g.num_experts * g.record_bytes(fp8=False) / 1024**3
    )
    # Disk pressure that the old "assume fp8 halves it" logic would have accepted.
    with pytest.raises(ValueError, match="does not fit|is free at"):
        decide(
            g,
            _env(vram=40.0, ram=512.0, disk=store_gib * 0.6),
            store_dir="/s",
        )

    # RAM pressure: the fallback must size with the record that is actually
    # smaller, and must not claim "even in fp8" when fp8 was never applicable.
    tight = decide(
        g,
        _env(vram=40.0, ram=store_gib, disk=500.0),
        store_dir="/s",
    )
    assert "fp8_store" not in tight.surgeon
    assert not any("even in fp8" in w for w in tight.warnings)
