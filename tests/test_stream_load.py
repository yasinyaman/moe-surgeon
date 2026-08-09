# SPDX-License-Identifier: Apache-2.0
"""Streaming load: does the intercepted loader produce a correct store?

These run without vLLM and without a GPU. ``StreamLoader`` takes a real
``DiskExpertStore`` and real tensors, so what is under test is the thing that matters:
the bytes that end up in the records, and whether the halves land the right way round.
A store whose gate and up are swapped loads without complaint and computes a different
function, so "it ran" proves nothing here.
"""

from __future__ import annotations

import pytest
import torch

from vllm_moe_surgeon.compat.stream_load import StreamLoader

E, INTER, HIDDEN = 4, 6, 8


def _loader(tmp_path, *, fp8=False, num_experts=E, dtype=torch.bfloat16):
    return StreamLoader.open(
        store_dir=str(tmp_path),
        layer_name="model.layers.0.mlp.experts",
        num_experts=num_experts,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        params_dtype=dtype,
        fp8=fp8,
        model="test/model",
        revision=None,
    )


def _read_back(store, expert_id):
    """(w13, w2) as the provider reconstructs them: record, field views, scales.

    For an fp8 store the field view holds raw e4m3 values; the weight is
    ``q.float() * scale.unsqueeze(-1)``. Reading the view alone gives numbers off by
    the row scale -- which is how the first version of this test "failed" by 400x.
    """
    row = torch.zeros(store.record_stride, dtype=torch.uint8)
    store.read_record(expert_id, row)
    w13 = store.field_view(row, "w13").clone()
    w2 = store.field_view(row, "w2").clone()
    if "w13_qs" in store.fields:
        w13 = w13.float() * store.field_view(row, "w13_qs").float().unsqueeze(-1)
        w2 = w2.float() * store.field_view(row, "w2_qs").float().unsqueeze(-1)
    return w13, w2


def _experts(num_experts=E, dtype=torch.bfloat16):
    """(gate, up, down) per expert, all distinct so a mix-up cannot pass."""
    generator = torch.Generator().manual_seed(7)
    return [
        (
            torch.randn(INTER, HIDDEN, generator=generator).to(dtype),
            torch.randn(INTER, HIDDEN, generator=generator).to(dtype),
            torch.randn(HIDDEN, INTER, generator=generator).to(dtype),
        )
        for _ in range(num_experts)
    ]


def _feed(streamer, experts, order=("w1", "w3", "w2")):
    for expert_id, (gate, up, down) in enumerate(experts):
        by_shard = {"w1": gate, "w3": up, "w2": down}
        for shard in order:
            streamer.accept(shard, by_shard[shard], expert_id)


def test_a_streamed_store_holds_the_experts_it_was_fed(tmp_path):
    experts = _experts()
    streamer = _loader(tmp_path)
    _feed(streamer, experts)
    store = streamer.finalize()

    assert streamer.records_written == E
    assert store.is_complete
    for expert_id, (gate, up, down) in enumerate(experts):
        w13, w2 = _read_back(store, expert_id)
        torch.testing.assert_close(w13[:INTER], gate, rtol=0, atol=0)
        torch.testing.assert_close(w13[INTER:], up, rtol=0, atol=0)
        torch.testing.assert_close(w2, down, rtol=0, atol=0)
    # Guard the guard: identical halves would make the assertions above vacuous.
    assert not torch.equal(experts[0][0], experts[0][1])


def test_the_halves_go_in_the_order_vllm_reads_them(tmp_path):
    """w1 is the FIRST half of w13 and w3 the second. Swapped is silently wrong."""
    experts = _experts()
    streamer = _loader(tmp_path)
    _feed(streamer, experts)
    store = streamer.finalize()

    w13, _ = _read_back(store, 0)
    gate, up, _ = experts[0]
    assert torch.equal(w13[:INTER], gate)
    assert not torch.equal(w13[:INTER], up)


def test_shard_arrival_order_does_not_matter(tmp_path):
    """Checkpoints do not promise an order, so the record must not depend on one."""
    experts = _experts()
    a = _loader(tmp_path / "a")
    _feed(a, experts, order=("w1", "w3", "w2"))
    store_a = a.finalize()

    b = _loader(tmp_path / "b")
    _feed(b, experts, order=("w2", "w3", "w1"))
    store_b = b.finalize()

    for expert_id in range(E):
        w13_a, w2_a = _read_back(store_a, expert_id)
        w13_b, w2_b = _read_back(store_b, expert_id)
        torch.testing.assert_close(w13_a, w13_b, rtol=0, atol=0)
        torch.testing.assert_close(w2_a, w2_b, rtol=0, atol=0)


def test_interleaved_experts_cost_records_not_the_model(tmp_path):
    """The saving is the point: staging holds only what is in flight."""
    experts = _experts()
    streamer = _loader(tmp_path)

    # Every expert's first shard before any expert's last: worst-case interleaving.
    for expert_id, (gate, _, _) in enumerate(experts):
        streamer.accept("w1", gate, expert_id)
    assert len(streamer._staging) == E, "all four are in flight"
    assert streamer.records_written == 0

    for expert_id, (_, up, down) in enumerate(experts):
        streamer.accept("w3", up, expert_id)
        streamer.accept("w2", down, expert_id)
        # Each record is written and dropped the moment it completes.
        assert expert_id not in streamer._staging

    assert streamer._staging == {}
    assert streamer.records_written == E
    streamer.finalize()


def test_an_incomplete_expert_refuses_to_seal(tmp_path):
    """A zeroed record is silent: the model loads and is wrong where it routes there."""
    experts = _experts()
    streamer = _loader(tmp_path)
    _feed(streamer, experts[:-1])
    gate, _, down = experts[-1]
    streamer.accept("w1", gate, E - 1)
    streamer.accept("w2", down, E - 1)  # w3 never arrives

    with pytest.raises(RuntimeError, match="incomplete experts"):
        streamer.finalize()
    # The message has to name what is missing, or the operator cannot act on it.
    try:
        streamer.finalize()
    except RuntimeError as exc:
        assert "'w3'" in str(exc) or "w3" in str(exc)


def test_a_shape_mismatch_is_refused_by_name(tmp_path):
    streamer = _loader(tmp_path)
    wrong = torch.randn(INTER + 1, HIDDEN).to(torch.bfloat16)
    with pytest.raises(ValueError, match="shape mismatch"):
        streamer.accept("w1", wrong, 0)


def test_fp8_records_round_trip_within_quantisation_error(tmp_path):
    experts = _experts()
    streamer = _loader(tmp_path, fp8=True)
    _feed(streamer, experts)
    store = streamer.finalize()

    assert store._quant == "fp8e4m3-row"
    for expert_id, (gate, up, down) in enumerate(experts):
        w13, w2 = _read_back(store, expert_id)
        # Row-wise e4m3: a few percent, not bit-exact. Loose enough to pass, tight
        # enough that a swapped half or a dropped scale fails -- both were checked by
        # breaking them.
        torch.testing.assert_close(w13[:INTER], gate.float(), rtol=0.1, atol=0.05)
        torch.testing.assert_close(w13[INTER:], up.float(), rtol=0.1, atol=0.05)
        torch.testing.assert_close(w2, down.float(), rtol=0.1, atol=0.05)


def test_fp8_is_refused_for_a_dtype_it_cannot_encode(tmp_path):
    with pytest.raises(ValueError, match="bf16/fp16"):
        _loader(tmp_path, fp8=True, dtype=torch.float32)


def test_a_matching_store_is_reused_instead_of_restreamed(tmp_path):
    """Second boot should not redo the first boot's work."""
    experts = _experts()
    first = _loader(tmp_path)
    _feed(first, experts)
    first.finalize()

    second = _loader(tmp_path)
    assert second.store.is_complete, "a fingerprint match must be reused"
    # Feeding a complete store is a no-op rather than an error: the loader still runs.
    _feed(second, experts)
    assert second.records_written == 0
    second.finalize()


# ------------------------------------------------------------------ the loader hook


def test_the_wrapper_intercepts_expert_shards_and_passes_the_rest_through(tmp_path):
    experts = _experts()
    streamer = _loader(tmp_path)
    seen = []

    def inner(**kwargs):
        seen.append(kwargs.get("shard_id"))
        return "delegated"

    loader = streamer.wrap(inner)

    for expert_id, (gate, up, down) in enumerate(experts):
        for shard, tensor in (("w1", gate), ("w3", up), ("w2", down)):
            assert loader(
                param=None,
                loaded_weight=tensor,
                weight_name=f"experts.{expert_id}.{shard}.weight",
                shard_id=shard,
                expert_id=expert_id,
                return_success=True,
            ) is True
    assert seen == [], "expert shards must not reach the inner loader"
    assert streamer.records_written == E

    # Anything else is vLLM's business, not this port's.
    assert loader(param=None, loaded_weight=torch.zeros(2), shard_id="w1_scale") == (
        "delegated"
    )
    assert seen == ["w1_scale"]
    streamer.finalize()


def test_a_bias_shard_is_delegated_even_though_its_shard_id_matches(tmp_path):
    """The tier refuses biased layers elsewhere; here the name is what disambiguates."""
    streamer = _loader(tmp_path)
    delegated = []
    loader = streamer.wrap(lambda **kw: delegated.append(kw.get("weight_name")))

    loader(
        param=None,
        loaded_weight=torch.zeros(INTER),
        weight_name="experts.w1_bias",
        shard_id="w1",
        expert_id=0,
        return_success=True,
    )
    assert delegated == ["experts.w1_bias"]
    assert streamer.records_written == 0


def test_return_success_is_honoured_both_ways(tmp_path):
    """The caller distinguishes loaded from skipped only when it asked to."""
    experts = _experts()
    streamer = _loader(tmp_path)
    loader = streamer.wrap(lambda **kw: None)
    gate = experts[0][0]

    assert loader(
        param=None, loaded_weight=gate, shard_id="w1", expert_id=0, return_success=True
    ) is True
    assert loader(
        param=None, loaded_weight=gate, shard_id="w1", expert_id=1, return_success=False
    ) is None


def test_a_positional_call_is_delegated_rather_than_guessed(tmp_path):
    """The contract read from vLLM is all-keyword. A positional call is not it.

    Delegating is the safe response: guessing an argument order would write a shard
    into the wrong half of the wrong expert, and nothing downstream would object.
    """
    streamer = _loader(tmp_path)
    got = []
    loader = streamer.wrap(lambda *a, **kw: got.append(("pos", len(a))))

    loader(
        object(), shard_id="w1", expert_id=0, loaded_weight=torch.zeros(INTER, HIDDEN)
    )
    assert got == [("pos", 1)]
    assert streamer.records_written == 0


def test_a_negative_local_id_is_skipped(tmp_path):
    """Expert-parallel maps non-local experts to -1. Refused elsewhere; safe here."""
    streamer = _loader(tmp_path)
    streamer._map_expert_id = lambda expert_id: -1
    streamer.accept("w1", torch.zeros(INTER, HIDDEN).to(torch.bfloat16), 3)
    assert streamer.records_written == 0
    assert streamer._staging == {}
