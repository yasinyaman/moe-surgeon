# SPDX-License-Identifier: Apache-2.0
"""The tiered artifact: an offline-built store, and the residency hint.

The store's byte layout is a correctness requirement, not a convention. ``w13``
rows ``[0, I)`` must be ``gate_proj`` and ``[I, 2I)`` must be ``up_proj``,
matching the in-tree writer, because a store with those swapped loads without
complaint and computes the wrong function. So the layout is asserted directly.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_moe_surgeon import hot_experts
from vllm_moe_surgeon.store.expert_disk_store import DiskExpertStore
from vllm_moe_surgeon.surgery import Plan
from vllm_moe_surgeon.surgery.descriptors import CheckpointIndex
from vllm_moe_surgeon.surgery.plan import ExpertPlacement
from vllm_moe_surgeon.surgery.tiered import (
    LAYER_NAME_TEMPLATE,
    build_layer_store,
    build_store,
    store_key,
    store_path,
)

H = 8
INTER = 6
NUM_EXPERTS = 3


def _make_checkpoint(root, *, dtype=torch.bfloat16, num_experts=NUM_EXPERTS):
    tensors = {"model.norm.weight": torch.ones(H, dtype=dtype)}
    rng = torch.Generator().manual_seed(5)
    truth = {}
    for expert in range(num_experts):
        gate = torch.randn(INTER, H, generator=rng).to(dtype)
        up = torch.randn(INTER, H, generator=rng).to(dtype)
        down = torch.randn(H, INTER, generator=rng).to(dtype)
        truth[expert] = (gate, up, down)
        base = f"model.layers.0.mlp.experts.{expert}"
        tensors[f"{base}.gate_proj.weight"] = gate
        tensors[f"{base}.up_proj.weight"] = up
        tensors[f"{base}.down_proj.weight"] = down
    tensors["model.layers.0.mlp.gate.weight"] = torch.randn(
        num_experts, H, generator=rng
    ).to(dtype)
    root.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    with open(root / "config.json", "w") as f:
        json.dump(
            {
                "num_hidden_layers": 1,
                "num_experts": num_experts,
                "num_experts_per_tok": 2,
                "hidden_size": H,
            },
            f,
        )
    return truth


def _read_record(store, expert):
    row = torch.empty(store.record_stride, dtype=torch.uint8)
    store.read_record(expert, row)
    return {name: store.field_view(row, name).clone() for name in store.fields}


# ------------------------------------------------------------------ store keys


def test_store_key_matches_the_in_tree_writer():
    """The serving side derives this name; a mismatch means a rebuilt store."""
    assert store_key("model.layers.7.mlp.experts") == "model_layers_7_mlp_experts"
    assert store_path("/d", "model.layers.7.mlp.experts").endswith(
        "model_layers_7_mlp_experts.experts"
    )


def test_layer_name_template_matches_vllms_naming():
    assert LAYER_NAME_TEMPLATE.format(layer=3) == "model.layers.3.mlp.experts"


# ---------------------------------------------------------------- store build


def test_unquantized_store_roundtrips_and_stacks_w13_correctly(tmp_path):
    """gate must land in w13[0:I] and up in w13[I:2I], not the other way."""
    source = tmp_path / "src"
    truth = _make_checkpoint(source, dtype=torch.float32)
    index = CheckpointIndex.open(str(source))

    store = build_layer_store(
        index, 0, str(tmp_path / "store"), model_id="m", revision="r", fp8=False
    )
    assert store.is_complete
    assert store.num_experts == NUM_EXPERTS

    for expert, (gate, up, down) in truth.items():
        fields = _read_record(store, expert)
        w13 = fields["w13"]
        torch.testing.assert_close(w13[:INTER], gate, rtol=0, atol=0)
        torch.testing.assert_close(w13[INTER:], up, rtol=0, atol=0)
        torch.testing.assert_close(fields["w2"], down, rtol=0, atol=0)


def test_gate_and_up_are_distinguishable_so_the_test_is_meaningful(tmp_path):
    """Guard the guard: if gate == up, the stacking assertion proves nothing."""
    source = tmp_path / "src"
    truth = _make_checkpoint(source, dtype=torch.float32)
    gate, up, _ = truth[0]
    assert not torch.allclose(gate, up)


def test_fp8_store_carries_scales_and_reconstructs(tmp_path):
    source = tmp_path / "src"
    truth = _make_checkpoint(source, dtype=torch.bfloat16)
    index = CheckpointIndex.open(str(source))

    store = build_layer_store(
        index, 0, str(tmp_path / "store"), model_id="m", revision="r", fp8=True
    )
    assert set(store.fields) == {"w13", "w2", "w13_qs", "w2_qs"}

    for expert, (gate, up, down) in truth.items():
        fields = _read_record(store, expert)
        recon13 = fields["w13"].float() * fields["w13_qs"].float().unsqueeze(-1)
        # Row scales are per output row, so gate and up reconstruct separately.
        torch.testing.assert_close(
            recon13[:INTER], gate.float(), rtol=0.08, atol=0.05
        )
        torch.testing.assert_close(
            recon13[INTER:], up.float(), rtol=0.08, atol=0.05
        )
        recon2 = fields["w2"].float() * fields["w2_qs"].float().unsqueeze(-1)
        torch.testing.assert_close(recon2, down.float(), rtol=0.08, atol=0.05)


def test_identity_fingerprint_is_recorded(tmp_path):
    source = tmp_path / "src"
    _make_checkpoint(source)
    index = CheckpointIndex.open(str(source))
    build_layer_store(
        index,
        0,
        str(tmp_path / "store"),
        model_id="org/model",
        revision="abc123",
        fp8=True,
    )
    directory = str(tmp_path / "store")
    sidecar = store_path(directory, "model.layers.0.mlp.experts") + ".json"
    with open(sidecar) as f:
        payload = json.load(f)
    assert payload["identity"] == {
        "model": "org/model",
        "revision": "abc123",
        "layer": "model.layers.0.mlp.experts",
    }
    assert payload["quant"] == "fp8e4m3-row"


def test_rebuilding_with_the_same_identity_reuses(tmp_path):
    source = tmp_path / "src"
    _make_checkpoint(source)
    index = CheckpointIndex.open(str(source))
    directory = str(tmp_path / "store")
    build_layer_store(index, 0, directory, model_id="m", revision="r", fp8=True)

    path = store_path(directory, "model.layers.0.mlp.experts")
    import os

    mtime = os.stat(path).st_mtime_ns
    again = build_layer_store(index, 0, directory, model_id="m", revision="r", fp8=True)
    assert again.is_complete
    assert os.stat(path).st_mtime_ns == mtime


def test_fp8_refuses_a_float32_checkpoint(tmp_path):
    source = tmp_path / "src"
    _make_checkpoint(source, dtype=torch.float32)
    index = CheckpointIndex.open(str(source))
    with pytest.raises(ValueError, match="target bf16/fp16"):
        build_layer_store(
            index, 0, str(tmp_path / "store"), model_id="m", revision="r", fp8=True
        )


def test_noncontiguous_expert_ids_are_refused(tmp_path):
    """A pruned checkpoint must be renumbered before a store is built."""
    source = tmp_path / "src"
    _make_checkpoint(source)
    index = CheckpointIndex.open(str(source))
    # Forge a gap by hiding expert 1 from the index.
    broken = CheckpointIndex(
        index.root,
        {k: v for k, v in index.weight_map.items() if ".experts.1." not in k},
    )
    with pytest.raises(ValueError, match="not contiguous"):
        build_layer_store(
            broken, 0, str(tmp_path / "store"), model_id="m", revision="r", fp8=True
        )


def test_build_store_summarizes_every_layer(tmp_path):
    source = tmp_path / "src"
    _make_checkpoint(source)
    summary = build_store(
        str(source), str(tmp_path / "store"), [0], model_id="m", fp8=True
    )
    assert summary["layers"] == [0]
    assert summary["fp8"] is True
    assert summary["bytes"] > 0


def test_store_is_readable_by_the_lifted_reader(tmp_path):
    """The point of matching the layout: the shipped reader opens it unchanged."""
    source = tmp_path / "src"
    _make_checkpoint(source)
    index = CheckpointIndex.open(str(source))
    directory = str(tmp_path / "store")
    build_layer_store(index, 0, directory, model_id="m", revision="r", fp8=True)

    path = store_path(directory, "model.layers.0.mlp.experts")
    reopened = DiskExpertStore.create_for_streaming(
        path,
        NUM_EXPERTS,
        [
            ("w13", (2 * INTER, H), torch.float8_e4m3fn),
            ("w2", (H, INTER), torch.float8_e4m3fn),
            ("w13_qs", (2 * INTER,), torch.float32),
            ("w2_qs", (H,), torch.float32),
        ],
        identity={"model": "m", "revision": "r", "layer": "model.layers.0.mlp.experts"},
        quant="fp8e4m3-row",
    )
    assert reopened.is_complete, "an identical spec must be recognised as complete"


# --------------------------------------------------------------- hot experts


def _plan_with_shares():
    placements = [
        ExpertPlacement(0, 0, "merge_into_core", tokens=500, share=0.5),
        ExpertPlacement(0, 1, "merge_into_core", tokens=300, share=0.3),
        ExpertPlacement(0, 2, "keep_on_disk", tokens=150, share=0.15),
        ExpertPlacement(0, 3, "drop", tokens=50, share=0.05),
    ]
    return Plan(
        model="test/m", revision="r1", budget={"core_experts": 2}, placements=placements
    )


def test_priors_track_share_not_rank():
    """A 50%-share expert must outrank a 30% one by 50/30, not by one step."""
    hints = hot_experts.from_plan(_plan_with_shares(), scale=8.0)
    priors = hints.for_layer(0)
    assert priors[0] == pytest.approx(4.0)
    assert priors[1] == pytest.approx(2.4)
    assert priors[0] / priors[1] == pytest.approx(0.5 / 0.3)


def test_dropped_experts_get_no_prior():
    priors = hot_experts.from_plan(_plan_with_shares()).for_layer(0)
    assert 3 not in priors, "a deleted expert cannot be resident"
    assert 2 in priors, "a disk expert can still earn a slot"


def test_importance_share_zero_is_not_bumped_to_raw_share():
    """The falsy trap, fixed per-plan: in a plan that carries importance shares, a
    rank-1 share of exactly 0.0 stays 0.0 rather than falling back to raw token
    share and outranking a genuine contributor."""
    placements = [
        ExpertPlacement(
            0, 0, "merge_into_core", tokens=500, share=0.5, importance_share=0.7
        ),
        ExpertPlacement(
            0, 1, "keep_on_disk", tokens=300, share=0.3, importance_share=0.0
        ),
    ]
    plan = Plan(model="m", revision="r", budget={}, placements=placements)
    priors = hot_experts.from_plan(plan, scale=8.0).for_layer(0)
    assert priors[0] == pytest.approx(5.6)  # 0.7 * 8
    assert priors[1] == pytest.approx(0.0)  # rank-1 share 0, not 0.3 * 8


def test_remap_keys_priors_in_the_applied_id_space():
    """Tiering the pruned checkpoint renumbers survivors; a remap rekeys the priors
    and drops deleted experts."""
    placements = [
        ExpertPlacement(
            0, 1, "merge_into_core", tokens=500, share=0.5, importance_share=0.6
        ),
        ExpertPlacement(
            0, 3, "keep_on_disk", tokens=300, share=0.3, importance_share=0.4
        ),
        ExpertPlacement(0, 0, "drop", tokens=10, share=0.01),
    ]
    plan = Plan(model="m", revision="r", budget={}, placements=placements)
    priors = hot_experts.from_plan(
        plan, scale=8.0, remap={0: {1: 0, 3: 1}}
    ).for_layer(0)
    assert set(priors) == {0, 1}  # new contiguous ids
    assert priors[0] == pytest.approx(4.8)  # old expert 1: 0.6 * 8
    assert priors[1] == pytest.approx(3.2)  # old expert 3: 0.4 * 8


def test_core_is_ordered_hottest_first():
    hints = hot_experts.from_plan(_plan_with_shares())
    assert hints.core[0] == [0, 1]


def test_prior_scale_lands_in_the_advisory_range():
    """Worth a few cache hits, not permanent dominance.

    EWMAPolicy's frequency term grows ~1.0 per event, so a prior in the single
    digits decides the cold-start window and is then overtaken. A prior of, say,
    1000 would pin an expert forever, which is a bug rather than a stronger hint.
    """
    priors = hot_experts.from_plan(_plan_with_shares()).for_layer(0)
    assert 1.0 <= max(priors.values()) <= 20.0


def test_layer_name_translation():
    assert hot_experts.layer_index_from_name("model.layers.7.mlp.experts") == 7
    assert hot_experts.layer_index_from_name("model.mlp.experts") is None


def test_for_layer_name_bridges_plan_and_provider_keying():
    hints = hot_experts.from_plan(_plan_with_shares())
    assert hints.for_layer_name("model.layers.0.mlp.experts") == hints.for_layer(0)
    assert hints.for_layer_name("nonsense") == {}


def test_save_load_roundtrip(tmp_path):
    hints = hot_experts.from_plan(_plan_with_shares())
    path = str(tmp_path / "hot_experts.json")
    hints.save(path)
    back = hot_experts.load(path)
    assert back.model == "test/m"
    assert back.core == hints.core
    assert back.for_layer(0) == hints.for_layer(0)
    # Keys must come back as ints, not the strings JSON stored.
    assert all(isinstance(k, int) for k in back.priors)
    assert all(isinstance(k, int) for k in back.for_layer(0))


def test_load_rejects_a_foreign_version(tmp_path):
    path = tmp_path / "hot.json"
    path.write_text(json.dumps({"manifest_version": 99, "priors": {}, "core": {}}))
    with pytest.raises(ValueError, match="manifest_version"):
        hot_experts.load(str(path))


# ---------------------------------------------------------------- the seam


def test_seeding_fills_the_ewma_prior_seam():
    """The seam that has been an empty dict since the prototype."""
    from vllm_moe_surgeon.store.expert_policy import EWMAPolicy

    policy = EWMAPolicy(num_experts=4)
    assert policy.prior == {}, "unseeded is the documented starting state"

    priors = hot_experts.from_plan(_plan_with_shares()).for_layer(0)
    assert hot_experts.seed_policy(policy, priors) is True
    assert policy.prior[0] == pytest.approx(4.0)

    # And it actually moves the score, which is the only reason to seed it.
    assert policy.score(0, freq=0, last=0, clock=0) > policy.score(
        2, freq=0, last=0, clock=0
    )


def test_seeding_reports_failure_on_a_policy_without_priors():
    """LFRU has no prior table; the caller must learn that, not assume."""
    from vllm_moe_surgeon.store.expert_policy import LFRUPolicy

    assert hot_experts.seed_policy(LFRUPolicy(num_experts=4), {0: 1.0}) is False


def test_seeded_prior_is_overtaken_by_real_traffic():
    """Advisory means yielding. A hint that never loses is a pin."""
    from vllm_moe_surgeon.store.expert_policy import EWMAPolicy

    policy = EWMAPolicy(num_experts=4)
    hot_experts.seed_policy(policy, {0: 4.0, 1: 0.1})
    assert policy.score(0, 0, 0, 0) > policy.score(1, 0, 0, 0)

    # Expert 1 turns out to be the busy one in this run.
    for _ in range(50):
        policy.on_hit(1)
    assert policy.score(1, 0, 0, 0) > policy.score(0, 0, 0, 0), (
        "observed traffic must be able to overrule the boot-time hint"
    )


def test_quantizing_from_bf16_or_widened_float32_is_identical(tmp_path):
    """The store builder widens bf16 to float32 before quantizing.

    That has to be a no-op for the records, or an offline-built store would
    disagree with one vLLM built from the same weights. Widening bf16 is exact,
    so both the row scales and the fp8 codes must come out bit-identical --
    checked here rather than assumed, because a low-bit scale difference shows up
    as sparse one-code differences that are easy to dismiss as noise.
    """
    from vllm_moe_surgeon.store.expert_disk_store import quantize_rowwise_fp8

    rng = torch.Generator().manual_seed(19)
    raw = torch.randn(INTER, H, generator=rng).to(torch.bfloat16)
    widened = raw.float()
    assert torch.equal(widened, raw.float())

    q_a, s_a = quantize_rowwise_fp8(raw)
    q_b, s_b = quantize_rowwise_fp8(widened)
    assert torch.equal(s_a, s_b), "row scales must not depend on the source dtype"
    assert torch.equal(q_a.view(torch.uint8), q_b.view(torch.uint8))
