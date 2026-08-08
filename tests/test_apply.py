# SPDX-License-Identifier: Apache-2.0
"""Applying a plan to a checkpoint.

The failure mode worth most of this file: a router row left at its old index
routes to a *different expert* than the one it was trained for. The checkpoint
loads, inference runs, nothing raises, and the model is quietly wrong. So the
renumbering is checked by identity, not by shape.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from vllm_moe_surgeon.surgery import Plan
from vllm_moe_surgeon.surgery.align import apply_permutation
from vllm_moe_surgeon.surgery.apply import apply_plan, derive_surgery, rewrite_config
from vllm_moe_surgeon.surgery.descriptors import CheckpointIndex
from vllm_moe_surgeon.surgery.plan import ExpertPlacement

H = 16
INTER = 12
NUM_EXPERTS = 6
NUM_LAYERS = 2
TOP_K = 2


def _distinct_expert(layer: int, expert: int):
    """Weights whose values identify the expert, so a mix-up is detectable."""
    tag = float(layer * 100 + expert + 1)
    gate = torch.full((INTER, H), tag, dtype=torch.float32)
    up = torch.full((INTER, H), tag + 0.25, dtype=torch.float32)
    down = torch.full((H, INTER), tag + 0.5, dtype=torch.float32)
    return gate, up, down


def _make_checkpoint(root, *, dtype=torch.float32, num_experts=NUM_EXPERTS):
    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.arange(8 * H, dtype=torch.float32).reshape(
            8, H
        ),
        "model.norm.weight": torch.ones(H),
        "lm_head.weight": torch.full((8, H), 3.0),
    }
    for layer in range(NUM_LAYERS):
        for expert in range(num_experts):
            gate, up, down = _distinct_expert(layer, expert)
            base = f"model.layers.{layer}.mlp.experts.{expert}"
            tensors[f"{base}.gate_proj.weight"] = gate
            tensors[f"{base}.up_proj.weight"] = up
            tensors[f"{base}.down_proj.weight"] = down
        # Router row e is filled with (e + 1) * 10 so identity is checkable.
        rows = [
            torch.full((H,), float((expert + 1) * 10)) for expert in range(num_experts)
        ]
        tensors[f"model.layers.{layer}.mlp.gate.weight"] = torch.stack(rows)
        tensors[f"model.layers.{layer}.self_attn.q_proj.weight"] = torch.full(
            (H, H), 7.0
        )

    root.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.to(dtype) for k, v in tensors.items()},
        str(root / "model.safetensors"),
        metadata={"format": "pt"},
    )
    config = {
        "architectures": ["OlmoeForCausalLM"],
        "model_type": "olmoe",
        "num_hidden_layers": NUM_LAYERS,
        "hidden_size": H,
        "num_experts": num_experts,
        "num_experts_per_tok": TOP_K,
    }
    with open(root / "config.json", "w") as f:
        json.dump(config, f)
    with open(root / "tokenizer_config.json", "w") as f:
        json.dump({"model_max_length": 128}, f)
    return config


def _plan(keep_by_layer, *, merges=None, model="test/model"):
    """Build a plan directly, bypassing the profile-driven engine."""
    placements = []
    merges = merges or {}
    for layer, keep in keep_by_layer.items():
        for expert in range(NUM_EXPERTS):
            if expert in keep:
                placements.append(
                    ExpertPlacement(
                        layer=layer,
                        expert=expert,
                        action="merge_into_core",
                        tokens=1000 - expert,
                        share=0.1,
                    )
                )
            else:
                target = merges.get((layer, expert))
                placements.append(
                    ExpertPlacement(
                        layer=layer,
                        expert=expert,
                        action="drop",
                        tokens=10,
                        share=0.01,
                        merge_target=target,
                    )
                )
    return Plan(
        model=model,
        revision="r1",
        budget={"core_experts": 0},
        placements=placements,
        # These tests exercise the surgery mechanics, not the gate, so they stand
        # in for a plan that has already been measured and approved.
        gate=_passing_gate(),
    )


def _passing_gate():
    return {
        "passed": True,
        "baseline_perplexity": 9.71,
        "perplexity": 12.13,
        "ratio": 1.25,
        "max_ratio": 1.3,
        "reason": "1.25x <= 1.3x",
    }


# ------------------------------------------------------------------- deriving


def test_derive_surgery_renumbers_survivors_contiguously():
    plan = _plan({0: {1, 3, 5}, 1: {0, 2, 4}})
    surgery = derive_surgery(plan)
    assert surgery[0].remap == {1: 0, 3: 1, 5: 2}
    assert surgery[1].remap == {0: 0, 2: 1, 4: 2}
    assert surgery[0].deleted == [0, 2, 4]


def test_rewrite_config_updates_count_and_clamps_top_k():
    config = {"num_experts": NUM_EXPERTS, "num_experts_per_tok": 4}
    surgery = derive_surgery(_plan({0: {0, 1}, 1: {0, 1}}))
    updated = rewrite_config(config, surgery)
    assert updated["num_experts"] == 2
    # Routing to 4 of 2 experts is impossible; it has to shrink.
    assert updated["num_experts_per_tok"] == 2


def test_rewrite_config_leaves_top_k_alone_when_it_still_fits():
    config = {"num_experts": NUM_EXPERTS, "num_experts_per_tok": 2}
    surgery = derive_surgery(_plan({0: {0, 1, 2, 3}, 1: {0, 1, 2, 3}}))
    assert rewrite_config(config, surgery)["num_experts_per_tok"] == 2


def test_rewrite_config_refuses_uneven_layers():
    """One config field cannot express a per-layer expert count."""
    surgery = derive_surgery(_plan({0: {0, 1}, 1: {0, 1, 2}}))
    with pytest.raises(ValueError, match="different expert counts"):
        rewrite_config({"num_experts": NUM_EXPERTS}, surgery)


def test_rewrite_config_handles_the_alias_key():
    config = {"num_local_experts": NUM_EXPERTS, "num_experts_per_tok": 2}
    surgery = derive_surgery(_plan({0: {0, 1}, 1: {0, 1}}))
    assert rewrite_config(config, surgery)["num_local_experts"] == 2


def test_rewrite_config_without_any_count_key_raises():
    surgery = derive_surgery(_plan({0: {0, 1}, 1: {0, 1}}))
    with pytest.raises(ValueError, match="cannot record the new expert count"):
        rewrite_config({"hidden_size": H}, surgery)


# ---------------------------------------------------------------- drop-only


def test_drop_only_produces_a_loadable_renumbered_checkpoint(tmp_path):
    source = tmp_path / "src"
    out = tmp_path / "out"
    _make_checkpoint(source)

    keep = {1, 3, 5}
    manifest = apply_plan(_plan({0: keep, 1: keep}), str(source), str(out))

    assert manifest["experts_before"] == NUM_EXPERTS
    assert manifest["experts_after"] == 3
    assert manifest["merges_applied"] == 0

    with open(out / "config.json") as f:
        assert json.load(f)["num_experts"] == 3

    index = CheckpointIndex.open(str(out))
    assert index.expert_ids(0) == [0, 1, 2]

    # New id 0 must be the *old expert 1* -- checked by its tag value.
    for new_id, old_id in enumerate(sorted(keep)):
        expected = float(0 * 100 + old_id + 1)
        gate = index.read(index.tensor_name(0, new_id, "gate_proj"))
        assert gate[0, 0] == pytest.approx(expected), (
            f"new expert {new_id} should be old expert {old_id}"
        )


def test_router_rows_follow_the_expert_renumbering(tmp_path):
    """The silent-failure case: a stale router row routes to the wrong expert."""
    source = tmp_path / "src"
    out = tmp_path / "out"
    _make_checkpoint(source)

    keep = sorted({1, 3, 5})
    apply_plan(_plan({0: set(keep), 1: set(keep)}), str(source), str(out))

    index = CheckpointIndex.open(str(out))
    router = index.read(index.router_name(0))
    assert router.shape == (3, H)
    for new_id, old_id in enumerate(keep):
        # Source row e was filled with (e + 1) * 10.
        assert router[new_id, 0] == pytest.approx((old_id + 1) * 10.0), (
            f"router row {new_id} must carry old expert {old_id}'s row"
        )


def test_non_expert_tensors_pass_through_untouched(tmp_path):
    source = tmp_path / "src"
    out = tmp_path / "out"
    _make_checkpoint(source)
    apply_plan(_plan({0: {0, 1}, 1: {0, 1}}), str(source), str(out))

    src_index = CheckpointIndex.open(str(source))
    out_index = CheckpointIndex.open(str(out))
    for name in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"):
        np.testing.assert_array_equal(
            src_index.read(name), out_index.read(name)
        )
    assert "model.layers.0.self_attn.q_proj.weight" in out_index.weight_map


def test_dropped_experts_are_gone_from_the_output(tmp_path):
    source = tmp_path / "src"
    out = tmp_path / "out"
    _make_checkpoint(source)
    apply_plan(_plan({0: {0, 1}, 1: {0, 1}}), str(source), str(out))

    out_index = CheckpointIndex.open(str(out))
    assert out_index.expert_ids(0) == [0, 1]
    assert not any(".experts.5." in name for name in out_index.weight_map)


def test_tokenizer_files_are_carried_over(tmp_path):
    source = tmp_path / "src"
    out = tmp_path / "out"
    _make_checkpoint(source)
    apply_plan(_plan({0: {0, 1}, 1: {0, 1}}), str(source), str(out))
    assert (out / "tokenizer_config.json").exists()


def test_dtype_is_preserved(tmp_path):
    source = tmp_path / "src"
    out = tmp_path / "out"
    _make_checkpoint(source, dtype=torch.bfloat16)
    apply_plan(_plan({0: {0, 1}, 1: {0, 1}}), str(source), str(out))

    from safetensors import safe_open

    with safe_open(str(out / "model-00001-of-00001.safetensors"), framework="pt") as f:
        tensor = f.get_tensor("model.layers.0.mlp.experts.0.gate_proj.weight")
    assert tensor.dtype == torch.bfloat16


# -------------------------------------------------------------------- merging


def test_merge_of_a_permuted_duplicate_recovers_the_original(tmp_path):
    """End to end through the checkpoint: alignment survives the file round trip."""
    source = tmp_path / "src"
    out = tmp_path / "out"
    root = source
    root.mkdir(parents=True)

    rng = np.random.default_rng(3)
    gate = torch.from_numpy(rng.standard_normal((INTER, H)).astype(np.float32))
    up = torch.from_numpy(rng.standard_normal((INTER, H)).astype(np.float32))
    down = torch.from_numpy(rng.standard_normal((H, INTER)).astype(np.float32))
    perm = rng.permutation(INTER)
    p_gate, p_up, p_down = apply_permutation(
        gate.numpy(), up.numpy(), down.numpy(), perm
    )

    tensors = {
        "model.norm.weight": torch.ones(H),
        "model.layers.0.mlp.experts.0.gate_proj.weight": gate,
        "model.layers.0.mlp.experts.0.up_proj.weight": up,
        "model.layers.0.mlp.experts.0.down_proj.weight": down,
        # Expert 1 is expert 0 with its neurons shuffled: same function.
        # np.ascontiguousarray because down[:, perm] is a strided view, and
        # safetensors refuses to write non-contiguous tensors.
        "model.layers.0.mlp.experts.1.gate_proj.weight": torch.from_numpy(
            np.ascontiguousarray(p_gate)
        ),
        "model.layers.0.mlp.experts.1.up_proj.weight": torch.from_numpy(
            np.ascontiguousarray(p_up)
        ),
        "model.layers.0.mlp.experts.1.down_proj.weight": torch.from_numpy(
            np.ascontiguousarray(p_down)
        ),
        "model.layers.0.mlp.gate.weight": torch.stack(
            [torch.full((H,), 10.0), torch.full((H,), 20.0)]
        ),
    }
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    with open(root / "config.json", "w") as f:
        json.dump(
            {
                "num_hidden_layers": 1,
                "num_experts": 2,
                "num_experts_per_tok": 2,
                "hidden_size": H,
            },
            f,
        )

    plan = Plan(
        model="test/dup",
        revision=None,
        budget={},
        placements=[
            ExpertPlacement(0, 0, "merge_into_core", tokens=100, share=0.9),
            ExpertPlacement(0, 1, "drop", tokens=100, share=0.1, merge_target=0),
        ],
        gate=_passing_gate(),
    )
    manifest = apply_plan(plan, str(source), str(out))
    assert manifest["merges_applied"] == 1
    assert manifest["experts_after"] == 1
    assert "least-squares refit" in manifest["router_rewrite"]

    index = CheckpointIndex.open(str(out))
    merged_gate = index.read(index.tensor_name(0, 0, "gate_proj"))
    # Equal-weight merge of an expert with a permuted copy of itself is that
    # expert. Any residual is misalignment.
    np.testing.assert_allclose(merged_gate, gate.numpy(), rtol=1e-5, atol=1e-5)
    merged_down = index.read(index.tensor_name(0, 0, "down_proj"))
    np.testing.assert_allclose(merged_down, down.numpy(), rtol=1e-5, atol=1e-5)


def test_merge_router_row_is_the_weighted_mean(tmp_path):
    source = tmp_path / "src"
    out = tmp_path / "out"
    _make_checkpoint(source)

    # Keep 0..3; expert 4 merges into 0, expert 5 is deleted outright.
    plan = _plan({0: {0, 1, 2, 3}, 1: {0, 1, 2, 3}}, merges={(0, 4): 0, (1, 4): 0})
    manifest = apply_plan(plan, str(source), str(out))
    assert manifest["merges_applied"] == 2
    assert manifest["layers"]["0"]["merged_away"] == {"0": [4]}
    assert manifest["layers"]["0"]["deleted"] == [5]

    index = CheckpointIndex.open(str(out))
    router = index.read(index.router_name(0))
    # target tokens 1000, donor tokens 10; rows are 10.0 and 50.0.
    expected = (1000 * 10.0 + 10 * 50.0) / 1010
    assert router[0, 0] == pytest.approx(expected, rel=1e-4)


def test_manifest_records_provenance_and_the_router_caveat(tmp_path):
    source = tmp_path / "src"
    out = tmp_path / "out"
    _make_checkpoint(source)
    manifest = apply_plan(_plan({0: {0, 1}, 1: {0, 1}}), str(source), str(out))

    assert manifest["base_model"] == "test/model"
    assert manifest["base_revision"] == "r1"
    assert manifest["top_k_before"] == TOP_K
    assert manifest["layers"]["0"]["remap"] == {"0": 0, "1": 1}
    with open(out / "surgeon_manifest.json") as f:
        assert json.load(f)["experts_after"] == 2


def test_sharding_splits_and_indexes_correctly(tmp_path):
    source = tmp_path / "src"
    out = tmp_path / "out"
    _make_checkpoint(source)
    # A byte budget far below one tensor forces a shard per flush.
    apply_plan(
        _plan({0: {0, 1}, 1: {0, 1}}), str(source), str(out), shard_bytes=1
    )

    with open(out / "model.safetensors.index.json") as f:
        index = json.load(f)
    shards = set(index["weight_map"].values())
    assert len(shards) > 1
    for shard in shards:
        assert (out / shard).exists()
        assert "-of-" in shard
    # Every tensor is reachable and correct through the index.
    reopened = CheckpointIndex.open(str(out))
    assert reopened.expert_ids(0) == [0, 1]
    assert reopened.read("model.norm.weight").shape == (H,)


# ------------------------------------------------------------------- the gate


def test_an_ungated_deletion_is_refused(tmp_path):
    """Deletion is the one irreversible step, and its cost is measurable first.

    So paying it unmeasured has to be an explicit choice, not the default.
    """
    source = tmp_path / "src"
    out = tmp_path / "out"
    _make_checkpoint(source)

    plan = _plan({0: {0, 1}, 1: {0, 1}})
    plan.gate = None
    with pytest.raises(RuntimeError, match="not passed the quality gate"):
        apply_plan(plan, str(source), str(out))


def test_a_failing_gate_is_refused_and_says_why(tmp_path):
    source = tmp_path / "src"
    out = tmp_path / "out"
    _make_checkpoint(source)

    plan = _plan({0: {0, 1}, 1: {0, 1}})
    plan.gate = {"passed": False, "reason": "2.80x > 1.3x", "ratio": 2.8}
    with pytest.raises(RuntimeError, match="2.80x > 1.3x"):
        apply_plan(plan, str(source), str(out))


def test_the_gate_can_be_waived_explicitly(tmp_path):
    source = tmp_path / "src"
    out = tmp_path / "out"
    _make_checkpoint(source)

    plan = _plan({0: {0, 1}, 1: {0, 1}})
    plan.gate = None
    manifest = apply_plan(plan, str(source), str(out), require_gate=False)
    assert manifest["experts_after"] == 2
    assert manifest["gate"] is None


def test_a_plan_that_deletes_nothing_needs_no_gate(tmp_path):
    """Re-placing experts between core and disk loses nothing to measure."""
    from vllm_moe_surgeon.surgery.plan import deletes_anything

    source = tmp_path / "src"
    out = tmp_path / "out"
    _make_checkpoint(source)

    placements = []
    for layer in (0, 1):
        for expert in range(NUM_EXPERTS):
            action = "merge_into_core" if expert < 2 else "keep_on_disk"
            placements.append(
                ExpertPlacement(layer, expert, action, tokens=100, share=0.1)
            )
    plan = Plan(model="m", revision=None, budget={}, placements=placements)
    assert not deletes_anything(plan)

    manifest = apply_plan(plan, str(source), str(out))
    assert manifest["experts_after"] == NUM_EXPERTS


def test_the_gate_verdict_travels_into_the_manifest(tmp_path):
    """Provenance: the artifact records what measurement authorised it."""
    source = tmp_path / "src"
    out = tmp_path / "out"
    _make_checkpoint(source)

    manifest = apply_plan(_plan({0: {0, 1}, 1: {0, 1}}), str(source), str(out))
    assert manifest["gate"]["passed"] is True
    assert manifest["gate"]["ratio"] == 1.25


def test_gate_survives_a_plan_roundtrip(tmp_path):
    from vllm_moe_surgeon.surgery import load_plan
    from vllm_moe_surgeon.surgery.plan import gate_passed

    plan = _plan({0: {0, 1}, 1: {0, 1}})
    path = str(tmp_path / "plan.json")
    plan.save(path)
    back = load_plan(path)
    assert gate_passed(back)
    assert back.gate["ratio"] == 1.25
