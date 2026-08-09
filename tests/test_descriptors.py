# SPDX-License-Identifier: Apache-2.0
"""The permutation-invariance claim, tested directly.

Everything about merging rests on one property: an FFN expert is unchanged by
permuting its intermediate neurons, so any similarity used to justify a merge
must be blind to that permutation. If these tests pass, the measure is sound. If
they fail, every merge decision the pipeline makes is arbitrary.
"""

from __future__ import annotations

import numpy as np
import pytest

from vllm_moe_surgeon.surgery.descriptors import (
    _as_float32,
    subspace_basis,
    subspace_similarity,
)

H = 32  # hidden
INTER = 24  # intermediate (not `I`: ruff flags it as ambiguous)
RANK = 8


def _rng(seed: int = 0):
    return np.random.default_rng(seed)


def test_identical_experts_have_similarity_one():
    w1 = _rng(1).standard_normal((INTER, H)).astype(np.float32)
    basis = subspace_basis(w1, RANK)
    assert subspace_similarity(basis, basis) == pytest.approx(1.0, abs=1e-5)


def test_permuted_expert_is_identical_the_whole_point():
    """Shuffling the intermediate neurons must not change the fingerprint.

    Entrywise cosine similarity fails this badly, which is exactly why the
    measure is a subspace comparison and not a dot product on flat weights.
    """
    rng = _rng(2)
    w1 = rng.standard_normal((INTER, H)).astype(np.float32)
    perm = rng.permutation(INTER)
    w1_permuted = w1[perm]

    basis = subspace_basis(w1, RANK)
    basis_permuted = subspace_basis(w1_permuted, RANK)
    assert subspace_similarity(basis, basis_permuted) == pytest.approx(1.0, abs=1e-4)

    # The naive alternative, for contrast: flattened cosine similarity collapses.
    flat_cos = float(
        np.dot(w1.ravel(), w1_permuted.ravel())
        / (np.linalg.norm(w1) * np.linalg.norm(w1_permuted))
    )
    assert abs(flat_cos) < 0.35, (
        "flattened cosine should be near zero for a permuted copy -- if it is "
        "not, this test is not demonstrating the hazard it claims to"
    )


def test_independent_experts_are_dissimilar():
    """Two unrelated experts must not look mergeable."""
    rng = _rng(3)
    a = subspace_basis(rng.standard_normal((INTER, H)).astype(np.float32), RANK)
    b = subspace_basis(rng.standard_normal((INTER, H)).astype(np.float32), RANK)
    similarity = subspace_similarity(a, b)
    # Two random rank-8 subspaces of R^32 overlap by roughly rank/H.
    assert similarity < 0.5
    assert similarity == pytest.approx(RANK / H, abs=0.2)


def test_similarity_is_symmetric_and_bounded():
    rng = _rng(4)
    a = subspace_basis(rng.standard_normal((INTER, H)).astype(np.float32), RANK)
    b = subspace_basis(rng.standard_normal((INTER, H)).astype(np.float32), RANK)
    ab = subspace_similarity(a, b)
    assert ab == pytest.approx(subspace_similarity(b, a), abs=1e-6)
    assert 0.0 <= ab <= 1.0


def test_singular_vector_sign_does_not_matter():
    """Singular-vector signs are arbitrary; the measure must ignore them."""
    rng = _rng(5)
    basis = subspace_basis(rng.standard_normal((INTER, H)).astype(np.float32), RANK)
    flipped = basis.copy()
    flipped[:, ::2] *= -1
    assert subspace_similarity(basis, flipped) == pytest.approx(1.0, abs=1e-5)


def test_scaling_an_expert_does_not_change_its_subspace():
    """Row space is scale-invariant, so magnitude is tracked separately."""
    rng = _rng(6)
    w1 = rng.standard_normal((INTER, H)).astype(np.float32)
    a = subspace_basis(w1, RANK)
    b = subspace_basis(w1 * 7.5, RANK)
    assert subspace_similarity(a, b) == pytest.approx(1.0, abs=1e-4)


def test_partially_shared_subspace_lands_in_between():
    """A blend must score between identical and unrelated, not saturate."""
    rng = _rng(7)
    shared = rng.standard_normal((INTER // 2, H)).astype(np.float32)
    a = np.vstack([shared, rng.standard_normal((INTER // 2, H)).astype(np.float32)])
    b = np.vstack([shared, rng.standard_normal((INTER // 2, H)).astype(np.float32)])
    similarity = subspace_similarity(
        subspace_basis(a, RANK), subspace_basis(b, RANK)
    )
    assert 0.2 < similarity < 0.95


def test_rank_is_clamped_to_the_matrix():
    basis = subspace_basis(_rng(8).standard_normal((4, H)).astype(np.float32), 64)
    assert basis.shape == (H, 4)


def test_non_matrix_input_is_rejected():
    with pytest.raises(ValueError, match="2-D matrix"):
        subspace_basis(np.zeros((2, 3, 4), dtype=np.float32), RANK)


def test_bfloat16_view_is_decoded():
    """Checkpoints are bf16; numpy has no bf16, so it arrives as raw 2-byte data.

    Decoding it wrong would silently produce garbage descriptors rather than an
    error, so the conversion is checked against known values.
    """
    values = np.array([1.0, -2.0, 0.5, 0.0], dtype=np.float32)
    # bf16 is the high 16 bits of the float32 pattern.
    as_bf16 = (values.view(np.uint32) >> 16).astype(np.uint16)
    decoded = _as_float32(as_bf16.view(np.uint16))
    np.testing.assert_allclose(decoded, values, rtol=0, atol=0)


# ------------------------------------------------- reading a real checkpoint


def _write_tiny_checkpoint(root, num_layers=1, num_experts=3, dtype=None):
    """A minimal bf16 safetensors checkpoint in the naming real models use."""
    import torch
    from safetensors.torch import save_file

    dtype = dtype or torch.bfloat16
    tensors = {}
    rng = torch.Generator().manual_seed(11)
    for layer in range(num_layers):
        for expert in range(num_experts):
            base = f"model.layers.{layer}.mlp.experts.{expert}"
            tensors[f"{base}.gate_proj.weight"] = torch.randn(
                INTER, H, generator=rng
            ).to(dtype)
            tensors[f"{base}.up_proj.weight"] = torch.randn(
                INTER, H, generator=rng
            ).to(dtype)
            tensors[f"{base}.down_proj.weight"] = torch.randn(
                H, INTER, generator=rng
            ).to(dtype)
        tensors[f"model.layers.{layer}.mlp.gate.weight"] = torch.randn(
            num_experts, H, generator=rng
        ).to(dtype)
    path = root / "model.safetensors"
    save_file(tensors, str(path))
    return path


def test_reads_bfloat16_checkpoint(tmp_path):
    """bf16 is what real checkpoints ship, and numpy cannot represent it.

    safetensors' numpy framework raises `data type 'bfloat16' not understood`
    at read time, so the read goes through torch. This test is the guard: it
    failed against the real OLMoE checkpoint before the fix.
    """
    from vllm_moe_surgeon.surgery.descriptors import CheckpointIndex, describe_layer

    _write_tiny_checkpoint(tmp_path)
    index = CheckpointIndex.open(str(tmp_path))

    assert index.expert_ids(0) == [0, 1, 2]
    weight = index.read(index.tensor_name(0, 0, "gate_proj"))
    assert weight.dtype == np.float32
    assert weight.shape == (INTER, H)

    descriptors = describe_layer(index, 0, rank=RANK)
    assert len(descriptors) == 3
    for descriptor in descriptors:
        assert set(descriptor.bases) == {"gate_proj", "up_proj", "down_proj"}
        assert descriptor.bases["gate_proj"].shape == (H, RANK)
        # down_proj is [H, I]; its invariant side is the column space, so the
        # basis must still land in R^H rather than R^I.
        assert descriptor.bases["down_proj"].shape == (H, RANK)


def test_similarity_matrix_on_a_real_checkpoint_is_well_formed(tmp_path):
    from vllm_moe_surgeon.surgery.descriptors import (
        CheckpointIndex,
        describe_layer,
        similarity_matrix,
    )

    _write_tiny_checkpoint(tmp_path, num_experts=4)
    index = CheckpointIndex.open(str(tmp_path))
    matrix = similarity_matrix(describe_layer(index, 0, rank=RANK))

    assert matrix.shape == (4, 4)
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-5)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)
    assert ((matrix >= 0.0) & (matrix <= 1.0)).all()


def test_missing_checkpoint_is_reported(tmp_path):
    from vllm_moe_surgeon.surgery.descriptors import CheckpointIndex

    with pytest.raises(FileNotFoundError, match="no safetensors checkpoint"):
        CheckpointIndex.open(str(tmp_path))


# --------------------------------------------------- the stacked (Granite) layout


def _write_stacked_checkpoint(root, num_experts=3):
    """IBM Granite's convention: all experts in one tensor, gate and up fused.

    ``input_linear`` is ``[E, 2I, H]`` with gate first then up -- the order vLLM's
    own granitemoe loader assumes when it does ``w1, w3 = p[e].chunk(2, dim=0)``.
    """
    import torch
    from safetensors.torch import save_file

    rng = torch.Generator().manual_seed(31)
    gate = torch.randn(num_experts, INTER, H, generator=rng)
    up = torch.randn(num_experts, INTER, H, generator=rng)
    down = torch.randn(num_experts, H, INTER, generator=rng)
    tensors = {
        "model.layers.0.block_sparse_moe.input_linear.weight": torch.cat(
            [gate, up], dim=1
        ),
        "model.layers.0.block_sparse_moe.output_linear.weight": down,
        "model.layers.0.block_sparse_moe.router.layer.weight": torch.randn(
            num_experts, H, generator=rng
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    return gate, up, down


def test_stacked_layout_is_detected(tmp_path):
    from vllm_moe_surgeon.surgery.descriptors import CheckpointIndex, detect_layout

    _write_stacked_checkpoint(tmp_path)
    index = CheckpointIndex.open(str(tmp_path))
    assert index.layout == "stacked"
    assert detect_layout({"model.layers.0.mlp.experts.0.gate_proj.weight": "s"}) == (
        "per_expert"
    )


def test_stacked_expert_ids_come_from_the_tensor_shape(tmp_path):
    from vllm_moe_surgeon.surgery.descriptors import CheckpointIndex

    _write_stacked_checkpoint(tmp_path, num_experts=5)
    assert CheckpointIndex.open(str(tmp_path)).expert_ids(0) == [0, 1, 2, 3, 4]


def test_stacked_reads_split_gate_and_up_the_right_way_round(tmp_path):
    """A swapped half loads without complaint and computes the wrong function."""
    from vllm_moe_surgeon.surgery.descriptors import CheckpointIndex

    gate, up, down = _write_stacked_checkpoint(tmp_path)
    index = CheckpointIndex.open(str(tmp_path))

    for expert in range(gate.shape[0]):
        np.testing.assert_allclose(
            index.read_expert(0, expert, "gate_proj"), gate[expert].numpy(), atol=0
        )
        np.testing.assert_allclose(
            index.read_expert(0, expert, "up_proj"), up[expert].numpy(), atol=0
        )
        np.testing.assert_allclose(
            index.read_expert(0, expert, "down_proj"), down[expert].numpy(), atol=0
        )
    # Guard the guard: if gate == up the assertions above prove nothing.
    assert not np.allclose(gate[0].numpy(), up[0].numpy())


def test_stacked_router_name_differs(tmp_path):
    from vllm_moe_surgeon.surgery.descriptors import CheckpointIndex

    _write_stacked_checkpoint(tmp_path)
    index = CheckpointIndex.open(str(tmp_path))
    assert index.router_name(0).endswith("block_sparse_moe.router.layer.weight")
    assert index.read(index.router_name(0)).shape == (3, H)


def test_descriptors_work_on_a_stacked_checkpoint(tmp_path):
    from vllm_moe_surgeon.surgery.descriptors import (
        CheckpointIndex,
        describe_layer,
        similarity_matrix,
    )

    _write_stacked_checkpoint(tmp_path, num_experts=4)
    index = CheckpointIndex.open(str(tmp_path))
    matrix = similarity_matrix(describe_layer(index, 0, rank=RANK))
    assert matrix.shape == (4, 4)
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-5)


def _stacked_source(tmp_path, num_experts=4):
    """A stacked checkpoint plus the config Granite carries."""
    import json

    gate, up, down = _write_stacked_checkpoint(tmp_path / "src", num_experts)
    with open(tmp_path / "src" / "config.json", "w") as f:
        json.dump(
            {
                "num_hidden_layers": 1,
                "num_local_experts": num_experts,
                "num_experts_per_tok": 2,
            },
            f,
        )
    return gate, up, down


def _drop_plan(dropped, kept):
    from vllm_moe_surgeon.surgery import Plan
    from vllm_moe_surgeon.surgery.plan import ExpertPlacement

    return Plan(
        model="m",
        revision=None,
        budget={},
        placements=[
            ExpertPlacement(0, e, "merge_into_core", tokens=10, share=0.5)
            for e in kept
        ]
        + [
            ExpertPlacement(0, e, "drop", tokens=1, share=0.0, merge_target=None)
            for e in dropped
        ],
        gate={"passed": True, "reason": "test"},
    )


def test_a_stacked_checkpoint_is_written_back_stacked(tmp_path):
    """The output keeps the source layout, and keeps the halves in Granite's order.

    Emitting per-expert tensors here would produce a checkpoint the granitemoe
    loader cannot load; emitting them with the halves swapped would produce one that
    loads silently and computes the wrong function. So this checks the surviving
    experts survive *bit-exactly* in the right half of the right slab.
    """
    from vllm_moe_surgeon.surgery.apply import apply_plan
    from vllm_moe_surgeon.surgery.descriptors import CheckpointIndex

    gate, up, down = _stacked_source(tmp_path, num_experts=4)
    kept = [0, 2, 3]
    manifest = apply_plan(
        _drop_plan(dropped=[1], kept=kept), str(tmp_path / "src"), str(tmp_path / "out")
    )

    out = CheckpointIndex.open(str(tmp_path / "out"))
    assert out.layout == "stacked", "a stacked source must not become per-expert"
    assert out.expert_ids(0) == [0, 1, 2]
    assert manifest["experts_before"] == 4
    assert manifest["experts_after"] == 3

    # new id i holds old id kept[i], gate still gate and up still up
    for new_id, old_id in enumerate(kept):
        np.testing.assert_allclose(
            out.read_expert(0, new_id, "gate_proj"), gate[old_id].numpy(), atol=0
        )
        np.testing.assert_allclose(
            out.read_expert(0, new_id, "up_proj"), up[old_id].numpy(), atol=0
        )
        np.testing.assert_allclose(
            out.read_expert(0, new_id, "down_proj"), down[old_id].numpy(), atol=0
        )
    assert not np.allclose(gate[0].numpy(), up[0].numpy())  # guard the guard


def test_a_stacked_write_carries_the_router_through_the_same_remap(tmp_path):
    """A router row left at its old index routes to the wrong expert, silently."""
    from vllm_moe_surgeon.surgery.apply import apply_plan
    from vllm_moe_surgeon.surgery.descriptors import CheckpointIndex

    _stacked_source(tmp_path, num_experts=4)
    src = CheckpointIndex.open(str(tmp_path / "src"))
    before = src.read(src.router_name(0))

    kept = [0, 2, 3]
    apply_plan(
        _drop_plan(dropped=[1], kept=kept), str(tmp_path / "src"), str(tmp_path / "out")
    )
    out = CheckpointIndex.open(str(tmp_path / "out"))
    after = out.read(out.router_name(0))

    assert after.shape == (3, H)
    for new_id, old_id in enumerate(kept):
        np.testing.assert_allclose(after[new_id], before[old_id], atol=1e-6)


def test_stacked_and_per_expert_surgery_agree(tmp_path):
    """Same weights in two layouts, same plan -- the survivors must match.

    The layouts differ only in storage, so any divergence here is the writer
    disagreeing with itself rather than a property of either format.
    """
    import json

    import torch
    from safetensors.torch import save_file

    from vllm_moe_surgeon.surgery.apply import apply_plan
    from vllm_moe_surgeon.surgery.descriptors import CheckpointIndex

    gate, up, down = _stacked_source(tmp_path, num_experts=4)
    router = CheckpointIndex.open(str(tmp_path / "src")).read(
        "model.layers.0.block_sparse_moe.router.layer.weight"
    )

    # the same tensors, per-expert
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    tensors = {"model.layers.0.mlp.gate.weight": torch.from_numpy(router)}
    for e in range(4):
        base = f"model.layers.0.mlp.experts.{e}"
        tensors[f"{base}.gate_proj.weight"] = gate[e]
        tensors[f"{base}.up_proj.weight"] = up[e]
        tensors[f"{base}.down_proj.weight"] = down[e]
    save_file(tensors, str(mirror / "model.safetensors"), metadata={"format": "pt"})
    with open(mirror / "config.json", "w") as f:
        json.dump(
            {"num_hidden_layers": 1, "num_local_experts": 4, "num_experts_per_tok": 2},
            f,
        )

    plan = _drop_plan(dropped=[1], kept=[0, 2, 3])
    apply_plan(plan, str(tmp_path / "src"), str(tmp_path / "out_stacked"))
    apply_plan(plan, str(mirror), str(tmp_path / "out_per_expert"))

    a = CheckpointIndex.open(str(tmp_path / "out_stacked"))
    b = CheckpointIndex.open(str(tmp_path / "out_per_expert"))
    assert a.layout == "stacked" and b.layout == "per_expert"
    for new_id in range(3):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            np.testing.assert_allclose(
                a.read_expert(0, new_id, projection),
                b.read_expert(0, new_id, projection),
                atol=0,
            )
    np.testing.assert_allclose(
        a.read(a.router_name(0)), b.read(b.router_name(0)), atol=0
    )


def test_stacked_merging_averages_into_the_right_slab(tmp_path):
    """A merge on the stacked path must land in the target's slab, not a new one."""
    from vllm_moe_surgeon.surgery import Plan
    from vllm_moe_surgeon.surgery.apply import apply_plan
    from vllm_moe_surgeon.surgery.descriptors import CheckpointIndex
    from vllm_moe_surgeon.surgery.plan import ExpertPlacement

    _stacked_source(tmp_path, num_experts=3)
    plan = Plan(
        model="m",
        revision=None,
        budget={},
        placements=[
            ExpertPlacement(0, 0, "merge_into_core", tokens=10, share=0.5),
            ExpertPlacement(0, 1, "merge_into_core", tokens=10, share=0.5),
            ExpertPlacement(
                0, 2, "drop", tokens=10, share=0.0, merge_target=0, similarity=0.99
            ),
        ],
        gate={"passed": True, "reason": "test"},
    )
    manifest = apply_plan(plan, str(tmp_path / "src"), str(tmp_path / "out"))

    assert manifest["merges_applied"] == 1
    out = CheckpointIndex.open(str(tmp_path / "out"))
    assert out.layout == "stacked"
    assert out.expert_ids(0) == [0, 1]
    # Shapes stay a valid stacked layer: [K, 2I, H] and [K, H, I].
    fused = out.read("model.layers.0.block_sparse_moe.input_linear.weight")
    assert fused.shape == (2, 2 * INTER, H)
    assert out.read(
        "model.layers.0.block_sparse_moe.output_linear.weight"
    ).shape == (2, H, INTER)
