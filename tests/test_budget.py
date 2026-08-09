# SPDX-License-Identifier: Apache-2.0
"""Static memory budgeting.

This is the one static analysis in the package that is arithmetic rather than
prediction, so the tests check the arithmetic against numbers measured on real
hardware -- and check the two places where the arithmetic is easy to get wrong in
a way that would flatter the result: fp8 does not shrink VRAM, and the capacity
floor is top_k.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_moe_surgeon.surgery.budget import (
    CheckpointGeometry,
    _gib,
    analyze,
    floor_plan,
    plan_for_vram,
    report,
)

H = 32
INTER = 16
E = 8
TOP_K = 2
LAYERS = 2


def _geometry(**overrides):
    base = dict(
        num_experts=E,
        top_k=TOP_K,
        moe_layers=list(range(LAYERS)),
        hidden=H,
        intermediate=INTER,
        expert_bytes=0,
        routed_expert_bytes=0,
        shared_expert_bytes=0,
        router_bytes=0,
        other_bytes=0,
        checkpoint_dtype_bytes=2,
    )
    base.update(overrides)
    return CheckpointGeometry(**base)


def _write(root, *, shared=False, dtype=torch.bfloat16):
    tensors = {
        "model.embed_tokens.weight": torch.zeros(64, H, dtype=dtype),
        "lm_head.weight": torch.zeros(64, H, dtype=dtype),
        "model.norm.weight": torch.ones(H, dtype=dtype),
    }
    for layer in range(LAYERS):
        for expert in range(E):
            base = f"model.layers.{layer}.mlp.experts.{expert}"
            tensors[f"{base}.gate_proj.weight"] = torch.zeros(INTER, H, dtype=dtype)
            tensors[f"{base}.up_proj.weight"] = torch.zeros(INTER, H, dtype=dtype)
            tensors[f"{base}.down_proj.weight"] = torch.zeros(H, INTER, dtype=dtype)
        tensors[f"model.layers.{layer}.mlp.gate.weight"] = torch.zeros(
            E, H, dtype=dtype
        )
        tensors[f"model.layers.{layer}.self_attn.q_proj.weight"] = torch.zeros(
            H, H, dtype=dtype
        )
        if shared:
            tensors[f"model.layers.{layer}.mlp.shared_expert.gate_proj.weight"] = (
                torch.zeros(INTER, H, dtype=dtype)
            )
    root.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    with open(root / "config.json", "w") as f:
        json.dump(
            {
                "num_hidden_layers": LAYERS,
                "num_experts": E,
                "num_experts_per_tok": TOP_K,
                "hidden_size": H,
            },
            f,
        )


# ------------------------------------------------------------------ arithmetic


def test_record_bytes_match_the_store_measured_on_gb10():
    """The one number the whole tool rests on, checked against a real store.

    OLMoE (64 experts, hidden 2048, moe intermediate 1024) produced an fp8 record
    stride of exactly 6,307,840 bytes and a 6.02 GiB store over 16 layers. If this
    drifts, every budget the tool prints is wrong.
    """
    olmoe = _geometry(
        num_experts=64,
        top_k=8,
        moe_layers=list(range(16)),
        hidden=2048,
        intermediate=1024,
    )
    assert olmoe.record_bytes(fp8=True) == 6_307_840
    total_gib = 16 * 64 * olmoe.record_bytes(fp8=True) / 1024**3
    assert total_gib == pytest.approx(6.02, abs=0.01)


def test_fp8_halves_the_record_but_not_the_gpu_slot():
    """The mistake that would make every small-VRAM plan too optimistic.

    The provider dequantizes fp8 records into model-dtype GPU slots, so fp8 buys
    disk, host RAM and transfer bytes -- never VRAM.

    Uses realistic dimensions on purpose: at toy sizes the 4096-byte record
    alignment swallows the fp8 saving entirely and both records round to one page,
    so a tiny fixture would assert nothing.
    """
    g = _geometry(hidden=2048, intermediate=1024, num_experts=64, top_k=8)
    assert g.record_bytes(fp8=True) < g.record_bytes(fp8=False)
    assert g.resident_gpu_record_bytes == g.record_bytes(fp8=False)

    small = plan_for_vram(g, 1 << 30, fp8_store=True)
    full = plan_for_vram(g, 1 << 30, fp8_store=False)
    assert small.capacity < g.num_experts, "must not be capped, or nothing varies"
    assert small.gpu_bytes == full.gpu_bytes, "fp8 must not change the GPU figure"
    assert small.host_ram_bytes < full.host_ram_bytes
    assert small.disk_bytes < full.disk_bytes


def test_records_are_page_aligned():
    """The store pads to 4096 so any pool row is O_DIRECT-aligned."""
    g = _geometry(hidden=2048, intermediate=1024)
    assert g.record_bytes(fp8=True) % 4096 == 0
    assert g.record_bytes(fp8=False) % 4096 == 0


# ---------------------------------------------------------------------- plans


def test_floor_plan_is_top_k_and_bit_exact():
    g = _geometry()
    floor = floor_plan(g)
    assert floor.capacity == TOP_K
    assert floor.needs_expert_split is False
    assert floor.resident_fraction == pytest.approx(TOP_K / E)


def test_capacity_below_top_k_demands_the_expert_split():
    """The token split cannot serve one token with fewer than top_k resident."""
    g = _geometry(other_bytes=0)
    per_slot = g.num_moe_layers * g.resident_gpu_record_bytes
    # Room for exactly one slot, which is below top_k = 2.
    plan = plan_for_vram(g, int(per_slot / 0.9) + 1, fp8_store=True)
    assert plan is not None
    assert plan.capacity == 1
    assert plan.needs_expert_split is True
    assert any("expert" in line for line in plan.env())


def test_a_model_that_cannot_fit_returns_none_rather_than_a_fantasy():
    g = _geometry(other_bytes=8 << 30)
    assert plan_for_vram(g, 1 << 30) is None


def test_capacity_never_exceeds_the_expert_count():
    g = _geometry()
    plan = plan_for_vram(g, 1 << 40)
    assert plan.capacity == E


def test_kv_reserve_reduces_capacity():
    g = _geometry(hidden=2048, intermediate=1024, num_experts=64, top_k=8)
    generous = plan_for_vram(g, 1 << 30, kv_cache_bytes=0)
    squeezed = plan_for_vram(g, 1 << 30, kv_cache_bytes=500 << 20)
    assert generous is not None and squeezed is not None
    assert squeezed.capacity < generous.capacity


def test_headroom_is_applied_not_ignored():
    """A plan that only fits with zero slack is not a plan."""
    g = _geometry()
    per_slot = g.num_moe_layers * g.resident_gpu_record_bytes
    exact = 4 * per_slot
    plan = plan_for_vram(g, exact, headroom=0.9)
    assert plan.capacity < 4


# -------------------------------------------------------------- from a real file


def test_analyze_classifies_tensors_by_role(tmp_path):
    _write(tmp_path)
    g = analyze(str(tmp_path))

    assert g.num_experts == E
    assert g.top_k == TOP_K
    assert g.moe_layers == [0, 1]
    assert g.intermediate == INTER
    assert g.hidden == H
    assert g.checkpoint_dtype_bytes == 2

    expected_routed = LAYERS * E * 2 * (2 * INTER * H + H * INTER)
    assert g.routed_expert_bytes == expected_routed
    assert g.router_bytes == LAYERS * 2 * E * H
    # Embeddings, lm_head, norm and attention -- not experts.
    assert g.other_bytes > 0
    assert g.shared_expert_bytes == 0


def test_shared_experts_are_counted_apart_from_routed(tmp_path):
    """Shared experts are always resident and never enter the cache.

    Folding them into the routed total would understate the fixed VRAM cost and
    overstate what the tier can offload.
    """
    _write(tmp_path, shared=True)
    g = analyze(str(tmp_path))
    assert g.shared_expert_bytes == LAYERS * 2 * INTER * H
    assert g.routed_expert_bytes == LAYERS * E * 2 * (2 * INTER * H + H * INTER)

    # And the fixed cost includes them.
    plan = plan_for_vram(g, 1 << 30)
    bare = plan_for_vram(_geometry(other_bytes=g.other_bytes), 1 << 30)
    assert plan.gpu_bytes >= bare.gpu_bytes


def test_analyze_refuses_a_dense_checkpoint(tmp_path):
    _write(tmp_path)
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"num_hidden_layers": LAYERS, "num_experts": 0}, f)
    with pytest.raises(ValueError, match="no MoE layers"):
        analyze(str(tmp_path))


# --------------------------------------------------------------------- report


def test_byte_formatting_scales_so_small_records_stay_visible():
    """Fixed GiB printed every record as '0.01 GiB', hiding the fp8 difference."""
    assert _gib(12 << 20) == "12.0 MiB"
    assert _gib(3 << 30) == "3.00 GiB"
    assert _gib(4096) == "4 KiB"


def test_report_states_that_fp8_does_not_help_vram(tmp_path):
    _write(tmp_path)
    text = report(analyze(str(tmp_path)), vram_gib=1.0, kv_cache_gib=0.0)
    assert "does NOT reduce VRAM" in text
    assert "bit-exact floor" in text


def test_report_explains_an_impossible_target(tmp_path):
    _write(tmp_path)
    g = analyze(str(tmp_path))
    g.other_bytes = 64 << 30
    text = report(g, vram_gib=1.0)
    assert "does not fit" in text
    assert "fixed cost alone" in text


def test_stacked_layout_expert_bytes_are_not_misfiled(tmp_path):
    """Classifying by name means the layout matters here too.

    Keying only on ".experts." filed every one of a stacked checkpoint's expert
    bytes under "everything else", which reported Granite as 0% experts with the
    whole model as fixed cost -- backwards, not just imprecise.
    """
    import json

    from safetensors.torch import save_file

    rng = torch.Generator().manual_seed(41)
    experts, inter, hidden = 4, 8, 16
    tensors = {
        "model.layers.0.block_sparse_moe.input_linear.weight": torch.randn(
            experts, 2 * inter, hidden, generator=rng, dtype=torch.float32
        ).to(torch.bfloat16),
        "model.layers.0.block_sparse_moe.output_linear.weight": torch.randn(
            experts, hidden, inter, generator=rng, dtype=torch.float32
        ).to(torch.bfloat16),
        "model.layers.0.block_sparse_moe.router.layer.weight": torch.randn(
            experts, hidden, generator=rng, dtype=torch.float32
        ).to(torch.bfloat16),
        "model.embed_tokens.weight": torch.zeros(8, hidden, dtype=torch.bfloat16),
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    with open(tmp_path / "config.json", "w") as f:
        json.dump(
            {
                "num_hidden_layers": 1,
                "num_local_experts": experts,
                "num_experts_per_tok": 2,
                "hidden_size": hidden,
            },
            f,
        )

    g = analyze(str(tmp_path))
    assert g.num_experts == experts
    assert g.intermediate == inter
    assert g.hidden == hidden
    expected = 2 * experts * (2 * inter * hidden + hidden * inter)
    assert g.routed_expert_bytes == expected
    assert g.router_bytes == 2 * experts * hidden
    # The experts must dominate, as they do in every real MoE checkpoint.
    assert g.routed_expert_bytes > g.other_bytes


# ------------------------------------- checkpoint dtype is not always serving dtype


def test_an_fp32_checkpoint_is_costed_at_the_width_vllm_downcasts_to():
    """Granite 3.0 stores fp32 weights and its config names no dtype at all.

    Measured: 12.57 GiB of checkpoint bytes against a 7.37 GiB boot floor, which only
    reconciles if resident weights are 6.29 GiB. vLLM's `_resolve_auto_dtype` does
    "Downcast for float32 models", and HF treats a silent config as fp32 -- so
    silence means downcast too. Trusting the checkpoint width here sizes a deployment
    twice too large and makes the arithmetic contradict the measurement.
    """
    from vllm_moe_surgeon.surgery.budget import _serving_dtype_bytes

    # The motivating case: fp32 file, config says nothing.
    assert _serving_dtype_bytes({}, 4) == 2
    assert _serving_dtype_bytes({"torch_dtype": "float32"}, 4) == 2
    assert _serving_dtype_bytes({"torch_dtype": None}, 4) == 2
    # An explicit 16-bit request, however it is spelled.
    assert _serving_dtype_bytes({"torch_dtype": "bfloat16"}, 4) == 2
    assert _serving_dtype_bytes({"torch_dtype": "torch.bfloat16"}, 4) == 2
    assert _serving_dtype_bytes({"dtype": "float16"}, 4) == 2
    # A 16-bit checkpoint is already at the serving width: nothing to say.
    assert _serving_dtype_bytes({"torch_dtype": "bfloat16"}, 2) == 2
    assert _serving_dtype_bytes({}, 2) == 2


def test_a_quantized_checkpoint_is_never_inflated_to_16_bit():
    """The same error in the other direction, and it would overstate the cost.

    fp8 checkpoints carry ``torch_dtype: bfloat16`` in their config, but vLLM keeps
    the fp8 weights resident. Costing them at 2 bytes would double the estimate.
    """
    from vllm_moe_surgeon.surgery.budget import _serving_dtype_bytes

    assert _serving_dtype_bytes({"torch_dtype": "bfloat16"}, 1) == 1
    assert _serving_dtype_bytes({}, 1) == 1


def test_the_scale_is_only_applied_when_the_dtypes_differ():
    from vllm_moe_surgeon.surgery.budget import CheckpointGeometry

    def geom(checkpoint_bytes, serving_bytes):
        return CheckpointGeometry(
            num_experts=4,
            top_k=2,
            moe_layers=[0],
            hidden=8,
            intermediate=4,
            expert_bytes=1,
            routed_expert_bytes=100,
            shared_expert_bytes=0,
            router_bytes=0,
            other_bytes=0,
            checkpoint_dtype_bytes=checkpoint_bytes,
            serving_dtype_bytes=serving_bytes,
        )

    same = geom(2, 2)
    assert same.serving_scale == 1.0
    assert not same.dtype_differs

    halved = geom(4, 2)
    assert halved.serving_scale == 0.5
    assert halved.dtype_differs

    # An unset serving width must not silently scale anything to zero.
    assert geom(4, 0).serving_scale == 1.0
