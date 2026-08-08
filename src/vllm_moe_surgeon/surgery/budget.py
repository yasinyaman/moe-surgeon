# SPDX-License-Identifier: Apache-2.0
"""What is the least GPU memory this model can be served in?

This is the static analysis that *is* reliable. Measured elsewhere in this
package: static weight signals cannot predict **which** experts a deployment will
use (rank correlation ~0.00 against a real profile). But they state exactly **how
much space** those experts occupy, and that is arithmetic rather than prediction --
so it cannot be wrong in the way a selection heuristic can.

Everything here comes from safetensors *headers* plus ``config.json``: tensor
shapes and dtypes, no weight data, no engine, no GPU. Answering "can this run in
4 GiB?" therefore costs a few hundred milliseconds and no accelerator at all.

The three tiers are sized separately because they do not shrink together, and
conflating them is the easy mistake:

GPU
    ``non-expert weights + num_moe_layers * capacity * record_bytes(model dtype)``.
    An fp8 store does **not** reduce this. The provider dequantizes into
    model-dtype GPU slots, so fp8 buys disk, host RAM and transfer bytes -- not
    VRAM. Only zero-copy mode removes the GPU tier, and that needs unified memory.
host RAM
    the pinned pool, ``num_moe_layers * ram_capacity * record_bytes(store dtype)``,
    where fp8 does halve the cost.
disk
    every expert, ``num_moe_layers * num_experts * record_bytes(store dtype)``.

The capacity floor is architectural: a forward pass needs at least ``top_k``
experts resident. Below that only the expert-split schedule works, which runs
``ceil(experts/capacity)`` kernel groups and rounds each group's partial sum to
model dtype rather than being bit-exact.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .._logging import init_logger
from ..store.expert_disk_store import ALIGN
from ..telemetry.layers import get_num_experts, get_top_k, resolve_moe_layers
from .descriptors import CheckpointIndex

logger = init_logger(__name__)

_DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def _align_up(value: int, to: int = ALIGN) -> int:
    return -(-value // to) * to


@dataclass
class CheckpointGeometry:
    """Byte accounting for a checkpoint, from headers alone."""

    num_experts: int
    top_k: int
    moe_layers: list[int]
    hidden: int
    intermediate: int
    #: Bytes of one expert's gate+up+down, at the checkpoint's own dtype.
    expert_bytes: int
    #: Every routed expert, all layers.
    routed_expert_bytes: int
    #: Shared / always-on expert MLPs, which are never part of the cache.
    shared_expert_bytes: int
    #: Router matrices.
    router_bytes: int
    #: Everything else: embeddings, attention, norms, lm_head.
    other_bytes: int
    checkpoint_dtype_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.routed_expert_bytes
            + self.shared_expert_bytes
            + self.router_bytes
            + self.other_bytes
        )

    @property
    def num_moe_layers(self) -> int:
        return len(self.moe_layers)

    def record_bytes(self, *, fp8: bool) -> int:
        """One store record: w13 + w2 (+ fp8 row scales), ALIGN-padded.

        Matches :mod:`.tiered` so the number is the one the store actually uses.
        """
        inter, hidden = self.intermediate, self.hidden
        if fp8:
            payload = (2 * inter * hidden) + (hidden * inter) + 4 * (2 * inter + hidden)
        else:
            width = self.checkpoint_dtype_bytes
            payload = width * ((2 * inter * hidden) + (hidden * inter))
        return _align_up(payload)

    @property
    def resident_gpu_record_bytes(self) -> int:
        """A GPU cache slot, which is always model dtype even with an fp8 store."""
        return self.record_bytes(fp8=False)


def analyze(checkpoint: str) -> CheckpointGeometry:
    """Read the geometry from headers. No weight data is touched."""
    from safetensors import safe_open

    with open(os.path.join(checkpoint, "config.json")) as f:
        config = json.load(f)

    index = CheckpointIndex.open(checkpoint)
    moe_layers = resolve_moe_layers(config)
    # Checked before top_k, which raises its own error on a dense config and
    # would bury the actual problem under a message about a missing field.
    if not moe_layers:
        raise ValueError(f"{checkpoint} has no MoE layers to budget")
    num_experts = get_num_experts(config)
    top_k = get_top_k(config)

    routed = shared = router = other = 0
    dtype_votes: dict[int, int] = {}

    by_shard: dict[str, list[str]] = {}
    for name, shard in index.weight_map.items():
        by_shard.setdefault(shard, []).append(name)

    for shard, names in by_shard.items():
        path = os.path.join(checkpoint, shard)
        with safe_open(path, framework="pt") as f:
            for name in names:
                slice_ = f.get_slice(name)
                shape = slice_.get_shape()
                width = _DTYPE_BYTES.get(slice_.get_dtype(), 2)
                nbytes = width
                for dim in shape:
                    nbytes *= dim

                if "shared_expert" in name:
                    shared += nbytes
                elif ".experts." in name:
                    routed += nbytes
                    dtype_votes[width] = dtype_votes.get(width, 0) + 1
                elif name.endswith("mlp.gate.weight") or "correction_bias" in name:
                    router += nbytes
                else:
                    other += nbytes

    dtype_bytes = max(dtype_votes, key=lambda k: dtype_votes[k]) if dtype_votes else 2

    # Geometry from a real expert tensor rather than the config, since config
    # field names for the MoE intermediate size vary by family.
    probe = index.read(index.tensor_name(moe_layers[0], 0, "gate_proj"))
    intermediate, hidden = probe.shape

    return CheckpointGeometry(
        num_experts=num_experts,
        top_k=top_k,
        moe_layers=moe_layers,
        hidden=hidden,
        intermediate=intermediate,
        expert_bytes=routed // max(len(moe_layers) * num_experts, 1),
        routed_expert_bytes=routed,
        shared_expert_bytes=shared,
        router_bytes=router,
        other_bytes=other,
        checkpoint_dtype_bytes=dtype_bytes,
    )


@dataclass
class TierPlan:
    """A concrete three-tier configuration and what it costs."""

    capacity: int
    ram_capacity: int
    fp8_store: bool
    gpu_bytes: int
    host_ram_bytes: int
    disk_bytes: int
    #: True when ``capacity < top_k``, which forces the expert-split schedule.
    needs_expert_split: bool
    #: Resident share of the routed experts.
    resident_fraction: float

    def env(self, store_dir: str = "<store>") -> list[str]:
        """The environment this plan corresponds to."""
        lines = [
            f"VLLM_MOE_DISK_STORE_DIR={store_dir}",
            f"VLLM_MOE_RAM_CACHE={self.ram_capacity}",
        ]
        if self.fp8_store:
            lines.append("VLLM_MOE_DISK_STORE_FP8=1")
        lines.append(f"--moe-expert-cache-size {self.capacity}")
        if self.needs_expert_split:
            lines.append("--moe-expert-cache-split expert   # capacity < top_k")
        return lines


def plan_for_vram(
    geometry: CheckpointGeometry,
    vram_bytes: int,
    *,
    fp8_store: bool = True,
    kv_cache_bytes: int = 0,
    ram_multiple: float = 2.0,
    headroom: float = 0.9,
) -> TierPlan | None:
    """Largest capacity that fits in ``vram_bytes``, or ``None`` if none does.

    ``headroom`` leaves room for activations, workspaces and fragmentation; the
    tool is for planning, and a plan that only fits with zero slack is not a plan.
    """
    fixed = geometry.other_bytes + geometry.shared_expert_bytes + geometry.router_bytes
    available = int(vram_bytes * headroom) - fixed - kv_cache_bytes
    per_slot = geometry.num_moe_layers * geometry.resident_gpu_record_bytes
    if available <= 0 or per_slot <= 0:
        return None

    capacity = available // per_slot
    if capacity < 1:
        return None
    capacity = min(int(capacity), geometry.num_experts)

    ram_capacity = min(
        geometry.num_experts, max(capacity, int(capacity * ram_multiple))
    )
    record = geometry.record_bytes(fp8=fp8_store)
    return TierPlan(
        capacity=capacity,
        ram_capacity=ram_capacity,
        fp8_store=fp8_store,
        gpu_bytes=fixed + capacity * per_slot + kv_cache_bytes,
        host_ram_bytes=geometry.num_moe_layers * ram_capacity * record,
        disk_bytes=geometry.num_moe_layers * geometry.num_experts * record,
        needs_expert_split=capacity < geometry.top_k,
        resident_fraction=capacity / geometry.num_experts,
    )


def floor_plan(
    geometry: CheckpointGeometry, *, fp8_store: bool = True
) -> TierPlan:
    """The smallest bit-exact configuration: exactly ``top_k`` resident.

    Below this the token-split schedule cannot serve a single token, so anything
    smaller has to accept the expert split's rounding.
    """
    per_slot = geometry.num_moe_layers * geometry.resident_gpu_record_bytes
    fixed = geometry.other_bytes + geometry.shared_expert_bytes + geometry.router_bytes
    capacity = geometry.top_k
    record = geometry.record_bytes(fp8=fp8_store)
    return TierPlan(
        capacity=capacity,
        ram_capacity=min(geometry.num_experts, 2 * capacity),
        fp8_store=fp8_store,
        gpu_bytes=fixed + capacity * per_slot,
        host_ram_bytes=geometry.num_moe_layers
        * min(geometry.num_experts, 2 * capacity)
        * record,
        disk_bytes=geometry.num_moe_layers * geometry.num_experts * record,
        needs_expert_split=False,
        resident_fraction=capacity / geometry.num_experts,
    )


def _gib(nbytes: int) -> str:
    """Auto-scaled bytes.

    Fixed GiB made the three per-record figures all print as "0.01 GiB", which
    hid the very thing they exist to show -- that the fp8 record is half the full
    one while the GPU slot is not.
    """
    if nbytes >= 1024**3:
        return f"{nbytes / 1024**3:.2f} GiB"
    if nbytes >= 1024**2:
        return f"{nbytes / 1024**2:.1f} MiB"
    return f"{nbytes / 1024:.0f} KiB"


def report(
    geometry: CheckpointGeometry,
    *,
    vram_gib: float | None = None,
    fp8_store: bool = True,
    kv_cache_gib: float = 1.0,
    model: str | None = None,
) -> str:
    """Numbers first: what the model is, what it costs, what fits."""
    lines = [
        f"model: {model or '(unnamed)'}",
        f"  experts/layer     {geometry.num_experts}   top_k {geometry.top_k}"
        f"   MoE layers {geometry.num_moe_layers}",
        f"  hidden {geometry.hidden}   moe intermediate {geometry.intermediate}"
        f"   dtype {geometry.checkpoint_dtype_bytes} bytes",
        "",
        "checkpoint bytes:",
        f"  routed experts    {_gib(geometry.routed_expert_bytes):>10}"
        f"   ({100 * geometry.routed_expert_bytes / max(geometry.total_bytes, 1):.0f}%"
        " of the model)",
        f"  shared experts    {_gib(geometry.shared_expert_bytes):>10}"
        "   always resident, never cached",
        f"  routers           {_gib(geometry.router_bytes):>10}",
        f"  everything else   {_gib(geometry.other_bytes):>10}"
        "   attention, embeddings, norms",
        f"  total             {_gib(geometry.total_bytes):>10}",
        "",
        "one cache slot / store record:",
        f"  GPU slot (model dtype)   {_gib(geometry.resident_gpu_record_bytes):>10}"
        "   per expert per layer",
        f"  store record fp8         {_gib(geometry.record_bytes(fp8=True)):>10}",
        f"  store record full        {_gib(geometry.record_bytes(fp8=False)):>10}",
        "",
        "  fp8 halves disk, host RAM and transfer bytes. It does NOT reduce VRAM:",
        "  the provider dequantizes into model-dtype GPU slots.",
    ]

    floor = floor_plan(geometry, fp8_store=fp8_store)
    lines += [
        "",
        f"bit-exact floor (capacity = top_k = {geometry.top_k}):",
        f"  GPU      {_gib(floor.gpu_bytes):>10}   (no KV cache included)",
        f"  host RAM {_gib(floor.host_ram_bytes):>10}",
        f"  disk     {_gib(floor.disk_bytes):>10}",
        f"  resident {100 * floor.resident_fraction:.1f}% of routed experts",
    ]

    if vram_gib is not None:
        fixed_cost = (
            geometry.other_bytes
            + geometry.shared_expert_bytes
            + geometry.router_bytes
        )
        kv = int(kv_cache_gib * 1024**3)
        plan = plan_for_vram(
            geometry,
            int(vram_gib * 1024**3),
            fp8_store=fp8_store,
            kv_cache_bytes=kv,
        )
        lines += [
            "",
            f"fitting {vram_gib:.2f} GiB VRAM (KV reserve {kv_cache_gib} GiB):",
        ]
        if plan is None:
            lines += [
                "  does not fit at any capacity >= 1.",
                f"  fixed cost alone is {_gib(fixed_cost)} before one expert slot.",
                "  options: reduce the KV reserve, use zero-copy on unified memory,",
                "  or serve a checkpoint with fewer/smaller experts.",
            ]
        else:
            lines += [
                f"  capacity {plan.capacity} of {geometry.num_experts} experts"
                f"   ({100 * plan.resident_fraction:.1f}% resident)",
                f"  GPU      {_gib(plan.gpu_bytes):>10}",
                f"  host RAM {_gib(plan.host_ram_bytes):>10}",
                f"  disk     {_gib(plan.disk_bytes):>10}",
            ]
            if plan.needs_expert_split:
                lines += [
                    f"  capacity is below top_k ({geometry.top_k}), so the token split",
                    "  cannot serve a token: the expert split is required, and it",
                    "  rounds each group's partial sum rather than being bit-exact.",
                ]
            lines += ["", "  serve with:"]
            lines += [f"    {line}" for line in plan.env()]
    return "\n".join(lines)


def analyze_and_report(checkpoint: str, **kwargs: Any) -> str:
    return report(analyze(checkpoint), **kwargs)
