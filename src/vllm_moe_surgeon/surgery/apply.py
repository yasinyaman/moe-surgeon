# SPDX-License-Identifier: Apache-2.0
"""Apply a plan to a checkpoint, producing a smaller one that vLLM can serve.

Mostly streaming: tensors are read one at a time through
:class:`~.descriptors.CheckpointIndex` and flushed into output shards as a byte
budget fills, so a model far larger than host RAM can be operated on. The one
exception is structural -- a *stacked* checkpoint (Granite) stores a layer's experts
in a single ``[E, 2I, H]`` tensor, which cannot be written incrementally, so that
path buffers one layer's survivors. Peak cost is one layer of experts, not one model.

The output keeps the source's layout, per-expert or stacked. That is not a
preference: per-expert tensors written for a loader that expects stacked ones fail
to load, and the same tensors written under stacked names load *wrong*.

Three things have to stay consistent or the result loads but is wrong:

**Expert renumbering.** Survivors are renumbered to a contiguous ``0..K-1``.
Router rows have to be reordered by the *same* mapping -- a router row left at
its old index routes to a different expert than the one it was trained for, and
nothing will complain.

**top_k.** If fewer experts survive than the config's ``num_experts_per_tok``,
the routing top-k has to shrink with it.

**Neuron order.** A merge averages the donor into the target only after aligning
neurons (see :mod:`.align`).

The router rewrite for merges is a usage-weighted mean of the member rows. That
is an approximation, and the plan says so: with ``bias=False`` on the gate -- true
for OLMoE and Qwen3-MoE -- there is no additive term available to represent the
*combined* selection mass of a merged cluster, so a least-squares refit against
calibration hidden states is the principled fix and is not done here.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._logging import init_logger
from .descriptors import LAYOUTS, CheckpointIndex
from .plan import Plan, deletes_anything, gate_passed, validate_plan

logger = init_logger(__name__)

DEFAULT_SHARD_BYTES = 4 * 1024**3

# Config keys that carry the routed-expert count, under any of the naming
# conventions; all present ones are rewritten so the artifact stays coherent
# whichever the loader reads.
_EXPERT_COUNT_KEYS = ("num_experts", "n_routed_experts", "num_local_experts")
_TOP_K_KEYS = ("num_experts_per_tok", "top_k_experts")


@dataclass
class LayerSurgery:
    """The concrete edit for one layer, derived from the plan."""

    layer: int
    #: old expert id -> new contiguous id, for survivors only
    remap: dict[int, int]
    #: surviving expert id -> [(donor id, donor token weight), ...]
    merges: dict[int, list[tuple[int, float]]]
    #: survivor id -> its own token weight, for the weighted average
    weights: dict[int, float]
    deleted: list[int] = field(default_factory=list)

    @property
    def survivors(self) -> list[int]:
        return sorted(self.remap)


def derive_surgery(plan: Plan) -> dict[int, LayerSurgery]:
    """Turn placements into a per-layer edit, and check it is applicable."""
    validate_plan(plan)
    layers = sorted({p.layer for p in plan.placements})
    out: dict[int, LayerSurgery] = {}

    for layer in layers:
        rows = plan.by_layer(layer)
        survivors = sorted(p.expert for p in rows if p.action != "drop")
        remap = {old: new for new, old in enumerate(survivors)}

        merges: dict[int, list[tuple[int, float]]] = {}
        deleted: list[int] = []
        for placement in rows:
            if placement.action != "drop":
                continue
            if placement.merge_target is None:
                deleted.append(placement.expert)
            else:
                merges.setdefault(placement.merge_target, []).append(
                    (placement.expert, float(max(placement.tokens, 1)))
                )

        weights = {
            p.expert: float(max(p.tokens, 1)) for p in rows if p.action != "drop"
        }
        out[layer] = LayerSurgery(
            layer=layer,
            remap=remap,
            merges=merges,
            weights=weights,
            deleted=sorted(deleted),
        )
    return out


def rewrite_config(
    config: dict[str, Any], surgery: dict[int, LayerSurgery]
) -> dict[str, Any]:
    """Set the expert count and clamp top_k, refusing an inconsistent plan.

    The config carries one expert count for the whole model, so every pruned
    layer must end with the same number of survivors. A plan that keeps
    different counts per layer is not representable in this format and has to
    fail here rather than produce a checkpoint that misloads.
    """
    updated = dict(config)
    counts = {len(s.remap) for s in surgery.values()}
    if len(counts) > 1:
        raise ValueError(
            f"layers end with different expert counts {sorted(counts)}; the HF "
            "config has a single num_experts, so a per-layer count cannot be "
            "expressed. Use a uniform core budget."
        )
    new_count = counts.pop()

    touched = False
    for key in _EXPERT_COUNT_KEYS:
        if key in updated:
            updated[key] = new_count
            touched = True
    if not touched:
        raise ValueError(
            f"config has none of {_EXPERT_COUNT_KEYS}; cannot record the new "
            "expert count"
        )

    for key in _TOP_K_KEYS:
        if key in updated and updated[key] is not None:
            old_top_k = int(updated[key])
            if old_top_k > new_count:
                logger.warning(
                    "clamping %s from %d to %d: fewer experts survive than the "
                    "model routed to",
                    key,
                    old_top_k,
                    new_count,
                )
                updated[key] = new_count
    return updated


class ShardWriter:
    """Accumulate tensors and flush them into safetensors shards."""

    def __init__(self, out_dir: str, shard_bytes: int = DEFAULT_SHARD_BYTES):
        self.out_dir = out_dir
        self.shard_bytes = shard_bytes
        self._buffer: dict[str, Any] = {}
        self._buffered = 0
        self._weight_map: dict[str, str] = {}
        self._shards: list[str] = []
        os.makedirs(out_dir, exist_ok=True)

    def add(self, name: str, tensor: Any) -> None:
        self._buffer[name] = tensor
        self._buffered += tensor.numel() * tensor.element_size()
        if self._buffered >= self.shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        from safetensors.torch import save_file

        shard = f"model-{len(self._shards) + 1:05d}.safetensors"
        # safetensors rejects shared storage, which .T views and slices produce.
        payload = {k: v.contiguous().clone() for k, v in self._buffer.items()}
        save_file(payload, os.path.join(self.out_dir, shard), metadata={"format": "pt"})
        for name in self._buffer:
            self._weight_map[name] = shard
        self._shards.append(shard)
        logger.info("wrote %s (%d tensors)", shard, len(self._buffer))
        self._buffer.clear()
        self._buffered = 0

    def finalize(self) -> dict[str, str]:
        self.flush()
        # Rename to the conventional of-N form now that N is known.
        total = len(self._shards)
        renames = {}
        for i, shard in enumerate(self._shards, start=1):
            final = f"model-{i:05d}-of-{total:05d}.safetensors"
            os.rename(
                os.path.join(self.out_dir, shard), os.path.join(self.out_dir, final)
            )
            renames[shard] = final
        self._weight_map = {k: renames[v] for k, v in self._weight_map.items()}

        index = {
            "metadata": {
                "total_size": sum(
                    os.path.getsize(os.path.join(self.out_dir, s))
                    for s in renames.values()
                )
            },
            "weight_map": self._weight_map,
        }
        with open(
            os.path.join(self.out_dir, "model.safetensors.index.json"), "w"
        ) as f:
            json.dump(index, f, indent=2)
        return self._weight_map


def _expert_tensors(index: CheckpointIndex, layer: int, expert: int):
    """``(gate, up, down)`` as float32 numpy, whatever the source layout.

    Through ``read_expert``, not ``read(tensor_name(...))``: those two coincide only
    for the per-expert layout. For a stacked checkpoint ``tensor_name`` names the
    whole ``[E, 2I, H]`` tensor, so reading it directly would hand every caller all
    the experts at once instead of the one it asked for.

    Both layouts yield the same per-expert shapes -- gate/up ``[I, H]``, down
    ``[H, I]`` -- which is why alignment and merging need no layout awareness at all.
    """
    return (
        index.read_expert(layer, expert, "gate_proj"),
        index.read_expert(layer, expert, "up_proj"),
        index.read_expert(layer, expert, "down_proj"),
    )


def _expert_tensor_names(index: CheckpointIndex, layer: int) -> set[str]:
    """Every source tensor holding this layer's expert weights.

    These are the names ``apply_plan`` must *not* copy through, because it rewrites
    them. For the per-expert layout it is a whole prefix; for the stacked layout it
    is exactly two tensors.
    """
    if index.layout == "stacked":
        return {
            index.tensor_name(layer, 0, "gate_proj"),
            index.tensor_name(layer, 0, "down_proj"),
        }
    prefix = f"model.layers.{layer}.mlp.experts."
    return {name for name in index.weight_map if name.startswith(prefix)}


def _write_experts_stacked(
    writer: ShardWriter,
    index: CheckpointIndex,
    layer: int,
    experts: list[tuple[Any, Any, Any]],
    dtype,
) -> None:
    """Re-stack survivors into ``input_linear`` / ``output_linear``.

    ``input_linear`` is ``[K, 2I, H]`` with gate before up in each expert's slab --
    the order vLLM's granitemoe loader assumes when it does
    ``w1, w3 = p[e].chunk(2, dim=0)``. Getting that order backwards would produce a
    checkpoint that loads without complaint and computes the wrong function, which is
    why the two halves are concatenated here explicitly rather than by reusing
    whatever order they arrived in.
    """
    import torch

    fused = torch.stack(
        [
            torch.cat(
                (
                    torch.from_numpy(np.ascontiguousarray(gate)),
                    torch.from_numpy(np.ascontiguousarray(up)),
                ),
                dim=0,
            )
            for gate, up, _ in experts
        ]
    ).to(dtype)
    down = torch.stack(
        [torch.from_numpy(np.ascontiguousarray(d)) for _, _, d in experts]
    ).to(dtype)
    writer.add(index.tensor_name(layer, 0, "gate_proj"), fused)
    writer.add(index.tensor_name(layer, 0, "down_proj"), down)


def _iter_source_tensors(index: CheckpointIndex) -> Iterator[str]:
    """Every tensor name in the source, in a stable order."""
    return iter(sorted(index.weight_map))


def apply_plan(
    plan: Plan,
    source: str,
    out_dir: str,
    *,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    copy_extra_files: bool = True,
    require_gate: bool = True,
) -> dict[str, Any]:
    """Write the operated-on checkpoint. Returns the manifest.

    Refuses to delete experts from a plan that has not passed the ablation gate.
    Deletion is the one irreversible step in the pipeline, and its cost is
    measurable before it is paid -- so paying it unmeasured is a choice that has to
    be made explicitly, not by default.
    """
    import torch

    if require_gate and deletes_anything(plan) and not gate_passed(plan):
        verdict = (plan.gate or {}).get("reason", "never measured")
        raise RuntimeError(
            "this plan deletes experts but has not passed the quality gate "
            f"({verdict}). Run `surgeon gate` to measure what the deletions cost, "
            "or pass require_gate=False to accept an unmeasured plan."
        )

    surgery = derive_surgery(plan)
    index = CheckpointIndex.open(source)
    if index.layout not in LAYOUTS:
        raise NotImplementedError(
            f"{source} uses the {index.layout!r} expert layout, which cannot be "
            f"written; known layouts are {list(LAYOUTS)}"
        )

    with open(os.path.join(source, "config.json")) as f:
        config = json.load(f)
    new_config = rewrite_config(config, surgery)

    writer = ShardWriter(out_dir, shard_bytes)
    # The output keeps the source's layout. Emitting per-expert tensors for a model
    # whose loader expects stacked ones produces a checkpoint that fails to load;
    # emitting them under stacked names produces one that loads wrong.
    rewritten = {index.router_name(layer) for layer in surgery}
    for layer in surgery:
        rewritten |= _expert_tensor_names(index, layer)

    n_merged = 0
    for name in _iter_source_tensors(index):
        if name in rewritten:
            continue  # rewritten below, per layer
        # Everything else -- embeddings, attention, norms, lm_head -- passes
        # through untouched and in its original dtype.
        writer.add(name, _read_torch(index, name))

    for layer, edit in surgery.items():
        first = index.tensor_name(layer, edit.survivors[0], "gate_proj")
        dtype = _read_torch(index, first).dtype
        # Survivors in new-id order. Per-expert writes each one as it is produced;
        # stacked has to concatenate them, so the order is what carries the remap and
        # is established once, here, for both paths.
        in_new_order = sorted(edit.survivors, key=lambda old: edit.remap[old])
        produced: list[tuple[Any, Any, Any]] = []

        for old_id in in_new_order:
            new_id = edit.remap[old_id]
            donors = edit.merges.get(old_id, [])
            target = _expert_tensors(index, layer, old_id)

            if donors:
                from .align import merge_experts

                donor_data = [
                    (_expert_tensors(index, layer, donor_id), weight)
                    for donor_id, weight in donors
                ]
                gate, up, down = merge_experts(
                    target, donor_data, edit.weights[old_id]
                )
                n_merged += len(donors)
                logger.info(
                    "layer %d expert %d <- merged %s",
                    layer,
                    old_id,
                    [d for d, _ in donors],
                )
            else:
                gate, up, down = target

            if index.layout == "stacked":
                produced.append((gate, up, down))
                continue

            base = f"model.layers.{layer}.mlp.experts.{new_id}"
            for suffix, array in (
                ("gate_proj", gate),
                ("up_proj", up),
                ("down_proj", down),
            ):
                writer.add(
                    f"{base}.{suffix}.weight",
                    torch.from_numpy(np.ascontiguousarray(array)).to(dtype),
                )

        if index.layout == "stacked":
            _write_experts_stacked(writer, index, layer, produced, dtype)

        writer.add(
            index.router_name(layer),
            _rewrite_router(index, layer, edit, dtype),
        )

    weight_map = writer.finalize()

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(new_config, f, indent=2)

    if copy_extra_files:
        _copy_aux_files(source, out_dir)

    manifest = {
        "manifest_version": 1,
        "base_model": plan.model,
        "base_revision": plan.revision,
        "source_path": os.path.abspath(source),
        "plan_budget": plan.budget,
        "plan_provenance": plan.provenance,
        "plan_warnings": plan.warnings,
        "gate": plan.gate,
        "experts_before": _expert_count(config),
        "experts_after": _expert_count(new_config),
        "top_k_before": _top_k(config),
        "top_k_after": _top_k(new_config),
        "merges_applied": n_merged,
        "router_rewrite": (
            "usage-weighted mean of member rows; the gate has no bias term, so "
            "the combined selection mass of a merged cluster is NOT represented "
            "-- a least-squares refit on calibration hidden states is the "
            "principled fix and was not applied"
            if n_merged
            else "rows of surviving experts, reordered to the new ids"
        ),
        "layers": {
            str(layer): {
                "remap": {str(k): v for k, v in edit.remap.items()},
                "deleted": edit.deleted,
                "merged_away": {
                    str(target): [d for d, _ in donors]
                    for target, donors in edit.merges.items()
                },
            }
            for layer, edit in surgery.items()
        },
        "shards": sorted(set(weight_map.values())),
    }
    with open(os.path.join(out_dir, "surgeon_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _read_torch(index: CheckpointIndex, name: str):
    """Read a tensor without widening -- pass-through keeps the source dtype."""
    from safetensors import safe_open

    shard = index.weight_map[name]
    with safe_open(os.path.join(index.root, shard), framework="pt") as f:
        return f.get_tensor(name)


def _rewrite_router(
    index: CheckpointIndex, layer: int, edit: LayerSurgery, dtype
):
    """Router rows for the survivors, in their new order."""
    import torch

    weight = _read_torch(index, index.router_name(layer))  # [E, H]
    rows = []
    for old_id in edit.survivors:
        donors = edit.merges.get(old_id, [])
        if not donors:
            rows.append(weight[old_id].to(torch.float32))
            continue
        # Weighted mean over the cluster. See the module docstring: without a
        # gate bias this cannot express the cluster's summed selection mass.
        total = edit.weights[old_id] + sum(w for _, w in donors)
        blended = weight[old_id].to(torch.float32) * (edit.weights[old_id] / total)
        for donor_id, donor_weight in donors:
            blended = blended + weight[donor_id].to(torch.float32) * (
                donor_weight / total
            )
        rows.append(blended)
    return torch.stack(rows).to(dtype)


def _expert_count(config: dict[str, Any]) -> int | None:
    for key in _EXPERT_COUNT_KEYS:
        if key in config:
            return int(config[key])
    return None


def _top_k(config: dict[str, Any]) -> int | None:
    for key in _TOP_K_KEYS:
        if config.get(key) is not None:
            return int(config[key])
    return None


def _copy_aux_files(source: str, out_dir: str) -> None:
    """Tokenizer and generation config -- without these the artifact is unusable."""
    wanted = (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.jinja",
        "generation_config.json",
    )
    for name in wanted:
        src = os.path.join(source, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, name))
