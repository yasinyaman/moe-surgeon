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
from .plan import (
    Plan,
    deletes_anything,
    drop_set_digest,
    gate_passed,
    merges_anything,
    validate_plan,
)

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


#: Per-expert router-side tensors we know how to renumber. A DeepSeek-V3-style
#: `e_score_correction_bias` is a learned per-expert balancing term; a plain
#: `mlp.gate.bias` (or the stacked router's bias) is one value per expert too. All
#: are `[E, ...]` and must be reordered by the survivor mapping, exactly like the
#: router rows -- left alone they keep old-id length against a new-id router.
KNOWN_ROUTER_AUX_SUFFIXES = (
    "mlp.gate.e_score_correction_bias",
    "mlp.gate.bias",
    "block_sparse_moe.router.layer.bias",
)


def _router_module_prefix(index: CheckpointIndex, layer: int) -> str:
    if index.layout == "stacked":
        return f"model.layers.{layer}.block_sparse_moe.router."
    return f"model.layers.{layer}.mlp.gate."


def _router_aux_names(index: CheckpointIndex, layer: int) -> set[str]:
    """Per-expert router-side tensors for ``layer`` that must be renumbered.

    Any tensor under the router module whose first dimension is the expert count is
    a per-expert quantity. Known ones are renumbered; an *unknown* one is refused
    rather than passed through at the wrong length, because a wrong per-expert
    balancing bias against renumbered experts is a silent correctness bug.
    """
    prefix = _router_module_prefix(index, layer)
    router = index.router_name(layer)
    n_experts = len(index.expert_ids(layer))
    aux: set[str] = set()
    unknown: list[str] = []
    for name in index.weight_map:
        if not name.startswith(prefix) or name == router:
            continue
        if _read_torch(index, name).shape[0] != n_experts:
            continue  # a shared router-module tensor, not per-expert
        if any(name.endswith(sfx) for sfx in KNOWN_ROUTER_AUX_SUFFIXES):
            aux.add(name)
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"layer {layer} has per-expert router tensor(s) {sorted(unknown)} that "
            "apply does not know how to renumber. Refusing rather than writing them "
            "at the old expert count against a renumbered router. Add them to "
            "KNOWN_ROUTER_AUX_SUFFIXES once their semantics are confirmed."
        )
    return aux


def _blend_expert_rows(weight: Any, edit: LayerSurgery) -> Any:
    """Survivors of a ``[E, ...]`` tensor in new-id order, merges weighted-averaged.

    Works for the router (`[E, H]`) and for a per-expert bias (`[E]`) alike, since
    both index the expert on axis 0. See the module docstring on why a merged
    cluster's combined selection mass cannot be represented without a gate bias.
    """
    import torch

    rows = []
    for old_id in edit.survivors:
        donors = edit.merges.get(old_id, [])
        base = weight[old_id].to(torch.float32)
        if not donors:
            rows.append(base)
            continue
        total = edit.weights[old_id] + sum(w for _, w in donors)
        blended = base * (edit.weights[old_id] / total)
        for donor_id, donor_weight in donors:
            blended = blended + weight[donor_id].to(torch.float32) * (
                donor_weight / total
            )
        rows.append(blended)
    return torch.stack(rows)


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


def _source_moe_layers(index: CheckpointIndex) -> set[int]:
    """Every layer that carries expert weights in the source checkpoint."""
    import re

    pattern = (
        r"model\.layers\.(\d+)\.block_sparse_moe\."
        if index.layout == "stacked"
        else r"model\.layers\.(\d+)\.mlp\.experts\."
    )
    layers = set()
    for name in index.weight_map:
        m = re.match(pattern, name)
        if m:
            layers.add(int(m.group(1)))
    return layers


def source_fingerprint(source: str) -> dict[str, Any]:
    """A cheap content fingerprint of the source checkpoint, for the manifest.

    Not a full content hash of gigabytes of weights -- that would dominate a 7-second
    apply. Hashes ``config.json`` and the (name, size) of every shard, which is enough
    to notice the artifact was built from a *different* checkpoint than a later reader
    assumes. Recorded so a published number is traceable to the bytes it came from.
    """
    import hashlib

    h = hashlib.sha256()
    try:
        with open(os.path.join(source, "config.json"), "rb") as f:
            h.update(f.read())
    except OSError:
        pass
    shards = sorted(
        p for p in os.listdir(source) if p.endswith(".safetensors")
    )
    manifest = []
    for name in shards:
        size = os.path.getsize(os.path.join(source, name))
        h.update(f"{name}:{size}".encode())
        manifest.append({"shard": name, "bytes": size})
    return {"sha256": h.hexdigest()[:32], "shards": manifest}


def environment() -> dict[str, Any]:
    """Best-effort record of what produced an artifact: python, torch, vLLM, device.

    Versions come from package metadata (no heavy import); the device name needs
    torch, imported lazily and tolerantly, since the offline writer must not depend
    on it. A number nobody can place in an environment is a number nobody can
    reproduce.
    """
    import platform
    import sys
    from importlib import metadata

    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in ("torch", "vllm"):
        try:
            env[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return env


def refuse_in_place(source: str, out: str) -> None:
    """Refuse to write into the directory being read.

    ``apply`` and ``tier`` emit HF-conventional shard names and overwrite
    ``config.json`` / the shard index, while reading the source shards lazily and
    interleaved. Pointing ``--out`` at ``--source`` -- the obvious in-place
    invocation for an operator short on disk -- corrupts the checkpoint mid-read
    with no error. Refuse it before the first byte is written, source or out nested
    in the other included.
    """
    s = os.path.realpath(source)
    o = os.path.realpath(out)
    if s == o or o.startswith(s + os.sep) or s.startswith(o + os.sep):
        raise RuntimeError(
            f"refusing in-place surgery: --out ({out!r}) is the same as or nested "
            f"with --source ({source!r}). The writer overwrites the checkpoint it is "
            "reading. Write to a fresh directory."
        )


def apply_plan(
    plan: Plan,
    source: str,
    out_dir: str,
    *,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    copy_extra_files: bool = True,
    require_gate: bool = True,
    amplitude: float | None = None,
) -> dict[str, Any]:
    """Write the operated-on checkpoint. Returns the manifest.

    Refuses to delete experts from a plan that has not passed the ablation gate.
    Deletion is the one irreversible step in the pipeline, and its cost is
    measurable before it is paid -- so paying it unmeasured is a choice that has to
    be made explicitly, not by default.

    ``amplitude`` folds a scalar into every survivor's ``down_proj``, correcting the
    gate inflation deletion introduces. See :func:`fold_amplitude` for why that is the
    only place it can go, and `surgeon calibrate` for how to find the value.
    """
    import torch

    refuse_in_place(source, out_dir)

    gate_skipped = False
    if deletes_anything(plan) and not gate_passed(plan):
        if require_gate:
            verdict = (plan.gate or {}).get("reason", "never measured")
            raise RuntimeError(
                "this plan deletes experts but has not passed the quality gate "
                f"({verdict}). Run `surgeon gate` to measure what the deletions "
                "cost, or pass require_gate=False to accept an unmeasured plan."
            )
        gate_skipped = True
    elif deletes_anything(plan) and gate_passed(plan):
        # The gate passed, but for *this* drop-set? A plan is hand-editable, and a
        # verdict is a plain dict; gating a small drop-set then editing in more
        # deletions must not apply under the stale pass.
        recorded = (plan.gate or {}).get("drop_digest")
        current = drop_set_digest(plan)
        if recorded is None:
            logger.warning(
                "the gate verdict predates drop-set binding (no digest), so it "
                "cannot be checked against the plan's current deletions"
            )
        elif recorded != current:
            raise RuntimeError(
                "the gate verdict was measured for a different drop-set than this "
                f"plan now carries (digest {recorded[:12]}... != {current[:12]}...). "
                "Re-run `surgeon gate`, or pass require_gate=False to override."
            )

    if merges_anything(plan):
        # A passing gate is not evidence about merges. The gate zeroes experts,
        # which emulates deletion; a merge instead rewrites a *surviving* expert's
        # weights, and no amount of zeroing reaches that. Loud rather than fatal:
        # the merge machinery is exact on synthetic candidates, and refusing would
        # block the only way to measure it on real ones.
        merged = sum(
            1
            for p in plan.placements
            if p.action == "drop" and p.merge_target is not None
        )
        logger.warning(
            "%d merges in this plan are NOT covered by the quality gate: zeroing "
            "cannot emulate a merge, so the verdict speaks only for the deletions. "
            "Measure the applied checkpoint's perplexity before trusting it.",
            merged,
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

    # An applied plan must cover every MoE layer the source has, or none: writing a
    # config with the pruned num_experts while an uncovered layer's expert tensors
    # pass through at full width is a checkpoint that misloads. This is the exact
    # inconsistency a hand-edit produces by deleting a silent layer's placements.
    new_count = _expert_count(new_config)
    uncovered = _source_moe_layers(index) - set(surgery)
    if uncovered and new_count is not None:
        mismatched = {
            layer: len(index.expert_ids(layer))
            for layer in sorted(uncovered)
            if len(index.expert_ids(layer)) != new_count
        }
        if mismatched:
            raise ValueError(
                f"plan does not cover MoE layer(s) {sorted(mismatched)} (experts "
                f"{mismatched}), but writes a config with num_experts={new_count}. "
                "Their tensors would pass through at full width against a pruned "
                "config -- an inconsistent checkpoint. Cover every MoE layer, or "
                "none."
            )

    writer = ShardWriter(out_dir, shard_bytes)
    # The output keeps the source's layout. Emitting per-expert tensors for a model
    # whose loader expects stacked ones produces a checkpoint that fails to load;
    # emitting them under stacked names produces one that loads wrong.
    # Per-expert router-side tensors (e.g. DeepSeek's e_score_correction_bias) are
    # renumbered alongside the router; an unknown one is refused here, before writing.
    router_aux = {layer: _router_aux_names(index, layer) for layer in surgery}
    rewritten = {index.router_name(layer) for layer in surgery}
    for layer in surgery:
        rewritten |= _expert_tensor_names(index, layer)
        rewritten |= router_aux[layer]

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

            if amplitude is not None:
                down = fold_amplitude(down, amplitude)

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
        # Per-expert router-side tensors (e_score_correction_bias, router bias) follow
        # the same survivor reordering, in their own source dtype.
        for aux_name in sorted(router_aux[layer]):
            aux = _read_torch(index, aux_name)
            writer.add(aux_name, _blend_expert_rows(aux, edit).to(aux.dtype))

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
        "source_fingerprint": source_fingerprint(source),
        "environment": environment(),
        "plan_budget": plan.budget,
        "plan_provenance": plan.provenance,
        "plan_warnings": plan.warnings,
        "gate": plan.gate,
        "gate_skipped": gate_skipped,
        "gate_note": (
            "deletions applied WITHOUT a passing quality gate (require_gate=False); "
            "this artifact's quality was not measured"
            if gate_skipped
            else None
        ),
        "experts_before": _expert_count(config),
        "experts_after": _expert_count(new_config),
        "top_k_before": _top_k(config),
        "top_k_after": _top_k(new_config),
        "merges_applied": n_merged,
        "amplitude": amplitude,
        "amplitude_note": (
            "down_proj scaled by this scalar to correct the gate inflation deletion "
            "introduces. Already folded in -- do not apply it again at serve time."
            if amplitude is not None
            else "not applied; run `surgeon calibrate` to measure one"
        ),
        "router_rewrite": (
            "usage-weighted mean of member rows; the gate has no bias term, so "
            "the combined selection mass of a merged cluster is NOT represented "
            "-- the principled fix is the log-sum-exp least-squares refit in "
            "surgery/refit.py, which needs calibration hidden states and was not "
            "applied here"
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


def fold_amplitude(down: Any, scale: float) -> Any:
    """Scale one expert's ``down_proj``, which is where a gate correction can live.

    Deleting experts changes the amplitude of what survives. With
    ``renormalize=False`` -- OLMoE's setting -- a surviving gate is the raw softmax
    probability, so restricting the softmax from E rows to K inflates every one of them
    by ``1 / (1 - P_D(x))``, where ``P_D`` is the mass the deleted set used to carry.
    The model was never trained to receive an inflated MoE branch.

    The correction cannot go in the router. ``softmax_K(W'x)`` sums to one over
    survivors for *any* rows, so a uniform amplitude change is unrepresentable there --
    the same obstruction as the gate having no bias term. But the layer's output is
    **linear** in ``down_proj``, so scaling it is exactly scaling the gate.

    ``gate_proj`` and ``up_proj`` must not be touched: SwiGLU is nonlinear in them, so
    scaling them changes the function rather than its amplitude.

    Measured on OLMoE pruned 64 -> 40 experts, held-out perplexity, fresh load per
    point: 11.8754 at 1.0 against 10.5306 at 0.85 on gsm8k (baseline 9.6230), and
    34.5260 against 29.0684 on hellaswag (baseline 24.5961) -- a corpus the scalar was
    not fitted on. Roughly 55-60% of deletion's perplexity cost, for one multiply.
    """
    if not scale > 0:
        raise ValueError(f"amplitude must be positive, got {scale}")
    return down * float(scale)


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
    weight = _read_torch(index, index.router_name(layer))  # [E, H]
    return _blend_expert_rows(weight, edit).to(dtype)


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
