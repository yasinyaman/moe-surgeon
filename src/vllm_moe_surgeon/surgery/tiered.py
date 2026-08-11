# SPDX-License-Identifier: Apache-2.0
"""Build the tiered artifact: a disk store plus a residency hint.

The measurements changed what this phase has to be. On OLMoE nothing is
redundant enough to merge and nothing is cold enough to delete, so editing the
checkpoint is not where the win is. What remains is **placement**: keep every
expert, put them all in an NVMe store, and tell the cache which ones deserve VRAM.

That makes the tiered artifact three things beside the checkpoint:

1. a store holding *every* expert of a layer (the existing
   :class:`~..store.expert_disk_store.DiskExpertStore` already stores all
   ``num_experts`` records -- the cache, not the store, decides residency),
2. ``hot_experts.json``, the per-layer residency hint, and
3. the checkpoint itself, edited only if the plan actually drops or merges.

Building the store offline matters on its own. Until now a store could only be
produced by booting vLLM with ``VLLM_MOE_STREAM_LOAD=1`` and intercepting the
weight loader, which needs a GPU and a working engine. Here it is a pure
safetensors-to-records pass, so it runs as an ordinary pipeline stage on any host.

The record layout is byte-compatible with the in-tree writer, and that is a
correctness requirement rather than a nicety: ``w13`` rows ``[0, I)`` are
``gate_proj`` and ``[I, 2I)`` are ``up_proj``. Getting that order backwards
produces a store that loads without complaint and computes the wrong function.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from .._logging import init_logger
from ..store.expert_disk_store import DiskExpertStore, quantize_rowwise_fp8
from .descriptors import CheckpointIndex

logger = init_logger(__name__)

#: How vLLM names a MoE layer's expert container, and therefore the store key.
#: ``RoutedExperts.layer_name`` is the factory prefix plus ``.experts``.
LAYER_NAME_TEMPLATE = "model.layers.{layer}.mlp.experts"


def store_key(layer_name: str) -> str:
    """The store's filename stem, matching the in-tree writer exactly."""
    return layer_name.replace("/", "_").replace(".", "_")


def store_path(directory: str, layer_name: str) -> str:
    return os.path.join(directory, f"{store_key(layer_name)}.experts")


def build_layer_store(
    index: CheckpointIndex,
    layer: int,
    directory: str,
    *,
    model_id: str,
    revision: str | None,
    fp8: bool = True,
    layer_name_template: str = LAYER_NAME_TEMPLATE,
) -> DiskExpertStore:
    """Write one layer's expert records, one expert at a time.

    ``model_id`` and ``revision`` go into the store's identity fingerprint and
    must match what the serving engine will report, or the store is rebuilt
    rather than reused. A mismatch is safe -- the check is exact-match and falls
    back to rebuilding -- which is why the fingerprint exists at all: two models
    of the same architecture must never silently share records.
    """
    import torch

    expert_ids = index.expert_ids(layer)
    if not expert_ids:
        raise ValueError(f"layer {layer} has no experts in the checkpoint")
    if expert_ids != list(range(len(expert_ids))):
        raise ValueError(
            f"layer {layer} expert ids are not contiguous from 0: "
            f"{expert_ids[:5]}... -- renumber the checkpoint first"
        )

    probe = index.read_expert(layer, expert_ids[0], "gate_proj")
    inter, hidden = probe.shape
    dtype = _source_dtype(index, layer, expert_ids[0])

    if fp8 and dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"fp8 records target bf16/fp16 checkpoints; this one is {dtype}"
        )

    if fp8:
        specs = [
            ("w13", (2 * inter, hidden), torch.float8_e4m3fn),
            ("w2", (hidden, inter), torch.float8_e4m3fn),
            ("w13_qs", (2 * inter,), torch.float32),
            ("w2_qs", (hidden,), torch.float32),
        ]
    else:
        specs = [
            ("w13", (2 * inter, hidden), dtype),
            ("w2", (hidden, inter), dtype),
        ]

    layer_name = layer_name_template.format(layer=layer)
    os.makedirs(directory, exist_ok=True)
    store = DiskExpertStore.create_for_streaming(
        store_path(directory, layer_name),
        len(expert_ids),
        specs,
        identity={
            "model": model_id,
            "revision": str(revision),
            "layer": layer_name,
        },
        quant="fp8e4m3-row" if fp8 else None,
    )
    if store.is_complete:
        # Reuse is keyed by identity {model, revision, layer} + shapes/dtypes, NOT by
        # content: a full content hash is deliberately not part of the fingerprint,
        # because the streamed boot builds the store without reading the source
        # weights and must stay byte-interchangeable with this offline build. So a
        # rebuilt checkpoint under the same model id and revision would be reused
        # silently -- warn, naming the identity, and note that revision is often unset
        # (str(None)). Give a changed checkpoint a fresh --store dir or a distinct
        # --revision.
        logger.warning(
            "layer %d: reusing existing store for identity "
            "{model=%s, revision=%s, layer=%s}; reuse is by identity + shapes, not "
            "content, so ensure this store was built from the same checkpoint",
            layer,
            model_id,
            revision,
            layer_name,
        )
        return store

    def read(expert: int, projection: str):
        return torch.from_numpy(index.read_expert(layer, expert, projection))

    for expert in expert_ids:
        row = torch.zeros(store.record_stride, dtype=torch.uint8)
        gate = read(expert, "gate_proj")
        up = read(expert, "up_proj")
        down = read(expert, "down_proj")

        if fp8:
            # Row-wise, so gate and up quantize independently even though they
            # share the w13 field: each supplies complete output rows.
            gate_q, gate_s = quantize_rowwise_fp8(gate)
            up_q, up_s = quantize_rowwise_fp8(up)
            down_q, down_s = quantize_rowwise_fp8(down)
            w13 = store.field_view(row, "w13")
            w13.narrow(0, 0, inter).copy_(gate_q)
            w13.narrow(0, inter, inter).copy_(up_q)
            store.field_view(row, "w2").copy_(down_q)
            qs = store.field_view(row, "w13_qs")
            qs.narrow(0, 0, inter).copy_(gate_s)
            qs.narrow(0, inter, inter).copy_(up_s)
            store.field_view(row, "w2_qs").copy_(down_s)
        else:
            w13 = store.field_view(row, "w13")
            w13.narrow(0, 0, inter).copy_(gate)
            w13.narrow(0, inter, inter).copy_(up)
            store.field_view(row, "w2").copy_(down)

        store.write_record(expert, row)

    store.finalize()
    logger.info(
        "layer %d: wrote %d records, stride %d bytes%s",
        layer,
        len(expert_ids),
        store.record_stride,
        " (fp8)" if fp8 else "",
    )
    return store


def _source_dtype(index: CheckpointIndex, layer: int, expert: int) -> Any:
    """The checkpoint's own torch dtype for an expert weight.

    Read from a one-row slice rather than from ``get_dtype()``: that returns
    safetensors' own spelling (``"BF16"``), and mapping those strings onto torch
    dtypes by hand is exactly the kind of table that silently goes wrong. A slice
    hands back a real tensor, so torch names the dtype for us.
    """
    from safetensors import safe_open

    name = index.tensor_name(layer, expert, "gate_proj")
    shard = index.weight_map[name]
    with safe_open(os.path.join(index.root, shard), framework="pt") as f:
        return f.get_slice(name)[0:1].dtype


def build_store(
    checkpoint: str,
    directory: str,
    layers: Sequence[int],
    *,
    model_id: str,
    revision: str | None = None,
    fp8: bool = True,
) -> dict[str, Any]:
    """Build the store for every named layer. Returns a summary."""
    from .apply import refuse_in_place

    refuse_in_place(checkpoint, directory)
    index = CheckpointIndex.open(checkpoint)
    total_bytes = 0
    built = []
    for layer in layers:
        build_layer_store(
            index,
            layer,
            directory,
            model_id=model_id,
            revision=revision,
            fp8=fp8,
        )
        path = store_path(directory, LAYER_NAME_TEMPLATE.format(layer=layer))
        total_bytes += os.path.getsize(path)
        built.append(layer)
    return {
        "directory": os.path.abspath(directory),
        "layers": built,
        "fp8": fp8,
        "model_id": model_id,
        "revision": revision,
        "bytes": total_bytes,
    }
