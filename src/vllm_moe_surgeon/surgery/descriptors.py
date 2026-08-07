# SPDX-License-Identifier: Apache-2.0
"""Permutation-invariant expert descriptors, read streaming from a checkpoint.

An MoE expert is an FFN: ``w1 = gate_proj [I, H]``, ``w3 = up_proj [I, H]``,
``w2 = down_proj [H, I]``. The intermediate dimension ``I`` is a *permutation
symmetry* of the function: permute the rows of ``w1`` and ``w3`` and the columns
of ``w2`` by the same permutation and the expert computes exactly the same thing.

That has a sharp consequence for similarity. Two experts can be functionally
near-identical while their weight matrices, compared entrywise, look unrelated --
cosine similarity on flattened weights is nearly meaningless here. Any measure
used to decide *which experts may be merged* has to be blind to the permutation.

The measure used here is the one the symmetry hands us for free: the **row space
of ``w1``** is a subspace of :math:`\\mathbb{R}^H` and ``P @ w1`` spans the same
row space as ``w1`` for any permutation ``P``. Likewise the column space of
``w2``. So comparing subspaces is exactly permutation-invariant, with no
alignment step and no approximation.

Similarity of two experts, per projection, is the mean squared cosine of the
principal angles between their rank-``r`` dominant subspaces,
:math:`\\|V_a^\\top V_b\\|_F^2 / r \\in [0, 1]` -- 1 when the subspaces coincide.
It is also invariant to the sign of each singular vector, which matters because
those signs are arbitrary.

No vLLM, no GPU: ``safetensors`` gives per-tensor lazy reads, so a model far
larger than host RAM is processed one expert at a time.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .._logging import init_logger

logger = init_logger(__name__)

# Naming used by OLMoE, Qwen2/Qwen3-MoE, Mixtral-style exports.
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")
SHARD_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class CheckpointIndex:
    """Where each tensor lives, so experts can be read without a full load."""

    root: str
    #: tensor name -> shard filename
    weight_map: dict[str, str]

    @classmethod
    def open(cls, root: str) -> CheckpointIndex:
        index_path = os.path.join(root, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            return cls(root, weight_map)
        # Single-shard checkpoint: no index file is written.
        single = "model.safetensors"
        if os.path.exists(os.path.join(root, single)):
            from safetensors import safe_open

            with safe_open(os.path.join(root, single), framework="pt") as f:
                weight_map = dict.fromkeys(f.keys(), single)
            return cls(root, weight_map)
        raise FileNotFoundError(f"no safetensors checkpoint under {root}")

    def expert_ids(self, layer: int) -> list[int]:
        prefix = f"model.layers.{layer}.mlp.experts."
        ids = {
            int(m.group(1))
            for name in self.weight_map
            if name.startswith(prefix) and (m := _EXPERT_RE.search(name))
        }
        return sorted(ids)

    def tensor_name(self, layer: int, expert: int, projection: str) -> str:
        return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"

    def router_name(self, layer: int) -> str:
        return f"model.layers.{layer}.mlp.gate.weight"

    def read(self, name: str) -> np.ndarray:
        """One tensor as a float32 numpy array.

        Read through torch rather than safetensors' numpy framework: real
        checkpoints are bfloat16, numpy has no bfloat16, and
        ``framework="numpy"`` raises ``TypeError: data type 'bfloat16' not
        understood`` at read time -- before any conversion of ours could run.
        torch reads bf16 natively, so the widening happens once, here.

        One expert's tensor at a time, so peak memory is a few MB regardless of
        model size.
        """
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"{name} not in checkpoint index")
        import torch
        from safetensors import safe_open

        with safe_open(os.path.join(self.root, shard), framework="pt") as f:
            tensor = f.get_tensor(name)
        return tensor.to(torch.float32).numpy()


def _as_float32(array: np.ndarray) -> np.ndarray:
    """Widen an already-numpy array, decoding a raw 2-byte bf16 view if needed.

    :meth:`CheckpointIndex.read` returns float32 already; this covers arrays
    handed in from elsewhere, including the raw-uint16 bf16 representation that
    turns up when a tensor is memory-mapped rather than read through torch.
    """
    if array.dtype == np.float32:
        return array
    if array.dtype.itemsize == 2 and array.dtype.kind in ("V", "u", "i"):
        # bfloat16: the 16 bits are the high half of the float32 pattern.
        raw = array.view(np.uint16).astype(np.uint32) << 16
        return raw.view(np.float32).reshape(array.shape)
    return array.astype(np.float32)


def subspace_basis(matrix: np.ndarray, rank: int) -> np.ndarray:
    """Orthonormal basis of the dominant ``rank`` row-space directions.

    ``matrix`` is ``[n, H]``; the result is ``[H, rank]``. Row permutations of
    the input leave the result's *span* unchanged, which is the property the
    whole module rests on.
    """
    matrix = _as_float32(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {matrix.shape}")
    rank = min(rank, *matrix.shape)
    # Right singular vectors span the row space.
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    return np.ascontiguousarray(vh[:rank].T)


def subspace_similarity(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Mean squared cosine of principal angles, in [0, 1].

    Sign-invariant per column, because the singular-vector signs are arbitrary.
    """
    rank = min(basis_a.shape[1], basis_b.shape[1])
    overlap = basis_a[:, :rank].T @ basis_b[:, :rank]
    return float(np.sum(overlap * overlap) / rank)


@dataclass
class ExpertDescriptor:
    """One expert's permutation-invariant fingerprint."""

    layer: int
    expert: int
    #: projection name -> [H, rank] orthonormal basis
    bases: dict[str, np.ndarray]
    #: Frobenius norm per projection, a scale signal the subspaces discard.
    norms: dict[str, float]

    def similarity(self, other: ExpertDescriptor) -> float:
        shared = [p for p in self.bases if p in other.bases]
        if not shared:
            raise ValueError("descriptors share no projection")
        return float(
            np.mean(
                [
                    subspace_similarity(self.bases[p], other.bases[p])
                    for p in shared
                ]
            )
        )


def describe_layer(
    index: CheckpointIndex,
    layer: int,
    *,
    rank: int = 16,
    projections: Sequence[str] = SHARD_PROJECTIONS,
) -> list[ExpertDescriptor]:
    """Descriptors for every expert in one layer, one expert at a time."""
    descriptors = []
    for expert in index.expert_ids(layer):
        bases: dict[str, np.ndarray] = {}
        norms: dict[str, float] = {}
        for projection in projections:
            name = index.tensor_name(layer, expert, projection)
            weight = _as_float32(index.read(name))
            # down_proj is [H, I]: its *column* space is the invariant one, so
            # transpose into the [n, H] convention subspace_basis expects.
            oriented = weight.T if projection == "down_proj" else weight
            bases[projection] = subspace_basis(oriented, rank)
            norms[projection] = float(np.linalg.norm(weight))
        descriptors.append(ExpertDescriptor(layer, expert, bases, norms))
    return descriptors


def similarity_matrix(descriptors: Sequence[ExpertDescriptor]) -> np.ndarray:
    """Symmetric ``[E, E]`` similarity with 1.0 on the diagonal."""
    n = len(descriptors)
    out = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            value = descriptors[i].similarity(descriptors[j])
            out[i, j] = out[j, i] = value
    return out


def iter_layer_similarity(
    index: CheckpointIndex,
    layers: Sequence[int],
    *,
    rank: int = 16,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(layer, similarity_matrix)``, holding one layer at a time."""
    for layer in layers:
        descriptors = describe_layer(index, layer, rank=rank)
        if not descriptors:
            logger.warning("layer %d has no experts in the checkpoint", layer)
            continue
        logger.info("layer %d: described %d experts", layer, len(descriptors))
        yield layer, similarity_matrix(descriptors)
