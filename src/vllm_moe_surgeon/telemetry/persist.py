# SPDX-License-Identifier: Apache-2.0
"""Save and load a profile.

``.npz`` rather than parquet: the payload is two or three dense integer arrays,
which is what npz is for, and it keeps the offline half free of a pyarrow
dependency. The provenance fields are not decoration -- a plan built from the
wrong profile is unfalsifiable after the fact, so the model, revision and the
counts of dropped rows travel with the numbers.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .stats import ExpertStats

FORMAT_VERSION = 1


def save(
    stats: ExpertStats,
    path: str,
    *,
    model: str | None = None,
    revision: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    meta = {
        "format_version": FORMAT_VERSION,
        "model": model,
        "revision": revision,
        "num_experts": stats.num_experts,
        "moe_layers": stats.moe_layers,
        "n_sequences": stats.n_sequences,
        "dropped_degenerate_rows": stats.dropped_degenerate_rows,
        "dropped_sentinels": stats.dropped_sentinels,
        "silent_layers": sorted(stats.silent_layers),
        "mean_experts_per_token": stats.top_k_hint,
        "position_order_correlation": stats.position_order_correlation(),
        **(extra or {}),
    }
    arrays = {
        "tokens": stats.tokens,
        "layer_token_rows": stats.layer_token_rows,
        "meta": np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
    }
    if stats.cooc is not None:
        arrays["cooc"] = stats.cooc
    if stats.positions is not None:
        arrays["positions"] = stats.positions
    np.savez_compressed(path, **arrays)


def load(path: str) -> tuple[ExpertStats, dict[str, Any]]:
    """Return the profile and its provenance metadata."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(bytes(data["meta"]).decode())
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"{path}: format_version {meta.get('format_version')} "
                f"!= {FORMAT_VERSION}"
            )
        stats = ExpertStats(
            num_experts=int(meta["num_experts"]),
            moe_layers=list(meta["moe_layers"]),
            tokens=data["tokens"],
            layer_token_rows=data["layer_token_rows"],
            cooc=data["cooc"] if "cooc" in data else None,
            positions=data["positions"] if "positions" in data else None,
            n_sequences=int(meta["n_sequences"]),
            dropped_degenerate_rows=int(meta["dropped_degenerate_rows"]),
            dropped_sentinels=int(meta["dropped_sentinels"]),
            silent_layers=set(meta["silent_layers"]),
        )
    return stats, meta
