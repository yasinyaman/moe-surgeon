# SPDX-License-Identifier: Apache-2.0
"""Per-expert usage accumulation from vLLM's routed-experts capture.

No vLLM import and no GPU: the input is the ``[seq_len, num_layers, top_k]``
array that ``CompletionOutput.routed_experts`` already hands us, so aggregation
is ordinary numpy. That is the whole reason this module is cheap -- vLLM's
``--enable-return-routed-experts`` does the hard part in the forward pass, and we
never touch a layer.

Two hazards are handled here rather than left to the caller, because getting
either wrong silently biases every downstream pruning decision:

**Dense rows read as expert 0.** The capture buffer is sized for every decoder
layer and zeroed each step; dense layers never write to it. See
:mod:`.layers`.

**Sentinel ids.** Some routing paths emit negative ids for "no expert". They must
be dropped, not counted.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .._logging import init_logger

logger = init_logger(__name__)


@dataclass
class ExpertStats:
    """Token counts per (layer, expert), plus the bookkeeping to trust them.

    ``tokens[layer_index, expert_id]`` counts how many *token slots* selected
    that expert -- i.e. token-weighted load, the quantity EPLB calls
    ``expert_load_pass``. A token contributes to ``top_k`` experts.
    """

    num_experts: int
    #: Global decoder-layer indices treated as MoE, in ascending order.
    moe_layers: list[int]
    #: (len(moe_layers), num_experts) int64
    tokens: np.ndarray
    #: Rows seen per layer, so a per-layer share can be computed honestly.
    layer_token_rows: np.ndarray
    #: Optional within-token co-occurrence, (len(moe_layers), E, E) int64.
    cooc: np.ndarray | None = None
    n_sequences: int = 0
    #: Rows rejected as uncaptured (all top_k entries identical).
    dropped_degenerate_rows: int = 0
    #: Individual (row, slot) entries dropped for being negative sentinels.
    dropped_sentinels: int = 0
    #: Layers the config called MoE but which never produced a valid row.
    silent_layers: set[int] = field(default_factory=set)

    @classmethod
    def empty(
        cls,
        num_experts: int,
        moe_layers: Sequence[int],
        with_cooc: bool = False,
    ) -> ExpertStats:
        layers = sorted(moe_layers)
        n = len(layers)
        return cls(
            num_experts=num_experts,
            moe_layers=layers,
            tokens=np.zeros((n, num_experts), dtype=np.int64),
            layer_token_rows=np.zeros(n, dtype=np.int64),
            cooc=(
                np.zeros((n, num_experts, num_experts), dtype=np.int64)
                if with_cooc
                else None
            ),
        )

    @property
    def top_k_hint(self) -> float:
        """Mean experts per token actually observed. Should match the config."""
        rows = int(self.layer_token_rows.sum())
        return float(self.tokens.sum() / rows) if rows else 0.0

    def add(self, routed_experts: np.ndarray) -> None:
        """Accumulate one sequence's capture, ``[seq_len, num_layers, top_k]``."""
        arr = np.asarray(routed_experts)
        if arr.ndim != 3:
            raise ValueError(
                f"expected [seq_len, num_layers, top_k], got shape {arr.shape}"
            )
        seq_len, num_layers, top_k = arr.shape

        if self.moe_layers and max(self.moe_layers) >= num_layers:
            raise ValueError(
                f"capture has {num_layers} layer rows but moe_layers names "
                f"index {max(self.moe_layers)}; the config and the model "
                "disagree about depth"
            )

        for slot, layer_idx in enumerate(self.moe_layers):
            rows = arr[:, layer_idx, :]

            # An uncaptured row is all-zeros. A genuine top-k row cannot repeat
            # an expert, so "every entry identical" identifies uncaptured rows
            # without trusting the config -- but only when top_k >= 2. At
            # top_k == 1 the two are indistinguishable and we must rely on the
            # caller's layer set alone.
            if top_k >= 2:
                degenerate = np.all(rows == rows[:, :1], axis=1)
                if degenerate.any():
                    self.dropped_degenerate_rows += int(degenerate.sum())
                    rows = rows[~degenerate]

            if rows.size == 0:
                continue

            flat = rows.reshape(-1)
            valid = flat >= 0
            dropped = int((~valid).sum())
            if dropped:
                self.dropped_sentinels += dropped
                flat = flat[valid]

            if flat.size == 0:
                continue
            if flat.max() >= self.num_experts:
                raise ValueError(
                    f"layer {layer_idx} selected expert id {int(flat.max())} but "
                    f"num_experts is {self.num_experts}; wrong config for this "
                    "capture"
                )

            self.tokens[slot] += np.bincount(flat, minlength=self.num_experts)
            self.layer_token_rows[slot] += rows.shape[0]

            if self.cooc is not None:
                self._add_cooc(slot, rows)

        self.n_sequences += 1
        if seq_len == 0:
            logger.warning_once("empty capture accumulated (seq_len=0)")

    def _add_cooc(self, slot: int, rows: np.ndarray) -> None:
        """Count unordered expert pairs selected by the same token."""
        assert self.cooc is not None
        top_k = rows.shape[1]
        if top_k < 2:
            return
        for i in range(top_k):
            for j in range(i + 1, top_k):
                a = rows[:, i]
                b = rows[:, j]
                keep = (a >= 0) & (b >= 0)
                if not keep.any():
                    continue
                a, b = a[keep], b[keep]
                # Symmetric: record both orderings so any later read is
                # orientation-independent.
                flat = a * self.num_experts + b
                counts = np.bincount(flat, minlength=self.num_experts**2)
                self.cooc[slot] += counts.reshape(
                    self.num_experts, self.num_experts
                )
                flat = b * self.num_experts + a
                counts = np.bincount(flat, minlength=self.num_experts**2)
                self.cooc[slot] += counts.reshape(
                    self.num_experts, self.num_experts
                )

    def finalize(self) -> None:
        """Flag configured layers that never yielded a usable row.

        A silent layer means the layer set is wrong -- either the config rule in
        :func:`~.layers.resolve_moe_layers` does not cover this architecture, or
        the capture was taken with a kernel that does not write routing. Either
        way the resulting profile is incomplete, and it must say so rather than
        report those layers as having no hot experts.
        """
        self.silent_layers = {
            layer
            for slot, layer in enumerate(self.moe_layers)
            if self.layer_token_rows[slot] == 0
        }
        if self.silent_layers:
            logger.warning(
                "%d of %d configured MoE layers produced no routing rows: %s. "
                "The profile is incomplete for those layers -- do not prune "
                "them from this run.",
                len(self.silent_layers),
                len(self.moe_layers),
                sorted(self.silent_layers),
            )

    def layer_share(self) -> np.ndarray:
        """Per-layer expert share, rows summing to 1 (0 for silent layers)."""
        totals = self.tokens.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(totals > 0, self.tokens / totals, 0.0)
        return share

    def merge(self, other: ExpertStats) -> None:
        """Fold another accumulator in. Used to combine workers or shards."""
        if other.num_experts != self.num_experts:
            raise ValueError("num_experts mismatch")
        if other.moe_layers != self.moe_layers:
            raise ValueError("moe_layers mismatch")
        self.tokens += other.tokens
        self.layer_token_rows += other.layer_token_rows
        self.n_sequences += other.n_sequences
        self.dropped_degenerate_rows += other.dropped_degenerate_rows
        self.dropped_sentinels += other.dropped_sentinels
        if self.cooc is not None and other.cooc is not None:
            self.cooc += other.cooc
        elif other.cooc is not None:
            self.cooc = other.cooc.copy()


def accumulate(
    captures: Iterable[np.ndarray],
    *,
    num_experts: int,
    moe_layers: Sequence[int],
    top_k: int,
    with_cooc: bool = False,
) -> ExpertStats:
    """Aggregate a stream of per-sequence captures.

    ``top_k`` is required rather than inferred: at ``top_k == 1`` uncaptured
    dense rows are indistinguishable from a genuine selection of expert 0, so
    the caller must have supplied a trustworthy ``moe_layers``. Refusing here is
    the difference between an incomplete profile and a wrong one.
    """
    if top_k < 2 and not moe_layers:
        raise ValueError(
            "top_k == 1 leaves uncaptured dense rows indistinguishable from a "
            "genuine selection of expert 0; pass an explicit moe_layers"
        )
    stats = ExpertStats.empty(num_experts, moe_layers, with_cooc=with_cooc)
    for capture in captures:
        stats.add(capture)
    stats.finalize()
    return stats


def from_hf_config(
    config: Mapping[str, object], *, with_cooc: bool = False
) -> ExpertStats:
    """Build an empty accumulator sized from an HF config mapping."""
    from .layers import get_num_experts, resolve_moe_layers

    return ExpertStats.empty(
        get_num_experts(config),
        resolve_moe_layers(config),
        with_cooc=with_cooc,
    )
