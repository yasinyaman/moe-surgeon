# SPDX-License-Identifier: Apache-2.0
"""Write experts into the store as the checkpoint arrives, out of tree.

Without this, serving through the tier costs *more* memory than not using it at all.
Measured on GB10, OLMoE-1B-7B: untiered boots in 14.60 GiB, the tier in 23.40 GiB. The
dominant term is the page-locked ``[num_experts, ...]`` set that ``create_weights``
allocates and the loader fills -- 12.0 GiB -- which on a unified-memory device is
charged against ``gpu_memory_utilization`` (``MemorySnapshot.measure`` substitutes
psutil availability for cudaMemGetInfo's free value on integrated GPUs).

Streaming removes that term rather than shrinking it: the parameters are allocated with
**zero** experts, each arriving shard is written into a staging record, and the record
is pwritten and dropped the moment its three shards complete. Peak cost is the experts
in flight, not the model. It also leaves nothing for ``device_loading_context`` to
hoist to the device before ``process_weights_after_loading`` runs, and nothing for
``replace_parameter`` to release.

**What this is not.** On a discrete GPU it saves zero *device* bytes, because under the
tier the full set was already never device-resident -- ``create_weights`` allocates
``device="cpu"`` deliberately. This is a host-memory feature that becomes a
device-memory feature only where the two come from one pool. Said plainly because the
opposite is the easy assumption to make.

The interception point is chosen to keep the port's one-substituted-class shape. The
prototype intercepted inside ``RoutedExperts.weight_loader``, which meant subclassing
a 1700-line class. Here the loader vLLM would have used is handed to ``create_weights``
in ``extra_weight_attrs`` and stamped onto each parameter -- so wrapping it there needs
no subclass, and anything that is not an expert weight passes through untouched.
"""

from __future__ import annotations

import os
from typing import Any

from .._logging import init_logger

logger = init_logger(__name__)

#: The three shards that make up one expert. ``w1`` is gate, ``w3`` is up -- together
#: they are the record's ``w13`` field, in that order -- and ``w2`` is down.
EXPERT_SHARDS = ("w1", "w2", "w3")

#: vLLM's ``weight_loader`` positional parameter order, so a positional call
#: (deepseek_v2, granitemoe) can be bound to names. Matches
#: ``RoutedExperts.weight_loader(param, loaded_weight, weight_name, shard_id,
#: expert_id, return_success)``.
LOADER_POSITIONAL = (
    "param",
    "loaded_weight",
    "weight_name",
    "shard_id",
    "expert_id",
    "return_success",
)


class StreamLoader:
    """Turns arriving expert shards into store records, one layer's worth.

    One instance per MoE layer, created before any weight arrives because the record
    layout follows from the config alone.
    """

    def __init__(
        self,
        store: Any,
        *,
        intermediate_size: int,
        params_dtype: Any,
        quantize: bool,
        layer_name: str,
        map_expert_id: Any = None,
    ) -> None:
        self.store = store
        self.intermediate_size = intermediate_size
        self.params_dtype = params_dtype
        self.quantize = quantize
        self.layer_name = layer_name
        self._map_expert_id = map_expert_id
        # Per in-flight expert. A checkpoint that interleaves experts costs a few
        # records, not the model -- which is the whole point of the exercise.
        self._staging: dict[int, Any] = {}
        self._parts: dict[int, set[str]] = {}
        self.records_written = 0

    # ---------------------------------------------------------------- construction

    @classmethod
    def open(
        cls,
        *,
        store_dir: str,
        layer_name: str,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Any,
        fp8: bool,
        model: str,
        revision: Any,
        map_expert_id: Any = None,
    ) -> StreamLoader:
        """Size and open the store from the config, before any weight arrives.

        The record specs and identity are byte-for-byte what the offline builder in
        :mod:`..surgery.tiered` writes, so a streamed store and a built one are
        interchangeable -- and a fingerprint match means the second boot reuses the
        first boot's work instead of restreaming it.
        """
        import torch

        from ..store.expert_disk_store import DiskExpertStore
        from ..surgery.tiered import store_path

        quantize = bool(fp8) and params_dtype in (torch.bfloat16, torch.float16)
        if fp8 and not quantize:
            raise ValueError(
                f"fp8 records target bf16/fp16 weights; this layer is {params_dtype}"
            )

        inter, hidden = intermediate_size, hidden_size
        if quantize:
            specs = [
                ("w13", (2 * inter, hidden), torch.float8_e4m3fn),
                ("w2", (hidden, inter), torch.float8_e4m3fn),
                ("w13_qs", (2 * inter,), torch.float32),
                ("w2_qs", (hidden,), torch.float32),
            ]
        else:
            specs = [
                ("w13", (2 * inter, hidden), params_dtype),
                ("w2", (hidden, inter), params_dtype),
            ]

        os.makedirs(store_dir, exist_ok=True)
        store = DiskExpertStore.create_for_streaming(
            store_path(store_dir, layer_name),
            num_experts,
            specs,
            identity={
                "model": model,
                "revision": str(revision),
                "layer": layer_name,
            },
            quant="fp8e4m3-row" if quantize else None,
        )
        return cls(
            store,
            intermediate_size=inter,
            params_dtype=params_dtype,
            quantize=quantize,
            layer_name=layer_name,
            map_expert_id=map_expert_id,
        )

    # -------------------------------------------------------------------- the hook

    def wrap(self, inner: Any) -> Any:
        """The loader to stamp on a parameter in place of ``inner``.

        Only the three expert shards are intercepted. Everything else -- biases,
        scales, anything a future vLLM adds -- goes to the loader vLLM chose, because
        this port has no business deciding how those load.
        """

        def stream_weight_loader(*args: Any, **kwargs: Any) -> Any:
            # vLLM's weight_loader contract, positional order. Some models call it by
            # keyword (the unified RoutedExperts path), others positionally with
            # shard_id/expert_id as kwargs (deepseek_v2, granitemoe). Bind both so a
            # positional expert shard is claimed rather than delegated to the
            # zero-expert placeholder, where param.data[expert_id] would index an
            # empty tensor and load nothing.
            bound = dict(zip(LOADER_POSITIONAL, args, strict=False))
            bound.update(kwargs)
            shard_id = bound.get("shard_id")
            expert_id = bound.get("expert_id")
            weight_name = bound.get("weight_name") or ""
            loaded_weight = bound.get("loaded_weight")
            return_success = bool(bound.get("return_success", False))

            mine = (
                shard_id in EXPERT_SHARDS
                and expert_id is not None
                and loaded_weight is not None
                and "bias" not in weight_name
            )
            if not mine:
                return inner(*args, **kwargs)

            # We own this shard. If more positional args arrived than the contract we
            # know, our name<-position mapping is not trustworthy, and writing a shard
            # into the wrong half of the wrong expert is silent. Refuse over guess.
            if len(args) > len(LOADER_POSITIONAL):
                raise ValueError(
                    f"stream_load received {len(args)} positional weight_loader args "
                    f"for an expert shard on {self.layer_name}, beyond the "
                    f"{len(LOADER_POSITIONAL)}-argument contract it can interpret. "
                    "Refusing to guess the argument order."
                )

            self.accept(shard_id, loaded_weight, int(expert_id))
            # The caller distinguishes "loaded" from "skipped" only when it asked to.
            return True if return_success else None

        return stream_weight_loader

    def accept(self, shard_id: str, loaded_weight: Any, expert_id: int) -> None:
        """Place one shard in its expert's staging record, writing when complete."""
        import torch

        store = self.store
        local_id = expert_id
        if self._map_expert_id is not None:
            local_id = int(self._map_expert_id(expert_id))
        if local_id < 0:
            return
        if store.is_complete:
            # A fingerprint-matched store was reused: there is nothing to write, and
            # the provider will read the existing records.
            return

        row = self._staging.get(local_id)
        if row is None:
            row = torch.zeros(store.record_stride, dtype=torch.uint8)
            self._staging[local_id] = row
            self._parts[local_id] = set()

        inter = self.intermediate_size
        scale_view = None
        if shard_id == "w2":
            view = store.field_view(row, "w2")
            if self.quantize:
                scale_view = store.field_view(row, "w2_qs")
        else:
            # w1 is the first `inter` rows of w13, w3 the second -- the order vLLM's
            # own loader uses. Swapping them loads without complaint and computes a
            # different function.
            start = 0 if shard_id == "w1" else inter
            view = store.field_view(row, "w13").narrow(0, start, inter)
            if self.quantize:
                scale_view = store.field_view(row, "w13_qs").narrow(0, start, inter)

        if tuple(view.shape) != tuple(loaded_weight.shape):
            raise ValueError(
                f"stream load shape mismatch for {self.layer_name} expert "
                f"{expert_id} {shard_id}: record {tuple(view.shape)} vs checkpoint "
                f"{tuple(loaded_weight.shape)}"
            )

        if scale_view is not None:
            from ..store.expert_disk_store import quantize_rowwise_fp8

            # Row-wise fp8 works shard-at-a-time because every shard delivers whole
            # output rows: w1/w3 are row ranges of w13 and w2 arrives entire. That is
            # what keeps the fp32 promotion inside quantize_rowwise_fp8 down to one
            # shard instead of the whole expert set.
            quantized, scales = quantize_rowwise_fp8(
                loaded_weight.to(self.params_dtype)
            )
            view.copy_(quantized)
            scale_view.copy_(scales)
        else:
            view.copy_(loaded_weight.to(view.dtype))

        parts = self._parts[local_id]
        parts.add(shard_id)
        if parts == set(EXPERT_SHARDS):
            store.write_record(local_id, row)
            self.records_written += 1
            del self._staging[local_id]
            del self._parts[local_id]

    # ------------------------------------------------------------------- finishing

    def finalize(self) -> Any:
        """Seal the store and return it, ready to hand to the provider.

        Refuses to seal with experts still in flight. An expert whose shards never all
        arrived would leave a zeroed record, and a zeroed expert is silent: the model
        loads and its output is quietly wrong for every token routed there.
        """
        if self._staging:
            incomplete = {
                expert: sorted(set(EXPERT_SHARDS) - self._parts.get(expert, set()))
                for expert in sorted(self._staging)
            }
            raise RuntimeError(
                f"stream load for {self.layer_name} has {len(self._staging)} "
                f"incomplete experts, missing shards {incomplete}. Sealing now would "
                "leave zeroed records, which serve silently wrong output."
            )
        self.store.finalize()
        if self.records_written:
            logger.info(
                "stream load %s: wrote %d records, no full expert tensor materialised",
                self.layer_name,
                self.records_written,
            )
        return self.store


__all__ = ["EXPERT_SHARDS", "StreamLoader"]
