# SPDX-License-Identifier: Apache-2.0
"""Probe the machine, size the tier for it, and remember the answer.

Everything needed to configure the tier well is measurable in under a second --
the checkpoint's geometry from headers, free device memory, host RAM, free disk --
and the rules connecting them to a configuration are the ones this project
measured rather than guessed. So the configuration should not be a thing a human
assembles by hand from three tables, and it should not be recomputed on every
boot either.

**What gets decided, and on what evidence.**

``expert_cache_size``
    The lever. 24 -> 48 slots was 2.59x decode on OLMoE at the same tokens, and
    ~80% of that is simply fetching fewer bytes over the host->device link. The
    rule is to cover the *per-layer expert union of the serving batch*, which is
    bounded above by ``min(num_experts, max_num_seqs * top_k)`` and in practice
    saturates below it (measured 35.3 mean / 46 max of 64 at batch 8, where the
    bound says 64). We take the bound, cap it at what the memory budget allows,
    and let the runtime correct us: it warns, once per layer, when a decode step
    routes to more experts than there are slots. Estimate, then measure.
``ram_cache``
    ``>= num_experts`` or every eviction becomes a disk read: 48 -> 64 was 1.66x
    and, more tellingly, it is the difference between steady and erratic. Given
    away only when host RAM cannot hold it.
``fp8_store``
    A **space** mechanism, not a speed one: on a non-fp8 checkpoint it costs 1.11x
    decode and bit-exactness to halve disk and host RAM. So it is off unless the
    store or the RAM tier does not fit without it.
``split``
    ``"expert"`` when capacity lands below ``top_k``, because the token split
    cannot serve a token at that capacity. Not bit-exact -- it rounds each group's
    partial sum -- so it is only ever chosen when the alternative is not running.

**Why it is cached, and what invalidates it.** The probe is cheap but not free,
and the answer only changes when the machine or the request does. The fingerprint
covers the checkpoint's identity and geometry, the serving shape, and the
resources -- with the resource figures **bucketed**, because free VRAM moves by a
few MiB between two consecutive reads and keying on exact bytes would mean the
cache never hits, which is the opposite of the point. Bucketing makes "the
environment is unchanged" a stable statement instead of an exact one.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from .._logging import init_logger
from .budget import CheckpointGeometry, analyze, plan_for_vram

logger = init_logger(__name__)

#: Resource readings are rounded down to this many GiB before they reach the
#: fingerprint. Small enough that a real change in the machine invalidates the
#: cache, coarse enough that ordinary jitter does not.
_BUCKET_GIB = 0.5

#: Bumped when the decision logic changes, so a cached answer from older logic is
#: not served for a machine that would now be configured differently.
_LOGIC_VERSION = 3


def _bucket(gib: float) -> float:
    return round(int(gib / _BUCKET_GIB) * _BUCKET_GIB, 2)


@dataclass
class Environment:
    """What the machine looks like right now, as far as the tier cares."""

    vram_gib: float = 0.0
    host_ram_gib: float = 0.0
    disk_free_gib: float = 0.0
    device: str = "unknown"
    #: Anything that could not be read, named rather than defaulted silently.
    unknown: list[str] = field(default_factory=list)


def probe(store_dir: str) -> Environment:
    """Read free device memory, host RAM and disk. Never raises.

    A probe that fails is reported as an unknown rather than as a zero: a zero
    would silently size the tier for a machine with no memory, and the caller can
    tell the difference between "measured small" and "could not measure".
    """
    env = Environment()

    try:
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            env.vram_gib = free / 1024**3
            env.device = torch.cuda.get_device_name(0)
        else:
            env.unknown.append("vram (no CUDA device visible)")
    except Exception as exc:  # pragma: no cover - torch absent or driver missing
        env.unknown.append(f"vram ({type(exc).__name__})")

    try:
        # os.sysconf is enough and avoids a psutil dependency for one number.
        env.host_ram_gib = (
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        )
    except Exception as exc:  # pragma: no cover - non-POSIX
        env.unknown.append(f"host ram ({type(exc).__name__})")

    try:
        target = store_dir
        while target and not os.path.exists(target):
            parent = os.path.dirname(os.path.abspath(target))
            if parent == target:
                break
            target = parent
        usage = os.statvfs(target or ".")
        free = usage.f_bavail * usage.f_frsize
        # Credit the bytes already inside the store back to "free": the question
        # decide() asks is "can the store live here", and the answer must not change
        # once the store exists. Without this, the second boot measured a disk
        # exactly one store smaller, concluded the store did not fit, and silently
        # flipped to fp8 -- which rebuilds the store it was worried about, slower
        # and no longer bit-exact.
        if os.path.isdir(store_dir):
            for root, _dirs, files in os.walk(store_dir):
                for name in files:
                    try:
                        free += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        pass
        env.disk_free_gib = free / 1024**3
    except Exception as exc:  # pragma: no cover - non-POSIX
        env.unknown.append(f"disk ({type(exc).__name__})")

    return env


@dataclass
class AutoConfig:
    """A serving configuration, and the reasoning that produced it."""

    surgeon: dict[str, Any]
    environment: Environment
    #: One line per decision, in the order they were made.
    why: list[str] = field(default_factory=list)
    #: What the caller should know but the config cannot express.
    warnings: list[str] = field(default_factory=list)
    fingerprint: str = ""
    cached: bool = False
    #: The serving batch the capacity was sized for. Part of the recipe, not an
    #: annotation: capacity covers the union a batch of this size can reach, so a
    #: boot that does not enforce the same bound can route to more experts than
    #: there are slots while the tool has just said no step should re-fetch.
    max_num_seqs: int = 8

    def serve_args(self, model: str) -> list[str]:
        """The `vllm serve` argv this configuration corresponds to."""
        return [
            "vllm",
            "serve",
            model,
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--additional-config",
            json.dumps({"surgeon": self.surgeon}),
        ]


def fingerprint(
    geometry: CheckpointGeometry,
    env: Environment,
    *,
    checkpoint: str,
    store_dir: str,
    max_num_seqs: int,
    kv_reserve_gib: float,
) -> str:
    """A stable identity for "this model, this machine, this serving shape"."""
    payload = {
        "logic": _LOGIC_VERSION,
        "checkpoint": os.path.realpath(checkpoint),
        "store": os.path.realpath(store_dir),
        "experts": geometry.num_experts,
        "top_k": geometry.top_k,
        "layers": geometry.num_moe_layers,
        "hidden": geometry.hidden,
        "intermediate": geometry.intermediate,
        "serving_dtype_bytes": geometry.serving_dtype_bytes,
        "max_num_seqs": max_num_seqs,
        "kv_reserve_gib": kv_reserve_gib,
        # Bucketed: see the module docstring.
        "vram_gib": _bucket(env.vram_gib),
        "host_ram_gib": _bucket(env.host_ram_gib),
        "disk_free_gib": _bucket(env.disk_free_gib),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def cache_path(fp: str) -> str:
    root = os.environ.get("VLLM_MOE_SURGEON_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "moe-surgeon", "autoconfig"
    )
    return os.path.join(root, f"{fp}.json")


def load_cached(fp: str) -> AutoConfig | None:
    path = cache_path(fp)
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if data.get("fingerprint") != fp:
        return None
    data["environment"] = Environment(**data["environment"])
    config = AutoConfig(**data)
    config.cached = True
    return config


def save_cached(config: AutoConfig) -> None:
    path = cache_path(config.fingerprint)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(config), f, indent=2)
    except OSError as exc:  # pragma: no cover - unwritable cache is not fatal
        logger.debug("could not cache the autoconfig: %s", exc)


def resolve_kv_reserve(env: Environment, kv_reserve_gib: float | None) -> float:
    """The KV reserve, scaled to the card unless the caller chose one.

    A flat default is a decision about every machine at once, and 2.0 GiB was the
    wrong decision for exactly the machines this package exists for: on a 3.5 GiB
    laptop GPU it reserved over half the card for KV and strangled the expert
    cache to one slot -- capacity 1 against a proven hand-tuned 4, below top_k,
    forcing the non-bit-exact expert split for no reason. Measured live on that
    laptop. Fifteen percent of free VRAM, clamped to [0.5, 2.0], keeps the old
    behaviour on large cards and stops starving small ones.
    """
    if kv_reserve_gib is not None:
        return kv_reserve_gib
    if env.vram_gib <= 0:
        return 2.0
    # From the *bucketed* reading, for the same reason the fingerprint buckets
    # its resources: the VRAM probe wobbles by tens of MiB between boots, and a
    # reserve that followed the wobble would change the fingerprint and defeat
    # the cache on exactly the small cards the scaling exists for.
    return round(min(2.0, max(0.5, 0.15 * _bucket(env.vram_gib))), 2)


def decide(
    geometry: CheckpointGeometry,
    env: Environment,
    *,
    store_dir: str,
    max_num_seqs: int = 8,
    kv_reserve_gib: float | None = None,
) -> AutoConfig:
    """Turn a machine and a checkpoint into a configuration, with reasons."""
    why: list[str] = []
    warnings: list[str] = []
    resolved_kv = resolve_kv_reserve(env, kv_reserve_gib)
    if kv_reserve_gib is None and env.vram_gib > 0:
        why.append(
            f"KV reserve scaled to the card: {resolved_kv} GiB "
            f"(15% of {_bucket(env.vram_gib):.1f} GiB free after bucketing, "
            "clamped to [0.5, 2.0])"
        )

    # The union the serving batch can reach, which is what capacity should cover.
    union_bound = min(geometry.num_experts, max_num_seqs * geometry.top_k)
    why.append(
        f"a batch of {max_num_seqs} at top_k={geometry.top_k} can route to at most "
        f"{union_bound} experts per layer, so that is the capacity to aim for"
    )

    if env.vram_gib <= 0:
        warnings.append(
            "device memory could not be measured, so capacity is the union bound "
            "rather than a fitted number; pass --vram to size it deliberately"
        )
        capacity = union_bound
        fp8 = False
    else:
        plan = plan_for_vram(
            geometry,
            int(env.vram_gib * 1024**3),
            fp8_store=False,
            kv_cache_bytes=int(resolved_kv * 1024**3),
        )
        if plan is None:
            raise ValueError(
                f"{env.vram_gib:.1f} GiB free (minus {resolved_kv} GiB reserved for "
                "KV) does not fit this model at any capacity >= 1. Free the device, "
                "lower --kv-reserve, or serve a smaller checkpoint."
            )
        fitted = plan.capacity
        capacity = min(union_bound, fitted)
        if fitted < union_bound:
            why.append(
                f"memory fits {fitted} slots, below the union bound, so capacity is "
                f"{capacity}: expect the runtime to warn that steps are re-fetching"
            )
        else:
            why.append(
                f"memory fits {fitted} slots, at or above the bound, so capacity is "
                f"{capacity} and no step should have to re-fetch"
            )
        fp8 = False

    record_bf16 = geometry.record_bytes(fp8=False)
    record_fp8 = geometry.record_bytes(fp8=True)
    layers = geometry.num_moe_layers

    # ram_cache >= num_experts unless the host cannot hold it. Half the host is the
    # ceiling: the pinned pool is charged against the process, and on unified memory
    # it is charged against the device budget too.
    ram_budget_gib = env.host_ram_gib / 2 if env.host_ram_gib else 0.0
    want_ram = geometry.num_experts
    ram_bytes_bf16 = layers * want_ram * record_bf16
    if not ram_budget_gib:
        ram_cache = want_ram
        warnings.append(
            "host RAM could not be measured; assuming it holds every expert"
        )
    elif ram_bytes_bf16 <= ram_budget_gib * 1024**3:
        ram_cache = want_ram
        why.append(
            f"host RAM holds all {want_ram} experts "
            f"({ram_bytes_bf16 / 1024**3:.1f} GiB), so nothing spills to disk in "
            "steady state"
        )
    elif (
        record_fp8 < record_bf16
        and layers * want_ram * record_fp8 <= ram_budget_gib * 1024**3
    ):
        ram_cache = want_ram
        fp8 = True
        why.append(
            "host RAM holds every expert only in fp8, so fp8_store is on: it halves "
            "the store for ~1.11x decode and costs bit-exactness, which is the trade "
            "worth making when the alternative is reading disk on every eviction"
        )
    else:
        # fp8 shrinks a record only when the checkpoint is wider than one byte --
        # on an already-1-byte checkpoint the fp8 record is *larger* (it adds row
        # scales to the same payload), so reaching for it under pressure would
        # approve a store that fits even less than the one just measured not to.
        fp8 = record_fp8 < record_bf16
        record_small = record_fp8 if fp8 else record_bf16
        ram_cache = max(
            capacity,
            int(ram_budget_gib * 1024**3 / max(1, layers * record_small)),
        )
        warnings.append(
            f"host RAM holds only {ram_cache} of {geometry.num_experts} experts"
            f"{' even in fp8' if fp8 else ''}, so evictions will read disk; decode "
            "will be slower and more variable than the numbers in DECISIONS.md"
        )

    store_bytes = layers * geometry.num_experts * (record_fp8 if fp8 else record_bf16)
    if env.disk_free_gib and store_bytes > env.disk_free_gib * 1024**3:
        store_fp8 = layers * geometry.num_experts * record_fp8
        # Gate on the real fp8 store size, not an assumed halving: on a 1-byte
        # checkpoint the fp8 record is larger, and "assume half" would approve a
        # store the build then runs out of disk writing.
        if (
            not fp8
            and record_fp8 < record_bf16
            and store_fp8 <= env.disk_free_gib * 1024**3
        ):
            fp8 = True
            why.append(
                f"the full-width store needs {store_bytes / 1024**3:.1f} GiB and "
                f"{env.disk_free_gib:.1f} GiB is free, so fp8_store is on: its store "
                f"is {store_fp8 / 1024**3:.1f} GiB"
            )
        else:
            raise ValueError(
                f"the store needs {store_bytes / 1024**3:.1f} GiB but only "
                f"{env.disk_free_gib:.1f} GiB is free at {store_dir}"
            )

    surgeon: dict[str, Any] = {
        "expert_cache_size": capacity,
        "store_dir": store_dir,
        "ram_cache": max(ram_cache, capacity),
    }
    if fp8:
        surgeon["fp8_store"] = True
    if capacity < geometry.top_k:
        surgeon["split"] = "expert"
        warnings.append(
            f"capacity {capacity} is below top_k {geometry.top_k}, so the expert "
            "split is required: it rounds each group's partial sum and is not "
            "bit-exact. This is the run-at-all configuration, not the fast one."
        )

    return AutoConfig(
        surgeon=surgeon,
        environment=env,
        why=why,
        warnings=warnings,
        max_num_seqs=max_num_seqs,
    )


def autoconfigure(
    checkpoint: str,
    *,
    store_dir: str = "./store",
    max_num_seqs: int = 8,
    kv_reserve_gib: float | None = None,
    vram_gib: float | None = None,
    refresh: bool = False,
) -> AutoConfig:
    """Probe, decide, and cache -- or return the cached answer unchanged.

    The cache hit is the point: an unchanged machine serving an unchanged model
    should not re-probe on every boot.
    """
    geometry = analyze(checkpoint)
    env = probe(store_dir)
    if vram_gib is not None:
        env.vram_gib = vram_gib
        env.unknown = [u for u in env.unknown if not u.startswith("vram")]

    fp = fingerprint(
        geometry,
        env,
        checkpoint=checkpoint,
        store_dir=store_dir,
        max_num_seqs=max_num_seqs,
        # The resolved value, so "scaled default" and an explicit identical
        # override share an identity, and a re-measured card re-decides.
        kv_reserve_gib=resolve_kv_reserve(env, kv_reserve_gib),
    )

    if not refresh:
        cached = load_cached(fp)
        if cached is not None:
            return cached

    config = decide(
        geometry,
        env,
        store_dir=store_dir,
        max_num_seqs=max_num_seqs,
        kv_reserve_gib=kv_reserve_gib,
    )
    config.fingerprint = fp
    save_cached(config)
    return config
