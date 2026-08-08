# SPDX-License-Identifier: Apache-2.0
"""Replay a routing trace through the cache policy, offline.

The residency prior exists to fix cold start: a fresh cache learns from traffic,
so the first requests after every restart pay misses on experts a previous run
already knew were hot. Whether the prior actually helps is a measurement, and this
is the cheap way to take it -- pure numpy over captures already collected, no GPU,
no engine, no store on disk.

It also restores a capability the prototype lost. Its docstrings referenced a
``bench/hit_rate_sweep.py`` as the consumer of the routing trace; that script is
not on the branch. This is the same idea, in the package, with tests.

What is simulated is deliberately narrow: **which experts are resident**, under the
real :mod:`.expert_policy` implementations, given the real per-token routing order.
Not simulated: read latency, pipelining, the RAM tier. So the output is hit rate
and its shape over time, not throughput -- the quantity a prior can move.

Accounting note that decides whether the numbers mean anything: a token needing
``top_k`` experts is charged ``top_k`` accesses, and misses within one token are
counted individually. Charging a token as one access would hide exactly the effect
a small cache has.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._logging import init_logger

logger = init_logger(__name__)


@dataclass
class ReplayResult:
    """Hit accounting for one layer, or summed over layers."""

    hits: int = 0
    misses: int = 0
    #: Hit/miss outcome per access, in order, so a prefix can be scored.
    outcomes: list[bool] = field(default_factory=list)

    @property
    def accesses(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0

    def prefix_hit_rate(self, n: int) -> float:
        """Hit rate over the first ``n`` accesses -- the cold-start window."""
        window = self.outcomes[:n]
        return sum(window) / len(window) if window else 0.0

    def merge(self, other: ReplayResult) -> None:
        self.hits += other.hits
        self.misses += other.misses
        self.outcomes.extend(other.outcomes)


def replay_layer(
    accesses: Iterable[Sequence[int]],
    *,
    num_experts: int,
    capacity: int,
    policy_factory: Callable[[int], Any],
    prior: dict[int, float] | None = None,
    prewarm: Sequence[int] | None = None,
) -> ReplayResult:
    """Walk one layer's per-token expert sets through a capacity-limited cache.

    ``prior`` seeds the policy's ``prior`` table (a warm start by scoring).
    ``prewarm`` additionally *places* those experts resident before the first
    token, which is what a real warm start would do -- the prior alone only biases
    eviction, and with an empty cache there is nothing to evict yet.
    """
    if capacity < 1:
        raise ValueError("capacity must be >= 1")

    policy = policy_factory(num_experts)
    if prior:
        if hasattr(policy, "prior"):
            policy.prior = dict(prior)
        else:
            logger.warning_once(
                "%s has no prior table; the warm-start arm will behave like cold",
                type(policy).__name__,
            )

    # expert -> (freq, last_touch), mirroring the provider's residency bookkeeping.
    resident: dict[int, list[int]] = {}
    clock = 0
    result = ReplayResult()

    if prewarm:
        for expert in list(prewarm)[:capacity]:
            resident[int(expert)] = [0, 0]
            policy.on_insert(int(expert))

    for token_experts in accesses:
        # Experts this token needs but does not have. Resolved one at a time so a
        # token wider than the cache thrashes, exactly as it would in the tier.
        for expert in token_experts:
            expert = int(expert)
            if expert < 0:
                continue
            clock += 1
            if expert in resident:
                result.hits += 1
                result.outcomes.append(True)
                entry = resident[expert]
                entry[0] += 1
                entry[1] = clock
                policy.on_hit(expert)
                continue

            result.misses += 1
            result.outcomes.append(False)
            if len(resident) >= capacity:
                # Never evict an expert this same token still needs -- the bug the
                # prototype hit, where a freshly loaded expert had the lowest
                # score and was evicted mid-token, silently serving wrong weights.
                protected = {int(e) for e in token_experts if int(e) >= 0}
                victim = None
                worst = None
                for candidate, (freq, last) in resident.items():
                    if candidate in protected:
                        continue
                    score = policy.score(candidate, freq, last, clock)
                    if worst is None or score < worst:
                        victim, worst = candidate, score
                if victim is None:
                    # Everything resident is needed by this token: the cache is
                    # smaller than the token's expert set.
                    raise RuntimeError(
                        f"capacity {capacity} cannot hold one token's "
                        f"{len(protected)} experts; the tier would have to split "
                        "the forward pass"
                    )
                del resident[victim]
                policy.on_evict(victim)

            resident[expert] = [1, clock]
            policy.on_insert(expert)

    return result


def replay_captures(
    captures: Iterable[np.ndarray],
    *,
    moe_layers: Sequence[int],
    num_experts: int,
    capacity: int,
    policy_factory: Callable[[int], Any],
    priors: dict[int, dict[int, float]] | None = None,
    prewarm: dict[int, Sequence[int]] | None = None,
) -> dict[int, ReplayResult]:
    """Replay every layer of a sequence of ``[seq_len, layers, top_k]`` captures.

    Captures are consumed once, so they are materialised per layer. Each layer has
    its own independent cache, which is how the tier works.
    """
    per_layer_rows: dict[int, list[np.ndarray]] = {layer: [] for layer in moe_layers}
    for capture in captures:
        arr = np.asarray(capture)
        for layer in moe_layers:
            per_layer_rows[layer].append(arr[:, layer, :])

    results: dict[int, ReplayResult] = {}
    for layer in moe_layers:
        rows = per_layer_rows[layer]
        if not rows:
            continue
        stacked = np.concatenate(rows, axis=0)
        # Uncaptured rows repeat one id; a genuine top-k row cannot.
        if stacked.shape[1] >= 2:
            stacked = stacked[~np.all(stacked == stacked[:, :1], axis=1)]
        results[layer] = replay_layer(
            stacked,
            num_experts=num_experts,
            capacity=capacity,
            policy_factory=policy_factory,
            prior=(priors or {}).get(layer),
            prewarm=(prewarm or {}).get(layer),
        )
    return results


def summarize(results: dict[int, ReplayResult], prefix: int = 2000) -> ReplayResult:
    """Sum layers into one result, for a single headline number."""
    total = ReplayResult()
    for layer in sorted(results):
        total.merge(results[layer])
    return total
