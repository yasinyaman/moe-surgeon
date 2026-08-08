# SPDX-License-Identifier: Apache-2.0
"""``hot_experts.json`` -- the residency hint, and the seam it finally fills.

``EWMAPolicy`` in the lifted disk tier has carried an empty ``prior`` dict since
the prototype, documented as "the manifest seam: a table loaded at boot that
biases residency toward experts that were hot in a previous run. Empty until
seeded". Nothing ever seeded it. This module is what does.

Why it matters more than it sounds: the cache starts cold and learns residency
from traffic, so the first requests after every restart pay misses on experts a
previous run already knew were hot. A profile is exactly the knowledge needed to
skip that, and the pipeline has one.

**Scale is the whole design question.** ``EWMAPolicy.score`` adds
``w_m * prior[e]`` to an EWMA term that grows by about 1.0 per cache event and
decays at 0.999. So a prior of 1.0 is worth roughly "one recent hit". The prior
must be large enough to decide residency during the cold-start window and small
enough that observed traffic overrules it soon after -- it is advisory, and a hint
that never yields to reality is a bug, not a stronger hint. :data:`DEFAULT_SCALE`
gives the hottest expert of a layer a head start of a few hits, which the EWMA
passes within a few hundred events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ._logging import init_logger

logger = init_logger(__name__)

MANIFEST_VERSION = 1

#: Multiplier on an expert's per-layer load share. See the module docstring: a
#: prior of ~1.0 is worth one recent cache hit, so this hands the hottest expert
#: of a layer a lead of a few hits and lets traffic overtake it quickly.
DEFAULT_SCALE = 8.0


@dataclass
class HotExperts:
    """Per-layer residency hints, keyed by global decoder-layer index."""

    #: layer -> {expert id -> prior score}
    priors: dict[int, dict[int, float]]
    #: layer -> the expert ids the plan wanted resident, hottest first
    core: dict[int, list[int]]
    model: str | None = None
    revision: str | None = None
    scale: float = DEFAULT_SCALE
    provenance: dict[str, Any] | None = None

    def for_layer(self, layer: int) -> dict[int, float]:
        return self.priors.get(layer, {})

    def for_layer_name(self, layer_name: str) -> dict[int, float]:
        """Look up by vLLM's ``RoutedExperts.layer_name``.

        The store and the provider are keyed by that name, while a plan is keyed
        by layer index, so the translation has to live somewhere. Here, with the
        parsing rule stated once.
        """
        index = layer_index_from_name(layer_name)
        if index is None:
            logger.warning_once(
                "could not read a layer index out of %r; residency hints will "
                "not be applied to it",
                layer_name,
            )
            return {}
        return self.for_layer(index)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {
                "manifest_version": MANIFEST_VERSION,
                "model": self.model,
                "revision": self.revision,
                "scale": self.scale,
                "provenance": self.provenance or {},
                # JSON keys are strings; the loader restores the ints.
                "priors": {
                    str(layer): {str(e): p for e, p in experts.items()}
                    for layer, experts in self.priors.items()
                },
                "core": {str(layer): ids for layer, ids in self.core.items()},
            },
            indent=indent,
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())
            f.write("\n")


def layer_index_from_name(layer_name: str) -> int | None:
    """``"model.layers.7.mlp.experts"`` -> ``7``.

    Takes the first integer component, matching vLLM's own
    ``extract_layer_index``.
    """
    for part in layer_name.split("."):
        if part.isdigit():
            return int(part)
    return None


def load(path: str) -> HotExperts:
    with open(path) as f:
        payload = json.load(f)
    if payload.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(
            f"{path}: manifest_version {payload.get('manifest_version')} "
            f"!= {MANIFEST_VERSION}"
        )
    return HotExperts(
        priors={
            int(layer): {int(e): float(p) for e, p in experts.items()}
            for layer, experts in payload["priors"].items()
        },
        core={int(layer): list(ids) for layer, ids in payload["core"].items()},
        model=payload.get("model"),
        revision=payload.get("revision"),
        scale=float(payload.get("scale", DEFAULT_SCALE)),
        provenance=payload.get("provenance") or {},
    )


def from_plan(plan: Any, *, scale: float = DEFAULT_SCALE) -> HotExperts:
    """Derive residency hints from a plan.

    The prior is the expert's per-layer *load share* times ``scale``, not its
    rank: sharing 20% of a layer's traffic should count for more than being
    third in an almost-flat distribution, and the share carries that where a
    rank does not.

    Experts the plan deletes get no entry -- they will not exist to be resident.
    Experts placed on disk get a share-proportional prior too, just a small one,
    so a genuinely busy "cold" expert can still earn a slot.
    """
    priors: dict[int, dict[int, float]] = {}
    core: dict[int, list[int]] = {}

    for placement in plan.placements:
        if placement.action == "drop":
            continue
        layer_priors = priors.setdefault(placement.layer, {})
        layer_priors[placement.expert] = float(placement.share) * scale
        if placement.action == "merge_into_core":
            core.setdefault(placement.layer, []).append(placement.expert)

    for layer, ids in core.items():
        ids.sort(key=lambda e: -priors[layer][e])

    return HotExperts(
        priors=priors,
        core=core,
        model=getattr(plan, "model", None),
        revision=getattr(plan, "revision", None),
        scale=scale,
        provenance={
            "from_plan_budget": getattr(plan, "budget", {}),
            "plan_provenance": getattr(plan, "provenance", {}),
        },
    )


def seed_policy(policy: Any, priors: dict[int, float]) -> bool:
    """Install ``priors`` on a cache policy that supports them.

    Returns whether it took. ``LFRUPolicy`` has no ``prior`` attribute, so the
    hint is silently inapplicable under the default policy -- which the caller
    needs to know rather than assume, since an unseeded cache is not an error,
    just a cold one.
    """
    if not hasattr(policy, "prior"):
        logger.warning_once(
            "%s has no prior table; residency hints need VLLM_MOE_CACHE_POLICY="
            "ewma to have any effect",
            type(policy).__name__,
        )
        return False
    policy.prior = dict(priors)
    return True
