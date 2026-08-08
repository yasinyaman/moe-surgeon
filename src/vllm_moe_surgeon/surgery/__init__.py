# SPDX-License-Identifier: Apache-2.0
"""Deciding and applying expert surgery. No vLLM, no GPU.

:mod:`.descriptors` fingerprints experts in a way that is blind to the FFN's
neuron-permutation symmetry -- without that, weight-space similarity is close to
meaningless and every merge decision would be arbitrary. :mod:`.plan` turns a
usage profile plus a budget into a reviewable, hand-editable placement.
"""

from .align import align_donor_to_target, apply_permutation, merge_experts
from .apply import apply_plan, derive_surgery, rewrite_config
from .budget import CheckpointGeometry, TierPlan, floor_plan, plan_for_vram
from .descriptors import (
    CheckpointIndex,
    ExpertDescriptor,
    describe_layer,
    iter_layer_similarity,
    similarity_matrix,
    subspace_basis,
    subspace_similarity,
)
from .inspect import (
    LayerSignals,
    effective_rank,
    inspect_checkpoint,
    inspect_layer,
    score_against_profile,
    spearman,
)
from .plan import (
    Budget,
    ExpertPlacement,
    Plan,
    ProfileTooThin,
    build_plan,
    coverage,
    load_plan,
    summarize_plan,
    validate_plan,
)
from .recommend import Recommendation, recommend

__all__ = [
    "Budget",
    "Recommendation",
    "recommend",
    "CheckpointGeometry",
    "TierPlan",
    "floor_plan",
    "plan_for_vram",
    "LayerSignals",
    "effective_rank",
    "inspect_checkpoint",
    "inspect_layer",
    "score_against_profile",
    "spearman",
    "align_donor_to_target",
    "apply_permutation",
    "apply_plan",
    "derive_surgery",
    "merge_experts",
    "rewrite_config",
    "CheckpointIndex",
    "ExpertDescriptor",
    "ExpertPlacement",
    "Plan",
    "ProfileTooThin",
    "build_plan",
    "coverage",
    "describe_layer",
    "iter_layer_similarity",
    "load_plan",
    "similarity_matrix",
    "subspace_basis",
    "subspace_similarity",
    "summarize_plan",
    "validate_plan",
]
