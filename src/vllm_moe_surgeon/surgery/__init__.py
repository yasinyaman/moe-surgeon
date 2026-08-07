# SPDX-License-Identifier: Apache-2.0
"""Deciding and applying expert surgery. No vLLM, no GPU.

:mod:`.descriptors` fingerprints experts in a way that is blind to the FFN's
neuron-permutation symmetry -- without that, weight-space similarity is close to
meaningless and every merge decision would be arbitrary. :mod:`.plan` turns a
usage profile plus a budget into a reviewable, hand-editable placement.
"""

from .descriptors import (
    CheckpointIndex,
    ExpertDescriptor,
    describe_layer,
    iter_layer_similarity,
    similarity_matrix,
    subspace_basis,
    subspace_similarity,
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

__all__ = [
    "Budget",
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
