# SPDX-License-Identifier: Apache-2.0
"""Stand-ins for the two vLLM test helpers the ported suite used.

The suite came from ``tests/kernels/moe/test_expert_lru_cache.py`` on the
in-tree prototype branch. It leaned on ``vllm.utils.torch_utils.set_random_seed``
and ``vllm.platforms.current_platform.has_device_capability``. Reimplementing
both here is what lets the disk-tier regression suite run without vLLM
installed -- the same reason the store modules themselves no longer import it.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Seed python, numpy and torch (host and device)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _has_device_capability(capability: int) -> bool:
    """True when the current CUDA device is at least ``capability``.

    ``capability`` is the packed major*10+minor form vLLM uses, e.g. 89 for
    sm_89 (the first architecture with native fp8-e4m3 support).
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= capability
