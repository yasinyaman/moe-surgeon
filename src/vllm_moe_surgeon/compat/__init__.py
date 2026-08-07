# SPDX-License-Identifier: Apache-2.0
"""The only place in this package that knows vLLM's internal names.

Nothing outside ``compat/`` may import ``vllm``. :mod:`.seams` declares what we
hold and why; the version modules bind it. ``tests/test_seams.py`` enforces the
declaration, and ``tests/test_layering.py`` enforces the import rule.
"""

from .seams import SEAMS, Seam, SeamProblem, check, check_all, check_all_static

__all__ = [
    "SEAMS",
    "Seam",
    "SeamProblem",
    "check",
    "check_all",
    "check_all_static",
]
