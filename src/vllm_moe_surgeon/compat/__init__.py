# SPDX-License-Identifier: Apache-2.0
"""The only place in this package that knows vLLM's internal names.

Nothing outside ``compat/`` may import ``vllm``. :mod:`.seams` declares what we
hold and why; the version modules bind it. ``tests/test_seams.py`` enforces the
declaration, and ``tests/test_layering.py`` enforces the import rule.
"""

from .seams import SEAMS, Seam, SeamProblem, check, check_all, check_all_static


def installed_version() -> str | None:
    """The installed vLLM's version, or ``None`` if vLLM is not importable.

    Lives here rather than in the CLI so that ``import vllm`` stays inside
    ``compat/`` -- the layering test treats even a function-local import as a
    dependency, which is the behaviour we want.
    """
    try:
        import vllm
    except ImportError:
        return None
    return getattr(vllm, "__version__", "unknown")


__all__ = [
    "SEAMS",
    "Seam",
    "SeamProblem",
    "check",
    "check_all",
    "check_all_static",
    "installed_version",
]
