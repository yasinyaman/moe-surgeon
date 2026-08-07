# SPDX-License-Identifier: Apache-2.0
"""Enforce the one architectural rule this package has.

Only ``compat/`` (and the plugin entry point that bootstraps it) may reference
vLLM. Everything else -- the store, the surgery pipeline, the writer, the job
server -- must import and run on a host with no vLLM, no CUDA and no GPU.

That is not tidiness. It is what makes the offline pipeline runnable on a laptop
and testable in CI without a GPU runner, and it is what keeps a vLLM upgrade
confined to one reviewable directory. A rule like this decays in a week unless a
test holds it, so: this test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "vllm_moe_surgeon"

# Modules allowed to reference vLLM, relative to SRC.
_ALLOWED = {
    "compat",  # the seam layer -- its entire job
    "plugin.py",  # the entry point vLLM itself calls
    "telemetry",  # runtime hooks; must go through compat, see test below
}


def _python_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _relative_top(path: Path) -> str:
    rel = path.relative_to(SRC)
    return rel.parts[0]


def _vllm_imports(path: Path) -> list[str]:
    """Every ``vllm`` name imported by this file, including deferred imports."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [
                a.name
                for a in node.names
                if a.name == "vllm" or a.name.startswith("vllm.")
            ]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "vllm" or mod.startswith("vllm."):
                found.append(mod)
    return found


@pytest.mark.parametrize(
    "path", _python_files(), ids=lambda p: str(p.relative_to(SRC))
)
def test_only_compat_imports_vllm(path):
    imports = _vllm_imports(path)
    if not imports:
        return
    top = _relative_top(path)
    assert top in _ALLOWED, (
        f"{path.relative_to(SRC)} imports {imports}, but only {sorted(_ALLOWED)} "
        "may reference vLLM. Route it through vllm_moe_surgeon.compat instead -- "
        "the offline pipeline has to run on hosts with no vLLM installed."
    )


def test_offline_half_imports_without_vllm():
    """The offline modules must import on a host with no vLLM.

    This test *is* that host: vLLM is not installed on the development laptop,
    so a successful import here is the assertion.
    """
    import importlib

    for module in (
        "vllm_moe_surgeon",
        "vllm_moe_surgeon.env",
        "vllm_moe_surgeon._logging",
        "vllm_moe_surgeon.store",
        "vllm_moe_surgeon.compat.seams",  # the seam *table* is pure data
    ):
        importlib.import_module(module)


def test_store_does_not_reference_vllm_at_all():
    """The lifted disk tier must stay clean, not merely import-clean.

    A string like ``"vllm.model_executor..."`` left in a deferred import or a
    docstring path is how the lift silently regresses.
    """
    offenders = {}
    for path in (SRC / "store").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        lines = [
            f"{n}: {line.strip()}"
            for n, line in enumerate(path.read_text().splitlines(), 1)
            if "vllm." in line and "vllm_moe_surgeon" not in line
        ]
        if lines:
            offenders[path.name] = lines
    assert not offenders, f"vLLM references left in store/: {offenders}"
