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
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "vllm_moe_surgeon"

# Modules allowed to reference vLLM, relative to SRC.
_ALLOWED = {
    "compat",  # the seam layer -- its entire job
    "plugin.py",  # the entry point vLLM itself calls
}


def _python_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _relative_top(path: Path) -> str:
    rel = path.relative_to(SRC)
    return rel.parts[0]


def _is_vllm(name: str) -> bool:
    return name == "vllm" or name.startswith("vllm.")


def _vllm_imports_from_source(source: str, filename: str = "<memory>") -> list[str]:
    """Every ``vllm`` name imported by this source, static *and* dynamic.

    Static ``import``/``from`` are the common case. Dynamic imports --
    ``importlib.import_module("vllm...")`` and ``__import__("vllm...")`` -- evade a
    plain import walk, so they are matched here too: a module-level dynamic vLLM
    import in the offline half would still break collection on a vLLM-free host, but
    a function-local one would not, and this is the check that catches it.
    """
    tree = ast.parse(source, filename=filename)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if _is_vllm(a.name)]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if _is_vllm(mod):
                found.append(mod)
        elif isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else None
            )
            if name in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if _is_vllm(arg.value):
                        found.append(arg.value)
    return found


def _vllm_imports(path: Path) -> list[str]:
    """Every ``vllm`` name imported by this file, including deferred imports."""
    return _vllm_imports_from_source(path.read_text(), filename=str(path))


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
    docstring path is how the lift silently regresses. ``store/`` is held to the
    strict line-level rule because it is kept byte-comparable to the fork.
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


_MODULE_PATH = re.compile(r"vllm(\.[A-Za-z_][A-Za-z0-9_]*)+\Z")


def test_no_offline_module_holds_a_vllm_import_string():
    """A string backstop over the whole offline half, precise enough for prose.

    Scans string *literals* (not lines) for a bare ``vllm`` module path like
    ``"vllm.config"`` -- the kind that would be handed to ``import_module`` -- across
    every module except ``compat/`` and the plugin. A docstring that merely mentions
    ``vllm.envs`` mid-sentence is one long string, not a module-path literal, so it
    is not a false positive; a standalone ``"vllm.config"`` is.
    """
    offenders: dict[str, list[str]] = {}
    for path in _python_files():
        if _relative_top(path) in _ALLOWED:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        hits = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _MODULE_PATH.fullmatch(node.value)
        ]
        if hits:
            offenders[str(path.relative_to(SRC))] = hits
    assert not offenders, f"vLLM module-path strings outside compat/: {offenders}"


def test_the_ast_check_catches_dynamic_imports():
    """importlib.import_module / __import__ evade a plain import walk; the layering
    check must see them, or a dynamic vLLM import in the offline half sails through."""
    assert _vllm_imports_from_source(
        "import importlib\nimportlib.import_module('vllm.config')"
    ) == ["vllm.config"]
    assert _vllm_imports_from_source("__import__('vllm')") == ["vllm"]
    assert _vllm_imports_from_source("from vllm.x import y") == ["vllm.x"]
    # A non-vLLM dynamic import is not a false positive.
    assert _vllm_imports_from_source("import_module('numpy')") == []
