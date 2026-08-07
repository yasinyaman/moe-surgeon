# SPDX-License-Identifier: Apache-2.0
"""The early-warning test for vLLM upgrades.

Each seam in :data:`vllm_moe_surgeon.compat.seams.SEAMS` gets its own test, so a
failure names the moved symbol and prints why we were holding it. That is the
whole point: without this, a vLLM upgrade that renames ``_apply_quant_method``
does not raise anywhere -- our override just stops being called, and the profiler
quietly reports that no expert was ever used.

Two levels, because they run in different places:

- **static** -- parses a vLLM source tree, no import, no torch, no CUDA. Runs on
  a development laptop. Point it at a checkout with ``MOE_SURGEON_VLLM_SRC``.
- **import** -- resolves each seam against the installed vLLM and checks
  signatures. Runs where vLLM is installed, i.e. the GPU boxes and CI.

Before shipping, run the import level against the pinned vLLM. To price an
upgrade, run it against the next release candidate.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vllm_moe_surgeon.compat.seams import SEAMS, check, check_static

_IDS = [s.target for s in SEAMS]


def _failure_message(seam, problem, version: str) -> str:
    return (
        f"\nSeam moved: {seam.target}"
        f"\n  kind:  {seam.kind}"
        f"\n  tier:  {seam.tier}"
        f"\n  found: {problem.detail}"
        f"\n  we hold it because: {seam.why}"
        f"\n\nvLLM: {version}"
    )


# ----------------------------------------------------------------------
# Table hygiene. No vLLM required, so these always run.
# ----------------------------------------------------------------------


def test_seam_table_is_well_formed():
    """No duplicates, and every entry explains itself."""
    targets = [s.target for s in SEAMS]
    duplicates = {t for t in targets if targets.count(t) > 1}
    assert not duplicates, f"duplicate seam entries: {duplicates}"

    for seam in SEAMS:
        assert ":" in seam.target, f"{seam.target} must be 'module:QualName'"
        assert seam.module.startswith("vllm"), f"{seam.target} is not a vLLM name"
        # A seam without a stated reason is a seam nobody can safely delete.
        assert len(seam.why) > 40, f"{seam.target} needs a real 'why'"


def test_optional_seams_are_excluded_from_the_pin_check():
    """A missing optional seam must never fail the pin.

    ``check_all`` is what a version bump gates on. Optional seams exist to
    record upstream helpers we want to adopt as they appear -- counting their
    absence as a break would make the gate unusable on the pinned version,
    which is precisely the version that does not have them yet.
    """
    from vllm_moe_surgeon.compat.seams import SEAMS as table

    optional = [s for s in table if not s.required]
    assert optional, "the table should track at least one emerging upstream seam"
    for seam in optional:
        assert seam.why, f"{seam.target} must say what it would buy us"


def test_no_reliance_on_fork_only_api():
    """``supports_expert_lru_cache`` was added by the in-tree prototype.

    It does not exist upstream. If a seam ever declares it, the package has
    quietly grown a dependency on the fork it is supposed to replace.
    """
    assert not any("supports_expert_lru_cache" in s.target for s in SEAMS)


def test_internal_seams_are_acknowledged():
    """Most of what we hold is private. Keep that visible, not surprising."""
    internal = [s for s in SEAMS if s.tier == "internal"]
    assert internal, "table should not claim everything is documented API"
    # Reviewers should see the ratio move when someone adds a private seam.
    assert len(internal) <= 14, (
        f"{len(internal)} internal seams -- if this grew on purpose, raise the "
        "bound deliberately and say why in the commit"
    )


# ----------------------------------------------------------------------
# Static level: needs only a source tree.
# ----------------------------------------------------------------------


def _source_root() -> str | None:
    explicit = os.environ.get("MOE_SURGEON_VLLM_SRC")
    if explicit:
        return explicit
    # Convenience for this workspace: the fork sits next to us.
    guess = Path(__file__).resolve().parents[2] / "vllm"
    return str(guess) if (guess / "vllm" / "__init__.py").exists() else None


@pytest.mark.parametrize("seam", SEAMS, ids=_IDS)
def test_seam_present_in_source_tree(seam):
    root = _source_root()
    if root is None:
        pytest.skip("set MOE_SURGEON_VLLM_SRC to a vLLM checkout")
    problem = check_static(seam, root)
    if problem is None:
        return
    if not seam.required:
        pytest.skip(f"optional seam unavailable here: {problem.detail}")
    pytest.fail(_failure_message(seam, problem, f"source tree at {root}"))


# ----------------------------------------------------------------------
# Import level: needs vLLM installed.
# ----------------------------------------------------------------------


@pytest.mark.needs_vllm
@pytest.mark.parametrize("seam", SEAMS, ids=_IDS)
def test_seam_still_holds(seam):
    vllm = pytest.importorskip("vllm", reason="import-level seam check needs vLLM")
    problem = check(seam)
    if problem is None:
        return
    if not seam.required:
        pytest.skip(f"optional seam unavailable here: {problem.detail}")
    pytest.fail(
        _failure_message(seam, problem, getattr(vllm, "__version__", "unknown"))
    )
