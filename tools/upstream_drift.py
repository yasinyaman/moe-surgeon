#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Price a vLLM upgrade before taking it.

Extracts only the modules the seam table names, at two git refs of a vLLM
checkout, and reports what each seam does at each ref. No vLLM install, no
torch, no GPU -- it reads blobs out of git and parses them, so it runs on a
laptop against a partial clone.

This is the operational form of the project's central claim. The predecessor
carried ~800 lines of hooks inside upstream files, so every release was a merge
argument with no way to see the bill in advance. Here the bill is a table.

    python tools/upstream_drift.py --repo ../vllm --from origin/main --to FETCH_HEAD

Exit status is 1 if a *required* seam broke at the target ref, so this can gate
a pin bump in CI. Optional seams that appeared are reported as adopt-me news and
never fail the run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm_moe_surgeon.compat.seams import SEAMS, check_static  # noqa: E402


def _git(repo: str, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _module_files() -> list[str]:
    """Repo-relative paths for every module the seam table touches."""
    paths = set()
    for seam in SEAMS:
        rel = seam.module.replace(".", "/")
        paths.add(rel)
    return sorted(paths)


def _materialize(repo: str, ref: str, dest: Path) -> list[str]:
    """Extract the seam modules at ``ref`` into ``dest``. Returns missing paths."""
    missing = []
    for rel in _module_files():
        blob = None
        for candidate in (f"{rel}.py", f"{rel}/__init__.py"):
            try:
                blob = _git(repo, "show", f"{ref}:{candidate}")
            except subprocess.CalledProcessError:
                continue
            out = dest / candidate
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(blob)
            break
        if blob is None:
            missing.append(rel)
    return missing


def _describe(ref: str, repo: str) -> str:
    try:
        return _git(repo, "log", "-1", "--format=%h %ci %s", ref).strip()
    except subprocess.CalledProcessError:
        return f"{ref} (unresolvable)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="path to a vLLM git checkout")
    parser.add_argument(
        "--from",
        dest="base",
        required=True,
        help="baseline ref, normally the currently pinned version",
    )
    parser.add_argument(
        "--to", dest="target", required=True, help="candidate ref to upgrade to"
    )
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    print(f"repo:   {repo}")
    print(f"from:   {_describe(args.base, repo)}")
    print(f"to:     {_describe(args.target, repo)}")
    try:
        span = _git(
            repo, "rev-list", "--count", f"{args.base}..{args.target}"
        ).strip()
        print(f"span:   {span} commits")
    except subprocess.CalledProcessError:
        print("span:   (unavailable -- refs not connected in this clone)")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        base_root = Path(tmp) / "base"
        target_root = Path(tmp) / "target"
        for ref, root in ((args.base, base_root), (args.target, target_root)):
            missing = _materialize(repo, ref, root)
            if missing:
                print(f"! modules absent at {ref}: {missing}")

        broke: list[str] = []
        appeared: list[str] = []
        rows = []
        for seam in SEAMS:
            before = check_static(seam, str(base_root))
            after = check_static(seam, str(target_root))
            if before is None and after is None:
                verdict, note = "holds", ""
            elif before is not None and after is None:
                verdict, note = "APPEARED", "adopt it"
                appeared.append(seam.target)
            elif before is None and after is not None:
                verdict, note = "BROKE", after.detail
                if seam.required:
                    broke.append(seam.target)
                else:
                    verdict, note = "absent", "optional, still unavailable"
            else:
                verdict, note = "absent", "optional, still unavailable"
                if seam.required:
                    verdict, note = "BROKE", f"already broken: {after.detail}"
                    broke.append(seam.target)
            flag = " " if seam.required else "?"
            rows.append((flag, verdict, seam.target, note))

        width = max(len(r[2]) for r in rows)
        for flag, verdict, target, note in rows:
            line = f"{flag} {verdict:<9} {target:<{width}}"
            print(f"{line}  {note}" if note else line)

        print()
        print(f"required seams broken: {len(broke)}")
        for target in broke:
            seam = next(s for s in SEAMS if s.target == target)
            print(f"  {target}\n    held for: {seam.why}")
        if appeared:
            print(f"newly available optional seams: {len(appeared)}")
            for target in appeared:
                print(f"  {target}")
        print("\n? = optional seam")

    return 1 if broke else 0


if __name__ == "__main__":
    sys.exit(main())
