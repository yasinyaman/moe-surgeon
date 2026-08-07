# SPDX-License-Identifier: Apache-2.0
"""``surgeon`` -- the command line for the expert-surgery pipeline.

vLLM is imported lazily inside the subcommands that need it, so ``surgeon
--help`` and the offline stages work on a host with no vLLM installed.
"""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_profile(args: argparse.Namespace) -> int:
    from .compat.profile_runner import (
        iter_prompts_from_jsonl,
        profile_offline,
        summarize,
    )

    if args.corpus:
        prompts = list(iter_prompts_from_jsonl(args.corpus, args.field))
    else:
        prompts = list(args.prompt)
    if not prompts:
        print("no prompts: pass --corpus or --prompt", file=sys.stderr)
        return 2

    llm_kwargs = json.loads(args.llm_kwargs) if args.llm_kwargs else {}
    stats = profile_offline(
        args.model,
        prompts,
        max_tokens=args.max_tokens,
        revision=args.revision,
        with_cooc=args.cooc,
        llm_kwargs=llm_kwargs,
    )
    print(summarize(stats))

    if args.out:
        from .telemetry.persist import save

        save(stats, args.out, model=args.model, revision=args.revision)
        print(f"\nwrote {args.out}")
    return 0


def _cmd_seams(args: argparse.Namespace) -> int:
    from .compat.seams import SEAMS, check, check_all_static

    if args.source:
        problems = check_all_static(args.source, include_optional=True)
        where = args.source
    else:
        from .compat import installed_version

        version = installed_version()
        if version is None:
            print("vLLM not installed; pass --source <vllm-checkout>", file=sys.stderr)
            return 2
        problems = [p for s in SEAMS if (p := check(s)) is not None]
        where = f"installed vLLM {version}"

    required = [p for p in problems if p.seam.required]
    print(f"{len(SEAMS)} seams checked against {where}")
    for problem in problems:
        mark = "BROKE" if problem.seam.required else "absent"
        print(f"  {mark:>6}  {problem.seam.target}\n          {problem.detail}")
    print(f"required seams broken: {len(required)}")
    return 1 if required else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="surgeon", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("profile", help="measure per-expert usage on a corpus")
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--corpus", help="JSONL calibration corpus")
    p.add_argument("--field", default="text", help="JSONL field holding the text")
    p.add_argument("--prompt", action="append", default=[], help="inline prompt")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--cooc", action="store_true", help="also record co-occurrence")
    p.add_argument("--out", help="write the profile here (.npz)")
    p.add_argument("--llm-kwargs", help="extra LLM() kwargs as JSON")
    p.set_defaults(func=_cmd_profile)

    p = sub.add_parser("seams", help="check the vLLM seams this package holds")
    p.add_argument("--source", help="check a vLLM source tree instead of the install")
    p.set_defaults(func=_cmd_seams)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
