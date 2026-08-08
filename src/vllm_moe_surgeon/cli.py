# SPDX-License-Identifier: Apache-2.0
"""``surgeon`` -- the command line for the expert-surgery pipeline.

vLLM is imported lazily inside the subcommands that need it, so ``surgeon
--help`` and the offline stages work on a host with no vLLM installed.
"""

from __future__ import annotations

import argparse
import json
import os
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


def _cmd_plan(args: argparse.Namespace) -> int:
    from .surgery import Budget, build_plan, iter_layer_similarity, summarize_plan
    from .surgery.descriptors import CheckpointIndex
    from .telemetry import load

    stats, meta = load(args.profile)

    similarity = None
    if args.similarity_cache and os.path.exists(args.similarity_cache):
        import numpy as np

        with np.load(args.similarity_cache) as data:
            similarity = {int(k): data[k] for k in data.files}
        print(f"loaded cached similarity for {len(similarity)} layers")
    elif args.checkpoint:
        index = CheckpointIndex.open(args.checkpoint)
        similarity = dict(
            iter_layer_similarity(index, stats.moe_layers, rank=args.rank)
        )
        if args.similarity_cache:
            import numpy as np

            # A full model is ~80s per layer of dense SVD; recomputing it on
            # every plan iteration would make the knobs impractical to tune.
            np.savez_compressed(
                args.similarity_cache,
                **{str(k): v for k, v in similarity.items()},
            )
            print(f"cached similarity to {args.similarity_cache}")

    budget = Budget(
        core_experts=args.core_experts,
        disk_experts=args.disk_experts,
        merge_threshold=args.merge_threshold,
        max_cooccurrence=args.max_cooccurrence,
        drop_share_below=args.drop_share_below,
        min_slots_per_expert=args.min_slots,
    )
    plan = build_plan(
        stats,
        budget,
        similarity=similarity,
        model=args.model or meta.get("model"),
        revision=meta.get("revision"),
        provenance={"profile": args.profile},
        force=args.force,
    )
    print(summarize_plan(plan))
    if args.out:
        plan.save(args.out)
        print(f"\nwrote {args.out}")
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    from .surgery import apply_plan, load_plan

    plan = load_plan(args.plan)
    manifest = apply_plan(
        plan,
        args.source,
        args.out,
        shard_bytes=int(args.shard_gb * 1024**3),
    )
    print(f"experts: {manifest['experts_before']} -> {manifest['experts_after']}")
    print(f"top_k:   {manifest['top_k_before']} -> {manifest['top_k_after']}")
    print(f"merges applied: {manifest['merges_applied']}")
    print(f"router:  {manifest['router_rewrite']}")
    print(f"shards:  {len(manifest['shards'])}")
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

    p = sub.add_parser("plan", help="turn a profile plus a budget into a placement")
    p.add_argument("--profile", required=True, help="profile .npz from `profile`")
    p.add_argument("--model", help="override the model name recorded in the plan")
    p.add_argument(
        "--checkpoint",
        help="model directory; enables merges via permutation-invariant similarity",
    )
    p.add_argument("--rank", type=int, default=16, help="descriptor subspace rank")
    p.add_argument(
        "--similarity-cache",
        help="reuse/store the similarity matrices (.npz); ~80s per layer to compute",
    )
    p.add_argument("--core-experts", type=int, required=True)
    p.add_argument(
        "--disk-experts",
        type=int,
        help="extra experts on the disk tier (default: all survivors)",
    )
    p.add_argument("--merge-threshold", type=float, default=0.85)
    p.add_argument("--max-cooccurrence", type=float, default=0.10)
    p.add_argument(
        "--drop-share-below",
        type=float,
        help="delete experts below this per-layer share (default: never delete)",
    )
    p.add_argument("--min-slots", type=float, default=200.0)
    p.add_argument(
        "--force", action="store_true", help="accept a plan built on a thin profile"
    )
    p.add_argument("--out", help="write plan.json here")
    p.set_defaults(func=_cmd_plan)

    p = sub.add_parser("apply", help="write the operated-on checkpoint")
    p.add_argument("--plan", required=True, help="plan.json from `plan`")
    p.add_argument("--source", required=True, help="the base model directory")
    p.add_argument("--out", required=True, help="output checkpoint directory")
    p.add_argument("--shard-gb", type=float, default=4.0)
    p.set_defaults(func=_cmd_apply)

    p = sub.add_parser("seams", help="check the vLLM seams this package holds")
    p.add_argument("--source", help="check a vLLM source tree instead of the install")
    p.set_defaults(func=_cmd_seams)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
