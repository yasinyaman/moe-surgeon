# SPDX-License-Identifier: Apache-2.0
"""``surgeon`` -- the command line for the expert-surgery pipeline.

vLLM is imported lazily inside the subcommands that need it, so ``surgeon
--help`` and the offline stages work on a host with no vLLM installed.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
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
        from .surgery.apply import environment
        from .telemetry.persist import save

        # Capture torch/vLLM/device here, where vLLM is already imported, so the
        # profile records what produced it -- a number without its environment is
        # not reproducible.
        save(
            stats,
            args.out,
            model=args.model,
            revision=args.revision,
            extra={"environment": environment()},
        )
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
        require_gate=not args.skip_gate,
        amplitude=args.amplitude,
    )
    print(f"experts: {manifest['experts_before']} -> {manifest['experts_after']}")
    print(f"top_k:   {manifest['top_k_before']} -> {manifest['top_k_after']}")
    print(f"merges applied: {manifest['merges_applied']}")
    if manifest.get("amplitude") is not None:
        print(f"amplitude: down_proj scaled by {manifest['amplitude']}")
    print(f"router:  {manifest['router_rewrite']}")
    print(f"shards:  {len(manifest['shards'])}")
    print(f"\nwrote {args.out}")
    return 0


def _cmd_tier(args: argparse.Namespace) -> int:
    from . import hot_experts
    from .surgery import load_plan
    from .surgery.tiered import build_store

    plan = load_plan(args.plan)
    hints = hot_experts.from_plan(plan, scale=args.prior_scale)

    layers = sorted(hints.priors)
    summary = build_store(
        args.source,
        args.store,
        layers,
        model_id=args.model_id or args.source,
        revision=plan.revision,
        fp8=not args.no_fp8,
    )
    hints.save(args.hot_experts)

    core = sum(len(v) for v in hints.core.values())
    total = sum(len(v) for v in hints.priors.values())
    print(f"store:   {summary['directory']}")
    print(f"layers:  {len(summary['layers'])}  fp8={summary['fp8']}")
    print(f"size:    {summary['bytes'] / 1024**3:.2f} GiB")
    print(f"hints:   {core} core of {total} resident-eligible experts")
    print(f"wrote    {args.hot_experts}")
    print()
    print("serve with:")
    print(f"  VLLM_MOE_DISK_STORE_DIR={summary['directory']} \\")
    if summary["fp8"]:
        print("  VLLM_MOE_DISK_STORE_FP8=1 \\")
    print("  VLLM_MOE_CACHE_POLICY=ewma \\")
    print(f"  VLLM_MOE_HOT_EXPERTS={args.hot_experts} \\")
    print("  VLLM_MOE_RAM_CACHE=<records> --moe-expert-cache-size <slots>")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    from .surgery.inspect import inspect_checkpoint, report

    profile = None
    if args.profile:
        from .telemetry import load

        profile, _ = load(args.profile)

    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    elif profile is not None:
        layers = list(profile.moe_layers)
    else:
        from .surgery.descriptors import CheckpointIndex

        index = CheckpointIndex.open(args.checkpoint)
        layers = [0] if index.expert_ids(0) else []

    results, model = inspect_checkpoint(
        args.checkpoint, layers, profile=profile, spectra=args.spectra
    )
    print(report(results, model=model or args.checkpoint))
    return 0


def _cmd_autoconfig(args: argparse.Namespace) -> int:
    from .surgery.autoconfig import autoconfigure

    try:
        config = autoconfigure(
            args.checkpoint,
            store_dir=args.store,
            max_num_seqs=args.max_num_seqs,
            kv_reserve_gib=args.kv_reserve,
            vram_gib=args.vram,
            refresh=args.refresh,
        )
    except ValueError as exc:
        # "this will not fit" is an answer, not a crash. The message already names
        # the numbers and what to change, so a traceback only buries it.
        print(f"cannot configure the tier here: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(config.surgeon))
        return 0

    env = config.environment
    source = "cached" if config.cached else "probed"
    print(f"{source} ({config.fingerprint}): {env.device}, "
          f"{env.vram_gib:.1f} GiB free VRAM, {env.host_ram_gib:.0f} GiB host RAM, "
          f"{env.disk_free_gib:.0f} GiB free disk")
    for item in env.unknown:
        print(f"  could not measure: {item}")
    print()
    for line in config.why:
        print(f"  - {line}")
    for line in config.warnings:
        print(f"  ! {line}")
    print()
    argv = config.serve_args(args.model or args.checkpoint)
    print("serve with:")
    print("  " + " ".join(shlex.quote(a) for a in argv))
    if config.cached:
        print()
        print("  (cached; the probe re-runs when the machine, model or batch changes,")
        print("   or pass --refresh)")

    if args.start:
        print()
        print("starting:", " ".join(shlex.quote(a) for a in argv))
        return subprocess.call(argv)
    return 0


def _cmd_budget(args: argparse.Namespace) -> int:
    from .surgery.budget import analyze, report

    geometry = analyze(args.checkpoint)
    print(
        report(
            geometry,
            vram_gib=args.vram,
            fp8_store=not args.no_fp8,
            kv_cache_gib=args.kv_reserve,
            model=args.model or args.checkpoint,
        )
    )
    return 0


def _cmd_ablate(args: argparse.Namespace) -> int:
    from .compat.ablation import arms_from_profile, run_study
    from .compat.profile_runner import iter_prompts_from_jsonl
    from .telemetry import load

    stats, meta = load(args.profile)
    prompts = list(iter_prompts_from_jsonl(args.corpus, args.field))
    if args.limit:
        prompts = prompts[: args.limit]
    if not prompts:
        print("no held-out prompts", file=sys.stderr)
        return 2

    arms = arms_from_profile(
        stats, args.keep, include_hot_control=not args.no_control
    )
    study = run_study(
        args.model or meta["model"],
        prompts,
        arms,
        llm_kwargs=json.loads(args.llm_kwargs) if args.llm_kwargs else None,
    )
    print(study.report())
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    from .compat.ablation import gate_plan
    from .compat.profile_runner import iter_prompts_from_jsonl
    from .surgery import load_plan

    plan = load_plan(args.plan)
    prompts = list(iter_prompts_from_jsonl(args.corpus, args.field))
    if args.limit:
        prompts = prompts[: args.limit]

    verdict = gate_plan(
        plan,
        prompts,
        max_ratio=args.max_ratio,
        model=args.model,
        llm_kwargs=json.loads(args.llm_kwargs) if args.llm_kwargs else None,
    )
    plan.gate = verdict
    plan.save(args.plan)

    mark = "PASS" if verdict["passed"] else "FAIL"
    print(f"verdict:    {mark}  ({verdict['reason']})")
    if "perplexity" in verdict:
        print(f"baseline:   {verdict['baseline_perplexity']:.4f}")
        print(f"ablated:    {verdict['perplexity']:.4f}")
        print(f"tokens:     {verdict.get('held_out_tokens')}")
    if verdict.get("note"):
        print()
        print(f"note: {verdict['note']}")
    print(f"\nwritten to  {args.plan}")
    return 0 if verdict["passed"] else 1


def _cmd_recommend(args: argparse.Namespace) -> int:
    from .surgery.budget import analyze
    from .surgery.recommend import recommend

    stats = None
    if args.profile:
        from .telemetry import load

        stats, _ = load(args.profile)

    max_similarity = None
    if args.similarity_cache and os.path.exists(args.similarity_cache):
        import numpy as np

        with np.load(args.similarity_cache) as data:
            best = 0.0
            for key in data.files:
                matrix = data[key]
                off = matrix[~np.eye(matrix.shape[0], dtype=bool)]
                best = max(best, float(off.max()))
        max_similarity = best

    rec = recommend(
        analyze(args.checkpoint),
        vram_gib=args.vram,
        kv_cache_gib=args.kv_reserve,
        stats=stats,
        max_similarity=max_similarity,
        restarts_often=args.restarts_often,
        latency_sensitive=args.latency_sensitive,
    )
    print(rec.report())
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "the job server needs the 'server' extra: pip install -e '.[server]'",
            file=sys.stderr,
        )
        return 2

    from .server.app import create_app

    token = args.token or os.environ.get("MOE_SURGEON_TOKEN")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not token:
        print(
            f"refusing to bind {args.host} without an auth token: the server runs "
            "subprocesses from request fields. Pass --token (or set "
            "MOE_SURGEON_TOKEN), or bind 127.0.0.1.",
            file=sys.stderr,
        )
        return 2

    uvicorn.run(
        create_app(args.state, timeout=args.stage_timeout, token=token),
        host=args.host,
        port=args.port,
    )
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from .compat.calibrate import calibrate_amplitude, to_dict
    from .compat.profile_runner import iter_prompts_from_jsonl

    prompts = list(iter_prompts_from_jsonl(args.corpus, args.field))
    if args.limit:
        prompts = prompts[: args.limit]
    scales = (
        tuple(float(x) for x in args.scales.split(","))
        if args.scales
        else None
    )
    kwargs = {"enforce_eager": True}
    if args.llm_kwargs:
        kwargs.update(json.loads(args.llm_kwargs))

    result = calibrate_amplitude(
        args.checkpoint,
        prompts,
        **({"scales": scales} if scales else {}),
        llm_kwargs=kwargs,
    )
    print(result.report())
    if args.out:
        with open(args.out, "w") as f:
            json.dump(to_dict(result), f, indent=2)
        print(f"\nwritten to  {args.out}")
    print(
        "\nfold it in with:  surgeon apply --plan <plan> --source <base> "
        f"--out <dir> --amplitude {result.best_scale:.3f}"
        if result.worth_applying
        else "\nnothing to fold in."
    )
    return 0


def _cmd_vram_floor(args: argparse.Namespace) -> int:
    from .compat.bisect_vram import bisect_floor, to_dict

    result = bisect_floor(
        args.model,
        name=args.name,
        low=args.low,
        high=args.high,
        tolerance=args.tolerance,
        llm_kwargs=json.loads(args.llm_kwargs) if args.llm_kwargs else None,
        timeout=args.timeout,
    )
    print(result.report())
    if args.out:
        with open(args.out, "w") as f:
            json.dump(to_dict(result), f, indent=2)
        print(f"\nwritten to  {args.out}")
    # A configuration that boots nowhere is a failure, not a floor of zero.
    return 0 if result.floor_fraction is not None else 1


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
    p.add_argument(
        "--amplitude",
        type=float,
        help="fold this scalar into every survivor's down_proj; measure it with "
        "`surgeon calibrate`",
    )
    p.add_argument(
        "--skip-gate",
        action="store_true",
        help="apply an unmeasured plan (deletion is irreversible)",
    )
    p.set_defaults(func=_cmd_apply)

    p = sub.add_parser("tier", help="build the disk store and the residency hint")
    p.add_argument("--plan", required=True)
    p.add_argument("--source", required=True, help="checkpoint to build the store from")
    p.add_argument("--store", required=True, help="output store directory")
    p.add_argument("--hot-experts", required=True, help="output hot_experts.json")
    p.add_argument(
        "--model-id",
        help="identity recorded in the store; must match what vLLM will report",
    )
    p.add_argument("--prior-scale", type=float, default=8.0)
    p.add_argument("--no-fp8", action="store_true", help="full-precision records")
    p.set_defaults(func=_cmd_tier)

    p = sub.add_parser(
        "inspect", help="which routing signals a checkpoint carries, and if they work"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--profile",
        help="profile .npz; without it nothing is validated, only reported",
    )
    p.add_argument("--layers", help="comma-separated layer indices")
    p.add_argument(
        "--spectra",
        action="store_true",
        help="add per-expert SVD spectra (slow: tens of seconds per layer)",
    )
    p.set_defaults(func=_cmd_inspect)

    p = sub.add_parser(
        "autoconfig",
        help="probe this machine and size the tier for it, once",
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", help="model id to serve; defaults to --checkpoint")
    p.add_argument("--store", default="./store")
    p.add_argument("--max-num-seqs", type=int, default=8,
                   help="serving batch; it sets how many experts a layer can reach")
    p.add_argument("--kv-reserve", type=float, default=2.0, help="KV cache GiB")
    p.add_argument("--vram", type=float, help="override the measured free VRAM")
    p.add_argument("--refresh", action="store_true", help="ignore the cached answer")
    p.add_argument("--json", action="store_true", help="print only the surgeon config")
    p.add_argument("--start", action="store_true", help="run `vllm serve` with it")
    p.set_defaults(func=_cmd_autoconfig)

    p = sub.add_parser(
        "budget", help="least GPU memory this model can be served in (headers only)"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", help="name to print")
    p.add_argument("--vram", type=float, help="target VRAM in GiB; solves for capacity")
    p.add_argument("--kv-reserve", type=float, default=1.0, help="KV cache GiB")
    p.add_argument("--no-fp8", action="store_true", help="full-precision store")
    p.set_defaults(func=_cmd_budget)

    p = sub.add_parser(
        "ablate", help="measure what experts contribute, by zeroing them"
    )
    p.add_argument("--profile", required=True, help="profile .npz for the ranking")
    p.add_argument("--corpus", required=True, help="HELD-OUT JSONL to score on")
    p.add_argument("--field", default="text")
    p.add_argument("--limit", type=int, help="cap the number of held-out prompts")
    p.add_argument("--model", help="override the model from the profile")
    p.add_argument(
        "--keep",
        type=int,
        required=True,
        help="experts kept per layer; the rest are ablated",
    )
    p.add_argument(
        "--no-control",
        action="store_true",
        help="skip the hottest-N control arm (not recommended)",
    )
    p.add_argument("--llm-kwargs", help="extra LLM() kwargs as JSON")
    p.set_defaults(func=_cmd_ablate)

    p = sub.add_parser(
        "gate", help="measure what a plan's deletions cost, and record the verdict"
    )
    p.add_argument(
        "--plan", required=True, help="plan.json; the verdict is written back"
    )
    p.add_argument("--corpus", required=True, help="HELD-OUT JSONL to score on")
    p.add_argument("--field", default="text")
    p.add_argument("--limit", type=int)
    p.add_argument("--model", help="override the model named in the plan")
    p.add_argument("--max-ratio", type=float, default=1.3, help="perplexity ceiling")
    p.add_argument("--llm-kwargs", help="extra LLM() kwargs as JSON")
    p.set_defaults(func=_cmd_gate)

    p = sub.add_parser(
        "recommend", help="choose a strategy for a target, from measured properties"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vram", type=float, help="target VRAM in GiB")
    p.add_argument("--kv-reserve", type=float, default=1.0)
    p.add_argument("--profile", help="profile .npz; without it nothing is called cold")
    p.add_argument("--similarity-cache", help="sim .npz; without it no merge is judged")
    p.add_argument("--restarts-often", action="store_true")
    p.add_argument("--latency-sensitive", action="store_true")
    p.set_defaults(func=_cmd_recommend)

    p = sub.add_parser("serve", help="run the job server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--token",
        default=None,
        help="require this token in the X-Surgeon-Token header; needed to bind a "
        "non-loopback host. Falls back to $MOE_SURGEON_TOKEN.",
    )
    p.add_argument("--state", default="./surgeon-state", help="where jobs are recorded")
    p.add_argument(
        "--stage-timeout",
        type=float,
        help="kill a stage that exceeds this many seconds",
    )
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser(
        "calibrate",
        help="find the down_proj amplitude a pruned checkpoint should carry",
    )
    p.add_argument("--checkpoint", required=True, help="the PRUNED checkpoint")
    p.add_argument("--corpus", required=True, help="HELD-OUT JSONL to score on")
    p.add_argument("--field", default="text")
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--scales",
        help="comma-separated scales to try; deletion inflates, so the useful "
        "bracket is below 1.0 (default 0.75..1.00)",
    )
    p.add_argument("--llm-kwargs", help="extra LLM() kwargs as JSON")
    p.add_argument("--out", help="write the curve as JSON here")
    p.set_defaults(func=_cmd_calibrate)

    p = sub.add_parser(
        "vram-floor",
        help="bisect the smallest memory budget a model boots and serves in",
    )
    p.add_argument("--model", required=True, help="model or checkpoint directory")
    p.add_argument("--name", help="label for the report")
    p.add_argument("--low", type=float, default=0.05)
    p.add_argument("--high", type=float, default=0.90)
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="stop when the bracket is this narrow, as a fraction of the device",
    )
    p.add_argument("--timeout", type=float, default=900.0, help="per boot attempt")
    p.add_argument("--llm-kwargs", help="extra LLM() kwargs as JSON")
    p.add_argument("--out", help="write the result as JSON here")
    p.set_defaults(func=_cmd_vram_floor)

    p = sub.add_parser("seams", help="check the vLLM seams this package holds")
    p.add_argument("--source", help="check a vLLM source tree instead of the install")
    p.set_defaults(func=_cmd_seams)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
