# SPDX-License-Identifier: Apache-2.0
"""``surgeon`` -- the command line for the expert-surgery pipeline.

vLLM is imported lazily inside the subcommands that need it, so ``surgeon
--help`` and the offline stages work on a host with no vLLM installed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from typing import Any


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
    from .telemetry.stats import concentration_report

    print()
    print(concentration_report(stats.tokens))

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


def _cmd_run(args: argparse.Namespace) -> int:
    from safetensors import SafetensorError

    from .compat.repl import run
    from .surgery.autoconfig import autoconfigure

    surgeon = None
    if not args.no_tier:
        checkpoint = args.checkpoint or args.model
        try:
            config = autoconfigure(
                checkpoint,
                store_dir=args.store,
                max_num_seqs=args.max_num_seqs,
                kv_reserve_gib=args.kv_reserve,
                vram_gib=args.vram,
            )
        except (ValueError, OSError, SafetensorError) as exc:
            # A bare repo id has no headers to read, a corrupt shard raises from
            # safetensors, and a machine that cannot fit the tier is a real answer.
            # Say which and offer the way out rather than failing with a stack
            # trace.
            print(f"could not size the tier: {exc}", file=sys.stderr)
            print(
                "pass --checkpoint <local dir> to size it, or --no-tier to serve "
                "untiered.",
                file=sys.stderr,
            )
            return 2
        surgeon = config.surgeon
        source = "cached" if config.cached else "probed"
        print(f"tier ({source}): {json.dumps(surgeon)}")

    return run(
        args.model,
        surgeon=surgeon,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        # The capacity above was sized for this batch bound; boot with the same
        # bound or the engine can route to more experts than there are slots
        # right after the tool said no step should have to re-fetch.
        llm_kwargs={"max_num_seqs": args.max_num_seqs} if surgeon else None,
    )


def _cmd_autoconfig(args: argparse.Namespace) -> int:
    from safetensors import SafetensorError

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
    except (ValueError, OSError, SafetensorError) as exc:
        # "this will not fit" is an answer, not a crash -- and so are a mistyped
        # checkpoint path and a corrupt shard. The message names the problem, so a
        # traceback only buries it.
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


def _cmd_headroom(args: argparse.Namespace) -> int:
    import tempfile

    from .compat.headroom import Headroom, report, score_model, stage_corpus
    from .compat.profile_runner import iter_prompts_from_jsonl

    if len(args.model) < 2:
        # The same rule the job server enforces on the request shape: a
        # one-row table is not a ranking. Refused here rather than after a
        # full engine boot has produced one.
        print(
            "headroom ranks candidates against each other, so it needs at least "
            "two --model arguments; got one. Add the checkpoint you would "
            "otherwise tier or prune as the baseline.",
            file=sys.stderr,
        )
        return 1

    prompts = list(iter_prompts_from_jsonl(args.corpus, args.field))
    if args.limit:
        prompts = prompts[: args.limit]
    if not prompts:
        print(f"no prompts in {args.corpus}", file=sys.stderr)
        return 1

    llm_kwargs = json.loads(args.llm_kwargs) if args.llm_kwargs else {}
    with tempfile.TemporaryDirectory() as tmp:
        # Staged once: every candidate must be scored on byte-identical text.
        staged = os.path.join(tmp, "corpus.jsonl")
        corpus_bytes = stage_corpus(prompts, staged)
        result = Headroom(
            corpus=args.corpus, corpus_bytes=corpus_bytes, n_prompts=len(prompts)
        )
        for model in args.model:
            print(f"scoring {model} ...", file=sys.stderr)
            result.scores.append(
                score_model(
                    model,
                    staged,
                    llm_kwargs=llm_kwargs,
                    timeout=args.timeout,
                    corpus_bytes=corpus_bytes,
                )
            )

    print(report(result))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "corpus": result.corpus,
                    "corpus_bytes": result.corpus_bytes,
                    "n_prompts": result.n_prompts,
                    "scores": [_jsonable(vars(s)) for s in result.scores],
                },
                f,
                indent=2,
            )
        print(f"\nwritten to  {args.out}")
    # Nothing scored is a failed measurement, not a verdict. A run where *some*
    # candidates failed still exits 0 on purpose: headroom is deliberately not a
    # gate, and it runs first in the job pipeline, so failing the stage over one
    # unreachable candidate would abort profile/plan/gate/tier behind it. The
    # "a ranking of one is not a ranking" rule is enforced up front instead,
    # where it costs no engine boot.
    return 0 if result.ranked else 1


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    """NaN is a Python extension, not JSON: a strict parser rejects the whole
    file. An unscored candidate is a deliberate, preserved outcome here, so its
    missing figures serialise as ``null`` -- which round-trips everywhere and
    still reads as "no number" beside the ``error`` that explains why."""
    return {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in row.items()
    }


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


def _user_input_errors() -> type[Exception]:
    """Exception classes that mean "your input", beyond the builtins.

    SafetensorError derives from Exception directly -- not ValueError -- so a
    corrupt shard would traceback without this. Resolved lazily because the CLI
    must import on hosts where optional extras are absent.
    """
    try:
        from safetensors import SafetensorError

        return SafetensorError
    except ImportError:  # pragma: no cover - safetensors is a core dep
        return ValueError


def main(argv: list[str] | None = None) -> int:
    # When stdout is a pipe or a captured log file it is block-buffered, and
    # vLLM's exit teardown can take the interpreter down before the buffer
    # flushes -- observed live: a piped `surgeon run` session ran to completion
    # and delivered zero output, every print lost. Every subcommand that boots
    # an engine prints its deliverable through this stdout, and the job server
    # captures stages the same way, so the durability fix belongs here, once.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:  # pragma: no cover - non-reconfigurable stream
        pass
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
        "run", help="interactive prompt, with what the tier costs per turn"
    )
    p.add_argument("model", help="model id or checkpoint directory to serve")
    p.add_argument("--checkpoint", help="local directory to size from, if `model` "
                                       "is a repo id")
    p.add_argument("--store", default="./store")
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--kv-reserve", type=float, default=None,
                   help="KV cache GiB (default: 15%% of free VRAM, clamped to "
                        "[0.5, 2.0] -- a flat 2.0 starved small cards)")
    p.add_argument("--vram", type=float, help="override the measured free VRAM")
    p.add_argument("--no-tier", action="store_true", help="serve untiered, for a "
                                                          "side-by-side comparison")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=512)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser(
        "autoconfig",
        help="probe this machine and size the tier for it, once",
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", help="model id to serve; defaults to --checkpoint")
    p.add_argument("--store", default="./store")
    p.add_argument("--max-num-seqs", type=int, default=8,
                   help="serving batch; it sets how many experts a layer can reach")
    p.add_argument("--kv-reserve", type=float, default=None,
                   help="KV cache GiB (default: 15%% of free VRAM, clamped to "
                        "[0.5, 2.0] -- a flat 2.0 starved small cards)")
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
        "headroom",
        help="score candidate checkpoints on a domain, before tiering or cutting",
    )
    p.add_argument(
        "--corpus", required=True, help="HELD-OUT JSONL for the target domain"
    )
    p.add_argument("--field", default="text")
    p.add_argument("--limit", type=int, help="cap the number of prompts")
    p.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="MODEL",
        help="candidate model or checkpoint; repeat for each one to compare",
    )
    p.add_argument("--timeout", type=float, default=1800.0, help="per candidate")
    p.add_argument("--llm-kwargs", help="extra LLM() kwargs as JSON")
    p.add_argument("--out", help="write the scores here as JSON")
    p.set_defaults(func=_cmd_headroom)

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
    try:
        return args.func(args)
    except (OSError, ValueError, _user_input_errors()) as exc:
        # A mistyped path, a missing config.json, a corrupt shard, a target that
        # does not fit: these are answers about the user's input, and every one of
        # them already carries its explanation in the exception message. A
        # traceback buries the answer under nine frames of plumbing -- found by
        # running each command against a bad path and reading what a user would
        # read. Genuine bugs (TypeError, KeyError, assertions) still crash loudly,
        # because hiding those would be worse than ugly output.
        print(f"surgeon {args.command}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
