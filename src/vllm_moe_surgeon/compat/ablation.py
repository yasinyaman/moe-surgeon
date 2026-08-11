# SPDX-License-Identifier: Apache-2.0
"""Soft ablation: measure what an expert actually contributes, before deleting it.

Everything measured so far counts how *often* an expert is selected. That is not
what it contributes. A top-k MoE output is a gate-weighted sum,
:math:`\\sum_i g_i E_i(x)`, so the eighth-ranked expert of a top-8 layer can be
selected constantly and still move the output very little. A pruning decision made
on selection counts is therefore made on the wrong quantity.

This module measures the right one, by the only method that cannot be fooled:
remove the expert and look at the loss.

**Zeroing, not deleting.** An ablated expert keeps its router row and is still
selected; only its weights become zero, so it contributes exactly nothing to the
weighted sum. That is deliberately *pessimistic* relative to deleting it: under
deletion the renormalised gate mass flows to the surviving experts, while zeroing
throws that mass away. So "zeroing did not hurt" implies "deleting will not hurt",
which is the safe direction for a gate that authorises irreversible surgery.

It also separates two things the earlier 64→40 experiment had conflated. Writing a
pruned checkpoint both removes expert capacity *and* rewrites the router (fewer
rows, different softmax denominator, different top-k for every token). Zeroing
holds the router fixed, so the two effects can finally be told apart.

**One load, many ablations.** The weights are zeroed in place in the live model
through ``collective_rpc``, which accepts a plain callable. No checkpoint copy per
variant -- which matters when a variant is 8.4 GiB and the disk is nearly full.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .._logging import init_logger

logger = init_logger(__name__)

# Backups of the rows we zeroed, keyed by (module name, expert id), living in the
# worker process. Module-level so the restore callable can find them.
_BACKUP: dict[tuple[str, int], Any] = {}


def _moe_modules(model: Any) -> list[tuple[str, Any]]:
    """Every module that owns stacked routed-expert weights, with its name."""
    found = []
    for name, module in model.named_modules():
        w13 = getattr(module, "w13_weight", None)
        w2 = getattr(module, "w2_weight", None)
        if w13 is None or w2 is None:
            continue
        if w13.numel() == 0 or w2.numel() == 0:
            # The expert cache replaces these with empty placeholders and keeps
            # the real weights in its own slots, so ablation would silently no-op.
            raise RuntimeError(
                f"{name} has zero-sized expert weights, which means the expert "
                "cache or streaming load is active. Ablation edits the model's "
                "own weights, so run it with the cache disabled."
            )
        found.append((name, module))
    return found


def _layer_index(name: str) -> int | None:
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return None


def zero_experts(worker: Any, spec: Mapping[int, Sequence[int]]) -> dict[str, int]:
    """Zero the named experts in place, backing up what was overwritten.

    ``spec`` maps a global decoder-layer index to the expert ids to silence.
    """
    import torch

    model = worker.model_runner.model
    zeroed = 0
    touched_layers = 0
    for name, module in _moe_modules(model):
        layer = _layer_index(name)
        if layer is None or layer not in spec:
            continue
        experts = spec[layer]
        if not experts:
            continue
        touched_layers += 1
        with torch.no_grad():
            for expert in experts:
                for attr in ("w13_weight", "w2_weight"):
                    param = getattr(module, attr)
                    key = (f"{name}.{attr}", int(expert))
                    if key not in _BACKUP:
                        _BACKUP[key] = param.data[expert].detach().to("cpu").clone()
                    param.data[expert].zero_()
                zeroed += 1
    return {"experts_zeroed": zeroed, "layers_touched": touched_layers}


def restore_experts(worker: Any) -> dict[str, int]:
    """Put every backed-up row back, so the next arm starts from the baseline."""
    import torch

    model = worker.model_runner.model
    by_name = {name: module for name, module in _moe_modules(model)}
    restored = 0
    with torch.no_grad():
        for (qualified, expert), saved in _BACKUP.items():
            module_name, attr = qualified.rsplit(".", 1)
            module = by_name.get(module_name)
            if module is None:
                continue
            getattr(module, attr).data[expert].copy_(saved.to(
                getattr(module, attr).device
            ))
            restored += 1
    _BACKUP.clear()
    return {"rows_restored": restored}


def count_moe_layers(worker: Any) -> list[str]:
    """Names of the expert-owning modules, for a sanity check before ablating."""
    return [name for name, _ in _moe_modules(worker.model_runner.model)]


# ----------------------------------------------------------------- the harness


@dataclass
class AblationResult:
    name: str
    nll: float
    tokens: int
    experts_zeroed: int

    @property
    def perplexity(self) -> float:
        return math.exp(self.nll / max(self.tokens, 1))


@dataclass
class AblationStudy:
    baseline: AblationResult
    arms: list[AblationResult] = field(default_factory=list)

    def report(self) -> str:
        base_ppl = self.baseline.perplexity
        lines = [
            f"held-out tokens: {self.baseline.tokens}",
            f"baseline perplexity: {base_ppl:.4f}",
            "",
            f"  {'arm':<28} {'experts':>7} {'ppl':>10} {'delta':>9} {'ratio':>7}",
        ]
        for arm in self.arms:
            ppl = arm.perplexity
            lines.append(
                f"  {arm.name:<28} {arm.experts_zeroed:>7} {ppl:>10.4f}"
                f" {ppl - base_ppl:>+9.4f} {ppl / base_ppl:>7.2f}x"
            )
        return "\n".join(lines)


def measure_nll(llm: Any, prompts: Sequence[str]) -> tuple[float, int]:
    """Total negative log-likelihood of ``prompts`` and the token count.

    Uses ``prompt_logprobs`` so the number is the model's loss on given text
    rather than anything about its own generations -- ablation has to be judged on
    held-out likelihood, not on whether the sampled text still looks plausible.
    """
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outputs = llm.generate(list(prompts), params)

    total = 0.0
    count = 0
    for output in outputs:
        logprobs = output.prompt_logprobs
        if not logprobs:
            continue
        for position, entry in enumerate(logprobs):
            if entry is None:
                continue  # the first token has no prediction
            token_id = output.prompt_token_ids[position]
            record = entry.get(token_id)
            if record is None:
                continue
            total -= record.logprob
            count += 1
    if count == 0:
        raise RuntimeError(
            "no prompt logprobs were returned; ablation cannot be scored. Check "
            "that the engine supports prompt_logprobs for this model."
        )
    return total, count


def run_study(
    model: str,
    prompts: Sequence[str],
    arms: Iterable[tuple[str, Mapping[int, Sequence[int]]]],
    *,
    llm_kwargs: dict[str, Any] | None = None,
) -> AblationStudy:
    """Baseline, then each arm, restoring between them. One model load.

    Arms are evaluated in order and the model is restored after each, so an arm
    cannot inherit the previous arm's damage -- the failure that would make every
    later arm look worse than it is.
    """
    from vllm import LLM

    kwargs: dict[str, Any] = {}
    kwargs.update(llm_kwargs or {})
    llm = LLM(model=model, **kwargs)

    names = llm.collective_rpc(count_moe_layers)[0]
    logger.info("ablation: %d expert-owning modules found", len(names))

    nll, tokens = measure_nll(llm, prompts)
    study = AblationStudy(
        baseline=AblationResult("baseline", nll, tokens, experts_zeroed=0)
    )

    for name, spec in arms:
        # zero_experts is inside the try so a failure partway through the zero pass
        # (an out-of-range id, an OOM cloning a row, a KeyboardInterrupt) still runs
        # restore in the finally. Otherwise the worker-global backup keeps rows nobody
        # replays, and the next arm inherits stale damage.
        try:
            stats = llm.collective_rpc(zero_experts, args=(spec,))[0]
            nll, tokens = measure_nll(llm, prompts)
            study.arms.append(
                AblationResult(name, nll, tokens, stats["experts_zeroed"])
            )
            logger.info(
                "arm %s: zeroed %d experts over %d layers",
                name,
                stats["experts_zeroed"],
                stats["layers_touched"],
            )
        finally:
            restored = llm.collective_rpc(restore_experts)[0]
            logger.info("arm %s: restored %d rows", name, restored["rows_restored"])

    return study


def arms_from_importance(
    stats: Any,
    keep: int,
    importance: Any,
    label: str,
) -> tuple[str, dict[int, list[int]]]:
    """One ablation arm: the coldest ``num_experts - keep`` under ``importance``.

    ``importance`` is ``(len(moe_layers), num_experts)``. Different importance
    definitions produce different keep-sets at the same budget, which is the
    point: the ablation, not the theory, decides which definition is better.
    """
    import numpy as np

    drop = stats.num_experts - keep
    spec: dict[int, list[int]] = {}
    for slot, layer in enumerate(stats.moe_layers):
        order = np.argsort(importance[slot], kind="stable")
        spec[layer] = [int(e) for e in order[:drop]]
    return (f"{label} (drop {drop})", spec)


def arms_from_profile(
    stats: Any,
    keep: int,
    *,
    include_hot_control: bool = True,
) -> list[tuple[str, dict[int, list[int]]]]:
    """Ablation arms derived from a usage profile.

    The coldest ``num_experts - keep`` experts per layer are the set a
    count-based plan would drop. The hot control ablates the *same number* of the
    hottest experts instead: without it, a small delta could just mean the model
    is insensitive to losing any 24 experts, which would say nothing about the
    ranking.
    """
    import numpy as np

    share = stats.layer_share()
    cold: dict[int, list[int]] = {}
    hot: dict[int, list[int]] = {}
    drop = stats.num_experts - keep
    for slot, layer in enumerate(stats.moe_layers):
        order = np.argsort(share[slot])  # ascending: coldest first
        cold[layer] = [int(e) for e in order[:drop]]
        hot[layer] = [int(e) for e in order[::-1][:drop]]

    arms = [(f"coldest {drop}/{stats.num_experts}", cold)]
    if include_hot_control:
        arms.append((f"hottest {drop}/{stats.num_experts} (control)", hot))
    return arms


def gate_plan(
    plan: Any,
    prompts: Sequence[str],
    *,
    max_ratio: float = 1.3,
    model: str | None = None,
    llm_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure what a plan's deletions cost, and return a gate verdict.

    Ablates exactly the set the plan drops -- one arm, one model load -- and
    compares held-out perplexity against the unablated baseline. The verdict goes
    into ``plan.gate`` and :func:`~..surgery.apply.apply_plan` refuses to delete
    without it.

    Zeroing is pessimistic relative to the deletion the plan will actually perform
    (the renormalised gate mass flows to survivors instead of being discarded), so
    a plan that passes here will do at least as well once applied. Measured on
    OLMoE: zeroed 12.65 vs deleted 12.13.

    Only deletions are gated, and merge donors are excluded rather than swept in.
    A donor carries ``action == "drop"`` with a ``merge_target``, but it is not
    deleted -- its weights are folded into a survivor. Zeroing it would measure the
    cost of throwing it away, which is not what the plan does, and would then report
    a verdict on a plan whose merges were never measured at all. Merge exactness is
    covered by the tests in :mod:`..surgery.align`; merge *quality* on real
    non-redundant experts is not something zeroing can reach, so the verdict says so
    by name instead of implying coverage it does not have.
    """
    dropped: dict[int, list[int]] = {}
    merged = 0
    for placement in plan.placements:
        if placement.action != "drop":
            continue
        if getattr(placement, "merge_target", None) is not None:
            merged += 1
            continue
        dropped.setdefault(placement.layer, []).append(placement.expert)

    ungated = (
        f" {merged} merge donors are folded into survivors, not deleted, and are "
        "excluded from this measurement -- zeroing cannot emulate a merge"
        if merged
        else ""
    )

    # Bind the verdict to the exact drop-set and corpus it measured, so a plan
    # edited after gating cannot be applied under a stale pass. Imported lazily to
    # keep this module free of a hard surgery dependency at import time.
    from ..surgery.plan import drop_set_digest

    digest = drop_set_digest(plan)
    corpus_id = _corpus_fingerprint(prompts)

    if not dropped:
        return {
            "passed": True,
            "reason": (
                "plan deletes nothing, so there is nothing to measure." + ungated
            ),
            "ratio": 1.0,
            "max_ratio": max_ratio,
            "merges_not_gated": merged,
            "drop_digest": digest,
            "corpus": corpus_id,
        }

    study = run_study(
        model or plan.model,
        prompts,
        [("plan deletions", dropped)],
        llm_kwargs=llm_kwargs,
    )
    arm = study.arms[0]
    base = study.baseline.perplexity
    ratio = arm.perplexity / base if base > 0 else float("inf")
    passed = ratio <= max_ratio

    # The ceiling is applied to a deliberately pessimistic measurement, so a plan
    # can fail by a hair and still be fine once applied -- on OLMoE the deletion
    # came in about 4% below its zeroed bound (12.13 vs 12.65). Say so rather than
    # letting a narrow FAIL read as a verdict on the plan itself.
    note = None
    if not passed and ratio <= max_ratio * 1.1:
        note = (
            f"failed narrowly: {ratio:.3f}x against a {max_ratio}x ceiling, and "
            "this measurement zeroes the experts, which is pessimistic relative to "
            "the deletion the plan performs (the renormalised gate mass is "
            "discarded rather than redistributed). Measured on OLMoE the applied "
            "checkpoint scored 4% better than its zeroed bound. Either raise the "
            "ceiling knowingly or measure the applied artifact."
        )

    verdict = {
        "passed": bool(passed),
        "reason": f"{ratio:.3f}x {'<=' if passed else '>'} {max_ratio}x" + ungated,
        "baseline_perplexity": base,
        "perplexity": arm.perplexity,
        "ratio": ratio,
        "max_ratio": max_ratio,
        "held_out_tokens": study.baseline.tokens,
        "experts_zeroed": arm.experts_zeroed,
        "method": "zeroed weights, router held fixed (pessimistic vs deletion)",
        "merges_not_gated": merged,
        "drop_digest": digest,
        "corpus": corpus_id,
    }
    if note:
        verdict["note"] = note
    return verdict


def _corpus_fingerprint(prompts: Sequence[str]) -> dict[str, Any]:
    """Enough to notice the held-out corpus changed under a reused verdict."""
    import hashlib

    joined = "\x00".join(prompts).encode()
    return {
        "n_prompts": len(prompts),
        "sha256": hashlib.sha256(joined).hexdigest()[:16],
    }
