# Profiling, planning and surgery

The offline pipeline: measure what a deployment uses, turn it into a plan,
measure what the plan's deletions cost, and only then write the checkpoint.
Also the measured case for why pruning is a last resort on these models.

## Profiling

```bash
surgeon profile --model allenai/OLMoE-1B-7B-0924 --corpus domain.jsonl --out profile.npz
```

Telemetry rides vLLM's own `--enable-return-routed-experts`, which captures
per-token, per-layer expert ids and returns them per request. So this costs
**zero seams** — the aggregator is numpy over a public output field, and adding it
*removed* three internal seams the plan had budgeted for a `MoERunner` subclass.

What the public API does not carry is router *weights*. Token-weighted counts and
co-occurrence are exact; gate mass would need a routing hook and is deferred.

One trap is handled for you. vLLM sizes the capture buffer by
`num_hidden_layers` and zeroes it each step, and only layers that route ever
write to it — so in a model with dense layers those rows read as *expert 0 chosen
by every token*. Aggregate naively and expert 0 looks like the hottest expert in
the model. `telemetry/layers.py` resolves the real MoE layer set from the config,
and `ExpertStats` cross-checks it against the data (a genuine top-k row cannot
repeat an expert, so an all-identical row is uncaptured).

## Planning

```bash
surgeon plan --profile profile.npz --core-experts 24 \
    --checkpoint /path/to/model --similarity-cache sim.npz --out plan.json
```

The plan is the reviewable artifact: JSON, one line of rationale per expert,
hand-editable, and re-validated on load. Every expert gets one of three
placements — `merge_into_core`, `keep_on_disk`, `drop` — so pruning is a
*placement* problem. Nothing is deleted unless you pass `--drop-share-below`.

Two things the engine refuses to do. It will not build a plan from a profile too
thin to justify one (drops are irreversible, and a ranking over a handful of
tokens is noise), and it will never touch a layer that produced no routing rows —
silence is not evidence that a layer's experts are cold.

### Merging needs permutation-invariant similarity

An FFN expert is unchanged by permuting its intermediate neurons. So two experts
can be functionally near-identical while their flattened weights look unrelated:
entrywise cosine similarity is close to meaningless here. `surgery/descriptors.py`
compares the *row space of `gate_proj`/`up_proj`* and the *column space of
`down_proj`* instead — subspaces that the permutation provably leaves fixed — via
the mean squared cosine of principal angles. `tests/test_descriptors.py` asserts
a permuted copy scores 1.0 where flattened cosine collapses to ~0.

### Measured: OLMoE has no mergeable experts

| layer | median | max | pairs ≥ 0.85 |
|---|---|---|---|
| 0 | 0.080 | 0.259 | 0 of 2016 |
| 7 | 0.033 | 0.289 | 0 of 2016 |
| 15 | 0.054 | 0.370 | 0 of 2016 |

Rank sweep on layer 0 (H = 2048): as descriptor rank grows 4 → 256, the median
rises 0.09 → 0.18 while the **max falls** 0.28 → 0.23. If a near-duplicate pair
existed its similarity would stay high as rank grew; instead the ceiling
compresses toward the median. There is no redundant pair to merge.

The similarity is still ~10x the random-subspace baseline, so the experts do
share structure — they are just nowhere near interchangeable. For this model that
makes the **disk tier the primary mechanism, not the fallback**: the cold tail
gets retained rather than merged away.

The caveat worth stating: weight-space orthogonality does not prove functional
distinctness *on a narrow domain*. On a domain corpus the hidden states occupy a
small subspace, and two experts could act almost identically there while spanning
different global subspaces. Activation-based similarity would settle it, and this
result promotes that from optional to necessary if merging is to be pursued.


#### Generalisation: Qwen3-30B-A3B

128 experts, 5 of 48 layers sampled, **81,280 pairs**: median similarity 0.034,
max 0.401, exactly **one** pair at or above 0.4. Doubling the expert count and
tripling the depth did not produce mergeable experts — if anything Qwen's experts
are *more* orthogonal than OLMoE's.

A third family screens the same way: DeepSeek-V2-Lite, max 0.331 across 6,048
pairs (2026-08-10). Three families at 0.33–0.40 max, none near the 0.85 merge
threshold.

### And when merging is forced anyway, it is worse than deleting

The similarity finding says no pair *qualifies*. It does not say what merging would
cost if the threshold were lowered until pairs did — so that was measured, because
"unexercised" is not the same as "unnecessary".

Dropping the threshold from 0.85 to 0.10 makes 233 of 384 removals merges (pair
similarity 0.10–0.29) instead of deletions. Same profile (6006 token slots per expert,
30× the minimum), same budget of 40 core experts, same 384 removals; the only
difference is whether a removed expert is deleted or folded into a survivor. All three
arms scored in one harness on the same 20 held-out gsm8k prompts, 1122 tokens:

| arm | perplexity | vs baseline |
|---|---|---|
| baseline, 64 experts | 10.3579 | — |
| delete only, 384 removals | 12.7026 | 1.226× |
| **merge 233 + delete 151** | **14.6689** | **1.416×** |

Merging costs **15% more perplexity than deleting the very same experts**. Folding a
weakly-similar expert into a survivor damages the survivor — it was carrying its own
function — where deletion at least leaves the remaining experts intact and lets the
renormalised gate mass flow to them. So the 0.85 threshold is not conservatism for its
own sake: below it, merging is worse than the thing it is meant to improve on.

This also exercises the merge path end-to-end on real weights for the first time:
permutation alignment, usage-weighted averaging and the router rewrite all ran over 233
real clusters, and the result loads and scores. The machinery is correct; the operation
is simply not worth doing on these models.

One gap this exposed, now fixed: the quality gate zeroes experts, which emulates
deletion. A merge donor also carries `action == "drop"`, so it was being swept into the
zeroed set — measuring the cost of discarding an expert the plan does not discard, and
then attaching a verdict to a plan whose merges were never examined. Donors are now
excluded, the verdict reports `merges_not_gated`, and `apply_plan` warns that a passing
gate says nothing about merges.

## Surgery

```bash
surgeon apply --plan plan.json --source /path/to/model --out ./pruned
```

Streams one tensor at a time, so a model larger than host RAM is fine. Survivors
are renumbered contiguously and **router rows are reordered by the same mapping** —
a row left at its old index routes to a different expert than the one it was
trained for, and nothing raises. `top_k` is clamped if fewer experts survive than
the model routed to. Merges align neurons first (`surgery/align.py`): merging an
expert with a permuted copy of itself returns the original to 1e-6, where naive
elementwise averaging of the same pair produces a function unlike either.

Verified on GB10: OLMoE 64 → 40 experts in 7 seconds, 8.4G in 3 shards, then
loaded and generated through a plain `LLM()` — no plugin, no flags, no env.
Artifact A is an ordinary HF checkpoint as far as vLLM is concerned.

### Measured: hard deletion is not viable for OLMoE

Profiled on 400 gsm8k prompts — 6,006 token slots per expert, 30× the threshold:

| keep | routing load deleted |
|---|---|
| 56/64 | 2.8% |
| 48/64 | 7.8% |
| 40/64 | 14.6% |
| 32/64 | 23.5% |

**There is no dead tail.** All 64 experts are used in every layer, and the
*coldest* still carries 0.5% of load where uniform would be 1.56% — only ~3×
below uniform, not 100×. The hottest quarter carries 46–60% against 25% uniform,
so concentration is real but mild.

Planning from the thin 169-token profile and from the 6,006-slot profile produced
keep-sets sharing only 26 of 40 experts, so the thin ranking *was* mostly noise, as
the refusal claimed.

**Corrected 2026-08-08.** An earlier version of this section called the pruned
model "badly degraded", read off three greedy samples. Measured on held-out
gsm8k[400:500] it is **9.71 → 15.26 perplexity, 1.57×** — a real cost, but not the
collapse three samples suggested. The qualitative read overstated it.

Every pruning number here is **in-domain** (profiled and evaluated on gsm8k). A
permanent artifact is served on whatever traffic arrives, and deletion is
irreversible, so the number that matters for a real deployment is the cost under
distribution shift — which is larger: the same pruned-40 artifact scores **1.404× on
hellaswag** against 1.234× in-domain (the amplitude section below, where the
out-of-domain cost is also what the amplitude fix is validated against). Read the
in-domain ratios as a floor on the cost, not the cost.

All perplexity figures below are single measurements without a confidence interval,
and the eval slice size varies by section (gsm8k[400:500] here, 20 prompts for the
merge arm, `--limit 50` for amplitude), so each carries its own baseline — labelled
where they are juxtaposed. Up to this point "accuracy" means held-out perplexity,
a proxy — the downstream **task**-accuracy measurement, which changes the reading,
is in [Measured on downstream tasks](#measured-on-downstream-tasks--and-this-changes-the-reading)
below.

The ablation study puts that number in context (`surgeon ablate`):

| arm | perplexity | vs baseline |
|---|---|---|
| baseline, 64 experts | 9.71 | — |
| coldest 24/64 **zeroed** | 16.90 | 1.74× |
| coldest 24/64 **deleted** | 15.26 | 1.57× |
| hottest 24/64 zeroed (control) | 548.09 | 56.45× |

Two things follow. The count-based ranking is **real** — a 32× separation between
ablating the coldest 24 and the hottest 24. And deletion scores *better* than
zeroing, which is the predicted ordering: deletion lets the renormalised gate mass
flow to survivors while zeroing discards it. That ordering is also the check that
`apply_plan` is not damaging the model beyond the loss of experts.

### Writing back the stacked (Granite) layout

Checkpoints disagree on how expert weights are stored, and the difference is
structural. The per-expert layout (OLMoE, Qwen, DeepSeek, Mixtral exports) gives one
tensor per expert per projection. The stacked layout (IBM Granite) puts a whole
layer in `input_linear.weight` of shape `[E, 2I, H]` with gate and up already fused,
plus `output_linear.weight` of `[E, H, I]`.

The output keeps the source's layout, and that is not a preference. Per-expert
tensors written for a loader that expects stacked ones fail to load; the same
tensors written under stacked names load **wrong**. Two things have to be exact:

- **Half order.** Gate before up in each expert's slab — the order vLLM's granitemoe
  loader assumes when it does `w1, w3 = p[e].chunk(2, dim=0)`. Swapped, the
  checkpoint loads without a complaint and computes a different function.
- **Slab order.** New id *i* holds the *i*-th survivor, and the router rows must be
  reordered by the same mapping.

Streaming survives this only partly. A stacked tensor cannot be written
incrementally, so that path buffers one layer's survivors — peak cost is one layer
of experts, not one model.

Verified end to end on `granite-3.0-3b-a800m-base` (40 experts, top-8, 32 layers,
stacked, shipped in fp32): pruned to 30 experts, written back stacked, and **loaded
and generated in vLLM** — 13.5 GiB → 9.8 GiB, `num_local_experts` 40 → 30, routers
`[30, 1536]`.

| | perplexity | vs baseline |
|---|---|---|
| baseline, 40 experts | 2.2543 | — |
| pruned to 30, applied | 5.2835 | **2.34×** |
| gate's zeroing proxy predicted | — | 2.68× |

The proxy over-predicted the damage by 0.34×, in the same direction as on OLMoE
(1.30× predicted, 1.25× applied). Zeroing an expert leaves its gate mass stranded
while deletion redistributes it across the renormalised survivors, so the gate is
conservative by construction — on two families and both layouts now.

A per-layer share threshold is refused here rather than approximated: dropping
"everything below 1.8%" ended layers at 30, 32 and 33 survivors, and the HF config
has a single `num_experts` that cannot express that. Uniform budgets only.

### Selection count is the wrong quantity — rank-1 frequency is better

A top-k output is a gate-weighted sum, so an expert selected constantly at the
*last* slot moves it far less than one selected first. Every ranking above used
plain selection count. The position histogram in a capture lets us do better at no
extra cost, and the ablation decides which weighting is right rather than an
argument:

| ranking | keep-set overlap with count | raw load deleted | perplexity |
|---|---|---|---|
| count (all ranks equal) | 100% | 14.6% | 16.90 (1.74×) |
| linear, K−j | 88.5% | 15.3% | 15.34 (1.58×) |
| harmonic, 1/(j+1) | 85.7% | 15.7% | 14.56 (1.50×) |
| **rank-1 only** | **67.2%** | **21.6%** | **12.65 (1.30×)** |

Read the middle column with the last: **rank-1's keep-set discards half again as
much raw routing load and costs 43% less perplexity.** That is the clearest
statement that count ≠ contribution. Four points moving monotonically as weight
concentrates on the top slot make it a trend, not noise.

Applied end to end, ranking by rank-1 frequency instead of count cuts the pruning
cost from **1.57× to 1.25×** — excess perplexity 5.55 → 2.42, a 56% reduction, from
reading data already being collected. `build_plan` uses it by default.

It self-checks rather than assuming. vLLM's fused `topk_softmax` carries no
ordering contract (only the grouped and bias routers honour a `sorted` flag), so
`position_order_correlation()` measures whether frequently-chosen experts really do
occupy better slots — −0.48 on OLMoE — and the engine falls back to plain counts,
saying so in the plan, when they do not. That check is a population-level proxy,
not a direct test: verifying slot *j* holds the *j*-th best score would need the
scores, which the public capture does not carry.

### Surgery serves the tier — it is not an alternative to it

The two are not competing options, and reading the numbers as a contest was a
framing error on my part. Every measurement above says the tier keeps quality that
deletion costs, so deletion is not the product. What surgery does is make the tier
cheaper:

- **A smaller store.** Deleting the experts that genuinely contribute nothing
  removes their records from disk entirely.
- **A smaller candidate set.** The cache chooses residency among fewer experts, so
  the same VRAM capacity covers a larger fraction of what routing asks for.
- **A better residency prior.** The question "which experts deserve VRAM" is the
  same question as "which experts contribute most", which is exactly what rank-1
  importance measures. So the ranking that improved pruning also improves the
  tier's warm start.

For reference, what deletion alone costs at 40/64, so the tradeoff is on the record:

| approach | experts kept | resident | perplexity |
|---|---|---|---|
| delete 24, count-ranked | 40 | 40 | 15.26 (1.57×) |
| delete 24, **rank-1 ranked** | 40 | 40 | **12.13 (1.25×)** |
| **disk tier alone** | **64** | **24** | **~9.71 (same tokens as base)** |

Deletion is worth doing where an expert is genuinely worthless — it shrinks the
store and the candidate set for free. It is not worth doing to *replace* the tier.

## Amplitude: half of pruning's damage was a constant

The largest quality win in the project, and it is one multiply.

Deleting experts changes the *amplitude* of what survives. With `renormalize=False` —
OLMoE's setting — a surviving gate is the raw softmax probability, so restricting the
softmax from 64 rows to 40 inflates every one of them by `1 / (1 - P_D(x))`, where
`P_D` is the mass the deleted set used to carry. The model was never trained to receive
an inflated MoE branch.

Two things follow, and both were checked before any code was written:

**A router refit cannot fix it, and cannot help pure deletion at all.** Softmax over
the survivors is exactly Bayes conditioning: `softmax_K(W_R x)_e = p(e | chosen ∈ R)`
for every token, so the *unchanged* rows already minimise the divergence from the
teacher's conditional routing — at zero loss, on any corpus, with no calibration data.
Restriction-then-softmax is monotone, so the selected set is also already the teacher's
best available. A row refit here can only be a no-op or a regression.

**The correction belongs in `down_proj`.** A softmax sums to one over whatever rows
remain, so a uniform amplitude change is unrepresentable in the router — the same
obstruction as the gate having no bias term. But the layer's output is *linear* in
`down_proj`, so scaling it is exactly scaling the gate. `gate_proj` and `up_proj` must
not be touched: SwiGLU is nonlinear in them, so scaling those changes the function
rather than its amplitude.

```bash
surgeon calibrate --checkpoint ./pruned --corpus heldout.jsonl --limit 50
surgeon apply --plan plan.json --source model --out ./pruned --amplitude 0.85
```

OLMoE-1B-7B pruned 64 → 40 experts, every figure a fresh load:

| corpus | baseline, 64 | pruned, 40 | **pruned + amplitude 0.85** | damage removed |
|---|---|---|---|---|
| gsm8k, 2863 tokens | 9.6230 | 11.8754 (1.234×) | **10.5306 (1.094×)** | **60%** |
| hellaswag, 1685 tokens | 24.5961 | 34.5260 (1.404×) | **29.0684 (1.182×)** | **55%** |

hellaswag is the corpus the scalar was **not** fitted on, and 0.85 wins there too — and
beats 0.90 there as well, so the optimum is not an artefact of the fitting set. Roughly
half to three-fifths of pruning's perplexity cost was a systematic amplitude error, and
0.85 ≈ the mass the 24 deleted experts used to carry.

`surgeon calibrate` finds the value by measuring held-out perplexity at several scales
on one engine, rather than by estimating `P_D` from captured activations: it optimises
the thing being claimed instead of a proxy for it, and needs no capture path. What it
gives up is stated in the output — the points share one engine, and successive in-place
multiplies accumulate bf16 rounding. Measured drift: the sweep read 10.5087 where a
fresh load of the folded artifact reads 10.5306, 0.2% apart. So the sweep locates the
optimum and the artifact is re-measured after folding. That check also confirms folding
at write time and scaling at serve time agree: 10.5306 and 29.0684 on both corpora,
matching the in-place numbers exactly.

The search reports the whole curve, refuses an empty or negative bracket before loading
anything, warns when the best point sits at an edge (the true optimum would be outside
what was measured), and declines to credit a win under 1% — several times the rounding
drift, because a one-parameter fit deserves no more.

### Measured on downstream tasks — and this changes the reading

Every number above is held-out **perplexity**. Perplexity is a proxy, so the
rank-1 pruned-40 artifact and its amplitude-corrected version were scored on three
downstream tasks through `lm-eval` (500 items each, GB10), with a paired **exact
McNemar** test per task — the same instrument the fp8 store was held to
(paired exact McNemar over `lm-eval` samples):

| task (metric) | baseline | pruned-40 | Δ (McNemar p) | pruned-40 + amp 0.85 |
|---|---|---|---|---|
| arc_challenge (acc_norm) | 0.468 | 0.352 | **−0.116** (4.8e‑08 \*\*\*) | 0.350 |
| hellaswag (acc_norm) | 0.662 | 0.618 | −0.044 (0.014 \*) | 0.622 |
| gsm8k (exact, strict) | 0.100 | 0.058 | −0.042 (0.0055 \*\*) | 0.068 |

Two results that perplexity alone would not have told us, and one of them is
uncomfortable:

- **Deletion costs real, statistically-significant task accuracy on every task** —
  arc_challenge loses a quarter of its accuracy (−25% relative, p = 5e‑08). So the
  quality gate's default 1.3× **perplexity** ceiling does **not** certify task
  quality; a plan that clears it can still gut a benchmark. The gate is a guardrail
  on perplexity, and this table is the reason to say so plainly rather than let
  "1.25× perplexity" read as "1.25× as costly."
- **The amplitude fix recovers perplexity but not task accuracy.** It removed ~60% of
  the *perplexity* damage, yet against pruned-40 its task deltas are within noise on
  all three tasks (arc −0.002, hellaswag +0.004, gsm8k +0.010; McNemar p = 1.0, 0.83,
  0.53). A one-scalar `down_proj` correction fixes the average log-likelihood without
  restoring the decisions — a concrete case of perplexity and task metrics parting
  ways, and the reason this section exists.

This is the strongest evidence in the document for the framing the rest of it
argues: on these models deletion is a real quality cost, so the tier — which keeps
those experts — is the primary mechanism and deletion is worth doing only where an
expert genuinely contributes nothing.

## The quality gate

```bash
surgeon gate --plan plan.json --corpus heldout.jsonl --max-ratio 1.3
surgeon apply --plan plan.json --source model --out ./pruned   # refuses without a pass
```

Deletion is the one irreversible step in the pipeline, and its cost is measurable
before it is paid — so `apply_plan` refuses to delete from a plan that has not been
measured. `surgeon gate` ablates exactly the set the plan drops, compares held-out
perplexity to baseline, and writes the verdict back into the plan; the verdict then
travels into the artifact's `surgeon_manifest.json`, so an artifact records what
measurement authorised it. `--skip-gate` exists and is a deliberate choice.

Only deletions are gated. A plan that merely re-places experts between core and
disk loses nothing to measure, and merges are covered instead by the exactness
tests in `surgery/align.py`, since zeroing cannot emulate a merge.

The ceiling applies to a **pessimistic** bound. On OLMoE the same 40/64 plan scored
1.303× zeroed and 1.25× once actually applied, so it fails a 1.3 ceiling while the
real artifact would have passed. Rather than fudge the number, a narrow failure
carries a note saying exactly that, and the operator raises the ceiling knowingly
or measures the applied artifact.
