# moe-surgeon

Permanent MoE expert pruning and merging for domain-specialised vLLM deployments.

A general MoE model carries experts for every domain it was trained on. A
deployment that only ever sees one domain pays for all of them. This package
measures which experts that deployment actually uses, then produces a smaller
checkpoint — permanently, offline — by dropping the unused ones and merging the
redundant ones. Experts too cold to keep resident but too useful to delete go to
an NVMe tier instead of being thrown away.

The gain is a smaller model, not a runtime tradeoff.

## Why it is a separate package

The predecessor of this work lives inside a vLLM fork: 20 files, +5063 lines,
about 800 of them hooks threaded through upstream files. Every vLLM release turned
that into a rebase argument — at one point upstream deleted the `FusedMoE` class
outright and every hunk lost its target.

So the rule here is structural, and a test enforces it:

> Only `compat/` may import vLLM.

Everything else — the surgery pipeline, the checkpoint writer, the disk store, the
job server — runs on a host with no vLLM, no CUDA and no GPU. That is what makes
the offline half testable on a laptop, and it confines the cost of a vLLM upgrade
to one reviewable directory.

`compat/seams.py` declares every vLLM internal we depend on as data, with the
reason we hold it. `tests/test_seams.py` checks each one. On an upgrade, that test
fails first — before a worker does, and with a message naming the symbol that
moved.

## Layout

| path | imports vLLM | what it is |
|---|---|---|
| `compat/` | **yes** | the seam layer; the only place that knows vLLM internals |
| `plugin.py` | **yes** | the `vllm.general_plugins` entry point |
| `telemetry/` | via compat | per-expert usage recording during a forward pass |
| `store/` | no | the NVMe → pinned RAM → VRAM expert tier |
| `surgery/` | no | prune, cluster, permutation-align, merge, router refit |
| `writer/` | no | safetensors + `config.json` + provenance manifest |
| `server/` | no | the job server |

`store/` is a verbatim lift from the prototype branch — only its imports were
rewritten (`vllm.envs` → `env.py`, `vllm.logger` → `_logging.py`). It is kept
diffable against `repos/vllm` on `disk-tier-proto` on purpose, which is why
`pyproject.toml` exempts it from two lint rules rather than reformatting it.

## Tests

```bash
python3 -m pytest tests/ -q
```

Three levels, and they run in different places:

- **CPU** (`test_store_cpu.py`, `test_layering.py`, table checks in
  `test_seams.py`) — run anywhere, no GPU, no vLLM. This is the laptop loop.
- **static seams** (`test_seams.py`) — parses a vLLM *source tree*; point it with
  `MOE_SURGEON_VLLM_SRC=/path/to/vllm`, or let it find the sibling checkout.
- **CUDA / installed vLLM** (`test_expert_cache.py`, import-level seam checks) —
  skip without a device. These run on the GPU boxes.

A large `skipped` count on a laptop is expected, not a warning sign.

## Pricing a vLLM upgrade

```bash
python tools/upstream_drift.py --repo ../vllm --from <pinned-ref> --to <candidate-ref>
```

Reads the seam modules straight out of git at both refs and reports, per seam,
whether it holds, broke, or newly appeared. No install, no torch, no GPU. Exit
status is 1 if a required seam broke, so it can gate a pin bump.

Measured 2026-08-07, merge base → upstream `main`, **296 commits**: 342 lines
changed across the five files we hold, **0 required seams broken**, 1 optional
seam newly available (`RoutedExperts._orient_fused_weight` — upstream extracted
the fused-weight orientation branch we would otherwise have duplicated).

That number is the argument for this package. The same window would have landed
on the fork's ~800 lines of in-file hooks: `moe_runner.py` alone lost 41 lines,
`config/vllm.py` gained 195.

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

## Cold start: what the residency prior is actually worth

`store/replay.py` replays a real routing trace through the real cache policies —
pure numpy, no GPU, no engine — so the prior can be evaluated instead of assumed.
(It also restores a capability the prototype lost: its docstrings referenced a
`bench/hit_rate_sweep.py` that is not on the branch.)

Measured at capacity 24 of 64, priors derived from **gsm8k** and the trace replayed
from **hellaswag** — a different domain on purpose, since scoring the prior on the
corpus it came from is scoring the answer key:

| arm | N=50 | N=100 | N=200 | N=500 | N=2000 | overall |
|---|---|---|---|---|---|---|
| cold, LFRU (today's default) | 32.0% | 41.0% | 47.0% | 57.0% | 58.2% | 70.1% |
| cold, EWMA | 32.0% | 42.0% | 50.0% | 59.6% | 60.5% | 71.3% |
| **warm, EWMA + prior** | **42.0%** | **49.0%** | **55.0%** | **62.0%** | 61.0% | 71.3% |

Two separable wins, and it matters not to conflate them:

- **The policy is worth more than the prior.** EWMA over LFRU is +2.3 to +4.0
  points on the cold-start window and +1.2 sustained, for one env var
  (`VLLM_MOE_CACHE_POLICY=ewma`).
- **The prior is real but short-lived**: +10.0 points over the first 50 accesses,
  decaying to +7, +5, +2.4, then nothing by 2000. That is exactly the shape
  `DEFAULT_SCALE` was specified for — "large enough to decide the cold-start
  window, small enough that observed traffic overrules it soon after". With
  capacity 24 and top_k 8 a cache saturates in about 4 tokens per layer, so the
  window a prior can win is genuinely narrow. It costs nothing to ship, since the
  manifest is a byproduct of the plan, and should not be oversold.

An earlier version of this measurement used a 2000-access window and found the
prior worth nothing. That window was simply too long to see it.

## Cold start, and what static analysis is worth

```bash
surgeon inspect --checkpoint /path/to/model --profile profile.npz
```

A cold cache learns residency from traffic, so every restart pays misses on
experts a previous run already knew were hot. Static analysis of the router is the
tempting fix — free, no engine, no data. `surgeon inspect` reports which signals a
checkpoint carries **and scores each one against a measured profile**, because
whether a signal carries information is a property of the model, not of the idea.

Measured on OLMoE (16 layers, 400-prompt profile), mean rank correlation:

| candidate signal | ρ | verdict |
|---|---|---|
| gate row norm → expert load | −0.03 | no signal, sign flips per layer |
| gate row norm, centered → load | −0.05 | no signal |
| gate cosine → co-occurrence | +0.00 | no signal |
| gate cosine, centered → co-occurrence | +0.00 | no signal |

The target is highly predictable — real load spreads 18.5× across experts — and
the static signals captured none of it. Centering the gate rows to remove their
shared component (which top-k cancels anyway) did not help. Gate rows are *not*
near-orthogonal — Qwen3-30B's median |cos| is 0.47–0.57, with pairs up to 0.99 —
so there is geometry there; it simply does not predict routing.

**What does work for cold start is a profile from a different domain.** gsm8k vs
hellaswag on OLMoE:

| | ρ | top-24 overlap | random |
|---|---|---|---|
| cross-domain load | **+0.43** | **54.2%** | 37.5% |

So roughly 1.4× better than random residency, for free, from any profile you
already have. That dominates static analysis outright, and a profiling run is
~5 minutes — the window where "no profile exists" is real is very short. Ship a
generic profile with the artifact rather than deriving a prior from weights.

Two static signals remain worth having, for different reasons. The aux-loss-free
balancing bias (`e_score_correction_bias`, DeepSeek-V3 style) is a *learned*
load-balancing correction, so a large negative value is a direct popularity
readout rather than a geometric proxy — `inspect` detects it and scores it with
the inverse sign. Neither OLMoE nor Qwen3-MoE has one, so it is unvalidated here.
Per-expert SVD spectra are captured under `--spectra` as a candidate
quantization-tolerance signal, but acting on them would need a store format
change: records carry one dtype per layer, not per expert.

Not pursued, with reasons: **correlation-driven disk layout** — the record stride
is already 6.0 MiB and this project's own NVMe probe found a single thread reaches
93% of the 6.9 GB/s ceiling at expert-sized reads (granularity only has to clear
~1 MB), so co-locating correlated experts has almost no bandwidth left to recover.
**Cross-expert block dedup** — untested, but the similarity results argue against
finding duplicate blocks in distinct trained bf16 experts.

### Generalisation: Qwen3-30B-A3B

128 experts, 5 of 48 layers sampled, **81,280 pairs**: median similarity 0.034,
max 0.401, exactly **one** pair at or above 0.4. Doubling the expert count and
tripling the depth did not produce mergeable experts — if anything Qwen's experts
are *more* orthogonal than OLMoE's.

## Status

Faz 0–4 complete, verified on GB10: 199 tests locally, 234 with CUDA and vLLM.

First real measurement, OLMoE-1B-7B (64 experts, top-8, 16 MoE layers), 3 prompts
/ 169 tokens: `mean experts/tok` came back exactly 8.000, all 16 layers captured,
no dropped rows. 54–64 of 64 experts were touched *per layer*, and the hottest
quarter of experts carried 50–66% of load against 25% for a uniform
distribution. So the concentration is real (~2–2.6x), but at 169 tokens this is a
pipeline check, not a pruning basis — it also reproduces the earlier finding that
a handful of prompts already reaches nearly the whole expert set.
