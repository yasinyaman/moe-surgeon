# moe-surgeon

Fit a domain-specialised MoE deployment into less memory: an NVMe expert tier,
with permanent offline pruning and merging where a model supports it.

A general MoE model carries experts for every domain it was trained on. A
deployment that only ever sees one domain pays for all of them. This package
measures which experts that deployment actually uses, then makes it cheaper to
serve — primarily by tiering cold experts to NVMe, and, where the measurements
justify it, by dropping the genuinely unused ones and merging redundant ones into
a smaller checkpoint.

**On the models measured here (OLMoE, Qwen3-30B) the disk tier is the primary
mechanism, not pruning** — they have no dead tail and no mergeable pairs, so
deletion costs quality (measured below) and exists to *support* the tier: a
smaller candidate set and store. Redundancy is a per-model property, so the
pruning and merging machinery is kept and tested; it is simply not the win on
these two families. See [DECISIONS.md](DECISIONS.md).

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

## The tier, served from outside the vLLM tree

```bash
pip install -e .   # registers a vllm.general_plugins entry point
vllm serve allenai/OLMoE-1B-7B-0924 \
  --additional-config '{"surgeon": {"expert_cache_size": 24,
      "store_dir": "./store", "ram_cache": 48, "fp8_store": true}}'
```

This is the claim the package exists to make good on. The prototype reaches the same
behaviour with ~800 lines of hooks threaded through upstream files; `compat/runtime.py`
does it with **one substituted quantisation method**, registered from a plugin,
touching no vLLM source. The simplification that made it small: the method already
receives `layer`, so the provider is built and stashed from inside it — no
`RoutedExperts` subclass, which drops the most churn-prone seam in the table.

Verified on GB10 against the in-tree implementation it replaces. In the same
residency shape (full-DRAM, no fp8) the out-of-tree tier is **token-identical to both
the in-tree tier and the untiered baseline**. With the fp8 store one prompt of three
diverges, which is quantisation and not the port.

Engagement is asserted, not inferred: 16/16 MoE layers hold a provider, 16/16 have
their full `w13_weight` released to `numel() == 0`, and layer 0 logged 128 hits /
93 misses. That check matters — the first attempt registered under the op name
`unquantized_fused_moe` when `CustomOp.__new__` looks up the *class* name, so the
substitution silently no-opped and **the tokens still matched**. Matching output was
not evidence of anything.

Activation is deliberately not `--moe-expert-cache-size`, which drives the in-tree
path; the two can therefore run side by side and be compared.

Scope, stated rather than implied: tensor parallelism is refused (the store identity
carries no rank component). `--enforce-eager` is **no longer required** —
see [Piecewise CUDA graphs](#piecewise-cuda-graphs-the-last-in-tree-only-capability).
Streaming the checkpoint into the store **is** ported and on by default —
see [Streaming load](#streaming-load-the-tier-now-wins-on-feasibility-too). **fp8
checkpoints are served too** (`compat/fp8_runtime.py`): verified token-identical to the
untiered fp8 baseline on `DeepSeek-Coder-V2-Lite-Instruct-FP8`. fp8 could not be the
unquantized path's one clean substitution — `Fp8MoEMethod` is not a `CustomOp`, so it
goes through a `Fp8Config` shadowing the `"fp8"` config, and the override copies
`_setup_kernel` to install the cache before the quant config captures the scales — so it
costs three more (optional) internal seams, the price recorded in
[DECISIONS.md](DECISIONS.md).

## Sizing it: the only knob that matters, and what it costs

Everything else in this README is a mechanism. This is the number to get right, and it
was measured after the mechanisms were built — which is how it came to be a surprise.

| `expert_cache_size` (of 64) | decode | GPU expert bytes | peak VRAM |
|---|---|---|---|
| 24 | 55.3/s | 4.50 GiB | 29.29 GiB |
| 32 | 75.5/s | 6.00 GiB | 29.29 GiB |
| 40 | 107.9/s | 7.50 GiB | 29.41 GiB |
| **48** | **143.0/s** | 9.00 GiB | 29.25 GiB |
| *untiered* | *218.2/s* | *12.00 GiB* | — |

OLMoE-1B-7B on GB10, `ram_cache 64`, eager, 8 prompts × 256 tokens, 3 repeats.
**2.59× across one integer.** Held-out perplexity is 11.6253 on every row including the
untiered one; the stronger check — a seed-pinned greedy run hashed to the token id — was
run at capacity 48 and matched untiered exactly, three repeats each.

Two things that are not obvious from the table:

- **Set it against the per-layer expert union of your serving batch, not against free
  VRAM.** At batch 8 that union measured 35.3 mean / 46 max, so 24 slots are 47%
  oversubscribed: every step re-fetches the shortfall over the host→device link, and the
  forward additionally splits into chunks. Roughly 80% of the 2.59× is simply moving
  fewer bytes (663 → 117 MB per token); the rest is the split. The runtime now warns,
  once per layer, when a decode step routes to more experts than there are slots.
- **The slots are paid for out of KV cache, not out of device memory.** The peak-VRAM
  column is flat: vLLM sizes KV to fill whatever `gpu_memory_utilization` leaves, so
  capacity buys throughput and spends context. `surgeon budget` prints the per-slot cost
  so the trade is explicit.

Corollaries worth stating because each was measured rather than assumed: `ram_cache`
below the expert count turns every eviction into a disk read (48 → 64 was 1.66× and cut
the median decode time markedly); `fp8_store` on a non-fp8 checkpoint is a **space**
mechanism, costing 1.11× decode and bit-exactness to halve disk and host RAM; and no
cache policy or prefetch substitutes for capacity — Belady's optimum is worth 1.068× and
known-future prefetch measured 1.03–1.075×, because residency is a byte count and
scheduling does not change how many bytes must cross the link.

## Three axes, and why nothing is judged on one

Every method is measured on **feasibility** (does it run, on how small a device),
**speed** (tokens/s) and **accuracy** (held-out perplexity) — and a method that
loses on one axis is kept if it wins on another. [DECISIONS.md](DECISIONS.md) is the
register, with the per-axis verdict for each method and the rules for choosing a
strategy per target and per model.

Measured on OLMoE-1B-7B, GB10, held-out gsm8k:

| configuration | boot floor | load | decode | perplexity |
|---|---|---|---|---|
| baseline, 64 resident | 14.60 GiB | 98.6 s | 689.4/s | 9.703 |
| disk tier, 24/64 resident | 23.40 GiB | 45.2 s | 265.1/s | 9.734 (1.003×) |
| **disk tier + streaming load** | **11.35 GiB** | — | — | — |
| pruned to 40, no tier | not measured | 83.4 s | 695.1/s | 12.115 (1.249×) |
| **pruned 40 + tier, 24/40** | not measured | **44.5 s** | **350.8/s** | 12.145 (1.252×) |

The fourth row is why surgery exists: **pruned+tier decodes 1.32× faster than tier
alone** at the same capacity, because a 24-slot cache covers more of a 40-expert
candidate set than a 64-expert one. And the tier **halves load time**, which no
accuracy or throughput number would have surfaced.

Two honesties about these numbers. Each is a **single run** — no repeats, no variance
(boot-floor probes vary ~15% in this very environment). And the `decode` column is
output tokens over **prefill+decode** wall time by construction (`compat/bench.py`
sets `prefill_seconds=0.0` and times the whole generate), with short prompts chosen so
prefill is ~1% of it; the contamination is not perfectly arm-neutral, since the tier
arms pay their prefill through disk reads inside that denominator. So read 0.38× and
1.32× as ratios with a few-percent error bar, not to three figures.

Two examples of the per-axis rule doing real work. Static gate geometry is
worthless for *choosing* experts (ρ ≈ 0.00) and genuinely useful for *sizing* them
(`surgeon budget`) — same analysis, different question. An fp8 store saves no VRAM
at all, and halves disk, host RAM and transfer bytes.

## Layout

| path | imports vLLM | what it is |
|---|---|---|
| `compat/` | **yes** | the seam layer; the only place that knows vLLM internals |
| `plugin.py` | **yes** | the `vllm.general_plugins` entry point |
| `telemetry/` | no | aggregation of vLLM's routed-experts capture |
| `store/` | no | the NVMe → pinned RAM → VRAM expert tier, plus its replay simulator |
| `surgery/` | no | inspect, budget, plan, align, merge, apply, tier |
| `server/` | no | the job server: stage orchestration, job records, HTTP |

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

## The job server

```bash
surgeon serve --port 8300 --state ./surgeon-state   # binds 127.0.0.1
```

The server runs subprocesses from request fields, so it binds loopback by default.
Exposing it on a non-loopback host requires `--token` (or `$MOE_SURGEON_TOKEN`),
which every request then carries in an `X-Surgeon-Token` header; `/health` stays
open. Engine kwargs in a request are allow-listed (`trust_remote_code` is refused),
and a stage never fetches an uncached model unless the job sets `allow_download`.

```bash
curl -sX POST localhost:8300/jobs -H 'content-type: application/json' -d '{
  "model": "allenai/OLMoE-1B-7B-0924",
  "corpus": "domain.jsonl", "heldout": "heldout.jsonl",
  "core_experts": 24, "stages": ["profile", "plan", "gate", "tier"]
}'
```

It adds orchestration, persistence and a record. It adds **no capability** — every
stage is a `surgeon` subcommand, and a job record stores each stage's argv, so any
failed run is reproducible by hand. That is deliberate: the debuggable version of a
pipeline server is one that cannot do anything you could not do yourself.

Three properties are load-bearing rather than incidental:

**Stages are subprocesses.** A `profile` or `gate` stage boots a vLLM engine, and an
engine does not release device memory when its Python object goes out of scope — the
same thing that broke the first benchmark harness, where four arms in one process
left two unable to boot. A crashing stage must not take the server with it either.

**One worker, not a pool.** Two pipelines would both claim the GPU and the second
would fail to boot with an out-of-memory error that reads like a model bug rather
than a scheduling mistake. Concurrency here manufactures confusing failures.

**Unrunnable requests are rejected before anything runs**, as a 400 naming the field.
This was learned the expensive way: a run spent 114 seconds profiling and then failed
in `tier` with `no safetensors checkpoint under allenai/OLMoE-1B-7B-0924`, because
`apply` and `tier` open safetensors directly and had been handed a repo id.

The fix was to *resolve* rather than to *demand*. A repo id resolves through the HF
cache with `local_files_only`, so a request never needs a hash-named snapshot path —
but only for the files those stages read (`*.safetensors`, the shard index,
`config.json`). A plain `local_files_only` resolution insists the **entire** repo be
cached and raises `IncompleteSnapshotError` otherwise, which is the normal state of a
cache vLLM filled: it never fetches a README.

A gate failure is recorded as a **verdict, not an error** — the job stops, the
pipeline is marked failed, and the reason is that the model did not clear the
threshold. That distinction matters, because "surgery would hurt too much" is a
successful measurement and should not read as a broken run.

Measured on GB10 from a bare repo id, OLMoE, 24 core experts:

| stage | | |
|---|---|---|
| `profile` | succeeded | 103.0 s |
| `plan` | succeeded | 1.4 s |
| `gate` | succeeded | 1.5 s |
| `tier` | succeeded | 17.5 s |

producing `profile.npz`, `plan.json`, a 6.1 GiB fp8 store over 16 layer files, and a
`hot_experts.json` carrying 24 core ids plus a prior for **all 64** experts per
layer — which is the seam the plan recorded as unfilled, now filled by a pipeline
rather than by hand.

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


#### And when merging is forced anyway, it is worse than deleting

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
where they are juxtaposed. There is no downstream **task**-accuracy measurement
anywhere in this document; "accuracy" throughout means held-out perplexity, a proxy.

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

## Feasibility: the measured floor, not the preallocated pool

`bench.py`'s peak-VRAM reading was never a floor. vLLM claims
`gpu_memory_utilization` of the device up front, so on a unified-memory box the
number tracks the fraction it was told to take — asking for 35% of 122 GiB reports
~45 GiB whether the model is 7 B or 3 B. `surgeon budget`'s arithmetic is right about
weights and silent about activation peaks, the graph pool and fragmentation.

`surgeon vram-floor` measures the quantity directly: boot at a budget, generate a
token, bisect for the smallest budget that survives. One process per attempt, because
an engine does not release device memory when its Python object goes out of scope.

```bash
surgeon vram-floor --model ./pruned --low 0.03 --high 0.10 --tolerance 0.003
```

Granite 3.0-3b-a800m on GB10 (122 GiB device), pruned from 40 experts to 30:

| arm | boot floor | resident weights (arithmetic) | overhead |
|---|---|---|---|
| baseline, 40 experts | **7.37 GiB** (failed 0.058, booted 0.061) | 6.29 GiB | 1.08 GiB |
| pruned, 30 experts | **6.04 GiB** (failed 0.048, booted 0.050) | 4.88 GiB | 1.16 GiB |

Deleting 25% of the experts lowered the floor by **1.33 GiB**, against 1.41 GiB
predicted by `surgeon budget` — agreement inside the ±0.36 GiB tolerance, with a
consistent ~1.1 GiB of non-weight overhead on both arms. That is the calibration the
feasibility axis was missing: the arithmetic is trustworthy, plus about a gigabyte.

Two traps this measurement exposed, both of which produce a confident wrong number:

**A search that never fails has not found a floor.** The first run reported both arms
at "floor 0.059" — identical, suggesting pruning bought nothing. Nothing had failed;
the bisection hit its tolerance while still above the boundary, so 0.059 was just the
smallest budget probed. An unbracketed result now says `UNBRACKETED` and reports
`<=`. The arms are 1.33 GiB apart, and the first run showed them as equal.

**Checkpoint dtype is not serving dtype.** Granite stores fp32 and its config names no
dtype at all, so `surgeon budget` counted 12.57 GiB against a measured 7.37 GiB
floor — an apparent contradiction. HF treats a silent config as fp32 and vLLM's
`_resolve_auto_dtype` downcasts fp32 to the platform's preferred 16-bit width, so
resident weights are 6.29 GiB. `surgeon budget` now says so, and only ever narrows —
an fp8 checkpoint whose config says `bfloat16` keeps fp8 weights resident, and
costing those at 2 bytes would be the same error inverted.

### Measured: without streaming load, the tier raises the boot floor

The uncomfortable result, and the reason the feasibility axis was worth building an
instrument for. OLMoE-1B-7B on GB10 (121.63 GiB unified memory), every arm bracketed
by an observed failure:

| arm | boot floor | bracket |
|---|---|---|
| untiered, 64 resident | **14.60 GiB** | failed 0.116, booted 0.120 |
| disk tier, 24 slots, `ram_cache` 48, fp8 store | **23.40 GiB** | failed 0.190, booted 0.192 |
| full-DRAM, 24 slots (`ram_cache` 0) | **31.10 GiB** | failed 0.254, booted 0.256 |
| full-DRAM, 10 slots (`ram_cache` 0) | **28.53 GiB** | failed 0.232, booted 0.235 |

**Correction to an earlier version of this table.** Rows three and four were first
written up as *tiered* arms with `ram_cache 0`, and they are not tiered at all.
`RuntimeConfig.use_disk` is `bool(store_dir) and ram_cache > 0`, so `ram_cache 0`
builds no store: the provider takes its full-DRAM branch and keeps `_cpu_w13` /
`_cpu_w2`, every expert page-locked for the process lifetime, where the disk branch
sets both to `None`. Both modes logged the same "expert cache active" line, so
nothing distinguished them. Three fixes came out of that, all in this repo now: the
combination is **refused** rather than silently resolved, the log **names the mode**,
and `BisectResult` **records the kwargs each arm booted with** — a floor without its
configuration is a number whose label nothing can check.

So there was never a `ram_cache` anomaly. The 7.7 GiB was the difference between
*building a bounded store* and *holding all 64 experts pinned forever* — the opposite
direction from "less host cache is cheaper".

What remains is real: the genuine tiered arm still needs **23.40 GiB against 14.60
untiered**. The accounting closes on known allocations (OLMoE: 16 layers × 64 experts,
12 MiB/expert, 12.0 GiB of expert weights):

| term | tiered, `ram_cache` 48 fp8 | full-DRAM, 24 slots |
|---|---|---|
| pinned set from `create_weights` | 12.0 | 12.0 |
| provider's own full mirror | — | 12.0 |
| warm pool (48 × 6.02 MiB × 16) | 4.5 | — |
| device slot buffers (12 MiB × cap × 16) | 4.5 | 4.5 |
| non-expert weights and overhead | ~2.6 | ~2.6 |
| **predicted** | **~23.6** | **~31.1** |
| measured | 23.40 | 31.10 |

The capacity axis checks independently: 24 → 10 slots predicts 2.63 GiB, measured
2.57 GiB.

**Why host bytes show up in a device floor.** On GB10, `MemorySnapshot.measure`
substitutes `psutil` available memory for `cudaMemGetInfo`'s free value on integrated
GPUs, so page-locked host memory is charged against `gpu_memory_utilization`. That is
what makes the 12.0 GiB pinned set from `create_weights` — allocated on the host
precisely so loading would *not* need device capacity — land in the floor anyway.

**And a second mechanism, which the port did not account for.** `create_weights`
allocates `device="cpu"` on purpose (`runtime.py`: "so loading never needs device
capacity for the full expert set"), but vLLM wraps `process_weights_after_loading` in
`device_loading_context`, which hoists every `cpu`-typed parameter to the device
first. So `build_provider` reads a **CUDA** `w13_weight` holding all 64 experts, alive
until `replace_parameter`. It is per-module, so ~768 MiB for one layer at a time
rather than 12 GiB — but it also means the provider's `_pinned_cpu_copy` copies *from
device*, which is why full-DRAM mode allocates a second full pinned mirror instead of
reusing the one already there.

### Streaming load: the tier now wins on feasibility too

The fix follows from the accounting: intercept the weight loader so each expert goes
straight into the store and the `[num_experts, ...]` tensor never exists. Parameters are
allocated with **zero** experts, each arriving shard is written into a staging record,
and the record is pwritten and dropped the moment its three shards complete — peak cost
is the experts in flight, not the model. It also leaves nothing for
`device_loading_context` to hoist and nothing for `replace_parameter` to release.

```bash
vllm serve allenai/OLMoE-1B-7B-0924 \
  --additional-config '{"surgeon": {"expert_cache_size": 24, "store_dir": "./store",
      "ram_cache": 48, "fp8_store": true}}'
```

Streaming is **on by default whenever the disk tier is on** — the non-streaming path
costs 12.0 GiB of page-locked host memory and buys nothing, so it is not somewhere to
land by leaving a flag unset. `"stream_load": false` forces the old path; a layer that
is not act-and-mul, whose record layout streaming cannot express, falls back with a
warning rather than failing.

| arm | boot floor | bracket |
|---|---|---|
| untiered, 64 resident | 14.60 GiB | failed 0.116, booted 0.120 |
| tier 24/64, fp8 store | 23.40 GiB | failed 0.190, booted 0.192 |
| **tier 24/64, fp8 store, streamed** | **11.35 GiB** | failed 0.091, booted 0.093 |

Streaming removed **12.05 GiB** where the accounting attributed 12.0 GiB to the
page-locked expert set — prediction and measurement agree to 0.05 GiB. The tier is now
**22% below untiered** rather than 60% above it, so it finally earns the feasibility
axis as well as the load-time one. Boots got faster too, 16–19 s per probe against
26–28 s, because the checkpoint is no longer read into host memory in full.

Where the port differs from the prototype: the prototype intercepted inside
`RoutedExperts.weight_loader`, which meant subclassing a 1700-line class. Here the
loader vLLM would have used arrives in `create_weights`'s `extra_weight_attrs` and is
stamped on each parameter, so wrapping it needs no subclass — and anything that is not
one of the three expert shards passes through to vLLM's own loader untouched. A
positional call is delegated rather than guessed, because guessing an argument order
would write a shard into the wrong half of the wrong expert and nothing downstream
would object.

Two things checked rather than assumed:

- **A streamed store and an offline-built one are byte-identical.** Same record specs,
  same identity, same fingerprint — verified in-process on a real expert: identical
  scales, identical quantised bytes, reconstruction delta 0.0. So `surgeon tier` and a
  streamed boot are interchangeable, and a second boot reuses the first boot's store
  rather than restreaming it. (An *older* store left on the GB10 does differ, by one
  e4m3 step: it predates the current code, and the fingerprint covers geometry and
  identity rather than the code that wrote the bytes.)
- **Sealing refuses an incomplete expert.** A record whose three shards never all
  arrived would be zeroed, and a zeroed expert is silent — the model loads and its
  output is quietly wrong for every token routed there.

What it is *not*: on a **discrete** GPU this saves zero device bytes, because under the
tier the full set was never device-resident — `create_weights` allocates `device="cpu"`
deliberately. It is a host-memory feature that becomes a device-memory feature only
where host and device draw on one pool. Stated plainly because the opposite is the easy
assumption to make.

The non-streaming numbers are kept rather than replaced: they are what the tier costs
without this, and the 23.40 → 11.35 GiB gap is the measurement that justifies the
mechanism. None of it overturns the rest — the tier still halves load time (98.6 s →
45.2 s) at 1.003× perplexity, and pruned+tier still decodes 1.32× faster than tier
alone.

### Piecewise CUDA graphs: the last in-tree-only capability

The one thing the prototype did that the plugin could not: run under CUDA graphs. The
cache's `prepare()` is dynamic host code — LFRU bookkeeping, a D2H routing sync, H2D
weight copies — and none of it may be captured into a graph. The in-tree version
handled this in two places: a `VllmConfig` post-init that adds `vllm::moe_forward` to
`compilation_config.splitting_ops` so the MoE op is *carved out* of the captured region
and runs eager, and a `MoERunner` that copies the eager op's output to a
**capture-stable address** so the next graph piece reads a fixed location. Neither is a
method substitution, so out of tree the plugin simply required `--enforce-eager`.

`compat/graph_runtime.py` pulls both in. The split is injected at config time by
**wrapping `VllmConfig.__post_init__`** — a plugin has no `offload_config` field to gate
on and cannot edit the config class, and the plugin entry point runs inside
`EngineArgs.__post_init__`, before any `VllmConfig` is built, so the wrap is in place
when the config that matters is post-initialised. (Injecting later, from the runner's
`__init__`, is too late: the split points are fixed before the runner is constructed,
and the MoE op ends up captured — it aborts with `operation not permitted when stream is
capturing`. That failed attempt is why the injection lives at config time.) The output
stabilisation is a genuine `MoERunner` substitution via `register_oot`, copying the MoE
output into a persistent per-shape buffer on piecewise passes.

Verified on GB10, OLMoE-1B-7B, greedy, four prompts, against three controls:

| arm | stabilised copies | output |
|---|---|---|
| tier, eager | 0 (no graphs) | **token-identical to untiered eager, all 4 prompts** |
| tier, piecewise graphs | 2464 | 3/4 prompts identical to tier-eager; 1 diverges in its tail |
| tier, graphs, **stabilisation off** | 0 | **garbage — one token id repeated 24×** on every prompt |

The three rows are the whole argument. Row 1: the cache is numerically transparent —
in eager mode the tier reconstructs weights bit-exactly, so it matches the untiered
baseline token-for-token. Row 3 is the control: split the MoE op out but *don't*
stabilise its address, and the captured downstream piece reads a frozen workspace view
— every step yields the same logits, so greedy decoding repeats one token. That is the
bug the stabilisation exists to prevent, and turning it off reproduces it exactly. Row
2 is the feature working: 2464 stabilised copies prove the piecewise path actually ran,
and the output tracks the eager tier except in one greedy-unstable tail — where
*untiered* graphs also diverge from *untiered* eager (CUDA graphs are not bit-identical
to eager in stock vLLM either), so the divergence is graph-vs-eager float noise, not the
tier. `--enforce-eager` is now optional; if `graph_runtime` fails to install (MoERunner
moved), `validate()` reinstates the requirement rather than risk a silent
capture-address bug.

**What it buys, measured — and it is not throughput.** Benchmarked on GB10 (OLMoE, 8
prompts × 256 tokens, 3 repeats) and on a 4 GiB laptop GPU:

| arm | decode | load | note |
|---|---|---|---|
| untiered, eager | 218.2/s (σ 0.001) | 117.9 s | |
| untiered, **graphs** | 226.4/s (σ 0.008) | 119.9 s | graphs buy the baseline **+3.8%** |
| tier 24/64, ram 64, eager | 55.8/s (σ 0.044) | 13.0 s | (undersized cache — see below) |
| tier 24/64, ram 64, **graphs** | 55.2/s (σ 0.044) | 35.0 s | graphs buy the tier **nothing** |

Those tier rows run a 24-slot cache against a measured 46-expert per-layer working set, which
is the wrong configuration; sized correctly (48 slots) the tier decodes at **143.0 tok/s**.
That does not change the graphs-vs-eager conclusion — it is the same MoE op staying eager
either way — but do not read 55.8 as what the tier costs. See
[DECISIONS.md](DECISIONS.md#the-capacity-sweep-and-the-finding-that-dominates-everything-else-here).

Graphs speed the *baseline* up by 3.8% and the *tier* not at all — which is the design
working as intended rather than a disappointment: the whole point of the split is that
the MoE op stays **eager**, so the part of the step that dominates under the tier is
exactly the part no graph can capture. Graph mode also costs ~22 s of capture at load
(13.0 → 35.0 s).

It costs device memory too, and the 4 GiB laptop GPU puts a number on it. At
`gpu_memory_utilization` 0.83 the tier runs fine eager (2.69 GiB peak, 7.7 tok/s, 6.6 s
load) while the same config with graphs cannot allocate a single KV block and vLLM
refuses at boot — loudly, with a clear message. Raise the budget to 0.90 and graphs do
fit (2.71 GiB peak, 29.4 s load): so the requirement is roughly **7 more points of
utilisation, ≈290 MiB of headroom, and 4.5× the load time** on that card, not an
impossibility. The decode difference there (7.8 → 8.3 tok/s) was a single run with no
repeats, so it is below the resolution of that measurement and no claim is made from it.

So the honest framing is that S5 is a **compatibility** win — the tier now composes with
vLLM's default (non-eager) serving mode instead of demanding a flag — and on
memory-tight or load-latency-sensitive deployments `--enforce-eager` remains the better
choice, now as a tuning decision rather than a hard requirement.

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
([notes/kalite-eval.md](../../notes/kalite-eval.md)):

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

Three caveats before reading this table. It is a **single-tier** `store/replay.py` run
(pure numpy) that has **no in-repo driver** — there is no `surgeon replay` subcommand,
so the table cannot be regenerated today. The warm arm is measured with the replay's
**prewarm placement** (experts placed resident before the first token); the shipped
runtime seeds only the prior *bias* (`hot_experts.seed_policy` sets `policy.prior`, it
does not place), so a real warm start realises less than the +10 shown. And +10 pts at
N=50 is ~5 cache hits (binomial SE ≈ 6.6 pts) with no repeats — indicative, not precise.

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

Faz 0–5 complete, and the tier now costs **less** memory than not using it: streaming
load took its boot floor from 23.40 GiB to 11.35 GiB against 14.60 GiB untiered. The
out-of-tree runtime is verified against the implementation it replaces, and the full `profile → plan → gate → tier` pipeline has run end-to-end
through the job server from nothing but a repository id. Both expert layouts are now
written as well as read, and all three axes are measured — feasibility last, with the
instrument that produced this project's one clearly negative result (the tier's boot
floor). `DECISIONS.md` lists the remaining open items with reasons, streaming load
first.

Three expert-storage layouts read (per-expert and stacked), three model families
measured (OLMoE, Qwen3-30B, DeepSeek-V2-Lite, Granite).

First real measurement, OLMoE-1B-7B (64 experts, top-8, 16 MoE layers), 3 prompts
/ 169 tokens: `mean experts/tok` came back exactly 8.000, all 16 layers captured,
no dropped rows. 54–64 of 64 experts were touched *per layer*, and the hottest
quarter of experts carried 50–66% of load against 25% for a uniform
distribution. So the concentration is real (~2–2.6x), but at 169 tokens this is a
pipeline check, not a pruning basis — it also reproduces the earlier finding that
a handful of prompts already reaches nearly the whole expert set.
