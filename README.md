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

Generation at 40/64 is badly degraded either way, and the isolating experiment
settles why. Planning from the thin 169-token profile and from the 6,006-slot
profile produced keep-sets sharing only 26 of 40 experts — so the thin ranking
*was* mostly noise, as the refusal claimed. But the real profile did not rescue
quality; it lost "The capital of France is Paris" that the noisy one kept. The
damage is the deleted 14.6% of routing, not the choice of which experts to delete.

Together with the similarity result — no mergeable pair anywhere in the model —
that leaves one viable path for this model: **keep every expert, and use the disk
tier to decide what stays resident.** Artifact B is not the fallback, it is the
answer. Hard deletion would need distillation to recover, which this pipeline
deliberately does not do.

## Tiering — where the win actually is

```bash
surgeon tier --plan plan.json --source /path/to/model \
    --store ./store --hot-experts ./store/hot_experts.json \
    --model-id org/model
```

Given the measurements above, this is the phase that matters. Nothing is deleted:
every expert goes into an NVMe store, and `hot_experts.json` tells the cache which
ones deserve VRAM. The store holds *all* `num_experts` records — the cache, not
the store, decides residency.

Building it offline is new. Until now a store could only be produced by booting
vLLM with `VLLM_MOE_STREAM_LOAD=1` and intercepting the weight loader, which needs
a GPU and a working engine. Here it is a safetensors-to-records pass: OLMoE's 16
layers, 6.02 GiB fp8, in **59 seconds** on any host.

`hot_experts.json` fills a seam that has been an empty dict since the prototype —
`EWMAPolicy.prior`, documented as "a table loaded at boot that biases residency
toward experts that were hot in a previous run. Empty until seeded." Priors are
share-proportional rather than rank-based, and deliberately small (worth a few
cache hits): a boot-time hint that traffic cannot overrule is a pin, not a hint,
and a test asserts that 50 hits on a cold expert overtake it.

### The result that decides the project

Serving OLMoE from the offline-built store with **24 of 64 experts resident**:

| approach | resident | output |
|---|---|---|
| base | 64/64 | reference |
| **disk tier** | **24/64** | **same tokens as base** |
| hard prune | 40/64 | degenerates into repetition |

Keeping 37.5% resident costs nothing measurable, while *deleting* 37.5% destroys
the model. Same fraction of experts, opposite outcomes — because the disk tier
still has every expert when the router asks for one.

(Same tokens under greedy decoding, not a bit-exactness claim: the fp8 store does
perturb logits, it just did not move any argmax here.)

## Status

Faz 0–4 complete, verified on GB10: 176 tests locally, 211 with CUDA and vLLM.

First real measurement, OLMoE-1B-7B (64 experts, top-8, 16 MoE layers), 3 prompts
/ 169 tokens: `mean experts/tok` came back exactly 8.000, all 16 layers captured,
no dropped rows. 54–64 of 64 experts were touched *per layer*, and the hottest
quarter of experts carried 50–66% of load against 25% for a uniform
distribution. So the concentration is real (~2–2.6x), but at 169 tokens this is a
pipeline check, not a pruning basis — it also reproduces the earlier finding that
a handful of prompts already reaches nearly the whole expert set.
