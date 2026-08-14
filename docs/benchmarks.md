# Benchmarks

Every number this package claims, with the machine and the method beside it.

**Method, applied throughout.** One arm per process — a vLLM engine does not
release its device memory when the object goes out of scope, so a second arm in
the same process starts against the first one's allocation. Decode is timed as
output tokens over wall time on a short-prompt / long-generation workload
(prompt share ~1%) after a warm-up pass, never as `full − prefill-only`.
Where a table states repeats, the figure is a median and the spread is beside
it; **a single boot is not a measurement** and is labelled as such where it
happens. Where token identity matters it is a sha256 over the generated ids —
and only within one configuration, since a different `gpu_memory_utilization`
sizes the KV cache differently and changes the reduction order.

**On spread.** A within-process σ is not a stability measurement. The same
configuration (cache 24, `ram_cache` 64, eager) has been measured at σ 0.044,
σ 0.69 and σ 0.055 in three separate sittings whose medians agree to ±0.9% —
the σ *estimate* swings 16× while the median does not move. So anything
resting on run-to-run variance here needs n ≥ 5 separate processes; the
medians, which move 2.59× and reproduce across every sitting, do not.

**Machines.**

| name | device | host |
|---|---|---|
| **GB10** | DGX Spark, 121 GB unified LPDDR5X | 20× Cortex-X925 |
| **laptop** | RTX 3050 Ti, 3.68 GiB usable | i7-12700H, 14.8 GB |

Model unless stated: OLMoE-1B-7B-0924 (64 experts, top-8, 16 MoE layers,
12.89 GiB bf16).

---

## Cache size

GB10, bf16 store, `ram_cache 64`, eager, 8 prompts × 256 tokens, 3 repeats.
Measured twice — in-tree (2026-08-11) and again through the out-of-tree runtime
(2026-08-12) — because "the package reproduces every in-tree capability" needed
a throughput number, not just a passing test.

| `expert_cache_size` (of 64) | in-tree decode | **OOT decode** | OOT load |
|---|---|---|---|
| 24 | 55.3 tok/s | **54.3 tok/s** | 33.0 s |
| 32 | 75.5 | **74.6** | 19.0 s |
| 40 | 107.9 | **109.3** | 16.8 s |
| **48** | 143.0 | **145.1** | 16.9 s |
| *untiered* | *218.2* | ***218.5*** | *128.9 s* |

**2.59× in-tree, 2.67× out-of-tree, across one integer**, and the two
implementations agree within 1–2% at every capacity. Perplexity was identical
on every OOT row including untiered, so the lever is free on the accuracy axis.

Load time settles at **17–19 s against 128.9 s untiered**, because expert
weights stream into the store instead of onto the device. The first tiered
arm's 33.0 s is a cold page cache, so the load column is not a "bigger cache
loads faster" trend.

The cause is oversubscription. At batch 8 the per-layer expert union measures
35.28 mean / 46 max, so 24 slots are 47% oversubscribed and every layer splits
into 2–3 chunks, each paying a blocking device-to-host sync, its own mapping
upload and a separate GEMM. At capacity ≥ 46 the split disappears.

## Host-RAM tier

GB10, cache 24, eager.

| `ram_cache` | decode |
|---|---|
| 48 | 33.6 tok/s |
| 64 | **55.8 tok/s** |

**1.66×** from removing the host-RAM spill. Both arms are 3-repeat medians
within one process; their σ (0.315 and 0.044) is *not* evidence that a spilling
tier is erratic — see the note on spread above, which measured the same
configuration at three different σ across sittings.

## Numerical transparency

Seed-pinned greedy generation, **eager**, sha256 over token ids, 3 repeats each
of {untiered, tier cache 48 / ram 64, zero-copy}, one arm per process.

**All nine runs produced the identical hash.** A correctly sized tier is
numerically transparent in eager mode, zero-copy included.

Three exceptions, all measured:

| | why |
|---|---|
| `split: "expert"` | a layer's experts are computed in groups; the reduction order changes |
| `fp8_store` | the record is quantised (11.7393 against 11.6253 perplexity) |
| **CUDA graphs** | graph arms diverge from eager and from each other |

The graph exception is not the tier's doing: stock vLLM is not bit-identical
between graphs and eager either, and the tier's MoE op runs eager between the
captured pieces. It is listed because the claim above is an *eager* claim.

## Feasibility

`surgeon vram-floor` bisects the real minimum by booting and generating.

| configuration | boot floor | fails below |
|---|---|---|
| untiered | 14.29 GiB | 0.115 |
| tier, streamed fp8 | **11.55 GiB** | 0.092 |

The tier boots **19% below** untiered. Without streaming load it was *above* it
(23.40 GiB): on unified memory, page-locked host memory is charged against
`gpu_memory_utilization`.

On the laptop the answer is blunter: the tier serves the 12.9 GiB model at
**2.69 GiB peak, 7.7 tok/s, 6.6 s load**. The untiered arm was never booted —
the attempt drove the 14.8 GB host into swap and lost the session.

Note that peak-VRAM readings on GB10 report the preallocated pool, not what a
model needs; feasibility numbers come from `surgeon budget` and `vram-floor`.

## CPU co-execution

Cold experts computed on the host instead of fetched over the link.

### Where it works, and where it does not

Single layer, no engine, real decode shape, 26 streaming weight sets, median of
15 repeats.

| | GB10 (unified) | laptop (discrete) |
|---|---|---|
| GPU-only | 6.07 ms/layer | 12.72 ms/layer |
| best co-exec | 8.44 ms | **3.45 ms** |
| ratio | **0.719× — a loss** | **3.7× — a win** |
| measured contention | 2.385 | 1.037 |
| host→device per record | 217 µs | 1139.7 µs |
| host GEMM per expert | ~300 µs | 368.8 µs |
| `BW_cpu_gemm / BW_h2d` | 0.78 | **3.09** |

Unified memory has one pool, so the host half and the device half compete for
it. A discrete card has two, and there the PCIe transfer is slow enough that
computing an expert beats moving it.

### In the engine

Laptop, bf16 store, batch 2, cache 4, expert split, eager, greedy, 2 prompts ×
128 tokens, 3 repeats. The flag is the only difference between arms.

| `ram_cache` | off | on | ratio |
|---|---|---|---|
| 16 | 2.78 tok/s | 3.58 tok/s | **1.29×** |
| 24 | 4.66 tok/s | **7.37 tok/s** | **1.58×** |

Compare within a pair, not across them: each pair ran back-to-back, while
absolute throughput drifts between sessions on this disk-bound configuration.

In-engine host cost measured 356–377 µs/expert against the isolated 368.8, so
contention inside the real engine is ≈1.0.

Not bit-exact. The GPU cache also stops adapting while co-execution covers the
whole miss set — every host-served expert is hidden from the planner, so the
resident set freezes.

## Pruning

GB10, held-out gsm8k, baseline perplexity 9.703.

| configuration | load | decode | perplexity |
|---|---|---|---|
| baseline, 64 experts resident | 98.6 s | 689.4 tok/s | 9.703 |
| tier, 24 of 64 resident | 45.2 s | 265.1 tok/s | 9.734 (1.003×) |
| pruned to 40, no tier | 83.4 s | 695.1 tok/s | 12.115 (1.249×) |
| **pruned to 40 + tier** | **44.5 s** | **350.8 tok/s** | 12.145 (1.252×) |

Pruned+tier decodes **1.32× faster than tier alone** at the same cache size: a
24-slot cache covers more of a 40-expert candidate set than a 64-expert one.

Each row here is a **single run** — no repeats, no variance, and boot-floor
probes vary ~15% in this environment. Read 1.32× as a ratio with a few-percent
error bar, not to three figures.

**Perplexity does not certify task accuracy.** The same pruned checkpoint,
`lm-eval` at 500 items/task with paired exact McNemar:

| task (metric) | baseline | pruned-40 | Δ (p) | + amplitude 0.85 |
|---|---|---|---|---|
| arc_challenge (acc_norm) | 0.468 | 0.352 | **−0.116** (4.8e‑08) | 0.350 |
| hellaswag (acc_norm) | 0.662 | 0.618 | −0.044 (0.014) | 0.622 |
| gsm8k (exact, strict) | 0.100 | 0.058 | −0.042 (0.0055) | 0.068 |

The amplitude fix removes ~60% of the *perplexity* damage and recovers **no**
task accuracy (p = 1.0, 0.83, 0.53).

### On a narrow domain

Real system logs (Linux + SSH + Apache triage prompts, held-out lines from the
same files), baseline 27.89.

| pruned to | applied | ratio |
|---|---|---|
| 56 of 64 | 30.64 | 1.10× |
| 40 of 64, no amplitude | 41.00 | **1.47×** |
| 40 of 64, amplitude 0.85 | 32.53 | **1.17×** |

Deletion inflates surviving gates by `1/(1−P_D)`; the plan predicts the
correction from its own deleted routing mass (0.861 predicted against 0.850
measured, 0.2% apart).

Gains, tier against tier at full coverage:

| configuration | GPU slot bytes | decode |
|---|---|---|
| **pruned-40 + tier** | **7.5 GiB** | **256.2 tok/s** |
| unpruned + tier | 12.0 GiB | 205.1 tok/s |
| unpruned, untiered | 12.0 GiB | 218.2 tok/s |

The pruned model beats even the untiered baseline: a 40-expert kernel is
smaller than a 64-expert one, so pruning buys compute as well as memory.

## Checkpoint selection

Scoring candidate checkpoints on a domain corpus, and what the winner serves at.
Bits per byte rather than perplexity, because per-token perplexity depends on
the tokenizer.

Speed and feasibility on the same GB10 harness as the capacity sweep (8 prompts
× 256 tokens, 3 repeats, one arm per process, untiered and eager on both sides);
OLMoE re-measured here at 218.33 tok/s against the 218.2 and 218.5 already on
record, so this shares the baseline every other number on this page uses. Task
accuracy is `lm-eval`, 500 items/task, paired exact McNemar.

| | OLMoE-1B-7B | granite-3.0-3b-a800m | |
|---|---|---|---|
| log-domain bits/byte | 1.4073 | **1.3395** | 4.8% better |
| decode | 218.33 tok/s | **329.55 tok/s** | 1.51× |
| load, this run | 115.8 s | **7.5 s** | ~15× |
| resident weights | 12.89 GiB | **6.29 GiB** | 2.0× smaller |
| bit-exact tier floor | 2.39 GiB | **1.79 GiB** | 1.3× smaller |
| checkpoint on disk | 12.89 GiB | 12.57 GiB | the same |
| arc_challenge acc_norm | 0.468 | 0.454 | −0.014, not significant (p=0.51) |
| arc_challenge raw acc | — | — | **−0.068, significant (p=0.0008)** |
| hellaswag acc_norm | 0.662 | 0.618 | **−0.044, significant (p=0.0032)** |
| gsm8k strict | 0.088 | **0.354** | +0.266 (p<1e‑4) |

**The two arc metrics disagree**, and the recorded protocol used `acc_norm` —
so "granite costs no arc_challenge accuracy" is an acc_norm statement, not a
general one. Only the delta was recorded for raw `acc`, which is why that row
carries no absolutes. Both directions are printed here rather than the
favourable one, because quoting the favourable metric is how a headline
outlives its own caveat. hellaswag is a real regression on this checkpoint: the
smaller model dominates on the domain and on gsm8k, not everywhere.

The gsm8k baselines on this page are two separate sessions and do not
reconcile: 0.100 in the pruning table above, 0.088 here. Compare each arm
against the baseline printed beside it, never across the two.

The checkpoint is not smaller on disk: granite ships fp32, so 3B parameters
occupy as much as OLMoE's 7B at bf16. What halves is *resident* weights,
because vLLM downcasts at load.

This compares two checkpoints, not two strategies, and it does not displace the
tier — granite's 6.29 GiB resident is still well above a 3.68 GiB card.

## fp8 store

| axis | effect |
|---|---|
| VRAM | **none** — the provider dequantizes into model-dtype slots |
| disk / host RAM / transfer | halved |
| decode (bf16 model, cache 48, in-tree session) | 128.6 vs 143.0 tok/s — **1.11× slower** |
| accuracy | not bit-exact (11.7393 vs 11.6253 perplexity) |

A **space** mechanism. Enable it when the store or the host RAM tier does not
otherwise fit, never for speed.

## CUDA graphs

Piecewise capture with the MoE op carved out. Worth **+3.8%** on the untiered
baseline and **~0** for the tier, by design: the split keeps the MoE op eager.
Capture costs ~22 s of load time.

Its value is compatibility — `--enforce-eager` is no longer required — not
throughput.
