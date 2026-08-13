# Benchmarks

Every number this package claims, in one place, with the machine and the
method next to it. The per-axis *verdicts* live in
[DECISIONS.md](../DECISIONS.md); this page is the measurements themselves.

**Method, applied to every table below.** One arm per process — a vLLM engine
does not release its device memory when the object goes out of scope, so a
second arm in the same process starts against the first one's allocation.
Decode is timed as output tokens over wall time on a short-prompt /
long-generation workload (prompt share ~1%) after a warm-up pass, never as
`full − prefill-only`. Repeats are medians with the spread reported; a single
boot is not a measurement. Perplexity is held-out, and where token identity
matters it is a sha256 over the generated ids, not a perplexity printed to
four decimals.

## The machines

| name | device | host | role |
|---|---|---|---|
| **GB10** | DGX Spark, 121 GB unified LPDDR5X | 20× Cortex-X925 | the big box: throughput, capacity, graphs |
| **laptop** | RTX 3050 Ti, 3.68 GiB usable | i7-12700H, 14.8 GB | the small box: feasibility, run-at-all |

---

## Capacity: the one knob that matters

OLMoE-1B-7B bf16, `ram_cache 64`, eager, 8 prompts × 256 tokens, 3 repeats.
Measured twice — in-tree (2026-08-11) and again through the out-of-tree
runtime (2026-08-12) — because "the OOT package reproduces every in-tree
capability" is a claim that needed a throughput number, not just a test pass.

| `expert_cache_size` (of 64) | in-tree decode | **OOT decode** | OOT load |
|---|---|---|---|
| 24 | 55.3 tok/s | **54.3 tok/s** | 33.0 s |
| 32 | 75.5 | **74.6** | 19.0 s |
| 40 | 107.9 | **109.3** | 16.8 s |
| 48 | 143.0 | **145.1** | 16.9 s |
| *untiered* | *218.2* | ***218.5*** | *128.9 s* |

**2.59× in-tree, 2.67× out-of-tree, across one integer.** The two
implementations agree within 1–2% at every capacity. Perplexity was identical
on all five OOT arms including untiered, so the lever is free on the accuracy
axis. Load time is ~7.7× shorter under the tier (streaming); the first tiered
arm's 33 s is a cold page cache, and later arms settle at 17–19 s.

The cause is oversubscription, not bandwidth: at batch 8 the per-layer expert
union measures 35.28 mean / 46 max, so 24 slots are 47% oversubscribed and
every layer splits into 2–3 chunks, each paying a blocking D2H, its own
mapping upload and a separate GEMM. At capacity ≥ 46 the split disappears.

Sizing rules derived from this: [docs/sizing.md](sizing.md).

## CPU expert co-execution

Cold experts computed on the host instead of fetched over the link. The
decisive quantity is `BW_cpu_gemm / BW_h2d` — whether the CPU's DRAM reads
and the GPU's inbound copies are separate bandwidth pools.

### The gate, before any code was written

Single layer, no engine, real decode shape, 26 streaming weight sets, median
of 15 reps. Go/no-go was fixed in advance at ≥1.4×.

| | GB10 (L5) | laptop (L6) |
|---|---|---|
| GPU-only | 6.07 ms/layer | 12.72 ms/layer |
| best co-exec | 8.44 ms (f=0.5) | **3.45 ms (f=1.0)** |
| **ratio vs the ≥1.4 gate** | **0.719 — NO-GO** | **3.7× — GO** |
| measured contention *s* | 2.385 | **1.037** |
| H2D per record | 217 µs | 1139.7 µs |
| CPU per expert (padded) | ~300 µs | 368.8 µs |
| `BW_cpu_gemm / BW_h2d` | 0.78 | **3.09** |

Unified memory has one pool, so the host half and the GPU half fight for it
and the co-exec arm loses. A discrete card has two, and the laptop's PCIe H2D
is so much slower than its CPU GEMM that the winning policy is **f = 1.0**:
never fetch a cold expert, compute all of them.

Scripts: `bench/l5_cpu_coexec_gate.py`, `bench/l6_cpu_coexec_gate.py`.

### End to end, in the engine

Laptop, OLMoE bf16 store, B=2, `expert_cache_size 4`, `split=expert`, eager,
greedy, 2 prompts × 128 tokens, 3 repeats. The flag is the only difference
between arms.

| arm | `ram_cache` | decode | ratio | GPU misses | host expert forwards |
|---|---|---|---|---|---|
| off | 16 | 2.78 tok/s | — | 95,405 | 0 |
| **on** | 16 | **3.58 tok/s** | **1.29×** | 11,013 | 73,925 |
| off | 24 | 4.66 tok/s | — | 95,405 | 0 |
| **on** | 24 | **7.37 tok/s** | **1.58×** | 1,414 | 85,165 |

**Compare within a pair, not across them.** Each pair was run back-to-back
with the flag as the only difference; absolute tok/s drifts between sessions
because the ram16 configuration is disk-bound and therefore sensitive to page
cache and thermal state (its off arm measured 3.72 tok/s in an earlier
window and 2.78 in this one, with byte-identical counters — same work,
different machine). The ratios are what the flag controls.

7.37 tok/s is the fastest this laptop has served this model; the fp8 recipe
topped out at 6.10 — an indirect comparison, since that arm used a different
store type. In-engine host cost measured **356–377 µs/expert against the
gate's solo 368.8**, so contention inside the real engine is ≈1.0 — the
gate's central prediction, confirmed where it counts.

These are post-review numbers: the whole table was re-measured after the
implementation's hostile review, because several of its fixes touch the hot
path (the per-forward counts pass, the in-place join, and the thread knob
moving to boot time where torch will actually honour it). The pre-review
build measured 1.37× and 1.52× on the same two configurations.

The residual gap to the gate's 3.7× is the disk tier: the RAM pool (16–24
rows) is far below the 64-expert working set, so both arms still read cold
rows off NVMe. The ram16 → ram24 trend is that gap closing.

**Read the miss column correctly.** The collapse from 95,405 to 1,414 is
*masking, not learning*: every CPU-served expert is hidden from the planners,
so the GPU cache only ever sees experts it already holds — it stops inserting
and stops evicting, and its resident set freezes. While co-execution covers
the whole miss set, the eviction policy is unreachable and a low GPU hit rate
means the opposite of what it means elsewhere in these docs.

Not bit-exact by construction (host reduction order, one fp32 join). The
first 12 greedy tokens matched the baseline on both prompts, which is an
observation, not a guarantee.

## Surgery: what deletion costs and buys

OLMoE-1B-7B, GB10, held-out gsm8k[400:500], baseline perplexity 9.703.

| configuration | load | decode | perplexity |
|---|---|---|---|
| baseline, 64 experts resident | 98.6 s | 689.4 tok/s | 9.703 |
| disk tier, 24/64 resident | 45.2 s | 265.1 | 9.734 (1.003×) |
| pruned to 40, no tier | 83.4 s | 695.1 | 12.115 (1.249×) |
| **pruned to 40 + tier, 24/40** | **44.5 s** | **350.8** | 12.145 (1.252×) |

Pruned+tier decodes **1.32× faster than tier alone** at the same capacity: a
24-slot cache covers more of a 40-expert candidate set than a 64-expert one.
This is why surgery exists to *support* the tier rather than compete with it.

**Perplexity is a proxy, and here is where it misleads.** The same pruned-40
checkpoint, `lm-eval` at 500 items/task with paired exact McNemar:
arc_challenge acc_norm 0.468 → **0.352** (p=5e-08), hellaswag 0.662 → 0.618
(p=0.014), gsm8k 0.100 → 0.058 (p=0.006). The amplitude fix removes ~60% of
the perplexity damage and recovers **no** task accuracy (all three p ≥ 0.5).

### When pruning does pay: a narrow domain

Real system logs (Linux+SSH+Apache triage), held-out log perplexity,
baseline 27.89. Full experiment in [DECISIONS.md](../DECISIONS.md).

| pruned to | applied | ratio |
|---|---|---|
| 56 of 64 (the dead tail) | 30.64 | 1.10× |
| 40 of 64, **without amplitude** | 41.00 | **1.47× — the trap** |
| 40 of 64, **with amplitude 0.85** | 32.53 | **1.17×** |

And the gain side, tier against tier at full coverage:

| configuration | GPU slot bytes | decode |
|---|---|---|
| **pruned-40 + tier, 40/40 resident** | **7.5 GiB** | **256.2 tok/s** |
| unpruned + tier, 64/64 resident | 12.0 GiB | 205.1 |
| unpruned, untiered | 12.0 GiB | 218.2 |

Pruning buys compute, not just memory — a 40-expert kernel is smaller than a
64-expert one, so the pruned model beats even the untiered baseline.

## Feasibility: the boot floor

`surgeon vram-floor` bisects the real minimum by booting, so this is measured
rather than accounted.

| configuration | boot floor | fails below |
|---|---|---|
| OLMoE untiered | 14.29 GiB | 0.115 |
| OLMoE tier, streamed fp8 | **11.55 GiB** | 0.092 |

The tier boots **19% below** untiered. Without streaming load it was *above*
it (23.40 GiB) — the streaming path is what turned feasibility from a loss
into a win.

On the laptop the same question has a blunter answer: the tier serves the
12.9 GiB model at **2.69 GiB peak / 7.7 tok/s**, and the untiered arm was
never booted — an attempt drove the 14 GB host into swap and lost the ssh
session. That *is* the feasibility result.

## What the fp8 store does and does not buy

| axis | effect |
|---|---|
| VRAM | **none** — the provider dequantizes into model-dtype slots |
| disk / host RAM / transfer | halved |
| decode (bf16 model, cap 48) | 128.6 vs 143.0 tok/s — **1.11× slower** |
| accuracy | not bit-exact (ppl 11.7393 vs 11.6253) |

`fp8_store` is a **space** mechanism. Enable it when the store or the host
RAM tier does not otherwise fit, never for speed.

## Reproducing

Scripts live in the workspace's `bench/` directory (outside the package, so
they are free to modify): `cpu_coexec_ab_arm.py` and the two gate scripts for
co-execution, plus the capacity-sweep and boot-floor harnesses. Raw results
for the co-execution work are checked in as `bench/cpu_coexec_ab_results.json`
and `bench/l6_result.json`.
