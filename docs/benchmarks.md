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

### How far the two non-exact configurations actually move

A hash says *that* two arms differ, never by how much — and the exceptions above
are not exotic: on a card too small to hold `top_k` experts the split is
**mandatory**, so the laptop runs one of them by default. `surgeon fidelity`
prices them. OLMoE, GB10, stock vLLM 0.27.1, gsm8k held-out (48 prompts, 2785
scored positions), every arm at the same `gpu_memory_utilization` 0.42:

| arm | top-1 agreement | KL(untiered ‖ arm) | RMS Δp | max Δp |
|---|---|---|---|---|
| `expert_cache_size: 48` (control) | **100.000%** | 0.00000 | 0.000% | 0.000% |
| `cap 4 + split: "expert"` | 98.205% | ≥0.00095 | 1.090% | +5.885% |
| `fp8_store` | 97.379% | ≥0.00095 | 1.961% | +19.880% |

The first row is a **positive control, not a result**: cap 48 is the
configuration whose token hashes have matched untiered nine times out of nine,
so a non-zero reading there would have meant the instrument was wrong.

So the expert split — which a card too small to hold `top_k` experts makes
mandatory — moves the sampled token 1.8% of the time, and `fp8_store` 2.6%. For
scale: 4-bit quantisation of the same model, measured with the same class of
statistic, moves it **8.0%**.

Two honest limits. The KL figures are lower bounds: they are computed over the
reference's top-K support, and 3.0% of reference tokens fell outside the arm's
own top-K and were scored at the most favourable value they could have had.
Raising K does not fix that — **32 → 128 moved coverage 89.29% → 94.50% and the
substitution rate only 3.22% → 2.97%**, so the tail is structural rather than a
capture setting. Top-1 agreement carries no such assumption: an argmax is inside
its own top-K by construction.

And none of this is task accuracy. That 8%-of-tokens change was measured
**invisible** to gsm8k@200 (0.095 vs 0.105, ±0.021) and to HellaSwag@400 (62.25%
vs 61.75%, overlapping intervals) — a change that moves the sampled token one
time in twelve, which neither benchmark could see. The axis this table measures
sits between perplexity and task accuracy precisely because both of those have
been measured here to miss things.

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

### Thread count is not the lever, and here is why

The host GEMM is a *weight read* — one OLMoE expert is 12.58 MB against ~25
MFLOP — so it scales with memory bandwidth, and bandwidth scales with threads.
Standalone (`bench/p1_cpu_threads.py`, i7-12700H, 20 CPUs, T=2, bf16):

| threads | 1 | 4 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|
| µs/expert | 1226.4 | 451.8 | 360.6 | 272.0 | 244.2 | **229.8** |
| GB/s | 10.26 | 27.85 | 34.89 | 46.25 | 51.53 | **54.75** |

**5.34× is available in the kernel** and the gate's recorded 368.8 µs sits at the
8-thread point of that curve. Raising it in the engine buys nothing: same served
benchmark, `OMP_NUM_THREADS` unset / 20 / 12 gives **12.27 / 12.40 / 12.32 tok/s**
(TPOT 278.7 / 271.0 / 265.2) — inside this box's drift, and the *slower* 12-thread
arm posted the best TPOT of the three, which is what noise looks like.

**Read that arm for exactly what it tested.** With `OMP_NUM_THREADS` unset torch
already takes 14 threads on this host, so unset → 20 is a 1.09× kernel change,
not a 5.34× one. The null result therefore says the top of the curve is flat; it
does **not** establish what share of the step the host GEMM is. The arm that
would establish it is `OMP_NUM_THREADS=1` — a real 5.34× — and it is listed as
open below rather than inferred here.

What is already established: the best available tuning of `cpu_expert_threads`
moves served TPOT by a few percent, so it is a small knob rather than a lever.
The measurement below says why — the host GEMM shares the step with a much larger
term.

### Concurrency is the gap, and a served TTFT is not a prefill number

Same laptop arm, same server, two client settings:

| | concurrency 4 | concurrency 1 |
|---|---|---|
| median TTFT | 7305 ms | **881 ms** |
| median TPOT | 274.8 ms | **96.4 ms** |

**Do not read a served median TTFT as prefill cost.** At concurrency 4 with
128-token outputs, a request waits behind other requests' decode — about 88% of
that 7305 ms is queueing. The prefill this configuration actually performs takes
881 ms.

Per-stream decode is 96.4 ms/token alone and 274.8 ms at four in flight — 2.85×
worse for the same work, since concurrency multiplies the per-layer expert union
and this card holds 4 experts of 64.

Raising the host tier confirms the mechanism, because it helps exactly where the
union is large and nowhere else:

| `ram_cache` | c4 tok/s | c4 TPOT | c1 TPOT | host RSS |
|---|---|---|---|---|
| 24 | 12.34 | 274.8 ms | 96.4 ms | 7878 MB |
| 36 | **14.26** | **208.6 ms** (1.32×) | 91.0 ms (1.06×) | 10183 MB |
| 48 | — | refused | — | — |

The 48 arm did not OOM: the pinned-pool rule refused it, *"only 3.8 GiB of 14.8
GiB is available and at least 3.7 GiB must stay reclaimable, or the host
livelocks"*. That guard exists because this box once went into swap and lost its
session; this is the first time it has fired on a real attempt, and the box
stayed up.

### The configuration matrix, in one sitting

Every figure above came from a different sitting, and absolute throughput on this
box drifts between them. Nineteen configurations run back to back are mutually
comparable; median ms per output token, single stream, lower is better:

| store · `ram_cache` | `cpu_experts` | O_DIRECT | page cache | Δ |
|---|---|---|---|---|
| fp8 · 8 | off | 134.59 | **108.17** | 1.24× |
| fp8 · 24 | off | 94.94 | 93.12 | 1.02× |
| fp8 · 64 | off | 89.87 | 89.63 | 1.00× |
| bf16 · 8 | on | 238.79 | **160.04** | 1.49× |
| bf16 · 24 | on | **109.86** | 132.64 | 0.83× |
| bf16 · 36 | on | **75.28** | 87.79 | 0.86× |
| bf16 · 8 | off | 277.23 | **182.94** | 1.52× |
| bf16 · 24 | off | 158.40 | 152.64 | 1.04× |
| bf16 · 36 | off | 146.28 | 146.26 | 1.00× |

Three things fall out of it.

**The read path's sign flips with `ram_cache`.** The page cache wins at 8
(1.24–1.52×), is neutral at 24, and loses at 36 (0.83–0.86×), where it is a
redundant second copy of records the pinned pool already holds. Choose it by the
pool's absolute size, not by whether the pool covers the store: at `ram_cache` 36
the pool is 7.25 GB against a 12.9 GB store and buffered is still the wrong
choice. The crossover is near 24 on this host.

**Co-execution is worth more the more host RAM it has** — 1.16× at `ram_cache` 8,
1.44× at 24, 1.94× at 36. What it buys is bytes never sent over PCIe, and the
only bytes it can avoid sending are the ones already in RAM.

**The fastest cell is not the fully-resident fp8 store.** bf16 · 36 · co-exec ·
O_DIRECT reaches **75.28 ms/token** against fp8 · 64's 89.87 — 1.19× better at
comparable pinned memory (7.25 vs 6.46 GB). The same configuration measured
91.0 ms in a separate sitting; only inside one sitting does its position show.

**How much of this table is readable.** The first cell was repeated as the last.
Single-stream latency returned within **0.3%** (134.59 → 134.98); the
concurrency-4 figure returned **45% higher** (466.73 → 678.61) doing identical
work by the read counters. So the single-stream column carries the findings and
**no concurrency comparison below 1.45× is real** — which is why this section
quotes none.

### What a slot fill costs

`t_disk_read` times the `preadv` that stages a record into the pinned pool.
Reading the same configuration twice separates the copy from the read, because
`VLLM_MOE_DISK_BUFFERED` with a warm page cache leaves no device inside it. fp8
store (6.31 MB per record), `ram_cache` 8, identical 10 534 fills:

| read path | µs per fill | effective |
|---|---|---|
| buffered, page cache warm | **889.3** | 7.1 GB/s — the copy alone |
| O_DIRECT | **2200.2** | 2.9 GB/s — device-bound |

So staging a record costs ~889 µs of pure memcpy, about 40% of a device-bound
fill. Residency is what removes it wholesale: at `ram_cache` 64 the same workload
performs **103 fills instead of 10 534** (99.2% RAM-tier hit rate).

Scaled to the fastest bf16 cell — 925 fills over 128 tokens, a 12.58 MB record —
the copy is ~12.8 ms of an 83.7 ms step, about **15%**. That is the ceiling on any
change that computes host-served experts in place rather than staging them, and
it is why no such change is shipped: 1.18× on one configuration, on a host that
cannot cache a 12.9 GB store beside a 4.8 GB engine anyway.

**Do not read `fill_s` as a fraction of the step.** It is summed across the
reader threads, so with `DISK_IO_THREADS` above 1 overlapping fills can total more
than wall time — one arm reported a "share" of 1.08. The per-fill figure is the
trustworthy one.

### What the step is actually made of

Counters rather than inference (`bench/p1e_share.py`, provider `cpu_execs`,
`t_cpu_gemm`, `n_disk_bytes`; needs `VLLM_ENABLE_V1_MULTIPROCESSING=0`, see
below). bf16 store, cache 4, `ram_cache` 24, `cpu_experts`, 128 tokens:

| | 1 sequence | 4 sequences |
|---|---|---|
| host GEMM share of wall | **29.6%** | **38.6%** |
| per expert | 335.3 µs | 525.5 µs |
| **read from disk per token** | **346.78 MiB** | 104.91 MiB |
| RAM tier hit rate | 76.1% | 74.9% |

**43 GiB of NVMe for 128 tokens.** The host GEMM is a large share but it runs
underneath that, which is why making it faster changed nothing.

### Fitting the whole store in RAM — what fp8 is actually for

An fp8 record is 6.31 MB against bf16's 12.58, so `ram_cache` 64 costs 6.46 GB
and the **entire store becomes resident** — the configuration the pinned-pool
rule refuses for bf16. Same client and workload as the rows above:

| configuration | c1 TTFT | c1 TPOT | c4 TPOT | c4 tok/s | disk/token |
|---|---|---|---|---|---|
| bf16 + `cpu_experts`, ram 24 | 881 ms | 96.4 ms | 274.8 ms | 12.34 | 346.78 MiB |
| bf16 + `cpu_experts`, ram 36 | 655 ms | 91.0 ms | **208.6 ms** | **14.26** | — |
| **fp8, ram 64 (fully resident)** | **608 ms** | **90.0 ms** | 296.2 ms | 12.78 | **4.84 MiB** |
| fp8, ram 12 | 814 ms | 110.7 ms | 416.8 ms | 8.75 | — |

Disk traffic falls **72×** and the RAM tier hits 99.2%. Single-stream, that arm
is the best of everything measured on this box. **At four sequences it loses to
co-execution anyway**, and the reason is mechanical: `cpu_experts` and
`fp8_store` are mutually exclusive today, so the fully-resident arm has no host
compute and every miss crosses PCIe at 1139.7 µs a record — which is precisely
what concurrency multiplies.

So the two mechanisms win in different regimes, and the honest recommendation for
a card this size is per-workload rather than universal: **fully-resident fp8 for
single-stream latency, bf16 + co-execution for concurrent serving.**

### Who should hold the warm tier — us, or the page cache

The store reads with `O_DIRECT`, which keeps the pinned RAM tier the only RAM
this path uses. `VLLM_MOE_DISK_BUFFERED=1` reads through the page cache instead.
Same client and workload, physical reads from `/proc/PID/io read_bytes` so
cache hits are excluded by construction:

| pinned pool | read path | c1 TPOT | c1 read | c4 TPOT | c4 read | c4 tok/s |
|---|---|---|---|---|---|---|
| 6.46 GB (`ram_cache` 64) | O_DIRECT | **89.5 ms** | 0 | 306.3 ms | 0 | 12.39 |
| 6.46 GB | buffered | 89.9 ms | 0 | 296.9 ms | 0 | 12.54 |
| 0.81 GB (`ram_cache` 8) | **buffered** | 107.5 ms | **0** | 387.8 ms | **5.4 GiB** | 9.96 |
| 0.81 GB | O_DIRECT | 126.1 ms | 21.4 GiB | 534.7 ms | **918 GiB** | 6.20 |

Two readings, and the second is the useful one. **When the pinned pool holds the
store, pinning wins** — 89.5 ms against 107.5 for the page cache, and the flag
itself is free (rows 1 and 2 are the same arm either way). **When it does not,
the page cache is worth 1.17× at one sequence and 1.61× on throughput at four**,
and it removes 99.4% of the physical reads — one benchmark of the O_DIRECT arm
pulled **918 GiB** off the device.

That second case is not hypothetical: it is what a card too small for the pool,
a model too large for the host, or the pinned-pool refusal all land on, and it is
where the current default behaves worst. The cgroup worry did not appear —
capped at 12G and uncapped measure the same (107.5 vs 105.8 ms) because a 6.1 GB
store fits beside a ~4 GB engine.

It cannot conjure residency that does not exist. With the **bf16** store (12.9 GB
on a 14.8 GB host) the same switch gives its largest relative win — 1.54× at c1,
2.19× at c4 — and still only reaches 140.9 ms, because nothing can cache a store
that does not fit.

`DISK_IO_THREADS` was swept in the same sitting and is flat (1/2/4 → 94.1 / 89.4
/ 97.2 ms at c1): the device saturates at queue depth 2, measured directly at
6.2 GB/s against a page-cache hit's 21–40 GB/s (`bench/f0_cache_probe.py`).

The combination that would take both is refused on purpose — *"the CPU path would
have to dequantize every forward, which is not implemented"*. Worth noting what
it would cost, because it is not obvious: a naive host dequant reads 6.31 MB and
writes 12.58, roughly **doubling** the ~335 µs host GEMM, while one fused into
the GEMM would read **half** the bytes bf16 does and could beat it. That is a
kernel decision, not a flag.

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
