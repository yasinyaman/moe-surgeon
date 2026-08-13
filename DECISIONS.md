# Decision register

Every method here is judged on **three axes**, never one:

| axis | what it asks | how it is measured |
|---|---|---|
| **feasibility** | does it run, and on how small a device | `surgeon budget` (validated arithmetic) + measured load time |
| **speed** | tokens per second | `compat/bench.py`, short prompts / long generations |
| **accuracy** | held-out perplexity | `compat/ablation.py`, the same metric the gate uses |

> **Perplexity is a proxy, and a downstream-task run (2026-08-10) shows where it
> misleads.** rank-1 pruned-40 vs baseline over `lm-eval` (500 items/task, paired
> exact McNemar): arc_challenge acc_norm 0.468 → **0.352** (p=5e‑08), hellaswag
> 0.662 → 0.618 (p=0.014), gsm8k 0.100 → 0.058 (p=0.006). So the gate's 1.3×
> *perplexity* ceiling does not certify task quality — deletion at this ratio is a
> statistically-significant task-accuracy loss. And the amplitude fix, which removes
> ~60% of the perplexity damage, recovers **no** task accuracy (all three McNemar
> p ≥ 0.5). Read every "accuracy" row below as perplexity; it is not task accuracy.

**A method that loses on one axis is not eliminated if it wins on another.** Most
of what follows would have been thrown away under a single-number verdict. The
register exists so the axis a method wins on is recorded next to the axis it loses
on, and so a later reader can tell "measured useless" from "useless for this one
purpose".

All numbers are OLMoE-1B-7B (64 experts, top-8, 16 MoE layers) on GB10, held-out
gsm8k[400:500], baseline perplexity 9.703. `n/a` means not measured, not zero.

## Measured configurations

| configuration | load | decode | perplexity |
|---|---|---|---|
| baseline, 64 experts resident | 98.6 s | 689.4/s | 9.703 |
| disk tier, 24/64 resident | 45.2 s | 265.1/s | 9.734 (1.003×) |
| pruned to 40, no tier | 83.4 s | 695.1/s | 12.115 (1.249×) |
| **pruned to 40 + tier, 24/40 resident** | **44.5 s** | **350.8/s** | 12.145 (1.252×) |

Two readings worth keeping:

- **Surgery supports the tier.** Pruned+tier decodes **1.32× faster than tier
  alone** at the same capacity, because a 24-slot cache covers more of a 40-expert
  candidate set than a 64-expert one. This is why deletion is worth doing even
  though the tier alone has better accuracy — they are not competing options.
- **The tier halves load time** (45 s vs 99 s), which was not anticipated: expert
  weights stream to the store instead of being staged onto the device. A
  feasibility win that no accuracy or throughput number would have surfaced.

## Per-method verdicts

| method | feasibility | speed | accuracy | verdict |
|---|---|---|---|---|
| **disk tier** | **strong win**: `surgeon budget` puts the bit-exact weight floor at 2.39 GiB (capacity=top_k=8, arithmetic, **plus ~1.1 GiB measured non-weight overhead**; `vram-floor` measured 11.35 GiB streamed / 23.40 GiB not, at capacity 24 — see the boot-floor tables); halves load time | **loses**: 0.66× decode correctly sized (143.0 vs 218.2 tok/s, the release benchmark; the older 24/64 arm measured 0.38×) | neutral: 1.003× | **keep** — the only method that changes what is possible at all |
| **pruning (rank-1 ranked)** | win, but only when forced: 34% smaller artifact; the only way to run below the tier's floor | neutral alone (1.01×); win in composition (1.32× over tier alone) | **loses: 1.249× perplexity AND measured task accuracy** — arc_challenge −25% relative (McNemar p=5e‑08), not recovered by amplitude | **last resort** — recommended only when the tier alone cannot fit the target; the tier keeps every expert at no accuracy cost |
| **rank-1 importance** vs token count | n/a | n/a | **strong win**: 1.57× → 1.25× | **keep, as default** |
| **permutation-aligned merging** | n/a | n/a | no mergeable pair in **three** screened families: OLMoE (max 0.37 of 64512), Qwen3-30B (0.401 of 81280), DeepSeek-V2-Lite (0.331 of 6048, 2026-08-10) | **keep the machinery** — tested exactly, and redundancy is a per-model property; but three families at 0.33–0.40 max, none near 0.85, make a natural candidate implausible on this class of model |
| **residency prior** (`hot_experts.json`) | **win, narrow**: +10.0 pts hit rate over the first 50 accesses (measured with prewarm placement; the shipped runtime applies only the prior bias — see caveat), decaying to 0 by 2000 | negligible in steady state | n/a | **keep** — free (a byproduct of the plan) and confined to cold start; do not oversell |
| **EWMA cache policy** vs LFRU | win: +2.3…+4.0 pts on the cold-start window (single-tier replay) | win: +1.2 pts sustained hit rate | n/a | **keep as an opt-in** (`VLLM_MOE_CACHE_POLICY=ewma`); default stays `lfru` |
| **fp8 store** | **loses on VRAM**: none saved, the provider dequantizes into model-dtype slots | win: halves disk, host RAM and transfer bytes | small loss: quantization error, no argmax moved in testing | **keep** — the axis it wins on is not the one people expect |
| **static gate geometry** (row norm, cosine) | n/a | n/a | **loses**: ρ ≈ 0.00 against measured load and co-occurrence | **drop as a selection signal** — but see below |
| **static byte accounting** (`surgeon budget`) | **strong win**: answers "least VRAM" in 1.5 s with no GPU | n/a | n/a | **keep** — the same static analysis, pointed at sizing instead of selection |
| **`e_score_correction_bias`** | n/a | n/a | untested: absent from every accessible small MoE | **keep detection** — principled (a *learned* balancing term), just unavailable here |
| **cross-domain profile** as a cold-start prior | win: ρ +0.43, top-24 overlap 54% vs 37.5% random | n/a | n/a | **keep** — dominates static analysis, and profiling costs minutes |
| **block dedup across experts** | n/a | n/a | n/a | **drop** — untested, and near-orthogonality makes duplicate blocks implausible in trained bf16 experts |
| **correlation-driven disk layout** | n/a | no headroom: 6.0 MiB records already reach 93% of the NVMe ceiling on one thread | n/a | **drop** — the lever is already banked by the record-stride design |
| **CPU expert co-execution** (`cpu_experts`) | win: zero GPU bytes; the run-at-all mode's misses stop paying H2D | **machine-split, measured both ways**: laptop 1.29–1.58× live (7.37 tok/s, the box's best ever; gate 3.7× at f=1.0, s=1.037); GB10 **0.719× — a loss** (unified memory, s=2.385) | small loss: not bit-exact (host reduction order + one fp32→bf16 join); first 12 greedy tokens matched in the A/B | **keep as a loud per-machine opt-in** — enable on discrete cards where `BW_cpu_gemm/BW_h2d` > 1, never on unified boxes; see the gate entry |

> **Attribution and caveats for the residency-prior and EWMA rows.** The 2026-08-05 predecessor
> measured the same question with a simulator validated bit-exact against live
> counters (an unpublished working note) and found
> the opposite on an *end-to-end, two-tier* trace (Qwen3-30B): EWMA worth −0.05% disk
> reads, cold-start seeding −5.3% in the boot window, decision "default stays lfru",
> reopen bar ">15–20%". The numbers here are not a refutation of that — they are a
> different quantity (single-tier GPU **hit rate** on OLMoE via `store/replay.py`, not
> E2E disk reads) — but they are weaker evidence (one replay, one trace, one model,
> one capacity), so both stay opt-in and the default is unchanged. The `store/replay.py`
> table is also **not regenerable from the repo today** (no `replay` CLI subcommand);
> and `+10 pts at N=50` is ~5 cache hits, binomial SE ≈ 6.6 pts, so treat it as
> indicative, not precise.

The two static-signal rows are the clearest case for judging per axis. Gate
geometry is worthless for *choosing* experts and genuinely useful for *sizing* them
— the same analysis, kept or dropped depending on which question it is asked.

## Choosing a strategy per target and per model

**The tier is the primary mechanism; deletion is a last resort.** The tier keeps
every expert (cold ones stream from NVMe), so it costs no accuracy — only decode
throughput and, without streaming load, some host memory. Deletion permanently
removes experts and costs measured downstream-task accuracy that the perplexity gate
does not catch (arc_challenge −25% relative on OLMoE rank-1 pruned-40, McNemar
p=5e‑08; the amplitude fix does not recover it). So deletion is chosen **only when
the tier alone cannot meet the target** — when even the tier's floor is above the
target's VRAM, a smaller model is the only way to run at all — and declined
otherwise, cold tail or not.

| if the target… | then |
|---|---|
| fits the model resident, latency-sensitive | skip the tier (even correctly sized it decodes at 0.66× untiered) |
| below resident, at or above the tier's floor (`surgeon budget --vram`) | **tier only** — it runs the model at no accuracy cost; do not delete |
| below the tier's own floor | deletion becomes the last resort: shrink the model enough to fit, sized by `surgeon budget --vram`, gated by `surgeon gate`, task cost accepted |
| restarts often | seed the residency prior and set `VLLM_MOE_CACHE_POLICY=ewma` |

| if the model… | then |
|---|---|
| any model, tier fits the target | keep every expert on the tier; delete nothing |
| has a genuine dead tail (coldest expert ≪ uniform) AND the tier cannot fit | deletion is cheapest here; measure with `surgeon gate` |
| has no dead tail (OLMoE: coldest is 0.5% vs 1.56% uniform) | the tier is the answer; deletion only if a hard VRAM limit forces it |
| has mergeable pairs (similarity ≥ merge threshold) AND deletion is forced | merge instead of deleting — no capacity lost; OLMoE/Qwen have none |
| carries `e_score_correction_bias` | a static popularity prior is available with no profiling |
| has shared/always-on experts | they are fixed VRAM cost, never cached — `surgeon budget` accounts for them separately |
| ships pre-stacked expert tensors (IBM Granite) | both directions are supported: read (inspect, budget, tier) and write-back (`apply`) |

`surgeon inspect` and `surgeon budget` between them report every property in the
second table without a GPU, and **`surgeon recommend` applies both tables**:

```bash
surgeon recommend --checkpoint model --vram 3.68 --profile p.npz --similarity-cache sim.npz
```

On OLMoE against a 3.68 GiB target it reproduces this session's conclusions from the
measurements alone — tier mandatory at capacity 10, delete nothing (no dead tail),
merge nothing (0.370 < 0.85), seed the prior, use EWMA — and lists what it could not
decide rather than guessing. Deletion quality is always deferred to `surgeon gate`.

## Release benchmark, 2026-08-11 (two machines, post-S5)

Run before publishing, on GB10 (OLMoE bf16 store, 8 prompts × 256 tokens, repeats=3,
median ± std) and on a 4 GiB laptop GPU (fp8 store, 4/64 slots, expert split). Held-out
perplexity here uses a fixed two-paragraph passage, **not** the gsm8k set the older
tables use, so its absolute value is not comparable across sections — only the
tier-vs-baseline ratio within this run is.

| GB10 arm | load | decode | σ | ppl |
|---|---|---|---|---|
| untiered, eager | 117.9 s | 218.2/s | 0.001 | 11.6253 |
| untiered, graphs | 119.9 s | 226.4/s | 0.008 | 11.4729 |
| tier 24/64, ram_cache 48, eager | 14.1 s | 33.6/s | 0.315 | 11.6253 |
| tier 24/64, ram_cache 48, graphs | 18.5 s | 31.7/s | 0.639 | 11.5874 |
| tier 24/64, **ram_cache 64**, eager | 13.0 s | 55.8/s | 0.044 | 11.6253 |
| tier 24/64, **ram_cache 64**, graphs | 35.0 s | 55.2/s | 0.044 | 11.5874 |

### The capacity sweep, and the finding that dominates everything else here

The arms above all ran at `expert_cache_size=24`. A follow-up investigation into whether
prefetching could hide the disk measured the actual per-layer
expert **union** under this benchmark's batch of 8: **35.28 mean, 46 max**. Twenty-four
slots against a 46-expert working set is 47% oversubscribed, so `plan_chunks` splits every
layer into 2–3 chunks — each with a blocking `topk_ids.tolist()` D2H, its own `prepare()`
and mapping upload, re-read of experts shared between chunks, and a separate expert GEMM
over a token subset. None of that is announced; it just runs slowly. Sweeping the capacity
(ram_cache 64, eager, otherwise identical) against the model's *pre-registered* predictions:

| `expert_cache_size` | GPU expert bytes | predicted | **measured** | σ | ppl |
|---|---|---|---|---|---|
| 24 | 4.50 GiB | 54.5/s | **55.3/s** | 0.69 | 11.6253 |
| 32 | 6.00 GiB | 76.0/s | **75.5/s** | 0.048 | 11.6253 |
| 40 | 7.50 GiB | 103.4/s | **107.9/s** | 0.054 | 11.6253 |
| 48 | 9.00 GiB | 129.3/s | **143.0/s** | 0.053 | 11.6253 |
| *untiered* | *12.00 GiB* | — | *218.2/s* | 0.001 | 11.6253 |

**24 → 48 is 2.59×, for one integer.** The falsification threshold set before the run
(cap 40 must clear 95 tok/s or the model was wrong) was cleared at 107.9. At 48 ≥ 46 the
chunk split disappears entirely, which is why that row beats its own prediction. Note the
σ column, with a caveat that matters more than the reading it invites. The same
configuration (cap 24, ram 64, eager) has now been measured at three separate sittings:

| sitting | median | σ |
|---|---|---|
| release table above | 55.8/s | 0.044 |
| this sweep | 55.3/s | 0.69 |
| re-run after the final code changes | 55.8/s | 0.055 |

**The medians agree to ±0.9%; the spreads differ by 16×.** So "an oversubscribed cache is
erratic" is *not* supported — the 0.69 is the outlier, and a single within-process σ is not
a stability measurement. (The disk-spilling arm's 0.315 should be read the same way.)
Anything resting on run-to-run variance here needs n ≥ 5 separate processes; the throughput
medians, which move 2.59× and reproduce across every sitting, do not.

**What the extra capacity actually costs is worth stating precisely, because the obvious
answer is wrong.** The four arms report peak VRAM 29.29 / 29.29 / 29.41 / 29.25 GiB — *flat*.
Twenty-four more slots is 4.5 GiB of expert residency, but at a fixed
`gpu_memory_utilization` vLLM sizes the KV cache to fill whatever is left, so the slots are
paid for out of **KV cache, not out of additional device memory**. Capacity and KV trade
roughly 1:1. So the real price of the 2.59× is context length and concurrency, and the
right way to choose the number is against a KV budget rather than against free VRAM. (This
also means `surgery/budget.py`'s model — spare bytes go to slots, KV is a fixed reserve —
describes the *floor* question correctly but not the *serving* question.)

Perplexity is **11.6253 at every capacity and untiered** — the speedup is free on the
accuracy axis, because none of this changes what the model computes.

**That claim was then tested properly, because a perplexity printed to four decimals is
not evidence of bit-exactness.** A controlled determinism run (seed-pinned, greedy,
`max_tokens=128`, sha256 of the token ids, 3 repeats per configuration, one arm per
process) covered three configurations: **untiered 3/3 identical, tier cap 48 / ram 64 3/3
identical, and `VLLM_MOE_ZERO_COPY=1` 3/3 identical — all nine runs the same hash**
(`370632d4…`). So the tier is bit-exact against untiered on the token stream, not merely
close on a rounded metric; every configuration is deterministic run to run; and that holds
on the zero-copy path too, which is the one with the most dangerous rollback semantics.

This matters because a preceding investigation reported the opposite — that a tiered arm
diverged from untiered — and it was wrong for an instructive reason: its `untiered`
control had **one** successful run (two others failed to boot) and had been taken at a
different memory configuration. A different `gpu_memory_utilization` sizes the KV pool
differently, which changes batch composition and hence which kernel configurations get
selected, which changes the float reduction order. That is a *configuration* difference
masquerading as nondeterminism. **Never compare token hashes across configurations, and
never accept a control with n=1.**

So the corrected headline is that a correctly sized tier decodes at **143.0 vs 218.2 tok/s
(0.66×)**, not the 0.25× the first arms suggested.

A follow-up investigation
decomposed that 2.59×: **76–89% of it is simply fetching fewer bytes** (loads fall 26.4 →
4.7 per layer per step, and a bf16 load is 217 µs of pure host→device DMA), and only 11–24%
is the chunk split. The split's cost is also not what the code comments claimed — re-fetching
shared experts over the bus is 2.5% of misses; the real cost is the GPU re-reading resident
weights from its own DRAM once per chunk. Since residency is a *byte count* and prefetch only
changes *when* bytes move, prefetch cannot substitute for capacity here; Belady's optimum on
the real trace is worth 1.068× and known-future chunk prefetch measured 1.03–1.075×.

That study ranked "make the MoE kernel consume fp8 weights directly" as the one lever that
cuts bytes without spending GPU memory, worth 1.92× per load. **Probing the live model shows
that is already how a native fp8 checkpoint runs**, so there is no kernel work to do there:

| path | slot dtype | slot bytes / 8 slots | scale slots | disk record |
|---|---|---|---|---|
| bf16 model + `fp8_store: true` (OLMoE) | `bfloat16` | 67.1 MB | none | 6.31 MB (fp8) |
| native fp8 checkpoint (DeepSeek-Coder-V2-Lite-FP8) | **`float8_e4m3fn`** | 46.1 MB | **yes**, fp32 | 8.65 MB |

The provider allocates its slots with `w13_weight.dtype`
([expert_weight_provider.py:622](src/vllm_moe_surgeon/store/expert_weight_provider.py)), so
when `Fp8MoEMethod` hands it fp8 weights the slots are fp8 and the scales get their own slot
buffers — no dequantisation anywhere, half the slot memory, and this is the path S1 verified
token-identical. The dequantisation the study costed belongs only to the *other* case, a
**bf16 model with an fp8 store** (`quantize_fp8=config.fp8_store`,
[runtime.py:309](src/vllm_moe_surgeon/compat/runtime.py)), where the model's own kernel
expects bf16 so something must expand the record. Making *that* path fp8-direct means serving
a bf16 checkpoint through the quantised path — i.e. quantising the model, which changes its
output. That is an accuracy decision to be judged on the three axes, not a free kernel win,
and the study's 1.92× should not be read as available for nothing.

**Which settles what `fp8_store` is for on a bf16 model, measured at the same capacity:**

| cap 48, ram 64 | decode | σ | perplexity | load |
|---|---|---|---|---|
| bf16 store | **143.0/s** | 0.053 | **11.6253** | 14.7 s |
| `fp8_store: true` | **128.6/s** | 0.008 | **11.7393** | 12.8 s |

It costs **1.11× decode, and it changes the output** — the three configurations put through
the token-hash control (untiered, tier cap 48 / ram 64, zero-copy) all reproduced the same
stream, and this one does not, because it quantises. (The eager tier arms also match on
perplexity to four decimals; the *graph* arms do not — 11.5874 against 11.6253 — which is the
graph-vs-eager float-order difference that shows up in stock vLLM too, not something the tier
introduces. Only the token-hash control speaks to bit-exactness; the perplexity column is
weaker evidence and is not being asked to carry that claim.) The per-load arithmetic predicts
the cost: an fp8 record is 6.31 MB (109 µs of DMA) but adds ~170 µs of dequantisation, so
279 µs against the bf16 store's 217 µs. Multiplied by the 4.7 loads per layer per step
measured at this capacity and 16 layers, that is **4.66 ms/step predicted against 6.27 ms
measured** — the right order and the right sign, 26% light, which is as much as a
single-term model should be trusted for. (An earlier draft quoted 5.65 ms by using 5.70
loads, a simulator-predicted figure this document supersedes with the measured 4.7 forty
lines above; the agreement it claimed was an artefact of mixing the two.)
**So `fp8_store` is a space mechanism, not a speed one: turn it on when the store or the host
RAM tier does not otherwise fit, and leave it off when they do.** (Its peak-VRAM column read
*higher* here, 35.29 vs 29.25 GiB, which the byte accounting does not explain and which is
recorded rather than rationalised.)

Four further findings, in order of how much they should change what anyone does:

- **Sizing beats everything else measured here, on both axes. `ram_cache` must be ≥ the
  expert count, and `expert_cache_size` must be ≥ the per-layer union at the serving
  batch** (the second axis is the sweep above, worth 2.59×; this bullet is the first).
  Going 48 → 64 (i.e. removing all host-RAM spill, so the disk is never touched during
  decode) moved decode **33.6 → 55.8 tok/s (1.66×)** and collapsed the run-to-run spread
  **0.315 → 0.044** — though see the caveat above: the same configuration produced 0.69 at
  a second sitting, so these within-process spreads cannot carry a claim about stability.
  What survives is the median. A `ram_cache` below `num_experts` silently converts every
  eviction into a disk read, which is the single most expensive misconfiguration in the
  system and is not currently refused (it is only refused when `ram_cache == 0`).
- **The tier's remaining cost is bytes over the host→device link, not disk.** With zero
  disk I/O and a correctly sized slot cache the gap to untiered is 143.0 vs 218.2 (1.53×).
  On this **bf16** store a load is **217 µs of pure DMA** (12.58 MB at 57.9 GB/s, matching
  the measured link rate with no other term hiding in it), and the tier moves 117–119
  MB/token at cap 48 against 663 MB/token at cap 24. (An earlier draft of this section
  attributed 61% of a load to on-GPU fp8→bf16 dequant. That figure is real but belongs to
  the **fp8** store, where a load is 276.5 µs = 170.1 dequant + 113.4 DMA; none of the
  arms benchmarked here use it. The correction matters: on the bf16 path there is no
  dequant to fuse.) Further speed work therefore has to cut **bytes per load** — a kernel
  that consumes fp8 weights directly halves the record and is the largest lever found —
  not the disk, not the cache policy, and not prefetch.
- **Load time is where the tier wins outright: 13.0 s vs 117.9 s, ~9×** (streaming, so
  the full `[num_experts, …]` tensors are never materialised).
- **Numerical transparency confirmed on a second axis.** tier-eager perplexity is
  **11.6253, matching untiered-eager to four decimals** — consistent with the S5
  token-identity result, though a rounded perplexity is not itself evidence of
  bit-exactness (see the determinism control below, which is).

Laptop (RTX 3050 Ti, 3.6 GiB free, fp8 store, 4/64 + expert split): the tier serves a
12.9 GiB model at **2.69 GiB peak, 7.7 tok/s, load 6.6 s** eager. Graphs need more
headroom on this card — refused at boot at `gpu_memory_utilization` 0.83, fitting at
0.90 (2.71 GiB peak, 29.4 s load), i.e. ≈290 MiB and 4.5× load. Untiered was **not**
booted —
`surgeon budget` puts the bit-exact floor at 2.39 GiB GPU and shows that fitting 3.6 GiB
allows capacity 7 < top_k 8 (so the expert split is mandatory and no longer bit-exact),
and an actual untiered boot attempt drove the 14 GB-RAM box into swap hard enough to
lose ssh — which is itself the feasibility result. This is the regime the tier exists
for.

**Re-benchmarked after the code changed.** The capacity numbers above were taken before the
oversubscription diagnostic existed; that diagnostic synchronises with the device, and on a
correctly sized cache its first version never retired, so the published figures had to be
re-earned rather than assumed. With the final code (diagnostic bounded to 8 probes per
layer): **cap 48 → 143.6 tok/s (σ 0.039), cap 24 → 55.8 (σ 0.055)**, perplexity 11.6253 at
both. Unchanged within noise, so the bounded diagnostic costs nothing measurable and the
table stands.

**Method note, recorded because the first attempt was wrong.** The first pass measured
the tier only at `ram_cache 48`, where decode is disk-bound; in that regime graphs and
eager are indistinguishable inside the noise, and the run could not have answered the
question it was run to answer. The `ram_cache 64` arms were added after that critique.
Absolute throughputs here also differ from the older tables in this file (which report
689/265 tok/s); those were taken at a different measurement point and are kept as-is
rather than silently overwritten — compare ratios within a run, never across runs.

## The vLLM pin, and what CI now measures for us

`runtime = ["vllm>=0.26.0,<0.27"]`. Two things about that line were checked rather
than assumed, on 2026-08-11, by cloning the tags and running `surgeon seams --source`:

- **The floor named a version that does not exist.** It said `>=0.26.1`, but upstream's
  tags run `v0.26.0` -> `v0.26.1rc0` -> `v0.27.0`: 0.26.1 was never released, and the
  runtime work was done against a dev snapshot of it. Nobody could have installed what
  the pin asked for. Corrected to `0.26.0`, which the seam check passes against (0
  required broken; only the optional `_orient_fused_weight`, which upstream had not
  extracted yet, is absent).
- **The ceiling is now stricter than the evidence.** Against **v0.27.1**, a full minor
  version past the ceiling, the check reports **0 required seams broken** — and
  `_orient_fused_weight` resolves, so a seam the table records as "emerging" has landed.
  That is the architecture's central claim measured rather than argued.

The ceiling stays at `<0.27` anyway, and the reason is the one this register keeps
insisting on: the static check parses names and signatures, **not behaviour**. Every
runtime verification in this document — token identity, the graph controls, the capacity
sweep — was run against the 0.26.1-dev fork. Widening the ceiling would claim 0.27
support on a check that cannot see a semantic change, which is the overclaim this
document exists to prevent. What it needs is one runtime pass on 0.27: boot the tier,
hash the tokens against untiered, and the ceiling can move.

`.github/workflows/vllm-compat.yml` now asks the question weekly and on demand, and
opens an issue naming the symbol when the answer changes.

## Narrow-domain pruning, measured on real logs (2026-08-12)

The recurring question — "surely a deployment that only ever analyses logs can
delete experts?" — was answered on real data (LogHub-style corpora: Linux + SSH +
Apache as the domain, task-shaped triage prompts, held-out lines from the same
files, Windows + HealthApp as the out-of-domain probe). Full pipeline: profile →
plan → gate → apply → calibrate, all on GB10.

| configuration | held-out log ppl | vs baseline |
|---|---|---|
| baseline, 64 experts | 27.89 | 1.000× |
| tier, nothing deleted | 27.89 | 1.000× |
| keep-56 applied (the near-dead tail deleted) | 30.64 | 1.099× |
| keep-40, zeroed gate bound | 33.26 | 1.192× |
| keep-40 applied, **no amplitude** | 41.00 | **1.470×** |
| **keep-40 applied + amplitude 0.850** | **32.53** | **1.166×** |

Three findings worth keeping:

- **A narrow domain does concentrate routing — but less than intuition says.** For
  the first time a near-dead tail exists (median 7 experts/layer under 10% of
  uniform; the coldest at ~0.00×), the hot quartile carries 69% (gsm8k: 52%), and
  deleting 24/64 costs 1.17× against gsm8k's 1.25× at the same depth. Cheaper,
  not free — and the hot sets still overlap gsm8k's at 56% (random: 37.5%), so
  even logs share the model's core experts.
- **The zeroed gate is NOT always a pessimistic bound.** On gsm8k, applied
  deletion beat zeroing (15.26 vs 16.90); on logs it was far *worse* (41.00 vs
  33.26) until the amplitude fix, which recovered it to 32.53 — below the bound
  again. Mechanism: deletion shrinks the softmax to the survivors and inflates
  their gates by 1/(1−P_D); the log domain concentrates more deleted mass into
  fewer layers, so the inflation bites harder than zeroing's discarded mass.
  Verified in a clean process before being believed, and the single 0.850
  multiply recovered 80% of the gap. **For narrow-domain pruning, `calibrate` is
  mandatory, not optional** — and a gate verdict on a plan that will be applied
  without amplitude understates the damage on domains like this one.
  **Follow-up, head-to-head:** the plan can *predict* this amplitude from its own
  deleted routing mass — (1 − P_D) with mean P_D = 13.9% gives 0.861 — and the
  prediction was applied and measured against the calibrated 0.850: **32.59 vs
  32.53 held-out ppl (0.2% apart)**. So the analytical amplitude the plan now
  prints is a valid first pass with zero GPU cost; `calibrate` remains the
  measured refinement. `apply` warns, with these numbers, when any layer loses
  more than 5% of routing mass and no amplitude is given.
  *(Review correction, same day: the mean is now taken over **all** layers the
  plan covers, not just the deleting ones — the amplitude is folded into every
  layer's survivors, so the deleting-layer mean would over-damp untouched layers
  on a concentrated plan. This experiment deleted on every layer, so 0.861
  stands. When the max deleted share sits far above the mean, one global scalar
  fits poorly — that is `calibrate`'s case.)*
- **Within the log universe there is no distribution cliff**: the same keep-40
  plan gated 1.16× on out-of-domain logs against 1.19× in-domain — unlike the
  gsm8k → hellaswag case (1.404× vs 1.234×). Log families share structure;
  task families do not.

**And the gain side, measured rather than assumed** (the table above is the cost
side). The applied keep-40+amplitude checkpoint was served from the tier at full
coverage and compared against the unpruned model at its own full coverage:

| arm | GPU slot bytes | decode |
|---|---|---|
| **pruned-40 + tier, 40/40 resident** | **7.5 GiB** | **256.2 tok/s** |
| unpruned + tier, 64/64 resident | 12.0 GiB | 205.1 tok/s |
| unpruned, untiered | 12.0 GiB | 218.2 tok/s |

Three measured gains for the 1.17× ppl cost: **decode +25%** over the unpruned
tier ceiling — the pruned model even beats the *untiered* baseline, because a
40-expert GEMM is smaller than a 64-expert one, so pruning buys compute, not
just memory; **~4.6 GiB freed and visible in the allocation** (the pruned run's
pool grew by almost exactly the slot bytes released — the slots↔KV trade,
confirmed a third time from the pruning direction); and a **37.5% smaller
store**. So in a genuinely narrow domain the composition earns its keep on all
three axes at once, which the gsm8k domain never managed.

The tier-first default earned its keep along the way: `plan` without
`--disk-experts 0` placed the cold tail on disk instead of deleting it, and the
gate answered "the plan deletes nothing". Deletion had to be asked for
explicitly, which is the design.

## Stress campaign, 2026-08-12 (both machines, limits deliberately pushed)

Nine scenarios on GB10 and four on the laptop, chosen to hit combinations nothing
had ever run together and regimes nothing had measured. Token streams are compared
by sha256 over every prompt's ids, one arm per process, greedy.

**Combinations proven safe, first time run:**

- **Zero-copy under piecewise CUDA graphs** is token-identical to the fill path
  (same hash, 1024 tokens across 8 prompts) and decodes ~5% faster (131.0 vs
  124.9 tok/s). The pairing had never been exercised; it is now a supported claim.
- **fp8 checkpoints under graphs**: the eager pair is token-identical (tier vs
  untiered, same hash over 192 tokens), so the tier remains transparent; the
  graph arms diverge from eager and from each other, which is the same
  graph-vs-eager float-reduction class measured on OLMoE (the tier's MoE op runs
  eager between graph pieces, so its reduction environment differs from a fully
  captured baseline). Attributed by running all four controls, not assumed.

**Limits measured, first numbers:**

- **The capacity lever grows with batch.** At batch 64 (per-layer union = the full
  64), cap 48 → 64 is **291.3 → 943.9 tok/s (3.24×)** — larger than the 2.59×
  at batch 8 — with identical token hashes, so capacity still changes only speed.
  The oversubscription warning fired exactly once per layer at cap 48, which is
  the scenario the old batch-32 threshold silenced.
- **The expert split's cost is now empirical, not asserted**: deterministic
  (hash-stable across repeats) but token-divergent from the unsplit reference,
  at 40.9 vs 125.5 tok/s for cap 4 vs 48 on GB10.
- **Sustained generation is stable**: 3 × 1024 tokens/prompt × 8 prompts, hash
  identical across repeats, spread under 1%, resident set flat — no leak signal.
- **On a severely bound device the capacity lever flattens**: the laptop serves
  at 5.5 tok/s with 1 slot and 5.8 with 4 — the bottleneck there is the fp8
  read+dequant path, not residency. Batching also inverts on tiny capacity:
  4 prompts ran *slower* than 1 (5.8 vs 7.7 tok/s) because the union quadruples
  against 4 slots and every layer splits into chunks.

**Defects the campaign found, fixed in this commit:**

- `autoconfig`'s flat 2.0 GiB KV reserve consumed over half of a 3.5 GiB card and
  strangled capacity to 1 — below top_k, forcing the non-bit-exact expert split
  against a proven hand-tuned 4. The reserve now scales (15% of the *bucketed*
  free-VRAM reading, clamped to [0.5, 2.0]); re-probed live, the same card now
  decides capacity 9 with no split. An explicit --kv-reserve still wins.
  *(Review correction, same day: the fix had only reached `surgeon autoconfig` —
  `surgeon run`, the command that boots the engine, still passed a flat 2.0; and
  resolving from the raw reading leaked probe jitter into the cache fingerprint.
  Both fixed; see the review entry below.)*
- `surgeon run` piped to another process delivered **zero output**: stdout is
  block-buffered on a pipe and vLLM's exit teardown can end the interpreter
  before the buffer flushes — the session ran to completion, every print lost.
  Line-buffering now lives at the top of `cli.main`, so every subcommand that
  boots an engine — and every job-server stage log, captured to a block-buffered
  file — gets the same durability, not just the prompt.
- A deliberate `split="expert"` configuration — the run-at-all mode whose whole
  point is that the device cannot hold more slots — was scolded with sixteen
  layers of WARNING advising it to raise the setting it had explicitly declined.
  An explicit expert split now suppresses the oversubscription warning.
  *(Review correction, same day: the check read the ambient vLLM config, which
  does not exist at forward time — so the suppression never fired for
  `additional_config`-configured runs, the documented channel, and only worked
  for the env-var recipe it happened to be tested with. It now reads the split
  off the provider's build-time snapshot.)*
- `autoconfigure`'s own decision (capacity 1, ram_cache 64 = 6 GiB pinned on a
  15 GiB host) was boot-tested on the fragile laptop under a timeout guard: it
  served correctly and the box stayed up, so the half-the-host pinned-pool rule
  holds at the boundary it was written for.

## CPU expert co-execution: gated, and measured dead on GB10 (2026-08-12)

The capacity-substitution study left one live candidate: compute cold experts
on the CPU from the host rows the tier already holds, instead of fetching them
over H2D (Fiddler-shaped; surveyed at 1.6–2.2× from isolated primitives). The
register's own rule — measure before committing — was applied as a single-layer
gate on GB10 before any runtime code: real decode shape (B=8, top_k 8, cap 24,
union ~40), 26 streaming weight sets, GPU arm vs a co-exec arm running the
coldest half on 10 cores concurrently with the GPU half.

**NO-GO, decisively.** GPU-only 6.07 ms/layer; co-exec 8.44 ms at f=0.5
(ratio **0.719**, the gate needed ≥ 1.4; every f in 0.336–0.568 and nt in
{10, 6} lost). The killer is measured contention: the CPU half ran **2.385×**
slower under the live GPU than solo — statistically the same as the 2.33×
"pessimistic" bound taken against a saturating GPU. On unified memory the
balanced duty cycle does not exist: CPU GEMM reads, H2D copies, and the GPU's
own weight reads all queue on one LPDDR5X controller, so the co-exec's premise
("zero GPU bytes" buys a second bandwidth pool) is false on this box. The
1.6–2.2× model assumed s ≤ ~1.5; the measurement refutes it where it was
always weakest — the term the survey admitted was never hostilely reviewed.

Two findings survive the death and are worth keeping:

- **The T=1 GEMV cliff and its fix are now measured, twice.** A single-token
  expert forward costs 1000.7–1497.7 µs on 10 cores; padding the token block
  to 2 rows recovers it to 282.3–294.0 µs (~3.5–5.1×). The "pad to ≥2" fix was
  asserted in the survey; it is now a number.
- **The cold tail is a T=1 population.** Under B=8 decode with the hottest 24
  resident, the cold experts carry almost exclusively one token each — a
  min-tokens ≥ 2 *exclusion* empties the candidate set entirely (the first
  gate run selected zero experts in every rep). Any future CPU-exec attempt
  must pad, not exclude — and its per-expert arithmetic must use the T=1
  padded cost, not the T=2 one.

**L6 follow-up, same day: on the laptop the answer is GO — decisively.** The
same gate, laptop shape (i7-12700H, RTX 3050 Ti, B=2, cap 4, pad-to-2
arithmetic): H2D costs **1139.7 µs/record** over PCIe (11.0 GB/s) against the
CPU's **368.8 µs/expert** (34.1 GB/s effective) — `BW_cpu_gemm/BW_h2d` =
**3.09**, inside the survey's predicted 1.3–4 window. The concurrent arm:
GPU-only 12.72 ms/layer vs **3.45 ms at f=1.0** (all eleven cold experts on
the CPU) — **3.7×**, and implied contention **s = 1.037**: on a discrete card
CPU DRAM reads and PCIe H2D genuinely are separate pools, the term that
killed GB10 simply is not there. Two policy consequences, both measured: the
winning fraction is **f = 1.0** — never H2D a cold expert on this class of
machine, compute every one of them on the CPU — and the T=1-dominated cold
tail costs 368.8 µs/expert *padded*, so the pad-never-exclude rule from the
GB10 run carries over unchanged. Gate scripts: `bench/l5_cpu_coexec_gate.py`
(GB10), `bench/l6_cpu_coexec_gate.py` (laptop). Implementation proceeds,
laptop-targeted, per the approved plan: the feature is a per-machine opt-in
whose one honest predictor is the measured `BW_cpu_gemm/BW_h2d` ratio — GB10
class (unified, ratio < 1) stays refused-by-record; laptop class (discrete,
ratio ~3) is the target.

**Implemented and measured end-to-end, same day** (laptop, OLMoE bf16 store,
B=2, cap 4, `split=expert`, eager, one arm per process, greedy, 2×128 tokens
×3 repeats):

| arm | ram_cache | decode | ratio | GPU misses | cpu_execs |
|---|---|---|---|---|---|
| off | 16 | 2.78 tok/s | — | 95,405 | 0 |
| on | 16 | 3.58 tok/s | **1.29×** | 11,013 | 73,925 |
| off | 24 | 4.66 tok/s | — | 95,405 | 0 |
| on | 24 | **7.37 tok/s** | **1.58×** | 1,414 | 85,165 |

Re-measured after this feature's own hostile review, since several fixes
touch the hot path; the pre-review build read 1.37× and 1.52×. Compare within
a pair, never across: each pair ran back-to-back with the flag as the only
difference, while absolute tok/s drifts between sessions — the ram16 off arm
measured 3.72 in one window and 2.78 in another with **byte-identical
counters**, i.e. the same work on a differently-warmed disk-bound machine.

7.37 tok/s is the fastest this laptop has served this model (the fp8 recipe
topped out at 6.10, indirectly comparable at best — different store type).
The mechanism is measured, not inferred: in-engine host cost is **356–377
µs/expert against the gate's solo 368.8**, so contention inside the real
engine is ≈1.0, exactly as the L6 gate predicted. The
residual gap to the gate's 3.7× is the disk tier — the RAM pool (16–24 rows)
is far below the 64-expert working set, so both arms still read cold rows off
NVMe; the ram16→ram24 trend is that gap closing.

**What the miss column does *not* mean.** 95,405 → 1,414 is not the cache
learning. Every CPU-served expert is masked out before the planners see it,
so `prepare()` only ever meets residents: no insertions, no evictions, the
resident set frozen at whatever the last uncovered forward left. The misses
became invisible. Two consequences to keep in mind when reading any counter
from a co-exec run: the GPU tier's eviction policy is unreachable while
co-exec covers the whole miss set, and a low GPU hit rate means the opposite
of what it means without this mode. (An earlier draft of this entry
attributed the collapse to "residency finally forming"; the hostile review
of the implementation refuted it from the masking path.)

The first 12 greedy tokens matched the baseline on both prompts — an
observation, not a guarantee; the mode stays declared not-bit-exact.

**Costs, recorded honestly.** v1 is silu-only and bf16-records-only (fp8
records would need a CPU dequant twin); refused with CUDA graphs, zero-copy,
fp8 checkpoints, non-silu activations and router-weight-on-input. One new
data coupling (`layer.activation` semantics), zero new internal seams. Live
catches worth keeping: `layer.activation` is an **enum** in this vLLM
(`MoEActivation.SILU`), not a string — the first laptop boot was refused by
our own check until the comparison normalized on `.value`; and the laptop's
three pinning-strictness tests fail identically on the pre-change provider,
i.e. environmental, verified by overlaying the old file.

**Hostile review of the implementation, same day: ten findings, all fixed.**
Four correctness bugs the A/B could not have surfaced — `cpu_release` outside
a `try/finally` (a leaked protection entry made the RAM victim scan's bare
assert reachable, killing the engine on a later forward); `out[rows] += …`
lowering to a non-accumulating `index_put_`, which silently drops one term
when a token routes to the same expert twice; `clamp_min(0)` folding `-1`
padding into expert 0's count and pushing a genuinely cold expert 0 back onto
the H2D path; and `cpu_views_for`'s fp8 guard testing `_store_fp8`, a flag
true only for row-quantized *records*, so an fp8 **checkpoint**'s per-expert
scales would have been dropped silently. Also fixed: `VLLM_MOE_CPU_EXPERTS=True`
parsed as *off* (two env channels disagreeing), the thread knob applied too
late for torch to honour it, a missing prefetch drain, and a second
`_DECODE_ROWS` constant at 64 shadowing runtime.py's 512. The provider now
serves as many experts as its pool can hold rather than asserting, and the
oversubscription warning stands down under co-exec instead of advising the
user to disable the mode.

## Model selection, gated against pruning — and it wins (2026-08-13)

The recurring proposal — "for a narrow domain, distil a small student instead
of tiering or cutting experts" — was gated before any training compute, the
same way CPU co-execution was. The gate never reached distillation, because
its first rung answered the question outright.

**D1, zero-shot headroom, no training.** `ibm-granite/granite-3.0-3b-a800m-base`
(3B total, **800M active**) scored on the *same* held-out log corpus as the
narrow-domain experiment, against OLMoE re-measured in the same session.
Per-token perplexity is not comparable across tokenizers — granite splits this
text at 2.64 bytes/token against OLMoE's 3.41 — so the honest metric is bits
per byte:

| model | active | bits/byte | per-token ppl |
|---|---|---|---|
| **granite-3.0-3b-a800m** | **800M** | **1.3395** | n/a (different tokenizer) |
| OLMoE-1B-7B, full | 1B | 1.4073 | 27.87 |
| OLMoE pruned-40 + amplitude | ~0.9B | 1.4727 | 32.53 |

OLMoE re-measured at 27.87 against the recorded 27.89, so the harness
reproduces the original experiment. A smaller off-the-shelf model models this
domain **4.8% better than the unpruned teacher and 9.0% better than the pruned
checkpoint**, with no training of any kind.

**D2, does it keep general capability?** The objection D1 could not answer: the
log domain still leans on the model's core (hot sets overlap gsm8k's at 56%),
so a smaller model might win the domain and lose everything else. Same lm-eval
protocol as the pruning run — 500 items/task, `--log_samples`, paired exact
McNemar. OLMoE reproduced its recorded 0.468 / 0.662 exactly.

| task (metric) | pruning cost (recorded) | granite vs OLMoE |
|---|---|---|
| arc_challenge acc_norm | −0.116 (p=4.8e‑08) | **−0.014 (p=0.51, ns)** |
| hellaswag acc_norm | −0.044 (p=0.014) | −0.044 (p=0.0032) |
| gsm8k strict | −0.042 (p=0.0055) | **+0.266 (p<1e‑4)** |
| arc_challenge acc | n/a | −0.068 (p=0.0008) |

**The uncomfortable reading, recorded because the register exists for exactly
this.** On the task pruning destroyed, switching model costs nothing
measurable; on gsm8k the smaller checkpoint is 3–5× better. **For this model
and this domain, choosing a better small model dominates pruning on every
axis** — better in-domain likelihood, no significant arc loss, far better
gsm8k, 3B instead of 7B, and 12.8 s load against 84.2 s. Pruning's whole
purpose was to buy footprint at a measured quality cost; here footprint comes
free and quality improves.

**Three limits, so this is not over-read.**

- **It compares models, not strategies.** OLMoE-1B-7B-0924 and granite-3.0 are
  different training efforts; a stronger model per active parameter beating a
  weaker one says nothing about whether the tier helps the stronger one.
- **It does not touch the tier.** The tier's job is fitting a model that does
  not fit, and granite at 3B bf16 is ~6.4 GiB — still above the 3.68 GiB laptop
  card. The tier composes with whatever checkpoint is chosen; only *pruning* is
  dominated here.
- **The domain metric is still a proxy.** D1 scores likelihood on prompt text
  that is mostly log lines, i.e. compression of structured text, not triage
  competence. And granite loses arc_challenge on raw `acc` even while matching
  on `acc_norm` — the two metrics disagree, and the recorded protocol used
  `acc_norm`.

**Consequence for the distillation proposal: it is not needed for this case and
was never reached.** If an off-the-shelf model of the target size already wins
the domain and holds general capability, there is nothing to distil. Training
compute stays unspent, the package stays training-free, and the standing advice
gains a step that costs minutes: **before tiering or cutting, measure whether a
smaller existing checkpoint already serves the domain better.** Distillation
remains open only for the case this gate did not produce — a target size with
no adequate off-the-shelf model.

Scripts: `bench/d1_headroom.py`, `bench/d2_capability.sh`, paired analysis via
`bench/eval_pair.py`.

## Hostile review of the day's commits (2026-08-12)

Eight-angle adversarial review of everything after the previous review commit;
sixteen candidates, ten confirmed, all fixed. The pattern that matters more than
any single finding: **three of the day's headline fixes had been verified live
through exactly one configuration recipe and did not generalise past it.**

- The expert-split warning suppression read `read_config()` at forward time,
  where there is no ambient vLLM config — the env fallback answered `"token"`,
  so the suppression never fired for `additional_config`-configured runs (the
  documented channel; the live test used the env-var recipe). Now reads the
  provider's build-time snapshot, which also removes a raised-and-caught
  exception per not-yet-retired layer forward.
- The scaled KV reserve landed only on `surgeon autoconfig`; `surgeon run` still
  passed a flat 2.0 as an "explicit" override. Both subcommands now default to
  None and the resolver owns the default.
- The resolved reserve fed the cache fingerprint from the **raw** VRAM reading
  while the fingerprint buckets that same reading to 0.5 GiB against probe
  jitter — so on exactly the small cards the scaling targets, the cache missed
  on nearly every boot. The reserve now resolves from the bucketed figure.
- `suggested_amplitude` averaged deleted share over deleting layers only, but
  the scalar folds into every layer's survivors — a plan deleting 30% on 2 of
  24 layers would advise damping the other 22 by 0.7. Mean is now over all
  covered layers; the apply warning triggers on the per-layer max.
- `concentration_report` counted silent (never-routed) layers as entire layers
  of near-dead experts — missing measurement read as a license to prune,
  directly under the profile's own do-not-prune warning. Silent layers are now
  excluded and named. The hot-quartile baseline also printed a hard-coded 25%
  for expert counts not divisible by 4; it now prints the actual slice share.
- The piped-output fix line-buffered only `repl.run`; profile/gate/ablate/
  calibrate and the job server's block-buffered stage logs had the identical
  loss mode. Moved to the top of `cli.main`.
- The pure-deletion predicate (`drop` with no merge target) existed in five
  copies across three files — the donor-exclusion mistake the gate once made,
  waiting to be re-made. Now one `pure_deletions()` helper; `deletion_mass`
  returns one shape always; the amplitude advice renders in one place; a
  hand-edited `share: null` is refused by `validate_plan` instead of crashing
  the advisory warning.

Verified from scratch after the fixes: fresh-venv installs on the Mac (474
passed / 116 skipped) and GB10 (551/39 — torch is a core dep, so the CUDA tests
run even in a bare venv there), plus the GB10 fork venv with vLLM installed:
**588 passed / 2 skipped** (the two: optional `_orient_fused_weight` seam).

## Open

- ~~CPU expert co-execution on the **laptop** (experiment L6).~~ **Gated GO
  (2026-08-12): 3.7× at f=1.0, s=1.037, BW ratio 3.09** — see the gate entry
  above. Runtime implementation in progress, laptop-targeted, per the approved
  plan (fork-first provider views, `store/expert_cpu_exec.py`, loud opt-in).
- ~~Streaming load.~~ **Done and measured.** The tier's boot floor went 23.40 GiB →
  **11.35 GiB**, against 14.60 GiB untiered — the tier is now 22% below untiered
  instead of 60% above. Removed 12.05 GiB where the accounting predicted 12.0 GiB for
  the page-locked expert set. `stream_load: true` in the surgeon config; records are
  byte-identical to an offline `surgeon tier` build, so the two are interchangeable.
  Precise about what it is: on a **discrete** GPU it buys zero device bytes, since
  under the tier the full set was never device-resident. Deletion's floor was already
  closed: 1.33 GiB measured against 1.41 GiB predicted on Granite.
- ~~UVA view to escape `device_loading_context`.~~ **Decided against, with the reason
  recorded.** vLLM hoists every `cpu`-typed parameter to the device before
  `process_weights_after_loading`, so the port's deliberate `device="cpu"` allocation is
  device-resident by the time `build_provider` reads it. A UVA accelerator view reports
  `device.type == "cuda"` and would carry it through untouched —
  `get_accelerator_view_from_cpu_tensor` is public, so it is buildable out-of-tree. It
  is not worth it. The hoist is **per-module**, so it costs one layer's expert set in
  transient device memory: 0.75 GiB on OLMoE, 3.2% of the non-streaming floor, and
  **zero** on the streaming path (there is no cpu-typed parameter left to hoist).
  Against that, `get_accelerator_view_from_cpu_tensor` returns a *copy* rather than a
  view when the tensor is not pinned — the loader would fill the copy, the stashed host
  tensor would stay zeroed, and the store would receive 64 zeroed experts, which serve
  silently wrong output. 3.2% of a superseded path is not worth a silent-corruption
  mode. If it is ever revisited, the guards are `assert host.is_pinned()` and
  `assert view.data_ptr() == host.data_ptr()`.
  What was done instead: **streaming load is now the default whenever the disk tier is
  on**, so the expensive path is not reachable by leaving a flag unset. Non-act-and-mul
  layers, whose record layout streaming cannot express, fall back with a warning rather
  than failing.
- ~~Undiagnosed `ram_cache` anomaly.~~ **Diagnosed and closed.** `ram_cache 0` did not
  shrink the pool, it disabled the disk tier (`use_disk` requires `ram_cache > 0`),
  leaving the provider holding all 64 experts pinned for the process lifetime. Now
  refused outright, the log names the mode, and `BisectResult` records each arm's
  kwargs.
- ~~Merging is unexercised on real candidates.~~ **Exercised and measured, and the
  answer is no.** Lowering the threshold to 0.10 makes 233 of 384 removals merges
  (similarity 0.10–0.29). Against a delete-only control at the same budget: baseline
  10.3579, delete-only 12.7026 (1.226×), **merge 14.6689 (1.416×)**. Merging costs 15%
  more perplexity than deleting the same experts, because folding a weakly-similar
  expert into a survivor damages the survivor. The machinery is correct — alignment,
  averaging and router rewrite ran over 233 real clusters and the artifact loads and
  scores — the operation is just not worth doing on these models. The 0.85 default is
  now an empirical floor, not caution.
- **The gate cannot measure merges, and no longer pretends to.** Zeroing emulates
  deletion; a merge rewrites a *surviving* expert's weights. Donors were being swept
  into the zeroed set (they carry `action == "drop"` too), which measured a deletion
  the plan does not perform and then attached the verdict to unexamined merges. Donors
  are excluded, the verdict carries `merges_not_gated`, and `apply_plan` warns.
- ~~fp8 in the out-of-tree runtime.~~ **Done and verified (2026-08-10).**
  `compat/fp8_runtime.py` serves the fp8 disk tier out-of-tree, and on
  `RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8` (static per-tensor, shared experts,
  `first_k_dense_replace: 1`) it is **token-identical to the untiered fp8 baseline**
  across three greedy prompts, with the cache engaged on all 26 MoE layers (24/64
  slots, disk-backed). The route confirms the analysis: `Fp8MoEMethod` is not a
  `CustomOp`, so it goes through a `Fp8Config` subclass registered under `"fp8"` via
  `register_quantization_config` (not `register_oot`); and because the cache-install
  ordering lives inside `Fp8MoEMethod._setup_kernel`, the override **copies that
  method's body and swaps one line** — the fork-only `_maybe_init_expert_lru_cache`
  for `build_fp8_provider`, which repoints the per-expert scale parameters at the
  cache's slot buffers *before* the quant config captures them. Token identity is the
  proof that repointing is correct: a wrong per-expert scale is silent (no exception),
  and it would have diverged the tokens.
  **The cost, recorded honestly:** this multiplied the OOT surface. It pulls in
  `convert_to_fp8_moe_kernel_format`, `make_fp8_moe_kernel` and `Fp8MoeBackend` as
  three new internal seams (raising the internal-seam bound 14 -> 17), all optional —
  `install_fp8` declines gracefully, so an fp8-internals rename disables the fp8 tier
  but never fails the pin or the unquantized path. fp8 was never "one clean
  substitution" the way the unquantized path was; it is done, and its price is visible
  in the seam count.
  **Scope, guarded not assumed.** Only **per-tensor / per-channel static** fp8 is
  verified. `refuse_unverified_fp8_scheme` **refuses block-quantized** fp8
  (`weight_block_size` / `weight_scale_inv`) loudly — its per-expert *block* scale
  cannot be slot-indexed and would be mishandled silently — and warns on `dynamic`
  activation (lower risk: it rescales inputs, not the per-expert weight scales the
  cache remaps). A second check refuses block-shaped scales (`ndim > 2`) in
  `build_fp8_provider`. Confirmed the guard does not refuse the verified case: the
  static checkpoint still boots token-identical with the guard in place.
  fp8 tier performance (DeepSeek-Coder-V2-Lite-FP8, GB10, repeats=3, both eager):
  load **203 s vs 132 s untiered** (a loss — fp8 streaming is not ported, so the store
  is built from materialized weights), decode **17.0 vs 177.5 tok/s** (0.10x), peak
  device **25.0 vs 35.3 GiB** (−29%, from caching 24 of 64, not from fp8). So the fp8
  tier is a pure feasibility mechanism: it loses on both time axes and wins only device
  memory. fp8's own saving (half the disk / host / transfer bytes) does not show in a
  device-memory number.
  (Streaming load, which shared this bullet, is done — see above.)
- ~~`--enforce-eager` required (no CUDA-graph support).~~ **Done and verified
  (2026-08-11).** `compat/graph_runtime.py` pulls the last in-tree-only capability out
  of tree: the tier now runs under **piecewise CUDA graphs**. Two pieces, mirroring the
  fork. (1) A **config-time split**: the MoE op must be carved out of the captured
  region so the cache's dynamic `prepare()` runs eager. The fork does this in
  `VllmConfig.__post_init__` gated on its own `offload_config`; a plugin has neither
  that field nor the right to edit the config class, so `graph_runtime` **wraps
  `VllmConfig.__post_init__`** and appends `vllm::moe_forward(_shared)` to
  `splitting_ops` gated on `additional_config['surgeon']`. It works because the plugin
  entry point runs in `EngineArgs.__post_init__`, before any `VllmConfig` is built. (2)
  An **output-address stabilisation**: a `MoERunner` substitution (`register_oot`, since
  `MoERunner` is a `PluggableLayer`) copies the eager MoE output into a persistent
  per-shape buffer on piecewise passes, so the next captured graph piece reads a fixed
  address.
  **The load-bearing lesson, recorded because it cost a run:** injecting the split from
  the runner's `__init__` is **too late** — the split points are fixed before the runner
  is constructed, so the MoE op is captured and the boot aborts with `operation not
  permitted when stream is capturing`. The injection has to be config-time; that is why
  it wraps `__post_init__` rather than living in the runner.
  **Verified on GB10, OLMoE-1B-7B, greedy, four prompts, three controls.** (a) *tier,
  eager* is **token-identical to untiered eager on all four prompts** — the cache is
  numerically transparent, so this is the clean correctness anchor. (b) *tier, piecewise
  graphs* logged **2464 stabilised copies** (proof the piecewise path actually ran) and
  matched the eager tier on 3/4 prompts, diverging only in one greedy-unstable tail —
  where *untiered* graphs also diverge from *untiered* eager, so it is graph-vs-eager
  float noise, not the tier. (c) *tier, graphs, stabilisation forced off* produced
  **garbage — one token id repeated 24× per prompt** — the exact stale-address failure
  the copy prevents, which makes the stabilisation demonstrably load-bearing and
  correct. `--enforce-eager` is now optional; if `graph_runtime` cannot install
  (MoERunner moved), `validate()` reinstates the requirement rather than risk a silent
  capture-address bug.
  **The cost, recorded honestly:** the graph path is the second surface expansion (after
  fp8), raising the internal-seam bound 17 -> 28 — `MoERunner` + `_forward_impl`,
  `PluggableLayer.register_oot`, `splitting_ops`, `max_cudagraph_capture_size`,
  `CUDAGraphMode`, three `forward_context` names, and the config-time injection's
  `VllmConfig.__post_init__` + `CompilationConfig.mode` + `CompilationMode`. All
  optional and all degrade to requiring `--enforce-eager`, never a crash. With this the
  out-of-tree tier reproduces every capability of the in-tree prototype it replaces.
  **What it is worth, benchmarked (see "Release benchmark, 2026-08-11"): compatibility,
  not throughput.** Graphs buy the untiered baseline +3.8% (218.2 → 226.4 tok/s) and the
  tier **nothing** (55.8 → 55.2, inside σ=0.044) — as designed, since the split exists
  precisely to keep the MoE op eager, and that op is what dominates a tiered step. They
  also cost ~22 s of capture at load (13.0 → 35.0 s) and real device memory, quantified
  on the 4 GiB laptop GPU: eager runs at `gpu_memory_utilization` 0.83 (2.69 GiB peak,
  6.6 s load) where graphs cannot allocate a single KV block and vLLM refuses at boot
  loudly; at 0.90 graphs do fit (2.71 GiB peak, 29.4 s load). So graphs cost ≈7 points of
  utilisation (≈290 MiB) and 4.5× load time on that card — a headroom requirement, not an
  impossibility. (An earlier draft of this entry claimed graphs simply did not fit on
  that card; the 0.90 arm was run precisely to check that claim and refuted it. The
  decode difference at 0.90, 7.8 → 8.3 tok/s, came from a single unrepeated run and is
  below that measurement's resolution, so nothing is claimed from it.) The win is that
  the tier now composes with vLLM's default serving mode; `--enforce-eager` survives as a
  tuning knob for memory-tight or load-latency-sensitive deployments rather than a hard
  requirement.
- ~~Router least-squares refit.~~ **Settled, and replaced by something better.** A row
  refit provably cannot help pure deletion: softmax over survivors is exactly Bayes
  conditioning, so the unchanged rows already minimise divergence from the teacher's
  conditional routing at zero loss, on any corpus. What deletion *does* leave is an
  amplitude error — surviving gates inflated by `1/(1 - P_D(x))` under
  `renormalize=False` — and a softmax cannot carry a uniform amplitude, so the fix goes
  in `down_proj`, where the output is linear. Measured on OLMoE pruned to 40 experts,
  fresh loads: gsm8k 11.8754 → **10.5306** (60% of the deletion damage removed),
  hellaswag 34.5260 → **29.0684** (55%), the second corpus not fitted on. `surgeon
  calibrate` finds the scalar, `surgeon apply --amplitude` folds it in.
  The refit for *merged* clusters is now **built as machinery** (`surgery/refit.py`),
  tested exactly on synthetic clusters, though it stays moot in practice: no screened
  family has a mergeable pair, and merging is dominated by deleting. The principled
  form is the **log-sum-exp least squares**: a softmax's mass on a cluster `C` is
  `exp(logsumexp_{e in C} l_e)/Z`, so the survivor row that reproduces that marginal
  is `argmin_w ||X w - logsumexp(X W_C.T)||` on calibration hidden states. The tests
  pin the two facts that matter: a single-expert "cluster" is fit exactly (its target
  is linear), and a real multi-expert cluster is **irreducibly lossy** — the
  `~log(k)` constant offset has no bias term to live in, which is the same
  no-bias obstruction the amplitude fix works around and the reason merging costs more
  than deleting. So the machinery exists and is correct; it is not wired into `apply`
  because there is nothing to wire it for.
