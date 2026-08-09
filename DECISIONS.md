# Decision register

Every method here is judged on **three axes**, never one:

| axis | what it asks | how it is measured |
|---|---|---|
| **feasibility** | does it run, and on how small a device | `surgeon budget` (validated arithmetic) + measured load time |
| **speed** | tokens per second | `compat/bench.py`, short prompts / long generations |
| **accuracy** | held-out perplexity | `compat/ablation.py`, the same metric the gate uses |

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
| **disk tier** | **strong win**: runs a 12.9 GiB model with a 2.39 GiB floor; halves load time | **loses**: 0.38× decode | neutral: 1.003× | **keep** — the only method that changes what is possible at all |
| **pruning (rank-1 ranked)** | win: 34% smaller artifact and store; shrinks the tier's candidate set | neutral alone (1.01×); **win in composition** (1.32× over tier alone) | loses: 1.249× | **keep** — its speed win only appears *with* the tier |
| **rank-1 importance** vs token count | n/a | n/a | **strong win**: 1.57× → 1.25× | **keep, as default** |
| **permutation-aligned merging** | n/a | n/a | not exercised: no mergeable pair in OLMoE (max similarity 0.37 of 64512) or Qwen3-30B (0.401 of 81280) | **keep the machinery** — tested exactly, and redundancy is a per-model property; absent here is not absent everywhere |
| **residency prior** (`hot_experts.json`) | **win, narrow**: +10.0 pts hit rate over the first 50 accesses, decaying to 0 by 2000 | negligible in steady state | n/a | **keep** — free (a byproduct of the plan) and confined to cold start; do not oversell |
| **EWMA cache policy** vs LFRU | win: +2.3…+4.0 pts on the cold-start window | win: +1.2 pts sustained hit rate | n/a | **keep** — one env var, wins on two axes |
| **fp8 store** | **loses on VRAM**: none saved, the provider dequantizes into model-dtype slots | win: halves disk, host RAM and transfer bytes | small loss: quantization error, no argmax moved in testing | **keep** — the axis it wins on is not the one people expect |
| **static gate geometry** (row norm, cosine) | n/a | n/a | **loses**: ρ ≈ 0.00 against measured load and co-occurrence | **drop as a selection signal** — but see below |
| **static byte accounting** (`surgeon budget`) | **strong win**: answers "least VRAM" in 1.5 s with no GPU | n/a | n/a | **keep** — the same static analysis, pointed at sizing instead of selection |
| **`e_score_correction_bias`** | n/a | n/a | untested: absent from every accessible small MoE | **keep detection** — principled (a *learned* balancing term), just unavailable here |
| **cross-domain profile** as a cold-start prior | win: ρ +0.43, top-24 overlap 54% vs 37.5% random | n/a | n/a | **keep** — dominates static analysis, and profiling costs minutes |
| **block dedup across experts** | n/a | n/a | n/a | **drop** — untested, and near-orthogonality makes duplicate blocks implausible in trained bf16 experts |
| **correlation-driven disk layout** | n/a | no headroom: 6.0 MiB records already reach 93% of the NVMe ceiling on one thread | n/a | **drop** — the lever is already banked by the record-stride design |

The two static-signal rows are the clearest case for judging per axis. Gate
geometry is worthless for *choosing* experts and genuinely useful for *sizing* them
— the same analysis, kept or dropped depending on which question it is asked.

## Choosing a strategy per target and per model

Different targets and different models want different subsets. The properties that
decide are measurable before committing:

| if the target… | then |
|---|---|
| has less VRAM than the bit-exact floor (`surgeon budget --vram`) | the tier is mandatory, not optional |
| is latency-sensitive and VRAM-rich | skip the tier; pruning alone costs no throughput |
| is VRAM-constrained *and* latency-sensitive | prune **and** tier — the composition recovers 1.32× over tier alone |
| restarts often | seed the residency prior and set `VLLM_MOE_CACHE_POLICY=ewma` |

| if the model… | then |
|---|---|
| has a genuine dead tail (coldest expert ≪ uniform) | deletion is cheap; measure with `surgeon gate` |
| has no dead tail (OLMoE: coldest is 0.5% vs 1.56% uniform) | prefer the tier; delete only under a gate |
| has mergeable pairs (similarity ≥ merge threshold) | merge instead of deleting — no capacity lost |
| carries `e_score_correction_bias` | a static popularity prior is available with no profiling |
| has shared/always-on experts | they are fixed VRAM cost, never cached — `surgeon budget` accounts for them separately |
| ships pre-stacked expert tensors (IBM Granite) | reading is supported (inspect, budget, tier); `apply` refuses to write that layout back |

`surgeon inspect` and `surgeon budget` between them report every property in the
second table without a GPU, and **`surgeon recommend` applies both tables**:

```bash
surgeon recommend --checkpoint model --vram 3.68 --profile p.npz --similarity-cache sim.npz
```

On OLMoE against a 3.68 GiB target it reproduces this session's conclusions from the
measurements alone — tier mandatory at capacity 10, delete nothing (no dead tail),
merge nothing (0.370 < 0.85), seed the prior, use EWMA — and lists what it could not
decide rather than guessing. Deletion quality is always deferred to `surgeon gate`.

## Open

- ~~Streaming load.~~ **Done and measured.** The tier's boot floor went 23.40 GiB →
  **11.35 GiB**, against 14.60 GiB untiered — the tier is now 22% below untiered
  instead of 60% above. Removed 12.05 GiB where the accounting predicted 12.0 GiB for
  the page-locked expert set. `stream_load: true` in the surgeon config; records are
  byte-identical to an offline `surgeon tier` build, so the two are interchangeable.
  Precise about what it is: on a **discrete** GPU it buys zero device bytes, since
  under the tier the full set was never device-resident. Deletion's floor was already
  closed: 1.33 GiB measured against 1.41 GiB predicted on Granite.
- **`device_loading_context` defeats the host allocation** on the non-streaming path.
  Second-order now that streaming load is in (with zero-expert placeholders there is
  nothing to hoist), but it still applies whenever `stream_load` is off. vLLM hoists
  every
  `cpu`-typed parameter to the device before `process_weights_after_loading`, so the
  port's deliberate `device="cpu"` allocation is device-resident by the time
  `build_provider` reads it (per-module: ~768 MiB at a time, not 12 GiB). A UVA
  accelerator view reports `device.type == "cuda"` and is how vLLM's own offloader
  carries host params through untouched — `get_accelerator_view_from_cpu_tensor` is
  public, so this is fixable out-of-tree. Second-order once streaming load lands.
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
- **fp8 and streaming load in the out-of-tree runtime.** `compat/runtime.py` covers
  the unquantized path with `--enforce-eager`; fp8 needs `Fp8MoEMethod` substituted
  with the cache installed before the quant config captures scale tensors.
- ~~Router least-squares refit.~~ **Settled, and replaced by something better.** A row
  refit provably cannot help pure deletion: softmax over survivors is exactly Bayes
  conditioning, so the unchanged rows already minimise divergence from the teacher's
  conditional routing at zero loss, on any corpus. What deletion *does* leave is an
  amplitude error — surviving gates inflated by `1/(1 - P_D(x))` under
  `renormalize=False` — and a softmax cannot carry a uniform amplitude, so the fix goes
  in `down_proj`, where the output is linear. Measured on OLMoE pruned to 40 experts,
  fresh loads: gsm8k 11.8754 → **10.5306** (60% of the deletion damage removed),
  hellaswag 34.5260 → **29.0684** (55%), the second corpus not fitted on. `surgeon
  calibrate` finds the scalar, `surgeon apply --amplitude` folds it in. The refit for
  *merged* clusters remains unbuilt and is now moot: merging is worse than deleting.
