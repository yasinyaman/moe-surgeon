# Serving from the tier

How the out-of-tree runtime works, what it is verified against, and what
running a conversation through it looks like. The numbers here are the
evidence behind the README's claims.

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
see [Piecewise CUDA graphs](sizing.md#piecewise-cuda-graphs-the-last-in-tree-only-capability).
Streaming the checkpoint into the store **is** ported and on by default —
see [Streaming load](sizing.md#streaming-load-the-tier-now-wins-on-feasibility-too). **fp8
checkpoints are served too** (`compat/fp8_runtime.py`): verified token-identical to the
untiered fp8 baseline on `DeepSeek-Coder-V2-Lite-Instruct-FP8`. fp8 could not be the
unquantized path's one clean substitution — `Fp8MoEMethod` is not a `CustomOp`, so it
goes through a `Fp8Config` shadowing the `"fp8"` config, and the override copies
`_setup_kernel` to install the cache before the quant config captures the scales — so it
costs three more (optional) internal seams, the price recorded in
[DECISIONS.md](../DECISIONS.md).

## Trying it interactively

```bash
surgeon run allenai/OLMoE-1B-7B-0924 --checkpoint /path/to/model
```

A prompt, like `ollama run` — with the part that is actually worth adding: every
turn says what the tier cost.

```
>>> explain a b-tree in two sentences
A B-tree is a self-balancing search tree ...
  [128 tok, 142.2 tok/s, cache 93% (15302/16384)]
```

`/stats` summarises the session, and names a misconfiguration when the counters show
one rather than leaving it as "it feels slow":

```
  GPU cache : 180 hits / 620 misses (22.5% hit rate)
  ! under half of expert lookups hit the GPU cache. Raising expert_cache_size toward
    the batch's per-layer union is the lever measured at 2.59x; `surgeon autoconfig`
    sizes it.
```

It boots the configuration `surgeon autoconfig` picked, so there is one implementation
of the sizing rules rather than two. `--no-tier` serves untiered for a side-by-side.

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
**Cross-expert block dedup** — untested, but the similarity results ([surgery.md](surgery.md#measured-olmoe-has-no-mergeable-experts)) argue against
finding duplicate blocks in distinct trained bf16 experts.
