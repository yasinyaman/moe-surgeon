# Sizing the tier

The measured rules, the automatic sizing that applies them, and the boot-floor
instrument that answers "will it run at all". This is where the 2.59x and
1.66x figures the README quotes come from.

## Sizing it automatically

`surgeon autoconfig` reads the checkpoint's geometry from headers, measures free
device memory, host RAM and disk, applies the rules below, and prints a command you
can paste:

```bash
surgeon autoconfig --checkpoint /path/to/model --max-num-seqs 8
```

```
probed (13c48f4333d97fc8): NVIDIA GB10, 41.2 GiB free VRAM, 122 GiB host RAM, 180 GiB free disk

  - a batch of 8 at top_k=8 can route to at most 64 experts per layer, so that is the capacity to aim for
  - memory fits 64 slots, at or above the bound, so capacity is 64 and no step should have to re-fetch
  - host RAM holds all 64 experts (12.0 GiB), so nothing spills to disk in steady state

serve with:
  vllm serve /path/to/model --max-num-seqs 8 --additional-config '{"surgeon": {"expert_cache_size": 64, "store_dir": "./store", "ram_cache": 64}}'
```

`--start` runs that command instead of printing it, and `--json` emits just the
config for a deployment script. **The answer is cached**, keyed on the checkpoint,
the serving batch and the machine's resources — so an unchanged deployment does not
re-probe on every boot. The resource figures are bucketed before they reach the key,
because free VRAM moves by a few MiB between two reads and an exact key would never
hit. `--refresh` forces a re-probe.

It states its reasoning line by line, and it says what it could not measure rather
than defaulting silently: a probe that reported zero free VRAM would quietly size the
tier for a machine with no memory. Where the numbers do not allow a good answer it
says so — a capacity below `top_k` comes with the warning that the expert split is
not bit-exact, and a target that cannot fit at all is an error naming the shortfall.

## Sizing it by hand: the only knob that matters, and what it costs

Everything else in these documents is a mechanism. This is the number to get right, and it
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
loses on one axis is kept if it wins on another. [DECISIONS.md](../DECISIONS.md) is the
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
[DECISIONS.md](../DECISIONS.md#the-capacity-sweep-and-the-finding-that-dominates-everything-else-here).

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
