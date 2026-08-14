# Configuration

Every setting, its default, and what it costs. Two channels reach the runtime:
`--additional-config '{"surgeon": {...}}'` on `vllm serve`, and `VLLM_MOE_*`
environment variables. The config payload wins where both are set.

## Serving settings

Passed as `--additional-config '{"surgeon": {...}}'`.

| key | default | what it does |
|---|---|---|
| `expert_cache_size` | `0` (off) | GPU cache slots per layer. **The one knob that matters** — size it to the batch's per-layer expert union, `max_num_seqs × top_k` capped at the expert count. |
| `store_dir` | none | Where the NVMe store lives. Built on first boot, reused after. |
| `ram_cache` | `0` | Host-RAM slots per layer. Set at or above the expert count, or every eviction reads disk. |
| `fp8_store` | `false` | Store records as row-scaled fp8. Halves disk, host RAM and transfer. Costs ~1.11× decode and bit-exactness. Must match between build and serve. |
| `split` | `"token"` | `"expert"` splits a layer's expert set into cache-sized groups, which is the only way to run below `top_k` slots. Not bit-exact. |
| `stream_load` | on with a store | Write experts into the store as the checkpoint arrives, so the full tensor never materialises. Leave it on. |
| `hot_experts` | none | Residency prior from `surgeon tier`. **Requires `VLLM_MOE_CACHE_POLICY=ewma`** — the default `lfru` policy has no prior table and discards the hint with one warning. See the caveat below the table. |
| `cpu_experts` | `false` | Compute cold experts on the host instead of fetching them. **Discrete cards only** — see below. |
| `cpu_expert_min_tokens` | `1` | Experts below this routed-token count stay on the GPU path. Single-token experts are padded inside the kernel, so 1 is correct; raising it shrinks the candidate set. |
| `cpu_expert_threads` | `0` | torch intra-op threads for the host GEMMs. `0` leaves the pool alone. 12 measured best on an i7-12700H. |

**The `hot_experts` caveat, because the number above needs it.** The `+~10
points at N=50` figure comes from a single-tier `store/replay.py` run (pure
numpy) whose warm arm used the replay's **prewarm placement** — experts placed
resident before the first token. The shipped runtime seeds only the prior
*bias*; it does not place, so a real warm start realises less than the table
shows. And +10 points at N=50 is about 5 cache hits, binomial SE ≈ 6.6 points,
with no repeats. Treat it as indicative, and note that the policy (`ewma` over
`lfru`, +2.3 to +4.0 on the same window) is worth more than the prior.

### Environment variables

`VLLM_MOE_EXPERT_CACHE_SIZE`, `VLLM_MOE_DISK_STORE_DIR`, `VLLM_MOE_RAM_CACHE`,
`VLLM_MOE_DISK_STORE_FP8`, `VLLM_MOE_EXPERT_CACHE_SPLIT`,
`VLLM_MOE_STREAM_LOAD`, `VLLM_MOE_HOT_EXPERTS`, `VLLM_MOE_CPU_EXPERTS`,
`VLLM_MOE_CPU_EXPERT_MIN_TOKENS`, `VLLM_MOE_CPU_EXPERT_THREADS` mirror the keys
above. These are store-side only and have no config-payload equivalent:

| variable | default | what it does |
|---|---|---|
| `VLLM_MOE_CACHE_POLICY` | `lfru` | `ewma` is worth ~1–4 points of hit rate; opt-in. Required for `hot_experts` to do anything. |
| `VLLM_MOE_CACHE_DECAY` | `0.999` | EWMA decay. |
| `VLLM_MOE_DISK_PIPELINE` | `true` | Route disk reads through the reader pool. |
| `VLLM_MOE_DISK_IO_THREADS` | `2` | Reader threads, clamped to [1, 4]. Past 4, p99 read latency degrades badly. |
| `VLLM_MOE_DISK_PREFETCH` | `false` | Cross-group prefetch. Needs `ram_cache ≥ 2 × expert_cache_size`; worth 1.03–1.075×. |
| `VLLM_MOE_ZERO_COPY` | `false` | Map the pinned pool as the kernel's buffer. Needs a disk store; incompatible with prefetch and with `cpu_experts`. |
| `VLLM_MOE_ZC_FP8_SLOTS` | `0` | Zero-copy over an fp8 store only: retain this many fp8 rows as a cold pool, so a later miss re-expands a row (0.134 ms) instead of reading disk (0.95 ms). Warns on use — validated in simulation and on GB10, **never tested on the small unified boxes it targets**. |
| `VLLM_MOE_ROUTING_TRACE` | none | Write a routing trace for offline replay. |
| `VLLM_MOE_RECORD_STATS` | `false` | Per-layer hit/miss and timing counters. |
| `VLLM_MOE_RECORD_COOC` | `false` | Also accumulate the per-layer expert co-occurrence matrix. `surgeon plan` uses it for one thing: vetoing a merge between two experts that fire together (`--max-cooccurrence`). Without it that veto is inert. |
| `VLLM_MOE_SURGEON_CACHE` | `~/.cache/moe-surgeon/autoconfig` | Where `surgeon autoconfig` caches its per-machine answer. |

## Sizing rules

Three rules decide nearly everything. `surgeon autoconfig` applies them and
caches the answer per machine.

| setting | rule | measured |
|---|---|---|
| `expert_cache_size` | cover the batch's per-layer expert union, not free VRAM | 24 → 48 slots: **2.67×** decode |
| `ram_cache` | at or above the expert count | 48 → 64: **1.66×** decode |
| `fp8_store` | only when the store or host RAM does not otherwise fit | 1.11× slower, not bit-exact |

Cache slots are paid for out of KV cache, not out of free device memory:
capacity buys throughput and spends context.

`surgeon autoconfig` also scales the KV reserve to the card (15% of the bucketed
free VRAM, clamped to [0.5, 2.0] GiB) rather than assuming a flat 2 GiB, which
starved small cards. `--kv-reserve` overrides it.

## CPU co-execution

`cpu_experts` computes cold experts on the host instead of fetching them over
the link, and joins the result in fp32.

**Enable it only where host DRAM reads and device transfers are separate
bandwidth pools** — that is, on a discrete GPU. Measured `BW_cpu_gemm / BW_h2d`
is 3.09 on an RTX 3050 Ti laptop and 0.78 on GB10's unified memory, and the
same code measures 1.58× on the first and 0.719× on the second.

It engages only on decode-shaped forwards whose expert union exceeds
`expert_cache_size`; otherwise it stands aside. `/stats` reports
`CPU co-exec: N expert forwards` with the per-expert host cost, so you can
confirm it ran.

Not bit-exact: the host GEMM's reduction order differs from the fused kernel's.
The runtime warns at boot.

## What is refused

These raise at boot rather than producing wrong output:

| refused | why |
|---|---|
| tensor parallelism | the store identity has no rank component, so ranks would share one store file |
| expert / data / sequence parallelism | not implemented |
| MoE layers with bias | not implemented |
| block-quantised fp8 | per-block scales cannot be slot-indexed |
| vLLM's in-tree expert cache at the same time | both would own the same weights |
| CUDA graphs without the MoE split | a captured graph replays against a stale workspace address |
| `ram_cache` below `expert_cache_size` | the warm tier would be smaller than the resident set it feeds |
| `store_dir` with `ram_cache: 0` | asks for the disk tier and then disables it |
| `cpu_experts` + fp8 store or checkpoint | the host path has no dequantizing twin |
| `cpu_experts` + zero-copy | the pool is already the kernel's buffer |
| `cpu_experts` + CUDA graphs | a captured graph replays without the host |
| `cpu_experts` + non-silu activation, or router weights on input | no verified host equivalent |

Deleting experts is refused unless a `surgeon gate` verdict is on the plan and
its recorded drop-set digest matches the plan's current deletions. `--skip-gate`
overrides.

## Merging, and why it is off by default

`surgeon plan --checkpoint <dir>` enables the fourth placement: folding an
expert into a sufficiently similar survivor instead of deleting it.

| flag | default | what it does |
|---|---|---|
| `--checkpoint` | none | Without it there is no similarity, so no merges are considered at all. |
| `--merge-threshold` | `0.85` | Minimum permutation-invariant subspace similarity for a pair to merge. |
| `--max-cooccurrence` | `0.10` | Refuse to merge two experts that fire together often; folding them would lose a distinction the router uses. |

**No real model has cleared the bar.** Maximum measured pairwise similarity is
0.37 on OLMoE, 0.40 on Qwen3-30B-A3B and 0.33 on DeepSeek-V2-Lite — three
families, none with a natural merge candidate. The 0.85 default is therefore
an empirical floor rather than caution: forced down to 0.10 on OLMoE, merging
233 experts and deleting 151 measured **14.6689 perplexity (1.416×)** against
**12.7026 (1.226×)** for deleting all 384 outright. Merging a weakly-similar
expert damages the survivor that was carrying its own function, so below the
threshold it is worse than the thing it is meant to improve on.

**A gate verdict says nothing about a merge.** The gate zeroes experts, which
emulates deletion pessimistically and does not model folding a donor's weights
into a survivor at all — so a plan with merges can pass a gate that never
examined the part most likely to hurt. `surgeon gate` reports this rather than
implying coverage.

## CUDA graphs

Supported. `--enforce-eager` is not required: the runtime carves the MoE op out
of the captured region at config time, so the cache's host code runs eager
between graph pieces. Worth +3.8% on the untiered baseline and ~0 for the tier.

`cpu_experts` is incompatible and refused.

## Surviving a vLLM upgrade

**Supported range: `vllm>=0.26.0,<0.28`.** The package holds a declared set of
vLLM internals. Check them against a new version before upgrading:

```bash
surgeon seams                      # against the installed vLLM
surgeon seams --source /path/to/vllm-checkout   # against a source tree, no GPU
```

Optional seams degrade rather than fail: if the fp8 internals move, the fp8
tier declines; if the graph internals move, `--enforce-eager` is required
again.

To see *what* moved rather than only that something did,
[`tools/upstream_drift.py`](../tools/upstream_drift.py) extracts the seam-named
modules at two git refs of a vLLM checkout and prints the bill as a table. It
reads blobs out of git, so it needs no install, no torch and no GPU, and it
exits 1 when a required seam broke — enough to gate a pin bump in CI.

**A clean report is necessary, not sufficient.** The check parses names and
signatures, **not behaviour**, so it cannot see a semantic change behind an
unchanged signature. A green parser is not a reason to move the ceiling.

**What moved it to `<0.28`.** vLLM 0.27.1 was installed from PyPI — stock, not
the fork — into its own environment on both machines, and the package was run
on it:

| check | result |
|---|---|
| seams, against the **installed** package | 33 seams, 0 required broken |
| plugin entry point | `moe_surgeon` loads |
| test suite | 633 passed / 1 skipped (GB10) |
| **tier vs untiered token identity** | **one sha256 across all six runs** |

The last row is the one that counts. OLMoE-1B-7B, greedy, 4 prompts × 128
tokens, `expert_cache_size` 48 with `ram_cache` 64 on the disk store, against a
plain untiered boot — 3 processes per arm, one arm per process, and every run
returned `c08aa685…`. Both arms held `gpu_memory_utilization` at 0.42: a token
hash is only comparable within one memory configuration, because a different KV
pool changes the batch composition and with it the reduction order.

This is also the first time the out-of-tree premise — that this runs on stock
vLLM, not only on the fork it grew out of — has been demonstrated by running it.
The earlier evidence was a symbol-by-symbol check against a source tree, which
establishes that the names are there and nothing about what they do.

Note the numbers in [benchmarks.md](benchmarks.md) were still taken against a
0.26.1-dev fork. The range says the package works on 0.27.1; it does not claim
the throughput tables were re-measured there.

## The job server

```bash
surgeon serve --port 8300 --state ./surgeon-state
```

Binds loopback by default. A non-loopback host requires `--token` (or
`$MOE_SURGEON_TOKEN`), sent as `X-Surgeon-Token`; `/health` stays open. Engine
kwargs in a request are allow-listed, and a stage never fetches an uncached
model unless the job sets `allow_download`.

```bash
curl -sX POST localhost:8300/jobs -H 'content-type: application/json' -d '{
  "model": "allenai/OLMoE-1B-7B-0924",
  "corpus": "domain.jsonl", "heldout": "heldout.jsonl",
  "core_experts": 24, "stages": ["profile", "plan", "gate", "tier"]
}'
```

Stage order is `headroom → profile → recommend → plan → gate → apply → tier`;
requested stages are run in that order regardless of how they were listed. The
default pipeline is `profile → plan → gate → tier`, which never writes a pruned
checkpoint — deletion is opt-in.

`headroom` needs `heldout` plus a `candidates` list. Every stage is a `surgeon`
subcommand and the job record stores each stage's argv, so any failed run is
reproducible by hand.
