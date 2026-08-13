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
| `hot_experts` | none | Residency prior from `surgeon tier`. Worth ~10 points of hit rate over the first 50 accesses, nothing after. |
| `cpu_experts` | `false` | Compute cold experts on the host instead of fetching them. **Discrete cards only** — see below. |
| `cpu_expert_min_tokens` | `1` | Experts below this routed-token count stay on the GPU path. Single-token experts are padded inside the kernel, so 1 is correct; raising it shrinks the candidate set. |
| `cpu_expert_threads` | `0` | torch intra-op threads for the host GEMMs. `0` leaves the pool alone. 12 measured best on an i7-12700H. |

### Environment variables

`VLLM_MOE_EXPERT_CACHE_SIZE`, `VLLM_MOE_DISK_STORE_DIR`, `VLLM_MOE_RAM_CACHE`,
`VLLM_MOE_DISK_STORE_FP8`, `VLLM_MOE_EXPERT_CACHE_SPLIT`,
`VLLM_MOE_STREAM_LOAD`, `VLLM_MOE_HOT_EXPERTS`, `VLLM_MOE_CPU_EXPERTS`,
`VLLM_MOE_CPU_EXPERT_MIN_TOKENS`, `VLLM_MOE_CPU_EXPERT_THREADS` mirror the keys
above. These are store-side only and have no config-payload equivalent:

| variable | default | what it does |
|---|---|---|
| `VLLM_MOE_CACHE_POLICY` | `lfru` | `ewma` is worth ~1–4 points of hit rate; opt-in. |
| `VLLM_MOE_CACHE_DECAY` | `0.999` | EWMA decay. |
| `VLLM_MOE_DISK_PIPELINE` | `true` | Route disk reads through the reader pool. |
| `VLLM_MOE_DISK_IO_THREADS` | `2` | Reader threads, clamped to [1, 4]. Past 4, p99 read latency degrades badly. |
| `VLLM_MOE_DISK_PREFETCH` | `false` | Cross-group prefetch. Needs `ram_cache ≥ 2 × expert_cache_size`; worth 1.03–1.075×. |
| `VLLM_MOE_ZERO_COPY` | `false` | Map the pinned pool as the kernel's buffer. Needs a disk store; incompatible with prefetch and with `cpu_experts`. |
| `VLLM_MOE_ROUTING_TRACE` | none | Write a routing trace for offline replay. |
| `VLLM_MOE_RECORD_STATS` / `_COOC` | `false` | Extra counters. |

## Sizing rules

Three rules decide nearly everything. `surgeon autoconfig` applies them and
caches the answer per machine.

| setting | rule | measured |
|---|---|---|
| `expert_cache_size` | cover the batch's per-layer expert union, not free VRAM | 24 → 48 slots: **2.67×** decode |
| `ram_cache` | at or above the expert count | 48 → 64: **1.66×**, and the run-to-run spread collapses |
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

## CUDA graphs

Supported. `--enforce-eager` is not required: the runtime carves the MoE op out
of the captured region at config time, so the cache's host code runs eager
between graph pieces. Worth +3.8% on the untiered baseline and ~0 for the tier.

`cpu_experts` is incompatible and refused.

## Surviving a vLLM upgrade

The package holds a declared set of vLLM internals. Check them against a new
version before upgrading:

```bash
surgeon seams                      # against the installed vLLM
surgeon seams --source /path/to/vllm-checkout   # against a source tree, no GPU
```

Optional seams degrade rather than fail: if the fp8 internals move, the fp8
tier declines; if the graph internals move, `--enforce-eager` is required
again.

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
