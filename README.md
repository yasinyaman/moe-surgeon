# moe-surgeon

Fit a domain-specialised MoE deployment into less memory: an NVMe expert tier,
with permanent offline pruning and merging where a model supports it.

A general MoE model carries experts for every domain it was trained on. A
deployment that only ever sees one domain pays for all of them. This package
measures which experts that deployment actually uses, then makes it cheaper to
serve — primarily by tiering cold experts to NVMe, and, where the measurements
justify it, by dropping the genuinely unused ones and merging redundant ones into
a smaller checkpoint.

**On the models measured here (OLMoE, Qwen3-30B) the disk tier is the primary
mechanism, not pruning** — they have no dead tail and no mergeable pairs, so
deletion costs quality (measured below) and exists to *support* the tier: a
smaller candidate set and store. Redundancy is a per-model property, so the
pruning and merging machinery is kept and tested; it is simply not the win on
these two families. See [DECISIONS.md](DECISIONS.md).

## Why use it

Because it changes what a machine can serve, and it proves what it costs. All
numbers below are measured, on hardware, and reproduced in this README:

- **A 12.9 GiB model on a 4 GiB laptop GPU.** The tier serves OLMoE-1B-7B at
  **2.69 GiB peak** (7.7 tok/s) on an RTX 3050 Ti that cannot even load the model
  untiered — the machine swaps itself unreachable trying. This is the regime the
  package exists for: the checkpoint no longer has to fit.
- **~8× faster load.** Expert weights stream into the store instead of being staged
  onto the device: 13–15 s against 118 s untiered on the same box. If your workflow
  boots models often, this alone may be the reason.
- **Bit-exact.** A correctly configured tier reproduces the untiered token stream
  exactly — verified by hashing seed-pinned greedy runs, nine for nine, including
  the zero-copy path — and runs under vLLM's default CUDA-graph serving mode.
- **The cost is stated, not hidden: decode.** At the measured-correct size the tier
  decodes at 0.66× untiered (143 vs 218 tok/s on GB10). It is a feasibility and
  load-time mechanism, not a throughput one. If the model already fits comfortably
  and load time does not matter, you do not need this package.
- **No fork.** A `pip install` and a `vllm.general_plugins` entry point on stock
  vLLM. Every internal it borrows is declared in [seams](src/vllm_moe_surgeon/compat/seams.py)
  and checked by CI weekly against vLLM's latest release — at last check, one full
  minor version past the pin, zero required seams broken.
- **It sizes itself.** The two misconfigurations worth 2.6× and 1.7× were found by
  measurement; `surgeon autoconfig` applies those rules to your machine so you do
  not rediscover them the slow way.

## Quick start

Not on PyPI yet — install from the repository:

```bash
pip install git+https://github.com/yasinyaman/moe-surgeon.git
```

Then either talk to a model interactively, with the tier's cost printed per turn:

```bash
surgeon run allenai/OLMoE-1B-7B-0924 --checkpoint /path/to/checkpoint
```

or size the tier for this machine and serve:

```bash
surgeon autoconfig --checkpoint /path/to/checkpoint --start
```

Both probe the machine once, cache the decision, and boot vLLM with an
`--additional-config '{"surgeon": {...}}'` — the tier is a plugin, so it is `vllm
serve` underneath and every other vLLM flag works as usual. The store builds itself
on the first boot (streamed, so the full expert tensors never materialise) and is
reused afterwards.

The checkpoint must be a **local directory** for sizing (the analysis reads
safetensors headers); pass a repo id as the model and `--checkpoint` for the local
copy if they differ.

### The commands

| stage | command | needs |
|---|---|---|
| serve, interactively | `surgeon run` | GPU |
| size + serve | `surgeon autoconfig [--start]` | nothing |
| "will it fit at all" | `surgeon budget` / `surgeon vram-floor` | nothing / GPU |
| what does this deployment use | `surgeon profile` | GPU |
| inspect routing signals | `surgeon inspect` | nothing |
| choose a strategy | `surgeon recommend` | nothing |
| prune (last resort) | `surgeon plan` → `gate` → `apply` | GPU for gate |
| build the store offline | `surgeon tier` | nothing |
| measure expert contribution | `surgeon ablate` / `surgeon calibrate` | GPU |
| before a vLLM upgrade | `surgeon seams` | nothing |
| pipeline over HTTP | `surgeon serve` | GPU for the stages it runs |

Pruning is deliberately not in the quick start: on the families measured here it
costs downstream-task accuracy the perplexity gate cannot see (arc_challenge −25%
relative), so `apply` refuses to delete without a measured, plan-bound gate verdict.
The tier keeps every expert and costs none.

## Sizing it

Three measured rules decide nearly everything; `surgeon autoconfig` applies them
to your machine and caches the answer:

| setting | rule | measured |
|---|---|---|
| `expert_cache_size` | cover the serving batch's per-layer expert union, not free VRAM | 24 → 48 slots: **2.59×** decode |
| `ram_cache` | ≥ the expert count, or every eviction reads disk | 48 → 64: **1.66×** |
| `fp8_store` | space mechanism only — on when the store/RAM does not otherwise fit | 1.11× slower, not bit-exact |

The slots are paid for out of KV cache, not free device memory — capacity buys
throughput and spends context. The full evidence, including the boot-floor
instrument and the capacity sweep, is in [docs/sizing.md](docs/sizing.md).

## Documentation

| doc | what it covers |
|---|---|
| [docs/serving.md](docs/serving.md) | the out-of-tree runtime, fp8, the interactive prompt, cold start |
| [docs/sizing.md](docs/sizing.md) | the measured sizing rules, autoconfig, boot floors, streaming load, CUDA graphs, the capacity sweep |
| [docs/surgery.md](docs/surgery.md) | profiling, planning, pruning/merging, the amplitude fix, the quality gate |
| [docs/architecture.md](docs/architecture.md) | the layering rule, the seam table, what a vLLM upgrade costs |
| [docs/server.md](docs/server.md) | the HTTP job pipeline |
| [DECISIONS.md](DECISIONS.md) | the decision register: per-axis verdicts and every benchmark behind them |

## Status

Published, CI-checked, and verified on two machines. The out-of-tree runtime is
token-identical to the in-tree implementation it replaces and to untiered vLLM;
the full `profile → plan → gate → tier` pipeline has run end-to-end through the
job server from nothing but a repository id; and all three axes are measured —
including the tier's own costs, stated above rather than discovered later.

- Serving: unquantized and fp8 checkpoints; piecewise CUDA graphs (no
  `--enforce-eager`); streaming load on by default; bit-exactness verified by
  token-stream hash across untiered, tiered and zero-copy runs.
- Sizing: measured rules (`expert_cache_size` toward the batch's per-layer union,
  `ram_cache` ≥ the expert count, `fp8_store` only under space pressure), applied
  automatically by `surgeon autoconfig` and cached per machine.
- Measured on: OLMoE-1B-7B and Qwen3-30B-A3B end to end; DeepSeek-V2-Lite and
  Granite-3.0 read and screened. Two expert-storage layouts read and written.
- Upstream: every borrowed vLLM internal is a declared seam; CI checks the pin on
  every push and vLLM's latest release weekly, opening an issue that names the
  symbol when one moves. At last check: v0.27.1, zero required seams broken.
- Known limits, refused loudly rather than mishandled: tensor parallelism (store
  identity has no rank component), block-quantised fp8, MoE layers with bias
  terms, in-tree and out-of-tree cache both enabled.

[DECISIONS.md](DECISIONS.md) is the decision register: per-method verdicts on the
three axes, the benchmarks behind every number quoted here, and the open items
with reasons.
