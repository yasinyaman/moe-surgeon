# moe-surgeon

Serve a Mixture-of-Experts model in less memory than it fits in. Cold experts
live on NVMe and stream to a GPU cache on demand; a correctly sized cache is
numerically transparent, so the tiered model produces the same tokens as the
untiered one.

It installs as a vLLM plugin. No fork, no patched vLLM.

```bash
pip install git+https://github.com/yasinyaman/moe-surgeon.git
surgeon autoconfig --checkpoint /path/to/model --start
```

## What it does

| you have | you get |
|---|---|
| a model too large for the card | it runs — 12.9 GiB model at 2.69 GiB peak |
| slow cold starts | 13–15 s instead of 118 s |
| a card that fits the model already | nothing; don't use this |

The cost is decode throughput: a correctly sized tier runs at 0.66× untiered.
This is a feasibility and load-time mechanism, not a speed one.

Optional, and measured separately: permanent expert pruning for deployments
that only ever see one domain, and CPU co-execution on discrete cards.

## Install

```bash
pip install git+https://github.com/yasinyaman/moe-surgeon.git
```

Requires vLLM at serve time. The offline commands (`budget`, `plan`, `tier`,
`inspect`, `recommend`) run without vLLM, without CUDA and without a GPU.

Registration is automatic through a `vllm.general_plugins` entry point. Verify
it took:

```bash
python -c "from importlib.metadata import entry_points; \
print([e.name for e in entry_points(group='vllm.general_plugins')])"
```

## Quick start

**Let it size itself and serve:**

```bash
surgeon autoconfig --checkpoint /path/to/model --start
```

It probes the machine once, caches the decision, and runs `vllm serve` with the
right settings. `--json` prints the config instead of starting; omit `--start`
to see the reasoning.

**Or talk to it interactively, with per-turn cost:**

```bash
surgeon run allenai/OLMoE-1B-7B-0924 --checkpoint /path/to/model
```

```
>>> explain a b-tree in two sentences
A B-tree is a self-balancing search tree ...
  [128 tok, 142.2 tok/s, cache 93% (15302/16384)]
```

`/stats` summarises the session and names a misconfiguration if the counters
show one. `/help` lists the commands. `--no-tier` serves untiered for a
side-by-side.

**Or configure vLLM yourself:**

```bash
vllm serve allenai/OLMoE-1B-7B-0924 \
  --additional-config '{"surgeon": {"expert_cache_size": 48,
      "store_dir": "./store", "ram_cache": 64}}'
```

The store builds itself on first boot and is reused afterwards.

The checkpoint must be a **local directory** for sizing, because the analysis
reads safetensors headers. Pass a repo id as the model and `--checkpoint` for
the local copy when they differ.

## Commands

| command | what it answers | needs |
|---|---|---|
| `surgeon budget` | will this model fit, and in how little | nothing |
| `surgeon vram-floor` | the smallest budget it actually boots in | GPU |
| `surgeon autoconfig` | how should this machine be configured | nothing |
| `surgeon run` | serve it interactively, with the cost shown | GPU |
| `surgeon headroom` | is a different checkpoint better for my domain | GPU |
| `surgeon recommend` | which methods does this target want | nothing |
| `surgeon profile` | which experts does my workload use | GPU |
| `surgeon inspect` | what routing signals does this checkpoint carry | nothing |
| `surgeon plan` | turn a profile and a budget into a placement | nothing |
| `surgeon gate` | what would this plan's deletions cost | GPU |
| `surgeon apply` | write the pruned checkpoint | nothing |
| `surgeon calibrate` | what amplitude should a pruned checkpoint carry | GPU |
| `surgeon ablate` | what do these experts contribute | GPU |
| `surgeon tier` | build the store offline | nothing |
| `surgeon serve` | run the pipeline over HTTP | GPU for its stages |
| `surgeon seams` | will this survive a vLLM upgrade | nothing |

Every command takes `--help`.

## Headline numbers

Measured on a DGX Spark (GB10, 121 GB unified) and an RTX 3050 Ti laptop
(3.68 GiB usable). Full tables and method: [docs/benchmarks.md](docs/benchmarks.md).

| | measured |
|---|---|
| serves a 12.9 GiB model on a 3.68 GiB card | 2.69 GiB peak, 7.7 tok/s |
| load time | 13–15 s vs 118 s untiered |
| **cache size, the one knob that matters** | **2.67× decode across one integer** |
| decode, correctly sized | 145.1 vs 218.5 tok/s untiered |
| numerical transparency | identical token hashes, 9 of 9 runs |
| CPU co-execution, discrete card | 1.58× |
| CPU co-execution, unified memory | 0.719× — a loss, refused |

## Documentation

| doc | what it covers |
|---|---|
| [docs/scenarios.md](docs/scenarios.md) | worked examples: small card, big box, narrow domain, multi-model |
| [docs/configuration.md](docs/configuration.md) | every setting, every default, every refusal |
| [docs/benchmarks.md](docs/benchmarks.md) | measured numbers and how they were taken |

## Limits

Refused by name rather than mishandled: tensor parallelism, block-quantised
fp8, MoE layers with bias terms, and running alongside vLLM's in-tree expert
cache. `cpu_experts` additionally refuses fp8 stores and checkpoints,
zero-copy, CUDA graphs, non-silu activations and router-weight-on-input models.

Expert pruning costs downstream task accuracy that a perplexity gate does not
catch, so `apply` refuses to delete without a measured, plan-bound verdict.

## Licence

Apache-2.0.
