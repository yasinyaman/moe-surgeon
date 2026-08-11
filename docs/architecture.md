# Architecture: one directory owns the vLLM dependency

Why this is a separate package, the layering rule and the seam table that
enforce it, and what a vLLM upgrade costs to evaluate.

## Why it is a separate package

The predecessor of this work lives inside a vLLM fork: 20 files, +5063 lines,
about 800 of them hooks threaded through upstream files. Every vLLM release turned
that into a rebase argument — at one point upstream deleted the `FusedMoE` class
outright and every hunk lost its target.

So the rule here is structural, and a test enforces it:

> Only `compat/` may import vLLM.

Everything else — the surgery pipeline, the checkpoint writer, the disk store, the
job server — runs on a host with no vLLM, no CUDA and no GPU. That is what makes
the offline half testable on a laptop, and it confines the cost of a vLLM upgrade
to one reviewable directory.

`compat/seams.py` declares every vLLM internal we depend on as data, with the
reason we hold it. `tests/test_seams.py` checks each one. On an upgrade, that test
fails first — before a worker does, and with a message naming the symbol that
moved.

## Layout

| path | imports vLLM | what it is |
|---|---|---|
| `compat/` | **yes** | the seam layer; the only place that knows vLLM internals |
| `plugin.py` | **yes** | the `vllm.general_plugins` entry point |
| `telemetry/` | no | aggregation of vLLM's routed-experts capture |
| `store/` | no | the NVMe → pinned RAM → VRAM expert tier, plus its replay simulator |
| `surgery/` | no | inspect, budget, plan, align, merge, apply, tier |
| `server/` | no | the job server: stage orchestration, job records, HTTP |

`store/` is a verbatim lift from the prototype branch — only its imports were
rewritten (`vllm.envs` → `env.py`, `vllm.logger` → `_logging.py`). It is kept
diffable against `repos/vllm` on `disk-tier-proto` on purpose, which is why
`pyproject.toml` exempts it from two lint rules rather than reformatting it.

## Tests

```bash
python3 -m pytest tests/ -q
```

Three levels, and they run in different places:

- **CPU** (`test_store_cpu.py`, `test_layering.py`, table checks in
  `test_seams.py`) — run anywhere, no GPU, no vLLM. This is the laptop loop.
- **static seams** (`test_seams.py`) — parses a vLLM *source tree*; point it with
  `MOE_SURGEON_VLLM_SRC=/path/to/vllm`, or let it find the sibling checkout.
- **CUDA / installed vLLM** (`test_expert_cache.py`, import-level seam checks) —
  skip without a device. These run on the GPU boxes.

A large `skipped` count on a laptop is expected, not a warning sign.

## Pricing a vLLM upgrade

```bash
python tools/upstream_drift.py --repo ../vllm --from <pinned-ref> --to <candidate-ref>
```

Reads the seam modules straight out of git at both refs and reports, per seam,
whether it holds, broke, or newly appeared. No install, no torch, no GPU. Exit
status is 1 if a required seam broke, so it can gate a pin bump.

Measured 2026-08-07, merge base → upstream `main`, **296 commits**: 342 lines
changed across the five files we hold, **0 required seams broken**, 1 optional
seam newly available (`RoutedExperts._orient_fused_weight` — upstream extracted
the fused-weight orientation branch we would otherwise have duplicated).

That number is the argument for this package. The same window would have landed
on the fork's ~800 lines of in-file hooks: `moe_runner.py` alone lost 41 lines,
`config/vllm.py` gained 195.
