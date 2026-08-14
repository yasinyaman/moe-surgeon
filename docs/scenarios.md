# Scenarios

Five situations, each with the commands to run and what the numbers came out
as. Machines: **GB10** (DGX Spark, 121 GB unified memory) and **laptop**
(RTX 3050 Ti, 3.68 GiB usable, 14.8 GB host RAM).

---

## 1. The model does not fit the card

*RTX 3050 Ti, 3.68 GiB. OLMoE-1B-7B is 12.9 GiB. Untiered it does not load —
the attempt drives the host into swap.*

Check first, without booting anything:

```bash
surgeon budget --checkpoint /path/to/OLMoE-1B-7B-0924 --vram 3.6
```

```
bit-exact floor (capacity = top_k = 8):
  GPU        2.39 GiB   (no KV cache included)
  host RAM   1.51 GiB
  disk       6.05 GiB
```

Then let it size itself and serve:

```bash
surgeon autoconfig --checkpoint /path/to/OLMoE-1B-7B-0924 --start
```

**Result:** serves at **2.69 GiB peak, 7.7 tok/s**, load 6.6 s.

On this card the cache lands below `top_k`, so autoconfig turns on the expert
split and says so — that path is not bit-exact. It also enables `fp8_store`,
because the 14.8 GB host cannot hold every expert at full width.

**If your card is discrete and starved like this one, also turn on CPU
co-execution** — see scenario 4.

---

## 2. The model fits, and you want it fast

*GB10, 121 GB. OLMoE fits untiered at 218.5 tok/s.*

The honest answer is **don't tier it**. Tiering a model that fits costs
throughput:

| configuration | decode |
|---|---|
| untiered | **218.5 tok/s** |
| tier, cache 48 of 64 | 145.1 tok/s |
| tier, cache 24 of 64 | 54.3 tok/s |

Use the tier here only for what it does win: **load time, ~17 s against 129 s**
at cache 48 — expert weights stream into the store instead of onto the device.
If you restart models often, that alone can be the reason.

If you do tier it, the cache size is the whole game:

```bash
surgeon autoconfig --checkpoint /path/to/model --max-num-seqs 8
```

`--max-num-seqs` is load-bearing: the cache should cover the union of experts
your serving batch can reach, which is `max_num_seqs × top_k` capped at the
expert count. Sizing from free VRAM instead is the mistake worth 2.67×.

---

## 3. One domain, and you are considering pruning

*A deployment that only ever analyses system logs.*

**Check the cheap thing first.** A different existing checkpoint may serve the
domain better than any surgery on this one:

```bash
surgeon headroom --corpus heldout.jsonl \
  --model allenai/OLMoE-1B-7B-0924 \
  --model ibm-granite/granite-3.0-3b-a800m-base
```

```
  model                                         bits/byte       ppl  B/token
  ibm-granite/granite-3.0-3b-a800m-base            1.3395     11.58     2.64
  allenai/OLMoE-1B-7B-0924                         1.4073     27.87     3.41

best on this domain: ibm-granite/granite-3.0-3b-a800m-base (4.8% better bits/byte)
```

Read **bits/byte**, not ppl: per-token perplexity depends on the tokenizer, and
these two split the same text at 2.64 against 3.41 bytes per token.

If a candidate wins, verify it on real tasks before adopting it — likelihood is
not accuracy. In the run above the follow-up mostly held: no significant
arc_challenge **acc_norm** loss and ~4× better gsm8k, but raw `acc` fell 0.068
(p=0.0008) and hellaswag 0.044 (p=0.0032). A win here is permission to run the
task evaluation, not its result.

**If you still need to prune**, the pipeline is profile → plan → gate → apply →
calibrate → apply again:

```bash
surgeon profile --model MODEL --corpus domain.jsonl --cooc --out profile.npz
surgeon plan --profile profile.npz --core-experts 40 --disk-experts 0 --out plan.json
surgeon gate --plan plan.json --corpus heldout.jsonl --max-ratio 1.3
surgeon apply --plan plan.json --source /original --out /pruned
surgeon calibrate --checkpoint /pruned --corpus heldout.jsonl
surgeon apply --plan plan.json --source /original --out /pruned-final --amplitude 0.85
```

Notes that decide the outcome:

- **Nothing is deleted unless you ask.** Without `--disk-experts 0` the cold
  tail goes to the disk tier and the plan reports "deletes nothing".
- **`apply` refuses an ungated plan.** The gate's verdict is bound to the exact
  drop set by digest, so editing the plan after gating is caught.
- **`apply` runs twice, and that is not a typo.** `calibrate` sweeps the
  amplitude on a checkpoint that already exists, so the first `apply` writes
  one to measure and the second folds in the scalar it reports. The `0.85`
  above is this domain's measured value, not a constant — `surgeon plan` also
  predicts it from the plan's own deleted routing mass (0.861 predicted against
  0.850 measured), which is a usable first pass when a sweep is too expensive.
- **Run `calibrate`.** Deletion inflates the surviving gates by `1/(1-P_D)`.
  On this domain, skipping the correction measured 1.47× perplexity where the
  corrected apply measured 1.17×.
- **Pin the corpus, not a slice notation.** Which held-out prompts you score on
  moves perplexity by more than the effects reported here.
  [`tools/derive_corpus.py`](../tools/derive_corpus.py) writes the JSONL and a
  sidecar recording dataset, split, slice, field, template and truncation, so
  the number stays reproducible.

**Result on real system logs**, held-out perplexity, baseline 27.89:

| pruned to | applied | ratio |
|---|---|---|
| 56 of 64 | 30.64 | 1.10× |
| 40 of 64, no amplitude | 41.00 | 1.47× |
| 40 of 64, amplitude 0.85 | 32.53 | **1.17×** |

And what it bought, tier against tier at full coverage:

| configuration | GPU slot bytes | decode |
|---|---|---|
| pruned-40 + tier | **7.5 GiB** | **256.2 tok/s** |
| unpruned + tier | 12.0 GiB | 205.1 tok/s |

**The caveat that matters:** on a broad benchmark the same pruning costs
arc_challenge 0.468 → 0.352. This is for deployments that never leave their
domain.

---

## 4. A discrete card, starved of cache

*RTX 3050 Ti. The cache holds 4 experts of 64, so most of every step is spent
fetching experts over PCIe.*

Compute them on the host instead:

```bash
vllm serve MODEL --enforce-eager --additional-config '{"surgeon": {
  "expert_cache_size": 4, "split": "expert",
  "store_dir": "./store", "ram_cache": 24,
  "cpu_experts": true, "cpu_expert_threads": 12}}'
```

`--enforce-eager` is required here and only here: `cpu_experts` joins a
host-computed partial into the MoE output on every forward, and a captured
graph replays without the host. The runtime refuses the combination at boot
rather than serving wrong output. (The plain tier does *not* need the flag — it
carves the MoE op out of the captured region instead.)

**Result**, flag as the only difference:

| ram_cache | off | on | |
|---|---|---|---|
| 16 | 2.78 tok/s | 3.58 tok/s | 1.29× |
| 24 | 4.66 tok/s | **7.37 tok/s** | **1.58×** |

The gain grows with `ram_cache`, because what remains is disk reads both arms
share.

**Only on discrete cards.** The mechanism needs host DRAM reads and device
transfers to draw on separate bandwidth pools. On unified memory (GB10) the
same code measured **0.719× — a loss**. The predictor is
`BW_cpu_gemm / BW_h2d`: 3.09 on this laptop, 0.78 on GB10.

Not bit-exact — the host reduction order differs from the fused kernel's, and
the runtime says so at boot.

---

## 5. Several models on one big machine

*GB10, 121 GB unified, four models wanted at once.*

Answer the memory question before booting anything:

```bash
surgeon budget --checkpoint /path/to/model
```

Four copies of a 12.89 GiB model is 51.6 GiB of weights against a 121 GB pool,
so they likely fit untiered — in which case **do not tier them**, and simply
give each vLLM instance a quarter of the budget
(`--gpu-memory-utilization 0.2` or so, leaving headroom).

Tier them only if the arithmetic says they do not fit. Then the tier buys
concurrency and context rather than speed: at the bit-exact floor the same four
models hold ~9.6 GiB of weights instead of 51.6 GiB, and the difference goes to
KV cache.

**Two traps specific to this shape.** On unified memory, page-locked host
memory is charged against `gpu_memory_utilization`, so each tiered instance
costs more of the budget than its device bytes suggest — keep `stream_load` on
(the default). And the pinned-pool ceiling is a whole-machine limit: 41.5 GB is
safe, 55.3 GB wedges the host. Four instances each sizing `ram_cache` as if
they were alone will cross it.

Multi-model concurrency has not been benchmarked here; the numbers above are
single-model measurements and per-model arithmetic.
