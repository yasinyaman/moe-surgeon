# The job server

The HTTP pipeline: one POST runs profile → plan → gate → tier, with the
hardening that request path carries.

## The job server

```bash
surgeon serve --port 8300 --state ./surgeon-state   # binds 127.0.0.1
```

The server runs subprocesses from request fields, so it binds loopback by default.
Exposing it on a non-loopback host requires `--token` (or `$MOE_SURGEON_TOKEN`),
which every request then carries in an `X-Surgeon-Token` header; `/health` stays
open. Engine kwargs in a request are allow-listed (`trust_remote_code` is refused),
and a stage never fetches an uncached model unless the job sets `allow_download`.

```bash
curl -sX POST localhost:8300/jobs -H 'content-type: application/json' -d '{
  "model": "allenai/OLMoE-1B-7B-0924",
  "corpus": "domain.jsonl", "heldout": "heldout.jsonl",
  "core_experts": 24, "stages": ["profile", "plan", "gate", "tier"]
}'
```

It adds orchestration, persistence and a record. It adds **no capability** — every
stage is a `surgeon` subcommand, and a job record stores each stage's argv, so any
failed run is reproducible by hand. That is deliberate: the debuggable version of a
pipeline server is one that cannot do anything you could not do yourself.

### headroom, and why it runs first

The full stage order is `headroom → profile → recommend → plan → gate → apply →
tier`, and `headroom` is first because a win there retires everything after it:
it asks whether a smaller existing checkpoint already serves the domain better,
which was measured once to be true at no quality cost where pruning cost 11.6
points of arc_challenge. Measuring that after a profile and a plan is measuring
it after the regret.

```bash
curl -sX POST localhost:8300/jobs -H 'content-type: application/json' -d '{
  "model": "allenai/OLMoE-1B-7B-0924",
  "heldout": "heldout.jsonl",
  "candidates": ["ibm-granite/granite-3.0-3b-a800m-base"],
  "stages": ["headroom"]
}'
```

It needs `heldout` and at least one entry in `candidates`; a ranking of one row
is not a ranking, so a request without candidates is refused before an engine
boots. The result lands as the job's `headroom` artifact
(`headroom.json`). Note it is **not** a gate: it scores held-out likelihood,
which this project has already watched part ways with task accuracy, so it
ranks candidates and says so rather than blocking the pipeline on the answer.

`surgeon recommend` names the same check as a prerequisite whenever it proposes
deletion — it runs without a GPU, so it cannot measure it, but it will not
recommend cutting experts without asking whether cutting is necessary at all.

Three properties are load-bearing rather than incidental:

**Stages are subprocesses.** A `profile` or `gate` stage boots a vLLM engine, and an
engine does not release device memory when its Python object goes out of scope — the
same thing that broke the first benchmark harness, where four arms in one process
left two unable to boot. A crashing stage must not take the server with it either.

**One worker, not a pool.** Two pipelines would both claim the GPU and the second
would fail to boot with an out-of-memory error that reads like a model bug rather
than a scheduling mistake. Concurrency here manufactures confusing failures.

**Unrunnable requests are rejected before anything runs**, as a 400 naming the field.
This was learned the expensive way: a run spent 114 seconds profiling and then failed
in `tier` with `no safetensors checkpoint under allenai/OLMoE-1B-7B-0924`, because
`apply` and `tier` open safetensors directly and had been handed a repo id.

The fix was to *resolve* rather than to *demand*. A repo id resolves through the HF
cache with `local_files_only`, so a request never needs a hash-named snapshot path —
but only for the files those stages read (`*.safetensors`, the shard index,
`config.json`). A plain `local_files_only` resolution insists the **entire** repo be
cached and raises `IncompleteSnapshotError` otherwise, which is the normal state of a
cache vLLM filled: it never fetches a README.

A gate failure is recorded as a **verdict, not an error** — the job stops, the
pipeline is marked failed, and the reason is that the model did not clear the
threshold. That distinction matters, because "surgery would hurt too much" is a
successful measurement and should not read as a broken run.

Measured on GB10 from a bare repo id, OLMoE, 24 core experts:

| stage | | |
|---|---|---|
| `profile` | succeeded | 103.0 s |
| `plan` | succeeded | 1.4 s |
| `gate` | succeeded | 1.5 s |
| `tier` | succeeded | 17.5 s |

producing `profile.npz`, `plan.json`, a 6.1 GiB fp8 store over 16 layer files, and a
`hot_experts.json` carrying 24 core ids plus a prior for **all 64** experts per
layer — which is the seam the plan recorded as unfilled, now filled by a pipeline
rather than by hand.
