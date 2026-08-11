#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Derive a calibration or held-out corpus as JSONL, reproducibly.

The profiling and gate stages read a ``.jsonl`` of ``{"text": ...}`` records. Which
prompts those are moves perplexity by more than the effects the project reports, so
"gsm8k[400:500]" is not a specification -- the split, the field, the template and the
truncation all have to be pinned. This script pins them, and writes a header record
recording exactly how the corpus was built, so a published number is reproducible.

    python tools/derive_corpus.py --dataset gsm8k --config main --split test \
        --start 400 --end 500 --field question --out heldout.jsonl

The ``.jsonl`` is pure ``{"text": ...}`` records, one per line, so the profiling
reader consumes it unchanged. A sidecar ``<out>.meta.json`` records exactly how the
corpus was built -- dataset, config, split, slice, field, template, truncation -- so a
published number is traceable to a reproducible corpus rather than to a slice notation.

Requires ``datasets`` (``HF_DATASETS_OFFLINE=1`` works once the set is cached).
"""

from __future__ import annotations

import argparse
import json
import sys


def build(args: argparse.Namespace) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.config or None, split=args.split)
    end = args.end if args.end is not None else len(ds)
    rows = ds.select(range(args.start, min(end, len(ds))))

    template = args.template  # e.g. "{question}" or "{ctx}\n{question}"
    out: list[dict] = []
    for row in rows:
        if template:
            text = template.format(**row)
        else:
            text = row[args.field]
        if args.max_chars and len(text) > args.max_chars:
            text = text[: args.max_chars]
        out.append({"text": text})
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="HF dataset id, e.g. gsm8k")
    p.add_argument("--config", default="", help="dataset config, e.g. main")
    p.add_argument("--split", default="test")
    p.add_argument("--start", type=int, default=0, help="slice start (inclusive)")
    p.add_argument("--end", type=int, default=None, help="slice end (exclusive)")
    p.add_argument(
        "--field", default="text", help="record field to take when no --template"
    )
    p.add_argument(
        "--template",
        default="",
        help="Python str.format template over the record, e.g. '{question}'. "
        "Overrides --field.",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="truncate each prompt to this many characters (0 = no truncation)",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()

    records = build(args)
    meta = {
        "dataset": args.dataset,
        "config": args.config or None,
        "split": args.split,
        "slice": [args.start, args.end],
        "field": args.field,
        "template": args.template or None,
        "max_chars": args.max_chars or None,
        "n_records": len(records),
    }
    with open(args.out, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {len(records)} prompts to {args.out} (+ {args.out}.meta.json)")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
