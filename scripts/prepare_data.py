#!/usr/bin/env python3
"""Stream, clean, deduplicate, tokenize, and pack public pretraining datasets into uint16 shards."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from coregrad.config import load_local_paths
from coregrad.data.pack import pack_domain_shards
from coregrad.data.sources import iter_parquet_texts, load_pretrain_specs
from coregrad.tokenizer import CoreGradTokenizer

# Default Phase 1 token targets (~6.7B unique tokens total across 4 high-signal English/code/math domains)
DEFAULT_TRAIN_TARGETS: dict[str, int] = {
    "edu-en": 3_200_000_000,
    "cosmo": 1_300_000_000,
    "code-starcoder": 1_200_000_000,
    "finemath": 1_000_000_000,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack public pretraining corpora into uint16 shards.")
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_TRAIN_TARGETS.keys()))
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiply default token targets by this factor (e.g. 0.05 for quick test)",
    )
    parser.add_argument("--val-tokens", type=int, default=4_000_000, help="Val tokens per domain")
    parser.add_argument("--shard-tokens", type=int, default=50_000_000, help="Tokens per train shard")
    args = parser.parse_args()

    paths = load_local_paths()
    tok_path = args.tokenizer or (paths.data_dir / "tokenizer" / "tokenizer.json")
    out_dir = args.out_dir or (paths.data_dir / "shards")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CoreGradTokenizer.from_file(tok_path)
    specs = load_pretrain_specs(domains=args.domains)
    global_seen: set[int] = set()

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_domains = existing.get("domains", {})
    else:
        existing_domains = {}

    manifest_out: dict[str, object] = {
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_path": str(tok_path.name),
        "domains": existing_domains,
    }

    t_start = time.perf_counter()
    for spec in specs:
        if not spec.files:
            print(f"[skip] {spec.name}: no files found")
            continue
        target_train = int(DEFAULT_TRAIN_TARGETS.get(spec.name, 500_000_000) * args.scale)
        target_val = max(200_000, int(args.val_tokens * min(args.scale, 1.0)))
        print(
            f"\nPacking domain '{spec.name}' ({len(spec.files)} files) -> "
            f"target train={target_train/1e6:,.1f}M tok, val={target_val/1e6:,.2f}M tok",
            flush=True,
        )
        t0 = time.perf_counter()
        texts = iter_parquet_texts(spec.files, spec.text_column, interleave_files=True)
        d_manifest = pack_domain_shards(
            domain=spec.name,
            texts=texts,
            tokenizer=tokenizer,
            out_dir=out_dir,
            max_train_tokens=target_train,
            max_val_tokens=target_val,
            shard_tokens=args.shard_tokens,
            dedup_seen=global_seen,
        )
        dt = time.perf_counter() - t0
        manifest_out["domains"][spec.name] = asdict(d_manifest)  # type: ignore[index]
        print(
            f"[done] {spec.name:16s} | train={d_manifest.train.tokens/1e6:7.1f}M tok "
            f"({len(d_manifest.train.shards)} shards, {d_manifest.train.bytes_per_token:.3f} B/tok) | "
            f"val={d_manifest.val.tokens/1e6:5.2f}M tok | dups={d_manifest.duplicates_dropped:,} | "
            f"{dt:.1f}s",
            flush=True,
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    total_train = sum(
        d["train"]["tokens"] for d in manifest_out["domains"].values()  # type: ignore[union-attr]
    )
    total_val = sum(
        d["val"]["tokens"] for d in manifest_out["domains"].values()  # type: ignore[union-attr]
    )
    print(
        f"\nAll shards written to {out_dir} in {(time.perf_counter() - t_start)/60:.1f} min | "
        f"Total train={total_train/1e9:.2f}B tok | Total val={total_val/1e6:.2f}M tok"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
