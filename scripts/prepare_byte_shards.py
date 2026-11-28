#!/usr/bin/env python3
"""Decode existing cleaned BPE shards into aligned raw UTF-8 byte shards (uint16, vocab=288).

Ensures Gate 4 (H-Net UTF-8 vs. 16k BPE) trains and evaluates on the EXACT same cleaned
documents and held-out validation splits so Bits-Per-Byte (BPB) is 100% apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from coregrad.config import load_local_paths
from coregrad.tokenizer import CoreGradTokenizer
from coregrad.tokenizer.byte_tok import BYTE_VOCAB_SIZE, ByteTokenizer


def _convert_split(
    src_domain_dir: Path,
    shard_names: list[str],
    bpe_tok: CoreGradTokenizer,
    byte_tok: ByteTokenizer,
    out_file: Path,
    max_bytes: int | None,
) -> dict[str, object]:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    bpe_eos = bpe_tok.eos_id
    byte_eos = byte_tok.eos_id

    total_tokens = 0
    total_utf8_bytes = 0
    chunk_toks = 1_000_000

    with out_file.open("wb") as f_out:
        for sname in shard_names:
            arr = np.memmap(src_domain_dir / sname, dtype=np.uint16, mode="r")
            n = len(arr)
            pos = 0
            while pos < n:
                end = min(n, pos + chunk_toks)
                block = arr[pos:end]
                pos = end

                # Split on BPE EOS so document boundaries map cleanly to byte_tok.eos_id (258)
                eos_positions = np.flatnonzero(block == bpe_eos)
                buf: list[int] = []
                start = 0
                for ep in eos_positions:
                    ep_i = int(ep)
                    if ep_i > start:
                        text = bpe_tok.decode(block[start:ep_i].tolist(), skip_special_tokens=True)
                        buf.extend(text.encode("utf-8"))
                    buf.append(byte_eos)
                    start = ep_i + 1
                if start < len(block):
                    text = bpe_tok.decode(block[start:].tolist(), skip_special_tokens=True)
                    buf.extend(text.encode("utf-8"))

                if buf:
                    if max_bytes is not None and total_tokens + len(buf) > max_bytes:
                        keep = max(0, max_bytes - total_tokens)
                        buf = buf[:keep]
                    np_buf = np.array(buf, dtype=np.uint16)
                    f_out.write(np_buf.tobytes())
                    total_tokens += len(buf)
                    total_utf8_bytes += int((np_buf < 256).sum())
                if max_bytes is not None and total_tokens >= max_bytes:
                    break
            if max_bytes is not None and total_tokens >= max_bytes:
                break

    return {
        "tokens": total_tokens,
        "utf8_bytes": total_utf8_bytes,
        "shards": [out_file.name],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build aligned UTF-8 byte shards from existing BPE shards.")
    parser.add_argument(
        "--max-train-bytes-per-domain",
        type=int,
        default=500_000_000,
        help="Cap train bytes per domain (default 500 MB/domain -> ~2.0 GB total across 4 domains)",
    )
    parser.add_argument(
        "--max-val-bytes-per-domain",
        type=int,
        default=12_000_000,
        help="Cap val bytes per domain (default 12 MB/domain)",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    paths = load_local_paths()
    src_root = paths.data_dir / "shards"
    dst_root = paths.data_dir / "byte_shards"
    dst_root.mkdir(parents=True, exist_ok=True)

    bpe_tok = CoreGradTokenizer.from_file(paths.data_dir / "tokenizer" / "tokenizer.json")
    byte_tok = ByteTokenizer()
    src_manifest = json.loads((src_root / "manifest.json").read_text(encoding="utf-8"))

    domains_out: dict[str, object] = {}
    for domain, info in src_manifest["domains"].items():
        print(f"[byte_shards] Converting domain={domain}...", flush=True)
        dom_dir = dst_root / domain
        tr = _convert_split(
            src_root / domain,
            info["train"]["shards"],
            bpe_tok,
            byte_tok,
            dom_dir / "train_0000.bin",
            args.max_train_bytes_per_domain,
        )
        va = _convert_split(
            src_root / domain,
            info["val"]["shards"],
            bpe_tok,
            byte_tok,
            dom_dir / "val_0000.bin",
            args.max_val_bytes_per_domain,
        )
        domains_out[domain] = {"train": tr, "val": va}
        print(
            f"  {domain:16s} | train={tr['tokens']/1e6:6.1f}M bytes | val={va['tokens']/1e6:5.2f}M bytes",
            flush=True,
        )

    manifest = {
        "vocab_size": BYTE_VOCAB_SIZE,
        "tokenizer": "utf8-byte-288",
        "domains": domains_out,
    }
    (dst_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done in {time.perf_counter() - t0:.1f}s -> {dst_root / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
