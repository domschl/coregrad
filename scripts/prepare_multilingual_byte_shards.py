#!/usr/bin/env python3
"""Stream & pack aligned multi-script UTF-8 byte shards from HuggingFaceFW/fineweb-2.

Uses HfFileSystem HTTP range requests over remote Parquet row groups so only the
exact target byte budget (~250 MB train + 10 MB val per language) is transferred.
English ('en') reuses our already-cleaned local FineWeb-Edu byte shard (data/byte_shards/edu-en).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from huggingface_hub.utils import HfHubHTTPError

from coregrad.config import load_local_paths
from coregrad.data.sources import clean_text, doc_hash_u64
from coregrad.tokenizer.byte_tok import BYTE_VOCAB_SIZE, ByteTokenizer

DEFAULT_LANGS: dict[str, str] = {
    "de": "deu_Latn",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "zh": "cmn_Hani",
}


def _pack_fineweb2_split(
    fs: HfFileSystem,
    repo_paths: list[str],
    out_file: Path,
    byte_eos: int,
    target_bytes: int,
    min_lang_score: float,
    seen_hashes: set[int],
) -> dict[str, object]:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    total_utf8_bytes = 0
    docs_kept = 0
    docs_seen = 0

    with out_file.open("wb") as f_out:
        for rpath in repo_paths:
            if total_tokens >= target_bytes:
                break
            with fs.open(f"datasets/HuggingFaceFW/fineweb-2/{rpath}", "rb") as fh:
                pf = pq.ParquetFile(fh)
                cols = ["text"]
                has_score = "language_score" in pf.schema_arrow.names
                if has_score:
                    cols.append("language_score")
                for rg_idx in range(pf.num_row_groups):
                    if total_tokens >= target_bytes:
                        break
                    tbl = pf.read_row_group(rg_idx, columns=cols)
                    texts = tbl.column("text").to_pylist()
                    scores = tbl.column("language_score").to_pylist() if has_score else [1.0] * len(texts)
                    buf: list[int] = []
                    for txt, sc in zip(texts, scores, strict=True):
                        docs_seen += 1
                        if not isinstance(txt, str):
                            continue
                        if sc is not None and float(sc) < min_lang_score:
                            continue
                        cleaned = clean_text(txt, min_chars=64, max_chars=200_000)
                        if cleaned is None:
                            continue
                        h = doc_hash_u64(cleaned)
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)
                        buf.extend(cleaned.encode("utf-8"))
                        buf.append(byte_eos)
                        docs_kept += 1
                    if buf:
                        if total_tokens + len(buf) > target_bytes:
                            buf = buf[: max(0, target_bytes - total_tokens)]
                        np_buf = np.array(buf, dtype=np.uint16)
                        f_out.write(np_buf.tobytes())
                        total_tokens += len(buf)
                        total_utf8_bytes += int((np_buf < 256).sum())
            if total_tokens >= target_bytes:
                break

    return {
        "tokens": total_tokens,
        "utf8_bytes": total_utf8_bytes,
        "docs": docs_kept,
        "docs_scanned": docs_seen,
        "shards": [out_file.name],
    }


def _list_split_parquets(api: HfApi, subset: str, split: str) -> list[str]:
    try:
        items = list(
            api.list_repo_tree("HuggingFaceFW/fineweb-2", path_in_repo=f"data/{subset}/{split}", repo_type="dataset")
        )
    except (HfHubHTTPError, OSError):
        return []
    return sorted(f.path for f in items if getattr(f, "path", "").endswith(".parquet"))


def _copy_capped_byte_shard(src: Path, dst: Path, max_tokens: int) -> dict[str, object]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(src, dtype=np.uint16, mode="r")
    n = min(len(arr), max_tokens)
    chunk = 8_000_000
    utf8_bytes = 0
    with dst.open("wb") as f_out:
        for i in range(0, n, chunk):
            sl = np.asarray(arr[i : min(n, i + chunk)], dtype=np.uint16)
            f_out.write(sl.tobytes())
            utf8_bytes += int((sl < 256).sum())
    return {"tokens": int(n), "utf8_bytes": int(utf8_bytes), "shards": [dst.name]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase 5.3B Stage 0 multi-script UTF-8 byte shards.")
    parser.add_argument("--out-subdir", type=str, default="multilingual_stage0")
    parser.add_argument("--train-bytes", type=int, default=250_000_000, help="Train bytes per language (default 250 MB)")
    parser.add_argument("--val-bytes", type=int, default=10_000_000, help="Val bytes per language (default 10 MB)")
    parser.add_argument("--min-lang-score", type=float, default=0.90)
    parser.add_argument(
        "--langs",
        type=str,
        default="de,ru,ar,zh",
        help="Comma-separated FineWeb-2 short codes from {de,ru,ar,zh}",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    paths = load_local_paths()
    dst_root = paths.data_dir / args.out_subdir
    dst_root.mkdir(parents=True, exist_ok=True)

    byte_tok = ByteTokenizer()
    manifest_path = dst_root / "manifest.json"
    domains_out: dict[str, object] = {}
    if manifest_path.exists() and not args.overwrite:
        domains_out = dict(json.loads(manifest_path.read_text(encoding="utf-8")).get("domains", {}))

    # 1. Copy/cap English from existing cleaned edu-en byte shards
    en_dir = dst_root / "en"
    if args.overwrite or "en" not in domains_out or not (en_dir / "train_0000.bin").exists():
        src_en = paths.data_dir / "byte_shards" / "edu-en"
        if not (src_en / "train_0000.bin").exists():
            raise FileNotFoundError(f"Missing English byte shard at {src_en}; run scripts/prepare_byte_shards.py first.")
        print("[stage0] Preparing en (from local cleaned FineWeb-Edu byte shards)...", flush=True)
        if en_dir.exists() and args.overwrite:
            shutil.rmtree(en_dir)
        tr = _copy_capped_byte_shard(src_en / "train_0000.bin", en_dir / "train_0000.bin", args.train_bytes)
        va = _copy_capped_byte_shard(src_en / "val_0000.bin", en_dir / "val_0000.bin", args.val_bytes)
        domains_out["en"] = {"train": tr, "val": va, "source": "HuggingFaceFW/fineweb-edu"}
        print(f"  en | train={tr['tokens']/1e6:.1f}MB | val={va['tokens']/1e6:.1f}MB", flush=True)

    # 2. Stream FineWeb-2 languages via HfFileSystem
    api = HfApi()
    fs = HfFileSystem()
    req_langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    for code in req_langs:
        if code not in DEFAULT_LANGS:
            raise ValueError(f"Unknown lang code {code!r}; expected subset of {sorted(DEFAULT_LANGS)}")
        subset = DEFAULT_LANGS[code]
        lang_dir = dst_root / code
        if not args.overwrite and code in domains_out and (lang_dir / "train_0000.bin").exists():
            print(f"[stage0] Skipping existing {code} ({subset})", flush=True)
            continue

        t_lang = time.perf_counter()
        print(f"[stage0] Streaming {code} ({subset}) from HuggingFaceFW/fineweb-2...", flush=True)
        seen_hashes: set[int] = set()
        val_pqs = _list_split_parquets(api, subset, "test")
        train_pqs = _list_split_parquets(api, subset, "train")
        if not train_pqs:
            raise RuntimeError(f"No train parquets found for {subset}")

        # Pack val first from official test split (falls back to tail of train[0] if test < target)
        va = _pack_fineweb2_split(
            fs,
            val_pqs if val_pqs else train_pqs[-1:],
            lang_dir / "val_0000.bin",
            byte_tok.eos_id,
            args.val_bytes,
            args.min_lang_score,
            seen_hashes,
        )
        tr = _pack_fineweb2_split(
            fs,
            train_pqs,
            lang_dir / "train_0000.bin",
            byte_tok.eos_id,
            args.train_bytes,
            args.min_lang_score,
            seen_hashes,
        )
        domains_out[code] = {"train": tr, "val": va, "source": f"HuggingFaceFW/fineweb-2/{subset}"}
        print(
            f"  {code} ({subset}) | train={tr['tokens']/1e6:.1f}MB ({tr['docs']:,} docs) | "
            f"val={va['tokens']/1e6:.1f}MB ({va['docs']:,} docs) | {time.perf_counter()-t_lang:.1f}s",
            flush=True,
        )

        # Checkpoint manifest incrementally after each language
        manifest = {"vocab_size": BYTE_VOCAB_SIZE, "tokenizer": "utf8-byte-288", "domains": domains_out}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    manifest = {"vocab_size": BYTE_VOCAB_SIZE, "tokenizer": "utf8-byte-288", "domains": domains_out}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done in {time.perf_counter()-t0:.1f}s -> {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
