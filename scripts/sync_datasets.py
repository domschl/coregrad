#!/usr/bin/env python3
"""Download additional public datasets into the configured local HF cache directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from coregrad.config import load_local_paths


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_files(
    name: str,
    repo_id: str,
    filenames: list[str],
    license_name: str,
    cache_root: Path,
    revision: str | None = None,
) -> None:
    api = HfApi()
    info = api.dataset_info(repo_id, revision=revision)
    pinned_rev = info.sha or revision or "main"
    target_dir = cache_root / repo_id
    target_dir.mkdir(parents=True, exist_ok=True)
    sources_log = cache_root / "SOURCES.jsonl"

    for rel_path in filenames:
        dest = target_dir / rel_path
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {repo_id}/{rel_path} already exists ({dest.stat().st_size/1e6:.1f} MB)")
            continue
        print(f"[download] {repo_id}/{rel_path} (rev {pinned_rev[:12]}) ...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=rel_path,
            repo_type="dataset",
            revision=pinned_rev,
            local_dir=target_dir,
        )
        p = Path(downloaded)
        record = {
            "name": name,
            "repo": repo_id,
            "path": rel_path,
            "revision": pinned_rev,
            "license": license_name,
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "time": int(time.time()),
        }
        with sources_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[ok] {repo_id}/{rel_path} ({record['bytes']/1e6:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync recommended public datasets to local HF cache.")
    parser.add_argument("--extra-edu", action="store_true", help="Download fineweb-edu shards 002 & 003")
    parser.add_argument("--smoltalk", action="store_true", help="Download HuggingFaceTB/smol-smoltalk")
    parser.add_argument("--all", action="store_true", help="Download all recommended Phase 1–3 additions")
    args = parser.parse_args()

    paths = load_local_paths()
    if paths.hf_cache_dir is None:
        raise SystemExit("hf_cache_dir not configured (set COREGRAD_HF_CACHE or configs/sources.local.yaml)")

    do_edu = args.extra_edu or args.all
    do_smoltalk = args.smoltalk or args.all
    if not (do_edu or do_smoltalk):
        parser.print_help()
        return 0

    if do_edu:
        download_files(
            name="edu-en",
            repo_id="HuggingFaceFW/fineweb-edu",
            filenames=["sample/10BT/002_00000.parquet", "sample/10BT/003_00000.parquet"],
            license_name="odc-by",
            cache_root=paths.hf_cache_dir,
            revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        )

    if do_smoltalk:
        api = HfApi()
        files = [
            f
            for f in api.list_repo_files("HuggingFaceTB/smol-smoltalk", repo_type="dataset")
            if f.endswith(".parquet")
        ]
        download_files(
            name="sft-smoltalk",
            repo_id="HuggingFaceTB/smol-smoltalk",
            filenames=sorted(files),
            license_name="apache-2.0",
            cache_root=paths.hf_cache_dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
