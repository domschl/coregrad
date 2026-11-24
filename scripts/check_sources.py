#!/usr/bin/env python3
"""Verify which public datasets from configs/datasets.yaml are present in the local HF cache."""

from __future__ import annotations

from coregrad.config import REPO_ROOT, load_local_paths, load_yaml


def main() -> int:
    paths = load_local_paths()
    manifest = load_yaml(REPO_ROOT / "configs" / "datasets.yaml")

    if paths.hf_cache_dir is None or not paths.hf_cache_dir.exists():
        print("HF cache directory is not configured or does not exist.")
        print("Set COREGRAD_HF_CACHE or create configs/sources.local.yaml.")
        return 1

    print(f"Checking public dataset manifest against configured HF cache ({paths.hf_cache_dir.name}/):")
    total_bytes = 0
    for section in ("pretrain", "tools_and_sft"):
        print(f"\n[{section}]")
        for name, spec in manifest.get(section, {}).items():
            repo_dir = paths.hf_cache_dir / spec["repo"]
            files = sorted(repo_dir.glob(spec["glob"])) if repo_dir.exists() else []
            n_bytes = sum(f.stat().st_size for f in files if f.is_file())
            total_bytes += n_bytes
            status = "OK" if files else "MISSING"
            print(
                f"  {status:7s} {name:18s} {spec['repo']:40s} "
                f"files={len(files):3d}  {n_bytes / 1e9:6.2f} GB"
            )
    print(f"\nTotal cached public data matched: {total_bytes / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
