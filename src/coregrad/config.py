"""Configuration loading for coregrad."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LocalPaths:
    hf_cache_dir: Path | None
    data_dir: Path
    runs_dir: Path


def _expand_path(raw: str | None, base: Path) -> Path | None:
    if not raw:
        return None
    expanded = os.path.expandvars(raw)
    if "${" in expanded:
        return None
    p = Path(expanded).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_local_paths(repo_root: Path = REPO_ROOT) -> LocalPaths:
    """Resolve dataset cache and working directories from env or configs/sources.local.yaml."""
    local_cfg_path = repo_root / "configs" / "sources.local.yaml"
    cfg: dict[str, Any] = load_yaml(local_cfg_path) if local_cfg_path.exists() else {}

    hf_cache_raw = os.environ.get("COREGRAD_HF_CACHE") or cfg.get("hf_cache_dir")
    data_raw = os.environ.get("COREGRAD_DATA_DIR") or cfg.get("data_dir", "data")
    runs_raw = os.environ.get("COREGRAD_RUNS_DIR") or cfg.get("runs_dir", "runs")

    return LocalPaths(
        hf_cache_dir=_expand_path(hf_cache_raw, repo_root),
        data_dir=_expand_path(data_raw, repo_root) or (repo_root / "data"),
        runs_dir=_expand_path(runs_raw, repo_root) or (repo_root / "runs"),
    )
