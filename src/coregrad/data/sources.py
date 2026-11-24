"""Streaming dataset readers, text cleaning, and deterministic train/val splitting."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from coregrad.config import REPO_ROOT, load_local_paths, load_yaml

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_BLANK_RE = re.compile(r"\n{4,}")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo: str
    text_column: str
    glob: str
    weight: float
    files: tuple[Path, ...]


def clean_text(text: str, min_chars: int = 64, max_chars: int = 200_000) -> str | None:
    """Normalize and filter raw document text; return None to drop."""
    if not text:
        return None
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n\n", text).strip()
    n = len(text)
    if n < min_chars or n > max_chars:
        return None
    return text


def doc_hash_u64(text: str) -> int:
    """Deterministic 64-bit hash of normalized text for dedup and train/val split."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def is_val_hash(h: int, val_fraction: float = 0.005) -> bool:
    """Deterministic document-level hold-out based on 64-bit hash."""
    threshold = int(val_fraction * (1 << 32))
    return (h & 0xFFFFFFFF) < threshold


def load_pretrain_specs(
    repo_root: Path = REPO_ROOT, domains: list[str] | None = None
) -> list[DatasetSpec]:
    paths = load_local_paths(repo_root)
    if paths.hf_cache_dir is None or not paths.hf_cache_dir.exists():
        raise FileNotFoundError(
            "Local HF cache directory not found. Configure COREGRAD_HF_CACHE or configs/sources.local.yaml."
        )
    manifest = load_yaml(repo_root / "configs" / "datasets.yaml")
    specs: list[DatasetSpec] = []
    for name, cfg in manifest.get("pretrain", {}).items():
        if domains is not None and name not in domains:
            continue
        repo_dir = paths.hf_cache_dir / cfg["repo"]
        files = tuple(sorted(repo_dir.glob(cfg["glob"]))) if repo_dir.exists() else ()
        specs.append(
            DatasetSpec(
                name=name,
                repo=cfg["repo"],
                text_column=cfg["text_column"],
                glob=cfg["glob"],
                weight=float(cfg["weight"]),
                files=files,
            )
        )
    return specs


def iter_parquet_texts(
    files: tuple[Path, ...],
    text_column: str,
    batch_size: int = 2048,
    min_chars: int = 64,
    max_chars: int = 200_000,
    interleave_files: bool = True,
) -> Iterator[str]:
    """Stream cleaned texts from Parquet files.

    When interleave_files=True, cycles row-batches round-robin across files so
    multi-language/multi-topic directories (e.g. starcoderdata, cosmopedia) are
    balanced even when capped at a target token count.
    """
    if not files:
        return

    if not interleave_files or len(files) == 1:
        for path in files:
            pf = pq.ParquetFile(path)
            if text_column not in pf.schema_arrow.names:
                continue
            for batch in pf.iter_batches(batch_size=batch_size, columns=[text_column]):
                for item in batch.column(0).to_pylist():
                    if isinstance(item, str):
                        cleaned = clean_text(item, min_chars=min_chars, max_chars=max_chars)
                        if cleaned is not None:
                            yield cleaned
        return

    iters: list[Iterator[list[str | None]]] = []
    for path in files:
        pf = pq.ParquetFile(path)
        if text_column not in pf.schema_arrow.names:
            continue

        def _file_batches(pf_obj: pq.ParquetFile = pf) -> Iterator[list[str | None]]:
            for batch in pf_obj.iter_batches(batch_size=batch_size, columns=[text_column]):
                yield batch.column(0).to_pylist()

        iters.append(_file_batches())

    active = iters
    while active:
        next_active: list[Iterator[list[str | None]]] = []
        for it in active:
            try:
                rows = next(it)
            except StopIteration:
                continue
            for item in rows:
                if isinstance(item, str):
                    cleaned = clean_text(item, min_chars=min_chars, max_chars=max_chars)
                    if cleaned is not None:
                        yield cleaned
            next_active.append(it)
        active = next_active
