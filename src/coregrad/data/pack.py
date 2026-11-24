"""Memory-mapped uint16 binary shard packer with exact deduplication and BPB stats."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from coregrad.data.sources import doc_hash_u64, is_val_hash
from coregrad.tokenizer import CoreGradTokenizer


@dataclass
class SplitStats:
    shards: list[str]
    tokens: int
    utf8_bytes: int
    docs: int

    @property
    def bytes_per_token(self) -> float:
        return self.utf8_bytes / max(self.tokens, 1)


@dataclass
class DomainManifest:
    domain: str
    vocab_size: int
    train: SplitStats
    val: SplitStats
    duplicates_dropped: int


class _ShardWriter:
    def __init__(self, out_dir: Path, split: str, shard_tokens: int) -> None:
        self.out_dir = out_dir
        self.split = split
        self.shard_tokens = shard_tokens
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.buf = np.empty(shard_tokens, dtype=np.uint16)
        self.pos = 0
        self.shard_idx = 0
        self.shards: list[str] = []
        self.total_tokens = 0
        self.total_bytes = 0
        self.total_docs = 0

    def add_batch(self, token_lists: list[list[int]], byte_counts: list[int]) -> None:
        for ids, n_bytes in zip(token_lists, byte_counts, strict=True):
            arr = np.asarray(ids, dtype=np.uint16)
            n = len(arr)
            if n == 0:
                continue
            self.total_docs += 1
            self.total_bytes += n_bytes
            self.total_tokens += n

            offset = 0
            while offset < n:
                space = self.shard_tokens - self.pos
                take = min(space, n - offset)
                self.buf[self.pos : self.pos + take] = arr[offset : offset + take]
                self.pos += take
                offset += take
                if self.pos == self.shard_tokens:
                    self._flush()

    def _flush(self) -> None:
        if self.pos == 0:
            return
        fname = f"{self.split}_{self.shard_idx:04d}.bin"
        out_path = self.out_dir / fname
        self.buf[: self.pos].tofile(out_path)
        self.shards.append(fname)
        self.shard_idx += 1
        self.pos = 0

    def close(self) -> SplitStats:
        self._flush()
        return SplitStats(
            shards=self.shards,
            tokens=self.total_tokens,
            utf8_bytes=self.total_bytes,
            docs=self.total_docs,
        )


def pack_domain_shards(
    domain: str,
    texts: Iterator[str],
    tokenizer: CoreGradTokenizer,
    out_dir: Path,
    max_train_tokens: int,
    max_val_tokens: int = 5_000_000,
    val_fraction: float = 0.005,
    shard_tokens: int = 50_000_000,
    encode_batch_size: int = 512,
    dedup_seen: set[int] | None = None,
) -> DomainManifest:
    """Stream, deduplicate, tokenize, and pack a domain into uint16 train/val shards."""
    if tokenizer.vocab_size > 65535:
        raise ValueError(f"uint16 shards require vocab_size <= 65535, got {tokenizer.vocab_size}")

    seen = dedup_seen if dedup_seen is not None else set()
    domain_dir = out_dir / domain
    train_writer = _ShardWriter(domain_dir, "train", shard_tokens)
    val_writer = _ShardWriter(domain_dir, "val", min(shard_tokens, max(max_val_tokens, 1_000_000)))

    train_batch: list[str] = []
    train_bytes: list[int] = []
    val_batch: list[str] = []
    val_bytes: list[int] = []
    dups = 0
    t0 = time.perf_counter()
    last_log = t0

    def flush_train() -> None:
        if not train_batch:
            return
        ids = tokenizer.encode_batch(train_batch, add_eos=True)
        train_writer.add_batch(ids, train_bytes)
        train_batch.clear()
        train_bytes.clear()

    def flush_val() -> None:
        if not val_batch:
            return
        ids = tokenizer.encode_batch(val_batch, add_eos=True)
        val_writer.add_batch(ids, val_bytes)
        val_batch.clear()
        val_bytes.clear()

    for text in texts:
        h = doc_hash_u64(text)
        if h in seen:
            dups += 1
            continue
        seen.add(h)

        n_bytes = len(text.encode("utf-8"))
        if is_val_hash(h, val_fraction=val_fraction):
            if val_writer.total_tokens < max_val_tokens:
                val_batch.append(text)
                val_bytes.append(n_bytes)
                if len(val_batch) >= encode_batch_size:
                    flush_val()
        else:
            if train_writer.total_tokens < max_train_tokens:
                train_batch.append(text)
                train_bytes.append(n_bytes)
                if len(train_batch) >= encode_batch_size:
                    flush_train()

        if train_writer.total_tokens >= max_train_tokens and val_writer.total_tokens >= max_val_tokens:
            break

        now = time.perf_counter()
        if now - last_log >= 15.0:
            elapsed = now - t0
            rate = (train_writer.total_tokens + val_writer.total_tokens) / max(elapsed, 1e-6)
            print(
                f"  [{domain}] train={train_writer.total_tokens/1e6:,.1f}M / {max_train_tokens/1e6:,.0f}M "
                f"val={val_writer.total_tokens/1e6:,.2f}M dups={dups:,} ({rate/1e6:.2f}M tok/s)",
                flush=True,
            )
            last_log = now

    flush_train()
    flush_val()

    train_stats = train_writer.close()
    val_stats = val_writer.close()

    manifest = DomainManifest(
        domain=domain,
        vocab_size=tokenizer.vocab_size,
        train=train_stats,
        val=val_stats,
        duplicates_dropped=dups,
    )
    (domain_dir / "domain_manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2), encoding="utf-8"
    )
    return manifest
