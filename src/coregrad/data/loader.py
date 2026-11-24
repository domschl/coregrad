"""Memory-mapped domain-weighted batch loader for pretraining and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class DomainShardSplit:
    """Memory-mapped view over all uint16 shards for one domain and split."""

    def __init__(self, domain_dir: Path, shard_names: list[str], bytes_per_token: float) -> None:
        self.domain_dir = domain_dir
        self.bytes_per_token = bytes_per_token
        self.arrays = [
            np.memmap(domain_dir / name, dtype=np.uint16, mode="r") for name in shard_names
        ]
        self.lengths = [len(a) for a in self.arrays]
        self.total_tokens = sum(self.lengths)
        self.cum_lengths = np.cumsum(self.lengths)

    def sample_windows(
        self, rng: np.random.Generator, batch_size: int, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample random contiguous windows of length seq_len + 1."""
        if self.total_tokens <= seq_len + 1:
            raise ValueError(
                f"Split in {self.domain_dir} has only {self.total_tokens} tokens, need > {seq_len + 1}"
            )
        x = np.empty((batch_size, seq_len), dtype=np.int64)
        y = np.empty((batch_size, seq_len), dtype=np.int64)

        # Sample shard proportional to usable tokens in each shard
        usable = np.array([max(0, n - (seq_len + 1)) for n in self.lengths], dtype=np.float64)
        probs = usable / usable.sum()
        shard_ids = rng.choice(len(self.arrays), size=batch_size, p=probs)

        for i, sid in enumerate(shard_ids):
            arr = self.arrays[int(sid)]
            start = int(rng.integers(0, len(arr) - seq_len - 1))
            chunk = arr[start : start + seq_len + 1]
            x[i] = chunk[:-1]
            y[i] = chunk[1:]

        return torch.from_numpy(x), torch.from_numpy(y)

    def iter_sequential_batches(
        self, batch_size: int, seq_len: int, max_batches: int | None = None
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Deterministic non-overlapping windows for exact validation BPB."""
        stride = seq_len
        windows: list[np.ndarray] = []
        target_windows = None if max_batches is None else max_batches * batch_size

        for arr in self.arrays:
            limit = len(arr) - seq_len - 1
            for start in range(0, max(0, limit), stride):
                windows.append(arr[start : start + seq_len + 1])
                if target_windows is not None and len(windows) >= target_windows:
                    break
            if target_windows is not None and len(windows) >= target_windows:
                break

        batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        n_full = len(windows) // batch_size
        for b in range(n_full):
            chunk = np.stack(windows[b * batch_size : (b + 1) * batch_size]).astype(np.int64)
            x = torch.from_numpy(np.ascontiguousarray(chunk[:, :-1]))
            y = torch.from_numpy(np.ascontiguousarray(chunk[:, 1:]))
            batches.append((x, y))
        return batches


class MixtureDataLoader:
    """Samples training batches dynamically across packed domain shards."""

    def __init__(
        self,
        shards_root: Path,
        weights: dict[str, float],
        seq_len: int,
        batch_size: int,
        seed: int = 1337,
        split: str = "train",
    ) -> None:
        manifest = json.loads((shards_root / "manifest.json").read_text(encoding="utf-8"))
        self.domains: list[str] = []
        self.splits: dict[str, DomainShardSplit] = {}
        raw_weights: list[float] = []

        for domain, w in weights.items():
            if w <= 0 or domain not in manifest["domains"]:
                continue
            d_info = manifest["domains"][domain]
            s_info = d_info[split]
            if not s_info["shards"]:
                continue
            bpt = s_info["utf8_bytes"] / max(s_info["tokens"], 1)
            self.domains.append(domain)
            self.splits[domain] = DomainShardSplit(shards_root / domain, s_info["shards"], bpt)
            raw_weights.append(float(w))

        if not self.domains:
            raise ValueError(f"No valid {split} domains found in {shards_root}")

        total_w = sum(raw_weights)
        self.probs = np.array([w / total_w for w in raw_weights], dtype=np.float64)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def next_batch(self, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Sample each sequence's domain independently according to mixture weights
        counts = self.rng.multinomial(self.batch_size, self.probs)
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        for domain, count in zip(self.domains, counts, strict=True):
            if count > 0:
                xd, yd = self.splits[domain].sample_windows(self.rng, int(count), self.seq_len)
                xs.append(xd)
                ys.append(yd)
        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)
        perm = torch.from_numpy(self.rng.permutation(self.batch_size))
        x, y = x[perm], y[perm]
        if device is not None:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        return x, y
