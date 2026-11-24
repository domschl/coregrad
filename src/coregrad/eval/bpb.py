"""Tokenizer-agnostic Bits-Per-Byte (BPB) validation evaluator."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from coregrad.data.loader import DomainShardSplit
from coregrad.model import CoreGradLM


@dataclass
class DomainBPBResult:
    domain: str
    loss_nats: float
    perplexity: float
    bpb: float
    bytes_per_token: float
    tokens_evaluated: int


@torch.no_grad()
def evaluate_validation_bpb(
    model: CoreGradLM,
    shards_root: Path,
    device: torch.device,
    seq_len: int = 1024,
    batch_size: int = 8,
    max_batches_per_domain: int = 32,
    use_bf16: bool = True,
) -> dict[str, DomainBPBResult]:
    """Compute exact cross-entropy loss and BPB across all held-out val domains."""
    manifest = json.loads((shards_root / "manifest.json").read_text(encoding="utf-8"))
    model.eval()
    results: dict[str, DomainBPBResult] = {}

    amp_enabled = use_bf16 and device.type == "cuda"
    for domain, d_info in manifest.get("domains", {}).items():
        val_info = d_info.get("val", {})
        if not val_info.get("shards"):
            continue
        bpt = val_info["utf8_bytes"] / max(val_info["tokens"], 1)
        split = DomainShardSplit(shards_root / domain, val_info["shards"], bpt)
        batches = split.iter_sequential_batches(
            batch_size=batch_size, seq_len=seq_len, max_batches=max_batches_per_domain
        )
        if not batches:
            continue

        total_loss = 0.0
        total_tokens = 0
        for x, y in batches:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                _, loss = model(x, y)
            assert loss is not None
            n_tok = int(y.numel())
            total_loss += float(loss.item()) * n_tok
            total_tokens += n_tok

        mean_loss = total_loss / max(total_tokens, 1)
        bpb = mean_loss / (math.log(2.0) * bpt)
        results[domain] = DomainBPBResult(
            domain=domain,
            loss_nats=mean_loss,
            perplexity=math.exp(min(mean_loss, 20.0)),
            bpb=bpb,
            bytes_per_token=bpt,
            tokens_evaluated=total_tokens,
        )
    return results


def summarize_bpb(results: dict[str, DomainBPBResult]) -> dict[str, object]:
    if not results:
        return {"mean_bpb": float("nan"), "domains": {}}
    mean_bpb = sum(r.bpb for r in results.values()) / len(results)
    return {
        "mean_bpb": mean_bpb,
        "domains": {k: asdict(v) for k, v in results.items()},
    }
