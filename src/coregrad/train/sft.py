"""Masked Supervised Fine-Tuning (SFT) trainer with optional 2048->4096 RoPE extension."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from coregrad.config import load_local_paths, load_yaml
from coregrad.model import CoreGradLM, ModelConfig
from coregrad.tokenizer import CoreGradTokenizer
from coregrad.tools import Turn, encode_masked_trajectory
from coregrad.train.monitor import RunLogger
from coregrad.train.optim import build_optimizer


@dataclass
class SFTConfig:
    run_name: str = "coregrad_125m_sft"
    init_from: str = "runs/coregrad_125m_base/best.pt"
    train_jsonl: str = "data/phase3/sft/train.jsonl"
    val_jsonl: str = "data/phase3/sft/val.jsonl"
    optimizer: str = "adamw"
    seq_len: int = 2048
    rope_theta_override: float | None = None
    micro_batch: int = 4
    grad_accum: int = 8
    epochs: int = 2
    max_steps: int | None = None
    peak_lr: float = 1.5e-4
    min_lr: float = 1.5e-5
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    warmup_fraction: float = 0.05
    compile_model: bool = True
    activation_checkpointing: bool = True
    seed: int = 1337
    log_interval: int = 10
    eval_interval: int = 100

    @classmethod
    def from_yaml(cls, path: Path | str, **overrides: Any) -> SFTConfig:
        data = load_yaml(Path(path))
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


def load_encoded_sft_examples(
    jsonl_path: Path,
    tokenizer: CoreGradTokenizer,
    max_seq_len: int,
    max_examples: int | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Load Turn JSONL records into masked (ids, labels) sequences."""
    examples: list[tuple[list[int], list[int]]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            turns = [Turn(t["role"], t["content"]) for t in obj["turns"]]
            ids, labels = encode_masked_trajectory(turns, tokenizer, max_seq_len=max_seq_len + 1)
            if len(ids) >= 4 and any(lbl != -100 for lbl in labels[1:]):
                examples.append((ids, labels))
                if max_examples is not None and len(examples) >= max_examples:
                    break
    return examples


def pack_sft_batch(
    batch_examples: list[tuple[list[int], list[int]]],
    seq_len: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad/truncate a mini-batch of masked examples to (B, seq_len) inputs and targets."""
    bsz = len(batch_examples)
    x = torch.full((bsz, seq_len), pad_id, dtype=torch.long)
    y = torch.full((bsz, seq_len), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(batch_examples):
        n = min(len(ids) - 1, seq_len)
        x[i, :n] = torch.tensor(ids[:n], dtype=torch.long)
        y[i, :n] = torch.tensor(labels[1 : n + 1], dtype=torch.long)
    return x, y


@torch.no_grad()
def evaluate_sft_loss(
    model: CoreGradLM,
    val_examples: list[tuple[list[int], list[int]]],
    seq_len: int,
    pad_id: int,
    batch_size: int,
    device: torch.device,
    max_batches: int = 32,
) -> float:
    model.eval()
    amp_enabled = device.type == "cuda"
    total_loss = 0.0
    count = 0
    for start in range(0, min(len(val_examples), max_batches * batch_size), batch_size):
        chunk = val_examples[start : start + batch_size]
        if not chunk:
            break
        x, y = pack_sft_batch(chunk, seq_len=seq_len, pad_id=pad_id)
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            _, loss = model(x, y)
        if loss is not None and torch.isfinite(loss):
            total_loss += float(loss.item())
            count += 1
    model.train()
    return total_loss / max(count, 1)


def train_sft(
    cfg: SFTConfig,
    *,
    tokenizer_path: Path | None = None,
    runs_dir: Path | None = None,
    device: torch.device | None = None,
    max_train_examples: int | None = None,
) -> Path:
    """Run masked SFT from a pretrained Gate 1 checkpoint."""
    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
    paths = load_local_paths()
    tok_path = tokenizer_path or (paths.data_dir / "tokenizer" / "tokenizer.json")
    run_root = (runs_dir or paths.runs_dir) / cfg.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CoreGradTokenizer.from_file(tok_path)

    ckpt = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
    model_cfg_dict = dict(ckpt["model_config"])
    model_cfg_dict["max_seq_len"] = max(int(model_cfg_dict.get("max_seq_len", 2048)), cfg.seq_len)
    model_cfg_dict["activation_checkpointing"] = cfg.activation_checkpointing
    if cfg.rope_theta_override is not None:
        model_cfg_dict["rope_theta"] = cfg.rope_theta_override
    model_cfg = ModelConfig(**model_cfg_dict)
    model = CoreGradLM(model_cfg).to(dev)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    train_examples = load_encoded_sft_examples(
        Path(cfg.train_jsonl), tokenizer, max_seq_len=cfg.seq_len, max_examples=max_train_examples
    )
    val_examples = load_encoded_sft_examples(
        Path(cfg.val_jsonl), tokenizer, max_seq_len=cfg.seq_len, max_examples=512
    )
    if not train_examples:
        raise ValueError(f"No valid SFT examples loaded from {cfg.train_jsonl}")

    steps_per_epoch = max(1, len(train_examples) // (cfg.micro_batch * cfg.grad_accum))
    total_steps = cfg.max_steps if cfg.max_steps is not None else max(1, steps_per_epoch * cfg.epochs)
    warmup_steps = int(total_steps * cfg.warmup_fraction)

    optimizer = build_optimizer(
        model, optimizer_type=cfg.optimizer, lr=cfg.peak_lr, weight_decay=cfg.weight_decay
    )
    train_model: torch.nn.Module = (
        torch.compile(model) if (cfg.compile_model and dev.type == "cuda") else model
    )

    tokens_per_step = cfg.seq_len * cfg.micro_batch * cfg.grad_accum
    logger = RunLogger(
        run_root,
        meta={
            "name": cfg.run_name,
            "stage": "sft",
            "model": model_cfg.name,
            "params": model.num_parameters(),
            "total_tokens": total_steps * tokens_per_step,
            "total_steps": total_steps,
            "seq_len": cfg.seq_len,
            "train_examples": len(train_examples),
        },
    )

    rng = random.Random(cfg.seed)
    order = list(range(len(train_examples)))
    rng.shuffle(order)
    cursor = 0

    def next_microbatch() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal cursor
        batch_items: list[tuple[list[int], list[int]]] = []
        for _ in range(cfg.micro_batch):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            batch_items.append(train_examples[order[cursor]])
            cursor += 1
        x, y = pack_sft_batch(batch_items, seq_len=cfg.seq_len, pad_id=tokenizer.pad_id)
        return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)

    amp_enabled = dev.type == "cuda"
    best_val = float("inf")
    tokens_seen = 0
    t_start = time.perf_counter()
    t_win = t_start
    tok_win = 0
    loss_acc = 0.0
    n_acc = 0

    model.train()
    for step in range(1, total_steps + 1):
        if warmup_steps > 0 and step <= warmup_steps:
            lr = cfg.peak_lr * (step / warmup_steps)
        else:
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr = cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))) * (cfg.peak_lr - cfg.min_lr)
        optimizer.set_lr_scale(lr / cfg.peak_lr)
        optimizer.zero_grad(set_to_none=True)

        step_loss = 0.0
        for _ in range(cfg.grad_accum):
            x, y = next_microbatch()
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp_enabled):
                _, loss = train_model(x, y)
                assert loss is not None
                scaled = loss / cfg.grad_accum
            scaled.backward()
            step_loss += float(scaled.detach().item())

        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item())
        optimizer.step()
        tokens_seen += tokens_per_step
        tok_win += tokens_per_step
        loss_acc += step_loss
        n_acc += 1

        if step % cfg.log_interval == 0 or step == total_steps:
            now = time.perf_counter()
            tok_s = tok_win / max(now - t_win, 1e-6)
            vram = torch.cuda.max_memory_allocated(dev) / (1024**3) if dev.type == "cuda" else 0.0
            avg_l = loss_acc / max(n_acc, 1)
            logger.log_step(
                step=step,
                tokens_seen=tokens_seen,
                train_loss=avg_l,
                lr=lr,
                grad_norm=gn,
                tok_per_sec=tok_s,
                vram_gb=vram,
                elapsed_s=now - t_start,
            )
            print(
                f"[sft] step {step:5d}/{total_steps} | loss {avg_l:.4f} | lr {lr:.2e} | "
                f"gn {gn:.2f} | {tok_s:,.0f} tok/s",
                flush=True,
            )
            t_win = now
            tok_win = 0
            loss_acc = 0.0
            n_acc = 0

        if step % cfg.eval_interval == 0 or step == total_steps:
            val_loss = evaluate_sft_loss(
                model, val_examples, cfg.seq_len, tokenizer.pad_id, cfg.micro_batch, dev
            )
            logger.log_eval(
                step=step,
                tokens_seen=tokens_seen,
                mean_bpb=val_loss,
                domain_bpb={"sft_val_loss": val_loss},
            )
            print(f"[sft eval step={step}] val_loss={val_loss:.4f}", flush=True)
            payload = {
                "step": step,
                "tokens_seen": tokens_seen,
                "val_loss": val_loss,
                "model_config": asdict(model.cfg),
                "sft_config": asdict(cfg),
                "model_state_dict": model.state_dict(),
            }
            torch.save(payload, run_root / "latest.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(payload, run_root / "best.pt")

    return run_root
