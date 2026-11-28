"""Masked Supervised Fine-Tuning (SFT) trainer with sequence packing and live agent rollout monitoring."""

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
from coregrad.model import CoreGradByteLM, CoreGradLM, load_checkpoint_model_and_tokenizer
from coregrad.tokenizer import CoreGradTokenizer
from coregrad.tools import (
    Document,
    InMemoryBM25Retriever,
    ToolExecutor,
    Turn,
    encode_masked_trajectory,
)
from coregrad.tools.rollout import answer_matches, run_agent_rollout
from coregrad.train.monitor import RunLogger
from coregrad.train.optim import build_optimizer


@dataclass
class SFTConfig:
    run_name: str = "coregrad_125m_sft"
    init_from: str = "runs/coregrad_125m_base/best.pt"
    train_jsonl: str = "data/phase3/sft/train.jsonl"
    val_jsonl: str = "data/phase3/sft/val.jsonl"
    eval_tasks_json: str = "data/phase3/eval_tasks.json"
    optimizer: str = "adamw"
    seq_len: int = 2048
    rope_theta_override: float | None = None
    pack_sequences: bool = True
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
    activation_checkpointing: bool = False
    seed: int = 1337
    log_interval: int = 10
    eval_interval: int = 200
    rollout_probe_count: int = 4

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


def pack_examples_to_rows(
    examples: list[tuple[list[int], list[int]]],
    seq_len: int,
    pad_id: int,
    seed: int = 1337,
) -> list[tuple[list[int], list[int]]]:
    """Pack multiple masked conversations into fixed seq_len+1 rows to eliminate padding waste.

    Because each trajectory starts with (bos_id, label=-100) and ends with (eos_id, label=eos_id),
    concatenating multiple conversations never trains across conversation boundaries.
    """
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)

    target_len = seq_len + 1
    rows: list[tuple[list[int], list[int]]] = []
    cur_ids: list[int] = []
    cur_lbl: list[int] = []

    for idx in order:
        ids, lbls = examples[idx]
        if len(ids) > target_len:
            ids = ids[:target_len]
            lbls = lbls[:target_len]
        if cur_ids and len(cur_ids) + len(ids) > target_len:
            pad_n = target_len - len(cur_ids)
            if pad_n > 0:
                cur_ids.extend([pad_id] * pad_n)
                cur_lbl.extend([-100] * pad_n)
            rows.append((cur_ids, cur_lbl))
            cur_ids = []
            cur_lbl = []
        cur_ids.extend(ids)
        cur_lbl.extend(lbls)

    if cur_ids and any(x != -100 for x in cur_lbl[1:]):
        pad_n = target_len - len(cur_ids)
        if pad_n > 0:
            cur_ids.extend([pad_id] * pad_n)
            cur_lbl.extend([-100] * pad_n)
        rows.append((cur_ids, cur_lbl))

    return rows


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


def _select_live_rollout_tasks(eval_tasks_path: Path, count: int) -> list[dict[str, Any]]:
    """Pick 1 representative task per category (calc, 1-hop, 2-hop, abstention) for live monitoring."""
    if count <= 0 or not eval_tasks_path.exists():
        return []
    data = json.loads(eval_tasks_path.read_text(encoding="utf-8"))
    picked: list[dict[str, Any]] = []
    calc_tasks = data.get("calc_tasks", [])
    if calc_tasks:
        picked.append({"kind": "calc", **calc_tasks[0]})

    ret_tasks = data.get("retrieval_tasks", [])
    seen_cats: set[str] = set()
    for t in ret_tasks:
        cat = str(t.get("category", ""))
        if cat not in seen_cats:
            seen_cats.add(cat)
            picked.append({"kind": "retrieval", **t})
            if len(picked) >= count:
                break
    return picked[:count]


@torch.no_grad()
def run_live_rollout_probes(
    model: CoreGradLM,
    tokenizer: CoreGradTokenizer,
    tasks: list[dict[str, Any]],
    device: torch.device,
) -> tuple[list[dict[str, str]], float, float]:
    """Run live agent rollouts for monitor display + syntax/answer accuracy tracking."""
    if not tasks:
        return [], 0.0, 0.0
    model.eval()
    samples: list[dict[str, str]] = []
    valid_calls = 0
    total_calls = 0
    hits = 0

    for t in tasks:
        if t["kind"] == "retrieval":
            docs = [Document(**d) for d in t["documents"]]
            executor = ToolExecutor(InMemoryBM25Retriever(docs))
            should_abstain = bool(t.get("should_abstain", False))
        else:
            executor = ToolExecutor(None)
            should_abstain = False

        tr = run_agent_rollout(
            model,
            tokenizer,
            t["question"],
            executor,
            device,
            max_tool_turns=3,
            max_new_tokens_per_turn=140,
            temperature=0.0,
        )
        valid_calls += tr.tool_calls_valid
        total_calls += tr.tool_calls_total
        ok = answer_matches(tr.final_answer, t["expected_answer"], should_abstain=should_abstain)
        hits += int(ok)

        status = "PASS" if ok else "MISS"
        calls_str = " | ".join(tr.tool_calls_raw) if tr.tool_calls_raw else "(no tool calls)"
        cites_str = ", ".join(tr.cited_doc_ids) if tr.cited_doc_ids else "none"
        completion = (
            f"[{status} | {t['category']} | expected={t['expected_answer']}]\n"
            f"Calls: {calls_str}\n"
            f"Citations: {cites_str}\n"
            f"Answer: {tr.final_answer}"
        )
        samples.append({"prompt": t["question"], "completion": completion})

    model.train()
    syntax_acc = valid_calls / max(total_calls, 1)
    ans_acc = hits / max(len(tasks), 1)
    return samples, syntax_acc, ans_acc


def train_sft(
    cfg: SFTConfig,
    *,
    tokenizer_path: Path | None = None,
    runs_dir: Path | None = None,
    device: torch.device | None = None,
    max_train_examples: int | None = None,
) -> Path:
    """Run masked SFT (with sequence packing & live agent rollout monitoring) from a Gate 1 checkpoint."""
    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
    paths = load_local_paths()
    tok_path = tokenizer_path or (paths.data_dir / "tokenizer" / "tokenizer.json")
    run_root = (runs_dir or paths.runs_dir) / cfg.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, tokenizer, _ckpt = load_checkpoint_model_and_tokenizer(
        cfg.init_from,
        dev,
        tokenizer_path=tok_path,
        min_seq_len=cfg.seq_len,
        activation_checkpointing=cfg.activation_checkpointing,
        rope_theta_override=cfg.rope_theta_override,
    )
    model_cfg = model.cfg
    is_byte_model = isinstance(model, CoreGradByteLM)

    print(f"Loading & encoding SFT examples from {cfg.train_jsonl} ...", flush=True)
    raw_train = load_encoded_sft_examples(
        Path(cfg.train_jsonl), tokenizer, max_seq_len=cfg.seq_len, max_examples=max_train_examples
    )
    raw_val = load_encoded_sft_examples(
        Path(cfg.val_jsonl), tokenizer, max_seq_len=cfg.seq_len, max_examples=1024
    )
    if not raw_train:
        raise ValueError(f"No valid SFT examples loaded from {cfg.train_jsonl}")

    if cfg.pack_sequences:
        train_rows: list[tuple[list[int], list[int]]] = []
        for ep in range(max(1, cfg.epochs)):
            train_rows.extend(
                pack_examples_to_rows(raw_train, cfg.seq_len, tokenizer.pad_id, seed=cfg.seed + ep)
            )
        val_rows = pack_examples_to_rows(raw_val, cfg.seq_len, tokenizer.pad_id, seed=cfg.seed + 999)
        print(
            f"Packed {len(raw_train):,} train examples x {cfg.epochs} epochs -> {len(train_rows):,} rows "
            f"(seq_len={cfg.seq_len}); val -> {len(val_rows):,} rows",
            flush=True,
        )
    else:
        train_rows = raw_train * max(1, cfg.epochs)
        val_rows = raw_val

    rows_per_step = cfg.micro_batch * cfg.grad_accum
    total_steps = cfg.max_steps if cfg.max_steps is not None else max(1, len(train_rows) // rows_per_step)
    warmup_steps = int(total_steps * cfg.warmup_fraction)

    optimizer = build_optimizer(
        model, optimizer_type=cfg.optimizer, lr=cfg.peak_lr, weight_decay=cfg.weight_decay
    )
    train_model: torch.nn.Module = (
        torch.compile(model) if (cfg.compile_model and dev.type == "cuda" and not is_byte_model) else model
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
            "train_examples": len(raw_train),
            "packed_rows": len(train_rows),
        },
    )

    probe_tasks = _select_live_rollout_tasks(Path(cfg.eval_tasks_json), cfg.rollout_probe_count)

    rng = random.Random(cfg.seed)
    order = list(range(len(train_rows)))
    rng.shuffle(order)
    cursor = 0

    def next_microbatch() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal cursor
        batch_items: list[tuple[list[int], list[int]]] = []
        for _ in range(cfg.micro_batch):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            batch_items.append(train_rows[order[cursor]])
            cursor += 1
        x, y = pack_sft_batch(batch_items, seq_len=cfg.seq_len, pad_id=tokenizer.pad_id)
        return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)

    amp_enabled = dev.type == "cuda"
    best_val = float("inf")
    tokens_seen = 0
    t_start = time.perf_counter()

    print(
        f"Starting SFT {cfg.run_name} | {model_cfg.name} ({model.num_parameters()/1e6:.2f}M) | "
        f"steps={total_steps:,} ({tokens_per_step:,} tok/step -> {total_steps*tokens_per_step/1e6:.1f}M tok) | "
        f"packed={cfg.pack_sequences} | device={dev}",
        flush=True,
    )

    # Step-0 baseline eval + rollout snapshot
    val0 = evaluate_sft_loss(model, val_rows, cfg.seq_len, tokenizer.pad_id, cfg.micro_batch, dev)
    samples0, syn0, ans0 = run_live_rollout_probes(model, tokenizer, probe_tasks, dev)
    logger.log_eval(
        step=0,
        tokens_seen=0,
        mean_bpb=val0,
        domain_bpb={"sft_val_loss": val0},
        grammar_acc=syn0 if probe_tasks else None,
        completion_acc=ans0 if probe_tasks else None,
    )
    if samples0:
        logger.log_samples(step=0, tokens_seen=0, samples=samples0)
    print(f"[sft eval step=0] val_loss={val0:.4f} | probe_syntax={syn0*100:.0f}% | probe_ans={ans0*100:.0f}%", flush=True)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    t_win = time.perf_counter()
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
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
                vram = torch.cuda.max_memory_allocated(dev) / (1024**3)
            else:
                vram = 0.0
            now = time.perf_counter()
            tok_s = tok_win / max(now - t_win, 1e-6)
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
                f"[sft] step {step:5d}/{total_steps} | tok {tokens_seen/1e6:6.1f}M | "
                f"loss {avg_l:.4f} | lr {lr:.2e} | gn {gn:.2f} | {tok_s:,.0f} tok/s | vram {vram:.2f}GB",
                flush=True,
            )
            t_win = now
            tok_win = 0
            loss_acc = 0.0
            n_acc = 0

        if step % cfg.eval_interval == 0 or step == total_steps:
            val_loss = evaluate_sft_loss(
                model, val_rows, cfg.seq_len, tokenizer.pad_id, cfg.micro_batch, dev
            )
            samples, syn_acc, ans_acc = run_live_rollout_probes(model, tokenizer, probe_tasks, dev)
            logger.log_eval(
                step=step,
                tokens_seen=tokens_seen,
                mean_bpb=val_loss,
                domain_bpb={"sft_val_loss": val_loss},
                grammar_acc=syn_acc if probe_tasks else None,
                completion_acc=ans_acc if probe_tasks else None,
            )
            if samples:
                logger.log_samples(step=step, tokens_seen=tokens_seen, samples=samples)
            print(
                f"[sft eval step={step}] val_loss={val_loss:.4f} | "
                f"probe_syntax={syn_acc*100:.0f}% | probe_ans={ans_acc*100:.0f}%",
                flush=True,
            )
            payload = {
                "step": step,
                "tokens_seen": tokens_seen,
                "val_loss": val_loss,
                "model_type": "hnet_byte" if is_byte_model else "bpe_lm",
                "model_config": asdict(model.cfg),
                "sft_config": asdict(cfg),
                "model_state_dict": model.state_dict(),
            }
            tmp_latest = run_root / "latest.pt.tmp"
            torch.save(payload, tmp_latest)
            tmp_latest.replace(run_root / "latest.pt")
            if val_loss < best_val:
                best_val = val_loss
                tmp_best = run_root / "best.pt.tmp"
                torch.save(payload, tmp_best)
                tmp_best.replace(run_root / "best.pt")
            t_win = time.perf_counter()
            tok_win = 0

    print(
        f"\nFinished SFT {cfg.run_name} in {(time.perf_counter() - t_start)/60:.1f} min | best_val_loss={best_val:.4f}",
        flush=True,
    )
    return run_root
