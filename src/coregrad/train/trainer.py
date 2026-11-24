"""Core pretraining loop for coregrad with WSD schedule, checkpointing, and live monitor logging."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from coregrad.config import REPO_ROOT, load_local_paths, load_yaml
from coregrad.data import MixtureDataLoader
from coregrad.eval import (
    evaluate_completion_probes,
    evaluate_grammar_pairs,
    evaluate_validation_bpb,
    run_qualitative_suite,
    summarize_bpb,
)
from coregrad.model import CoreGradLM, ModelConfig
from coregrad.tokenizer import CoreGradTokenizer
from coregrad.train.monitor import RunLogger
from coregrad.train.optim import build_optimizer
from coregrad.train.schedule import wsd_lr


@dataclass
class TrainConfig:
    run_name: str = "coregrad_12m_smoke"
    model_config: str = "configs/model_12m.yaml"
    optimizer: str = "adamw"
    total_tokens: int = 150_000_000
    seq_len: int = 1024
    micro_batch: int = 16
    grad_accum: int = 4
    peak_lr: float = 1.0e-3
    min_lr: float = 1.0e-4
    muon_lr: float = 0.02
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_fraction: float = 0.03
    decay_fraction: float = 0.15
    decay_style: str = "cosine"
    compile_model: bool = True
    activation_checkpointing: bool = False
    seed: int = 1337
    log_interval: int = 10
    eval_interval: int = 200
    eval_val_batches: int = 16
    sample_count: int = 6
    sample_max_tokens: int = 96
    save_interval: int = 500
    mixture: dict[str, float] = field(
        default_factory=lambda: {
            "edu-en": 0.45,
            "cosmo": 0.25,
            "code-starcoder": 0.17,
            "finemath": 0.13,
        }
    )

    @property
    def tokens_per_step(self) -> int:
        return self.seq_len * self.micro_batch * self.grad_accum

    @property
    def total_steps(self) -> int:
        return max(1, self.total_tokens // self.tokens_per_step)

    @classmethod
    def from_yaml(cls, path: Path | str, **overrides: Any) -> TrainConfig:
        data = load_yaml(Path(path))
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


def _save_checkpoint(
    path: Path,
    *,
    model: CoreGradLM,
    optimizer: Any,
    step: int,
    tokens_seen: int,
    best_bpb: float,
    train_cfg: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "step": step,
            "tokens_seen": tokens_seen,
            "best_bpb": best_bpb,
            "model_config": asdict(model.cfg),
            "train_config": asdict(train_cfg),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        tmp_path,
    )
    tmp_path.replace(path)


def train_pretrain(
    cfg: TrainConfig,
    *,
    resume: bool = False,
    init_from: Path | None = None,
    shards_dir: Path | None = None,
    tokenizer_path: Path | None = None,
    runs_dir: Path | None = None,
    device: torch.device | None = None,
) -> Path:
    """Run WSD pretraining with live monitor logging, periodic evals, and sample generations."""
    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
    paths = load_local_paths()
    shards_root = shards_dir or (paths.data_dir / "shards")
    tok_path = tokenizer_path or (paths.data_dir / "tokenizer" / "tokenizer.json")
    run_root = (runs_dir or paths.runs_dir) / cfg.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(cfg.seed)
    tokenizer = CoreGradTokenizer.from_file(tok_path)

    model_cfg_path = Path(cfg.model_config)
    if not model_cfg_path.is_absolute():
        model_cfg_path = REPO_ROOT / model_cfg_path
    model_cfg = ModelConfig.from_yaml(
        model_cfg_path,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=max(cfg.seq_len, 2048),
        activation_checkpointing=cfg.activation_checkpointing,
    )
    model = CoreGradLM(model_cfg).to(dev)
    optimizer = build_optimizer(
        model,
        optimizer_type=cfg.optimizer,
        lr=cfg.peak_lr,
        muon_lr=cfg.muon_lr,
        weight_decay=cfg.weight_decay,
    )

    start_step = 0
    tokens_seen = 0
    best_bpb = float("inf")

    latest_ckpt = run_root / "latest.pt"
    if resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt["step"])
        tokens_seen = int(ckpt["tokens_seen"])
        best_bpb = float(ckpt.get("best_bpb", float("inf")))
        print(f"[resume] Loaded {latest_ckpt} at step={start_step:,} ({tokens_seen/1e6:.2f}M tok)", flush=True)
    elif init_from is not None:
        ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[init-from] Loaded weights from {init_from}", flush=True)

    train_model: torch.nn.Module = (
        torch.compile(model) if (cfg.compile_model and dev.type == "cuda") else model
    )

    total_steps = cfg.total_steps
    warmup_steps = int(total_steps * cfg.warmup_fraction)
    decay_steps = int(total_steps * cfg.decay_fraction)

    logger = RunLogger(
        run_root,
        meta={
            "name": cfg.run_name,
            "model": model_cfg.name,
            "params": model.num_parameters(),
            "non_emb_params": model.num_parameters(non_embedding=True),
            "optimizer": cfg.optimizer,
            "total_tokens": cfg.total_tokens,
            "total_steps": total_steps,
            "tokens_per_step": cfg.tokens_per_step,
            "seq_len": cfg.seq_len,
            "peak_lr": cfg.peak_lr,
            "mixture": cfg.mixture,
        },
    )

    # Shift loader seed on resume so resumed runs see fresh batches
    loader = MixtureDataLoader(
        shards_root,
        weights=cfg.mixture,
        seq_len=cfg.seq_len,
        batch_size=cfg.micro_batch,
        seed=cfg.seed + start_step,
        split="train",
    )

    print(
        f"Starting {cfg.run_name} | {model_cfg.name} ({model.num_parameters()/1e6:.2f}M params) | "
        f"opt={cfg.optimizer} | steps={start_step:,}..{total_steps:,} "
        f"({cfg.tokens_per_step:,} tok/step -> {cfg.total_tokens/1e6:,.1f}M tok) | device={dev}",
        flush=True,
    )

    amp_enabled = dev.type == "cuda"
    t_run_start = time.perf_counter()

    def run_eval_and_samples(cur_step: int, cur_tokens: int) -> float:
        model.eval()
        bpb_res = evaluate_validation_bpb(
            model,
            shards_root,
            device=dev,
            seq_len=min(cfg.seq_len, 1024),
            batch_size=max(1, min(cfg.micro_batch, 8)),
            max_batches_per_domain=cfg.eval_val_batches,
            use_bf16=amp_enabled,
        )
        summary = summarize_bpb(bpb_res)
        mean_bpb = float(summary["mean_bpb"])  # type: ignore[arg-type]
        dom_bpb = {d: r.bpb for d, r in bpb_res.items()}
        gram = evaluate_grammar_pairs(model, tokenizer, device=dev, use_bf16=amp_enabled)
        comp = evaluate_completion_probes(model, tokenizer, device=dev, use_bf16=amp_enabled)
        samples = run_qualitative_suite(
            model,
            tokenizer,
            device=dev,
            max_new_tokens=cfg.sample_max_tokens,
            temperature=0.7,
            top_k=40,
            limit=cfg.sample_count,
        )
        logger.log_eval(
            step=cur_step,
            tokens_seen=cur_tokens,
            mean_bpb=mean_bpb,
            domain_bpb=dom_bpb,
            grammar_acc=float(gram["accuracy"]),
            completion_acc=float(comp["accuracy"]),  # type: ignore[arg-type]
        )
        logger.log_samples(step=cur_step, tokens_seen=cur_tokens, samples=samples)
        dom_str = " ".join(f"{k}={v:.3f}" for k, v in sorted(dom_bpb.items()))
        print(
            f"[eval step={cur_step:,} tok={cur_tokens/1e6:.1f}M] mean_bpb={mean_bpb:.4f} "
            f"({dom_str}) | grammar={gram['accuracy']*100:.1f}% | 4-way={float(comp['accuracy'])*100:.1f}%",  # type: ignore[arg-type]
            flush=True,
        )
        model.train()
        return mean_bpb

    if start_step == 0:
        run_eval_and_samples(0, 0)

    model.train()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    t_window = time.perf_counter()
    tok_window = 0
    loss_accum_log = 0.0
    steps_in_log = 0

    for step in range(start_step + 1, total_steps + 1):
        lr = wsd_lr(
            step - 1,
            total_steps=total_steps,
            peak_lr=cfg.peak_lr,
            min_lr=cfg.min_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            decay_style=cfg.decay_style,
        )
        optimizer.set_lr_scale(lr / cfg.peak_lr)
        optimizer.zero_grad(set_to_none=True)

        step_loss = 0.0
        for _ in range(cfg.grad_accum):
            x, y = loader.next_batch(device=dev)
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp_enabled):
                _, loss = train_model(x, y)
                assert loss is not None
                scaled_loss = loss / cfg.grad_accum
            scaled_loss.backward()
            step_loss += float(scaled_loss.detach().item())

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item())
        optimizer.step()

        tokens_seen += cfg.tokens_per_step
        tok_window += cfg.tokens_per_step
        loss_accum_log += step_loss
        steps_in_log += 1

        if step % cfg.log_interval == 0 or step == total_steps:
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
                vram_gb = torch.cuda.max_memory_allocated(dev) / (1024**3)
            else:
                vram_gb = 0.0
            now = time.perf_counter()
            dt = max(now - t_window, 1e-6)
            tok_s = tok_window / dt
            avg_loss = loss_accum_log / max(steps_in_log, 1)
            elapsed_s = now - t_run_start

            logger.log_step(
                step=step,
                tokens_seen=tokens_seen,
                train_loss=avg_loss,
                lr=lr,
                grad_norm=grad_norm,
                tok_per_sec=tok_s,
                vram_gb=vram_gb,
                elapsed_s=elapsed_s,
            )
            print(
                f"step {step:6d}/{total_steps} | tok {tokens_seen/1e6:7.2f}M | "
                f"loss {avg_loss:.4f} | lr {lr:.2e} | gn {grad_norm:.2f} | "
                f"{tok_s:8,.0f} tok/s | vram {vram_gb:.2f}GB",
                flush=True,
            )
            t_window = now
            tok_window = 0
            loss_accum_log = 0.0
            steps_in_log = 0

        if step % cfg.eval_interval == 0 or step == total_steps:
            mean_bpb = run_eval_and_samples(step, tokens_seen)
            if mean_bpb < best_bpb:
                best_bpb = mean_bpb
                _save_checkpoint(
                    run_root / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    tokens_seen=tokens_seen,
                    best_bpb=best_bpb,
                    train_cfg=cfg,
                )
            _save_checkpoint(
                latest_ckpt,
                model=model,
                optimizer=optimizer,
                step=step,
                tokens_seen=tokens_seen,
                best_bpb=best_bpb,
                train_cfg=cfg,
            )
            t_window = time.perf_counter()
            tok_window = 0

        if cfg.save_interval > 0 and step % cfg.save_interval == 0:
            _save_checkpoint(
                run_root / f"ckpt_step_{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                tokens_seen=tokens_seen,
                best_bpb=best_bpb,
                train_cfg=cfg,
            )

    print(
        f"\nFinished {cfg.run_name}: {tokens_seen/1e6:.2f}M tokens in "
        f"{(time.perf_counter() - t_run_start)/60:.1f} min | best_bpb={best_bpb:.4f}",
        flush=True,
    )
    return run_root
