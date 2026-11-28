#!/usr/bin/env python3
"""Gate 4 (Phase 5.3A) WSD Trainer for coregrad-Byte (H-Net Dynamic UTF-8 Patching + Outer GRU)."""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from coregrad.config import REPO_ROOT, load_local_paths
from coregrad.data import MixtureDataLoader
from coregrad.eval import (
    evaluate_completion_probes,
    evaluate_grammar_pairs,
    evaluate_validation_bpb,
    run_qualitative_suite,
    summarize_bpb,
)
from coregrad.model.hnet import CoreGradByteLM, HNetConfig
from coregrad.tokenizer.byte_tok import ByteTokenizer
from coregrad.train.monitor import RunLogger
from coregrad.train.optim import build_optimizer
from coregrad.train.schedule import wsd_lr

_CHUNK_PROBES = (
    "The quick brown fox jumps over the lazy dog.",
    "def fib(n): return n if n <= 1 else fib(n - 1) + fib(n - 2)",
    "prove([ x : treldon, treldon -> plendor ], x : plendor)",
)


def _mean_rep4(sample_list: list[dict[str, str]]) -> float:
    rates: list[float] = []
    for s in sample_list:
        words = s.get("completion", "").lower().split()
        if len(words) < 6:
            rates.append(0.0)
            continue
        grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
        rates.append(1.0 - len(set(grams)) / len(grams))
    return round(sum(rates) / max(len(rates), 1), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train coregrad-Byte (H-Net + Outer GRU) on UTF-8 byte shards.")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_32m_hnet_gru.yaml"))
    parser.add_argument("--run-name", type=str, default="gate4_32m_hnet_gru_1.2gb")
    parser.add_argument("--total-bytes", type=int, default=1_200_000_000, help="Total UTF-8 bytes (~300M BPE tok)")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    parser.add_argument("--peak-lr", type=float, default=8.0e-4)
    parser.add_argument("--min-lr", type=float, default=8.0e-5)
    parser.add_argument("--muon-lr", type=float, default=0.015)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-fraction", type=float, default=0.04)
    parser.add_argument("--decay-fraction", type=float, default=0.20)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shards-dir", type=Path, default=None)
    args = parser.parse_args()

    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
    paths = load_local_paths()
    shards_root = args.shards_dir or (paths.data_dir / "byte_shards")
    run_root = paths.runs_dir / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(args.seed)
    tokenizer = ByteTokenizer()

    mcfg_path = args.model_config if args.model_config.is_absolute() else (REPO_ROOT / args.model_config)
    mcfg = HNetConfig.from_yaml(mcfg_path, max_seq_len=max(args.seq_len, 4096))
    model = CoreGradByteLM(mcfg).to(dev)
    optimizer = build_optimizer(
        model,
        optimizer_type=args.optimizer,
        lr=args.peak_lr,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
    )

    mixture = {"edu-en": 0.45, "cosmo": 0.25, "code-starcoder": 0.17, "finemath": 0.13}
    bytes_per_step = args.seq_len * args.micro_batch * args.grad_accum
    total_steps = max(1, args.total_bytes // bytes_per_step)
    warmup_steps = int(total_steps * args.warmup_fraction)
    decay_steps = int(total_steps * args.decay_fraction)

    loader = MixtureDataLoader(
        shards_root, weights=mixture, seq_len=args.seq_len, batch_size=args.micro_batch, seed=args.seed, split="train"
    )
    logger = RunLogger(
        run_root,
        meta={
            "name": args.run_name,
            "model": mcfg.name,
            "params": model.num_parameters(),
            "non_emb_params": model.num_parameters(non_embedding=True),
            "optimizer": args.optimizer,
            "total_tokens": args.total_bytes,
            "total_steps": total_steps,
            "tokens_per_step": bytes_per_step,
            "seq_len": args.seq_len,
            "peak_lr": args.peak_lr,
            "mixture": mixture,
            "byte_mode": True,
        },
    )

    print(
        f"Starting {args.run_name} | {mcfg.name} ({model.num_parameters()/1e6:.2f}M params, "
        f"{model.num_parameters(non_embedding=True)/1e6:.2f}M non-emb, eff_L={mcfg.effective_layers}) | "
        f"opt={args.optimizer} | steps=0..{total_steps:,} ({bytes_per_step:,} B/step -> {args.total_bytes/1e6:,.1f}MB) | "
        f"target_R={mcfg.target_ratio} | device={dev}",
        flush=True,
    )

    amp_enabled = dev.type == "cuda"
    best_bpb = float("inf")
    t_run_start = time.perf_counter()

    def _step_vitals() -> tuple[float, float]:
        max_lgn = 0.0
        max_w = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None:
                max_lgn = max(max_lgn, float(p.grad.detach().float().norm().item()))
            if p.ndim >= 2 and "tok_emb" not in name and "lm_head" not in name and "router." not in name:
                max_w = max(max_w, float(p.detach().float().pow(2).mean().sqrt().item()))
        return max_lgn, max_w

    def _save(path: Path, step: int, bytes_seen: int) -> None:
        tmp = path.with_suffix(".pt.tmp")
        torch.save(
            {
                "step": step,
                "tokens_seen": bytes_seen,
                "best_bpb": best_bpb,
                "model_type": "hnet_byte",
                "model_config": asdict(mcfg),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            tmp,
        )
        tmp.replace(path)

    def run_eval(cur_step: int, cur_bytes: int) -> float:
        model.eval()
        # Evaluate ~524k UTF-8 bytes/domain (~131k BPE tokens/domain, matching BPE eval window!)
        bpb_res = evaluate_validation_bpb(
            model,
            shards_root,
            device=dev,
            seq_len=min(args.seq_len, 4096),
            batch_size=4,
            max_batches_per_domain=32,
            use_bf16=amp_enabled,
        )
        summary = summarize_bpb(bpb_res)
        mean_bpb = float(summary["mean_bpb"])  # type: ignore[arg-type]
        dom_bpb = {d: r.bpb for d, r in bpb_res.items()}
        gram = evaluate_grammar_pairs(model, tokenizer, device=dev, use_bf16=amp_enabled)
        comp = evaluate_completion_probes(model, tokenizer, device=dev, use_bf16=amp_enabled)
        samples = run_qualitative_suite(
            model, tokenizer, device=dev, max_new_tokens=240, temperature=0.7, top_k=40, limit=4
        )
        greedy = run_qualitative_suite(
            model, tokenizer, device=dev, max_new_tokens=200, temperature=0.0, top_k=None, limit=3
        )
        probe_x, _ = loader.next_batch(device=dev)
        vitals = model.compute_health_vitals(probe_x[:1, :1024])
        vitals["rep4_t07"] = _mean_rep4(samples)
        vitals["rep4_t0"] = _mean_rep4(greedy)

        logger.log_eval(
            step=cur_step,
            tokens_seen=cur_bytes,
            mean_bpb=mean_bpb,
            domain_bpb=dom_bpb,
            grammar_acc=float(gram["accuracy"]),
            completion_acc=float(comp["accuracy"]),  # type: ignore[arg-type]
            vitals=vitals,
        )
        logger.log_samples(step=cur_step, tokens_seen=cur_bytes, samples=samples)
        dom_str = " ".join(f"{k}={v:.3f}" for k, v in sorted(dom_bpb.items()))
        print(
            f"[eval step={cur_step:,} B={cur_bytes/1e6:.1f}M] mean_bpb={mean_bpb:.4f} ({dom_str}) | "
            f"grammar={gram['accuracy']*100:.1f}% | 4-way={float(comp['accuracy'])*100:.1f}% | "  # type: ignore[arg-type]
            f"R={vitals['compression_ratio']:.2f} wRMS={vitals['max_w_rms']:.3f} actRMS={vitals['act_rms']:.1f} "
            f"rep4(T0/T.7)={vitals['rep4_t0']*100:.0f}%/{vitals['rep4_t07']*100:.0f}% | "
            f"loop_cos={vitals['loop_cos']:.3f} gate={vitals['loop_gate_mean']:.2f}",
            flush=True,
        )
        for p_txt in _CHUNK_PROBES:
            chunks = model.inspect_chunks(p_txt, tokenizer, dev)
            preview = "|".join(chunks)
            print(f"  [chunks R={len(p_txt.encode('utf-8'))/max(len(chunks),1):.2f}] {preview}", flush=True)
        model.train()
        return mean_bpb

    run_eval(0, 0)
    model.train()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    t_win = time.perf_counter()
    bytes_win = 0
    ce_win = 0.0
    ratio_win = 0.0
    steps_win = 0
    bytes_seen = 0

    for step in range(1, total_steps + 1):
        lr = wsd_lr(
            step - 1,
            total_steps=total_steps,
            peak_lr=args.peak_lr,
            min_lr=args.min_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            decay_style="cosine",
        )
        optimizer.set_lr_scale(lr / args.peak_lr)
        optimizer.zero_grad(set_to_none=True)

        step_ce = 0.0
        step_r = 0.0
        for _ in range(args.grad_accum):
            x, y = loader.next_batch(device=dev)
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp_enabled):
                _, loss, aux = model(x, y, return_aux=True)  # type: ignore[misc]
                assert loss is not None
                (loss / args.grad_accum).backward()
            step_ce += float(aux["ce_loss"]) / args.grad_accum
            step_r += float(aux["compression_ratio"]) / args.grad_accum

        is_log = step % args.log_interval == 0 or step == total_steps
        max_lgn: float | None = None
        max_w: float | None = None
        if is_log:
            max_lgn, max_w = _step_vitals()

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()

        bytes_seen += bytes_per_step
        bytes_win += bytes_per_step
        ce_win += step_ce
        ratio_win += step_r
        steps_win += 1

        if is_log:
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
                vram_gb = torch.cuda.max_memory_allocated(dev) / (1024**3)
            else:
                vram_gb = 0.0
            now = time.perf_counter()
            b_per_s = bytes_win / max(now - t_win, 1e-6)
            avg_ce = ce_win / max(steps_win, 1)
            avg_bpb = avg_ce / math.log(2.0)
            avg_r = ratio_win / max(steps_win, 1)
            logger.log_step(
                step=step,
                tokens_seen=bytes_seen,
                train_loss=avg_ce,
                lr=lr,
                grad_norm=grad_norm,
                tok_per_sec=b_per_s,
                vram_gb=vram_gb,
                elapsed_s=now - t_run_start,
                max_layer_gn=max_lgn,
                max_w_rms=max_w,
            )
            print(
                f"step {step:5d}/{total_steps} | B {bytes_seen/1e6:7.2f}M | ce {avg_ce:.4f} (bpb {avg_bpb:.3f}) | "
                f"R {avg_r:.2f} | lr {lr:.2e} | gn {grad_norm:.2f} | wRMS {max_w or 0.0:.3f} | "
                f"{b_per_s:9,.0f} B/s | vram {vram_gb:.2f}GB",
                flush=True,
            )
            t_win = now
            bytes_win = 0
            ce_win = 0.0
            ratio_win = 0.0
            steps_win = 0

        if step % args.eval_interval == 0 or step == total_steps:
            mean_bpb = run_eval(step, bytes_seen)
            if mean_bpb < best_bpb:
                best_bpb = mean_bpb
                _save(run_root / "best.pt", step, bytes_seen)
            _save(run_root / "latest.pt", step, bytes_seen)
            t_win = time.perf_counter()
            bytes_win = 0

    print(
        f"\nFinished {args.run_name}: {bytes_seen/1e6:.2f}MB UTF-8 in {(time.perf_counter() - t_run_start)/60:.1f} min | "
        f"best_bpb={best_bpb:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
