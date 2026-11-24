#!/usr/bin/env python3
"""GPU micro-benchmark for coregrad models on a single consumer GPU.

Measures real training throughput (tokens/sec), effective MFU, and peak VRAM
across context lengths and micro-batch sizes with BF16 + Flash SDPA + torch.compile.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from coregrad.model import CoreGradLM, ModelConfig


def estimate_flops_per_token(cfg: ModelConfig, seq_len: int, non_emb_params: int) -> float:
    # Standard 6*N dense matmul FLOPs + 12*L*H*D*T causal attention FLOPs (6x fwd+bwd)
    dense_flops = 6.0 * non_emb_params
    attn_flops = 6.0 * cfg.n_layer * cfg.n_heads * cfg.head_dim * seq_len
    return dense_flops + attn_flops


def run_bench(
    config_path: Path,
    seq_len: int,
    micro_batch: int,
    grad_accum: int,
    steps: int,
    warmup: int,
    compile_model: bool,
    activation_ckpt: bool,
    peak_bf16_tflops: float,
) -> dict[str, float]:
    assert torch.cuda.is_available(), "CUDA GPU is required for bench_gpu.py"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    cfg = ModelConfig.from_yaml(
        config_path,
        max_seq_len=max(seq_len, 2048),
        activation_checkpointing=activation_ckpt,
    )
    model = CoreGradLM(cfg).to(device)
    total_params = model.num_parameters(non_embedding=False)
    non_emb_params = model.num_parameters(non_embedding=True)

    train_model = torch.compile(model) if compile_model else model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True
    )

    x = torch.randint(0, cfg.vocab_size, (micro_batch, seq_len), device=device)
    y = torch.randint(0, cfg.vocab_size, (micro_batch, seq_len), device=device)

    def one_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        loss_val = 0.0
        for _ in range(grad_accum):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = train_model(x, y)
                assert loss is not None
                loss = loss / grad_accum
            loss.backward()
            loss_val += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss_val

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    last_loss = 0.0
    for _ in range(steps):
        last_loss = one_step()
    torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0

    tokens_total = steps * grad_accum * micro_batch * seq_len
    tok_per_sec = tokens_total / dt
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    flops_per_tok = estimate_flops_per_token(cfg, seq_len, non_emb_params)
    achieved_tflops = (tok_per_sec * flops_per_tok) / 1e12
    mfu = achieved_tflops / peak_bf16_tflops * 100.0
    hrs_per_1b = (1e9 / tok_per_sec) / 3600.0

    print(
        f"[{cfg.name}] seq={seq_len} mb={micro_batch} ga={grad_accum} "
        f"compile={compile_model} ckpt={activation_ckpt} | "
        f"params={total_params/1e6:.2f}M (non-emb {non_emb_params/1e6:.2f}M) | "
        f"{tok_per_sec:,.0f} tok/s | {achieved_tflops:.1f} TFLOPS ({mfu:.1f}% MFU) | "
        f"peak={peak_gb:.2f} GB | {hrs_per_1b:.2f} h/1B tok | loss={last_loss:.3f}"
    )

    return {
        "tok_per_sec": tok_per_sec,
        "peak_gb": peak_gb,
        "achieved_tflops": achieved_tflops,
        "mfu": mfu,
        "hrs_per_1b": hrs_per_1b,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark coregrad training throughput on GPU.")
    parser.add_argument("--config", type=Path, default=Path("configs/model_125m.yaml"))
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--ckpt", action="store_true", help="Enable activation checkpointing")
    parser.add_argument(
        "--peak-bf16-tflops",
        type=float,
        default=160.0,
        help="GPU dense BF16 tensor core peak TFLOPS (RTX 4070 Ti ~160 TFLOPS)",
    )
    parser.add_argument("--sweep", action="store_true", help="Run standard Phase 0 benchmark sweep")
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / (1024**3)
    total_gb = total_bytes / (1024**3)
    print(
        f"Device: {gpu_name} ({free_gb:.2f} / {total_gb:.2f} GB VRAM free) | "
        f"PyTorch {torch.__version__}"
    )
    if free_gb < 4.0:
        print(
            "WARNING: Less than 4 GB VRAM is free — another process is likely occupying the GPU."
        )

    if args.sweep:
        cases = [
            (Path("configs/model_45m.yaml"), 2048, 8, 4, True, False),
            (Path("configs/model_125m.yaml"), 1024, 8, 4, True, False),
            (Path("configs/model_125m.yaml"), 2048, 4, 8, True, False),
            (Path("configs/model_125m.yaml"), 2048, 8, 4, True, True),
        ]
        for cfg_path, seq_len, mb, ga, comp, ckpt in cases:
            try:
                run_bench(cfg_path, seq_len, mb, ga, args.steps, args.warmup, comp, ckpt, args.peak_bf16_tflops)
            except torch.cuda.OutOfMemoryError:
                print(f"[{cfg_path.stem}] seq={seq_len} mb={mb} ckpt={ckpt} -> OOM")
                torch.cuda.empty_cache()
    else:
        run_bench(
            args.config,
            args.seq_len,
            args.micro_batch,
            args.grad_accum,
            args.steps,
            args.warmup,
            not args.no_compile,
            args.ckpt,
            args.peak_bf16_tflops,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
