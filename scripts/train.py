#!/usr/bin/env python3
"""Train a coregrad language model with WSD schedule and live monitor logging."""

from __future__ import annotations

import argparse
from pathlib import Path

from coregrad.train import TrainConfig, train_pretrain


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrain a coregrad model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to train_*.yaml config")
    parser.add_argument("--run-name", type=str, default=None, help="Override run_name")
    parser.add_argument("--optimizer", type=str, default=None, choices=["adamw", "muon"])
    parser.add_argument("--total-tokens", type=int, default=None, help="Override total_tokens")
    parser.add_argument("--muon-lr", type=float, default=None, help="Override muon_lr")
    parser.add_argument("--peak-lr", type=float, default=None, help="Override peak_lr")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override weight_decay")
    parser.add_argument("--resume", action="store_true", help="Resume from latest.pt in run dir")
    parser.add_argument("--init-from", type=Path, default=None, help="Initialize weights from checkpoint")
    parser.add_argument("--init-optimizer", action="store_true", help="Also restore optimizer state from --init-from")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    overrides = {
        "run_name": args.run_name,
        "optimizer": args.optimizer,
        "total_tokens": args.total_tokens,
        "muon_lr": args.muon_lr,
        "peak_lr": args.peak_lr,
        "weight_decay": args.weight_decay,
    }
    if args.no_compile:
        overrides["compile_model"] = False

    cfg = TrainConfig.from_yaml(args.config, **overrides)
    train_pretrain(
        cfg,
        resume=args.resume,
        init_from=args.init_from,
        init_optimizer=True if args.init_optimizer else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
