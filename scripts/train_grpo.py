#!/usr/bin/env python3
"""Run verifiable GRPO (RLVR) polish (Stage 2C) on SQuAD v2 + synthetic retrieval + calc."""

from __future__ import annotations

import argparse
from pathlib import Path

from coregrad.train.grpo import GRPOConfig, train_grpo


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GRPO verifiable RL for coregrad Gate 2.")
    parser.add_argument("--init-from", type=Path, default=Path("runs/coregrad_125m_sft_v2/best.pt"))
    parser.add_argument("--run-name", type=str, default="coregrad_125m_grpo")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--kl-beta", type=float, default=0.05)
    parser.add_argument("--eval-interval", type=int, default=40)
    parser.add_argument("--no-squad", action="store_true")
    args = parser.parse_args()

    cfg = GRPOConfig(
        run_name=args.run_name,
        init_from=str(args.init_from),
        steps=args.steps,
        group_size=args.group_size,
        lr=args.lr,
        kl_beta=args.kl_beta,
        eval_interval=args.eval_interval,
        use_squad_v2=not args.no_squad,
    )
    train_grpo(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
