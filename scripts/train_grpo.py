#!/usr/bin/env python3
"""Run verifiable GRPO (RLVR) polish (Stage 2C) on calc & fictional retrieval environments."""

from __future__ import annotations

import argparse
from pathlib import Path

from coregrad.train.grpo import GRPOConfig, train_grpo


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GRPO verifiable RL for coregrad Gate 2.")
    parser.add_argument("--init-from", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="coregrad_125m_grpo")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    cfg = GRPOConfig(
        run_name=args.run_name,
        init_from=str(args.init_from),
        steps=args.steps,
        group_size=args.group_size,
        lr=args.lr,
    )
    train_grpo(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
