#!/usr/bin/env python3
"""Train masked SFT (Stage 2B) from a pretrained Gate 1 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from coregrad.train.sft import SFTConfig, train_sft


def main() -> int:
    parser = argparse.ArgumentParser(description="Run masked SFT for coregrad Gate 2.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_125m_sft.yaml"))
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    overrides: dict[str, object] = {
        "init_from": str(args.init_from) if args.init_from else None,
        "run_name": args.run_name,
        "max_steps": args.max_steps,
    }
    if args.no_compile:
        overrides["compile_model"] = False

    cfg = SFTConfig.from_yaml(args.config, **overrides)
    train_sft(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
