#!/usr/bin/env python3
"""Evaluate a coregrad checkpoint on the locked Gate 2 Cognitive Core suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from coregrad.config import load_local_paths
from coregrad.eval.gate2 import evaluate_gate2
from coregrad.model import load_checkpoint_model_and_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gate 2 tool-use & grounded retrieval evaluation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-tasks", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--no-tools", action="store_true", help="Disable tools for baseline ablation")
    parser.add_argument("--limit", type=int, default=None, help="Cap tasks per category for quick check")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    paths = load_local_paths()
    tok_path = args.tokenizer or (paths.data_dir / "tokenizer" / "tokenizer.json")
    eval_path = args.eval_tasks or (paths.data_dir / "phase3" / "eval_tasks.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, _ = load_checkpoint_model_and_tokenizer(args.checkpoint, device, tokenizer_path=tok_path)

    report = evaluate_gate2(
        model,
        tokenizer,
        eval_path,
        device=device,
        enable_tools=not args.no_tools,
        max_calc_tasks=args.limit,
        max_retrieval_tasks=args.limit,
    )
    print(f"Gate 2 Evaluation ({args.checkpoint.name}, tools={'ON' if not args.no_tools else 'OFF'}):")
    print(
        f"  Tool syntax validity       : {report['tool_syntax_validity']*100:5.1f}% "
        f"({report['tool_calls_valid']}/{report['tool_calls_total']})"
    )
    print(
        f"  Calc exact accuracy        : {report['calc_accuracy']*100:5.1f}% "
        f"({report['calc_hits']}/{report['calc_total']})"
    )
    print(
        f"  Grounded retrieval accuracy: {report['grounded_retrieval_accuracy']*100:5.1f}% "
        f"({report['grounded_retrieval_hits']}/{report['grounded_retrieval_total']})"
    )
    print(
        f"  Grounded citation rate     : {report['grounded_citation_rate']*100:5.1f}% "
        f"(hallucinated={report['hallucinated_citations']})"
    )
    print(
        f"  Abstention accuracy        : {report['abstention_accuracy']*100:5.1f}% "
        f"({report['abstention_hits']}/{report['abstention_total']})"
    )

    out_path = args.out or (paths.runs_dir / "eval" / f"gate2_{args.checkpoint.stem}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
