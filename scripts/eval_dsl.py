#!/usr/bin/env python3
"""Evaluate any coregrad checkpoint on the locked Phase 5 coregrad-DSL benchmark."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch

from coregrad.config import load_local_paths
from coregrad.model import load_checkpoint_model_and_tokenizer
from coregrad.tools import ToolExecutor
from coregrad.tools.rollout import answer_matches, run_agent_rollout


def _dsl_answer_matches(final_answer: str, expected: str, level: str) -> bool:
    if answer_matches(final_answer, expected):
        return True
    # Normalize whitespace/formatting for algebraic expressions, lists, and dicts
    fa_compact = re.sub(r"\s+", "", final_answer.lower())
    ex_compact = re.sub(r"\s+", "", expected.lower())
    if ex_compact and ex_compact in fa_compact:
        return True
    if level == "logic":
        matches = re.findall(r"\b(true|false|unknown)\b", final_answer, re.IGNORECASE)
        if matches and matches[-1].lower() == expected.lower():
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on locked Phase 5 coregrad-DSL benchmark.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    paths = load_local_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, tokenizer, _ = load_checkpoint_model_and_tokenizer(args.checkpoint, device)
    model.eval()

    eval_path = args.eval_json or (paths.data_dir / "phase5" / "dsl_eval_tasks.json")
    tasks: list[dict[str, Any]] = json.loads(eval_path.read_text(encoding="utf-8"))["tasks"]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    executor = ToolExecutor(None)
    by_level: dict[str, list[int]] = {}
    tool_valid = 0
    tool_total = 0
    hits = 0
    samples: dict[str, dict[str, Any]] = {}

    for t in tasks:
        level = str(t["level"])
        max_toks = 260 if level == "logic" else 180
        tr = run_agent_rollout(
            model,
            tokenizer,
            str(t["question"]),
            executor,
            device,
            max_tool_turns=2,
            max_new_tokens_per_turn=max_toks,
            temperature=0.0,
        )
        ok = int(_dsl_answer_matches(tr.final_answer, str(t["expected_answer"]), level))
        hits += ok
        tool_valid += tr.tool_calls_valid
        tool_total += tr.tool_calls_total
        by_level.setdefault(level, []).append(ok)
        if level not in samples:
            samples[level] = {
                "question": str(t["question"])[:220],
                "expected": t["expected_answer"],
                "calls": tr.tool_calls_raw,
                "final_answer": tr.final_answer[:220],
                "ok": bool(ok),
            }

    level_scores = {
        lvl: round(sum(vals) / max(len(vals), 1), 4) for lvl, vals in sorted(by_level.items())
    }
    report = {
        "checkpoint": str(args.checkpoint),
        "model_name": model.cfg.name,
        "params_m": round(model.num_parameters() / 1e6, 2),
        "overall_accuracy": round(hits / max(len(tasks), 1), 4),
        "hits": hits,
        "total": len(tasks),
        "tool_valid_rate": round(tool_valid / max(tool_total, 1), 4),
        "tool_calls_valid": tool_valid,
        "tool_calls_total": tool_total,
        "by_level": level_scores,
        "samples": samples,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }

    out_path = args.out or (paths.runs_dir / "eval" / f"dsl_eval_{args.checkpoint.parent.name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lvl_str = " | ".join(f"{k}={v*100:.1f}%" for k, v in level_scores.items())
    print(
        f"[eval_dsl] {model.cfg.name} ({args.checkpoint.parent.name}): "
        f"overall={report['overall_accuracy']*100:.1f}% ({hits}/{len(tasks)}) | "
        f"tool_valid={report['tool_valid_rate']*100:.1f}% ({tool_valid}/{tool_total}) | "
        f"{lvl_str} -> {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
