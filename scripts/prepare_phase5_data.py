#!/usr/bin/env python3
"""Prepare Phase 5 unified cognitive-core dataset (Phase 3 SFT v2 + BPE-aligned coregrad-DSL).

Outputs (under data/phase5/):
- sft/train.jsonl & sft/val.jsonl : unified masked trajectories (Phase 3 + 4-level DSL + multi-depth logic DSL)
- dsl_eval_tasks.json             : locked held-out 160-task coregrad-DSL benchmark (40 per level)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from coregrad.config import load_local_paths
from coregrad.synthetic.dsl_gen import generate_dsl_tasks
from coregrad.synthetic.logic_gen import generate_logic_tasks
from coregrad.tools import Turn


def _turns_to_record(source: str, turns: list[Turn]) -> dict[str, object]:
    return {"source": source, "turns": [{"role": t.role, "content": t.content} for t in turns]}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase 5 unified SFT + coregrad-DSL dataset.")
    parser.add_argument("--n-dsl", type=int, default=24000, help="4-level coregrad-DSL trajectories")
    parser.add_argument("--n-logic-dsl", type=int, default=8000, help="Open-vocab multi-depth prove() trajectories")
    parser.add_argument("--n-eval-per-level", type=int, default=40, help="Held-out tasks per DSL level")
    parser.add_argument("--seed", type=int, default=5026)
    args = parser.parse_args()

    paths = load_local_paths()
    p3_train = paths.data_dir / "phase3" / "sft" / "train.jsonl"
    p3_val = paths.data_dir / "phase3" / "sft" / "val.jsonl"
    if not p3_train.exists():
        raise FileNotFoundError(f"Missing Phase 3 SFT data: {p3_train}. Run scripts/prepare_phase3_data.py first.")

    out_root = paths.data_dir / "phase5"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading existing Phase 3 SFT trajectories from {p3_train}...", flush=True)
    train_records = _read_jsonl(p3_train)
    val_records = _read_jsonl(p3_val) if p3_val.exists() else []
    print(f"  Phase 3 base: train={len(train_records):,}, val={len(val_records):,}", flush=True)

    print("Generating open-vocab BPE-aligned 4-level coregrad-DSL trajectories...", flush=True)
    dsl_tasks = generate_dsl_tasks(args.n_dsl, seed=args.seed + 1)
    dsl_records = [_turns_to_record(f"dsl-{t.level}", t.turns) for t in dsl_tasks]

    print("Generating open-vocab BPE-aligned multi-depth logic prove() trajectories...", flush=True)
    logic_tasks = generate_logic_tasks(
        args.n_logic_dsl,
        depths=(2, 3, 4, 5),
        split="train",
        seed=args.seed + 2,
        n_distractors=3,
        cot=False,
        notation="dsl",
        open_vocab=True,
    )
    logic_records = [_turns_to_record(f"dsl-logic-d{t.depth}", t.turns) for t in logic_tasks]

    new_records = dsl_records + logic_records
    rng = random.Random(args.seed)
    rng.shuffle(new_records)

    n_new_val = max(200, min(600, len(new_records) // 50))
    val_records.extend(new_records[:n_new_val])
    train_records.extend(new_records[n_new_val:])
    rng.shuffle(train_records)
    rng.shuffle(val_records)

    _write_jsonl(out_root / "sft" / "train.jsonl", train_records)
    _write_jsonl(out_root / "sft" / "val.jsonl", val_records)
    print(
        f"Wrote Phase 5 SFT JSONL: train={len(train_records):,} (+{len(new_records) - n_new_val:,} DSL), "
        f"val={len(val_records):,} (+{n_new_val:,} DSL)",
        flush=True,
    )

    # Locked held-out coregrad-DSL eval suite (40 per level; logic uses strictly disjoint _EVAL_PREDS)
    eval_tasks: list[dict[str, object]] = []
    eval_logic = generate_logic_tasks(
        args.n_eval_per_level,
        depths=(2, 4, 5, 6, 8),
        split="eval",
        seed=95001,
        n_distractors=3,
        cot=False,
        notation="dsl",
    )
    for t in eval_logic:
        eval_tasks.append(
            {
                "task_id": t.task_id,
                "level": "logic",
                "category": f"{t.category}_d{t.depth}",
                "question": t.question,
                "expected_answer": t.expected_answer,
            }
        )

    # Pull held-out bindings / pipelines / symbolic from strictly disjoint eval seed
    other_pool = generate_dsl_tasks(args.n_eval_per_level * 8, seed=95002)
    counts = {"bindings": 0, "pipelines": 0, "symbolic": 0}
    for t in other_pool:
        if t.level in counts and counts[t.level] < args.n_eval_per_level:
            counts[t.level] += 1
            eval_tasks.append(
                {
                    "task_id": t.task_id,
                    "level": t.level,
                    "category": t.level,
                    "question": t.question,
                    "expected_answer": t.expected_answer,
                }
            )

    eval_path = out_root / "dsl_eval_tasks.json"
    eval_path.write_text(json.dumps({"tasks": eval_tasks}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote locked Phase 5 DSL eval suite: {len(eval_tasks)} tasks -> {eval_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
