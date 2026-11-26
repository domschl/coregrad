#!/usr/bin/env python3
"""Prepare Phase 3 (Gate 2) datasets on CPU: synthetic calc/retrieval + normalized public tool/chat SFT.

Outputs (under data/phase3/):
- sft/train.jsonl & sft/val.jsonl : structured Turn trajectories for masked SFT
- midtrain/                       : packed uint16 shards for Stage 2A cognitive mid-training
- eval_tasks.json                 : locked held-out Gate 2 verifiable benchmark tasks
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from coregrad.config import load_local_paths
from coregrad.data.pack import pack_domain_shards
from coregrad.synthetic import (
    generate_calc_tasks,
    generate_retrieval_tasks,
    iter_glaive_trajectories,
    iter_gsm8k_trajectories,
    iter_hermes_trajectories,
    iter_smoltalk_trajectories,
    iter_squad_v2_trajectories,
    iter_xlam_trajectories,
    load_cached_public_tool_files,
)
from coregrad.tokenizer import CoreGradTokenizer
from coregrad.tools import Turn, render_trajectory


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _turns_to_record(source: str, turns: list[Turn]) -> dict[str, object]:
    return {"source": source, "turns": [{"role": t.role, "content": t.content} for t in turns]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase 3 synthetic + public tool SFT/midtrain datasets.")
    parser.add_argument("--n-calc", type=int, default=15000)
    parser.add_argument("--n-retrieval", type=int, default=20000)
    parser.add_argument("--n-squad", type=int, default=55000)
    parser.add_argument("--n-gsm8k-repeats", type=int, default=2)
    parser.add_argument("--n-xlam", type=int, default=5000)
    parser.add_argument("--n-hermes", type=int, default=5000)
    parser.add_argument("--n-glaive", type=int, default=5000)
    parser.add_argument("--n-smoltalk", type=int, default=45000)
    parser.add_argument("--n-eval-calc", type=int, default=100)
    parser.add_argument("--n-eval-retrieval", type=int, default=120)
    parser.add_argument("--skip-midtrain-shards", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    paths = load_local_paths()
    out_root = paths.data_dir / "phase3"
    out_root.mkdir(parents=True, exist_ok=True)

    print("Generating deterministic synthetic calc & fictional multi-hop retrieval trajectories...", flush=True)
    calc_train = generate_calc_tasks(args.n_calc, seed=args.seed + 1)
    ret_train = generate_retrieval_tasks(args.n_retrieval, seed=args.seed + 2, split="train")

    # Strictly disjoint eval seed + disjoint fictional entity vocabulary (split="eval")
    calc_eval = generate_calc_tasks(args.n_eval_calc, seed=args.seed + 9001)
    ret_eval = generate_retrieval_tasks(args.n_eval_retrieval, seed=args.seed + 9002, split="eval")

    records: list[dict[str, object]] = []
    records.extend(_turns_to_record(t.category, t.turns) for t in calc_train)
    records.extend(_turns_to_record(t.category, t.turns) for t in ret_train)
    print(f"  synthetic: calc={len(calc_train):,}, retrieval={len(ret_train):,}", flush=True)

    cached = load_cached_public_tool_files()
    for name, fn, limit in (
        ("rag-squad-v2", lambda fs, limit: iter_squad_v2_trajectories(fs, limit=limit, seed=args.seed), args.n_squad),
        ("tool-xlam", iter_xlam_trajectories, args.n_xlam),
        ("tool-hermes", iter_hermes_trajectories, args.n_hermes),
        ("tool-glaive", iter_glaive_trajectories, args.n_glaive),
        ("sft-smoltalk", iter_smoltalk_trajectories, args.n_smoltalk),
    ):
        files = cached.get(name, [])
        if not files or limit <= 0:
            print(f"  [skip] {name} (files={len(files)})", flush=True)
            continue
        loaded = [_turns_to_record(name, turns) for turns in fn(files, limit=limit)]
        records.extend(loaded)
        print(f"  public {name:14s}: {len(loaded):,} trajectories", flush=True)

    gsm_files = cached.get("math-gsm8k", [])
    if gsm_files and args.n_gsm8k_repeats > 0:
        gsm_loaded = [_turns_to_record("math-gsm8k", turns) for turns in iter_gsm8k_trajectories(gsm_files)]
        records.extend(gsm_loaded * args.n_gsm8k_repeats)
        print(
            f"  public {'math-gsm8k':14s}: {len(gsm_loaded):,} unique x {args.n_gsm8k_repeats} "
            f"= {len(gsm_loaded) * args.n_gsm8k_repeats:,} trajectories",
            flush=True,
        )

    rng = random.Random(args.seed)
    rng.shuffle(records)

    n_val = max(200, min(2000, len(records) // 50))
    val_records = records[:n_val]
    train_records = records[n_val:]

    _write_jsonl(out_root / "sft" / "train.jsonl", train_records)
    _write_jsonl(out_root / "sft" / "val.jsonl", val_records)
    print(f"Wrote SFT JSONL: train={len(train_records):,}, val={len(val_records):,}", flush=True)

    eval_payload = {
        "calc_tasks": [
            {
                "task_id": t.task_id,
                "category": t.category,
                "question": t.question,
                "expected_answer": t.expected_answer,
                "requires_tool": t.requires_tool,
            }
            for t in calc_eval
        ],
        "retrieval_tasks": [
            {
                "task_id": t.task_id,
                "category": t.category,
                "question": t.question,
                "expected_answer": t.expected_answer,
                "expected_doc_ids": list(t.expected_doc_ids),
                "should_abstain": t.should_abstain,
                "documents": [asdict(d) for d in t.documents],
            }
            for t in ret_eval
        ],
    }
    (out_root / "eval_tasks.json").write_text(
        json.dumps(eval_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Wrote locked Gate 2 eval suite: calc={len(calc_eval)}, retrieval={len(ret_eval)}",
        flush=True,
    )

    tok_path = paths.data_dir / "tokenizer" / "tokenizer.json"
    if not args.skip_midtrain_shards and tok_path.exists():
        tokenizer = CoreGradTokenizer.from_file(tok_path)
        rendered = (
            render_trajectory([Turn(t["role"], t["content"]) for t in r["turns"]], add_bos=True, add_eos=False)  # type: ignore[index]
            for r in records
        )
        mid_dir = out_root / "midtrain"
        dm = pack_domain_shards(
            domain="cognitive-core",
            texts=rendered,
            tokenizer=tokenizer,
            out_dir=mid_dir,
            max_train_tokens=500_000_000,
            max_val_tokens=4_000_000,
            val_fraction=0.01,
            shard_tokens=50_000_000,
        )
        manifest = {
            "vocab_size": tokenizer.vocab_size,
            "tokenizer_path": tok_path.name,
            "domains": {"cognitive-core": asdict(dm)},
        }
        (mid_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            f"Packed mid-training shards: train={dm.train.tokens/1e6:.2f}M tok, "
            f"val={dm.val.tokens/1e6:.2f}M tok",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
