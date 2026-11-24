#!/usr/bin/env python3
"""Run Gate 1 evaluation suite (validation BPB + grammar/completion probes + qualitative prompts)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from coregrad.config import load_local_paths
from coregrad.eval import (
    evaluate_completion_probes,
    evaluate_grammar_pairs,
    evaluate_validation_bpb,
    run_qualitative_suite,
    summarize_bpb,
)
from coregrad.model import CoreGradLM, ModelConfig
from coregrad.tokenizer import CoreGradTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a coregrad model on Gate 1 metrics.")
    parser.add_argument("--config", type=Path, default=Path("configs/model_12m.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--shards-dir", type=Path, default=None)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=16)
    parser.add_argument("--qual-limit", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    paths = load_local_paths()
    tok_path = args.tokenizer or (paths.data_dir / "tokenizer" / "tokenizer.json")
    shards_dir = args.shards_dir or (paths.data_dir / "shards")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CoreGradTokenizer.from_file(tok_path)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**ckpt["model_config"])
        model = CoreGradLM(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        tag = args.checkpoint.stem
    else:
        cfg = ModelConfig.from_yaml(args.config, vocab_size=tokenizer.vocab_size)
        model = CoreGradLM(cfg)
        tag = f"{cfg.name}_random_init"

    model = model.to(device)
    print(f"Evaluating {tag} ({model.num_parameters()/1e6:.2f}M params) on {device}")

    report: dict[str, object] = {
        "tag": tag,
        "params": model.num_parameters(),
        "timestamp": int(time.time()),
    }

    if (shards_dir / "manifest.json").exists():
        bpb_res = evaluate_validation_bpb(
            model,
            shards_dir,
            device=device,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_batches_per_domain=args.max_val_batches,
        )
        summary = summarize_bpb(bpb_res)
        report["bpb"] = summary
        print(f"Mean validation BPB: {summary['mean_bpb']:.4f}")
        for dom, stats in bpb_res.items():
            print(
                f"  {dom:16s} | loss={stats.loss_nats:.4f} | ppl={stats.perplexity:8.2f} | "
                f"bpb={stats.bpb:.4f} ({stats.tokens_evaluated:,} tok)"
            )
    else:
        print(f"[skip] No shard manifest found in {shards_dir}")

    grammar = evaluate_grammar_pairs(model, tokenizer, device=device)
    completions = evaluate_completion_probes(model, tokenizer, device=device)
    qual = run_qualitative_suite(model, tokenizer, device=device, limit=args.qual_limit)

    report["grammar_pairs"] = grammar
    report["completion_probes"] = completions
    report["qualitative"] = qual

    print(
        f"Grammar minimal pairs: {grammar['accuracy']*100:.1f}% "
        f"({int(grammar['correct'])}/{int(grammar['total'])})"
    )
    print(
        f"4-way completion probes: {completions['accuracy']*100:.1f}% "
        f"({completions['correct']}/{completions['total']}) | {completions['by_category']}"
    )

    out_path = args.out or (paths.runs_dir / "eval" / f"{tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved evaluation report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
