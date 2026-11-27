#!/usr/bin/env python3
"""Phase 4A Option A: Multi-Step Syllogism & Horn-Clause Depth Generalization Probe.

Fine-tunes a lightweight probe branch from any coregrad checkpoint on procedural logic chains
of depths D in {2, 3, 4, 5}, then evaluates strict accuracy across both in-distribution
depths (2, 4, 5) and out-of-distribution depths (6, 8, 10, 12) on disjoint fictional vocabularies.

Supports both:
  --mode cot    : Explicit <|think|> proof chains (cognitive-core scratchpad scaling)
  --mode direct : Direct answer without <|think|> (tests latent depth of dense vs looped models)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from coregrad.config import load_local_paths
from coregrad.model import CoreGradLM, ModelConfig
from coregrad.synthetic.logic_gen import VerifiableLogicTask, generate_logic_tasks
from coregrad.tokenizer import CoreGradTokenizer
from coregrad.tokenizer.special_tokens import (
    ASSISTANT_END,
    CALL_END,
    RESULT_END,
    RESULT_START,
    SYSTEM_END,
    SYSTEM_START,
    USER_END,
    USER_START,
)
from coregrad.tools.executor import ToolExecutor
from coregrad.tools.protocol import (
    DEFAULT_SYSTEM_PROMPT,
    encode_masked_trajectory,
    extract_tool_calls,
    strip_think_blocks,
)

_LABEL_RE = re.compile(r"\b(true|false|unknown)\b", re.IGNORECASE)


def _parse_logic_label(text: str) -> str:
    clean = strip_think_blocks(text).strip()
    matches = _LABEL_RE.findall(clean)
    if not matches:
        return ""
    return matches[-1].capitalize()


def _collate_masked(
    batch: list[tuple[list[int], list[int]]], pad_id: int, max_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    mlen = min(max_len, max(len(ids) for ids, _ in batch))
    bsz = len(batch)
    x = torch.full((bsz, mlen - 1), pad_id, dtype=torch.long)
    y = torch.full((bsz, mlen - 1), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(batch):
        ids_c = ids[:mlen]
        lab_c = labels[:mlen]
        n = len(ids_c) - 1
        x[i, :n] = torch.tensor(ids_c[:n], dtype=torch.long)
        y[i, :n] = torch.tensor(lab_c[1 : n + 1], dtype=torch.long)
    return x, y


@torch.no_grad()
def _eval_depth_tasks(
    model: CoreGradLM,
    tokenizer: CoreGradTokenizer,
    tasks: list[VerifiableLogicTask],
    device: torch.device,
    *,
    max_new_tokens: int,
    allow_tools: bool = False,
) -> dict[str, Any]:
    model.eval()
    eos_id = tokenizer.eos_id
    end_id = tokenizer.token_to_id(ASSISTANT_END)
    call_end_id = tokenizer.token_to_id(CALL_END)
    amp = device.type == "cuda"
    executor = ToolExecutor(None) if allow_tools else None

    hits = 0
    tool_ok = 0
    by_cat: dict[str, list[int]] = {}
    sample_out: str | None = None
    for task in tasks:
        # Match encode_masked_trajectory exactly: end prompt at <|user|>...<|/user|> so the model
        # emits <|think|>...<|/think|><|assistant|>... (cot/lot), <|call|>... (dsl), or <|assistant|>... (direct).
        prompt = f"{SYSTEM_START}{DEFAULT_SYSTEM_PROMPT}{SYSTEM_END}{USER_START}{task.question}{USER_END}"
        ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        gen: list[int] = []
        used_tool = False
        for _ in range(max_new_tokens):
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits, _ = model(x[:, -model.cfg.max_seq_len :])
            nxt = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            if nxt in (eos_id, end_id):
                break
            gen.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
            if executor is not None and not used_tool and nxt == call_end_id:
                used_tool = True
                so_far = tokenizer.decode(gen, skip_special_tokens=False)
                calls = extract_tool_calls(so_far)
                if calls:
                    res = executor.execute_raw(calls[-1])
                    if res.ok:
                        tool_ok += 1
                    res_block = f"{RESULT_START}{res.output}{RESULT_END}"
                    res_ids = tokenizer.encode(res_block, add_bos=False, add_eos=False)
                    gen.extend(res_ids)
                    x = torch.cat([x, torch.tensor([res_ids], dtype=torch.long, device=device)], dim=1)
        raw_gen = tokenizer.decode(gen, skip_special_tokens=False)
        if sample_out is None:
            sample_out = raw_gen
        pred = _parse_logic_label(raw_gen)
        ok = int(pred == task.expected_answer)
        hits += ok
        by_cat.setdefault(task.category, []).append(ok)

    out: dict[str, Any] = {
        "accuracy": round(hits / max(len(tasks), 1), 4),
        "hits": hits,
        "total": len(tasks),
        "by_category": {
            cat: round(sum(vals) / max(len(vals), 1), 4) for cat, vals in sorted(by_cat.items())
        },
        "sample_output": (sample_out or "")[:400],
    }
    if allow_tools:
        out["tool_valid_rate"] = round(tool_ok / max(len(tasks), 1), 4)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-step syllogism & Horn-clause depth probe.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=["cot", "direct", "lot", "dsl"], default="cot")
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--eval-per-depth", type=int, default=40)
    parser.add_argument("--eval-depths", type=str, default="2,4,5,6,8,10,12")
    parser.add_argument("--open-vocab", action="store_true", help="Use >3,000 disjoint train pseudowords instead of 32")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    paths = load_local_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = CoreGradTokenizer.from_file(paths.data_dir / "tokenizer" / "tokenizer.json")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CoreGradLM(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    use_cot = args.mode in ("cot", "lot")
    notation = args.mode if args.mode in ("lot", "dsl") else "english"
    train_depths = (2, 3, 4, 5)
    eval_depths = tuple(int(d.strip()) for d in args.eval_depths.split(",") if d.strip())

    print(
        f"[logic_probe:{args.mode}] {model.cfg.name} ({model.num_parameters()/1e6:.2f}M, eff_L={model.cfg.effective_layers}) | "
        f"train_depths={train_depths} ({args.n_train:,} ex) | eval_depths={eval_depths}",
        flush=True,
    )
    train_tasks = generate_logic_tasks(
        args.n_train,
        depths=train_depths,
        split="train",
        seed=2026,
        n_distractors=3,
        cot=use_cot,
        notation=notation,
        open_vocab=args.open_vocab,
    )
    encoded_train = [encode_masked_trajectory(t.turns, tokenizer) for t in train_tasks]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
    amp = device.type == "cuda"
    rng = random.Random(1337)
    max_len = min(1024, model.cfg.max_seq_len)

    for step in range(1, args.steps + 1):
        model.train()
        warm = max(1, int(args.steps * 0.08))
        if step <= warm:
            lr = args.lr * (step / warm)
        else:
            prog = (step - warm) / max(1, args.steps - warm)
            lr = args.lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * prog)))
        for pg in opt.param_groups:
            pg["lr"] = lr

        batch = [encoded_train[rng.randrange(len(encoded_train))] for _ in range(args.batch_size)]
        x, y = _collate_masked(batch, tokenizer.pad_id, max_len=max_len)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0 or step == args.steps:
            print(f"[logic_probe:{args.mode}] step {step:4d}/{args.steps} | loss={float(loss.item()):.4f} | lr={lr:.2e}", flush=True)

    # Evaluate across ID and OOD depths on strictly disjoint fictional vocabulary
    depth_results: dict[str, Any] = {}
    for d in eval_depths:
        tasks = generate_logic_tasks(
            args.eval_per_depth,
            depths=(d,),
            split="eval",
            seed=9000 + d,
            n_distractors=3,
            cot=use_cot,
            notation=notation,
        )
        if args.mode == "dsl":
            max_new = 24 * d + 128
        elif use_cot:
            max_new = (24 * d + 80) if notation == "lot" else (36 * d + 48)
        else:
            max_new = 12
        res = _eval_depth_tasks(model, tokenizer, tasks, device, max_new_tokens=max_new, allow_tools=(args.mode == "dsl"))
        depth_results[str(d)] = res
        tag = "ID " if d in train_depths else "OOD"
        print(
            f"  depth={d:2d} ({tag}) | acc={res['accuracy']*100:5.1f}% ({res['hits']:2d}/{res['total']}) | "
            f"cats={res['by_category']}",
            flush=True,
        )

    id_accs = [depth_results[str(d)]["accuracy"] for d in eval_depths if d in train_depths]
    ood_accs = [depth_results[str(d)]["accuracy"] for d in eval_depths if d not in train_depths]
    report = {
        "checkpoint": str(args.checkpoint),
        "model_name": model.cfg.name,
        "params_m": round(model.num_parameters() / 1e6, 2),
        "effective_layers": model.cfg.effective_layers,
        "mode": args.mode,
        "train_depths": list(train_depths),
        "eval_depths": list(eval_depths),
        "mean_id_accuracy": round(sum(id_accs) / max(len(id_accs), 1), 4),
        "mean_ood_accuracy": round(sum(ood_accs) / max(len(ood_accs), 1), 4),
        "by_depth": depth_results,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }

    out_path = args.out or (paths.runs_dir / "eval" / f"logic_depth_{args.checkpoint.parent.name}_{args.mode}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Summary ({model.cfg.name}, {args.mode}): ID={report['mean_id_accuracy']*100:.1f}% | "
        f"OOD={report['mean_ood_accuracy']*100:.1f}% -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
