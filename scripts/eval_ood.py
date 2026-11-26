#!/usr/bin/env python3
"""Locked Out-of-Distribution (OOD) evaluator for Phase 4A Pareto comparisons.

Evaluates any coregrad SFT/GRPO checkpoint on strictly held-out public benchmarks:
- 50 SQuAD v2 validation answerable QA tasks (6-doc BM25 pool): target-doc citation & span EM
- 50 SQuAD v2 validation unanswerable QA tasks (adversarial traps): abstention rate
- 50 GSM8K test word problems: valid calc syntax & exact numeric match
- 4 no-tool chat routing prompts: clean direct response (0 spurious tool calls)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

from coregrad.config import load_local_paths
from coregrad.model import CoreGradLM, ModelConfig
from coregrad.tokenizer import CoreGradTokenizer
from coregrad.tools import Document, InMemoryBM25Retriever, ToolExecutor
from coregrad.tools.rollout import answer_matches, run_agent_rollout

_CHAT_PROMPTS = (
    "Hello! How are you doing today?",
    "Write a short haiku about winter rain.",
    "Explain what recursion is in one simple sentence.",
    "Name the three primary colors of light.",
)
_GSM_ANS_RE = re.compile(r"####\s*([-+]?[0-9,]+(?:\.[0-9]+)?)")


def _extract_gsm8k_answer(ans_field: str) -> str:
    m = _GSM_ANS_RE.search(ans_field)
    if not m:
        return ""
    return m.group(1).replace(",", "").strip()


def evaluate_ood(
    model: CoreGradLM,
    tokenizer: CoreGradTokenizer,
    hf_cache_dir: Path,
    device: torch.device,
    *,
    n_squad_ans: int = 50,
    n_squad_unans: int = 50,
    n_gsm8k: int = 50,
) -> dict[str, Any]:
    rng = random.Random(4242)

    # 1. No-tool chat routing
    chat_clean = 0
    for q in _CHAT_PROMPTS:
        tr = run_agent_rollout(model, tokenizer, q, ToolExecutor(None), device, max_tool_turns=2, temperature=0.0)
        if tr.tool_calls_total == 0 and len(tr.final_answer.strip()) >= 8:
            chat_clean += 1

    # 2. Held-out SQuAD v2 validation
    sq_path = hf_cache_dir / "rajpurkar/squad_v2/squad_v2/validation-00000-of-00001.parquet"
    rows = pq.read_table(sq_path, columns=["title", "context", "question", "answers"]).to_pylist()
    ans_rows = [r for r in rows if r["answers"]["text"] and len(r["context"]) <= 1400]
    unans_rows = [r for r in rows if not r["answers"]["text"] and len(r["context"]) <= 1400]
    rng.shuffle(ans_rows)
    rng.shuffle(unans_rows)

    all_ctx_pool = [(str(r["title"]).replace("_", " "), str(r["context"]).strip()) for r in ans_rows[n_squad_ans : n_squad_ans + 300]]

    sq_cite_hits = 0
    sq_span_hits = 0
    sq_halluc_cites = 0
    sq_syntax_ok = 0
    sq_syntax_tot = 0

    for idx, r in enumerate(ans_rows[:n_squad_ans]):
        target_id = f"d{1000 + idx:04d}"
        title = str(r["title"]).replace("_", " ").strip()
        ctx = str(r["context"]).strip()
        docs = [Document(doc_id=target_id, title=title, text=ctx)]
        for j in range(5):
            dt, dc = all_ctx_pool[(idx * 5 + j) % len(all_ctx_pool)]
            if dc != ctx:
                docs.append(Document(doc_id=f"d{2000 + idx * 5 + j:04d}", title=dt, text=dc))
        rng.shuffle(docs)
        tr = run_agent_rollout(
            model,
            tokenizer,
            str(r["question"]).strip(),
            ToolExecutor(InMemoryBM25Retriever(docs)),
            device,
            max_tool_turns=3,
            max_new_tokens_per_turn=140,
            temperature=0.0,
        )
        sq_syntax_ok += tr.tool_calls_valid
        sq_syntax_tot += tr.tool_calls_total
        golds = [str(a).strip().lower() for a in r["answers"]["text"] if str(a).strip()]
        low_ans = tr.final_answer.lower()
        if target_id in tr.cited_doc_ids:
            sq_cite_hits += 1
        if any(g in low_ans for g in golds):
            sq_span_hits += 1
        for cid in tr.cited_doc_ids:
            if cid not in tr.retrieved_doc_ids:
                sq_halluc_cites += 1

    sq_abstain_hits = 0
    for idx, r in enumerate(unans_rows[:n_squad_unans]):
        title = str(r["title"]).replace("_", " ").strip()
        ctx = str(r["context"]).strip()
        docs = [Document(doc_id=f"d{3000 + idx:04d}", title=title, text=ctx)]
        for j in range(4):
            dt, dc = all_ctx_pool[(idx * 4 + j + 100) % len(all_ctx_pool)]
            docs.append(Document(doc_id=f"d{4000 + idx * 4 + j:04d}", title=dt, text=dc))
        tr = run_agent_rollout(
            model,
            tokenizer,
            str(r["question"]).strip(),
            ToolExecutor(InMemoryBM25Retriever(docs)),
            device,
            max_tool_turns=3,
            max_new_tokens_per_turn=140,
            temperature=0.0,
        )
        sq_syntax_ok += tr.tool_calls_valid
        sq_syntax_tot += tr.tool_calls_total
        if answer_matches(tr.final_answer, "", should_abstain=True):
            sq_abstain_hits += 1

    # 3. Held-out GSM8K test
    gsm_path = hf_cache_dir / "openai/gsm8k/main/test-00000-of-00001.parquet"
    gsm_rows = pq.read_table(gsm_path, columns=["question", "answer"]).to_pylist()
    rng.shuffle(gsm_rows)
    gsm_hits = 0
    gsm_calls_ok = 0
    gsm_calls_tot = 0
    for r in gsm_rows[:n_gsm8k]:
        gold = _extract_gsm8k_answer(str(r["answer"]))
        tr = run_agent_rollout(
            model,
            tokenizer,
            str(r["question"]).strip(),
            ToolExecutor(None),
            device,
            max_tool_turns=4,
            max_new_tokens_per_turn=180,
            temperature=0.0,
        )
        gsm_calls_ok += tr.tool_calls_valid
        gsm_calls_tot += tr.tool_calls_total
        if gold and answer_matches(tr.final_answer, gold, should_abstain=False):
            gsm_hits += 1

    return {
        "chat_clean_routing": round(chat_clean / len(_CHAT_PROMPTS), 4),
        "chat_clean_hits": chat_clean,
        "chat_total": len(_CHAT_PROMPTS),
        "squad_cite_rate": round(sq_cite_hits / max(n_squad_ans, 1), 4),
        "squad_span_em": round(sq_span_hits / max(n_squad_ans, 1), 4),
        "squad_halluc_cites": sq_halluc_cites,
        "squad_unans_abstain": round(sq_abstain_hits / max(n_squad_unans, 1), 4),
        "squad_tool_syntax": round(sq_syntax_ok / max(sq_syntax_tot, 1), 4),
        "gsm8k_exact_match": round(gsm_hits / max(n_gsm8k, 1), 4),
        "gsm8k_hits": gsm_hits,
        "gsm8k_total": n_gsm8k,
        "gsm8k_tool_syntax": round(gsm_calls_ok / max(gsm_calls_tot, 1), 4),
        "gsm8k_calls_valid": gsm_calls_ok,
        "gsm8k_calls_total": gsm_calls_tot,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run locked OOD evaluation (SQuAD v2 + GSM8K + chat routing).")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    paths = load_local_paths()
    assert paths.hf_cache_dir is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CoreGradTokenizer.from_file(paths.data_dir / "tokenizer" / "tokenizer.json")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CoreGradLM(ModelConfig(**ckpt["model_config"])).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    report = evaluate_ood(model, tokenizer, paths.hf_cache_dir, device)
    report["checkpoint"] = str(args.checkpoint)
    report["model_name"] = model.cfg.name
    report["params_m"] = round(model.num_parameters() / 1e6, 2)
    report["elapsed_s"] = round(time.perf_counter() - t0, 1)

    print(f"OOD Evaluation ({model.cfg.name}, {report['params_m']}M, {report['elapsed_s']}s):")
    print(f"  Chat no-tool routing : {report['chat_clean_routing']*100:5.1f}% ({report['chat_clean_hits']}/{report['chat_total']})")
    print(f"  SQuAD v2 target cite : {report['squad_cite_rate']*100:5.1f}% (halluc={report['squad_halluc_cites']})")
    print(f"  SQuAD v2 span EM     : {report['squad_span_em']*100:5.1f}%")
    print(f"  SQuAD v2 unans abst. : {report['squad_unans_abstain']*100:5.1f}%")
    print(
        f"  GSM8K EM / syntax    : {report['gsm8k_exact_match']*100:5.1f}% ({report['gsm8k_hits']}/{report['gsm8k_total']}) | "
        f"syntax={report['gsm8k_tool_syntax']*100:5.1f}% ({report['gsm8k_calls_valid']}/{report['gsm8k_calls_total']})"
    )

    out_path = args.out or (paths.runs_dir / "eval" / f"ood_{args.checkpoint.parent.name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved OOD report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
