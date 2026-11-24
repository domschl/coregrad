"""Automated Gate 2 Cognitive Core benchmark evaluator.

Measures all 4 Gate 2 pass criteria on locked synthetic & public retrieval tasks:
1. Tool syntax validity (%)
2. Routing & abstention accuracy (%)
3. 1-to-3-step calc exact accuracy (%)
4. 1-hop & 2-hop grounded retrieval accuracy & citation precision (%)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from coregrad.model import CoreGradLM
from coregrad.tokenizer import CoreGradTokenizer
from coregrad.tools import Document, InMemoryBM25Retriever, ToolExecutor
from coregrad.tools.rollout import answer_matches, run_agent_rollout


def evaluate_gate2(
    model: CoreGradLM,
    tokenizer: CoreGradTokenizer,
    eval_tasks_path: Path,
    device: torch.device,
    *,
    enable_tools: bool = True,
    max_calc_tasks: int | None = None,
    max_retrieval_tasks: int | None = None,
    max_new_tokens_per_turn: int = 160,
) -> dict[str, Any]:
    data = json.loads(eval_tasks_path.read_text(encoding="utf-8"))
    calc_tasks = data.get("calc_tasks", [])[:max_calc_tasks]
    ret_tasks = data.get("retrieval_tasks", [])[:max_retrieval_tasks]

    total_calls = 0
    valid_calls = 0
    calc_hits = 0
    ret_grounded_hits = 0
    ret_grounded_total = 0
    ret_cited_hits = 0
    abstain_hits = 0
    abstain_total = 0
    hallucinated_cites = 0
    sample_traces: list[dict[str, Any]] = []

    for task in calc_tasks:
        executor = ToolExecutor(None) if enable_tools else None
        tr = run_agent_rollout(
            model,
            tokenizer,
            task["question"],
            executor,
            device,
            max_tool_turns=4 if enable_tools else 0,
            max_new_tokens_per_turn=max_new_tokens_per_turn,
            temperature=0.0,
        )
        total_calls += tr.tool_calls_total
        valid_calls += tr.tool_calls_valid
        ok = answer_matches(tr.final_answer, task["expected_answer"], should_abstain=False)
        calc_hits += int(ok)
        if len(sample_traces) < 3:
            sample_traces.append(
                {
                    "task_id": task["task_id"],
                    "question": task["question"],
                    "expected": task["expected_answer"],
                    "final_answer": tr.final_answer,
                    "calls": tr.tool_calls_raw,
                    "correct": ok,
                }
            )

    for task in ret_tasks:
        docs = [Document(**d) for d in task["documents"]]
        executor = ToolExecutor(InMemoryBM25Retriever(docs)) if enable_tools else None
        tr = run_agent_rollout(
            model,
            tokenizer,
            task["question"],
            executor,
            device,
            max_tool_turns=4 if enable_tools else 0,
            max_new_tokens_per_turn=max_new_tokens_per_turn,
            temperature=0.0,
        )
        total_calls += tr.tool_calls_total
        valid_calls += tr.tool_calls_valid
        should_abstain = bool(task["should_abstain"])
        ok = answer_matches(tr.final_answer, task["expected_answer"], should_abstain=should_abstain)

        if should_abstain:
            abstain_total += 1
            abstain_hits += int(ok)
        else:
            ret_grounded_total += 1
            ret_grounded_hits += int(ok)
            expected_docs = set(task["expected_doc_ids"])
            if tr.cited_doc_ids:
                if any(c not in tr.retrieved_doc_ids for c in tr.cited_doc_ids):
                    hallucinated_cites += 1
                elif any(c in expected_docs for c in tr.cited_doc_ids):
                    ret_cited_hits += 1

        if len(sample_traces) < 6:
            sample_traces.append(
                {
                    "task_id": task["task_id"],
                    "question": task["question"],
                    "expected": task["expected_answer"],
                    "final_answer": tr.final_answer,
                    "calls": tr.tool_calls_raw,
                    "citations": tr.cited_doc_ids,
                    "correct": ok,
                }
            )

    return {
        "enable_tools": enable_tools,
        "tool_syntax_validity": valid_calls / max(total_calls, 1),
        "tool_calls_valid": valid_calls,
        "tool_calls_total": total_calls,
        "calc_accuracy": calc_hits / max(len(calc_tasks), 1),
        "calc_hits": calc_hits,
        "calc_total": len(calc_tasks),
        "grounded_retrieval_accuracy": ret_grounded_hits / max(ret_grounded_total, 1),
        "grounded_retrieval_hits": ret_grounded_hits,
        "grounded_retrieval_total": ret_grounded_total,
        "grounded_citation_rate": ret_cited_hits / max(ret_grounded_total, 1),
        "hallucinated_citations": hallucinated_cites,
        "abstention_accuracy": abstain_hits / max(abstain_total, 1),
        "abstention_hits": abstain_hits,
        "abstention_total": abstain_total,
        "sample_traces": sample_traces,
    }
