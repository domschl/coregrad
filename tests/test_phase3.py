"""Phase 3 (Gate 2) CPU-only unit tests: tools, hybrid index, synthetic gens, SFT, GRPO, Gate 2 eval."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from coregrad.eval.gate2 import evaluate_gate2
from coregrad.model import CoreGradLM, ModelConfig
from coregrad.synthetic import convert_gsm8k_row, generate_calc_tasks, generate_retrieval_tasks
from coregrad.synthetic.retrieval_gen import EVAL_SURNAMES, TRAIN_SURNAMES
from coregrad.tokenizer import train_bpe_tokenizer
from coregrad.tools import (
    HybridMarkdownIndex,
    InMemoryBM25Retriever,
    ToolExecutor,
    Turn,
    encode_masked_trajectory,
    extract_peak_excerpt,
    format_tool_call,
    parse_tool_call,
    safe_calc,
    validate_tool_call,
)
from coregrad.tools.rollout import RolloutTrace, answer_matches, score_rollout_reward
from coregrad.train.grpo import GRPOConfig, train_grpo
from coregrad.train.sft import SFTConfig, train_sft


class _MockEmbedder:
    dim = 8

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for ch in t.lower().encode("utf-8"):
                vecs[i, ch % self.dim] += 1.0
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)


def test_safe_calc_and_security_guards() -> None:
    assert safe_calc("1979 - 1962") == "17"
    assert safe_calc("round(sqrt(3**2 + 4**2), 2)") == "5"
    assert safe_calc("round(100 * (1 + 15/100), 2)") == "115"
    with pytest.raises(ValueError):
        safe_calc("__import__('os').system('echo nope')")
    with pytest.raises(ValueError):
        safe_calc("2 ** 100000")


def test_tool_protocol_parsing_and_masked_sft_labels(tmp_path: Path) -> None:
    # Ensure tools having a keyword argument literally named 'name' format cleanly
    assert format_tool_call("lookup", name="Ada", limit=5) == 'lookup(name="Ada", limit=5)'
    call = parse_tool_call('search(query="Darwin 1859", k=3)')
    ok, msg = validate_tool_call(call)
    assert ok, msg
    assert call.name == "search" and call.args["k"] == 3

    bad = parse_tool_call("unknown_tool(x=1)")
    ok_bad, _ = validate_tool_call(bad)
    assert not ok_bad

    corpus = [
        '<|system|>Sys<|/system|><|user|>Hi<|/user|><|think|>Plan<|/think|><|call|>calc(expr="2+2")<|/call|><|result|>4<|/result|><|assistant|>4<|/assistant|><|eos|>'
        for _ in range(30)
    ]
    tok = train_bpe_tokenizer(iter(corpus), vocab_size=256, output_path=tmp_path / "tok.json")
    turns = [
        Turn("system", "Sys"),
        Turn("user", "What is 2+2?"),
        Turn("think", "Compute 2+2."),
        Turn("call", 'calc(expr="2+2")'),
        Turn("result", "4"),
        Turn("assistant", "The answer is 4."),
    ]
    ids, labels = encode_masked_trajectory(turns, tok)
    assert len(ids) == len(labels)
    assert labels[0] == -100  # BOS masked
    assert labels[-1] == tok.eos_id  # EOS trainable
    # Must contain both masked (-100) and unmasked tokens
    assert any(x == -100 for x in labels[1:-1])
    assert any(x != -100 for x in labels[1:-1])


def test_hybrid_markdown_index_bm25_and_rrf(tmp_path: Path) -> None:
    md_dir = tmp_path / "books"
    md_dir.mkdir()
    (md_dir / "optics.md").write_text(
        "# Optics Notes\n\nIn a vacuum, the speed of light is 299792458 m/s, and glass prisms disperse white light into its constituent spectrum of colors.\n",
        encoding="utf-8",
    )
    (md_dir / "botany.md").write_text(
        "# Alpine Botany\n\nHigh-altitude mosses survive the freezing winter months through desiccation and cellular protection mechanisms that have evolved over millennia.\n",
        encoding="utf-8",
    )
    (md_dir / "deutsch.md").write_text(
        "# Alpenbotanik\n\nDie hochalpinen Moose überleben kalte Winter durch Austrocknung und besondere Schutzmechanismen in den hohen Bergen Europas und der Schweiz.\n",
        encoding="utf-8",
    )

    idx = HybridMarkdownIndex(tmp_path / "index", embedder=_MockEmbedder())
    try:
        # Rebuild twice on the same directory to verify FTS5 & chunk_id stay cleanly aligned
        idx.index_markdown_directory(md_dir, english_only=True)
        stats = idx.index_markdown_directory(md_dir, english_only=True)
        assert stats["docs"] == 2
        assert stats["skipped_non_english"] == 1
        assert stats["has_vectors"] == 1
        hits = idx.search("speed of light vacuum prism", k=2)
        assert len(hits) >= 1
        assert hits[0].title == "Optics Notes"
        # Porter stemming check: "travels" matches "traveling" / "survives" matches "surviving"
        stem_hits = idx.search("freezing winter moss surviving", k=2)
        assert len(stem_hits) >= 1
        assert stem_hits[0].title == "Alpine Botany"
        page = idx.read(hits[0].doc_id, start=0, length=200)
        assert "299792458" in str(page["text"])
        listed = idx.list_docs("Botany")
        assert len(listed) == 1
    finally:
        idx.close()

    long_txt = ("alpha beta " * 100) + "target quantum prism 1859 " + ("gamma delta " * 100)
    excerpt = extract_peak_excerpt(long_txt, "prism 1859", max_chars=200)
    assert "1859" in excerpt


def test_synthetic_generators_disjoint_and_executable() -> None:
    assert set(TRAIN_SURNAMES).isdisjoint(set(EVAL_SURNAMES))

    calc_tasks = generate_calc_tasks(24, seed=7)
    exec_calc = ToolExecutor(None)
    for task in calc_tasks:
        calls = [t.content for t in task.turns if t.role == "call"]
        assert len(calls) >= 1
        for c in calls:
            res = exec_calc.execute_raw(c)
            assert res.ok, res.output

    ret_tasks = generate_retrieval_tasks(16, seed=9, split="eval")
    for task in ret_tasks:
        ex = ToolExecutor(InMemoryBM25Retriever(task.documents))
        calls = [t.content for t in task.turns if t.role == "call"]
        assert len(calls) >= 1
        for c in calls:
            res = ex.execute_raw(c)
            assert res.ok, res.output

    gsm_turns = convert_gsm8k_row(
        "Natalia sold 48 clips in April and half as many in May. Total?",
        "May: 48/2 = <<48/2=24>>24.\nTotal: 48+24 = <<48+24=72>>72.\n#### 72",
    )
    assert gsm_turns is not None
    gsm_calls = [t.content for t in gsm_turns if t.role == "call"]
    assert gsm_calls == ['calc(expr="48/2")', 'calc(expr="48+24")']


def test_reward_scoring_and_cpu_sft_grpo_gate2_pipeline(tmp_path: Path) -> None:
    # Verify verifiable reward logic directly
    good_trace = RolloutTrace(
        question="Year?",
        full_ids=[1, 2, 3],
        assistant_mask=[False, True, True],
        rendered_text="",
        final_answer="Published in 1859 <|cite|>d001<|/cite|>.",
        tool_calls_raw=['search(query="Year", k=3)'],
        tool_calls_valid=1,
        tool_calls_total=1,
        retrieved_doc_ids={"d001"},
        cited_doc_ids=["d001"],
    )
    assert score_rollout_reward(good_trace, expected_answer="1859", expected_doc_ids=("d001",)) == pytest.approx(0.9)
    # Hallucinated citation gets penalized
    bad_cite = RolloutTrace(**{**asdict(good_trace), "cited_doc_ids": ["d999"]})
    assert score_rollout_reward(bad_cite, expected_answer="1859", expected_doc_ids=("d001",)) < 0.3
    assert answer_matches("I could not find this in the documents.", "could not find", should_abstain=True)

    # Mini end-to-end CPU SFT -> GRPO -> Gate 2 eval smoke check
    calc_tasks = generate_calc_tasks(12, seed=1)
    ret_tasks = generate_retrieval_tasks(8, seed=2, split="train")
    docs_for_tok = [
        " ".join(t.content for t in task.turns) for task in (*calc_tasks, *ret_tasks)
    ] * 10
    tok_path = tmp_path / "tok.json"
    tok = train_bpe_tokenizer(iter(docs_for_tok), vocab_size=384, output_path=tok_path)

    cfg = ModelConfig(
        name="tiny-phase3",
        vocab_size=tok.vocab_size,
        max_seq_len=512,
        n_layer=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
    )
    model = CoreGradLM(cfg)
    base_ckpt = tmp_path / "base.pt"
    torch.save({"model_config": asdict(cfg), "model_state_dict": model.state_dict()}, base_ckpt)

    sft_dir = tmp_path / "sft"
    sft_dir.mkdir()
    records = [
        {"source": t.category, "turns": [{"role": x.role, "content": x.content} for x in t.turns]}
        for t in (*calc_tasks, *ret_tasks)
    ]
    for fname, subset in (("train.jsonl", records[:14]), ("val.jsonl", records[14:])):
        (sft_dir / fname).write_text("\n".join(json.dumps(r) for r in subset) + "\n", encoding="utf-8")

    sft_run = train_sft(
        SFTConfig(
            run_name="tiny_sft",
            init_from=str(base_ckpt),
            train_jsonl=str(sft_dir / "train.jsonl"),
            val_jsonl=str(sft_dir / "val.jsonl"),
            eval_tasks_json=str(tmp_path / "missing_eval.json"),
            seq_len=256,
            pack_sequences=True,
            micro_batch=2,
            grad_accum=1,
            max_steps=3,
            log_interval=1,
            eval_interval=3,
            compile_model=False,
            activation_checkpointing=False,
        ),
        tokenizer_path=tok_path,
        runs_dir=tmp_path / "runs",
        device=torch.device("cpu"),
    )
    assert (sft_run / "best.pt").exists()

    grpo_run = train_grpo(
        GRPOConfig(
            run_name="tiny_grpo",
            init_from=str(sft_run / "best.pt"),
            steps=2,
            group_size=2,
            max_tool_turns=1,
            max_new_tokens_per_turn=12,
            eval_interval=0,
            use_squad_v2=False,
        ),
        tokenizer_path=tok_path,
        runs_dir=tmp_path / "runs",
        device=torch.device("cpu"),
    )
    assert (grpo_run / "best.pt").exists()

    eval_path = tmp_path / "eval_tasks.json"
    eval_path.write_text(
        json.dumps(
            {
                "calc_tasks": [
                    {
                        "task_id": t.task_id,
                        "category": t.category,
                        "question": t.question,
                        "expected_answer": t.expected_answer,
                        "requires_tool": True,
                    }
                    for t in calc_tasks[:2]
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
                    for t in ret_tasks[:2]
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = torch.load(grpo_run / "best.pt", map_location="cpu", weights_only=False)
    eval_model = CoreGradLM(ModelConfig(**loaded["model_config"]))
    eval_model.load_state_dict(loaded["model_state_dict"])
    rep = evaluate_gate2(eval_model, tok, eval_path, device=torch.device("cpu"), max_new_tokens_per_turn=12)
    assert rep["calc_total"] == 2
    assert rep["grounded_retrieval_total"] + rep["abstention_total"] == 2
