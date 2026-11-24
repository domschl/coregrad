"""Phase 1 unit tests: tokenizer, special tokens, shard packing, mixture loader, eval harness."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from coregrad.data import (
    MixtureDataLoader,
    clean_text,
    doc_hash_u64,
    is_val_hash,
    pack_domain_shards,
)
from coregrad.eval import (
    evaluate_completion_probes,
    evaluate_grammar_pairs,
    evaluate_validation_bpb,
)
from coregrad.model import CoreGradLM, ModelConfig
from coregrad.tokenizer import ALL_SPECIAL_TOKENS, CoreGradTokenizer, train_bpe_tokenizer


def _make_sample_corpus() -> list[str]:
    base = [
        "The scientific method relies on testable hypotheses, careful measurements, and reproducible experiments.",
        "In 1962 and 1979, researchers published monographs examining thermodynamics and statistical mechanics.",
        "def fibonacci(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
        "<|user|>Compute 1979 - 1962.<|/user|><|assistant|><|call|>calc(expr=\"1979 - 1962\")<|/call|><|result|>17<|/result|>17<|eos|>",
        "Every rational number can be expressed as a quotient p/q of two integers where q is nonzero.",
    ]
    return [f"{s} Document index {i}." for i in range(60) for s in base]


def test_tokenizer_special_tokens_and_single_digit_splitting(tmp_path: Path) -> None:
    tok_path = tmp_path / "tok.json"
    train_bpe_tokenizer(iter(_make_sample_corpus()), vocab_size=512, output_path=tok_path)
    reloaded = CoreGradTokenizer.from_file(tok_path)

    assert reloaded.vocab_size <= 512
    for idx, sym in enumerate(ALL_SPECIAL_TOKENS):
        assert reloaded.token_to_id(sym) == idx

    # Special tokens must remain single atomic IDs
    tool_str = '<|think|>Check<|/think|><|call|>calc(expr="2+2")<|/call|><|result|>4<|/result|>'
    ids = reloaded.encode(tool_str)
    assert reloaded.token_to_id("<|think|>") in ids
    assert reloaded.token_to_id("<|call|>") in ids
    assert reloaded.token_to_id("<|result|>") in ids

    # Single-digit splitting: "1979" must encode to 4 individual digit tokens
    digit_ids = reloaded.encode("1979")
    assert len(digit_ids) == 4
    assert reloaded.decode(digit_ids) == "1979"


def test_clean_text_and_deterministic_split() -> None:
    assert clean_text("short") is None
    raw = "Hello\x00 world!\r\n\r\n\r\n\r\n\r\n" + ("A" * 80)
    cleaned = clean_text(raw)
    assert cleaned is not None
    assert "\x00" not in cleaned
    assert "\r" not in cleaned
    assert "\n\n\n\n" not in cleaned

    h1 = doc_hash_u64(cleaned)
    h2 = doc_hash_u64(cleaned)
    assert h1 == h2
    assert isinstance(is_val_hash(h1, val_fraction=0.5), bool)


def test_pack_loader_and_eval_harness_end_to_end(tmp_path: Path) -> None:
    docs = _make_sample_corpus()
    # Add exact duplicates to verify deduplication
    docs_with_dups = docs + docs[:20]

    tok = train_bpe_tokenizer(iter(docs), vocab_size=512, output_path=tmp_path / "tok.json")
    shards_root = tmp_path / "shards"
    d_manifest = pack_domain_shards(
        domain="edu-en",
        texts=iter(docs_with_dups),
        tokenizer=tok,
        out_dir=shards_root,
        max_train_tokens=50_000,
        max_val_tokens=10_000,
        val_fraction=0.25,
        shard_tokens=20_000,
    )
    assert d_manifest.duplicates_dropped == 20
    assert d_manifest.train.tokens > 500
    assert d_manifest.val.tokens > 100

    manifest = {
        "vocab_size": tok.vocab_size,
        "domains": {
            "edu-en": {
                "train": {
                    "shards": d_manifest.train.shards,
                    "tokens": d_manifest.train.tokens,
                    "utf8_bytes": d_manifest.train.utf8_bytes,
                    "docs": d_manifest.train.docs,
                },
                "val": {
                    "shards": d_manifest.val.shards,
                    "tokens": d_manifest.val.tokens,
                    "utf8_bytes": d_manifest.val.utf8_bytes,
                    "docs": d_manifest.val.docs,
                },
            }
        },
    }
    (shards_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loader = MixtureDataLoader(shards_root, {"edu-en": 1.0}, seq_len=32, batch_size=4, seed=42)
    x, y = loader.next_batch()
    assert x.shape == (4, 32)
    assert y.shape == (4, 32)
    assert torch.equal(x[:, 1:], y[:, :-1])

    cfg = ModelConfig(
        name="tiny-eval",
        vocab_size=tok.vocab_size,
        max_seq_len=128,
        n_layer=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
    )
    model = CoreGradLM(cfg)
    device = torch.device("cpu")

    bpb_res = evaluate_validation_bpb(
        model, shards_root, device=device, seq_len=32, batch_size=2, max_batches_per_domain=4, use_bf16=False
    )
    assert "edu-en" in bpb_res
    assert bpb_res["edu-en"].bpb > 0.0

    gram = evaluate_grammar_pairs(model, tok, device=device, use_bf16=False)
    assert 0.0 <= gram["accuracy"] <= 1.0

    comp = evaluate_completion_probes(model, tok, device=device, use_bf16=False)
    assert 0.0 <= float(comp["accuracy"]) <= 1.0  # type: ignore[arg-type]
