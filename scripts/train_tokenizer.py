#!/usr/bin/env python3
"""Train public BPE tokenizer(s) and compare 16k vs 32k fertility across domains."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from coregrad.config import load_local_paths
from coregrad.data.sources import iter_parquet_texts, load_pretrain_specs
from coregrad.tokenizer import measure_fertility, train_bpe_tokenizer

TOOL_PROTOCOL_SAMPLES = [
    (
        "<|user|>What year did Darwin publish On the Origin of Species, and how many years before 1953?<|/user|>"
        "<|assistant|><|think|>Look up the publication year and compute the difference.<|/think|>"
        '<|call|>search(query="Darwin On the Origin of Species publication year", k=3)<|/call|>'
        '<|result|>[{"doc_id": "d104", "span": [120, 410], "text": "Published in 1859."}]<|/result|>'
        '<|call|>calc(expr="1953 - 1859")<|/call|>'
        "<|result|>94<|/result|>"
        "<|think|>1859 to 1953 is 94 years.<|/think|>"
        "Charles Darwin published On the Origin of Species in 1859 <|cite|>d104<|/cite|>, "
        "which was 94 years before 1953.<|eos|>"
    ),
    (
        "<|user|>Compare the boiling points of ethanol and water in Kelvin.<|/user|>"
        "<|assistant|><|think|>Search for both boiling points and convert Celsius to Kelvin.<|/think|>"
        '<|call|>search(query="boiling point ethanol water Celsius", k=3)<|/call|>'
        '<|result|>[{"doc_id": "d201", "text": "Ethanol boils at 78.37 C; water boils at 100.0 C."}]<|/result|>'
        '<|call|>calc(expr="78.37 + 273.15")<|/call|>'
        "<|result|>351.52<|/result|>"
        '<|call|>calc(expr="100.0 + 273.15")<|/call|>'
        "<|result|>373.15<|/result|>"
        "Ethanol boils at 351.52 K and water boils at 373.15 K <|cite|>d201<|/cite|>.<|eos|>"
    ),
]


def sample_texts_for_tokenizer(
    total_mb: float,
    eval_docs_per_domain: int = 400,
) -> tuple[list[str], dict[str, list[str]]]:
    specs = load_pretrain_specs()
    # Skip code-github-clean duplication for tokenizer sampling; use the 4 core domains
    domain_budgets = {
        "edu-en": 0.45,
        "cosmo": 0.25,
        "code-starcoder": 0.17,
        "finemath": 0.13,
    }
    target_bytes_total = int(total_mb * 1024 * 1024)
    train_texts: list[str] = []
    eval_samples: dict[str, list[str]] = {"tool-dsl": TOOL_PROTOCOL_SAMPLES * 20}

    for spec in specs:
        if spec.name not in domain_budgets or not spec.files:
            continue
        budget = int(target_bytes_total * domain_budgets[spec.name])
        collected = 0
        eval_list: list[str] = []
        for text in iter_parquet_texts(spec.files, spec.text_column, interleave_files=True):
            if len(eval_list) < eval_docs_per_domain:
                eval_list.append(text)
                continue
            train_texts.append(text)
            collected += len(text.encode("utf-8"))
            if collected >= budget:
                break
        eval_samples[spec.name] = eval_list
        print(
            f"[sample] {spec.name:16s} train={collected/1e6:6.1f} MB | eval_docs={len(eval_list)}",
            flush=True,
        )
    return train_texts, eval_samples


def _iter_chunks(texts: list[str], chunk_size: int = 1000) -> Iterator[list[str]]:
    for i in range(0, len(texts), chunk_size):
        yield texts[i : i + chunk_size]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train public BPE tokenizer and benchmark fertility.")
    parser.add_argument("--sample-mb", type=float, default=200.0, help="Total public text sample in MB")
    parser.add_argument("--vocab-size", type=int, default=16384, help="Final locked vocab size")
    parser.add_argument("--compare-32k", action="store_true", help="Also train 32768 to compare fertility")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    paths = load_local_paths()
    out_dir = args.out_dir or (paths.data_dir / "tokenizer")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_texts, eval_samples = sample_texts_for_tokenizer(total_mb=args.sample_mb)
    print(f"Total training documents sampled: {len(train_texts):,}", flush=True)

    sizes = [args.vocab_size]
    if args.compare_32k and 32768 not in sizes:
        sizes.append(32768)

    reports: dict[str, dict[str, dict[str, float]]] = {}
    for vs in sizes:
        out_path = out_dir / (
            "tokenizer.json" if vs == args.vocab_size else f"tokenizer_{vs}.json"
        )
        print(f"\nTraining byte-level BPE tokenizer (vocab_size={vs}) -> {out_path}", flush=True)
        tok = train_bpe_tokenizer(iter(train_texts), vocab_size=vs, output_path=out_path)
        fert = measure_fertility(tok, eval_samples)
        reports[str(vs)] = fert
        print(f"Fertility report (vocab={tok.vocab_size}):")
        for dom, stats in fert.items():
            print(
                f"  {dom:16s} | {stats['bytes_per_token']:.3f} bytes/tok | "
                f"{stats['tokens_per_char']:.3f} tok/char"
            )

    (out_dir / "fertility_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
