#!/usr/bin/env python3
"""Build a reproducible public Markdown demo library from held-out SQuAD v2 validation articles.

Creates clean multi-paragraph Markdown files under data/demo_library/ (gitignored) so anyone
can verify hybrid EmbeddingGemma 2 + SQLite FTS5 BM25 indexing and live Gate 2 research CLI
strictly on open public data.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

from coregrad.config import load_local_paths


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    return slug[:64] or "article"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export held-out public SQuAD v2 articles as Markdown.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-articles", type=int, default=35)
    args = parser.parse_args()

    paths = load_local_paths()
    if paths.hf_cache_dir is None:
        raise SystemExit("hf_cache_dir not configured.")

    sq_val = paths.hf_cache_dir / "rajpurkar/squad_v2/squad_v2/validation-00000-of-00001.parquet"
    if not sq_val.exists():
        raise SystemExit(f"Missing {sq_val}. Run: uv run python scripts/sync_datasets.py --squad-gsm8k")

    out_dir = args.out_dir or (paths.data_dir / "demo_library")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = pq.read_table(sq_val, columns=["title", "context", "question", "answers"]).to_pylist()
    articles: dict[str, list[str]] = {}
    sample_qa: list[dict[str, str]] = []

    for r in rows:
        title = str(r["title"]).replace("_", " ").strip()
        ctx = str(r["context"]).strip()
        if title not in articles and len(articles) >= args.max_articles:
            continue
        paras = articles.setdefault(title, [])
        if ctx not in paras and len(paras) < 6:
            paras.append(ctx)
        ans = r["answers"]["text"]
        if ans and len(sample_qa) < 12 and all(q["title"] != title for q in sample_qa):
            sample_qa.append({"title": title, "question": r["question"], "expected": ans[0]})

    for title, paras in articles.items():
        fname = f"{_slugify(title)}.md"
        body = f"# {title}\n\n" + "\n\n".join(paras) + "\n"
        (out_dir / fname).write_text(body, encoding="utf-8")

    (out_dir / "_demo_questions.json").write_text(json.dumps(sample_qa, indent=2), encoding="utf-8")
    print(f"Exported {len(articles)} public Markdown articles + {len(sample_qa)} sample QAs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
