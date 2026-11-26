#!/usr/bin/env python3
"""Local Markdown research CLI: index any Markdown folder and chat with coregrad."""

from __future__ import annotations

import argparse
from pathlib import Path

from coregrad.app.cli import build_markdown_index, run_interactive_chat
from coregrad.config import load_local_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="coregrad local Markdown research CLI.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="Index one or more folders of Markdown files (BM25 + optional EmbeddingGemma 2)")
    p_idx.add_argument("markdown_dirs", type=Path, nargs="+", help="One or more Markdown folders (e.g. books + summaries)")
    p_idx.add_argument("--out", type=Path, default=None)
    p_idx.add_argument("--use-embeddinggemma2", action="store_true")
    p_idx.add_argument("--mrl-dim", type=int, default=256)
    p_idx.add_argument("--chunk-chars", type=int, default=1000)
    p_idx.add_argument(
        "--english-only",
        action="store_true",
        help="Skip non-English Markdown files (recommended for English-only coregrad models)",
    )

    p_chat = sub.add_parser("chat", help="Start interactive research session")
    p_chat.add_argument("--checkpoint", type=Path, required=True)
    p_chat.add_argument("--index", type=Path, default=None)
    p_chat.add_argument("--tokenizer", type=Path, default=None)
    p_chat.add_argument("--use-embeddinggemma2", action="store_true")
    p_chat.add_argument("--mrl-dim", type=int, default=256)
    p_chat.add_argument("--temperature", type=float, default=0.0)
    p_chat.add_argument("--ask", type=str, default=None, help="Run a single non-interactive query and exit")
    p_chat.add_argument(
        "--kv-cache",
        action="store_true",
        help="Opt-in Phase 4B GQA KV-bank injection mode (for additional latent-memory tests)",
    )
    p_chat.add_argument(
        "--query-head",
        type=Path,
        default=None,
        help="Optional trained LatentQueryHead checkpoint for zero-token latent retrieval with --kv-cache",
    )
    p_chat.add_argument("--kv-top-k", type=int, default=4, help="Passages injected into GQA KV bank")

    args = parser.parse_args()
    paths = load_local_paths()

    if args.cmd == "index":
        out_dir = args.out or (paths.data_dir / "index" / args.markdown_dirs[0].name)
        stats = build_markdown_index(
            args.markdown_dirs,
            out_dir,
            use_embeddinggemma2=args.use_embeddinggemma2,
            mrl_dim=args.mrl_dim,
            chunk_chars=args.chunk_chars,
            english_only=args.english_only,
        )
        print(
            f"Indexed {stats['docs']} docs ({stats['chunks']} chunks, vectors={bool(stats['has_vectors'])}, "
            f"skipped_non_english={stats.get('skipped_non_english', 0)}) -> {out_dir}"
        )
        return 0

    tok_path = args.tokenizer or (paths.data_dir / "tokenizer" / "tokenizer.json")
    if args.ask is not None:
        from coregrad.app.cli import run_single_query

        run_single_query(
            args.checkpoint,
            tok_path,
            args.ask,
            index_dir=args.index,
            use_embeddinggemma2=args.use_embeddinggemma2,
            mrl_dim=args.mrl_dim,
            temperature=args.temperature,
            use_kv_cache=args.kv_cache,
            query_head_path=args.query_head,
            kv_top_k=args.kv_top_k,
        )
        return 0

    run_interactive_chat(
        args.checkpoint,
        tok_path,
        index_dir=args.index,
        use_embeddinggemma2=args.use_embeddinggemma2,
        mrl_dim=args.mrl_dim,
        temperature=args.temperature,
        use_kv_cache=args.kv_cache,
        query_head_path=args.query_head,
        kv_top_k=args.kv_top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
