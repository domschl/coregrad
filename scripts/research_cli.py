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

    p_idx = sub.add_parser("index", help="Index a folder of Markdown files (BM25 + optional EmbeddingGemma 2)")
    p_idx.add_argument("markdown_dir", type=Path)
    p_idx.add_argument("--out", type=Path, default=None)
    p_idx.add_argument("--use-embeddinggemma2", action="store_true")
    p_idx.add_argument("--mrl-dim", type=int, default=256)
    p_idx.add_argument("--chunk-chars", type=int, default=1000)

    p_chat = sub.add_parser("chat", help="Start interactive research session")
    p_chat.add_argument("--checkpoint", type=Path, required=True)
    p_chat.add_argument("--index", type=Path, default=None)
    p_chat.add_argument("--tokenizer", type=Path, default=None)
    p_chat.add_argument("--use-embeddinggemma2", action="store_true")
    p_chat.add_argument("--mrl-dim", type=int, default=256)
    p_chat.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()
    paths = load_local_paths()

    if args.cmd == "index":
        out_dir = args.out or (paths.data_dir / "index" / args.markdown_dir.name)
        stats = build_markdown_index(
            args.markdown_dir,
            out_dir,
            use_embeddinggemma2=args.use_embeddinggemma2,
            mrl_dim=args.mrl_dim,
            chunk_chars=args.chunk_chars,
        )
        print(f"Indexed {stats['docs']} docs ({stats['chunks']} chunks, vectors={bool(stats['has_vectors'])}) -> {out_dir}")
        return 0

    tok_path = args.tokenizer or (paths.data_dir / "tokenizer" / "tokenizer.json")
    run_interactive_chat(
        args.checkpoint,
        tok_path,
        index_dir=args.index,
        use_embeddinggemma2=args.use_embeddinggemma2,
        mrl_dim=args.mrl_dim,
        temperature=args.temperature,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
