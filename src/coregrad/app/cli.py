"""Interactive local Markdown research CLI powered by coregrad + Hybrid EmbeddingGemma 2/BM25."""

from __future__ import annotations

from pathlib import Path

import torch

from coregrad.model import CoreGradLM, ModelConfig
from coregrad.tokenizer import CoreGradTokenizer
from coregrad.tools import EmbeddingGemma2Embedder, HybridMarkdownIndex, ToolExecutor
from coregrad.tools.rollout import run_agent_rollout


def build_markdown_index(
    markdown_dir: Path,
    index_dir: Path,
    *,
    use_embeddinggemma2: bool = False,
    mrl_dim: int = 256,
    chunk_chars: int = 1000,
) -> dict[str, int]:
    embedder = EmbeddingGemma2Embedder(mrl_dim=mrl_dim) if use_embeddinggemma2 else None
    index = HybridMarkdownIndex(index_dir, embedder=embedder)
    try:
        stats = index.index_markdown_directory(markdown_dir, chunk_chars=chunk_chars)
    finally:
        index.close()
    return stats


def run_interactive_chat(
    checkpoint_path: Path,
    tokenizer_path: Path,
    index_dir: Path | None = None,
    *,
    use_embeddinggemma2: bool = False,
    mrl_dim: int = 256,
    temperature: float = 0.0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CoreGradTokenizer.from_file(tokenizer_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CoreGradLM(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    index: HybridMarkdownIndex | None = None
    if index_dir is not None and index_dir.exists():
        embedder = EmbeddingGemma2Embedder(mrl_dim=mrl_dim) if use_embeddinggemma2 else None
        index = HybridMarkdownIndex(index_dir, embedder=embedder)

    executor = ToolExecutor(index)
    print(f"coregrad research CLI ready ({model.cfg.name} on {device}). Type 'exit' to quit.")
    try:
        while True:
            try:
                question = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question or question.lower() in {"exit", "quit", ":q"}:
                break
            trace = run_agent_rollout(
                model,
                tokenizer,
                question,
                executor,
                device,
                max_tool_turns=4,
                temperature=temperature,
            )
            if trace.tool_calls_raw:
                print(f"[tools] {trace.tool_calls_raw}")
            print(f"coregrad> {trace.final_answer}")
            if trace.cited_doc_ids:
                print(f"[citations] {', '.join(sorted(set(trace.cited_doc_ids)))}")
    finally:
        if index is not None:
            index.close()
