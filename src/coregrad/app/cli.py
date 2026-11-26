"""Interactive local Markdown research CLI powered by coregrad + Hybrid EmbeddingGemma 2/BM25."""

from __future__ import annotations

from pathlib import Path

import torch

from coregrad.model import CoreGradLM, ModelConfig
from coregrad.model.memory_bridge import (
    CoreGradMemoryBridge,
    LatentQueryHead,
    MemoryBridgeConfig,
    encode_kv_passage,
    forward_with_kv_bank,
)
from coregrad.tokenizer import CoreGradTokenizer
from coregrad.tools import EmbeddingGemma2Embedder, HybridMarkdownIndex, SearchHit, ToolExecutor
from coregrad.tools.rollout import run_agent_rollout


class _LatentKVSession:
    """Opt-in Phase 4B runtime: optional LatentQueryHead retrieval + cached GQA KV-bank injection."""

    def __init__(
        self,
        model: CoreGradLM,
        tokenizer: CoreGradTokenizer,
        index: HybridMarkdownIndex,
        device: torch.device,
        *,
        query_head_path: Path | None = None,
        top_k: int = 4,
        max_passage_tokens: int = 220,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.index = index
        self.device = device
        self.top_k = top_k
        self.max_passage_tokens = max_passage_tokens
        self._kv_memo: dict[tuple[str, int], list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._query_head: LatentQueryHead | None = None
        self._query_layer: int = model.cfg.n_layer
        self._bridge: CoreGradMemoryBridge | None = None

        if query_head_path is not None:
            qh_ckpt = torch.load(query_head_path, map_location="cpu", weights_only=False)
            self._query_layer = int(qh_ckpt["layer"])
            self._query_head = LatentQueryHead(
                d_model=int(qh_ckpt["d_model"]),
                query_dim=int(qh_ckpt["query_dim"]),
            ).to(device).eval()
            self._query_head.load_state_dict(qh_ckpt["state_dict"])
            self._bridge = CoreGradMemoryBridge(
                model, MemoryBridgeConfig(bridge_layers=(), freeze_base=True), memory_mode="retro"
            ).to(device).eval()

    @torch.no_grad()
    def _retrieve(self, question: str) -> tuple[list[SearchHit], str]:
        if self._query_head is not None and self._bridge is not None:
            ids = self.tokenizer.encode(question, add_bos=True, add_eos=False)[:128]
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            amp = self.device.type == "cuda"
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=amp):
                h = self._bridge.extract_layer_hidden(x, (self._query_layer,))[self._query_layer][:, -1, :]
                q_vec = self._query_head(h.float()).squeeze(0).cpu().numpy()
            try:
                hits = self.index.search_by_vector(q_vec, k=self.top_k)
                if hits:
                    return hits, f"latent_query(layer={self._query_layer})"
            except ValueError:
                pass
        return self.index.search(question, k=self.top_k), "hybrid_search"

    @torch.no_grad()
    def _get_kv(self, hit: SearchHit) -> list[tuple[torch.Tensor, torch.Tensor]]:
        key = (hit.doc_id, hit.start)
        if key not in self._kv_memo:
            ids = self.tokenizer.encode(
                f"Passage [{hit.doc_id} | {hit.title}]: {hit.text}\n\n",
                add_bos=True,
                add_eos=False,
            )[: self.max_passage_tokens]
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            amp = self.device.type == "cuda"
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=amp):
                kv = encode_kv_passage(self.model, x)
            self._kv_memo[key] = [(k.to(torch.bfloat16).cpu(), v.to(torch.bfloat16).cpu()) for k, v in kv]
        return self._kv_memo[key]

    @torch.no_grad()
    def answer(self, question: str, max_new_tokens: int = 96) -> tuple[str, list[str], str]:
        hits, mode = self._retrieve(question)
        kv_bank: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if hits:
            kvs = [self._get_kv(h) for h in hits]
            max_mem = self.model.cfg.max_seq_len - 192
            n_layer = len(kvs[0])
            kv_bank = []
            for li in range(n_layer):
                ks = torch.cat(
                    [kv[li][0] if i == 0 else kv[li][0][:, :, 1:, :] for i, kv in enumerate(kvs)], dim=2
                )[:, :, :max_mem, :].to(self.device)
                vs = torch.cat(
                    [kv[li][1] if i == 0 else kv[li][1][:, :, 1:, :] for i, kv in enumerate(kvs)], dim=2
                )[:, :, :max_mem, :].to(self.device)
                kv_bank.append((ks, vs))

        p_ids = self.tokenizer.encode(
            f"\n\nQuestion: {question}\nAnswer:", add_bos=(kv_bank is None), add_eos=False
        )
        x = torch.tensor([p_ids], dtype=torch.long, device=self.device)
        eos_id = self.tokenizer.eos_id
        amp = self.device.type == "cuda"
        gen: list[int] = []
        for _ in range(max_new_tokens):
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=amp):
                logits, _ = forward_with_kv_bank(self.model, x, kv_bank=kv_bank)
            nxt = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            if nxt == eos_id:
                break
            gen.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], dtype=torch.long, device=self.device)], dim=1)
        ans = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        return ans, sorted({h.doc_id for h in hits}), mode


def build_markdown_index(
    markdown_dirs: Path | list[Path],
    index_dir: Path,
    *,
    use_embeddinggemma2: bool = False,
    mrl_dim: int = 256,
    chunk_chars: int = 1000,
    english_only: bool = False,
) -> dict[str, int]:
    embedder = EmbeddingGemma2Embedder(mrl_dim=mrl_dim) if use_embeddinggemma2 else None
    index = HybridMarkdownIndex(index_dir, embedder=embedder)
    try:
        stats = index.index_markdown_directory(
            markdown_dirs, chunk_chars=chunk_chars, english_only=english_only
        )
    finally:
        index.close()
    return stats


def run_single_query(
    checkpoint_path: Path,
    tokenizer_path: Path,
    question: str,
    index_dir: Path | None = None,
    *,
    use_embeddinggemma2: bool = False,
    mrl_dim: int = 256,
    temperature: float = 0.0,
    use_kv_cache: bool = False,
    query_head_path: Path | None = None,
    kv_top_k: int = 4,
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

    try:
        if use_kv_cache and index is not None:
            session = _LatentKVSession(
                model, tokenizer, index, device, query_head_path=query_head_path, top_k=kv_top_k
            )
            answer, docs, mode = session.answer(question)
            print(f"Question : {question}")
            print(f"Mode     : kv-cache ({mode}, k={kv_top_k})")
            print(f"Answer   : {answer}")
            if docs:
                print(f"KV Docs  : {', '.join(docs)}")
            return

        trace = run_agent_rollout(
            model,
            tokenizer,
            question,
            ToolExecutor(index),
            device,
            max_tool_turns=4,
            temperature=temperature,
        )
        print(f"Question : {question}")
        if trace.tool_calls_raw:
            print(f"Tools    : {trace.tool_calls_raw}")
        print(f"Answer   : {trace.final_answer}")
        if trace.cited_doc_ids:
            print(f"Citations: {', '.join(sorted(set(trace.cited_doc_ids)))}")
    finally:
        if index is not None:
            index.close()


def run_interactive_chat(
    checkpoint_path: Path,
    tokenizer_path: Path,
    index_dir: Path | None = None,
    *,
    use_embeddinggemma2: bool = False,
    mrl_dim: int = 256,
    temperature: float = 0.0,
    use_kv_cache: bool = False,
    query_head_path: Path | None = None,
    kv_top_k: int = 4,
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
    kv_session = (
        _LatentKVSession(model, tokenizer, index, device, query_head_path=query_head_path, top_k=kv_top_k)
        if (use_kv_cache and index is not None)
        else None
    )
    mode_tag = f"kv-cache(k={kv_top_k})" if kv_session is not None else "4-tool"
    print(f"coregrad research CLI ready ({model.cfg.name} on {device}, mode={mode_tag}). Type 'exit' to quit.")
    try:
        while True:
            try:
                question = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question or question.lower() in {"exit", "quit", ":q"}:
                break
            if kv_session is not None:
                answer, docs, mode = kv_session.answer(question)
                print(f"[kv-cache:{mode}] docs={docs}")
                print(f"coregrad> {answer}")
                continue
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
