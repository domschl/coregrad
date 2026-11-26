# coregrad

A compact, open-data **cognitive core** LLM built and trained from scratch in clean PyTorch on a single consumer GPU (12 GB VRAM).

`coregrad` combines the minimal, readable engineering ethos of `micrograd` / `nanoGPT` with the cognitive-core thesis: dedicate a tiny model's parameter budget to **language, reasoning, and tool orchestration**, while outsourcing **world and personal knowledge** to external local retrieval (hybrid `google/embeddinggemma-2` + SQLite FTS5 BM25).

## Documentation & Study Template
- **[Study Guide](docs/STUDY_GUIDE.md):** Chapter-by-chapter walkthrough of every technique, ablation table, and further-study pointers mapped directly to `src/coregrad/`.
- **[Research Experiments](docs/EXPERIMENTS.md):** Phase 4 empirical findings (Latent Memory Bridge: Latent Query Head, Gated Cross-Attention falsification, and GQA KV-Bank Injection; plus upcoming Gate 3 essentialization ablations).
- **[Local Research Guide](docs/LOCAL_RESEARCH_GUIDE.md):** How to index and query your own Markdown libraries (hybrid `EmbeddingGemma 2` + BM25, multilingual filtering, external vector DB adapters).
- **[Datasets & Licenses](docs/DATASETS.md):** Exact Hugging Face repos, pinned revisions, licenses, mixture weights, and token/dedup counts (100% open data).
- **[Prior Research & References](docs/REFERENCES.md):** Full bibliography linking papers directly to code modules.
- **[Project Plan](plan/project_plan.md):** Phased roadmap, hardware budget, and Go/No-Go gates.

## Quickstart

```bash
# 1. Provision environment (Python 3.14 + PyTorch CUDA 13.4)
uv sync --python 3.14 --extra dev

# 2. Configure local dataset cache (gitignored)
cp configs/sources.example.yaml configs/sources.local.yaml
# Edit configs/sources.local.yaml or export COREGRAD_HF_CACHE=/path/to/hf-cache

# 3. Verify repo hygiene, unit tests, and cached public datasets
uv run python scripts/check_hygiene.py
uv run pytest
uv run python scripts/check_sources.py

# 4. Run GPU throughput micro-benchmark
uv run python scripts/bench_gpu.py --sweep

# 5. Live training monitor (loss/BPB/probe plots + sample generations)
uv run python scripts/monitor.py --port 8765

# 6. Pretrain (Gate 1) -> Prepare Phase 3 data -> SFT & GRPO (Gate 2) -> Research CLI
uv run python scripts/train.py --config configs/train_125m_base.yaml
uv run python scripts/prepare_phase3_data.py
uv run python scripts/train_sft.py --config configs/train_125m_sft.yaml
uv run python scripts/train_grpo.py --init-from runs/coregrad_125m_sft/best.pt
uv run python scripts/eval_gate2.py --checkpoint runs/coregrad_125m_grpo/best.pt
uv run python scripts/research_cli.py index ./my_markdown_docs --use-embeddinggemma2
uv run python scripts/research_cli.py chat --checkpoint runs/coregrad_125m_grpo/best.pt --index data/index/my_markdown_docs
```

## Architecture Highlights (`coregrad-125m`)
- Decoder-only Transformer (~126M params): 18 layers, `d_model=768`, GQA (12 Q / 4 KV heads), SwiGLU (`d_ff=2048`), RoPE, Pre-RMSNorm + QK-Norm, tied 16k BPE embeddings, Flash SDPA, `torch.compile`.
- 100% open-licensed English pretraining, tool-use, and deterministic synthetic reasoning data through Gates 1–3.

## Authors & AI Collaboration
- **Project Concept & Direction:** `domschl`
- **Architecture, Implementation & Planning:** Antigravity (Gemini 4 Neon, Google DeepMind) in collaboration with `domschl`

## License
`coregrad` source code, configs, documentation, and trained open-data artifacts are released under the **[MIT License](LICENSE)**. Upstream public training datasets retain their original open licenses as documented in [docs/DATASETS.md](docs/DATASETS.md).
