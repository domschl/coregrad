# coregrad

A compact, open-data **cognitive core** LLM built and trained from scratch in clean PyTorch on a single consumer GPU (12 GB VRAM).

`coregrad` combines the minimal, readable engineering ethos of `micrograd` / `nanoGPT` with the cognitive-core thesis: dedicate a tiny model's parameter budget to **language, reasoning, and tool orchestration**, while outsourcing **world and personal knowledge** to external local retrieval (hybrid `google/embeddinggemma-2` + SQLite FTS5 BM25).

See [plan/project_plan.md](plan/project_plan.md) for the full phased roadmap and Go/No-Go gates.

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
```

## Architecture Highlights (`coregrad-125m`)
- Decoder-only Transformer (~124M params): 16 layers, `d_model=768`, GQA (12 Q / 4 KV heads), SwiGLU (`d_ff=2048`), RoPE, Pre-RMSNorm + QK-Norm, tied 16k BPE embeddings, Flash SDPA, `torch.compile`.
- 100% open-licensed English pretraining, tool-use, and deterministic synthetic reasoning data through Gates 1–3.
