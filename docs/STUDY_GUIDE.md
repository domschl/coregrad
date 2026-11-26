# The `coregrad` Study Guide
## Building a Compact Cognitive Core LLM from Scratch on One 12 GB GPU

This guide walks through every engineering and research choice in `coregrad` — **why** each technique was chosen, **where** it lives in the codebase, **what** we measured empirically on an RTX 4070 Ti (12 GB VRAM), and **where to go next** if you want to replicate or extend this work.

> See [`DATASETS.md`](DATASETS.md) for exact open dataset revisions/licenses and [`REFERENCES.md`](REFERENCES.md) for full academic citations.

---

## Chapter 1 — The Cognitive Core Thesis & Hardware Budgeting
**Code:** [`scripts/bench_gpu.py`](../scripts/bench_gpu.py) · [`plan/project_plan.md`](../plan/project_plan.md)

### Intuition
Large frontier LLMs spend vast parameter capacity acting as lossy encyclopedias. On a single 12 GB GPU, trying to memorize Wikipedia inside a 125M-parameter model is a losing battle. The **Cognitive Core** strategy flips the objective:
- Put **language fluency, planning, logical control flow, and tool orchestration** in the weights.
- Keep **facts, books, notes, and exact arithmetic** outside the weights, accessed via crisp tools (`search`, `read`, `list_docs`, `calc`).

### Hardware Math First (Measure Before Training!)
A consumer Ada Lovelace RTX 4070 Ti has ~80 TFLOPS dense BF16 peak (with FP32 accumulation) and 11.68 GB usable VRAM. Before writing any training loop, we wrote [`bench_gpu.py`](../scripts/bench_gpu.py) and locked real throughput:

| Model | Params | Seq Len | Micro-Batch | Act. Ckpt | Measured Throughput | Peak VRAM | Hours / 1B Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `coregrad-12m` | 12.25M | 1024 | 16 | Off | ~370k tok/s | 2.9 GB | ~0.75 h |
| `coregrad-45m` | 43.79M | 2048 | 8 | Off | 131k tok/s | 6.0 GB | 2.12 h |
| `coregrad-125m` | 125.86M | 2048 | 4 | Off | 52.6k tok/s | 6.9 GB | 5.29 h |
| `coregrad-125m` | 125.86M | 2048 | 8 | On | 43.0k tok/s | 4.3 GB | 6.46 h |

**Lesson:** Always run a 60-second micro-benchmark first. At 52.6k tok/s, a 6B-token run on a 125M model takes ~31 hours — totally doable across 1–2 weekend sessions with crash-safe checkpointing.

- **Further Study:** Hoffmann et al. (2022) *Chinchilla*; Austin et al. *How to Scale Your Model* (JAX Scaling Book).
- **Try It:** Run `uv run python scripts/bench_gpu.py --sweep` on your own GPU to compute your exact token/hour budget.

---

## Chapter 2 — Tokenizer Design for Tiny Models
**Code:** [`src/coregrad/tokenizer/bpe.py`](../src/coregrad/tokenizer/bpe.py) · [`special_tokens.py`](../src/coregrad/tokenizer/special_tokens.py) · [`scripts/train_tokenizer.py`](../scripts/train_tokenizer.py)

### Why 16,384 Vocab (Not 50k or 128k)?
An embedding matrix has $V \times d_{\text{model}}$ parameters. At $d_{\text{model}} = 768$:
- $V = 50{,}000 \rightarrow 38.4\text{M}$ embedding params (~30% of a 125M model, and >100% of a 30M model!).
- $V = 16{,}384 \rightarrow 12.6\text{M}$ embedding params — AND cuts the final softmax logit tensor VRAM by 3×!

We trained both 16k and 32k byte-level BPE tokenizers on 160 MB of balanced public text and compared **bytes per token** (higher = better compression):

| Domain | `vocab=16,384` | `vocab=32,768` | 32k Gain |
| :--- | :---: | :---: | :---: |
| `edu-en` (web English) | **4.036 B/tok** | 4.348 B/tok | +7.7% |
| `cosmo` (textbooks) | **4.227 B/tok** | 4.424 B/tok | +4.7% |
| `code-starcoder` | **2.977 B/tok** | 3.226 B/tok | +8.4% |
| `finemath` | **3.294 B/tok** | 3.461 B/tok | +5.1% |
| `tool-dsl` | **3.029 B/tok** | 3.285 B/tok | +8.5% |

16k already exceeds 4.0 bytes/token on English prose for a tiny fraction of the parameter cost.

### Two Subtle Tokenizer Wins Built In from Day 1
1. **Single-digit splitting (`\p{N}` in `PRETOKENIZER_REGEX`):** Forces `1979` to tokenize as `["1", "9", "7", "9"]` instead of arbitrary multi-digit chunks like `["19", "79"]`. This makes numeric reasoning and `calc(expr)` arguments far cleaner for small models.
2. **32 Reserved Special Tokens (`IDs 0..31`):** `<|think|>`, `<|call|>`, `<|result|>`, `<|cite|>`, etc. exist as native single tokens from the first pretraining step, so we never need to resize embeddings during post-training.

- **Further Study:** Karpathy's *Let's build the GPT Tokenizer* lecture; Hugging Face `tokenizers` library docs.
- **Try It:** Re-train a 4,096 or 8,192 vocab tokenizer and compare BPB on a 12M model (a classic Gate 3 essentialization experiment!).

---

## Chapter 3 — Data Curation, Deduplication & Tokenizer-Agnostic BPB
**Code:** [`src/coregrad/data/sources.py`](../src/coregrad/data/sources.py) · [`pack.py`](../src/coregrad/data/pack.py) · [`loader.py`](../src/coregrad/data/loader.py) · [`src/coregrad/eval/bpb.py`](../src/coregrad/eval/bpb.py)

### Key Design Choices
1. **Quality over Volume:** Mixing `fineweb-edu` (45%), `cosmopedia` textbooks (25%), `starcoderdata` (17%), and `finemath` (13%) packs dramatically more reasoning structure per token than raw web crawls.
2. **Separate Domain Shards + Dynamic Mixture Loader:** Instead of baking one fixed mixture into giant files, each domain is packed into its own memory-mapped `uint16` shards (`data/shards/<domain>/*.bin`). [`MixtureDataLoader`](../src/coregrad/data/loader.py) mixes domains dynamically at runtime — letting you adjust domain weights for annealing or ablations in 1 second without re-tokenizing.
3. **Zero Train/Val Leakage:** Validation hold-out is decided by a deterministic 64-bit `blake2b` hash of each normalized document *after* exact deduplication.
4. **Why Bits-Per-Byte (BPB) Instead of Raw Loss?** Cross-entropy loss in nats/token depends on vocabulary size (a 4k vocab looks artificially "lower loss per token" than a 32k vocab because it predicts shorter pieces!). Converting to **BPB**:
   $$\text{BPB} = \frac{\mathcal{L}_{\text{nats}}}{\ln(2) \cdot \text{bytes\_per\_token}}$$
   gives a **100% fair, tokenizer-agnostic** metric across every model and vocabulary size.

- **Further Study:** FineWeb-Edu technical report (`arXiv:2406.17557`); SmolLM2 data curation (`arXiv:2502.02737`).

---

## Chapter 4 — Modern Compact Transformer Architecture
**Code:** [`src/coregrad/model/transformer.py`](../src/coregrad/model/transformer.py) (~250 lines)

Every line of `CoreGradLM` reflects lessons from post-2023 LLM engineering:
- **Pre-RMSNorm (`F.rms_norm`):** Faster and simpler than LayerNorm (no mean subtraction or bias).
- **QK-Norm:** Applies RMSNorm to `Q` and `K` per head before dot-product attention — completely eliminates attention-logit blowup, letting us train stably with aggressive learning rates.
- **RoPE (Rotary Position Embeddings):** Encodes relative position via complex rotations in `Q`/`K`, enabling clean context extension later.
- **GQA (Grouped-Query Attention, 12 Q / 4 KV heads):** Cuts KV cache size 3× at inference with virtually zero quality loss.
- **SwiGLU MLP (bias-free):** Consistently outperforms GELU/ReLU MLPs at matched parameter count.
- **Deep-and-Narrow Geometry (`18 layers × d_model=768`):** Following *MobileLLM*, deeper narrow networks outperform shallow wide networks at ~100M scale on reasoning tasks.
- **Tied Input/Output Embeddings + Flash SDPA + `torch.compile`:** Saves 12.6M params and fuses kernels automatically.

- **Further Study:** *MobileLLM* (`arXiv:2402.14905`); *GQA* (`arXiv:2305.13245`); *RoFormer* (`arXiv:2104.09864`).

---

## Chapter 5 — Muon Optimizer & WSD Learning-Rate Schedule
**Code:** [`src/coregrad/train/optim.py`](../src/coregrad/train/optim.py) · [`schedule.py`](../src/coregrad/train/schedule.py) · [`trainer.py`](../src/coregrad/train/trainer.py)

### Why Muon Beat AdamW in Our Controlled A/B Test
**Muon** applies Nesterov momentum followed by a 5-step quintic **Newton-Schulz orthogonalization** to 2D hidden weight matrices (keeping fused AdamW for embeddings and 1D norms). Intuitively, standard SGD/Adam updates are often dominated by a few high-variance singular directions; orthogonalizing the update matrix spreads learning evenly across all feature directions.

On our exact same 12M / 150M-token smoke benchmark (identical seed, batches, and schedule):

| Metric | Fused AdamW | **Hybrid Muon + AdamW** | Delta |
| :--- | :---: | :---: | :---: |
| Final Train Loss | 3.3658 | **3.2949** | **-0.0709** |
| Mean Validation BPB | 1.1415 | **1.1091** | **-0.0324 BPB** |
| Peak VRAM / Wall Time | 2.91 GB / 7.4 min | **2.88 GB / 7.6 min** | Lower VRAM, +2.7% time |

### Why WSD (Warmup–Stable–Decay) Instead of Fixed Cosine?
With a classic cosine schedule, you must commit to `total_tokens` upfront — extending a finished run requires awkward restarts. **WSD** stays at constant peak LR during the stable phase and only decays at the end. Saving milestone checkpoints along the stable plateau (`ckpt_step_020000.pt`) lets us branch new decay/annealing/mid-training runs from any point!

### Measured Pretraining Scaling Ladder (Gate 1)

| Run | Params | Tokens | Mean Val BPB | `edu-en` BPB | `cosmo` BPB | `code` BPB | `finemath` BPB | Grammar (32) | 4-Way (24) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `12m_smoke_muon` | 12.25M | 150M | 1.1091 | 1.324 | 0.889 | 0.904 | 1.320 | 87.5% | 29.2% |
| `45m_proto` | 43.79M | 1.5B | 0.8609 | 1.111 | 0.672 | 0.660 | 1.001 | 96.9% | 50.0% |
| **`125m_base`** | **125.86M** | **6.0B** | **0.7371** | **0.990** | **0.543** | **0.561** | **0.854** | **93.8%** | **62.5%** |

- **Further Study:** Keller Jordan's Muon blog; Moonshot AI *Moonlight* (`arXiv:2502.16982`); Hägele et al. *WSD Scaling* (`arXiv:2405.18392`).

---

## Chapter 6 — Tool Protocol, Safe Sandboxing & Synthetic Environments
**Code:** [`src/coregrad/tools/`](../src/coregrad/tools/) · [`src/coregrad/synthetic/`](../src/coregrad/synthetic/)

### Narrow DSL over Arbitrary Deep JSON
A 125M model can become rock-solid on a **small, unambiguous 4-tool grammar**:
- `search(query, k=3)` · `read(doc_id, start, length)` · `calc(expr)` · `list_docs(filter)`
- Parsed strictly via Python's `ast.parse(..., mode="eval")` (zero `eval()` injection risk!) and wrapped in native special tokens (`<|think|>`, `<|call|>`, `<|result|>`, `<|cite|>`).

### Why Fictional-World Synthetic Data Matters
If you test RAG on real Wikipedia facts (e.g., *"When was the Eiffel Tower built?"*), you cannot tell whether the model retrieved the fact or memorized it during pretraining. [`retrieval_gen.py`](../src/coregrad/synthetic/retrieval_gen.py) generates **procedural fictional libraries** with **strictly disjoint train and eval entity vocabularies** — forcing 0% parametric memorization and proving genuine 1-hop, 2-hop, retrieval+`calc`, and abstention capability!

- **Further Study:** *ReAct* (`arXiv:2210.03629`); *Toolformer* (`arXiv:2302.04761`).

---

## Chapter 7 — Packed Masked SFT & Verifiable RL (GRPO)
**Code:** [`src/coregrad/train/sft.py`](../src/coregrad/train/sft.py) · [`src/coregrad/train/grpo.py`](../src/coregrad/train/grpo.py) · [`src/coregrad/tools/rollout.py`](../src/coregrad/tools/rollout.py)

### Two Critical SFT Engineering Details
1. **Strict Target Masking (`-100`):** During SFT, loss is computed **only** on assistant-generated tokens (`<|think|>`, `<|call|>`, final text, `<|eos|>`). Prompt (`<|system|>`, `<|user|>`) and tool output (`<|result|>`) tokens are masked to `-100` so the model never learns to hallucinate fake `<|result|>` blocks.
2. **Sequence Packing (3× Speedup):** Our 153k SFT trajectories have a median length of 267 tokens (mean 520). Padding each to 2,048 wastes ~75% of GPU compute. Packing multiple conversations per row (with `bos` masked to `-100` at every boundary) cuts 2 full epochs from ~4 hours to **~50 minutes** on an RTX 4070 Ti.

### Verifiable RL (GRPO / RLVR) Without a Reward Model
In [`grpo.py`](../src/coregrad/train/grpo.py), the model generates $G$ live multi-turn rollouts per task interacting with real `ToolExecutor` calls. Rewards are 100% programmatic:
- `+0.2` valid tool syntax & execution, `+0.5` exact answer or calibrated abstention, `+0.2` valid grounded citation, `-0.5` hallucinated citation.
Group-normalized advantages + PPO clipping + KL anchor optimize multi-step tool control directly.

### Measured Gate 2 Results & The "100% Synthetic Isn't Enough" Lesson

A core scientific lesson from `coregrad` is **always test out-of-distribution (OOD)** even when your held-out benchmark hits 100%.

1. **SFT v1** (80k narrow synthetic + 45k public tool + 30k `smol-smoltalk` *without* `DEFAULT_SYSTEM_PROMPT`) scored **100.0% across all 220 locked Gate 2 tasks** (disjoint fictional entities and numbers), yet failed an 8-prompt real-Markdown + chat OOD test:
   - Called `search(...)` on pure chat (*"Hello! Who are you?"*) due to system-prompt shortcutting.
   - Retrieved the right unseen Markdown file (`voyager.md` at Rank #1) but failed open-domain span extraction because synthetic docs shared one rigid bio layout.
2. **SFT v2** fixed all three gaps by mixing **55k SQuAD v2** real Wikipedia RAG/abstention, **13.8k GSM8K** verified `calc` word problems, and **45k `smol-smoltalk`** with aligned `DEFAULT_SYSTEM_PROMPT` (plus BM25 stopword filtering & SQLite FTS5 `porter` stemming):

| Benchmark Tier | Metric | SFT v1 (`tools=ON`) | **SFT v2 (`tools=ON`)** | `tools=OFF` Baseline |
| :--- | :--- | :---: | :---: | :---: |
| **Locked Gate 2 Suite** *(220 disjoint tasks)* | Tool syntax validity | 100.0% (372/372) | **100.0% (377/377)** | 0.0% |
| | Multi-step `calc` exact | 100.0% (100/100) | **100.0% (100/100)** | 0.0% |
| | Grounded retrieval (1-hop, 2-hop, ret+calc) | 100.0% (90/90) | **94.4% (85/90)** | 0.0% |
| | Grounded citations (hallucinated = 0) | 100.0% | **96.7% (0 halluc.)** | 0.0% |
| | Calibrated abstention | 100.0% (30/30) | **100.0% (30/30)** | 0.0% |
| **OOD Stress Test** *(real MD + chat)* | Clean no-tool chat routing | 0/2 (spurious calls) | **2/2 (100%)** | — |
| | Unseen MD retrieval + grounded citation | 0/3 (looped) | **3/3 cited (`d0005`, `d0003`, `d0004`)** | — |
| **Held-Out SQuAD v2 Val** *(100 real tasks)* | Target doc retrieved & cited (0 halluc.) | — | **80.0% (40/50)** | — |
| | Exact open-domain answer span match | — | **28.0% (14/50)** | — |
| | Adversarial unanswerable abstention | — | **16.0% (8/50)** | — |
| **Held-Out GSM8K Test** *(50 human problems)* | Valid `calc(expr)` syntax | — | **100.0% (186/186)** | — |
| | Multi-step free-form exact match | — | **2.0% (1/50)** | — |

**Key takeaway:** At 125M parameters trained from scratch on one 12 GB GPU, the cognitive core already masters **tool syntax (100% even on unseen GSM8K)**, **structured multi-step tool chains (94–100%)**, **no-tool chat routing**, and **faithful document citation (80% on unseen Wikipedia with 0 hallucinated citations)** — while fine-grained OOD span extraction (28% SQuAD v2), adversarial abstention (16%), and 4–6 step free-form GSM8K word problems mark the exact frontier where **RLVR (GRPO)** and **Phase 4 looped/latent-memory architectures** can push next.

- **Further Study:** *DeepSeekMath / GRPO* (`arXiv:2402.03300`); *DeepSeek-R1* (`arXiv:2501.12948`); *SQuAD v2* (`arXiv:1806.03822`).

---

## Chapter 8 — Hybrid Retrieval (`EmbeddingGemma 2` + SQLite FTS5 BM25)
**Code:** [`src/coregrad/tools/retriever.py`](../src/coregrad/tools/retriever.py) · [`src/coregrad/app/cli.py`](../src/coregrad/app/cli.py)

### Why Hybrid Beats Either Alone
- **Dense Semantic Vectors (`google/embeddinggemma-2`, Apache-2.0, MRL 256d):** Understands paraphrases and conceptual queries (*"monetary crisis"* matching *"currency collapse"*). Loading only the ~270M text backbone takes ~550 MB BF16.
- **Sparse Lexical BM25 (SQLite FTS5):** Nails exact entity names, catalog codes (`ALD-59`), dates, and rare identifiers where dense embeddings can blur neighbors.
- **Reciprocal Rank Fusion (RRF):** Combines both rankings via $\sum_r \frac{1}{60 + \text{rank}_r(d)}$ — parameter-free, immune to score-scale mismatch, and ~15 lines of code.
- **Context-Aware Chunk Sizing:** Chunks are sized to ~1,000 chars (~250 tokens) so `k=3` hits take ~750 tokens, leaving plenty of working room inside a 2,048-token context window.

- **Further Study:** Robertson & Zaragoza *BM25* (2009); Cormack et al. *RRF* (2009); Kusupati et al. *Matryoshka Representation Learning* (`arXiv:2205.13147`).

---

## Chapter 9 — Where to Go Next (Phase 4 Open Research Exercises)
Once you reproduce Gates 1 & 2, `coregrad` is designed as a fast single-GPU laboratory for cutting-edge small-LM research:
1. **Essentialization Ladder:** Train `60M → 30M → 15M → 8M` cores on the locked Gate 2 suite to find the exact parameter cliff for tool use.
2. **Looped / Recursive Transformers:** Replace 18 unique layers with 4–6 weight-shared layers looped 3–4× — can depth-without-parameters preserve 2-hop reasoning at 25M params?
3. **Cognitive Core Purity Test:** Add Wikipedia back into pretraining at 30M scale — does memorizing trivia *hurt* externalized tool-use accuracy?
4. **Latent Memory Bridge (Phase 4B):** Attach zero-initialized gated cross-attention (`tanh(α)`) + a middle-layer latent query head to let layers 9–15 attend directly to EmbeddingGemma 2 memory vectors without consuming text context tokens.
