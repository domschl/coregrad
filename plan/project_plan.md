# `coregrad` — Project Plan

> **Goal:** Build, train, and systematically essentialize a compact **cognitive core** LLM from scratch on a single consumer GPU (RTX 4070 Ti, 12 GB VRAM) — combining the clean, readable engineering ethos of `micrograd` / `nanoGPT` with the thesis that a tiny model can master **English language, reasoning, and tool use** while outsourcing **world and personal knowledge** to external local retrieval.
>
> **Open-First Design:** Phases 0–4 use **exclusively public, openly licensed datasets**. Every script, tokenizer, benchmark, training recipe, and resulting model checkpoint can be published and reproduced without restriction. Personal/private libraries work out of the box at **inference time** via the local retrieval tools (no retraining required), with an optional plug-in path for private local training later.

---

## 1. Guiding Principles & Scope

1. **Cognitive Core over World Encyclopedia:** A 10M–150M parameter model cannot and should not memorize an encyclopedia. Its limited parameter budget should be dedicated to linguistic coherence, problem decomposition, logical control flow, knowing when it lacks information, calling tools accurately, and synthesizing cited answers from retrieved context.
2. **100% Public & Reproducible Core (Phases 0–4):** All pretraining, mid-training, SFT, synthetic generation, and evaluation artifacts are built strictly from public open datasets so the entire repository and trained weights can be shared openly.
3. **English-First:** Focus exclusively on English (plus code and structured tool syntax) through Gates 1–3 to maximize reasoning and tool-use capability per parameter.
4. **Staged Value Delivery:** Every phase ends with a concrete, testable milestone. Even partial completion produces a standalone, useful, publishable artifact (a modern compact PyTorch trainer at Gate 1; a working local-research agent at Gate 2; empirical minimal-architecture research at Gate 3).
5. **Measure Before Scaling:** Build the automated evaluation harness and hardware micro-benchmark *before* launching multi-day runs. Use a Warmup–Stable–Decay (WSD) learning-rate schedule so runs are checkpointable, branchable, and never wasted.

---

## 2. Hardware Envelope & Compute Reality Check

Target hardware constraint: **1× NVIDIA RTX 4070 Ti (12 GB VRAM, Ada Lovelace SM 8.9, BF16/FP8 Tensor Cores), 128 GB RAM, 20-thread desktop CPU.**

### Estimated Training Budget (BF16 + `torch.compile` + Flash SDPA, ~35–45% MFU ≈ 55–70 TFLOPS effective)

| Model Tier | Active Params | Est. VRAM (Weights + Opt + Acts, ctx 2048) | Est. Throughput | Time / 1B Tokens | Practical Budget |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Smoke** | ~12M | ~2–3 GB | ~220k–300k tok/s | ~1 hr | 100M–300M tok (15–30 min) |
| **Essentialized / Proto** | ~45M | ~4–6 GB | ~110k–150k tok/s | ~2–2.5 hrs | 1.5B–3B tok (3–7 hrs) |
| **Gate 1–2 Reference** | **~125M** | **~7–10 GB** (grad accum, micro-batch 8–16) | **~55k–80k tok/s** | **~3.5–5 hrs** | **5B–8B tok (~20–36 hrs)** |
| **Upper Bound** | ~180M | ~9–11.5 GB (with activation ckpt) | ~35k–50k tok/s | ~6–8 hrs | 5B–8B tok (~35–60 hrs) |

**Key takeaway:**
- **~125M parameters trained on 5B–8B high-signal tokens** is the sweet spot for Gate 1 & Gate 2: comfortably fits in 12 GB VRAM, finishes across 2–3 overnight/weekend sessions, and sits in the proven capability regime (comparable in scale to GPT-2 124M / SmolLM-135M, with modern architecture and far higher data quality).
- **128 GB system RAM** accelerates CPU-side deduplication, tokenization, shard shuffling, synthetic trajectory generation, and local hybrid search indexing without bottlenecking the GPU.

---

## 3. Repository Structure & Configuration Hygiene

```text
coregrad/
├── configs/
│   ├── sources.example.yaml     # Public schema pointing to HF repos & env-configured cache root
│   ├── sources.local.yaml       # GITIGNORED: local dataset cache/backup root (and future optional private sources)
│   ├── model_*.yaml             # Architecture presets (12m, 45m, 125m, looped_30m, ...)
│   └── train_*.yaml             # Schedules, mixtures, hyperparameters
├── data/                        # GITIGNORED: tokenized shards, eval caches, local RAG index
├── runs/                        # GITIGNORED: checkpoints, logs, metrics
├── src/coregrad/
│   ├── data/                    # HF/local loaders, cleaners, dedup, shard packer, mixture sampler
│   ├── synthetic/               # Deterministic generators (calc, multi-hop retrieval, abstention)
│   ├── tokenizer/               # Open BPE trainer & special-token registry
│   ├── model/                   # Clean PyTorch Transformer (+ looped/essentialized variants)
│   ├── train/                   # Pretrain (WSD + AdamW/Muon), mid-train, SFT, GRPO/RLVR
│   ├── eval/                    # BPB, language probes, tool-use & RAG faithfulness suite
│   ├── tools/                   # Sandboxed calc, search, read, list_docs, protocol parser
│   └── app/                     # Local Markdown research CLI & document indexer
├── scripts/                     # GPU benchmark, dataset sync, leak-check, training entrypoints
└── tests/                       # Unit tests for model shapes, masking, tool parser, hygiene guard
```

### Repository Hygiene Rules (Enforced from Day 1)
- **Zero Local Paths in Git:** All dataset cache and output roots are configured via environment variables (`COREGRAD_DATA_CACHE`, `COREGRAD_WORK_DIR`) or gitignored `configs/sources.local.yaml`.
- **Reproducible Dataset Manifest:** Every public dataset used is pinned in a public manifest (`configs/datasets.yaml`) by HuggingFace repo ID, revision/commit hash, license, and file list so anyone can reproduce the exact corpus.
- **Automated Hygiene Check:** `scripts/check_hygiene.py` verifies staged files contain no machine-specific paths or large binary artifacts (`.bin`, `.parquet`, `.pt`).

---

## 4. Phased Execution Plan & Go/No-Go Gates

### Phase 0: Environment, Hygiene & Hardware Baseline (Days 1–2)
**Objective:** Establish a clean, reproducible project skeleton and verify real hardware throughput before writing training pipelines.

- [x] Initialize Python project with `uv` (`pyproject.toml`, Python 3.14, PyTorch CUDA 13.4, `tokenizers`, `pyarrow`, `huggingface-hub`, `pytest`, `pyyaml`).
- [x] Set up `.gitignore`, `configs/datasets.yaml`, `configs/sources.example.yaml`, and `scripts/check_hygiene.py`.
- [x] Write a standalone GPU micro-benchmark (`scripts/bench_gpu.py`) testing a 125M transformer with `torch.compile`, BF16, and Flash SDPA at context lengths 1024 and 2048 to measure actual tok/s and peak VRAM on the 4070 Ti.
- [x] **Exit check:** Reproducible environment, verified **52.6k tok/s** at 125M (`seq=2048, mb=4, ckpt=False`, **6.94 GB** peak VRAM, **5.29 h / 1B tok**) and **131.3k tok/s** at 45M (`seq=2048, mb=8`, **6.00 GB** peak VRAM, **2.12 h / 1B tok**); hygiene checks and unit tests passing.

---

### Phase 1: Public Dataset Pipeline, Tokenizer & Eval Harness (Week 1)
**Objective:** Build a 100% public, English-focused, high-signal token pipeline (~6B–8B tokens) and a locked evaluation harness.

#### 1A. Public Pretraining Mixture (~6B–8B Tokens Packed into `uint16` Shards)
Prioritize **reasoning density, clean modern prose, and structured syntax** over raw encyclopedic trivia:

| Category | Target Share | Public Open Sources | License | Rationale |
| :--- | :---: | :--- | :--- | :--- |
| **Educational Web Prose** | ~45% | `HuggingFaceFW/fineweb-edu` (`sample/10BT` shards) | ODC-By | Clean, coherent, modern informative English |
| **Pedagogical Textbooks & Stories** | ~25% | `HuggingFaceTB/cosmopedia` (OpenStax, Khan Academy, Stanford, stories) | Apache-2.0 | Dense step-by-step explanations & long-form discourse |
| **Code & Structured Syntax** | ~15% | `bigcode/starcoderdata` (Python, JSON, YAML, Shell, JS/TS, Rust, C/C++) / `codeparrot/github-code-clean` | Permissive | Algorithmic logic, strict syntax, schema discipline |
| **Math & Formal Reasoning** | ~12% | `HuggingFaceTB/finemath` | ODC-By | Multi-step logical chains & symbolic manipulation |
| **Synthetic Bootstrap** | ~3% | Early format/QA/arithmetic mix | Open (generated) | Smooth bridge into Gate 2 mid-training |

*Additional public downloads (only if needed, cached to the configured HuggingFace backup root):*
- 1–2 extra shards of `HuggingFaceFW/fineweb-edu` (`sample/10BT`) if we want >3B tokens of pure educational English prose.
- A clean public instruction/chat SFT dataset (e.g., `HuggingFaceTB/smol-smoltalk`, Apache-2.0) for Phase 3 general conversational polish alongside tool use.
- *Encyclopedic dumps (e.g., Wikipedia):* Deliberately excluded from Gate 1 pretraining; reserved both as an optional public **external retrieval corpus** for Gate 2 demos and as the control condition in Gate 3's cognitive-core purity experiment.

#### 1B. Data Processing Pipeline & Tokenizer
- Streaming cleaners per source (Parquet/JSONL): Unicode normalization, whitespace cleanup, length/quality filters, strict document-level train/val split.
- Fast CPU-side deduplication (exact SHA-256 + MinHash LSH near-dedup) leveraging 128 GB RAM.
- Train a **16,384-token byte-level BPE tokenizer** (100% public data) with dedicated reserved special tokens:
  `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|think|>`, `<|/think|>`, `<|call|>`, `<|/call|>`, `<|result|>`, `<|/result|>`, `<|cite|>`, `<|/cite|>`.
  *(Why 16,384? At `d_model=768`, 16k vocab saves ~26M embedding parameters vs. 50k — critical for Gate 3 essentialization — while maintaining strong English + Python + JSON fertility. Benchmark 16k vs. 32k fertility on a sample before locking.)*
- Pack documents with `<|eos|>` boundaries into memory-mapped binary shards (`train/*.bin`, `val/*.bin`) with a deterministic domain-weighted sampler.

#### 1C. Automated Evaluation Harness (Built Before Training)
- **Language Quality (Gate 1):**
  - Tokenizer-agnostic **Bits-Per-Byte (BPB)** on fixed held-out validation splits (Edu-Web, Textbooks, Code, Math).
  - Lightweight zero-shot probes: HellaSwag (subset), ARC-Easy, LAMBADA, BLiMP (grammar).
  - Fixed 20-prompt qualitative suite logged at every checkpoint.
  - Calibration run against an open reference model (e.g., `HuggingFaceTB/SmolLM2-360M` and/or `SmolLM2-135M`) to ground target scores.
- **Cognitive Core Suite (Gate 2 — scaffolded now, scored in Phase 3):**
  - Tool-call syntax validity & argument schema pass rate.
  - Tool-routing accuracy (call tool vs. answer directly vs. abstain).
  - 1-hop & 2-hop synthetic retrieval QA with mandatory citations.
  - 1-to-3-step calculator/reasoning accuracy.
  - Public Markdown library QA benchmark (built from held-out OpenStax/Wikipedia articles converted to Markdown).

- [x] **Exit check:** Tokenized train/val shards built (**6.70B unique train tokens + 16.48M val tokens** across `edu-en`, `cosmo`, `code-starcoder`, `finemath`), 16,384-vocab public BPE tokenizer locked (>4.0 B/tok English), eval harness and live training monitor verified end-to-end (9/9 tests & hygiene passing).

---

### Phase 2: Gate 1 — Coherent Minimal Language Model (Weeks 2–3)
**Objective:** Train a clean, modern ~125M English language model from scratch that generates coherent, grammatical, on-topic prose.

#### 2A. Reference Architecture (`coregrad-125m`)
Modern dense decoder-only transformer in ~300–400 lines of readable PyTorch:
- **Geometry:** `n_layer=16`, `d_model=768`, `n_heads=12`, `n_kv_heads=4` (GQA 3:1), `d_ff=2048` (SwiGLU), `vocab_size=16384`, `max_seq_len=2048`, tied input/output embeddings → **~124M parameters** (deep-and-narrow ratio favors reasoning over memorization at small scale).
- **Stabilizers & Speed:** Pre-RMSNorm, **QK-Norm**, RoPE (`theta=10000`), bias-free linear layers, Flash SDPA, `torch.compile`.
- **Optimization:**
  - Start with fused **AdamW** baseline; test **Muon** on 2D hidden matrices during the 45M prototype run (consistently 20–35% faster convergence in compact LLM benchmarks).
  - **WSD (Warmup–Stable–Decay) LR schedule:** warmup 2% → constant peak LR → checkpoint every 500M tokens → branch short 10% decay runs to evaluate or anneal without restarting.
  - Effective batch size: **262,144 tokens/step** (e.g., micro-batch 8 × seq_len 2048 × grad_accum 16).

#### 2B. Progressive De-Risking Ladder
1. **Run 0 — Smoke (`coregrad-12m`, 150M tokens, ~20 min):** Verify gradient flow, loss smoothness, checkpoint save/resume, eval logging, zero NaNs.
2. **Run 1 — Fast Prototype (`coregrad-45m`, 1.5B tokens, ~4 hrs):** Validate data mixture balance, LR/WD, and AdamW vs. Muon before committing >24 hours.
3. **Run 2 — Gate 1 Base (`coregrad-125m`, 5B–8B tokens, ~24–36 hrs across resumable sessions):** Stable pretraining + 500M-token high-quality annealing decay (upsampling Cosmopedia textbooks + FineMath + clean Python/JSON + top-scoring FineWeb-Edu).

#### Gate 1 Pass Criteria
- [x] Smooth, converged validation BPB across all 4 domains (`coregrad_125m_base`, 6.0B tokens, Muon + WSD: **mean BPB `0.7371`** — `cosmo=0.5430` ppl 6.03, `code-starcoder=0.5610` ppl 3.13, `finemath=0.8539` ppl 6.78, `edu-en=0.9904` ppl 17.08).
- [x] Matches or beats GPT-2 124M quality on clean English validation BPB and grammar/completion probes (**93.8% (30/32)** grammar minimal pairs; **62.5% (15/24)** 4-way completions including **100% science**, **66.7% logic**, **66.7% commonsense**).
- [x] Qualitative 20-prompt review: coherent, grammatical, on-topic English, valid JSON/Python blocks, and clear confirmation that offloading exact arithmetic (`calc`) and exact facts (`search`/`read`) is the right Gate 2 target.

---

### Phase 3: Gate 2 — Cognitive Core: Tool-Use, Reasoning & Local Research (Weeks 4–6)
**Objective:** Transform the Gate 1 base model into a 100% open cognitive core that thinks, invokes a compact tool set, reasons over retrieved Markdown documents, and abstains honestly when evidence is absent.

#### 3A. Minimal Canonical Tool Protocol
A compact, unambiguous 4-tool interface designed for high reliability at ~125M parameters:

```text
<|user|>What year did Author X publish Book Y, and how many years before Z?<|/user|>
<|assistant|><|think|>I should look up the publication years of Book Y and Z in the document index, then calculate the difference.<|/think|>
<|call|>search(query="Author X Book Y publication year", k=3)<|/call|>
<|result|>[{"doc_id": "d412", "span": [1200, 1650], "text": "...Book Y was first published in 1962..."}]<|/result|>
<|call|>search(query="Z publication year", k=3)<|/call|>
<|result|>[{"doc_id": "d889", "span": [400, 820], "text": "...Z appeared in 1979..."}]<|/result|>
<|call|>calc(expr="1979 - 1962")<|/call|>
<|result|>17<|/result|>
<|think|>Book Y was published in 1962 (d412) and Z in 1979 (d889). The difference is 17 years.<|/think|>
Author X published Book Y in 1962 <|cite|>d412<|/cite|>, which was 17 years before Z appeared in 1979 <|cite|>d889<|/cite|>.<|eos|>
```

**Core Tool Set (4 Primitives):**
1. `search(query, k=3)` — **Hybrid Retriever (EmbeddingGemma 2 + BM25 via RRF):**
   - Uses `google/embeddinggemma-2` (Apache-2.0, selective text-only backbone ~270M params / ~550 MB BF16, MRL 256d/768d) for semantic/conceptual retrieval combined with SQLite FTS5 BM25 for exact entities, dates, and identifiers, fused via Reciprocal Rank Fusion (RRF).
   - Returns **compact ~200–300 token passages (`k=3`)** sized to fit cleanly inside a 2048-token context window (trimming larger external chunks around the peak similarity span when bridging to external indexes).
   - Decoupled behind a clean `Retriever` interface so high-throughput synthetic training/GRPO rollouts can use instantaneous BM25 while live research & public benchmarks use full hybrid EmbeddingGemma 2 + BM25 (and external vector indexes can be plugged in via a thin adapter).
2. `read(doc_id, start, length)` — expand surrounding context around a retrieved chunk.
3. `calc(expr)` — safe AST-evaluated arithmetic/Python expression runner.
4. `list_docs(filter)` — browse available document titles/metadata in the mounted library.

#### 3B. 100% Public + Deterministic Synthetic Post-Training Data
1. **Normalized Public Tool-Calling Corpora:** Convert `NousResearch/hermes-function-calling-v1` (Apache-2.0), `Salesforce/xlam-function-calling-60k` (CC-BY-4.0), and `glaiveai/glaive-function-calling-v2` into the canonical `<|call|>` / `<|result|>` format. Filter to clean 1–4 argument calls and preserve "no tool required" turns.
2. **Clean Public General Chat SFT:** Optional slice of `HuggingFaceTB/smol-smoltalk` (Apache-2.0) for natural instruction following and concise answers when no tool is needed.
3. **Deterministic Python Synthetic Generators (`src/coregrad/synthetic/` — fully open & reproducible):**
   - **Calculator & Multi-Step Reasoning (~100k trajectories):** Arithmetic, unit conversions, percentages, date differences, and 2–3 step chained word problems offloaded to `calc(...)`.
   - **Fictional-World Multi-Hop Retrieval (~100k trajectories):** Procedurally generated mini-libraries with *fictional* entities/dates/facts (impossible to memorize!), paired with 1-hop and 2-hop questions requiring `search` → `read` → grounded `<|cite|>` synthesis.
   - **Negative, Retry & Abstention (~40k trajectories):** Queries where the first search returns noisy hits (teaching query reformulation) or where the document store genuinely lacks the answer (teaching calibrated abstention instead of hallucination).
   - **Public Open-Textbook Grounded QA (~30k trajectories):** Passage-grounded QA & summarization traces built from public OpenStax / Wikipedia Markdown articles (with a strictly held-out subset reserved for evaluation).

#### 3C. Three-Stage Training Recipe
1. **Stage 2A — Cognitive Mid-Training & Context Extension (Annealing from Gate 1 stable checkpoint, ~1.5B tokens, ~6–8 hrs):**
   - Extend working context **`2048 → 4096`** via RoPE base scaling (`theta: 10000 → 100000`) with activation checkpointing (~4.3–5.5 GB VRAM on 4070 Ti) to double room for multi-passage tool outputs.
   - Mix 50% clean pretraining replay (prevents forgetting) + 25% synthetic reasoning/retrieval/calc trajectories + 15% normalized public tool/instruction traces + 10% code/math; decay LR cleanly.
2. **Stage 2B — Supervised Fine-Tuning (SFT, ~50M–150M tokens, ~1–2 hrs):**
   - Loss computed **only** on assistant tokens (`<|think|>`, `<|call|>`, final response); `<|user|>` and `<|result|>` tokens masked.
3. **Stage 2C — Verifiable RL Polish (GRPO / RLVR, ~2–6 hrs, high-leverage stretch):**
   - Online rollouts on deterministic synthetic environments (calculator + fictional multi-hop retrieval + abstention).
   - Purely programmatic rewards: `+0.2` valid tool syntax & execution, `+0.5` exact answer match, `+0.2` grounded citation matching retrieved `doc_id`, `-0.5` hallucinated citation or fabricated tool output.

#### 3D. Local Markdown Research CLI (`src/coregrad/app/`)
- Standalone **EmbeddingGemma 2 (text-only, MRL) + SQLite FTS5 BM25** indexer for **any user-specified folder of Markdown files**, with compact ~250-token chunking (or peak-relevance excerpt extraction for externally indexed longer chunks).
- Tested publicly on a reproducible open Markdown demo library (e.g., OpenStax chapters / public Wikipedia articles).
- Works immediately on any personal library of Markdown books and notes at runtime — **without ever training on private data**.
- Exposes a clean retriever/backend interface so external vector-DB indexes or research-agent harnesses can integrate via a thin adapter (or an OpenAI-compatible/subprocess agent endpoint).

#### Gate 2 Pass Criteria
- [x] **Tool Syntax:** ≥ 95% valid, parseable tool calls on held-out test prompts (**100.0% (377/377)** on Gate 2 suite; **100.0% (186/186)** on held-out GSM8K).
- [x] **Routing & Abstention:** ≥ 85% accuracy deciding when to call a tool vs. answer directly vs. abstain when evidence is absent (**100.0% (30/30)** Gate 2 abstention; **2/2** clean OOD no-tool chat routing after SFT v2 system-prompt alignment; 16% on adversarial SQuAD v2 traps — target for GRPO).
- [x] **Offloaded Reasoning:** ≥ 75% exact accuracy on held-out 1-to-3-step `calc` reasoning problems (**100.0% (100/100)** on locked Gate 2 calc suite; vs. 0% with tools disabled).
- [x] **Grounded Retrieval:** ≥ 70% factually accurate, properly cited answers on held-out 1-hop & 2-hop synthetic retrieval tasks (**94.4% (85/90)** accuracy, **96.7%** citation rate, **0** hallucinated citations vs. 0% without tools; plus **80.0%** target-doc citation / **28.0%** exact span match on 100 unseen SQuAD v2 Wikipedia tasks).
- [ ] **End-to-End CLI Demo:** Interactive local research session answering multi-passage questions with verified citations (CLI & hybrid `EmbeddingGemma 2` + FTS5 BM25 indexer implemented; ready for live demo & GRPO polish).

---

### Phase 3.5: Core Study Template, Provenance & Study Guide (`docs/`)
**Objective:** Turn `coregrad` into a self-contained, fully cited open study template so anyone with a single consumer GPU can understand, replicate, and extend Gates 1–2.

- [x] `docs/DATASETS.md` — Complete dataset & reference-model registry (repo IDs, pinned revisions, licenses, citations, pretraining/SFT mixtures, exact token & dedup counts).
- [x] `docs/REFERENCES.md` — Full structured bibliography of prior research used across architecture, optimization, tokenization, data curation, tool-use, verifiable RL, retrieval, and Phase 4 directions.
- [x] `docs/STUDY_GUIDE.md` — Chapter-by-chapter technical study guide mapped directly to `src/coregrad/`, grounded in our measured ablations (16k vs. 32k fertility, Muon vs. AdamW, `12M → 45M → 125M` scaling, SFT sequence packing, Gate 2 tools ON vs. OFF & OOD SQuAD v2 / GSM8K findings), with *"Further Study"* pointers and suggested reader exercises.

---

### Phase 4: Gate 3 — Essentializing the Cognitive Core (Weeks 7–10+)
**Objective:** Systematically strip down parameter count, architecture, and training budget on the locked Gate 1 + Gate 2 suite to map the **minimum viable cognitive core** Pareto frontier.

1. **Parameter Compression Ladder:** Dense cores at **125M → 60M → 30M → 15M → 8M** to locate the exact capability cliffs for tool syntax, routing, and grounded synthesis.
2. **Looped / Recursive Cognitive Cores:** Weight-shared depth (e.g., **4–6 unique layers looped 3–4×** with lightweight per-step norms/embeddings) — decoupling reasoning depth from parameter count, tested head-to-head against iso-parameter dense baselines on 2-hop retrieval and multi-step `calc`.
3. **Tokenizer & Embedding Minimization:** Compare 16k vs. 8k vs. 4k BPE and factorized embeddings (`d_embed < d_model`) where embeddings dominate tiny parameter budgets.
4. **Distillation & Data-Efficiency Ablations:** Distill logits/trajectories from the Gate 2 125M core into 15M–30M students; measure minimum token budgets needed to cross Gate 2 thresholds.
5. **Cognitive Core Purity Experiment:** Controlled A/B test at 30M–45M scale comparing pretraining *with* vs. *without* encyclopedic Wikipedia text — directly testing whether stripping world-fact memorization improves externalized tool-use and reasoning accuracy per parameter.

#### Gate 3 Pass Criteria
- [ ] Reproducible Pareto table & plots comparing ≥ 5 compressed/alternative core variants against the 125M reference.
- [ ] At least one **≤ 35M parameter** model achieving ≥ 85% of the Gate 2 125M reference score on tool-use & grounded retrieval.
- [ ] Documented empirical findings on looped vs. dense depth, vocab size, distillation, and cognitive-core purity.

---

### Phase 4B (Research Track): Latent Memory Bridge (Middle-Layer Knowledge Injection)
**Objective:** Go beyond text-only tool outputs by letting the model's middle layers directly query and cross-attend to an external vector knowledge base in latent space — creating a **"Peripheral (Latent) + Foveal (Tools)"** cognitive memory architecture without blowing up the context window.

1. **Zero-Initialized Gated Middle-Layer Cross-Attention (RETRO-style, Retrofittable):**
   - Bolt lightweight gated cross-attention blocks (`x = x + tanh(α) · CrossAttn(x, mem_K, mem_V)`, initialized at `α = 0` so pretrained Gate 1/2 behavior is preserved at step 0) onto middle/upper layers (e.g., layers 9, 12, 15 of `coregrad-125m`).
   - Pre-encode indexed knowledge passages into compact latent memory representations (starting from projected `google/embeddinggemma-2` 768d vectors for lightweight xRAG-style gist injection, scaling up to 16–32 pooled latent tokens per chunk for high-fidelity RETRO-style recall), stored in RAM for fast lookup.
2. **Learned Middle-Layer Latent Query Head:**
   - Attach a small MLP query projector `q = MLP_q(h_mid)` (`d_model=768 → 768`) at middle layer $L/2$ aligned to EmbeddingGemma 2's vector space (initialized via cosine/InfoNCE alignment and optionally refined end-to-end via differentiable top-$k$ softmax retrieval weights from next-token loss).
3. **Peripheral + Foveal Synergy:**
   - **Latent Memory Bridge (Peripheral):** Cheaply injects background awareness across 10–30 relevant passages at **zero context-window token cost**.
   - **Explicit Tools (`read`, `calc`) (Foveal):** Retained for exact verbatim dates, numbers, quotes, and verifiable citations where single-vector compression is lossy.
4. **Evaluation:** Measure held-out document BPB drop, context-token reduction, and factual QA accuracy head-to-head against pure text RAG on the locked Gate 2 benchmark suite.

---

### Phase 4.5: Research Findings & Essentialization Addendum (`docs/EXPERIMENTS.md`)
**Objective:** Publish the complete Gate 3 & Latent Memory Bridge empirical findings so the repository stands as an open mini-monograph.

- [ ] `docs/EXPERIMENTS.md` — Pareto frontier tables & plots (`125M → 60M → 30M → 15M → 8M`, looped vs. dense depth, vocabulary compression, distillation, cognitive-core purity A/B, and Latent Memory Bridge vs. pure text RAG).

---

### Phase 5 (Optional Future Extension): Private Library Integration
Once the open cognitive core works cleanly:
1. **Zero-Training Personal Research (Default — Recommended):** Simply run `coregrad index <local-markdown-dir>` and `coregrad chat` against any private books/notes. Because the cognitive core externalizes knowledge by design, the public weights never absorb private text and remain freely shareable.
2. **Optional Local Domain Adaptation (Opt-In):** Add a clean `optional_local_sources` hook in gitignored `configs/sources.local.yaml` allowing users to locally mix private Markdown books into pretraining or grounded-QA SFT (producing local-only private checkpoints kept strictly outside git).

---

## 5. Risks, Mitigations & Partial-Success Guarantees

| Risk | Likelihood | Mitigation | Graceful Fallback |
| :--- | :---: | :--- | :--- |
| **125M struggles with multi-hop (2+ step) tool chains** | Medium | Narrow 4-tool DSL, explicit `<|think|>` traces, fictional multi-hop synthetic generator, GRPO verifiable rewards | Even a rock-solid **1-hop search + read + calc** assistant is genuinely useful for local research |
| **Overfitting to synthetic templates** | Medium–High | Mix 3 diverse public tool datasets + general chat SFT, randomize synthetic phrasings, replay pretraining data, evaluate on unseen phrasings | Add lightweight local-teacher paraphrasing pass (e.g., open local model) to diversify synthetic questions |
| **Single-GPU wall-clock fatigue** | Medium | WSD schedule with 500M-token checkpoints; validate all ideas on 45M (~4 hrs) before 125M runs | 45M–60M model trained on 3B–4B tokens (~10–15 hrs) still achieves strong Gate 1 + basic Gate 2 |
| **12 GB VRAM OOM at 2048 context** | Low–Medium | BF16 + Flash SDPA + gradient accumulation + optional activation checkpointing; benchmark in Phase 0 | Train base at 1024 ctx, extend to 2048 during mid-training/annealing |

---

## 6. Immediate Next Steps

1. **Start Phase 0:** Scaffold the `uv` + PyTorch repo, create `configs/datasets.yaml`, `configs/sources.example.yaml`, gitignored `configs/sources.local.yaml`, and `scripts/check_hygiene.py`.
2. **Run `scripts/bench_gpu.py`:** Lock real tok/s and peak VRAM numbers on the RTX 4070 Ti before launching data packing.
3. **Start Phase 1:** Inventory the existing public dataset cache, download any missing public slices if needed (backed up to the configured HuggingFace cache root), train the public 16k BPE tokenizer, pack shards, and verify the eval harness.
