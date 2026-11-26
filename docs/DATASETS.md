# Datasets, Licenses & Provenance

Phases 0–4 of `coregrad` are built **100% from publicly available, openly licensed datasets** plus deterministic Python-generated synthetic environments. Every external source is pinned by repository ID and commit revision in [`configs/datasets.yaml`](../configs/datasets.yaml) so the entire training corpus, tokenizer, and benchmark suite can be reproduced without restriction.

---

## 1. Pretraining Corpora (Gate 1 — 6.70B Unique Tokens)

Rather than spending a 125M model's limited capacity memorizing raw encyclopedic trivia, the pretraining mixture prioritizes **high-signal educational prose, structured textbook explanations, clean source code, and mathematical reasoning**.

| Domain Key | Hugging Face Dataset | Pinned Revision | License | Mixture Weight | Packed Train Tokens | Packed Val Tokens | Bytes/Tok (16k BPE) | Exact Dups Dropped |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `edu-en` | [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample/10BT`, shards `000`–`003`) | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | **ODC-By 1.0** | 45% | 3,200.4M | 4.14M | 4.131 | 32,305 |
| `cosmo` | [`HuggingFaceTB/cosmopedia`](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) (OpenStax, Khan Academy, Stanford, stories, math) | `0ae6ec63f91742bd2d1eaef4f02232c55d719385` | **Apache-2.0** | 25% | 1,300.2M | 4.24M | 4.778 | 2 |
| `code-starcoder` | [`bigcode/starcoderdata`](https://huggingface.co/datasets/bigcode/starcoderdata) (Python, C, C++, Rust, Go, JS, TS, Shell, JSON, YAML, Jupyter) | `9fc30b578cedaec69e47302df72cf00feed7c8c4` | **Permissive per-file** *(The Stack ToU)* | 17% | 1,200.2M | 4.09M | 2.893 | 48 |
| `finemath` | [`HuggingFaceTB/finemath`](https://huggingface.co/datasets/HuggingFaceTB/finemath) | `e92b25a616738fe95dc186b64dfb19f9c8525594` | **ODC-By 1.0** | 13% | 1,000.3M | 4.01M | 3.220 | 0 |
| **Total Packed** | — | — | **Open / Permissive** | **100%** | **6,701.1M** | **16.48M** | — | **32,355** |

*Additional compatible public code dataset declared in manifest (optional):* [`codeparrot/github-code-clean`](https://huggingface.co/datasets/codeparrot/github-code-clean) (`c48d40f9e70f0196f8236901ee35807f7d6c44c0`, filtered to permissive licenses).

### Processing & Zero-Leakage Guarantee
1. **Cleaning (`src/coregrad/data/sources.py`):** Unicode NFC normalization, newline standardization, control-character stripping, minimum/maximum document length filtering (`64 .. 200,000` chars).
2. **File Interleaving:** Multi-topic/multi-language Parquet directories (`cosmopedia`, `starcoderdata`) are interleaved round-robin across files so capped token budgets remain evenly balanced across languages and subjects.
3. **Deterministic Split & Deduplication (`src/coregrad/data/pack.py`):** Every normalized document is hashed with 64-bit `blake2b`. Exact duplicates are dropped across the entire stream, and validation documents (`val_fraction = 0.005`) are assigned deterministically by hash before packing — making train/validation leakage impossible.
4. **Public-Only Tokenizer (`scripts/train_tokenizer.py`):** The 16,384-token byte-level BPE tokenizer is trained on a 160 MB balanced sample (35,545 documents) strictly from these four public datasets.

---

## 2. Post-Training, Tool-Use & Synthetic Corpora (Gate 2 — 155,007 Trajectories)

All Phase 3 post-training trajectories are normalized into a single compact cognitive-core special-token protocol (`<|system|>`, `<|user|>`, `<|think|>`, `<|call|>`, `<|result|>`, `<|assistant|>`, `<|cite|>`, `<|eos|>`).

### 2A. Public Tool-Calling & Chat Datasets

| Key | Hugging Face Dataset | Pinned Revision | License | Normalized Trajectories Used (SFT v2) | Role |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `rag-squad-v2` | [`rajpurkar/squad_v2`](https://huggingface.co/datasets/rajpurkar/squad_v2) (`train`) | `3ffb306f725f7f2ce8431b538a0b28e504731317` | **CC-BY-SA-4.0** | 55,000 | Real open-domain reading-comprehension RAG + unanswerable abstention |
| `math-gsm8k` | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) (`main/train`) | `740312add88f50c849f834f9b763858a74596f54` | **MIT** | 6,925 (×2 = 13,850) | Human-written multi-step math word problems with verified `calc(expr)` calls |
| `sft-smoltalk` | [`HuggingFaceTB/smol-smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) | `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc` | **Apache-2.0** | 45,000 | Clean general English chat & no-tool routing (aligned `DEFAULT_SYSTEM_PROMPT`) |
| `tool-xlam` | [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | `26d14ebfe18b1f7b524bd39b404b50af5dc97866` | **CC-BY-4.0** | 5,000 | Clean 1–2 call function argument binding |
| `tool-hermes` | [`NousResearch/hermes-function-calling-v1`](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | `dae3e1d28cfbcf4b915c04ea1e072030529b4bda` | **Apache-2.0** | 5,000 | Multi-turn tool invocation & response synthesis |
| `tool-glaive` | [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | `e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac` | **Apache-2.0** | 5,000 | Tool routing & "no tool required" refusal/chat turns |

### 2B. Deterministic Programmatic Synthetic Generators (Open Code in `src/coregrad/synthetic/`)

| Generator | Module | Train Trajectories | Locked Eval Tasks | Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **Multi-Step Calculator** | [`calc_gen.py`](../src/coregrad/synthetic/calc_gen.py) | 40,000 | 100 | 1-to-3-step year spans, percentages, unit conversions, orders+tax, geometry (`sqrt`), compound interest — verified verbatim via `safe_calc()` |
| **Fictional Multi-Hop Retrieval & Abstention** | [`retrieval_gen.py`](../src/coregrad/synthetic/retrieval_gen.py) | 40,000 | 120 | Procedurally generated mini-libraries with **strictly disjoint train/eval entity vocabularies** covering 1-hop QA, 2-hop bridge QA, retrieval+`calc`, and calibrated abstention |

- **Final SFT Split (`data/phase3/sft/`):** `153,007` train / `2,000` val trajectories (~79.6M real tokens; median 267 tok, mean 520 tok, p95 1,182 tok).
- **Mid-Training Shards (`data/phase3/midtrain/`):** `73.76M` deduplicated tokens + `0.73M` val tokens.

---

## 3. External Open Reference Models

| Role | Hugging Face Model | License | Usage in `coregrad` |
| :--- | :--- | :--- | :--- |
| **Hybrid Retriever Embedder** | [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) | **Apache-2.0** | Text-only backbone (~270M params) with Matryoshka truncation (`256d`/`768d`), fused with SQLite FTS5 BM25 via RRF |
| **Scale Calibration Reference** | [`HuggingFaceTB/SmolLM2-360M`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) | **Apache-2.0** | Optional external reference baseline for small-LM capability expectations |

---

## 4. Dataset Attribution & Citations

- **FineWeb-Edu:** Penedo, G., Kydlíček, H., Ben Allal, L., Lozhkov, A., Mitchell, M., Raffel, C., von Werra, L., & Wolf, T. (2024). *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* NeurIPS 2024 Datasets & Benchmarks (`arXiv:2406.17557`).
- **Cosmopedia & SmolTalk / SmolLM2:** Ben Allal, L., Lozhkov, A., Bakouch, E., Blázquez, G. M., Penedo, G., Tunstall, L., Marafioti, A., Kydlíček, H., Lajarín, A. P., Srivastav, V., Lochner, J., Fahlgren, C., Nguyen, X.-S., Fourrier, C., Burtenshaw, B., Larcher, H., Zhao, H., Zakka, C., Morlon, M., Raffel, C., von Werra, L., & Wolf, T. (2024–2025). *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model* (`arXiv:2502.02737`).
- **FineMath:** Lozhkov, A., Ben Allal, L., et al. (2024). *FineMath: Open Mathematical Pretraining Data*, HuggingFaceTB.
- **StarCoderData:** Li, R., Ben Allal, L., Zi, Y., Muennighoff, N., Kocetkov, D., et al. (2023). *StarCoder: May the Source Be With You!* (`arXiv:2305.06161`).
- **xLAM-Function-Calling-60k:** Liu, Z., Hoang, T., Zhang, J., Zhu, M., Lan, T., Kokane, S., Tan, J., Yao, W., Liu, Z., Feng, Y., Murthy, R., Yang, L., Savarese, S., Niebles, J. C., Wang, H., Heinecke, S., & Xiong, C. (2024). *APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets* (`arXiv:2406.18518`).
- **Hermes Function Calling v1:** Teknium / Nous Research (2024). *Hermes Function Calling v1 Dataset*.
- **Glaive Function Calling v2:** Glaive AI (2024). *Glaive Function Calling v2 Dataset*.
- **SQuAD v2:** Rajpurkar, P., Jia, R., & Liang, P. (2018). *Know What You Don't Know: Unanswerable Questions for SQuAD.* ACL 2018 (`arXiv:1806.03822`).
- **GSM8K:** Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, Ł., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). *Training Verifiers to Solve Math Word Problems* (`arXiv:2110.14168`).
- **EmbeddingGemma 2:** Google DeepMind (2026). *EmbeddingGemma 2: Unified and Efficient Multimodal Representations* (`arXiv:2610.37364`).

---

## 5. Reproduction Commands

```bash
# Configure cache root (env var or gitignored configs/sources.local.yaml)
export COREGRAD_HF_CACHE=/path/to/hf-cache

# Verify cached datasets & sync recommended additions
uv run python scripts/check_sources.py
uv run python scripts/sync_datasets.py --all

# Train public 16k BPE tokenizer & pack 6.70B uint16 pretraining shards
uv run python scripts/train_tokenizer.py --sample-mb 160 --vocab-size 16384 --compare-32k
uv run python scripts/prepare_data.py

# Generate deterministic synthetic + normalized public Phase 3 datasets
uv run python scripts/prepare_phase3_data.py
```
