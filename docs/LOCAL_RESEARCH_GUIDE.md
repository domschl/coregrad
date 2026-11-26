# Local Research Guide: Integrating Your Own Markdown Libraries

Because `coregrad` is designed as a **cognitive core**, its weights do **not** need to be retrained on your private books, papers, or notes. You can keep the open-data model weights completely unchanged and mount any local folder (or multiple folders) of Markdown files at **inference time**.

---

## 1. Preparing Your Markdown Library

`coregrad`'s indexer recursively scans one or more directories for `*.md` files. Common research workflows include:
- **Ebook & PDF libraries:** Export books/articles (e.g., from Calibre, Zotero, or PDF archives) to clean Markdown via Pandoc, Marker, or OCR pipelines, placing one `.md` file per book or paper.
- **Structured summaries:** If you maintain generated or hand-written English summaries alongside full-text books, you can pass **both** directories to the indexer (`books/ summaries/`).
- **Personal notes & wikis:** Standard Markdown folders (Obsidian, Foam, Logseq, plain notes) work directly; `# H1` headers are used as document titles (falling back to the filename).

---

## 2. Handling Multilingual Libraries (Important for English-Only Cores)

There is an important asymmetry between the **retriever** and the **cognitive core generator**:

- **Retriever (`google/embeddinggemma-2`):** Supports **100+ languages natively**, so cross-lingual semantic retrieval works out of the box.
- **Generator (`coregrad-125m` v1/v2):** Pretrained and SFT-tuned **English-only** (16,384-token English+code BPE). If `search()` returns a raw German, French, or Spanish passage, an English-only 125M core will fragment the tokens (~2× longer) and struggle to synthesize accurate answers.

### Recommended Strategies for Mixed-Language Libraries
1. **English full texts + English summaries of non-English works (Recommended):**
   Index your mixed-language full-text directory with `--english-only` (which uses a fast stopword-density filter to skip non-English full texts) **alongside** a directory of English summaries covering your non-English books and articles:
   ```bash
   uv run python scripts/research_cli.py index \
     /path/to/MarkdownBooks /path/to/EnglishSummaries \
     --out data/index/my_library \
     --english-only \
     --use-embeddinggemma2 --mrl-dim 256
   ```
2. **Pure English libraries:** Omit `--english-only` if your library is already English.
3. **Future multilingual cores:** If you want native multi-language reading without English summaries, train a bilingual/multilingual tokenizer and pretraining mix (e.g., adding `HuggingFaceFW/fineweb-2`) — keeping in mind the capacity tradeoff at ~125M parameters.

---

## 3. Indexing & Embedding Options

First install the optional retrieval dependencies (`transformers`, `sentence-transformers`):

```bash
uv sync --extra dev --extra retrieval
```

### Option A: Hybrid `EmbeddingGemma 2` + SQLite FTS5 BM25 (Recommended)
Combines dense semantic vectors from [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) (Apache-2.0, Matryoshka truncation to `256d` or `768d`) with SQLite FTS5 BM25 (`porter unicode61` stemming + stopword filtering), fused via Reciprocal Rank Fusion (RRF):

```bash
uv run python scripts/research_cli.py index /path/to/markdown_dir \
  --out data/index/my_library \
  --use-embeddinggemma2 \
  --mrl-dim 256 \
  --chunk-chars 1000
```

- **Why `--chunk-chars 1000`?** Compact ~250-token chunks mean `search(query, k=3)` uses ~750 tokens, fitting comfortably inside a 2,048-token context window alongside `<|think|>` traces and follow-up `read()` / `calc()` calls.
- **Why `--mrl-dim 256`?** Matryoshka truncation cuts vector storage and dot-product latency by 3× vs. 768d with minimal retrieval quality loss.

### Option B: Zero-Download BM25-Only Mode
If you want instant indexing with zero embedding model downloads (or no GPU memory used by the embedder), omit `--use-embeddinggemma2`:

```bash
uv run python scripts/research_cli.py index /path/to/markdown_dir --out data/index/my_library
```

---

## 4. Querying Your Library

### Interactive Research Session
```bash
uv run python scripts/research_cli.py chat \
  --checkpoint runs/coregrad_125m_grpo_v2/latest.pt \
  --index data/index/my_library \
  --use-embeddinggemma2 --mrl-dim 256
```

### Single Non-Interactive Query
```bash
uv run python scripts/research_cli.py chat \
  --checkpoint runs/coregrad_125m_grpo_v2/latest.pt \
  --index data/index/my_library \
  --use-embeddinggemma2 --mrl-dim 256 \
  --ask "What are the main causes discussed across the indexed articles?"
```

---

## 5. Reproducible Public Demo (Try It Without Private Data)

To verify the full hybrid indexing + CLI pipeline strictly on open-licensed data:

```bash
# 1. Export 35 held-out public Wikipedia articles from SQuAD v2 validation as Markdown
uv run python scripts/build_public_demo_library.py

# 2. Index with EmbeddingGemma 2 + SQLite FTS5 BM25
uv run python scripts/research_cli.py index data/demo_library \
  --out data/index/demo_hybrid --use-embeddinggemma2 --mrl-dim 256

# 3. Query interactively or via --ask
uv run python scripts/research_cli.py chat \
  --checkpoint runs/coregrad_125m_grpo_v2/latest.pt \
  --index data/index/demo_hybrid --use-embeddinggemma2 --mrl-dim 256
```

---

## 6. Bridging an External Pre-Built Vector Database

If you already maintain an external vector database or personal wiki engine (for example, a system indexing long 3,000-character book chunks), you do **not** need to duplicate your index. Implement `coregrad`'s 3-method [`Retriever`](../src/coregrad/tools/retriever.py) protocol (`search`, `read`, `list_docs`) and use `extract_peak_excerpt(chunk_text, query, max_chars=900)` to trim long external chunks around the highest-relevance span before returning them to `coregrad`'s 2,048-token context window.

---

## 7. Privacy Guarantee

- All indexes under `data/index/` and local configs (`configs/sources.local.yaml`) are **gitignored** and verified by `scripts/check_hygiene.py`.
- Pointing `research_cli.py` at private Markdown folders never modifies model weights, keeping your `coregrad` checkpoints 100% open and shareable.
