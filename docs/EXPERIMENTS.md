# Research Experiments & Empirical Findings

This document records controlled, falsifiable experiments conducted in **Phase 4** of `coregrad`. Every claim is backed by held-out metrics, explicit negative controls, and reproducible scripts under [`scripts/`](../scripts/).

---

## 1. Phase 4B — Latent Memory Bridge (Peripheral Latent Memory vs. Text Tools)

**Research Question:** Can a compact 125M-parameter cognitive core (`coregrad-125m`) query and read an external document store directly in **latent space** — bypassing text-token prompt prefill — and how do three distinct latent-memory mechanisms compare under a strict causal control?

**Evaluation Protocol (Locked, Held-Out SQuAD v2 Validation):**
- **Dataset:** 500 held-out answerable SQuAD v2 validation questions across 406–417 unique Wikipedia passages (strictly disjoint from the 8,000–20,000 training questions).
- **Mandatory Causal Control (`OFF` vs. `SHUFFLED` vs. `MATCHED`):**
  - `OFF`: Memory disabled (`memory=None`) — measures pure parametric baseline.
  - `SHUFFLED`: $k$ *wrong* passages injected into latent memory — exposes any content-blind format-adaptation shortcut.
  - `MATCHED`: Gold passage + $(k-1)$ random distractors injected into latent memory (**zero passage text tokens in the prompt**).
  - **Pass rule:** Genuine factual transfer requires `MATCHED` to substantially beat `SHUFFLED` ($\Delta_{\text{causal}} = \text{BPB}_{\text{MATCHED}} - \text{BPB}_{\text{SHUFFLED}} \ll 0$ and higher Exact Match / Token-F1), not merely beat `OFF`.

### 1.1 Mechanism 1: Learned Latent Query Head (Test 4 — **PASS**)

Can a frozen `coregrad-125m` generate dense retrieval queries directly from an internal hidden state $h_\ell$ — emitting **zero search tokens** — and retrieve the right passage from an `EmbeddingGemma 2` (768d) index?

We swept a lightweight 2-layer MLP (`LatentQueryHead` in [`memory_bridge.py`](../src/coregrad/model/memory_bridge.py), trained in **18.8s** via symmetric InfoNCE + cosine loss on 8,000 train questions) across 6 depths of frozen `coregrad_125m_base` ([`run_phase4b_query_head.py`](../scripts/run_phase4b_query_head.py)):

| Query Source | Search Tokens Generated | R@1 | R@3 | R@5 | R@10 | % of EG2 R@5 |
|---|---:|---:|---:|---:|---:|---:|
| **Upper bound:** `google/embeddinggemma-2` text query | N/A (external encoder) | 84.0% | 94.0% | 96.4% | 98.4% | 100.0% |
| **Sparse baseline:** BM25 text query | N/A (lexical) | 85.4% | 94.2% | 96.2% | 97.8% | 99.8% |
| Frozen `coregrad-125m` Layer 3 / 18 | **0** | 31.6% | 53.6% | 63.6% | 77.6% | 66.0% |
| Frozen `coregrad-125m` Layer 6 / 18 | **0** | 36.2% | 59.6% | 68.2% | 79.8% | 70.7% |
| Frozen `coregrad-125m` Layer 9 / 18 | **0** | 42.2% | 63.8% | 71.2% | 80.4% | 73.9% |
| Frozen `coregrad-125m` Layer 12 / 18 | **0** | 46.2% | 65.0% | 75.0% | 84.4% | 77.8% |
| Frozen `coregrad-125m` Layer 15 / 18 | **0** | 46.4% | 68.6% | 76.2% | 88.4% | 79.0% |
| **Frozen `coregrad-125m` Layer 18 / 18** | **0** | **49.6%** | **71.4%** | **79.8%** | **88.8%** | **82.8%** |

**Key Takeaway:** Semantic alignment with `EmbeddingGemma 2` emerges **monotonically all the way to layer 18/18**, reaching **82.8% of `EmbeddingGemma 2`'s R@5** (`88.8% R@10`) from a frozen 125M base model without generating a single search token.

---

### 1.2 Mechanism 2: Retrofitted Zero-Init Gated Cross-Attention (Tests 1–3 — **Falsified at Small Scale**)

We bolted zero-initialized `GatedCrossAttention` layers (`x = x + tanh(α) · CrossAttn(x, mem)`, initialized at `α = 0` for exact step-0 identity) onto layers `(9, 12, 15)` of `coregrad_125m_base` ([`run_phase4b_bridge.py`](../scripts/run_phase4b_bridge.py)) and tested **five configurations** spanning passage compression and base-model co-adaptation on 20,000 SQuAD v2 QA pairs (12,618 unique passages):

1. **Variant A (`embeddinggemma`, frozen):** 1 `EmbeddingGemma 2` (768d) vector per passage → 4 learned slots (10.62M trainable params, 1,200 steps).
2. **Variant B (`retro`, frozen):** 8 average-pooled latent tokens per passage from frozen bottom 6 layers (5.31M trainable params, 1,200 steps).
3. **Variant C1 (`retro_tokens`, frozen):** Full unpooled layer-6 hidden states (up to 160 tokens/passage with `memory_mask`, zero pooling loss, 5.31M trainable params, 1,600 steps).
4. **Variant C2 (`retro_tokens_coadapt`):** Unpooled layer-6 states + unfrozen upper blocks 9–17 (`lr = 5e-5`) + `20%` memory dropout so the decoder owns QA formatting (61.95M trainable params, 1,600 steps).
5. **Variant A2 (`embeddinggemma_coadapt`):** 1-vector `EmbeddingGemma 2` + unfrozen upper blocks 9–17 + `20%` memory dropout (67.26M trainable params, 1,600 steps).

| Configuration | Trainable Params | `OFF` BPB | `SHUFFLED k=4` BPB / EM / F1 | `MATCHED k=1` BPB / EM / F1 | `MATCHED k=4` BPB / EM / F1 | `MATCHED k=20` BPB / EM / F1 | Causal $\Delta_{\text{BPB}}$ (`k=1 − SHUF`) |
|---|---:|---:|---|---|---|---|---:|
| **Text-in-Prompt Upper Bound** (base, 0 training) | 0M | 1.8324 | — | **1.1222 / 20.8% / 9.7%** | — | — | — |
| Variant A: 1-vec EG2 (frozen) | 10.62M | 1.8324 | 1.2544 / 2.5% / 12.3% | 1.2548 / 2.5% / 11.9% | 1.2544 / 2.5% / 12.3% | 1.2549 / 2.5% / 11.9% | `+0.0004` |
| Variant B: 8-slot RETRO (frozen) | 5.31M | 1.8324 | 1.2689 / 2.5% / 10.5% | 1.2753 / 2.5% / 10.2% | 1.2694 / 2.5% / 10.5% | 1.2695 / 2.5% / 10.5% | `+0.0064` |
| Variant C1: Unpooled Layer-6 (frozen) | 5.31M | 1.8324 | 1.2483 / 2.5% / 11.7% | 1.2481 / 2.5% / 11.4% | 1.2485 / 2.5% / 11.3% | 1.2484 / 2.5% / 11.4% | `-0.0002` |
| Variant C2: Unpooled Layer-6 (co-adapt 9–17) | 61.95M | 1.2382 | 1.2312 / 2.5% / 12.0% | 1.2310 / 2.5% / 12.0% | 1.2311 / 2.5% / 12.2% | 1.2311 / 2.5% / 12.0% | `-0.0002` |
| Variant A2: 1-vec EG2 (co-adapt 9–17) | 67.26M | 1.2408 | 1.2285 / 2.5% / 11.5% | 1.2282 / 2.5% / 11.3% | 1.2283 / 2.5% / 11.8% | 1.2284 / 2.5% / 11.5% | `-0.0003` |

**Why the `SHUFFLED` Control Was Critical:**
- Looking only at `OFF → MATCHED` (`1.8324 → 1.2544 BPB`, F1 `5.3% → 12.3%`) would have falsely declared a huge win — when in reality `SHUFFLED` achieved the **exact same score** (`Δ_causal ≈ 0.000`). The zero-init cross-attention adapters learned a **content-blind QA-format bias**, not factual retrieval.
- **Root Cause (Mechanistic Bottleneck):** Even after removing pooling completely (`retro_tokens`) and co-adapting layers 9–17 (`retro_tokens_coadapt`), randomly initialized cross-attention has **no pretrained induction/copy circuits**. On a small supervised QA budget (~100k answer tokens — 60,000× smaller than pretraining), `Q/K/V/O` cross-attention starts with uniform attention over 160–640 tokens and stalls at a bilinear saddle point, whereas RETRO trains cross-attention over billions of autoregressive continuation tokens.

---

### 1.3 Mechanism 3: Independent Precomputed GQA KV-Bank Injection (TurboRAG / CAG — **STRONG PASS, Zero Training**)

If the bottleneck in Mechanism 2 is the absence of pretrained induction/copy circuits in freshly initialized cross-attention, **what happens when we inject independently precomputed latent memory directly into `coregrad-125m`'s existing 6B-token self-attention heads?**

We implemented `encode_kv_passage` and `forward_with_kv_bank` ([`memory_bridge.py`](../src/coregrad/model/memory_bridge.py), [`run_phase4b_kv_memory.py`](../scripts/run_phase4b_kv_memory.py)):
- **Offline (once per indexed passage):** Run each passage *independently* (zero cross-passage attention) and cache its post-norm, **pre-RoPE** GQA `(K_raw, V)` tensors across all 18 layers. Thanks to `coregrad-125m`'s GQA (4 KV heads vs. 12 Q heads), each passage averages **2.93 MB in BF16** (~163 tokens; 3× smaller than standard MHA) and **406 passages encode in 3.5s** on an RTX 4070 Ti.
- **Online (query time, zero training & zero passage prefill):** Concatenate the $k$ retrieved passages' cached `(K_raw, V)` tensors (keeping `<|bos|>` only on the first passage), apply contiguous RoPE positions `[0, M)`, and run only the short question at positions `[M, M+Q)`.
- **Exact identity verified:** Unit test `test_kv_bank_matches_exact_full_text_prefill` confirms single-passage KV injection matches full joint text prefill to `atol=1e-5`.

Evaluated on the **exact same 500 held-out SQuAD v2 validation questions** with **zero additional training**:

#### A. `coregrad_125m_base` (Gate 1 Base Checkpoint)
| Condition | Answer NLL | Answer BPB | Exact Match | Token-F1 | Causal $\Delta_{\text{BPB}}$ (`MATCHED − SHUF`) |
|---|---:|---:|---:|---:|---:|
| `OFF` (no memory) | 4.1444 | 1.8288 | 1.7% | 6.4% | — |
| `kv_shuffled_k1` (1 wrong KV bank) | 4.3157 | 1.9044 | 0.8% | 5.4% | — |
| **`kv_matched_k1`** (1 gold KV bank) | **2.5857** | **1.1410** | **20.0%** | **12.5%** | **`−0.7634` (25× EM)** |
| `kv_shuffled_k4` (4 wrong KV banks) | 4.3124 | 1.9029 | 0.8% | 5.1% | — |
| **`kv_matched_k4`** (4 independent KV banks, ~650 tok) | **2.8383** | **1.2524** | **9.2%** | **8.3%** | **`−0.6505` (11.5× EM)** |
| `kv_shuffled_k8` (8 wrong KV banks) | 4.2995 | 1.8973 | 0.8% | 5.1% | — |
| **`kv_matched_k8`** (8 independent KV banks, ~1,300 tok) | **3.3038** | **1.4579** | **6.7%** | **6.2%** | **`−0.4394` (8.4× EM)** |
| *Reference: `joint_text_k4` (full text re-prefill)* | *2.7445* | *1.2111* | *15.8%* | *9.8%* | *upper bound* |

#### B. `coregrad_125m_sft_v2` (Gate 2 Tool-Use SFT Checkpoint)
| Condition | Answer NLL | Answer BPB | Exact Match | Token-F1 | Causal $\Delta_{\text{BPB}}$ (`MATCHED − SHUF`) |
|---|---:|---:|---:|---:|---:|
| `OFF` (no memory) | 4.6139 | 2.0360 | 3.3% | 8.3% | — |
| `kv_shuffled_k1` (1 wrong KV bank) | 4.5825 | 2.0221 | 1.7% | 5.8% | — |
| **`kv_matched_k1`** (1 gold KV bank) | **2.6001** | **1.1473** | **20.0%** | **13.9%** | **`−0.8748` (12× EM)** |
| `kv_shuffled_k4` (4 wrong KV banks) | 4.6146 | 2.0363 | 2.5% | 5.3% | — |
| **`kv_matched_k4`** (4 independent KV banks, ~650 tok) | **2.8384** | **1.2525** | **12.5%** | **9.6%** | **`−0.7838` (5× EM)** |
| `kv_shuffled_k8` (8 wrong KV banks) | 4.6175 | 2.0376 | 1.7% | 5.6% | — |
| **`kv_matched_k8`** (8 independent KV banks, ~1,300 tok) | **3.4325** | **1.5146** | **6.7%** | **8.6%** | **`−0.5230` (3.9× EM)** |
| *Reference: `joint_text_k4` (full text re-prefill)* | *2.7585* | *1.2173* | *14.2%* | *9.0%* | *upper bound* |

### 1.4 Phase 4B Architectural Verdict

Combining **Mechanism 1 (`LatentQueryHead` at Layer 18)** with **Mechanism 3 (Independent Precomputed GQA KV-Bank Injection)** gives a compact 125M model a working, falsifiably verified **Latent Memory Bridge** without 100B-token RETRO pretraining:
1. **Latent Querying works out-of-the-box:** Layer 18 hidden states reach **82.8% of `EmbeddingGemma 2`'s R@5** (`88.8% R@10`) with **zero generated search tokens** and 18.8s of adapter training.
2. **Retrofitted Cross-Attention fails at small fine-tuning budgets:** Without billions of continuation tokens to build cross-attention copy circuits, `GatedCrossAttention` collapses to format adaptation (`Δ_causal ≈ 0.000`).
3. **Precomputed GQA KV-Bank Injection succeeds with zero training:** Reusing `coregrad-125m`'s pretrained self-attention induction heads achieves **`Δ = −0.78 BPB`** at `k=4` on `sft_v2`, recovering **88% of full joint text-in-prompt EM (`12.5%` vs. `14.2%`) and beating joint Token-F1 (`9.6%` vs. `9.0%`)** with independently encoded passages and zero query-time passage prefill.
4. **Recommended System Architecture ("Peripheral + Foveal"):** Use **Latent Query Head + GQA KV-Bank Injection** for low-latency peripheral background context over cached library passages, and retain **explicit tools (`search`, `read`, `calc`)** whenever exact verifiable citations, multi-hop synthesis, or arithmetic are required.

---

## 2. Phase 4A — Gate 3: Essentializing the Cognitive Core *(Upcoming)*

Planned Pareto frontier ablations on the locked Gate 1 + Gate 2 suite:
- Parameter compression ladder (`125M → 45M → 25M → 12M`)
- Looped / recursive weight-shared Transformers vs. iso-parameter dense baselines
- Vocabulary & embedding compression (`16k → 8k → 4k`)
- Cognitive Core Purity A/B (pretraining with vs. without encyclopedic Wikipedia text)
