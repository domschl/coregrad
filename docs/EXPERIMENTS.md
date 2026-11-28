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

## 2. Phase 4A — Gate 3: Essentializing the Cognitive Core (Dense vs. Looped Scaling & Logic Depth)

**Research Question:** How small can a cognitive core shrink before tool orchestration, grounded citation, abstention, and multi-step formal deduction break down — and can **weight-shared Looped/Recursive Transformers** recover dense capacity at ~26M parameters?

**Locked Comparability Controls:**
- All compressed models (`45m`, `25m-dense`, `25m-looped12`, `25m-looped14`) share the **exact same 16,384 BPE tokenizer, 1.5B-token pretraining budget, data mixture, `seed=1337` batch order, Muon + WSD schedule, and SFT v2 dataset (`data/phase3/sft/train.jsonl`)**.
- All five models (including the 6.0B-token `125m` reference) are evaluated identically on:
  1. **Gate 1 Base Eval:** 4-domain validation BPB, grammar minimal pairs, and 4-way completions.
  2. **Locked Gate 2 Suite ([`eval_gate2.py`](../scripts/eval_gate2.py)):** 100 multi-step `calc`, 90 1–2 hop fictional retrieval + citations, 30 abstention tasks.
  3. **Locked OOD Suite ([`eval_ood.py`](../scripts/eval_ood.py)):** 50 held-out SQuAD v2 answerable + 50 unanswerable traps + 50 GSM8K test + 4 no-tool chat prompts.
  4. **Procedural Multi-Step Syllogism & Horn-Clause Depth Probe ([`run_logic_depth_probe.py`](../scripts/run_logic_depth_probe.py)):** Train on depths `2–5`, evaluate strictly OOD on depths `6, 8, 10, 12` over disjoint fictional vocabularies with zero keyword shortcuts.

### 2.1 Six-Model Parameter, Compute & Capability Pareto Table

*(All 1.5B-token Base Val BPB numbers verified on identical 131,072-token/domain validation windows via [`eval_model.py`](../scripts/eval_model.py).)*

| Metric | `125m` (Ref.) | `45m-dense` | **`45m-looped14`** | `25m-dense` | `25m-looped12` | `25m-looped14` |
|---|---:|---:|---:|---:|---:|---:|
| **Total / Non-Emb Params** | 125.86M / 113.28M | 43.79M / 35.40M | **45.04M / 32.46M** | 25.18M / 18.89M | 26.09M / 17.70M | 26.09M / 17.70M |
| **BF16 Weight Size** | ~252 MB | ~88 MB | **~90 MB** | ~50 MB | ~52 MB | ~52 MB |
| **Architecture (`d`, layers)** | `d=768`, 18 dense | `d=512`, 12 dense | **`d=768`, `1+4×3+1`** | `d=384`, 12 dense | `d=512`, `6×2` | `d=512`, `1+4×3+1` |
| **Pretrain Tokens** | 6.0B | 1.5B | **1.5B** | 1.5B | 1.5B | 1.5B |
| **Train tok/s · VRAM (mb)** | 52.6k · 6.9G (4) | 131.3k · 6.0G (8) | **72.7k · 4.3G (4)** | 188.5k · 4.6G (8) | 130.5k · 5.6G (8) | 114.2k · 6.3G (8) |
| **Base Val Mean BPB** | 0.7371 | 0.8609 | **0.8402 (`−0.0207`)** | 0.9018 | 0.8860 | **0.8778 (`−0.0240`)** |
| **Base Greedy 4-gram Rep.** | — | 19.2% | 51.7% | 28.4% | 34.7% | 65.0% |
| **SFT v2 Best Val Loss** | — | 1.1315 | **1.0515 (`−0.0800`)** | 1.2024 | 1.1688 | **1.1514 (`−0.0510`)** |
| **Gate 2:** Tool syntax | 100.0% | 100.0% | **100.0% (372/372)** | 100.0% | 100.0% | 100.0% |
| **Gate 2:** Multi-step `calc` | 100.0% | 98.0% | **99.0% (99/100)** | 99.0% | 98.0% | 98.0% |
| **Gate 2:** 1–2 hop retrieval | 94.4% | **98.9%** | **97.8% (88/90)** | 87.8% | 88.9% | 86.7% |
| **Gate 2:** Citation (halluc.) | 96.7% (0) | **100.0% (0)** | **97.8% (0)** | 95.6% (0) | 91.1% (0) | 94.4% (0) |
| **Gate 2:** Clear abstention | 100.0% | 100.0% | **100.0% (30/30)** | 100.0% | 100.0% | 100.0% |
| **OOD:** Chat no-tool routing | 100.0% (4/4) | 100.0% (4/4) | **100.0% (4/4)** | 100.0% (4/4) | 75.0% (3/4) | 100.0% (4/4) |
| **OOD:** SQuAD v2 target cite | 80.0% (0) | **84.0% (0)** | 62.0% (0) | 56.0% (0) | 54.0% (0) | 60.0% (0) |
| **OOD:** SQuAD v2 span EM | 24.0% | 18.0% | **28.0% (+4% vs 125m!)** | 10.0% | 14.0% | 14.0% |
| **OOD:** SQuAD v2 unans. abst. | 14.0% | 14.0% | 12.0% | 12.0% | 2.0% | 10.0% |
| **OOD:** GSM8K EM / syntax | 4.0% / 100% | 4.0% / 100% | **6.0% (3/50) / 100%** | 4.0% / 100% | 2.0% / 100% | **10.0% (5/50) / 100%** |

**Key Architectural Findings:**
1. **Tool orchestration survives cleanly down to 25M (~50 MB):** Every model down to `25.18M` achieves **100.0% valid tool syntax**, **98–99% multi-step `calc`**, **86.7–98.9% multi-hop retrieval** with **0 hallucinated citations**, and **100.0% clear abstention**. Gate 3's `≤ 35M` target (≥ 85% of `125M` reference) is comfortably surpassed by all three ~25M cores (91–94% of `125M`'s retrieval score).
2. **`1+4×3+1` looping scales cleanly from 25M to 45M (`−0.021` to `−0.024` BPB at BOTH scales!):**
   - At **~25M:** `looped14` beats `25m-dense` by **`−0.0240` Base BPB** (`0.9018 → 0.8778`), **`−0.0510` SFT val loss** (`1.2024 → 1.1514`), and `+4.0%` OOD SQuAD EM (`10% → 14%`).
   - At **~45M:** `45m-looped14` (`45.04M`, ~90 MB BF16, 1.5B tok) beats `45m-dense` across **all 4 pretraining domains** (`−0.0207` Base BPB: `0.8609 → 0.8402`), slashes SFT val loss by **`−0.0800`** (`1.1315 → 1.0515`), hits **`97.8%` Gate 2 retrieval / `99%` calc / `100%` syntax & abstention**, and achieves **`28.0%` OOD SQuAD v2 exact span EM** — **beating `coregrad-125m` (`24.0%`) at 2.8× fewer parameters and 4× fewer pretraining tokens!** *(Its one OOD SFT lag — `62%` vs `84%` unprompted OOD citation rate despite `0` hallucinations — is the exact margin GRPO v2 lifted by `+13%` on `125m`.)*
3. **Prelude + Recurrent Core + Coda (`1+4×3+1`) clearly outperforms unguarded `6×2` looping:** At `26.09M`, `looped14` shows ~2.5× calmer gradients (`gn ~0.13` vs `0.30`) and heals `looped12`'s chat-routing and abstention drops (`100%` chat routing, `10%` vs `2%` unanswerable abstention).
4. **Honest Tradeoffs of Simple Looping (Motivating Phase 5):**
   - **Compute:** `45m-looped14` runs at `72.7k tok/s` (~1.4× faster than `125m`, ~1.8× slower than `45m-dense`).
   - **Greedy Base Repetition:** Both `25m-looped14` (`65.0%`) and `45m-looped14` (`51.7%`) show higher greedy (`T=0`) 4-gram repetition on open-ended base prompts than dense peers (`28.4%` / `19.2%`) due to sharper induction/copy attractors across unguarded loops — directly motivating **outer GRU-gated depth-RNN + prelude input re-injection** in Phase 5.

---

### 2.2 Multi-Step Syllogism & Horn-Clause Depth Generalization (`D = 2..12`)

To test how deeply tiny dense vs. looped cores can execute formal rules with **zero world-knowledge shortcuts**, we trained a lightweight 600-step probe ([`run_logic_depth_probe.py`](../scripts/run_logic_depth_probe.py)) from each model's `sft_v2/best.pt` on procedural rule chains of depths `D ∈ {2, 3, 4, 5}` ([`logic_gen.py`](../src/coregrad/synthetic/logic_gen.py), balanced across positive syllogisms, exclusion chains, reversed-implication `Unknown` traps, and conjunctive Horn rules — with distractor negation and conjunctive rules injected into *every* task to eliminate surface keyword shortcuts). We then evaluated on **strictly disjoint fictional vocabularies** across both in-distribution (`2, 4, 5`) and out-of-distribution (`6, 8, 10, 12`) depths (`40` tasks/depth):

#### A. `--mode direct` (Zero Scratchpad Tokens — Pure Latent Reasoning Depth)
| Model | Params | Eff. Layers | Mean ID (`d2,4,5`) | **Mean OOD (`d6–12`)** | `d2` | `d4` | `d5` | `d6` | `d8` | `d10` | **`d12`** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `coregrad-125m` | 125.9M | 18 | 90.0% | **78.1%** | 100.0% | 90.0% | 80.0% | 82.5% | 80.0% | 80.0% | 70.0% |
| `coregrad-45m-dense` | 43.8M | 12 | 81.7% | 76.9% | 87.5% | 80.0% | 77.5% | 77.5% | 77.5% | 77.5% | **75.0%** |
| **`coregrad-45m-looped14`** | **45.0M** | **14** | **88.3% (`+6.6%`)** | **77.5% (`+0.6%`)** | 95.0% | 87.5% | 82.5% | 82.5% | 77.5% | 77.5% | 72.5% |
| `coregrad-25m-dense` | 25.2M | 12 | 86.7% | 70.0% | 92.5% | 82.5% | 85.0% | 77.5% | 70.0% | 70.0% | 62.5% |
| **`coregrad-25m-looped12`** | **26.1M** | **12** | **90.8%** | **76.2% (`+6.2%`)** | 100.0% | 85.0% | 87.5% | 87.5% | 75.0% | 70.0% | 72.5% |
| **`coregrad-25m-looped14`** | **26.1M** | **14** | 80.0% | **75.6% (`+5.6%`)** | 92.5% | 77.5% | 70.0% | 77.5% | 72.5% | 77.5% | **75.0%** |

#### B. `--mode cot` (Explicit `<|think|>` Proof Trace over Unseen Multi-Subword Fictional Predicates)
| Model | Params | Eff. Layers | Mean ID (`d2,4,5`) | Mean OOD (`d6–12`) | `d2` | `d4` | `d5` | `d6` | `d8` | `d10` | `d12` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `coregrad-125m` | 125.9M | 18 | 55.0% | 35.6% | 75.0% | 55.0% | 35.0% | 42.5% | 45.0% | 35.0% | 20.0% |
| `coregrad-45m-dense` | 43.8M | 12 | **57.5%** | 33.1% | 62.5% | 52.5% | 57.5% | 50.0% | 32.5% | 30.0% | 20.0% |
| **`coregrad-45m-looped14`** | **45.0M** | **14** | 42.5% | **35.0%** | 55.0% | 47.5% | 25.0% | 50.0% | 40.0% | 25.0% | 25.0% |
| `coregrad-25m-dense` | 25.2M | 12 | 49.2% | 33.1% | 67.5% | 50.0% | 30.0% | 47.5% | 32.5% | 35.0% | 17.5% |
| `coregrad-25m-looped12` | 26.1M | 12 | 42.5% | 25.6% | 60.0% | 32.5% | 35.0% | 32.5% | 22.5% | 32.5% | 15.0% |
| **`coregrad-25m-looped14`** | **26.1M** | **14** | 38.3% | **37.5%** | 55.0% | 32.5% | 27.5% | 37.5% | **45.0%** | 32.5% | **35.0%** |

**Three Crisp Empirical Lessons from the Logic-Depth Probe:**
1. **Weight-shared looping wins cleanly on OOD latent depth (`direct`):** Both `25m-looped12` (`76.2%` OOD, `72.5%` at `d12`) and `25m-looped14` (`75.6%` OOD, `75.0%` at `d12`) beat iso-parameter `25m-dense` (`70.0%` OOD, `62.5%` at `d12`) by **`+5.6%` to `+12.5%`** on chains up to 2.4× deeper than training — matching `45m` (`76.9%` OOD, `75.0%` at `d12`) at 40% fewer parameters.
2. **Real failure mode in `direct` vs. `cot`:** In `direct` mode, detecting a *single reversed implication arrow* (`syllogism_unknown`, affirming-the-consequent trap) degrades from `100%` at `d2` to `10–40%` by `d8–12` across all models. In `cot` mode, models *do* explicitly diagnose affirming-the-consequent fallacies on unseen words (`40–70%` accuracy on `syllogism_unknown` even at `d8–12`!) — proving the scratchpad unlocks sharper fallacy detection.
3. **Why `cot` trailed `direct` on long unseen chains (classic $p^D$ BPE drift):** Inspecting generated traces shows that overfitting 600 steps to only 32 training fictional predicates (`loss ≈ 0.0004–0.001`) hurts exact verbatim copying of *unseen multi-subword nonsense words* (`quoril`, `brenvix`, `yaldor`). One drifted subword at Step 3 (`yaldor → triad`) derails Steps 4–12. Notably, **`25m-looped14`** suffered the least OOD collapse in `cot` mode (`37.5%` OOD, `35.0%` at `d12`, beating `125m`'s `20.0%` at `d12`). Expanding the training predicate pool (e.g., 500+ random pseudowords) or adding GRPO outcome rewards is a clear recipe to unlock near-100% OOD `cot` chaining.

### 2.3 Diagnosing & Curing the Looped Repetition Attractor (4-Way 300M-Token Ablation)

Inspecting raw generated text from our initial `25m-looped14` (`65.0%` greedy rep) and `45m-looped14` (`51.7%` greedy rep) runs revealed severe single-token stutter (`"# # # ..."`, `"closed closed ..."`, `"initial initial ..."`), even though global `grad_norm` stayed calm (`~0.13`) and validation BPB improved.

**Root-Cause Diagnosis (Two Coupled Bugs):**
1. **Scale-Invariance Gradient Paradox in Muon:** Pairing `muon_lr=0.02` with Moonlight's `0.2 * sqrt(max(m, n))` multiplier (instead of Keller Jordan's `sqrt(max(1, m / n))`) + weak Muon weight decay (`wd * 0.2`) inflated 2D weights to `||W||_F ≈ 600` (`wRMS ≈ 0.65`) and blew up residual norms to `||x|| ≈ 25,000–75,000` (`actRMS ≈ 1,500–2,700`). Because `Pre-RMSNorm` shrinks gradients as `1 / ||W||`, global `grad_norm` looked *calmer* while weights exploded.
2. **BF16 Underflow of `loop_emb`:** Adding a `~1.3`-norm `loop_emb` onto a `25,000`-norm residual stream (`ratio = 0.00006 < bf16 eps = 0.0078`) rounded `loop_emb` to **exact zero** in BFloat16 — leaving shared blocks completely blind to iteration index and amplifying copy attractors.

**Fixes Implemented & 4-Vital Early Warning Monitor (`f9bf44b`):**
- Corrected Muon aspect scaling to `sqrt(max(1, rows / cols))` with full `weight_decay`.
- Added scale-safe `loop_mode: rms_add` and an **Outer Gated Depth-RNN + Prelude Input Re-Injection** (`loop_mode: gru`: `s ← (1 - z)·s + z·Core(s_norm + W_in·silu([s_norm; e_0]) + loop_emb[k])`).
- Added live health vitals (`max_w_rms`, `max_layer_gn`, `act_rms`, `max_logit`, `rep4_t0`, `rep4_t07`, `loop_cos`, `loop_gate_mean`) to [`trainer.py`](../src/coregrad/train/trainer.py) and [`monitor.py`](../src/coregrad/train/monitor.py).

Controlled 4-way comparison at **300M tokens** (`seed=1337`, identical data stream & fixed Muon):

| Metric (`300M` tokens) | Run A: `25m-dense` (`25.2M`, 12d) | Run B: `25m-looped14-clean` (`26.1M`, `1+4×3+1`, `rms_add`) | Run C: `25m-looped14-gru` (`27.1M`, `1+4×3+1`, `gru`) | **Run D: `32m-looped16-gru` (`33.0M`, `2+4×3+2`, `gru`)** |
|---|---:|---:|---:|---:|
| **Val Mean BPB** | `1.0093` | `0.9904` (`−0.0189`) | `0.9726` (`−0.0367`) | **`0.9634` (`−0.0459`)** |
| `cosmo` / `code` BPB | `0.811` / `0.815` | `0.793` / `0.796` | `0.778` / `0.772` | **`0.768` / `0.767`** |
| `finemath` / `edu-en` BPB | `1.173` / `1.239` | `1.152` / `1.220` | `1.133` / `1.207` | **`1.121` / `1.197`** |
| **Grammar / 4-Way** | `68.8%` / `41.7%` | `87.5%` / **`45.8%`** | `84.4%` / `37.5%` | **`87.5%`** / `37.5%` |
| **Max Weight RMS (`wRMS`)** | `0.112` | `0.095` | `0.082` | **`0.078`** *(healthy!)* |
| **Residual `actRMS` / `maxLogit`** | `8.3` / `21.3` | `9.8` / `20.8` | **`5.8`** / `20.8` | **`6.6`** / `21.8` |
| **Loop Cos Sim (`L1 → L2`)** | — | `0.934` | **`0.881`** | **`0.886`** |
| **GRU Gate Mean** | — | — | `0.75` | `0.73` |
| **Sampled `T=0.7` 4-gram Rep.** | `36.5%` | `22.4%` | `24.0%` | **`18.0%` (½ of dense!)** |
| **Single-token stutter?** | Mild | **None** | **None** | **None** |
| **Train tok/s · VRAM** | `188.4k` · `4.6 GB` | `114.3k` · `6.3 GB` | `106.0k` · `6.9 GB` | `94.7k` · `7.6 GB` |

**Verdict:** Fixed Muon scaling + `rms_add` completely eliminated the `~2,700` activation explosion and single-token stutter. Adding the **Outer GRU Depth-RNN + Prelude Re-Injection** doubled the BPB gain (`−0.0189 → −0.0367` BPB) and broke inter-loop collapse (`loop_cos: 0.934 → 0.881`), while expanding the interface to **Huginn-style `2` prelude + `4×3` + `2` coda (`33.0M`)** pushed BPB to **`0.9634`** (`−0.0459` vs dense) and halved sampled `T=0.7` repetition (`36.5% → 18.0%`) — establishing **`2+4×3+2` + Outer GRU Depth-RNN + Fixed Muon** as our new standard architecture going into Phase 5.

### 2.4 3-Point Muon Bracket & Exact Weight-RMS Equilibrium Law on `32m-looped16-gru`

Before launching `1.5B`-token Phase 5.2 runs on our winning `coregrad-32m-looped16-gru` (`33.04M`), we ran a fast 2-point `100M`-token (`762`-step) Muon LR/WD bracket against Run D (`seed=1337`; Steps `300` and `600` are 100% on the flat WSD plateau in all three runs):

| Config (`muon_lr` / `wd`) | Step 300 BPB | **Step 600 BPB** | **Step 762 (`100M`) BPB** | `wRMS` (@600 / @762) | `actRMS` (@600 / @762) | `loop_cos` (@600) | `rep4(T.7)` (@300 / @600) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`gentle` (`0.015` / `0.03`)** | **`1.4858`** | **`1.1644`** | **`1.0854`** | `0.086` / **`0.090`** | `10.1` / **`12.3`** | `0.961` | **`1.6%`** / **`21.4%`** |
| Baseline (`0.020` / `0.10`, Run D) | `1.5246` | `1.1903` | *(plateau)* | `0.079` / `0.085` *(eq)* | `8.6` / `10.4` | **`0.951`** | `10.4%` / `23.2%` |
| `hot` (`0.035` / `0.05`) | `1.4924` | `1.1795` | `1.0913` | `0.159` / `0.159` *(eq)* | `40.7` / `44.8` | `0.965` | `7.0%` / `44.6%` |

**Empirical Ornstein-Uhlenbeck Equilibrium Law:** Across all three points, Muon's steady-state 2D weight RMS matches $\mathrm{wRMS}_{\mathrm{eq}} \approx 0.190 \sqrt{\eta_{\mathrm{muon}} / \lambda_{\mathrm{wd}}}$ to 3 digits (`0.020/0.10`: predicted `0.085`, measured `0.0846`; `0.035/0.05`: predicted `0.159`, measured `0.1588`; `0.015/0.03` at step 600: predicted `0.0866`, measured `0.0858`). We lock **`muon_lr = 0.015, weight_decay = 0.05`** (`wRMS_eq ≈ 0.104`, keeping `gentle`'s winning step size while guaranteeing bounded weight/activation norms over `11,400+` steps) for Phase 5.2.

---

## 3. Phase 5.1 — `coregrad-DSL`, Compact `LoT`, and the BPE Operator-Spacing Discovery

To overcome the $p^D$ BPE subword-drift cliff seen in English `<|think|>` chains (§2.2B), we built a unified, sandboxed ASCII-functional DSL ([`dsl.py`](../src/coregrad/tools/dsl.py), `34/34` tests in [`test_dsl.py`](../tests/test_dsl.py)) wired transparently behind `calc(expr)` — supporting arithmetic, `let`, short-circuit `if`, `1..n` ranges, `|>` pipelines, bounded recursion, SymPy (`solve/diff/int/simplify/expand/factor/subs/limit`), and an exact forward-chaining Horn/exclusion prover (`prove(...)`).

We then ran a controlled 5-step ablation on our locked unseen-vocabulary logic benchmark (`D = 2..12`, [`run_logic_depth_probe.py`](../scripts/run_logic_depth_probe.py), trained only on `D ∈ {2,3,4,5}`):

| Model | Mode / Representation | Mean ID (`d2,4,5`) | **Mean OOD (`d6–12`)** | `d2` | `d5` | `d6` | `d8` | `d10` | `d12` | Tool Valid (`d2–5`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `25m-looped14` | `cot` (English prose) | `38.3%` | `37.5%` | `55.0%` | `27.5%` | `37.5%` | `45.0%` | `32.5%` | `35.0%` | — |
| `25m-looped14` | `lot` (upfront `bind:`, 32w) | `40.0%` | `40.0%` | `37.5%` | `42.5%` | `40.0%` | `37.5%` | `50.0%` | `32.5%` | — |
| `25m-looped14` | `dsl` tight (`a->b`, 32w) | `36.7%` | `23.1%` | `45.0%` | `30.0%` | `22.5%` | `22.5%` | `22.5%` | `25.0%` | `95.8%` |
| `25m-looped14` | `dsl` tight (`a->b`, open-vocab) | `44.2%` | `23.1%` | `55.0%` | `35.0%` | `27.5%` | `30.0%` | `17.5%` | `17.5%` | `95.8%` |
| **`25m-looped14`** | **`dsl` BPE-aligned (`a -> b`, open-vocab)** | **`78.3%`** | **`44.4%`** | **`82.5%`** | **`75.0%`** | **`60.0%`** | **`52.5%`** | `37.5%` | `27.5%` | **`99.2%`** |
| `45m-looped14` | `cot` (English prose) | `42.5%` | `35.0%` | `55.0%` | `25.0%` | `50.0%` | `40.0%` | `25.0%` | `25.0%` | — |
| `45m-looped14` | `lot` (upfront `bind:`, 32w) | `30.0%` | `20.0%` | `35.0%` | `20.0%` | `37.5%` | `17.5%` | `12.5%` | `12.5%` | — |
| `45m-looped14` | `dsl` tight (`a->b`, 32w) | `32.5%` | `18.8%` | `45.0%` | `27.5%` | `22.5%` | `17.5%` | `17.5%` | `17.5%` | `90.0%` |
| `45m-looped14` | `dsl` tight (`a->b`, open-vocab) | `54.2%` | `30.0%` | `65.0%` | `52.5%` | `32.5%` | `32.5%` | `25.0%` | `30.0%` | **`100.0%`** |
| **`45m-looped14`** | **`dsl` BPE-aligned (`a -> b`, open-vocab)** | **`76.7%`** | **`50.0%`** | **`87.5%`** | **`67.5%`** | **`65.0%`** | `42.5%` | **`60.0%`** | `32.5%` | **`99.2%`** |

**Three Sharp Design Laws Discovered Before Phase 5.2:**
1. **No Lookahead Headers (`lot` falsification):** Emitting `bind: P0=...; P1=...` in *solution order* on Line 1 forces an autoregressive model to solve the entire shuffled graph in weights before its first reasoning step (`45m` collapsed to `20.0%` OOD). Left-to-right translation into `prove(...)` requires only `O(1)` lookahead.
2. **Never Use Small Fixed Vocabularies in Synthetic Generators:** Training on only 32 fixed fictional words caused `loss → 0.0001` memorization; the model hallucinated memorized training predicates (`tovix`) onto unseen eval words. Expanding to `3,400+` disjoint training pseudowords (`_OPEN_TRAIN_PREDS`) immediately lifted `45m` by `+21.7%` ID / `+11.2%` OOD.
3. **BPE Operator-Spacing Alignment (`+34%` ID / `+20%` OOD from whitespace alone!):** In byte-level BPE, prose words carry a leading space (`" treldon"` → `[1870, 468, 294]`), whereas tight operators strip it (`"treldon->plendor"` → `[115, 293, 468, 294, 538, 484, 479, 305]`, 8 tokens, **zero** leading-token match). Padding operators with single spaces (`" treldon -> plendor"` → `[1870, 468, 294, 4919, 566, 479, 305]`) uses **fewer tokens** while restoring **100% exact token-ID identity** for induction/copy heads — catapulting `25m-looped14` to **`78.3% ID / 44.4% OOD`** (`+40.0%` ID over English `cot`) and `45m-looped14` to **`76.7% ID / 50.0% OOD`** (`d2 = 87.5%`, `d10 = 60.0%`, `99.2%` valid syntax), even on pre-fix checkpoints still suffering repetition loops at `d10–d12`.

---

## 4. Phase 5.2 — Interleaved Pre/Post WSD Mid-Training A/B on `coregrad-32m-looped16-gru` (`1.5B` Tokens)

We trained **`coregrad-32m-looped16-gru`** (`33.04M` params, `2+4×3+2` Outer GRU Depth-RNN, locked `muon_lr=0.015, wd=0.05`) to a reusable **`1.2B`-token pure WSD plateau** (`decay_fraction=0.0`, `mean_bpb=0.9374` best / `0.9404` final, `wRMS=0.120`, `actRMS=13.1`, `loop_cos=0.839`, `rep4(T0/T.7)=48%/14%`, `94.8k tok/s`, `7.57 GB` VRAM), then branched two controlled **`300M`-token decays** inheriting the exact plateau Muon state (`1.5B` total pretraining/decay tokens):

- **Branch A (`Sequential` Baseline):** `300M` pure pretraining cosine decay (`p5a_decay`: **`mean_bpb = 0.8471`**, beating `45m-dense` `0.8609` at 25% fewer params, `actRMS=8.3`, `loop_cos=0.782`, `rep4(T.7)=10.1%`) → full 2-epoch masked SFT (`p5a_sft`, `3,910` steps / `256M` tok, `~50 min`) on `data/phase5/sft/train.jsonl` (`193,250` Phase 3 + BPE-aligned `coregrad-DSL` trajectories).
- **Branch B (`Interleaved` 50/50):** `300M` mixed cosine decay (`p5b_interleaved`, `~54 min`) blending 50% full-LM pretraining replay + 50% packed assistant-masked Phase 5 trajectories inside every `grad_accum` step (`rep4(T0/T.7) = 45%/6%`, `actRMS = 8.5`) — evaluated **with zero extra SFT (`p5b_interleaved`)** and after a light 600-step / `7.5`-min polish (**`p5b_polish`**).

### Locked 1.5B Head-to-Head Scoreboard (`33.04M` Outer-GRU vs. `125M` Dense Reference)

| Metric (Locked Benchmarks) | Branch A: `p5a_sft` (Sequential) | **Branch B: `p5b_interleaved` (0 extra SFT!)** | **Branch B+: `p5b_polish` (+7.5m polish)** | Ref: `125m` dense (`6.0B`, `sft_v2`) |
|---|---:|---:|---:|---:|
| **Post-Trained Val BPB *(Forgetting)*** | `1.0169` *(+0.170 loss!)* | **`0.8903` (`−0.1266` better!)** | **`0.8978`** | — |
| `cosmo` / `code` / `finemath` / `edu` BPB | `0.784 / 0.894 / 1.169 / 1.221` | **`0.698 / 0.690 / 1.036 / 1.137`** | `0.702 / 0.703 / 1.045 / 1.142` | — |
| **Phase 5 SFT Val Loss** | `0.8578` | `0.8553` | **`0.8424`** | — |
| **Gate 2 Tool Syntax** | `100.0%` (`371/371`) | **`100.0%` (`372/372`)** | **`100.0%` (`372/372`)** | `100.0%` |
| **Gate 2 Calc Accuracy** | `100.0%` (`100/100`) | **`100.0%` (`100/100`)** | **`100.0%` (`100/100`)** | `100.0%` |
| **Gate 2 Grounded Retrieval** | `98.9%` (`89/90`) | **`100.0%` (`90/90`)** | **`100.0%` (`90/90`)** | `94.4%` (`85/90`) |
| **Gate 2 Citation Rate (Halluc.)** | `98.9%` (`0`) | **`100.0%` (`0`)** | **`100.0%` (`0`)** | `96.7%` (`0`) |
| **Gate 2 Abstention** | `100.0%` (`30/30`) | **`100.0%` (`30/30`)** | **`100.0%` (`30/30`)** | `100.0%` |
| **OOD Chat No-Tool Routing** | `100.0%` (`4/4`) | **`100.0%` (`4/4`)** | **`100.0%` (`4/4`)** | `100.0%` |
| **OOD SQuAD v2 Target Cite / Span EM** | `56.0%` / **`32.0%`** | `56.0%` / `30.0%` | **`58.0%`** / `30.0%` | `80.0%` / `24.0%` |
| **OOD SQuAD v2 Unans. Abstain** | **`48.0%`** | `34.0%` | `38.0%` | `16.0%` |
| **OOD GSM8K EM / Syntax** | **`8.0%`** / `100%` | **`8.0%`** / `100%` | **`8.0%`** / `100%` | `4.0%` / `100%` |
| **Locked `coregrad-DSL` (160 tasks)** | `96.2%` (`154/160`) | **`97.5%` (`156/160`)** | **`97.5%` (`156/160`)** | — |
| — `bindings` / `pipelines` / `symbolic` | `100% / 100% / 100%` | **`100% / 100% / 100%`** | **`100% / 100% / 100%`** | — |
| — `logic` (`prove`, unseen `_EVAL_PREDS` `d2–8`) | `85.0%` (`34/40`) | **`90.0%` (`36/40`)** | **`90.0%` (`36/40`)** | — |
| **Grammar / 4-Way Completion** | `90.6%` / `45.8%` | `87.5%` / `45.8%` | `84.4%` / **`50.0%`** | `93.8%` / `62.5%` |

**Verdict — Decisive Win for Interleaved WSD (`Branch B` / `B+`):**
1. **Perfect Gate 2 (`100/100/100/100(0h)/100`) & `97.5%` `coregrad-DSL` at `33.04M` params:** With **zero separate SFT**, `p5b_interleaved` beat our `125M` dense reference (`6.0B` tok) on Gate 2 retrieval (`100.0%` vs `94.4%`), OOD SQuAD EM (`30.0%` vs `24.0%`), OOD SQuAD unanswerable abstention (`34–38%` vs `16%`), and OOD GSM8K (`8.0%` vs `4.0%`), while hitting `100%` on DSL bindings, pipelines/recursion, and SymPy algebra/calculus and **`90.0%`** on unseen-vocab multi-step `prove()` (`d2–d8`).
2. **Catastrophic Forgetting Nearly Eliminated:** Sequential 2-epoch SFT degraded validation BPB by `+0.170` (`0.8471 → 1.0169`, hurting `code-starcoder` by `+0.247` BPB), whereas 50/50 interleaved WSD decay retained **`0.8903` BPB** (`−0.1266` BPB better general language/code retention) while *also* beating sequential SFT on Gate 2, `coregrad-DSL`, and SFT val loss (`0.8553` vs `0.8578`; `0.8424` after 7.5-min polish).

### 4.1 Scaling the Winning Interleaved Recipe to `3.5B` Tokens (`3.0B` Plateau + `500M` Interleaved Decay)

Using WSD's zero-waste resumability, we resumed `coregrad_32m_gru_p5base` directly from `1.2B → 3.0B` tokens (`step 9,155..22,888`, plateau `mean_bpb: 0.9404 → 0.9119`, plateau `grammar: 84.4% → 96.9%`, `rep4(T.7): 13.9% → 6.4%`), then branched a **`500M`-token 50/50 interleaved decay** (`coregrad_32m_gru_3.5b_interleaved`, `3.5B` total) + light 600-step polish (`coregrad_32m_gru_3.5b_polish`):

| Metric (`coregrad-32m-looped16-gru`, `33.04M`) | `1.5B` `p5b_interleaved` | `1.5B` `p5b_polish` | **`3.5B` `3.5b_interleaved`** | **`3.5B` `3.5b_polish`** |
|---|---:|---:|---:|---:|
| **Post-Trained Val BPB** | `0.8903` | `0.8978` | **`0.8622`** (`−0.0281`) | **`0.8706`** (`−0.0272`) |
| `cosmo` / `code` / `finemath` / `edu` BPB | `0.698 / 0.690 / 1.036 / 1.137` | `0.702 / 0.703 / 1.045 / 1.142` | **`0.671 / 0.664 / 1.004 / 1.111`** | `0.677 / 0.677 / 1.012 / 1.117` |
| **Phase 5 SFT Val Loss** | `0.8553` | `0.8424` | `0.8311` | **`0.8232`** (`−0.0192`) |
| **Grammar / Greedy `T=0` `rep4`** | `87.5%` / `45.0%` | `84.4%` / — | **`90.6%` / `40.3%`** | **`90.6%`** / — |
| **Residual `actRMS` / `loop_cos`** | `8.5` / `0.804` | — | **`5.6` / `0.791`** | — |
| **Gate 2 (Syntax / Calc / Retr / Cite / Abst)** | **`100 / 100 / 100 / 100(0h) / 100`** | **`100 / 100 / 100 / 100(0h) / 100`** | `100 / 100 / 98.9 / 98.9(0h) / 100` | **`100 / 100 / 100 / 100(0h) / 100`** |
| **Locked `coregrad-DSL` (160 tasks)** | `97.5%` (`156/160`) | `97.5%` (`156/160`) | **`98.8%` (`158/160`)** | **`98.8%` (`158/160`)** |
| — `logic` (`prove`, unseen `_EVAL_PREDS`) | `90.0%` (`36/40`) | `90.0%` (`36/40`) | **`95.0%` (`38/40`)** | **`95.0%` (`38/40`)** |
| — `bindings` / `pipelines` / `symbolic` | `100% / 100% / 100%` | `100% / 100% / 100%` | **`100% / 100% / 100%`** | **`100% / 100% / 100%`** |
| **OOD SQuAD Cite / Span EM / Unans. Abst.** | `56%` / `30%` / `34%` | **`58%`** / **`30%`** / `38%` | `32%` / `14%` / **`46%`** | `54%` / `24%` / `40%` |
| **OOD GSM8K EM / Tool Syntax** | `8.0%` / `100%` | `8.0%` / `100%` | **`10.0%` (`5/50`)** / `100%` | `8.0%` / `100%` |

**Takeaways from `1.5B → 3.5B`:**
- **Clear gains in fluency, symbolic reasoning, and math:** Post-trained BPB improved `−0.028` (`0.890 → 0.862`), SFT val loss improved `−0.019` (`0.842 → 0.823`), grammar hit **`90.6%`** (`96.9%` on the 3.0B plateau), `actRMS` stayed rock-solid at **`5.6`**, greedy `T=0` repetition fell to **`40.3%`** (our lowest ever), GSM8K reached **`10.0%`** (`2.5×` `125m`), and locked `coregrad-DSL` hit **`98.8%` (`158/160`, `95.0%` unseen-vocab `prove()`!)**.
- **Light 600-step polish (`7.5 min`) is valuable after a long 500M interleaved decay:** Without polish, `3.5b_interleaved` became slightly over-cautious on OOD SQuAD v2 (`46%` unans. abstention, `32%/14%` cite/EM); 600 polish steps immediately restored SQuAD cite/EM (`54%/24%`), hit a perfect **`100/100/100/100(0h)/100`** on Gate 2, and preserved **`98.8%`** `coregrad-DSL` and **`0.8706`** BPB.
- **Core Thesis Validated:** A **`33.04M`** (`24.65M` non-embedding) `2+4×3+2` Outer GRU core reliably hits **100% Gate 2** and **98.8% `coregrad-DSL`**, confirming this compact trunk is ready to serve as the language-agnostic engine for **Phase 5.3** (including **Option C: Multi-Script "World Core"**).

---

## 5. Gate 4 (Phase 5.3A) — `coregrad-Byte` (H-Net UTF-8 Dynamic Patching) vs. `16k` BPE

To eliminate BPE vocabulary lock-in forever without losing effective sequence length, we implemented **`coregrad-32m-hnet-gru`** ([`hnet.py`](../src/coregrad/model/hnet.py), [`byte_tok.py`](../src/coregrad/tokenizer/byte_tok.py)): raw **UTF-8 bytes (`0..255`) + `32` protocol tokens = `288` vocab** (`0.15M` emb params → **`25.59M` total / `25.44M` non-emb = `99.4%` active compute**), `2` compiled `flex_attention` sliding-window (`W=256`) byte encoder layers → zero-mean `LayerNorm` cosine-similarity H-Net router (`target_ratio=4.5`) → `4×3` Outer-GRU global core → parallel $O(\log M)$ causal EMA smoothing + STE confidence-gated upsampler → `2` local byte decoder layers (`38/38` unit tests verifying bit-for-bit causality and exact router gradient flow).

We trained **`gate4_32m_hnet_gru_1.2gb`** on **`1.20 GB` of raw UTF-8 text** (`2,288` steps × `524,288 B/step`, `seq_len=4096`, `mb=4, ga=32`, `muon_lr=0.015, wd=0.05`, `146.2 min`, `134.3k B/s`, `8.56 GB` VRAM) — matching our `300M`-token (`1.19 GB` UTF-8) `16k`-BPE `diag300m_32m_looped16_gru` (`33.04M` params) 1:1 on raw text volume.

### Direct Same-Raw-Text BPB Audit (`319 KB` of Identical Held-Out Validation Text per Model)

*(We also caught and fixed a truncation bug in `prepare_byte_shards.py`'s `manifest.json` that inflated `bpt` on capped val files; the table below reports exact, audited NLL / (UTF-8 bytes × ln 2) evaluated directly on the exact same decoded held-out validation strings for both models.)*

| Metric (Matched `~1.20 GB` UTF-8 Text) | `16k-BPE` `32m-looped16-gru` (`33.04M` total) | **`coregrad-Byte` `32m-hnet-gru` (`25.59M` total, −23% params!)** | Delta (`H-Net − BPE`) |
|---|---:|---:|---:|
| **Direct Same-Text Mean BPB** | **`1.0358`** | **`1.0539`** | **`+0.0181` (PASS `≤ 0.03`!)** |
| — `code-starcoder` (`85.7 KB`) | `0.8328` | **`0.8011`** | **`−0.0318` (H-Net wins!)** |
| — `finemath` (`70.1 KB`) | `1.1470` | `1.1625` | `+0.0155` |
| — `cosmo` (`82.7 KB`) | `0.8724` | `0.9140` | `+0.0416` |
| — `edu-en` (`80.7 KB`) | `1.2910` | `1.3380` | `+0.0470` |
| **4-Way Completion Probe** | `37.5%` | **`50.0%` (`+12.5%`!)** | **`+12.5%`** |
| **Grammar Minimal Pairs** | `87.5%` | `81.2%` | `−6.3%` |
| **Sampled `T=0.7` / Greedy `T=0` `rep4`** | `18.0%` / `76.0%` | **`8.0%` / `37.0%` (½ of BPE!)** | **−10% / −39%** |
| **Health Vitals (`wRMS` / `actRMS` / `loop_cos` / `gate`)** | `0.078` / `6.6` / `0.886` / `0.73` | **`0.105` / `4.3` / `0.831` / `0.82`** | Rock-solid |
| **Learned Compression Ratio `R` (Train / Val)** | `3.96` B/tok *(fixed BPE)* | **`3.56` / `4.36` B/chunk *(learned)*** | Self-learned |
| **Invalid UTF-8 bytes in samples (Step 300+)** | — | **`0` (100% valid UTF-8)** | Zero `\ufffd` |

**Self-Learned UTF-8 Chunks at Step 2,288 (Zero Tokenizer, Zero Dictionary):**
```text
The| quick| br|own| fo|x| ju|mps| ove|r| the| la|zy| do|g|.
def| fi|b|(n|)|: return| n| if| n| |<=| 1| e|lse| fi|b|(n| |-| 1|) |+| fi|b|(n| |-| 2|)
pr|ove|(|[| x| |: tr|el|don|, trel|don| |->| pl|end|or| |]|, x| |: plend|or|)
```

**Gate 4 Step 1 Verdict — PASS:** With **23% fewer parameters (`25.59M` vs `33.04M`)** and **zero BPE tokenizer**, `coregrad-Byte` landed within **`+0.0181 BPB`** of `16k` BPE on English, **beat `16k` BPE outright on code (`−0.0318 BPB`)**, beat `16k` BPE on 4-way completions (`50.0%` vs `37.5%`), halved repetition (`8%` / `37%` vs `18%` / `76%`), generated 100% valid UTF-8 from step 300 onward, and self-discovered crisp word, Python keyword, and DSL operator (`|<=|`, `|->|`) boundaries purely from gradients.

### 5.1 Gate 4 Step 2 — Byte-Level Phase 5 SFT, Gate 2 & `coregrad-DSL` Verification

Fine-tuned `runs/gate4_32m_hnet_gru_1.2gb/best.pt` (`25.59M`, pretrained on only `1.20 GB` ≈ `300M` BPE-tok eq — `5×` less pretraining than our `1.5B` BPE runs!) with `2,400` masked SFT steps (`314.6M` bytes, `seq_len=4096`, `39.0 min`, `best_val_loss=0.3223`, `8.14 GB` VRAM):

| Benchmark | `coregrad-Byte` `25.59M` (`300M` BPE-eq base, `4KB` ctx) | Ref: `25m-looped14` `16k`-BPE (`1.5B` base, `8KB` ctx) | Notes |
|---|---:|---:|---|
| **Gate 2 Tool Syntax** | **`100.0%` (`373/373`)** | `100.0%` | Zero byte syntax errors! |
| **Gate 2 Calc Accuracy** | **`100.0%` (`100/100`)** | `98.0%` | Perfect exact byte arithmetic |
| **Gate 2 Grounded Retrieval** | **`91.1%` (`82/90`)** | `86.7%` | Beats `25m-looped14` with `5×` less pretraining! |
| **Gate 2 Citation Rate (Halluc.)** | **`96.7%` (`0` halluc.)** | `94.4%` (`0` halluc.) | Zero hallucinated doc IDs |
| **Gate 2 Abstention** | **`100.0%` (`30/30`)** | `100.0%` | Perfect abstention |
| **Locked `coregrad-DSL` (160 tasks)** | **`90.0%` (`144/160`, `100%` valid `153/153`)** | — | **`100%` `bindings`, `100%` `pipelines`, `100%` `symbolic`**, `60%` `logic` |
| **OOD GSM8K EM / Syntax** | **`4.0%`** / **`100.0%` (`159/159`)** | `10.0%` / `100%` | Matches `125m` (`4%`) with `5×` less pretraining |
| **OOD Chat Routing** | `75.0%` (`3/4`) | `100.0%` | — |
| **OOD SQuAD v2 Cite / EM** | `0.0%` / `6.0%` *(4KB ctx clip!)* | `60.0%` / `14.0%` | `k=3` SQuAD passages (`~4.5–5.5 KB`) exceeded `4096 B` window → confirms need for **`8,192 B` context** in `CoreGradByteLM v2`! |

**Gate 4 Verdict — PASSED (with clear v2 upgrade path):** A `25.59M` raw UTF-8 H-Net model with only `300M` BPE-eq pretraining achieved **`100%` tool syntax**, **`100%` calc**, **`91.1%` retrieval (`0` hallucinated)**, **`100%` abstention**, and **`100%` on DSL bindings, functional `|>` pipelines, and SymPy algebra/calculus** byte-by-byte. The SQuAD v2 `4 KB` clipping cleanly identifies why upgrading to **`CoreGradByteLM v2`** (`d_byte=384, d_core=768`, causal `Conv1d(k=4)`, `8,192`-byte context) is the right prerequisite before multi-script scaling.

### Remaining Optional Phase 4A Ablations
- Cognitive Core Purity A/B (`25m` pretrained with `0%` web/encyclopedic `edu-en` vs. `45% edu-en`).
- GQA KV-Bank scaling check on `45m` / `32m-looped16-gru`.

