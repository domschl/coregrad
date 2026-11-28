# Prior Research & Technical References

`coregrad` synthesizes proven ideas from modern open LLM research into a compact, readable codebase runnable on a single 12 GB consumer GPU. Below is a structured reference map linking each research contribution directly to where it is implemented in `src/coregrad/`.

---

## 1. Philosophy & Pedagogical Inspiration

- **Micrograd, nanoGPT & nanochat:** Karpathy, A. (2020–2025). [`micrograd`](https://github.com/karpathy/micrograd), [`nanoGPT`](https://github.com/karpathy/nanoGPT), and the **"Cognitive Core"** thesis: separating compact reasoning/language/tool-orchestration machinery from world knowledge stored externally.
- **modded-nanoGPT:** Jordan, K., et al. (2024–2025). [`KellerJordan/modded-nanogpt`](https://github.com/KellerJordan/modded-nanogpt) — community speedrunning benchmark demonstrating dramatic sample-efficiency gains at ~124M scale via modern norms, RoPE, QK-Norm, and Muon.

---

## 2. Compact Transformer Architecture (`src/coregrad/model/transformer.py`)

- **Transformer Decoder:** Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need.* NeurIPS 2017 (`arXiv:1706.03762`).
- **Pre-RMSNorm:** Zhang, B., & Sennrich, R. (2019). *Root Mean Square Layer Normalization.* NeurIPS 2019 (`arXiv:1910.07467`). Implemented in `RMSNorm` via `F.rms_norm`.
- **SwiGLU Feed-Forward Network:** Shazeer, N. (2020). *GLU Variants Improve Transformer* (`arXiv:2002.05202`). Implemented in `SwiGLU`.
- **Rotary Position Embeddings (RoPE):** Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding* (`arXiv:2104.09864`). Implemented in `precompute_rope_freqs` & `apply_rotary_emb`.
- **Grouped-Query Attention (GQA):** Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023 (`arXiv:2305.13245`). Implemented in `CausalSelfAttention` (12 Q / 4 KV heads in `coregrad-125m`).
- **QK-Norm (Attention Logit Stability):** Dehghani, M., et al. (2023). *Scaling Vision Transformers to 22 Billion Parameters* (`arXiv:2302.05442`); Henry, A., et al. (2020) *Query-Key Normalization for Transformers* (`arXiv:2010.04245`). Prevents attention entropy collapse at high learning rates.
- **FlashAttention / Memory-Efficient SDPA:** Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022 (`arXiv:2205.14135`); Dao, T. (2023) *FlashAttention-2* (`arXiv:2307.08691`). Invoked natively via `F.scaled_dot_product_attention(..., is_causal=True, enable_gqa=...)`.
- **Deep-and-Narrow Sub-Billion Scaling & Tied Embeddings:** Liu, Z., Zhao, C., Iandola, F., Lai, C., Tian, Y., Fedorov, I., Xiong, Y., Chang, E., Shi, Y., Krishnamoorthi, R., Lai, L., & Chandra, V. (2024). *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases.* ICML 2024 (`arXiv:2402.14905`). Motivates our 18-layer / `d_model=768` geometry and weight tying.
- **Residual Projection Init Scaling:** Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2). Scales `o_proj` and `w_down` initialization by $1 / \sqrt{2 L}$.

---

## 3. Tokenization & Data Curation (`src/coregrad/tokenizer/`, `src/coregrad/data/`)

- **Byte-Level BPE:** Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units* (`arXiv:1508.07909`); Radford, A., et al. (2019) GPT-2 byte-level BPE.
- **Single-Digit Number Splitting:** Grattafiori, A., Dubey, A., et al. (2024). *The Llama 3 Herd of Models* (`arXiv:2407.21783`); OpenAI GPT-4 tokenizer regex design. Splitting `\p{N}` into single digits improves arithmetic consistency in small models.
- **Scaling Laws & Overtraining Small Models:** Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla, `arXiv:2203.15556`); Touvron, H., et al. (2023) *LLaMA* (`arXiv:2302.13971`); Ben Allal, L., et al. (2025) *SmolLM2* (`arXiv:2502.02737`). Justifies training a 125M model to ~6B high-signal tokens (~2.4× Chinchilla) for strong inference-time quality.
- **Bits-Per-Byte (BPB) Evaluation:** Gao, L., et al. (2020). *The Pile: An 800GB Dataset of Diverse Text for Language Modeling* (`arXiv:2101.00027`). Enables tokenizer-agnostic comparison across vocabulary sizes (16k vs. 32k vs. 8k).
- **Grammatical Minimal Pairs (BLiMP):** Warstadt, A., Parrish, A., Liu, H., Mohananey, A., Peng, W., Wang, S.-F., & Bowman, S. R. (2020). *BLiMP: The Benchmark of Linguistic Minimal Pairs for English.* TACL (`arXiv:1912.00582`). Inspires `GRAMMAR_MINIMAL_PAIRS` in `src/coregrad/eval/probes.py`.

---

## 4. Optimization & Learning Rate Schedules (`src/coregrad/train/`)

- **AdamW (Decoupled Weight Decay):** Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR 2019 (`arXiv:1711.05101`).
- **Muon (MomentUm Orthogonalized by Newton-Schulz):**
  - Jordan, K., Jin, Y., Boza, V., You, J., Cesista, F., Newhouse, L., & Bernstein, J. (2024). *Muon: An Optimizer for Hidden Layers in Neural Networks* ([blog post](https://kellerjordan.github.io/posts/muon/)).
  - Liu, J., et al. / Moonshot AI (2025). *Muon is Scalable for LLM Training* (Moonlight, `arXiv:2502.16982`) — motivates decoupled weight decay for Muon (note: `src/coregrad/train/optim.py` uses Keller Jordan's `sqrt(max(1, rows/cols))` aspect scaling with full weight decay, verified via our Phase 4C diagnostic).
- **Warmup–Stable–Decay (WSD) Schedule:**
  - Hu, S., Tu, Y., Han, X., et al. (2024). *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (`arXiv:2404.06395`).
  - Hägele, A., Bakouch, E., Kosson, A., Ben Allal, L., von Werra, L., & Jaggi, M. (2024). *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations.* NeurIPS 2024 (`arXiv:2405.18392`). Enables checkpoint-and-branch annealing without restarting pretraining.

---

## 5. Tool-Use, Post-Training & Verifiable RL (`src/coregrad/tools/`, `src/coregrad/train/sft.py`, `grpo.py`)

- **ReAct (Interleaved Reasoning & Tool Action):** Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023 (`arXiv:2210.03629`). Directly motivates `<|think|>` → `<|call|>` → `<|result|>` trajectories.
- **Toolformer:** Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools.* NeurIPS 2023 (`arXiv:2302.04761`). Demonstrates compact LMs offloading arithmetic and lookup via special tool tokens.
- **Sequence Packing for SFT:** Krell, M. M., Kosec, M., Perez, S. P., & Fitzgibbon, A. (2022). *Efficient Sequence Packing without Cross-Contamination* (`arXiv:2107.02027`). Implemented in `pack_examples_to_rows` (`src/coregrad/train/sft.py`).
- **GRPO / Reinforcement Learning with Verifiable Rewards (RLVR):**
  - Shao, Z., Wang, P., Zhu, Q., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (`arXiv:2402.03300`).
  - DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (`arXiv:2501.12948`). Implemented in `src/coregrad/train/grpo.py` with programmatic rewards over tool syntax, exact answers, and grounded citations.
- **Long-Context RoPE Extension (YaRN / Base Scaling):** Peng, B., Quesnelle, J., Fan, H., & Shippole, E. (2024). *YaRN: Efficient Context Window Extension of Large Language Models.* ICLR 2024 (`arXiv:2309.00071`); Xiong, W., et al. (2023) *Effective Long-Context Scaling of Foundation Models* (`arXiv:2309.16039`).

---

## 6. Hybrid Retrieval & Local Research (`src/coregrad/tools/retriever.py`)

- **Okapi BM25:** Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval, 3(4), 333–389. Implemented in `InMemoryBM25Retriever` and SQLite FTS5.
- **Reciprocal Rank Fusion (RRF):** Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR 2009. Fuses BM25 and dense vector rankings cleanly without score-scale calibration.
- **Matryoshka Representation Learning (MRL):** Kusupati, A., Bhatt, G., Rege, A., Wallingford, M., Sinha, A., Ramanujan, V., Howard-Snyder, W., Chen, K., Kakade, S., Jain, P., & Farhadi, A. (2022). *Matryoshka Representation Learning.* NeurIPS 2022 (`arXiv:2205.13147`). Allows truncating 768d embeddings to 256d for fast in-RAM search.
- **EmbeddingGemma 2:** Google DeepMind (2026). *EmbeddingGemma 2: Unified and Efficient Multimodal Representations* (`arXiv:2610.37364`).

---

## 7. Phase 4 Research Frontier (Essentialization & Latent Memory Bridge)

- **Universal & Looped Transformers (Weight-Shared Depth):**
  - Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., & Kaiser, Ł. (2019). *Universal Transformers.* ICLR 2019 (`arXiv:1807.03819`).
  - Giannou, A., Rajput, S., Sohn, J., Lee, K., Lee, J. D., & Papailiopoulos, D. (2023). *Looped Transformers as Programmable Computers.* ICML 2023 (`arXiv:2301.13196`).
  - Geiping, J., McLeish, S., Jain, N., Kirchenbauer, J., Singh, S., Bartoldson, B. R., Kailkhura, B., Bhatele, A., & Goldstein, T. (2025). *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* (`arXiv:2502.05171`).
- **RETRO (Chunked Cross-Attention Retrieval):** Borgeaud, S., Mensch, A., Hoffmann, J., et al. (2022). *Improving Language Models by Retrieving from Trillions of Tokens.* ICML 2022 (`arXiv:2112.04426`).
- **Memorizing Transformers:** Wu, Y., Rabe, M. N., Hutchins, D., & Szegedy, C. (2022). *Memorizing Transformers.* ICLR 2022 (`arXiv:2203.08913`).
- **xRAG / Latent Embedding Injection:** Cheng, X., Wang, X., Zhang, X., Ge, T., Chen, S.-Q., Wei, F., Zhang, H., & Zhao, D. (2024). *xRAG: Extreme Context Compression for Retrieval-Augmented Generation with One Token.* NeurIPS 2024 (`arXiv:2405.13792`).
- **Zero-Initialized Gated Cross-Attention (Flamingo-style retrofitting):** Alayrac, J.-B., Donahue, J., Luc, P., et al. (2022). *Flamingo: a Visual Language Model for Few-Shot Learning.* NeurIPS 2022 (`arXiv:2204.14198`).

---

## 8. Phase 5 Frontier (Dynamic Byte Patching & Multilingual Scaling Laws)

- **Byte Latent Transformer (BLT):** Pagnoni, A., Pasunuru, R., Rodriguez, P., Nguyen, J., Muller, B., Li, M., Zhou, C., Yu, L., Weston, J., Zettlemoyer, L., Ghosh, G., Lewis, M., Holtzman, A., & Iyer, S. (2024). *Byte Latent Transformer: Patches Scale Better Than Tokens* (`arXiv:2412.09871`). Introduces entropy-based dynamic byte patching to eliminate fixed subword tokenizers without losing effective context length.
- **H-Net (End-to-End Dynamic Chunking):** Hwang, S., Wang, B., & Gu, A. (2025). *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (`arXiv:2507.07955`). Introduces cosine-similarity boundary routing, ratio load-balancing loss, and EMA/STE smoothing upsampling so byte-to-chunk compression is learned 100% end-to-end with gradients.
- **Multilingual Capacity Interference & Scaling Laws:**
  - Conneau, A., Khandelwal, K., Goyal, N., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale (XLM-R).* ACL 2020 (`arXiv:1911.02116`) — characterizes the Curse of Multilinguality.
  - Fernandes, P., Ghorbani, B., Garcia, X., Freitag, M., & Firat, O. (2023). *Scaling Laws for Multilingual Neural Machine Translation.* ICML 2023 (`arXiv:2302.09650`).
  - Chang, T. A., Arnett, C., Tu, Z., & Bergen, B. K. (2024). *When is Multilinguality a Curse? Language Modeling for 250 High- and Low-Resource Languages* (`arXiv:2311.09205`).
- **FineWeb-2 (Open Multilingual Corpus):** Penedo, G., Kydlíček, H., Sabolčec, V., Messmer, B., Foroutan, N., Jaggi, M., von Werra, L., & Wolf, T. (2025). *FineWeb2: One Pipeline to Scale Them All — Adapting Pre-Training Data Processing to Every Language* (`HuggingFaceFW/fineweb-2`, ODC-By).
