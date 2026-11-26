#!/usr/bin/env python3
"""Phase 4B Test 4: Middle-Layer Latent Query Alignment & Layer-Depth Sweep.

Trains a lightweight MLP query head h_L -> EmbeddingGemma 2 (768d) across multiple
depths of frozen coregrad-125m (layers 3, 6, 9, 12, 15, 18) and benchmarks Recall@1/3/5/10
on 500 held-out SQuAD v2 validation questions against raw EmbeddingGemma 2 and BM25.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from coregrad.config import load_local_paths
from coregrad.model import CoreGradLM, ModelConfig
from coregrad.model.memory_bridge import CoreGradMemoryBridge, LatentQueryHead, MemoryBridgeConfig
from coregrad.tokenizer import CoreGradTokenizer
from coregrad.tools import Document, EmbeddingGemma2Embedder, InMemoryBM25Retriever


def _extract_pooled_hidden_states(
    bridge: CoreGradMemoryBridge,
    tokenizer: CoreGradTokenizer,
    questions: list[str],
    layers: tuple[int, ...],
    device: torch.device,
    batch_size: int = 32,
) -> dict[int, torch.Tensor]:
    """Extract last-token + mean-pooled hidden representations at each target layer."""
    storage: dict[int, list[torch.Tensor]] = {l_idx: [] for l_idx in layers}
    pad_id = tokenizer.pad_id

    for start in range(0, len(questions), batch_size):
        batch_q = questions[start : start + batch_size]
        encoded = [tokenizer.encode(q, add_bos=True, add_eos=False)[:128] for q in batch_q]
        max_len = max(len(ids) for ids in encoded)
        x = torch.full((len(encoded), max_len), pad_id, dtype=torch.long, device=device)
        lens = torch.tensor([len(ids) for ids in encoded], dtype=torch.long, device=device)
        for i, ids in enumerate(encoded):
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            layer_outs = bridge.extract_layer_hidden(x, layers)

        mask = (torch.arange(max_len, device=device)[None, :] < lens[:, None]).unsqueeze(-1).float()
        for l_idx, h in layer_outs.items():
            h_f = h.float()
            mean_h = (h_f * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            last_h = h_f[torch.arange(len(encoded), device=device), lens - 1]
            # Average last-token and mean-pooled representations for strong semantic summary
            storage[l_idx].append((0.5 * mean_h + 0.5 * last_h).cpu())

    return {l_idx: torch.cat(chunks, dim=0) for l_idx, chunks in storage.items()}


def _compute_recall_at_k(
    query_vecs: np.ndarray, doc_vecs: np.ndarray, gold_doc_indices: list[int], ks: tuple[int, ...] = (1, 3, 5, 10)
) -> dict[str, float]:
    sims = query_vecs @ doc_vecs.T  # (Q, D)
    ranks = np.argsort(-sims, axis=1)
    out: dict[str, float] = {}
    for k in ks:
        hits = sum(int(gold_doc_indices[i] in ranks[i, :k]) for i in range(len(gold_doc_indices)))
        out[f"R@{k}"] = hits / max(len(gold_doc_indices), 1)
    return out


def _train_and_eval_head(
    h_train: torch.Tensor,
    target_q_train: torch.Tensor,
    h_val: torch.Tensor,
    doc_vecs_val: np.ndarray,
    gold_doc_idx: list[int],
    device: torch.device,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 2e-3,
) -> tuple[dict[str, float], float, LatentQueryHead]:
    head = LatentQueryHead(d_model=h_train.shape[1], query_dim=target_q_train.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    n = len(h_train)
    temp = 0.07

    h_tr = h_train.to(device)
    y_tr = F.normalize(target_q_train.to(device), p=2, dim=-1)

    for _ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            if len(idx) < 8:
                continue
            pred = head(h_tr[idx])
            tgt = y_tr[idx]
            cos_loss = (1.0 - (pred * tgt).sum(dim=-1)).mean()
            # In-batch symmetric contrastive loss sharpens retrieval separation
            logits = (pred @ tgt.T) / temp
            labels = torch.arange(len(idx), device=device)
            nce_loss = F.cross_entropy(logits, labels)
            loss = cos_loss + 0.35 * nce_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    head.eval()
    with torch.no_grad():
        val_pred = head(h_val.to(device)).cpu().numpy()
        val_cos = float(
            (torch.from_numpy(val_pred) * F.normalize(torch.from_numpy(doc_vecs_val[gold_doc_idx]), p=2, dim=-1))
            .sum(dim=-1)
            .mean()
            .item()
        )
    recalls = _compute_recall_at_k(val_pred, doc_vecs_val, gold_doc_idx)
    return recalls, val_cos, head


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4B Test 4: Middle-layer latent query alignment.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/coregrad_125m_base/best.pt"))
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("runs/eval/phase4b_latent_query.json"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    paths = load_local_paths()
    assert paths.hf_cache_dir is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CoreGradTokenizer.from_file(paths.data_dir / "tokenizer" / "tokenizer.json")

    sq_train = paths.hf_cache_dir / "rajpurkar/squad_v2/squad_v2/train-00000-of-00001.parquet"
    sq_val = paths.hf_cache_dir / "rajpurkar/squad_v2/squad_v2/validation-00000-of-00001.parquet"

    rng = random.Random(42)
    tr_rows = [r for r in pq.read_table(sq_train, columns=["question", "context"]).to_pylist() if r["question"]]
    rng.shuffle(tr_rows)
    train_questions = [str(r["question"]).strip() for r in tr_rows[: args.n_train]]

    val_rows = [
        r
        for r in pq.read_table(sq_val, columns=["title", "question", "context", "answers"]).to_pylist()
        if r["answers"]["text"] and len(r["context"]) <= 1400
    ]
    rng.shuffle(val_rows)
    val_rows = val_rows[: args.n_val]
    val_questions = [str(r["question"]).strip() for r in val_rows]

    # Build deduplicated candidate passage bank from all val rows
    ctx_to_idx: dict[str, int] = {}
    val_docs: list[Document] = []
    val_doc_texts: list[str] = []
    gold_doc_idx: list[int] = []
    for r in val_rows:
        ctx = str(r["context"]).strip()
        title = str(r["title"]).replace("_", " ").strip()
        if ctx not in ctx_to_idx:
            cid = len(val_doc_texts)
            ctx_to_idx[ctx] = cid
            val_doc_texts.append(f"{title}\n{ctx}")
            val_docs.append(Document(doc_id=f"d{cid:04d}", title=title, text=ctx))
        gold_doc_idx.append(ctx_to_idx[ctx])

    print(
        f"Loaded {len(train_questions):,} train Qs, {len(val_questions)} val Qs across "
        f"{len(val_doc_texts)} unique held-out passages.",
        flush=True,
    )

    # 1. Embed train/val queries and val passages with EmbeddingGemma 2 (768d)
    embedder = EmbeddingGemma2Embedder(mrl_dim=768, device=str(device))
    print("Encoding queries & val passages with google/embeddinggemma-2 (768d)...", flush=True)
    q_tr_vecs = embedder.encode_queries(train_questions)
    q_val_vecs = embedder.encode_queries(val_questions)
    d_val_vecs = embedder.encode_documents(val_doc_texts)

    # Upper bound: raw EmbeddingGemma 2 text query
    eg2_recalls = _compute_recall_at_k(q_val_vecs, d_val_vecs, gold_doc_idx)
    print(f"[Upper Bound] Raw EmbeddingGemma 2 text query: {eg2_recalls}", flush=True)

    # Sparse baseline: BM25
    bm25 = InMemoryBM25Retriever(val_docs)
    bm25_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    for i, q in enumerate(val_questions):
        gold_id = f"d{gold_doc_idx[i]:04d}"
        top_ids = [h.doc_id for h in bm25.search(q, k=10)]
        for k in bm25_hits:
            bm25_hits[k] += int(gold_id in top_ids[:k])
    bm25_recalls = {f"R@{k}": v / len(val_questions) for k, v in bm25_hits.items()}
    print(f"[Sparse Base] BM25 text query                : {bm25_recalls}", flush=True)

    # Free embedder VRAM before loading coregrad-125m
    del embedder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. Extract frozen coregrad-125m hidden states across layer depths
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_model = CoreGradLM(ModelConfig(**ckpt["model_config"])).to(device).eval()
    base_model.load_state_dict(ckpt["model_state_dict"])
    bridge = CoreGradMemoryBridge(base_model, MemoryBridgeConfig(freeze_base=True))

    sweep_layers = (3, 6, 9, 12, 15, 18)
    print(f"Extracting frozen coregrad-125m hidden states at layers {sweep_layers}...", flush=True)
    h_tr_by_layer = _extract_pooled_hidden_states(bridge, tokenizer, train_questions, sweep_layers, device)
    h_val_by_layer = _extract_pooled_hidden_states(bridge, tokenizer, val_questions, sweep_layers, device)

    target_q_tr = torch.from_numpy(q_tr_vecs)
    layer_results: dict[str, dict[str, float]] = {}
    best_layer = -1
    best_r5 = -1.0
    best_head: LatentQueryHead | None = None

    for l_idx in sweep_layers:
        recalls, mean_cos, head = _train_and_eval_head(
            h_tr_by_layer[l_idx],
            target_q_tr,
            h_val_by_layer[l_idx],
            d_val_vecs,
            gold_doc_idx,
            device=device,
            epochs=args.epochs,
        )
        pct_of_eg2 = recalls["R@5"] / max(eg2_recalls["R@5"], 1e-6) * 100.0
        layer_results[str(l_idx)] = {**recalls, "mean_q_doc_cos": mean_cos, "pct_of_eg2_R5": pct_of_eg2}
        print(
            f"[Layer {l_idx:2d}/18] R@1={recalls['R@1']*100:5.1f}% | R@3={recalls['R@3']*100:5.1f}% | "
            f"R@5={recalls['R@5']*100:5.1f}% ({pct_of_eg2:5.1f}% of EG2) | R@10={recalls['R@10']*100:5.1f}%",
            flush=True,
        )
        if recalls["R@5"] > best_r5:
            best_r5 = recalls["R@5"]
            best_layer = l_idx
            best_head = head

    assert best_head is not None
    head_out = paths.runs_dir / "phase4b_query_head" / f"query_head_layer{best_layer}.pt"
    head_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"layer": best_layer, "d_model": base_model.cfg.d_model, "query_dim": 768, "state_dict": best_head.state_dict()},
        head_out,
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "n_train": len(train_questions),
        "n_val": len(val_questions),
        "n_unique_val_passages": len(val_doc_texts),
        "embeddinggemma2_upper_bound": eg2_recalls,
        "bm25_baseline": bm25_recalls,
        "layer_sweep": layer_results,
        "best_layer": best_layer,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved Test 4 report -> {args.out} (best layer={best_layer}, elapsed={report['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
