#!/usr/bin/env python3
"""Phase 4B Mechanism 3: Independent Precomputed GQA KV-Bank Injection (TurboRAG / CAG).

Precomputes each passage's pre-RoPE GQA (K, V) tensors independently offline (zero cross-passage
prefill), then drops k retrieved passages' latent KV tensors directly into coregrad-125m's
pretrained self-attention heads at query time (ZERO training, ZERO query-time passage prefill!).

Evaluates strictly on the exact same 500 held-out SQuAD v2 validation items under:
  - OFF (no memory)
  - SHUFFLED k=1, k=4, k=8 (wrong cached KV passages)
  - MATCHED k=1, k=4, k=8 (gold + independently-encoded distractor KV passages)
  - Full Joint Text-in-Prompt k=4 (exact upper bound for multi-passage reading)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from coregrad.config import load_local_paths
from coregrad.model import CoreGradLM, ModelConfig
from coregrad.model.memory_bridge import encode_kv_passage, forward_with_kv_bank
from coregrad.tokenizer import CoreGradTokenizer

_WORD_RE = re.compile(r"[a-z0-9]+")


def _token_f1(pred: str, gold: str) -> float:
    p = _WORD_RE.findall(pred.lower())
    g = _WORD_RE.findall(gold.lower())
    if not p or not g:
        return 0.0
    counts: dict[str, int] = {}
    for w in p:
        counts[w] = counts.get(w, 0) + 1
    ov = 0
    for w in g:
        if counts.get(w, 0) > 0:
            ov += 1
            counts[w] -= 1
    if ov == 0:
        return 0.0
    return 2.0 * ov / (len(p) + len(g))


def _concat_kv(
    kv_list: list[list[tuple[torch.Tensor, torch.Tensor]]],
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    n_layer = len(kv_list[0])
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_i in range(n_layer):
        # Keep BOS sink only on the first passage; strip duplicate BOS from subsequent passages
        ks = torch.cat(
            [kv[layer_i][0] if idx == 0 else kv[layer_i][0][:, :, 1:, :] for idx, kv in enumerate(kv_list)],
            dim=2,
        ).to(device)
        vs = torch.cat(
            [kv[layer_i][1] if idx == 0 else kv[layer_i][1][:, :, 1:, :] for idx, kv in enumerate(kv_list)],
            dim=2,
        ).to(device)
        out.append((ks, vs))
    return out


@torch.no_grad()
def _generate_kv(
    model: CoreGradLM,
    tokenizer: CoreGradTokenizer,
    prompt_ids: list[int],
    kv_bank: list[tuple[torch.Tensor, torch.Tensor]] | None,
    device: torch.device,
    max_new_tokens: int = 24,
) -> str:
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos_id = tokenizer.eos_id
    amp_enabled = device.type == "cuda"
    gen: list[int] = []
    for _ in range(max_new_tokens):
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            logits, _ = forward_with_kv_bank(model, x, kv_bank=kv_bank)
        nxt = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        if nxt == eos_id:
            break
        gen.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


@torch.no_grad()
def eval_kv_condition(
    model: CoreGradLM,
    tokenizer: CoreGradTokenizer,
    val_raw: list[tuple[str, list[str], int]],
    kv_cache: list[list[tuple[torch.Tensor, torch.Tensor]]],
    device: torch.device,
    *,
    condition: str,
    k_passages: int,
    gen_limit: int = 120,
    max_mem_tokens: int = 1900,
) -> dict[str, float]:
    rng = random.Random(777)
    rng_g = random.Random(888)
    amp_enabled = device.type == "cuda"
    n_bank = len(kv_cache)

    total_nll = 0.0
    total_toks = 0
    total_bytes = 0
    t_q = time.perf_counter()

    def pick_kv(gold_idx: int, r: random.Random) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        if condition == "off":
            return None
        chosen: list[int] = [gold_idx] if condition == "matched" else []
        while len(chosen) < k_passages:
            c = r.randrange(n_bank)
            if c != gold_idx and c not in chosen:
                chosen.append(c)
        r.shuffle(chosen)
        kv = _concat_kv([kv_cache[i] for i in chosen], device=device)
        if kv[0][0].shape[2] > max_mem_tokens:
            kv = [(k[:, :, :max_mem_tokens, :], v[:, :, :max_mem_tokens, :]) for k, v in kv]
        return kv

    for q, ans_list, gold_pidx in val_raw:
        prompt_str = f"\n\nQuestion: {q}\nAnswer:"
        ans_str = f" {ans_list[0]}"
        p_ids = tokenizer.encode(prompt_str, add_bos=(condition == "off"), add_eos=False)
        a_ids = tokenizer.encode(ans_str, add_bos=False, add_eos=True)
        ids = p_ids + a_ids
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        labels = [-100] * len(p_ids) + a_ids
        y = torch.tensor([labels[1:]], dtype=torch.long, device=device)
        kv = pick_kv(gold_pidx, rng)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            logits, _ = forward_with_kv_bank(model, x, kv_bank=kv)
            loss_sum = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        total_nll += float(loss_sum.item())
        total_toks += int((y != -100).sum().item())
        total_bytes += len(ans_str.encode("utf-8"))

    nll_ms_per_q = (time.perf_counter() - t_q) * 1000.0 / max(len(val_raw), 1)

    em_hits = 0
    f1_sum = 0.0
    n_gen = min(gen_limit, len(val_raw))
    for i in range(n_gen):
        q, gold_list, gold_pidx = val_raw[i]
        p_ids = tokenizer.encode(f"\n\nQuestion: {q}\nAnswer:", add_bos=(condition == "off"), add_eos=False)
        kv = pick_kv(gold_pidx, rng_g)
        pred = _generate_kv(model, tokenizer, p_ids, kv, device=device)
        first_line = pred.split("\n", 1)[0].strip()
        em_hits += int(any(g.lower() in first_line.lower() for g in gold_list))
        f1_sum += max(_token_f1(first_line, g) for g in gold_list)

    return {
        "answer_nll": round(total_nll / max(total_toks, 1), 4),
        "answer_bpb": round(total_nll / (math.log(2.0) * max(total_bytes, 1)), 4),
        "exact_match": round(em_hits / max(n_gen, 1), 4),
        "token_f1": round(f1_sum / max(n_gen, 1), 4),
        "nll_ms_per_query": round(nll_ms_per_q, 2),
        "n_eval": len(val_raw),
        "n_gen": n_gen,
    }


@torch.no_grad()
def eval_joint_text_k(
    model: CoreGradLM,
    tokenizer: CoreGradTokenizer,
    val_passages: list[str],
    val_raw: list[tuple[str, list[str], int]],
    device: torch.device,
    *,
    k_passages: int = 4,
    gen_limit: int = 120,
) -> dict[str, float]:
    """Full joint text-in-prompt reference (re-prefills all k passages per query)."""
    rng = random.Random(777)
    rng_g = random.Random(888)
    amp_enabled = device.type == "cuda"
    n_bank = len(val_passages)

    def build_prompt(gold_idx: int, q: str, r: random.Random) -> list[int]:
        chosen = [gold_idx]
        while len(chosen) < k_passages:
            c = r.randrange(n_bank)
            if c != gold_idx and c not in chosen:
                chosen.append(c)
        r.shuffle(chosen)
        joined = "\n\n".join(f"Passage: {val_passages[i]}" for i in chosen)
        return tokenizer.encode(f"{joined}\n\nQuestion: {q}\nAnswer:", add_bos=True, add_eos=False)[:1900]

    total_nll = 0.0
    total_toks = 0
    total_bytes = 0
    t_q = time.perf_counter()
    for q, ans_list, gold_pidx in val_raw:
        p_ids = build_prompt(gold_pidx, q, rng)
        ans_str = f" {ans_list[0]}"
        a_ids = tokenizer.encode(ans_str, add_bos=False, add_eos=True)
        ids = p_ids + a_ids
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        labels = [-100] * len(p_ids) + a_ids
        y = torch.tensor([labels[1:]], dtype=torch.long, device=device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            logits, _ = model(x)
            loss_sum = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        total_nll += float(loss_sum.item())
        total_toks += int((y != -100).sum().item())
        total_bytes += len(ans_str.encode("utf-8"))
    nll_ms_per_q = (time.perf_counter() - t_q) * 1000.0 / max(len(val_raw), 1)

    em_hits = 0
    f1_sum = 0.0
    n_gen = min(gen_limit, len(val_raw))
    for i in range(n_gen):
        q, gold_list, gold_pidx = val_raw[i]
        p_ids = build_prompt(gold_pidx, q, rng_g)
        pred = _generate_kv(model, tokenizer, p_ids, None, device=device)
        first_line = pred.split("\n", 1)[0].strip()
        em_hits += int(any(g.lower() in first_line.lower() for g in gold_list))
        f1_sum += max(_token_f1(first_line, g) for g in gold_list)

    return {
        "answer_nll": round(total_nll / max(total_toks, 1), 4),
        "answer_bpb": round(total_nll / (math.log(2.0) * max(total_bytes, 1)), 4),
        "exact_match": round(em_hits / max(n_gen, 1), 4),
        "token_f1": round(f1_sum / max(n_gen, 1), 4),
        "nll_ms_per_query": round(nll_ms_per_q, 2),
        "n_eval": len(val_raw),
        "n_gen": n_gen,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4B Mechanism 3: Precomputed GQA KV-Bank Injection.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/coregrad_125m_base/best.pt"))
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--max-passage-tokens", type=int, default=180)
    parser.add_argument("--out", type=Path, default=Path("runs/eval/phase4b_kv_memory.json"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    paths = load_local_paths()
    assert paths.hf_cache_dir is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CoreGradTokenizer.from_file(paths.data_dir / "tokenizer" / "tokenizer.json")
    sq_val = paths.hf_cache_dir / "rajpurkar/squad_v2/squad_v2/validation-00000-of-00001.parquet"

    # Match exact val split sampling seed as run_phase4b_bridge.py (after train shuffle)
    rng = random.Random(2026)
    sq_train = paths.hf_cache_dir / "rajpurkar/squad_v2/squad_v2/train-00000-of-00001.parquet"
    tr_rows = [
        r
        for r in pq.read_table(sq_train, columns=["title", "context", "question", "answers"]).to_pylist()
        if r["answers"]["text"] and len(r["context"]) <= 1400
    ]
    rng.shuffle(tr_rows)
    del tr_rows

    val_rows = [
        r
        for r in pq.read_table(sq_val, columns=["title", "context", "question", "answers"]).to_pylist()
        if r["answers"]["text"] and len(r["context"]) <= 1400
    ]
    rng.shuffle(val_rows)
    val_rows = val_rows[: args.n_val]

    ctx_map: dict[str, int] = {}
    val_passages: list[str] = []
    val_raw: list[tuple[str, list[str], int]] = []
    for r in val_rows:
        title = str(r["title"]).replace("_", " ").strip()
        ctx = str(r["context"]).strip()
        full_p = f"{title}\n{ctx}"
        if full_p not in ctx_map:
            ctx_map[full_p] = len(val_passages)
            val_passages.append(full_p)
        pidx = ctx_map[full_p]
        q = str(r["question"]).strip()
        ans_list = [str(a).strip() for a in r["answers"]["text"] if str(a).strip()]
        val_raw.append((q, ans_list, pidx))

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CoreGradLM(ModelConfig(**ckpt["model_config"])).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    print(
        f"[phase4b:kv_memory] checkpoint={args.checkpoint} | val={len(val_raw)} QAs ({len(val_passages)} unique passages)",
        flush=True,
    )
    t_enc = time.perf_counter()
    kv_cache: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    total_kv_tokens = 0
    amp_enabled = device.type == "cuda"
    for p in val_passages:
        ids = tokenizer.encode(f"Passage: {p}\n\n", add_bos=True, add_eos=False)[: args.max_passage_tokens]
        total_kv_tokens += len(ids)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            kv = encode_kv_passage(model, x)
        kv_cache.append([(k.to(torch.bfloat16).cpu(), v.to(torch.bfloat16).cpu()) for k, v in kv])
    enc_s = time.perf_counter() - t_enc
    bytes_per_tok = model.cfg.n_layer * 2 * model.cfg.n_kv_heads * model.cfg.head_dim * 2
    avg_kb_per_passage = (total_kv_tokens / len(val_passages)) * bytes_per_tok / 1024.0
    print(
        f"Precomputed {len(kv_cache)} independent GQA KV banks in {enc_s:.2f}s "
        f"(avg {total_kv_tokens/len(val_passages):.1f} tok/passage, {avg_kb_per_passage:.1f} KB/passage in bf16)",
        flush=True,
    )

    results: dict[str, dict[str, float]] = {}
    for name, cond, k in [
        ("off", "off", 1),
        ("kv_shuffled_k1", "shuffled", 1),
        ("kv_matched_k1", "matched", 1),
        ("kv_shuffled_k4", "shuffled", 4),
        ("kv_matched_k4", "matched", 4),
        ("kv_shuffled_k8", "shuffled", 8),
        ("kv_matched_k8", "matched", 8),
    ]:
        results[name] = eval_kv_condition(model, tokenizer, val_raw, kv_cache, device, condition=cond, k_passages=k)
        r = results[name]
        print(
            f"  {name:18s} | nll={r['answer_nll']:.4f} | bpb={r['answer_bpb']:.4f} | "
            f"EM={r['exact_match']*100:5.1f}% | F1={r['token_f1']*100:5.1f}% | {r['nll_ms_per_query']:5.2f} ms/q",
            flush=True,
        )

    results["joint_text_k4"] = eval_joint_text_k(model, tokenizer, val_passages, val_raw, device, k_passages=4)
    r = results["joint_text_k4"]
    print(
        f"  {'joint_text_k4':18s} | nll={r['answer_nll']:.4f} | bpb={r['answer_bpb']:.4f} | "
        f"EM={r['exact_match']*100:5.1f}% | F1={r['token_f1']*100:5.1f}% | {r['nll_ms_per_query']:5.2f} ms/q",
        flush=True,
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "n_val": len(val_raw),
        "n_unique_passages": len(val_passages),
        "avg_tokens_per_passage": round(total_kv_tokens / len(val_passages), 1),
        "avg_kb_per_passage_bf16": round(avg_kb_per_passage, 1),
        "offline_encode_s": round(enc_s, 2),
        "results": results,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved KV-bank report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
