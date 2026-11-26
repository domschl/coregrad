#!/usr/bin/env python3
"""Phase 4B Tests 1, 2 & 3: Gated Middle-Layer Latent Memory Bridge (Variant A vs Variant B).

Trains zero-initialized gated cross-attention adapters on frozen coregrad-125m_base:
- Variant A ('embeddinggemma'): 1 EmbeddingGemma 2 (768d) vector per passage -> 4 memory slots
- Variant B ('retro')         : 8 pooled latent tokens per passage from frozen bottom 6 layers

Evaluates strictly on held-out SQuAD v2 validation tasks with ZERO passage text in the prompt:
- Test 1: Causal control (OFF vs SHUFFLED vs MATCHED)
- Test 2: Answer NLL / BPB drop from latent memory alone
- Test 3: 20-passage overflow test (~4,500 text tokens compressed to 0 text prompt tokens!)
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
from coregrad.model.memory_bridge import CoreGradMemoryBridge, MemoryBridgeConfig
from coregrad.tokenizer import CoreGradTokenizer
from coregrad.tools import EmbeddingGemma2Embedder
from coregrad.train.monitor import RunLogger

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


def _encode_qa_pair(
    tokenizer: CoreGradTokenizer, question: str, answer: str, max_len: int = 128
) -> tuple[list[int], list[int], int]:
    """Encode prompt='Question: ...\nAnswer:' + target=' <answer><|eos|>' with prompt masked."""
    prompt_str = f"Question: {question}\nAnswer:"
    ans_str = f" {answer}"
    p_ids = tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
    a_ids = tokenizer.encode(ans_str, add_bos=False, add_eos=True)
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    ans_bytes = len(ans_str.encode("utf-8"))
    return ids, labels, ans_bytes


@torch.no_grad()
def _precompute_passage_memories(
    mode: str,
    passages: list[str],
    bridge: CoreGradMemoryBridge,
    tokenizer: CoreGradTokenizer,
    device: torch.device,
    retro_slots: int = 8,
    max_passage_tokens: int = 160,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Precompute compressed or unpooled latent memory + optional bool mask for every unique passage."""
    if mode == "embeddinggemma":
        embedder = EmbeddingGemma2Embedder(mrl_dim=768, device=str(device))
        vecs = embedder.encode_documents(passages)  # (N, 768)
        del embedder
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return torch.from_numpy(vecs).unsqueeze(1), None  # (N, 1, 768)

    pad_id = tokenizer.pad_id
    out_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    bsz = 32
    for start in range(0, len(passages), bsz):
        batch_txt = passages[start : start + bsz]
        enc = [tokenizer.encode(t, add_bos=True, add_eos=False)[:max_passage_tokens] for t in batch_txt]
        lengths = [len(ids) for ids in enc]
        mlen = max_passage_tokens if mode == "retro_tokens" else max(lengths)
        x = torch.full((len(enc), mlen), pad_id, dtype=torch.long, device=device)
        m = torch.zeros((len(enc), mlen), dtype=torch.bool)
        for i, ids in enumerate(enc):
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            m[i, : len(ids)] = True
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if mode == "retro_tokens":
                states = bridge.encode_retro_token_states(x)
                out_chunks.append(states.to(torch.bfloat16).cpu())
                mask_chunks.append(m)
            else:
                pooled = bridge.encode_retro_passages(x, slots_per_passage=retro_slots, lengths=lengths)
                out_chunks.append(pooled.float().cpu())
    mem_bank = torch.cat(out_chunks, dim=0)
    mask_bank = torch.cat(mask_chunks, dim=0) if mask_chunks else None
    return mem_bank, mask_bank


def _collate_qa(
    items: list[tuple[list[int], list[int], int, int]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], int]:
    max_l = max(len(it[0]) for it in items)
    bsz = len(items)
    x = torch.full((bsz, max_l - 1), pad_id, dtype=torch.long)
    y = torch.full((bsz, max_l - 1), -100, dtype=torch.long)
    p_indices: list[int] = []
    total_bytes = 0
    for i, (ids, labels, pidx, n_bytes) in enumerate(items):
        n = len(ids) - 1
        x[i, :n] = torch.tensor(ids[:n], dtype=torch.long)
        y[i, :n] = torch.tensor(labels[1 : n + 1], dtype=torch.long)
        p_indices.append(pidx)
        total_bytes += n_bytes
    return x, y, p_indices, total_bytes


def _build_memory_batch(
    mem_bank: torch.Tensor,
    mask_bank: torch.Tensor | None,
    p_indices: list[int],
    k_passages: int,
    rng: random.Random,
    condition: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Assemble (B, k_passages * S, D) memory tensor + optional bool mask under OFF / SHUFFLED / MATCHED."""
    if condition == "off":
        return None, None
    n_bank = mem_bank.shape[0]
    batch_mems: list[torch.Tensor] = []
    batch_masks: list[torch.Tensor] = []
    for gold_idx in p_indices:
        chosen: list[int] = []
        if condition == "matched":
            chosen.append(gold_idx)
        while len(chosen) < k_passages:
            cand = rng.randrange(n_bank)
            if cand != gold_idx and cand not in chosen:
                chosen.append(cand)
        rng.shuffle(chosen)
        batch_mems.append(mem_bank[chosen].reshape(-1, mem_bank.shape[-1]))
        if mask_bank is not None:
            batch_masks.append(mask_bank[chosen].reshape(-1))
    mem = torch.stack(batch_mems, dim=0)
    mask = torch.stack(batch_masks, dim=0) if mask_bank is not None else None
    return mem, mask


@torch.no_grad()
def _generate_answer(
    bridge: CoreGradMemoryBridge,
    tokenizer: CoreGradTokenizer,
    question: str,
    mem: torch.Tensor | None,
    mem_mask: torch.Tensor | None,
    device: torch.device,
    max_new_tokens: int = 24,
    prompt_prefix: str = "",
) -> str:
    bridge.eval()
    ids = tokenizer.encode(f"{prompt_prefix}Question: {question}\nAnswer:", add_bos=True, add_eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    eos_id = tokenizer.eos_id
    amp_enabled = device.type == "cuda"
    gen: list[int] = []
    for _ in range(max_new_tokens):
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            logits, _ = bridge(x, memory=mem, memory_mask=mem_mask)
        nxt = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        if nxt == eos_id:
            break
        gen.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


@torch.no_grad()
def evaluate_text_prompt_baseline(
    bridge: CoreGradMemoryBridge,
    tokenizer: CoreGradTokenizer,
    val_passages: list[str],
    val_raw: list[tuple[str, list[str], int]],
    device: torch.device,
    gen_limit: int = 120,
) -> dict[str, float]:
    """Upper-bound reference: gold passage placed directly in text prompt (memory=OFF)."""
    bridge.eval()
    amp_enabled = device.type == "cuda"
    total_nll = 0.0
    total_toks = 0
    total_bytes = 0
    items: list[tuple[list[int], list[int], int, int]] = []
    for q, ans_list, pidx in val_raw:
        prompt_str = f"Passage: {val_passages[pidx]}\n\nQuestion: {q}\nAnswer:"
        ans_str = f" {ans_list[0]}"
        p_ids = tokenizer.encode(prompt_str, add_bos=True, add_eos=False)[:440]
        a_ids = tokenizer.encode(ans_str, add_bos=False, add_eos=True)
        ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids
        items.append((ids, labels, pidx, len(ans_str.encode("utf-8"))))

    bsz = 16
    for start in range(0, len(items), bsz):
        chunk = items[start : start + bsz]
        x, y, _, n_bytes = _collate_qa(chunk, tokenizer.pad_id)
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            logits, _ = bridge(x, memory=None)
            loss_sum = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        total_nll += float(loss_sum.item())
        total_toks += int((y != -100).sum().item())
        total_bytes += n_bytes

    em_hits = 0
    f1_sum = 0.0
    n_gen = min(gen_limit, len(val_raw))
    for i in range(n_gen):
        q, gold_list, pidx = val_raw[i]
        pred = _generate_answer(
            bridge, tokenizer, q, None, None, device=device, prompt_prefix=f"Passage: {val_passages[pidx]}\n\n"
        )
        em_hits += int(any(g.lower() in pred.lower() for g in gold_list))
        f1_sum += max(_token_f1(pred, g) for g in gold_list)

    return {
        "answer_nll": round(total_nll / max(total_toks, 1), 4),
        "answer_bpb": round(total_nll / (math.log(2.0) * max(total_bytes, 1)), 4),
        "exact_match": round(em_hits / max(n_gen, 1), 4),
        "token_f1": round(f1_sum / max(n_gen, 1), 4),
        "n_eval": len(items),
        "n_gen": n_gen,
    }


@torch.no_grad()
def evaluate_bridge(
    bridge: CoreGradMemoryBridge,
    tokenizer: CoreGradTokenizer,
    val_items: list[tuple[list[int], list[int], int, int]],
    val_raw: list[tuple[str, list[str], int]],
    val_mem_bank: torch.Tensor,
    val_mask_bank: torch.Tensor | None,
    device: torch.device,
    *,
    condition: str,
    k_passages: int,
    gen_limit: int = 120,
) -> dict[str, float]:
    bridge.eval()
    rng = random.Random(777)
    amp_enabled = device.type == "cuda"
    total_nll = 0.0
    total_toks = 0
    total_bytes = 0

    bsz = 16
    for start in range(0, len(val_items), bsz):
        chunk = val_items[start : start + bsz]
        x, y, p_idx, n_bytes = _collate_qa(chunk, tokenizer.pad_id)
        x, y = x.to(device), y.to(device)
        mem, mem_mask = _build_memory_batch(
            val_mem_bank, val_mask_bank, p_idx, k_passages=k_passages, rng=rng, condition=condition
        )
        if mem is not None:
            mem = mem.to(device)
        if mem_mask is not None:
            mem_mask = mem_mask.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            logits, _ = bridge(x, memory=mem, memory_mask=mem_mask)
            # Sum exact cross-entropy over non-masked answer tokens
            loss_sum = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        n_tok = int((y != -100).sum().item())
        total_nll += float(loss_sum.item())
        total_toks += n_tok
        total_bytes += n_bytes

    mean_nll = total_nll / max(total_toks, 1)
    bpb = total_nll / (math.log(2.0) * max(total_bytes, 1))

    # Generative exact match + token-F1 on first gen_limit val items
    em_hits = 0
    f1_sum = 0.0
    n_gen = min(gen_limit, len(val_raw))
    rng_g = random.Random(888)
    for i in range(n_gen):
        q, gold_list, gold_pidx = val_raw[i]
        mem, mem_mask = _build_memory_batch(
            val_mem_bank, val_mask_bank, [gold_pidx], k_passages=k_passages, rng=rng_g, condition=condition
        )
        if mem is not None:
            mem = mem.to(device)
        if mem_mask is not None:
            mem_mask = mem_mask.to(device)
        pred = _generate_answer(bridge, tokenizer, q, mem, mem_mask, device=device)
        em_hits += int(any(g.lower() in pred.lower() for g in gold_list))
        f1_sum += max(_token_f1(pred, g) for g in gold_list)

    return {
        "answer_nll": round(mean_nll, 4),
        "answer_bpb": round(bpb, 4),
        "exact_match": round(em_hits / max(n_gen, 1), 4),
        "token_f1": round(f1_sum / max(n_gen, 1), 4),
        "n_eval": len(val_items),
        "n_gen": n_gen,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train & test Phase 4B Latent Memory Bridge.")
    parser.add_argument("--mode", choices=["embeddinggemma", "retro", "retro_tokens"], required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/coregrad_125m_base/best.pt"))
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--k-train", type=int, default=4, help="Passages in memory during training")
    parser.add_argument("--retro-slots", type=int, default=8)
    parser.add_argument("--max-passage-tokens", type=int, default=160)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--unfreeze-top", action="store_true", help="Co-adapt upper blocks (>= min bridge layer)")
    parser.add_argument("--mem-dropout", type=float, default=0.0, help="Prob of memory=None step during training")
    parser.add_argument("--eval-text-baseline", action="store_true", help="Also score text-in-prompt upper bound")
    parser.add_argument("--run-tag", type=str, default=None, help="Optional suffix for run_name and output report")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    paths = load_local_paths()
    assert paths.hf_cache_dir is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = CoreGradTokenizer.from_file(paths.data_dir / "tokenizer" / "tokenizer.json")
    sq_train = paths.hf_cache_dir / "rajpurkar/squad_v2/squad_v2/train-00000-of-00001.parquet"
    sq_val = paths.hf_cache_dir / "rajpurkar/squad_v2/squad_v2/validation-00000-of-00001.parquet"

    rng = random.Random(2026)

    def load_split(path: Path, limit: int) -> tuple[list[str], list[tuple[list[int], list[int], int, int]], list[tuple[str, list[str], int]]]:
        rows = [
            r
            for r in pq.read_table(path, columns=["title", "context", "question", "answers"]).to_pylist()
            if r["answers"]["text"] and len(r["context"]) <= 1400
        ]
        rng.shuffle(rows)
        rows = rows[:limit]
        ctx_map: dict[str, int] = {}
        passages: list[str] = []
        items: list[tuple[list[int], list[int], int, int]] = []
        raw: list[tuple[str, list[str], int]] = []
        for r in rows:
            title = str(r["title"]).replace("_", " ").strip()
            ctx = str(r["context"]).strip()
            full_p = f"{title}\n{ctx}"
            if full_p not in ctx_map:
                ctx_map[full_p] = len(passages)
                passages.append(full_p)
            pidx = ctx_map[full_p]
            q = str(r["question"]).strip()
            ans_list = [str(a).strip() for a in r["answers"]["text"] if str(a).strip()]
            ids, labels, n_bytes = _encode_qa_pair(tokenizer, q, ans_list[0])
            items.append((ids, labels, pidx, n_bytes))
            raw.append((q, ans_list, pidx))
        return passages, items, raw

    tr_passages, tr_items, _ = load_split(sq_train, args.n_train)
    val_passages, val_items, val_raw = load_split(sq_val, args.n_val)
    print(
        f"[phase4b:{args.mode}] train={len(tr_items):,} QAs ({len(tr_passages):,} unique passages) | "
        f"val={len(val_items)} QAs ({len(val_passages)} unique passages)",
        flush=True,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_model = CoreGradLM(ModelConfig(**ckpt["model_config"])).to(device)
    base_model.load_state_dict(ckpt["model_state_dict"])

    bridge_cfg = MemoryBridgeConfig(
        bridge_layers=(9, 12, 15),
        retro_slots_per_passage=args.retro_slots,
        freeze_base=True,
    )
    bridge = CoreGradMemoryBridge(base_model, bridge_cfg, memory_mode=args.mode).to(device)
    top_params: list[torch.nn.Parameter] = []
    if args.unfreeze_top:
        start_layer = min(bridge_cfg.bridge_layers)
        for idx in range(start_layer, len(bridge.base.blocks)):
            for p in bridge.base.blocks[idx].parameters():
                p.requires_grad_(True)
                top_params.append(p)
        for p in bridge.base.final_norm.parameters():
            p.requires_grad_(True)
            top_params.append(p)

    bridge_params = [p for p in bridge.parameters() if p.requires_grad and not any(p is tp for tp in top_params)]
    tag = args.run_tag or (f"{args.mode}_coadapt" if args.unfreeze_top else args.mode)
    print(
        f"[phase4b:{tag}] Base={base_model.num_parameters()/1e6:.2f}M | "
        f"Trainable={bridge.trainable_parameters()/1e6:.2f}M "
        f"(bridge={sum(p.numel() for p in bridge_params)/1e6:.2f}M, top={sum(p.numel() for p in top_params)/1e6:.2f}M)",
        flush=True,
    )

    text_baseline: dict[str, float] | None = None
    if args.eval_text_baseline:
        text_baseline = evaluate_text_prompt_baseline(bridge, tokenizer, val_passages, val_raw, device, gen_limit=120)
        print(f"[text-in-prompt upper bound (base model)] {text_baseline}", flush=True)

    print(f"Precomputing latent memory banks ({args.mode})...", flush=True)
    tr_mem_bank, tr_mask_bank = _precompute_passage_memories(
        args.mode, tr_passages, bridge, tokenizer, device,
        retro_slots=args.retro_slots, max_passage_tokens=args.max_passage_tokens,
    )
    val_mem_bank, val_mask_bank = _precompute_passage_memories(
        args.mode, val_passages, bridge, tokenizer, device,
        retro_slots=args.retro_slots, max_passage_tokens=args.max_passage_tokens,
    )
    print(f"Memory bank shapes: train={tuple(tr_mem_bank.shape)}, val={tuple(val_mem_bank.shape)}", flush=True)

    run_name = f"phase4b_bridge_{tag}"
    run_root = paths.runs_dir / run_name
    logger = RunLogger(
        run_root,
        meta={
            "name": run_name,
            "mode": args.mode,
            "unfreeze_top": args.unfreeze_top,
            "mem_dropout": args.mem_dropout,
            "trainable_params": bridge.trainable_parameters(),
            "steps": args.steps,
            "total_tokens": args.steps * args.batch_size,
        },
    )

    param_groups = [{"params": bridge_params, "lr": args.lr, "lr_scale": 1.0}]
    if top_params:
        param_groups.append({"params": top_params, "lr": args.lr * 0.05, "lr_scale": 0.05})
    opt = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.02)
    amp_enabled = device.type == "cuda"

    # Verify step-0 identity (OFF == MATCHED before any training)
    off_0 = evaluate_bridge(
        bridge, tokenizer, val_items, val_raw, val_mem_bank, val_mask_bank, device,
        condition="off", k_passages=args.k_train, gen_limit=60,
    )
    print(f"[step 0 baseline OFF] {off_0}", flush=True)

    tr_rng = random.Random(1337)
    t_train = time.perf_counter()
    for step in range(1, args.steps + 1):
        bridge.train()
        # Cosine LR decay with 5% warmup
        warm = max(1, int(args.steps * 0.05))
        if step <= warm:
            lr = args.lr * (step / warm)
        else:
            prog = (step - warm) / max(1, args.steps - warm)
            lr = args.lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * prog)))
        for pg in opt.param_groups:
            pg["lr"] = lr * pg["lr_scale"]

        batch = [tr_items[tr_rng.randrange(len(tr_items))] for _ in range(args.batch_size)]
        x, y, p_idx, _ = _collate_qa(batch, tokenizer.pad_id)
        drop_mem = args.mem_dropout > 0.0 and tr_rng.random() < args.mem_dropout
        if drop_mem:
            mem, mem_mask = None, None
        else:
            k_cur = tr_rng.choice([1, 2, args.k_train])
            mem, mem_mask = _build_memory_batch(
                tr_mem_bank, tr_mask_bank, p_idx, k_passages=k_cur, rng=tr_rng, condition="matched"
            )
        x, y = x.to(device), y.to(device)
        if mem is not None:
            mem = mem.to(device)
        if mem_mask is not None:
            mem_mask = mem_mask.to(device)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            _, loss = bridge(x, y, memory=mem, memory_mask=mem_mask)
            assert loss is not None
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0).item())
        opt.step()

        if step % 100 == 0 or step == args.steps:
            vram = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
            logger.log_step(
                step=step,
                tokens_seen=step * args.batch_size,
                train_loss=float(loss.item()),
                lr=lr,
                grad_norm=gn,
                tok_per_sec=float(step * args.batch_size) / max(time.perf_counter() - t_train, 1e-6),
                vram_gb=vram,
                elapsed_s=time.perf_counter() - t_train,
            )
            gates = {k: round(v, 3) for k, v in bridge.gate_values().items()}
            print(
                f"[{tag}] step {step:4d}/{args.steps} | loss={float(loss.item()):.4f} | "
                f"lr={lr:.2e} | gates={gates}",
                flush=True,
            )

        if step % 400 == 0 or step == args.steps:
            m_res = evaluate_bridge(
                bridge, tokenizer, val_items, val_raw, val_mem_bank, val_mask_bank, device,
                condition="matched", k_passages=args.k_train, gen_limit=60,
            )
            s_res = evaluate_bridge(
                bridge, tokenizer, val_items, val_raw, val_mem_bank, val_mask_bank, device,
                condition="shuffled", k_passages=args.k_train, gen_limit=60,
            )
            logger.log_eval(
                step=step,
                tokens_seen=step * args.batch_size,
                mean_bpb=m_res["answer_bpb"],
                domain_bpb={"matched_bpb": m_res["answer_bpb"], "shuffled_bpb": s_res["answer_bpb"]},
                grammar_acc=m_res["exact_match"],
                completion_acc=m_res["token_f1"],
            )
            print(
                f"[eval step={step}] MATCHED bpb={m_res['answer_bpb']:.3f} em={m_res['exact_match']*100:.1f}% f1={m_res['token_f1']*100:.1f}% | "
                f"SHUFFLED bpb={s_res['answer_bpb']:.3f} em={s_res['exact_match']*100:.1f}% f1={s_res['token_f1']*100:.1f}%",
                flush=True,
            )

    # Final comprehensive causal & 20-passage overflow evaluation
    print(f"\n=== FINAL FALSIFIABLE EVALUATION ({tag}) ===", flush=True)
    res_off = evaluate_bridge(
        bridge, tokenizer, val_items, val_raw, val_mem_bank, val_mask_bank, device, condition="off", k_passages=args.k_train, gen_limit=120
    )
    res_shuf_4 = evaluate_bridge(
        bridge, tokenizer, val_items, val_raw, val_mem_bank, val_mask_bank, device, condition="shuffled", k_passages=4, gen_limit=120
    )
    res_match_1 = evaluate_bridge(
        bridge, tokenizer, val_items, val_raw, val_mem_bank, val_mask_bank, device, condition="matched", k_passages=1, gen_limit=120
    )
    res_match_4 = evaluate_bridge(
        bridge, tokenizer, val_items, val_raw, val_mem_bank, val_mask_bank, device, condition="matched", k_passages=4, gen_limit=120
    )
    res_match_20 = evaluate_bridge(
        bridge, tokenizer, val_items, val_raw, val_mem_bank, val_mask_bank, device, condition="matched", k_passages=20, gen_limit=120
    )
    res_shuf_20 = evaluate_bridge(
        bridge, tokenizer, val_items, val_raw, val_mem_bank, val_mask_bank, device, condition="shuffled", k_passages=20, gen_limit=120
    )

    report = {
        "mode": args.mode,
        "tag": tag,
        "unfreeze_top": args.unfreeze_top,
        "mem_dropout": args.mem_dropout,
        "checkpoint": str(args.checkpoint),
        "trainable_params": bridge.trainable_parameters(),
        "gates": bridge.gate_values(),
        "text_in_prompt_baseline": text_baseline,
        "results": {
            "off": res_off,
            "shuffled_k4": res_shuf_4,
            "matched_k1": res_match_1,
            "matched_k4": res_match_4,
            "shuffled_k20": res_shuf_20,
            "matched_k20": res_match_20,
        },
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    for name, r in report["results"].items():
        print(
            f"  {name:14s} | nll={r['answer_nll']:.4f} | bpb={r['answer_bpb']:.4f} | "
            f"EM={r['exact_match']*100:5.1f}% | F1={r['token_f1']*100:5.1f}%"
        )

    out_path = args.out or Path(f"runs/eval/phase4b_bridge_{tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(
        {
            "mode": args.mode,
            "bridge_cfg": bridge_cfg.__dict__,
            "cross_attns": bridge.cross_attns.state_dict(),
            "mem_proj": bridge.mem_proj.state_dict(),
        },
        run_root / "bridge_weights.pt",
    )
    print(f"Saved report -> {out_path} and weights -> {run_root / 'bridge_weights.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
