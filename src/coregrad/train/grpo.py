"""Verifiable RL (GRPO / RLVR) trainer for coregrad Gate 2 research skills.

Optimizes multi-turn tool use, open-domain reading comprehension (SQuAD v2 train),
adversarial abstention, multi-hop retrieval, and exact calculator reasoning using
100% programmatic verifiable rewards (no learned reward model).
"""

from __future__ import annotations

import copy
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from coregrad.config import load_local_paths
from coregrad.model import CoreGradLM, ModelConfig
from coregrad.synthetic import generate_calc_tasks, generate_retrieval_tasks
from coregrad.tokenizer import CoreGradTokenizer
from coregrad.tools import Document, InMemoryBM25Retriever, ToolExecutor
from coregrad.tools.rollout import RolloutTrace, answer_matches, run_agent_rollout
from coregrad.train.monitor import RunLogger

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class GRPOConfig:
    run_name: str = "coregrad_125m_grpo"
    init_from: str = "runs/coregrad_125m_sft_v2/best.pt"
    steps: int = 240
    group_size: int = 6
    lr: float = 1.5e-5
    kl_beta: float = 0.05
    clip_eps: float = 0.2
    grad_clip: float = 1.0
    temperature: float = 0.75
    max_tool_turns: int = 3
    max_new_tokens_per_turn: int = 96
    eval_interval: int = 40
    eval_squad_n: int = 30
    use_squad_v2: bool = True
    seed: int = 2026


@dataclass(frozen=True)
class _EnvTask:
    category: str
    question: str
    expected_answers: tuple[str, ...]
    expected_doc_ids: tuple[str, ...]
    should_abstain: bool
    requires_tool: bool
    documents: tuple[Document, ...]


def _token_f1(pred: str, gold: str) -> float:
    p_toks = _WORD_RE.findall(pred.lower())
    g_toks = _WORD_RE.findall(gold.lower())
    if not p_toks or not g_toks:
        return 0.0
    common: dict[str, int] = {}
    for t in p_toks:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g_toks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    prec = overlap / len(p_toks)
    rec = overlap / len(g_toks)
    return 2.0 * prec * rec / (prec + rec)


def score_research_reward(trace: RolloutTrace, task: _EnvTask) -> float:
    """Programmatic verifiable reward balancing exact span extraction, citations, and abstention."""
    reward = 0.0

    # 1. Tool execution validity
    if trace.tool_calls_total > 0:
        if trace.tool_calls_valid == trace.tool_calls_total:
            reward += 0.15
        else:
            reward -= 0.25
    elif task.requires_tool:
        reward -= 0.25

    abstained = answer_matches(trace.final_answer, "could not find", should_abstain=True)

    # 2. Answer / abstention accuracy (calibrated so "always abstain" has strictly negative EV)
    if task.should_abstain:
        if abstained and trace.tool_calls_valid > 0 and not trace.cited_doc_ids:
            reward += 0.50
        elif abstained:
            reward += 0.20
        else:
            reward -= 0.35
    else:
        exact_hit = any(
            answer_matches(trace.final_answer, ans, should_abstain=False)
            for ans in task.expected_answers
        )
        if exact_hit:
            reward += 1.00
        elif abstained:
            # Heavy penalty for lazy abstention when the answer IS present in the passages
            reward -= 1.00
        else:
            best_f1 = max((_token_f1(trace.final_answer, a) for a in task.expected_answers), default=0.0)
            if best_f1 >= 0.35:
                reward += 0.45 * best_f1

        # 3. Grounded citation reward & strict anti-hallucination penalty
        if trace.cited_doc_ids:
            hallucinated = [c for c in trace.cited_doc_ids if c not in trace.retrieved_doc_ids]
            if hallucinated:
                reward -= 0.50
            elif task.expected_doc_ids and any(c in task.expected_doc_ids for c in trace.cited_doc_ids):
                reward += 0.20
        elif task.expected_doc_ids:
            reward -= 0.10

    return reward


def _load_squad_env_tasks(
    parquet_path: Path, n_ans: int, n_unans: int, seed: int
) -> list[_EnvTask]:
    if not parquet_path.exists():
        return []
    rows = pq.read_table(parquet_path, columns=["title", "context", "question", "answers"]).to_pylist()
    rng = random.Random(seed)
    ans_rows = [r for r in rows if r["answers"]["text"] and len(r["context"]) <= 1300]
    unans_rows = [r for r in rows if not r["answers"]["text"] and len(r["context"]) <= 1300]
    rng.shuffle(ans_rows)
    rng.shuffle(unans_rows)
    pool_ctxs = [(r["title"].replace("_", " "), r["context"]) for r in rows[:1000]]

    tasks: list[_EnvTask] = []
    for r in ans_rows[:n_ans] + unans_rows[:n_unans]:
        title = r["title"].replace("_", " ").strip()
        docs = [Document(doc_id="d001", title=title, text=r["context"].strip())]
        for j in range(2):
            ot, oc = rng.choice(pool_ctxs)
            if ot != title:
                docs.append(Document(doc_id=f"d{j + 2:03d}", title=ot, text=oc[:850]))
        ans_list = tuple(str(a).strip() for a in r["answers"]["text"] if str(a).strip())
        is_unans = len(ans_list) == 0
        tasks.append(
            _EnvTask(
                category="squad_abstain" if is_unans else "squad_qa",
                question=str(r["question"]).strip(),
                expected_answers=("could not find",) if is_unans else ans_list,
                expected_doc_ids=() if is_unans else ("d001",),
                should_abstain=is_unans,
                requires_tool=True,
                documents=tuple(docs),
            )
        )
    rng.shuffle(tasks)
    return tasks


def _sequence_token_logprobs(
    model: CoreGradLM, ids: list[int], device: torch.device, use_bf16: bool = True
) -> torch.Tensor:
    ctx = ids[-model.cfg.max_seq_len :]
    x = torch.tensor([ctx[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ctx[1:]], dtype=torch.long, device=device)
    amp_enabled = use_bf16 and device.type == "cuda"
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
        logits, _ = model(x)
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)[0]


@torch.no_grad()
def _eval_squad_mini(
    model: CoreGradLM,
    tokenizer: CoreGradTokenizer,
    eval_tasks: list[_EnvTask],
    device: torch.device,
) -> tuple[float, float, float, list[dict[str, str]]]:
    if not eval_tasks:
        return 0.0, 0.0, 0.0, []
    model.eval()
    ans_hits = ans_tot = abs_hits = abs_tot = cite_hits = 0
    samples: list[dict[str, str]] = []
    for task in eval_tasks:
        ex = ToolExecutor(InMemoryBM25Retriever(list(task.documents)))
        tr = run_agent_rollout(
            model,
            tokenizer,
            task.question,
            ex,
            device,
            max_tool_turns=3,
            max_new_tokens_per_turn=96,
            temperature=0.0,
        )
        if task.should_abstain:
            abs_tot += 1
            ok = answer_matches(tr.final_answer, "could not find", should_abstain=True)
            abs_hits += int(ok)
        else:
            ans_tot += 1
            ok = any(answer_matches(tr.final_answer, a, should_abstain=False) for a in task.expected_answers)
            ans_hits += int(ok)
            if "d001" in tr.cited_doc_ids and all(c in tr.retrieved_doc_ids for c in tr.cited_doc_ids):
                cite_hits += 1
        if len(samples) < 4:
            samples.append(
                {
                    "prompt": task.question,
                    "completion": (
                        f"[{'PASS' if ok else 'MISS'} | {task.category} | exp={task.expected_answers[0]}]\n"
                        f"Calls: {' | '.join(tr.tool_calls_raw)}\n"
                        f"Cites: {', '.join(tr.cited_doc_ids) or 'none'}\n"
                        f"Ans: {tr.final_answer}"
                    ),
                }
            )
    ans_acc = ans_hits / max(ans_tot, 1)
    abs_acc = abs_hits / max(abs_tot, 1)
    cite_acc = cite_hits / max(ans_tot, 1)
    return ans_acc, abs_acc, cite_acc, samples


def train_grpo(
    cfg: GRPOConfig,
    *,
    tokenizer_path: Path | None = None,
    runs_dir: Path | None = None,
    device: torch.device | None = None,
) -> Path:
    """Run GRPO verifiable RL on SQuAD v2 + synthetic multi-hop retrieval + calc environments."""
    paths = load_local_paths()
    tok_path = tokenizer_path or (paths.data_dir / "tokenizer" / "tokenizer.json")
    run_root = (runs_dir or paths.runs_dir) / cfg.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = CoreGradTokenizer.from_file(tok_path)

    ckpt = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_config"])
    policy = CoreGradLM(model_cfg).to(dev)
    policy.load_state_dict(ckpt["model_state_dict"])

    ref_policy = copy.deepcopy(policy).eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.01)
    logger = RunLogger(
        run_root,
        meta={
            "name": cfg.run_name,
            "stage": "grpo",
            "model": model_cfg.name,
            "total_tokens": cfg.steps * cfg.group_size,
            "steps": cfg.steps,
            "group_size": cfg.group_size,
        },
    )

    # Build verifiable training task pools
    calc_pool = [
        _EnvTask(
            category=t.category,
            question=t.question,
            expected_answers=(t.expected_answer,),
            expected_doc_ids=(),
            should_abstain=False,
            requires_tool=t.requires_tool,
            documents=(),
        )
        for t in generate_calc_tasks(max(cfg.steps, 64), seed=cfg.seed + 10)
    ]
    ret_pool = [
        _EnvTask(
            category=t.category,
            question=t.question,
            expected_answers=(t.expected_answer,),
            expected_doc_ids=t.expected_doc_ids,
            should_abstain=t.should_abstain,
            requires_tool=True,
            documents=tuple(t.documents),
        )
        for t in generate_retrieval_tasks(max(cfg.steps, 64), seed=cfg.seed + 20, split="train")
    ]

    squad_train_pool: list[_EnvTask] = []
    squad_val_tasks: list[_EnvTask] = []
    if cfg.use_squad_v2 and paths.hf_cache_dir is not None:
        sq_train = paths.hf_cache_dir / "rajpurkar/squad_v2/squad_v2/train-00000-of-00001.parquet"
        sq_val = paths.hf_cache_dir / "rajpurkar/squad_v2/squad_v2/validation-00000-of-00001.parquet"
        # 70% answerable / 30% unanswerable prevents collapse into degenerate refusal
        squad_train_pool = _load_squad_env_tasks(sq_train, n_ans=560, n_unans=240, seed=cfg.seed + 30)
        squad_val_tasks = _load_squad_env_tasks(
            sq_val, n_ans=cfg.eval_squad_n, n_unans=cfg.eval_squad_n, seed=42
        )
        print(
            f"[grpo] Loaded SQuAD v2 envs: train={len(squad_train_pool)}, val={len(squad_val_tasks)}",
            flush=True,
        )

    rng = random.Random(cfg.seed)
    t_start = time.perf_counter()
    best_score = -1.0

    def run_eval_and_save(cur_step: int) -> None:
        nonlocal best_score
        if not squad_val_tasks:
            return
        ans_acc, abs_acc, cite_acc, samples = _eval_squad_mini(policy, tokenizer, squad_val_tasks, dev)
        # Harmonic mean prevents degenerate "always abstain" or "never abstain" solutions from winning best.pt
        combo = (2.0 * ans_acc * abs_acc / (ans_acc + abs_acc)) if (ans_acc + abs_acc) > 0 else 0.0
        logger.log_eval(
            step=cur_step,
            tokens_seen=cur_step * cfg.group_size,
            mean_bpb=1.0 - combo,
            domain_bpb={"squad_ans": ans_acc, "squad_abstain": abs_acc, "squad_cite": cite_acc},
            grammar_acc=ans_acc,
            completion_acc=abs_acc,
        )
        if samples:
            logger.log_samples(step=cur_step, tokens_seen=cur_step * cfg.group_size, samples=samples)
        print(
            f"[grpo eval step={cur_step:4d}] squad_ans={ans_acc*100:5.1f}% | "
            f"squad_abstain={abs_acc*100:5.1f}% | squad_cite={cite_acc*100:5.1f}% | hmean={combo*100:5.1f}%",
            flush=True,
        )
        payload: dict[str, Any] = {
            "step": cur_step,
            "squad_combo": combo,
            "model_config": asdict(policy.cfg),
            "grpo_config": asdict(cfg),
            "model_state_dict": policy.state_dict(),
        }
        torch.save(payload, run_root / "latest.pt")
        if combo >= best_score:
            best_score = combo
            torch.save(payload, run_root / "best.pt")

    run_eval_and_save(0)

    skipped_zero_var = 0
    for step in range(1, cfg.steps + 1):
        u = rng.random()
        if squad_train_pool and u < 0.65:
            task = rng.choice(squad_train_pool)
        elif u < 0.85:
            task = rng.choice(ret_pool)
        else:
            task = rng.choice(calc_pool)

        executor = ToolExecutor(InMemoryBM25Retriever(list(task.documents)) if task.documents else None)

        traces: list[RolloutTrace] = []
        rewards: list[float] = []
        old_logps: list[torch.Tensor] = []

        policy.eval()
        for _ in range(cfg.group_size):
            tr = run_agent_rollout(
                policy,
                tokenizer,
                task.question,
                executor,
                dev,
                max_tool_turns=cfg.max_tool_turns,
                max_new_tokens_per_turn=cfg.max_new_tokens_per_turn,
                temperature=cfg.temperature,
            )
            r = score_research_reward(tr, task)
            traces.append(tr)
            rewards.append(r)
            with torch.no_grad():
                old_logps.append(_sequence_token_logprobs(policy, tr.full_ids, dev).detach())

        r_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean_r = float(r_tensor.mean().item())
        std_r = float(r_tensor.std().item())

        if std_r < 1e-6:
            skipped_zero_var += 1
            gn = 0.0
            loss_val = 0.0
        else:
            adv = (r_tensor - mean_r) / (std_r + 1e-6)
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            total_obj = torch.tensor(0.0, device=dev)
            valid_groups = 0

            for tr, old_lp, a_val in zip(traces, old_logps, adv.tolist(), strict=True):
                if len(tr.full_ids) < 3:
                    continue
                ctx_len = min(len(tr.full_ids), policy.cfg.max_seq_len)
                mask_slice = tr.assistant_mask[-ctx_len:][1:]
                mask = torch.tensor(mask_slice, dtype=torch.bool, device=dev)
                if not mask.any():
                    continue
                new_lp = _sequence_token_logprobs(policy, tr.full_ids, dev)
                with torch.no_grad():
                    ref_lp = _sequence_token_logprobs(ref_policy, tr.full_ids, dev)

                ratio = torch.exp((new_lp - old_lp).clamp(-10.0, 10.0))
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
                pg_loss = -torch.minimum(ratio * a_val, clipped * a_val)
                kl = torch.exp((ref_lp - new_lp).clamp(-10.0, 10.0)) - (ref_lp - new_lp) - 1.0
                token_loss = (pg_loss + cfg.kl_beta * kl)[mask].mean()
                total_obj = total_obj + token_loss
                valid_groups += 1

            if valid_groups > 0:
                loss = total_obj / valid_groups
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip).item())
                optimizer.step()
                loss_val = float(loss.item())
            else:
                gn = 0.0
                loss_val = 0.0

        if step % 5 == 0 or step == cfg.steps:
            vram = torch.cuda.max_memory_allocated(dev) / (1024**3) if dev.type == "cuda" else 0.0
            logger.log_step(
                step=step,
                tokens_seen=step * cfg.group_size,
                train_loss=loss_val,
                lr=cfg.lr,
                grad_norm=gn,
                tok_per_sec=mean_r,
                vram_gb=vram,
                elapsed_s=time.perf_counter() - t_start,
            )
            print(
                f"[grpo] step {step:4d}/{cfg.steps} | {task.category:14s} | "
                f"mean_r={mean_r:+.3f} (std={std_r:.2f}) | loss={loss_val:.4f} | gn={gn:.2f} | "
                f"skip0={skipped_zero_var}",
                flush=True,
            )

        if cfg.eval_interval > 0 and (step % cfg.eval_interval == 0 or step == cfg.steps):
            run_eval_and_save(step)

    if not (run_root / "best.pt").exists():
        torch.save(
            {
                "step": cfg.steps,
                "model_config": asdict(policy.cfg),
                "grpo_config": asdict(cfg),
                "model_state_dict": policy.state_dict(),
            },
            run_root / "best.pt",
        )
    print(f"\nFinished GRPO {cfg.run_name} in {(time.perf_counter()-t_start)/60:.1f} min", flush=True)
    return run_root
