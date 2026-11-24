"""Lightweight Verifiable RL (GRPO / RLVR) trainer for tool-use and grounded retrieval."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from coregrad.config import load_local_paths
from coregrad.model import CoreGradLM, ModelConfig
from coregrad.synthetic import generate_calc_tasks, generate_retrieval_tasks
from coregrad.tokenizer import CoreGradTokenizer
from coregrad.tools.executor import ToolExecutor
from coregrad.tools.retriever import InMemoryBM25Retriever
from coregrad.tools.rollout import run_agent_rollout, score_rollout_reward
from coregrad.train.monitor import RunLogger


@dataclass
class GRPOConfig:
    run_name: str = "coregrad_125m_grpo"
    init_from: str = "runs/coregrad_125m_sft/best.pt"
    steps: int = 200
    group_size: int = 4
    lr: float = 2.0e-5
    kl_beta: float = 0.04
    clip_eps: float = 0.2
    grad_clip: float = 1.0
    temperature: float = 0.8
    max_tool_turns: int = 3
    max_new_tokens_per_turn: int = 128
    seed: int = 2026


def _sequence_token_logprobs(
    model: CoreGradLM, ids: list[int], device: torch.device
) -> torch.Tensor:
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    logits, _ = model(x)
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)[0]


def train_grpo(
    cfg: GRPOConfig,
    *,
    tokenizer_path: Path | None = None,
    runs_dir: Path | None = None,
    device: torch.device | None = None,
) -> Path:
    """Run Group Relative Policy Optimization on verifiable calc & retrieval environments."""
    paths = load_local_paths()
    tok_path = tokenizer_path or (paths.data_dir / "tokenizer" / "tokenizer.json")
    run_root = (runs_dir or paths.runs_dir) / cfg.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CoreGradTokenizer.from_file(tok_path)

    ckpt = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_config"])
    policy = CoreGradLM(model_cfg).to(dev)
    policy.load_state_dict(ckpt["model_state_dict"])

    ref_policy = copy.deepcopy(policy).eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.01)
    logger = RunLogger(run_root, meta={"name": cfg.run_name, "stage": "grpo", "steps": cfg.steps})

    calc_pool = generate_calc_tasks(max(cfg.steps, 64), seed=cfg.seed + 10)
    ret_pool = generate_retrieval_tasks(max(cfg.steps, 64), seed=cfg.seed + 20, split="train")
    rng = random.Random(cfg.seed)

    for step in range(1, cfg.steps + 1):
        use_ret = (step % 2 == 0)
        if use_ret:
            task_r = rng.choice(ret_pool)
            question = task_r.question
            executor = ToolExecutor(InMemoryBM25Retriever(task_r.documents))
            expected_ans = task_r.expected_answer
            expected_docs = task_r.expected_doc_ids
            should_abstain = task_r.should_abstain
            requires_tool = True
        else:
            task_c = rng.choice(calc_pool)
            question = task_c.question
            executor = ToolExecutor(None)
            expected_ans = task_c.expected_answer
            expected_docs = ()
            should_abstain = False
            requires_tool = task_c.requires_tool

        traces = []
        rewards: list[float] = []
        old_logps: list[torch.Tensor] = []

        policy.eval()
        for _ in range(cfg.group_size):
            tr = run_agent_rollout(
                policy,
                tokenizer,
                question,
                executor,
                dev,
                max_tool_turns=cfg.max_tool_turns,
                max_new_tokens_per_turn=cfg.max_new_tokens_per_turn,
                temperature=cfg.temperature,
            )
            r = score_rollout_reward(
                tr,
                expected_answer=expected_ans,
                expected_doc_ids=expected_docs,
                should_abstain=should_abstain,
                requires_tool=requires_tool,
            )
            traces.append(tr)
            rewards.append(r)
            with torch.no_grad():
                old_logps.append(_sequence_token_logprobs(policy, tr.full_ids, dev).detach())

        r_tensor = torch.tensor(rewards, dtype=torch.float32)
        adv = (r_tensor - r_tensor.mean()) / (r_tensor.std() + 1e-6)

        policy.train()
        optimizer.zero_grad(set_to_none=True)
        total_obj = torch.tensor(0.0, device=dev)
        valid_groups = 0

        for tr, old_lp, a_val in zip(traces, old_logps, adv.tolist(), strict=True):
            if len(tr.full_ids) < 3:
                continue
            mask = torch.tensor(tr.assistant_mask[1:], dtype=torch.bool, device=dev)
            if not mask.any():
                continue
            new_lp = _sequence_token_logprobs(policy, tr.full_ids, dev)
            with torch.no_grad():
                ref_lp = _sequence_token_logprobs(ref_policy, tr.full_ids, dev)

            ratio = torch.exp((new_lp - old_lp).clamp(-10.0, 10.0))
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
            pg_loss = -torch.minimum(ratio * a_val, clipped * a_val)
            kl = torch.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1.0
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

        mean_r = float(r_tensor.mean().item())
        if step % 5 == 0 or step == cfg.steps:
            logger.log_step(
                step=step,
                tokens_seen=step * cfg.group_size,
                train_loss=loss_val,
                lr=cfg.lr,
                grad_norm=gn,
                tok_per_sec=mean_r,
                vram_gb=torch.cuda.max_memory_allocated(dev) / (1024**3) if dev.type == "cuda" else 0.0,
                elapsed_s=float(step),
            )
            print(f"[grpo] step {step:4d}/{cfg.steps} | mean_reward={mean_r:+.3f} | loss={loss_val:.4f}", flush=True)

    torch.save(
        {
            "step": cfg.steps,
            "model_config": asdict(policy.cfg),
            "grpo_config": asdict(cfg),
            "model_state_dict": policy.state_dict(),
        },
        run_root / "best.pt",
    )
    return run_root
