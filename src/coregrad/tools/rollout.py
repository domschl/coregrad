"""Interactive multi-turn agent rollout engine and verifiable reward scoring for Gate 2 & GRPO."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from coregrad.model import CoreGradByteLM, CoreGradLM
from coregrad.tokenizer import ByteTokenizer, CoreGradTokenizer
from coregrad.tokenizer.special_tokens import (
    ASSISTANT_END,
    ASSISTANT_START,
    CALL_END,
    CALL_START,
    RESULT_END,
    RESULT_START,
    SYSTEM_END,
    SYSTEM_START,
    USER_END,
    USER_START,
)
from coregrad.tools import (
    DEFAULT_SYSTEM_PROMPT,
    ToolExecutor,
    extract_citations,
    extract_tool_calls,
    strip_think_blocks,
)

_ABSTAIN_PATTERNS = (
    "could not find",
    "cannot find",
    "not found",
    "no information",
    "do not contain",
    "doesn't contain",
    "not mentioned",
    "unable to find",
)


@dataclass
class RolloutTrace:
    question: str
    full_ids: list[int]
    assistant_mask: list[bool]
    rendered_text: str
    final_answer: str
    tool_calls_raw: list[str]
    tool_calls_valid: int
    tool_calls_total: int
    retrieved_doc_ids: set[str]
    cited_doc_ids: list[str]


@torch.no_grad()
def run_agent_rollout(
    model: CoreGradLM | CoreGradByteLM,
    tokenizer: CoreGradTokenizer | ByteTokenizer,
    question: str,
    executor: ToolExecutor | None,
    device: torch.device,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tool_turns: int = 4,
    max_new_tokens_per_turn: int = 160,
    temperature: float = 0.0,
    top_k: int = 40,
) -> RolloutTrace:
    """Run a live multi-turn model <-> ToolExecutor rollout, tracking assistant token masks."""
    model.eval()
    # Scale per-turn generation budget cleanly for raw UTF-8 byte models (~4 bytes per BPE token)
    turn_budget = max_new_tokens_per_turn * 4 if isinstance(tokenizer, ByteTokenizer) else max_new_tokens_per_turn
    prefix = f"{SYSTEM_START}{system_prompt}{SYSTEM_END}{USER_START}{question}{USER_END}"
    prompt_ids = tokenizer.encode(prefix, add_bos=True, add_eos=False)

    full_ids: list[int] = list(prompt_ids)
    assistant_mask: list[bool] = [False] * len(prompt_ids)

    call_end_id = tokenizer.token_to_id(CALL_END)
    eos_id = tokenizer.eos_id
    amp_enabled = device.type == "cuda"

    raw_calls: list[str] = []
    valid_calls = 0
    retrieved_docs: set[str] = set()

    for _ in range(max_tool_turns + 1):
        turn_gen: list[int] = []
        stopped_on_call = False
        stopped_on_eos = False

        for _step in range(turn_budget):
            ctx = full_ids[-model.cfg.max_seq_len :]
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                logits, _ = model(x)
            next_logits = logits[:, -1, :].float()
            if temperature <= 0.0:
                next_tok = int(torch.argmax(next_logits, dim=-1).item())
            else:
                scaled = next_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled[scaled < v[:, [-1]]] = -float("inf")
                probs = F.softmax(scaled, dim=-1)
                next_tok = int(torch.multinomial(probs, num_samples=1).item())

            full_ids.append(next_tok)
            assistant_mask.append(True)
            turn_gen.append(next_tok)

            if next_tok == eos_id:
                stopped_on_eos = True
                break
            if next_tok == call_end_id:
                stopped_on_call = True
                break

        if stopped_on_eos or not stopped_on_call or executor is None:
            break

        turn_text = tokenizer.decode(turn_gen, skip_special_tokens=False)
        calls = extract_tool_calls(turn_text)
        if not calls:
            break

        call_str = calls[-1]
        raw_calls.append(call_str)
        exec_res = executor.execute_raw(call_str)
        if exec_res.ok:
            valid_calls += 1
            retrieved_docs.update(exec_res.retrieved_doc_ids)

        res_block = f"{RESULT_START}{exec_res.output}{RESULT_END}"
        res_ids = tokenizer.encode(res_block, add_bos=False, add_eos=False)
        full_ids.extend(res_ids)
        assistant_mask.extend([False] * len(res_ids))

    rendered = tokenizer.decode(full_ids, skip_special_tokens=False)
    # Extract final assistant segment after last <|assistant|> or last <|/result|>
    if ASSISTANT_START in rendered:
        tail = rendered.rsplit(ASSISTANT_START, 1)[-1]
        tail = tail.split(ASSISTANT_END, 1)[0]
    elif RESULT_END in rendered:
        tail = rendered.rsplit(RESULT_END, 1)[-1]
    else:
        tail = rendered.rsplit(USER_END, 1)[-1]

    final_clean = strip_think_blocks(tail)
    for tok in (CALL_START, CALL_END, ASSISTANT_START, ASSISTANT_END, "<|eos|>"):
        final_clean = final_clean.replace(tok, "")
    final_clean = final_clean.strip()
    cited = extract_citations(tail)

    return RolloutTrace(
        question=question,
        full_ids=full_ids,
        assistant_mask=assistant_mask,
        rendered_text=rendered,
        final_answer=final_clean,
        tool_calls_raw=raw_calls,
        tool_calls_valid=valid_calls,
        tool_calls_total=len(raw_calls),
        retrieved_doc_ids=retrieved_docs,
        cited_doc_ids=cited,
    )


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def answer_matches(final_answer: str, expected: str, should_abstain: bool = False) -> bool:
    norm = _normalize_text(final_answer)
    if should_abstain:
        return any(pat in norm for pat in _ABSTAIN_PATTERNS)
    exp_norm = _normalize_text(expected)
    if not exp_norm:
        return False
    return exp_norm in norm


def score_rollout_reward(
    trace: RolloutTrace,
    *,
    expected_answer: str,
    expected_doc_ids: tuple[str, ...] = (),
    should_abstain: bool = False,
    requires_tool: bool = True,
) -> float:
    """Purely programmatic verifiable reward for GRPO / RLVR."""
    reward = 0.0

    # 1. Tool syntax & execution reward
    if trace.tool_calls_total > 0:
        if trace.tool_calls_valid == trace.tool_calls_total:
            reward += 0.2
        else:
            reward -= 0.2
    elif requires_tool:
        reward -= 0.1

    # 2. Exact answer or calibrated abstention reward
    if answer_matches(trace.final_answer, expected_answer, should_abstain=should_abstain):
        reward += 0.5

    # 3. Citation grounding & anti-hallucination penalty
    if trace.cited_doc_ids:
        hallucinated = [c for c in trace.cited_doc_ids if c not in trace.retrieved_doc_ids]
        if hallucinated:
            reward -= 0.5
        elif expected_doc_ids and any(c in expected_doc_ids for c in trace.cited_doc_ids):
            reward += 0.2
    elif expected_doc_ids and not should_abstain:
        reward -= 0.05

    return reward
