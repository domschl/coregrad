"""Canonical cognitive-core tool protocol, AST call parser, and masked SFT encoder."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

from coregrad.tokenizer import CoreGradTokenizer
from coregrad.tokenizer.special_tokens import (
    ASSISTANT_END,
    ASSISTANT_START,
    BOS_TOKEN,
    CALL_END,
    CALL_START,
    CITE_END,
    CITE_START,
    EOS_TOKEN,
    RESULT_END,
    RESULT_START,
    SYSTEM_END,
    SYSTEM_START,
    THINK_END,
    THINK_START,
    USER_END,
    USER_START,
)

ALLOWED_TOOLS: dict[str, dict[str, type | tuple[type, ...]]] = {
    "search": {"query": str, "k": int},
    "read": {"doc_id": str, "start": int, "length": int},
    "calc": {"expr": str},
    "list_docs": {"filter": str},
}

REQUIRED_TOOL_ARGS: dict[str, tuple[str, ...]] = {
    "search": ("query",),
    "read": ("doc_id",),
    "calc": ("expr",),
    "list_docs": (),
}

_CALL_BLOCK_RE = re.compile(re.escape(CALL_START) + r"(.*?)" + re.escape(CALL_END), re.DOTALL)
_CITE_RE = re.compile(re.escape(CITE_START) + r"([^<]+?)" + re.escape(CITE_END))
_THINK_RE = re.compile(re.escape(THINK_START) + r".*?" + re.escape(THINK_END), re.DOTALL)

DEFAULT_SYSTEM_PROMPT = (
    "You are coregrad, a compact research assistant. "
    "Use tools when external knowledge or exact calculation is needed: "
    "search(query, k=3), read(doc_id, start=0, length=800), calc(expr), list_docs(filter=''). "
    "Cite retrieved document IDs with <|cite|>doc_id<|/cite|>. "
    "If the documents do not contain the answer, state that clearly instead of guessing."
)


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Turn:
    """One segment of a trajectory.

    role in {"system", "user", "think", "call", "result", "assistant"}:
    - Non-trainable roles (loss masked to -100): system, user, result
    - Trainable roles: think, call, assistant
    """

    role: str
    content: str


TRAINABLE_ROLES = frozenset({"think", "call", "assistant"})


def format_tool_call(name: str, **kwargs: Any) -> str:
    """Format a canonical single-line tool call like search(query="...", k=3)."""
    parts: list[str] = []
    for k, v in kwargs.items():
        if isinstance(v, str):
            parts.append(f"{k}={json.dumps(v, ensure_ascii=False)}")
        else:
            parts.append(f"{k}={v!r}")
    return f"{name}({', '.join(parts)})"


def parse_tool_call(raw: str) -> ToolCall:
    """Safely parse a tool call expression using Python AST (no eval)."""
    text = raw.strip()
    if not text:
        raise ValueError("Empty tool call")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid tool call syntax: {exc}") from exc

    node = tree.body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        msg = "Tool call must be a simple function call like search(query='...')"
        raise TypeError(msg)

    name = node.func.id
    args: dict[str, Any] = {}

    if node.args:
        # Map positional arguments onto canonical parameter order if known
        param_order = list(ALLOWED_TOOLS.get(name, {}).keys())
        if len(node.args) > len(param_order):
            raise ValueError(f"Too many positional arguments for {name}")
        for pname, arg_node in zip(param_order, node.args, strict=False):
            args[pname] = _literal_value(arg_node)

    for kw in node.keywords:
        if kw.arg is None:
            raise ValueError("**kwargs unpacking is not allowed in tool calls")
        args[kw.arg] = _literal_value(kw.value)

    return ToolCall(name=name, args=args)


def _literal_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError) as exc:
        raise ValueError("Tool arguments must be literal constants") from exc


def validate_tool_call(call: ToolCall, allow_generic: bool = False) -> tuple[bool, str]:
    """Validate tool name, required parameters, and argument types."""
    if call.name not in ALLOWED_TOOLS:
        if allow_generic:
            return True, "ok"
        return False, f"Unknown tool: {call.name!r}"

    schema = ALLOWED_TOOLS[call.name]
    for req in REQUIRED_TOOL_ARGS[call.name]:
        if req not in call.args:
            return False, f"Missing required argument {req!r} for {call.name}"

    for key, val in call.args.items():
        if key not in schema:
            return False, f"Unexpected argument {key!r} for {call.name}"
        expected = schema[key]
        if isinstance(val, bool) and expected is int:
            return False, f"Argument {key!r} must be int, not bool"
        if not isinstance(val, expected):
            return False, f"Argument {key!r} expected {expected}, got {type(val).__name__}"

    return True, "ok"


def extract_tool_calls(text: str) -> list[str]:
    return [m.group(1).strip() for m in _CALL_BLOCK_RE.finditer(text)]


def extract_citations(text: str) -> list[str]:
    return [m.group(1).strip() for m in _CITE_RE.finditer(text)]


def strip_think_blocks(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def format_turn(turn: Turn) -> str:
    """Wrap one Turn with canonical special tokens."""
    role = turn.role
    c = turn.content
    if role == "system":
        return f"{SYSTEM_START}{c}{SYSTEM_END}"
    if role == "user":
        return f"{USER_START}{c}{USER_END}"
    if role == "think":
        return f"{THINK_START}{c}{THINK_END}"
    if role == "call":
        return f"{CALL_START}{c}{CALL_END}"
    if role == "result":
        return f"{RESULT_START}{c}{RESULT_END}"
    if role == "assistant":
        return f"{ASSISTANT_START}{c}{ASSISTANT_END}"
    raise ValueError(f"Unknown turn role: {role!r}")


def render_trajectory(turns: list[Turn], add_bos: bool = True, add_eos: bool = True) -> str:
    """Render a multi-turn cognitive-core trajectory as a single canonical string."""
    parts = [BOS_TOKEN] if add_bos else []
    parts.extend(format_turn(t) for t in turns)
    if add_eos:
        parts.append(EOS_TOKEN)
    return "".join(parts)


def encode_masked_trajectory(
    turns: list[Turn],
    tokenizer: CoreGradTokenizer,
    max_seq_len: int | None = None,
) -> tuple[list[int], list[int]]:
    """Encode turns into (input_ids, labels) with non-assistant spans masked to -100.

    Labels are aligned for next-token prediction by the caller (i.e. inputs=ids[:-1],
    targets=labels[1:]), so labels[i] is either ids[i] (trainable) or -100 (masked).
    """
    ids: list[int] = [tokenizer.bos_id]
    labels: list[int] = [-100]

    for turn in turns:
        seg_ids = tokenizer.encode(format_turn(turn), add_bos=False, add_eos=False)
        ids.extend(seg_ids)
        if turn.role in TRAINABLE_ROLES:
            labels.extend(seg_ids)
        else:
            labels.extend([-100] * len(seg_ids))

    ids.append(tokenizer.eos_id)
    labels.append(tokenizer.eos_id)

    if max_seq_len is not None and len(ids) > max_seq_len:
        ids = ids[:max_seq_len]
        labels = labels[:max_seq_len]
    return ids, labels
