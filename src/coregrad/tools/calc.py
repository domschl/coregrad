"""Safe, sandboxed AST arithmetic & math evaluator for coregrad's calc(expr) tool."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_FUNCS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": pow,
}

_SAFE_CONSTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}

MAX_AST_NODES = 128
MAX_EXPONENT = 1000.0


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)

    if isinstance(node, ast.Name):
        if node.id in _SAFE_CONSTS:
            return _SAFE_CONSTS[node.id]
        raise ValueError(f"Unknown symbol: {node.id!r}")

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_node(node.operand)))

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise ValueError(f"Exponent too large: {right}")
            return float(left**right)
        op_fn = _BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return float(op_fn(left, right))

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _SAFE_FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"Unsupported function: {node.func.id!r}")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported in calc()")
        args: list[Any] = [_eval_node(a) for a in node.args]
        if node.func.id == "round" and len(args) == 2:
            args[1] = int(args[1])
        return float(fn(*args))

    raise ValueError(f"Disallowed expression node: {type(node).__name__}")


def format_number(val: float) -> str:
    """Format numeric results cleanly (integers without .0, floats trimmed)."""
    if not math.isfinite(val):
        raise ValueError(f"Non-finite result: {val}")
    if abs(val - round(val)) < 1e-10 and abs(val) < 1e15:
        return str(round(val))
    return f"{val:.10g}"


def safe_calc(expr: str) -> str:
    """Safely evaluate a mathematical expression and return a clean string."""
    cleaned = expr.strip()
    if not cleaned:
        raise ValueError("Empty expression")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid math syntax: {exc}") from exc

    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError("Expression too complex")

    result = _eval_node(tree.body)
    return format_number(result)
