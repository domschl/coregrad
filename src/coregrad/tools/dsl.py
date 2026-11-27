"""Unified terse ASCII-functional Logic, Arithmetic, Array, Recursion & SymPy DSL (coregrad-DSL).

100% sandboxed custom recursive-descent parser & AST evaluator (zero eval/exec, zero OS/import
surface, strict gas limit and recursion depth cap). Designed as a compact, token-efficient
universal coprocessor for 25M-125M cognitive cores:

  - Exact arithmetic & variables : let x = 12; x * 3 + 5
  - APL-terse pipelines & arrays : 1..10 |> map(x -> x^2) |> sum
  - Bounded recursion (Turing)   : let f(n) = if(n <= 1, 1, n * f(n - 1)); f(7)
  - Symbolic algebra (SymPy)     : solve(3*x + 7 = 22, x), diff(x^3, x), int(x^2, x, 0, 3)
  - Formal logic prover          : prove([x:A, A->B, B&C->D, x:C], x:D)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import sympy as sp

_MAX_GAS = 2500
_MAX_DEPTH = 32
_MAX_LIST = 1000
_MAX_EXPONENT = 256

_SYMPY_FUNCS = frozenset({"solve", "diff", "int", "simplify", "expand", "factor", "subs", "limit"})


class DSLError(ValueError):
    """Raised when a coregrad-DSL expression fails parsing, safety bounds, or evaluation."""


# ---------------------------------------------------------------------------
# Formal Logic Prover (`prove([x:A, A->B, B&C->D, D-/>E], x:E)`)
# ---------------------------------------------------------------------------

def _eval_prove_call(raw_args: str) -> str:
    """Parse and run exact forward-chaining prover over compact Horn / exclusion clauses."""
    m = re.match(r"^\s*\[(.*)\]\s*,\s*([A-Za-z0-9_]+)\s*:\s*([A-Za-z0-9_]+)\s*$", raw_args, re.DOTALL)
    if not m:
        raise DSLError("prove expects prove([clause, ...], entity:Pred)")
    clauses_str, q_ent, q_pred = m.group(1), m.group(2), m.group(3)
    clauses = [c.strip() for c in clauses_str.split(",") if c.strip()]
    if len(clauses) > 128:
        raise DSLError("Too many logic clauses (max 128)")

    pos: dict[str, set[str]] = {}
    neg: dict[str, set[str]] = {}
    unary: list[tuple[str, str]] = []
    conj: list[tuple[list[str], str]] = []
    excl: list[tuple[str, str]] = []

    for cl in clauses:
        if "-/>" in cl:
            lhs, rhs = (p.strip() for p in cl.split("-/>", 1))
            if not lhs or not rhs:
                raise DSLError(f"Invalid exclusion clause: {cl!r}")
            excl.append((lhs, rhs))
        elif "->" in cl:
            lhs, rhs = (p.strip() for p in cl.split("->", 1))
            if not lhs or not rhs:
                raise DSLError(f"Invalid implication clause: {cl!r}")
            if "&" in lhs:
                ants = [a.strip() for a in lhs.split("&") if a.strip()]
                if len(ants) < 2:
                    raise DSLError(f"Invalid conjunction clause: {cl!r}")
                conj.append((ants, rhs))
            else:
                unary.append((lhs, rhs))
        elif ":" in cl:
            ent, pred = (p.strip() for p in cl.split(":", 1))
            if pred.startswith("!"):
                neg.setdefault(ent, set()).add(pred[1:].strip())
            else:
                pos.setdefault(ent, set()).add(pred)
        else:
            raise DSLError(f"Unrecognized logic clause: {cl!r}")

    known_p = set(pos.get(q_ent, set()))
    known_n = set(neg.get(q_ent, set()))
    changed = True
    iters = 0
    while changed and iters < 256:
        changed = False
        iters += 1
        for a, b in unary:
            if a in known_p and b not in known_p:
                known_p.add(b)
                changed = True
        for ants, c in conj:
            if all(a in known_p for a in ants) and c not in known_p:
                known_p.add(c)
                changed = True
        for a, b in excl:
            if a in known_p and b not in known_n:
                known_n.add(b)
                changed = True
            if b in known_p and a not in known_n:
                known_n.add(a)
                changed = True

    if q_pred in known_p and q_pred in known_n:
        return "Contradiction"
    if q_pred in known_p:
        return "True"
    if q_pred in known_n:
        return "False"
    return "Unknown"


# ---------------------------------------------------------------------------
# SymPy Symbolic Bridge (`solve`, `diff`, `int`, `simplify`, `expand`, `factor`, `subs`, `limit`)
# ---------------------------------------------------------------------------

_SYMPY_LOCALS: dict[str, Any] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "pi": sp.pi,
    "e": sp.E,
    "oo": sp.oo,
}


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur or parts:
        tail = "".join(cur).strip()
        if tail:
            parts.append(tail)
    return parts


def _parse_sympy_expr(expr_s: str, env_vars: dict[str, Any]) -> sp.Expr:
    cleaned = expr_s.strip().replace("^", "**")
    if len(cleaned) > 256:
        raise DSLError("Symbolic expression too long")
    # Allow only safe alphanumeric / math punctuation characters
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/().,\s=]+", cleaned):
        raise DSLError(f"Disallowed characters in symbolic expression: {expr_s!r}")
    local_dict = dict(_SYMPY_LOCALS)
    for k, v in env_vars.items():
        if isinstance(v, (int, float)):
            local_dict[k] = sp.nsimplify(v) if isinstance(v, float) else sp.Integer(v)
    # Implicitly treat any unknown identifier as a real SymPy symbol
    for ident in set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", cleaned)):
        if ident not in local_dict:
            local_dict[ident] = sp.Symbol(ident)
    try:
        if "=" in cleaned and "==" not in cleaned:
            lhs, rhs = cleaned.split("=", 1)
            return sp.Eq(sp.sympify(lhs, locals=local_dict), sp.sympify(rhs, locals=local_dict))
        return sp.sympify(cleaned, locals=local_dict)
    except Exception as exc:
        raise DSLError(f"Invalid symbolic expression {expr_s!r}: {exc}") from exc


def _fmt_sympy(val: Any) -> str:
    if isinstance(val, list):
        return "[" + ", ".join(_fmt_sympy(v) for v in val) + "]"
    if isinstance(val, dict):
        items = sorted(val.items(), key=lambda kv: str(kv[0]))
        return "{" + ", ".join(f"{k}: {_fmt_sympy(v)}" for k, v in items) + "}"
    s = str(val).replace("**", "^")
    return s


def _eval_sympy_call(fn_name: str, raw_args: str, env_vars: dict[str, Any]) -> str:
    args = _split_top_level(raw_args, ",")
    if not args:
        raise DSLError(f"{fn_name}() requires at least 1 argument")
    if fn_name == "solve":
        if len(args) < 1 or len(args) > 2:
            raise DSLError("solve(eq, var) expects 1 or 2 arguments")
        eq_arg = args[0]
        if eq_arg.startswith("[") and eq_arg.endswith("]"):
            eqs = [_parse_sympy_expr(e, env_vars) for e in _split_top_level(eq_arg[1:-1], ",")]
        else:
            eqs = _parse_sympy_expr(eq_arg, env_vars)
        if len(args) == 2:
            v_arg = args[1]
            if v_arg.startswith("[") and v_arg.endswith("]"):
                vs = [sp.Symbol(v.strip()) for v in _split_top_level(v_arg[1:-1], ",")]
            else:
                vs = sp.Symbol(v_arg.strip())
            sol = sp.solve(eqs, vs)
        else:
            sol = sp.solve(eqs)
        if isinstance(sol, list) and len(sol) == 1:
            return _fmt_sympy(sol[0])
        return _fmt_sympy(sol)

    if fn_name == "diff":
        if len(args) not in (2, 3):
            raise DSLError("diff(expr, var[, order]) expects 2 or 3 arguments")
        expr = _parse_sympy_expr(args[0], env_vars)
        var = sp.Symbol(args[1].strip())
        order = int(args[2]) if len(args) == 3 else 1
        if not (1 <= order <= 5):
            raise DSLError("Derivative order must be 1..5")
        return _fmt_sympy(sp.diff(expr, var, order))

    if fn_name == "int":
        if len(args) not in (2, 4):
            raise DSLError("int(expr, var[, a, b]) expects 2 or 4 arguments")
        expr = _parse_sympy_expr(args[0], env_vars)
        var = sp.Symbol(args[1].strip())
        if len(args) == 4:
            lo = _parse_sympy_expr(args[2], env_vars)
            hi = _parse_sympy_expr(args[3], env_vars)
            return _fmt_sympy(sp.integrate(expr, (var, lo, hi)))
        return _fmt_sympy(sp.integrate(expr, var))

    if fn_name == "simplify":
        return _fmt_sympy(sp.simplify(_parse_sympy_expr(args[0], env_vars)))
    if fn_name == "expand":
        return _fmt_sympy(sp.expand(_parse_sympy_expr(args[0], env_vars)))
    if fn_name == "factor":
        return _fmt_sympy(sp.factor(_parse_sympy_expr(args[0], env_vars)))
    if fn_name == "subs":
        if len(args) != 3:
            raise DSLError("subs(expr, var, val) expects 3 arguments")
        expr = _parse_sympy_expr(args[0], env_vars)
        var = sp.Symbol(args[1].strip())
        val = _parse_sympy_expr(args[2], env_vars)
        return _fmt_sympy(sp.simplify(expr.subs(var, val)))
    if fn_name == "limit":
        if len(args) != 3:
            raise DSLError("limit(expr, var, point) expects 3 arguments")
        expr = _parse_sympy_expr(args[0], env_vars)
        var = sp.Symbol(args[1].strip())
        pt = _parse_sympy_expr(args[2], env_vars)
        return _fmt_sympy(sp.limit(expr, var, pt))
    raise DSLError(f"Unknown symbolic op: {fn_name}")


# ---------------------------------------------------------------------------
# Tokenizer & Recursive-Descent Parser for Expressions, Pipes, Lambdas & Let
# ---------------------------------------------------------------------------

_TOK_SPEC = re.compile(
    r"""
    (?P<SPACE>\s+)
  | (?P<NUM>\d+(?:\.\d+)?)
  | (?P<PIPE>\|>)
  | (?P<ARROW>->)
  | (?P<DOTDOT>\.\.)
  | (?P<POW>\*\*)
  | (?P<FLOORDIV>//)
  | (?P<CMP>==|!=|<=|>=)
  | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<OP>[+\-*/%^=<>(),;\[\]])
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class _Tok:
    kind: str
    val: str


def _tokenize(src: str) -> list[_Tok]:
    toks: list[_Tok] = []
    pos = 0
    while pos < len(src):
        m = _TOK_SPEC.match(src, pos)
        if not m:
            raise DSLError(f"Unexpected character {src[pos]!r} at position {pos}")
        kind = m.lastgroup
        assert kind is not None
        if kind != "SPACE":
            toks.append(_Tok(kind, m.group()))
        pos = m.end()
    toks.append(_Tok("EOF", ""))
    return toks


@dataclass
class _Closure:
    params: list[str]
    body: Any
    env: dict[str, Any]


class _Parser:
    def __init__(self, toks: list[_Tok], raw_src: str) -> None:
        self.toks = toks
        self.raw_src = raw_src
        self.i = 0

    def peek(self, offset: int = 0) -> _Tok:
        idx = min(self.i + offset, len(self.toks) - 1)
        return self.toks[idx]

    def eat(self, kind: str | None = None, val: str | None = None) -> _Tok:
        t = self.peek()
        if kind is not None and t.kind != kind:
            raise DSLError(f"Expected {kind}, got {t.kind} ({t.val!r})")
        if val is not None and t.val != val:
            raise DSLError(f"Expected {val!r}, got {t.val!r}")
        self.i += 1
        return t

    def parse_program(self) -> list[Any]:
        stmts: list[Any] = []
        while self.peek().kind != "EOF":
            stmts.append(self.parse_stmt())
            if self.peek().val == ";":
                self.eat(val=";")
            else:
                break
        if self.peek().kind != "EOF":
            raise DSLError(f"Unexpected trailing token {self.peek().val!r}")
        if not stmts:
            raise DSLError("Empty DSL expression")
        return stmts

    def parse_stmt(self) -> Any:
        if self.peek().kind == "IDENT" and self.peek().val == "let":
            self.eat(val="let")
            name = self.eat("IDENT").val
            if self.peek().val == "(":
                self.eat(val="(")
                params: list[str] = []
                if self.peek().val != ")":
                    params.append(self.eat("IDENT").val)
                    while self.peek().val == ",":
                        self.eat(val=",")
                        params.append(self.eat("IDENT").val)
                self.eat(val=")")
                self.eat(val="=")
                body = self.parse_expr()
                return ("let_fn", name, params, body)
            self.eat(val="=")
            val = self.parse_expr()
            return ("let", name, val)
        return self.parse_expr()

    def parse_expr(self) -> Any:
        # Check for lambda: x -> expr  OR  (a, b) -> expr
        if self.peek().kind == "IDENT" and self.peek(1).kind == "ARROW":
            p = self.eat("IDENT").val
            self.eat("ARROW")
            return ("lambda", [p], self.parse_expr())
        if self.peek().val == "(":
            saved = self.i
            try:
                self.eat(val="(")
                ps: list[str] = []
                if self.peek().kind == "IDENT":
                    ps.append(self.eat("IDENT").val)
                    while self.peek().val == ",":
                        self.eat(val=",")
                        ps.append(self.eat("IDENT").val)
                if self.peek().val == ")" and self.peek(1).kind == "ARROW":
                    self.eat(val=")")
                    self.eat("ARROW")
                    return ("lambda", ps, self.parse_expr())
            except DSLError:
                pass
            self.i = saved
        return self.parse_pipe()

    def parse_pipe(self) -> Any:
        node = self.parse_or()
        while self.peek().kind == "PIPE":
            self.eat("PIPE")
            rhs = self.parse_or()
            node = ("pipe", node, rhs)
        return node

    def parse_or(self) -> Any:
        node = self.parse_and()
        while self.peek().kind == "IDENT" and self.peek().val == "or":
            self.eat()
            node = ("binop", "or", node, self.parse_and())
        return node

    def parse_and(self) -> Any:
        node = self.parse_cmp()
        while self.peek().kind == "IDENT" and self.peek().val == "and":
            self.eat()
            node = ("binop", "and", node, self.parse_cmp())
        return node

    def parse_cmp(self) -> Any:
        node = self.parse_range()
        while self.peek().kind == "CMP" or self.peek().val in ("<", ">"):
            op = self.eat().val
            node = ("binop", op, node, self.parse_range())
        return node

    def parse_range(self) -> Any:
        node = self.parse_add()
        if self.peek().kind == "DOTDOT":
            self.eat("DOTDOT")
            end = self.parse_add()
            return ("range", node, end)
        return node

    def parse_add(self) -> Any:
        node = self.parse_mul()
        while self.peek().val in ("+", "-"):
            op = self.eat().val
            node = ("binop", op, node, self.parse_mul())
        return node

    def parse_mul(self) -> Any:
        node = self.parse_unary()
        while self.peek().val in ("*", "/", "%") or self.peek().kind == "FLOORDIV":
            op = self.eat().val
            node = ("binop", op, node, self.parse_unary())
        return node

    def parse_unary(self) -> Any:
        if self.peek().val in ("+", "-"):
            op = self.eat().val
            return ("unary", op, self.parse_unary())
        if self.peek().kind == "IDENT" and self.peek().val == "not":
            self.eat()
            return ("unary", "not", self.parse_unary())
        return self.parse_pow()

    def parse_pow(self) -> Any:
        node = self.parse_primary()
        if self.peek().val == "^" or self.peek().kind == "POW":
            self.eat()
            rhs = self.parse_unary()
            return ("binop", "^", node, rhs)
        return node

    def parse_primary(self) -> Any:
        t = self.peek()
        if t.kind == "NUM":
            self.eat()
            return ("num", float(t.val) if "." in t.val else int(t.val))
        if t.val == "[":
            self.eat(val="[")
            items: list[Any] = []
            if self.peek().val != "]":
                items.append(self.parse_expr())
                while self.peek().val == ",":
                    self.eat(val=",")
                    if self.peek().val == "]":
                        break
                    items.append(self.parse_expr())
            self.eat(val="]")
            return ("list", items)
        if t.val == "(":
            self.eat(val="(")
            inner = self.parse_expr()
            self.eat(val=")")
            return inner
        if t.kind == "IDENT":
            name = self.eat("IDENT").val
            if name == "True":
                return ("bool", True)
            if name == "False":
                return ("bool", False)
            if self.peek().val == "(":
                self.eat(val="(")
                args: list[Any] = []
                if self.peek().val != ")":
                    args.append(self.parse_expr())
                    while self.peek().val == ",":
                        self.eat(val=",")
                        args.append(self.parse_expr())
                self.eat(val=")")
                return ("call", name, args)
            return ("var", name)
        raise DSLError(f"Unexpected token {t.kind} ({t.val!r})")


class _Evaluator:
    def __init__(self) -> None:
        self.gas = _MAX_GAS

    def _tick(self) -> None:
        self.gas -= 1
        if self.gas <= 0:
            raise DSLError("Execution step limit exceeded")

    def eval_program(self, stmts: list[Any], env: dict[str, Any]) -> Any:
        res: Any = None
        for st in stmts:
            res = self.eval(st, env, depth=0)
        return res

    def _call_fn(self, fn: Any, args: list[Any], depth: int) -> Any:
        if depth >= _MAX_DEPTH:
            raise DSLError("Maximum recursion depth exceeded")
        if isinstance(fn, _Closure):
            if len(args) != len(fn.params):
                raise DSLError(f"Expected {len(fn.params)} args, got {len(args)}")
            local = dict(fn.env)
            for k, v in zip(fn.params, args, strict=True):
                local[k] = v
            return self.eval(fn.body, local, depth + 1)
        if callable(fn):
            return fn(*args)
        raise DSLError(f"Not callable: {fn!r}")

    def eval(self, node: Any, env: dict[str, Any], depth: int) -> Any:
        self._tick()
        tag = node[0]
        if tag in ("num", "bool"):
            return node[1]
        if tag == "var":
            name = node[1]
            if name not in env:
                raise DSLError(f"Unknown variable: {name!r}")
            return env[name]
        if tag == "let":
            _, name, expr = node
            val = self.eval(expr, env, depth)
            env[name] = val
            return val
        if tag == "let_fn":
            _, name, params, body = node
            fn = _Closure(params=params, body=body, env=env)
            env[name] = fn
            return fn
        if tag == "lambda":
            _, params, body = node
            return _Closure(params=params, body=body, env=dict(env))
        if tag == "list":
            items = [self.eval(it, env, depth) for it in node[1]]
            if len(items) > _MAX_LIST:
                raise DSLError("List length limit exceeded")
            return items
        if tag == "range":
            lo = int(self.eval(node[1], env, depth))
            hi = int(self.eval(node[2], env, depth))
            step = 1 if hi >= lo else -1
            if abs(hi - lo) + 1 > _MAX_LIST:
                raise DSLError("Range length limit exceeded")
            return list(range(lo, hi + step, step))
        if tag == "unary":
            _, op, sub = node
            v = self.eval(sub, env, depth)
            if op == "+":
                return +v
            if op == "-":
                return -v
            if op == "not":
                return not v
            raise DSLError(f"Unknown unary op: {op}")
        if tag == "binop":
            _, op, a_n, b_n = node
            if op == "and":
                return bool(self.eval(a_n, env, depth)) and bool(self.eval(b_n, env, depth))
            if op == "or":
                return bool(self.eval(a_n, env, depth)) or bool(self.eval(b_n, env, depth))
            a = self.eval(a_n, env, depth)
            b = self.eval(b_n, env, depth)
            if op == "+":
                return a + b
            if op == "-":
                return a - b
            if op == "*":
                return a * b
            if op == "/":
                if b == 0:
                    raise DSLError("Division by zero")
                return a / b
            if op == "//":
                if b == 0:
                    raise DSLError("Division by zero")
                return a // b
            if op == "%":
                if b == 0:
                    raise DSLError("Modulo by zero")
                return a % b
            if op == "^":
                if isinstance(b, (int, float)) and abs(b) > _MAX_EXPONENT:
                    raise DSLError("Exponent too large")
                return a**b
            if op == "==":
                return a == b
            if op == "!=":
                return a != b
            if op == "<":
                return a < b
            if op == "<=":
                return a <= b
            if op == ">":
                return a > b
            if op == ">=":
                return a >= b
            raise DSLError(f"Unknown binary op: {op}")
        if tag == "pipe":
            _, lhs, rhs = node
            lv = self.eval(lhs, env, depth)
            if rhs[0] == "var":
                return self._call_builtin_or_env(rhs[1], [lv], env, depth)
            if rhs[0] == "call":
                _, fn_name, arg_nodes = rhs
                # Inject piped value as the FIRST argument: xs |> map(x -> x^2) == map(xs, x -> x^2)
                return self._eval_call(fn_name, [lv] + [self.eval(a, env, depth) for a in arg_nodes], env, depth)
            raise DSLError("Pipe RHS must be a function name or call")
        if tag == "call":
            _, fn_name, arg_nodes = node
            # Short-circuit if(cond, then_expr, else_expr)
            if fn_name == "if":
                if len(arg_nodes) != 3:
                    raise DSLError("if(cond, then, else) requires 3 arguments")
                return (
                    self.eval(arg_nodes[1], env, depth)
                    if self.eval(arg_nodes[0], env, depth)
                    else self.eval(arg_nodes[2], env, depth)
                )
            args = [self.eval(a, env, depth) for a in arg_nodes]
            return self._eval_call(fn_name, args, env, depth)
        raise DSLError(f"Unknown AST node: {tag}")

    def _call_builtin_or_env(self, name: str, args: list[Any], env: dict[str, Any], depth: int) -> Any:
        return self._eval_call(name, args, env, depth)

    def _eval_call(self, name: str, args: list[Any], env: dict[str, Any], depth: int) -> Any:
        # Higher-order array primitives
        if name == "map":
            if len(args) != 2 or not isinstance(args[0], list):
                raise DSLError("map(list, fn) expects a list and function")
            return [self._call_fn(args[1], [x], depth) for x in args[0]]
        if name == "filter":
            if len(args) != 2 or not isinstance(args[0], list):
                raise DSLError("filter(list, pred) expects a list and predicate")
            return [x for x in args[0] if self._call_fn(args[1], [x], depth)]
        if name == "fold":
            if len(args) != 3 or not isinstance(args[0], list):
                raise DSLError("fold(list, init, fn) expects (list, init, (acc, x) -> ...)")
            acc = args[1]
            for x in args[0]:
                acc = self._call_fn(args[2], [acc, x], depth)
            return acc
        if name == "sum":
            return sum(args[0]) if (len(args) == 1 and isinstance(args[0], list)) else sum(args)
        if name == "prod":
            xs = args[0] if (len(args) == 1 and isinstance(args[0], list)) else args
            return math.prod(xs)
        if name == "mean":
            xs = args[0] if (len(args) == 1 and isinstance(args[0], list)) else args
            if not xs:
                raise DSLError("mean of empty list")
            return sum(xs) / len(xs)
        if name == "len":
            return len(args[0])
        if name == "sort":
            return sorted(args[0])
        if name == "rev":
            return list(reversed(args[0]))
        if name == "uniq":
            return list(dict.fromkeys(args[0]))
        if name == "take":
            return args[0][: int(args[1])]
        if name == "drop":
            return args[0][int(args[1]) :]
        if name == "min":
            return min(args[0]) if (len(args) == 1 and isinstance(args[0], list)) else min(args)
        if name == "max":
            return max(args[0]) if (len(args) == 1 and isinstance(args[0], list)) else max(args)
        if name == "range":
            if len(args) == 1:
                n = int(args[0])
                if n > _MAX_LIST:
                    raise DSLError("range limit exceeded")
                return list(range(n))
            if len(args) == 2:
                a, b = int(args[0]), int(args[1])
                if abs(b - a) > _MAX_LIST:
                    raise DSLError("range limit exceeded")
                return list(range(a, b))
            raise DSLError("range expects 1 or 2 args")
        if name == "round":
            return round(args[0], int(args[1])) if len(args) == 2 else round(args[0])
        if name == "abs":
            return abs(args[0])
        if name == "sqrt":
            return math.sqrt(args[0])
        if name == "floor":
            return math.floor(args[0])
        if name == "ceil":
            return math.ceil(args[0])
        if name == "gcd":
            return math.gcd(*(int(x) for x in args))
        if name == "lcm":
            return math.lcm(*(int(x) for x in args))
        if name == "log":
            return math.log(args[0], args[1]) if len(args) == 2 else math.log(args[0])
        if name == "exp":
            return math.exp(args[0])
        if name == "sin":
            return math.sin(args[0])
        if name == "cos":
            return math.cos(args[0])
        if name == "any":
            return any(args[0])
        if name == "all":
            return all(args[0])
        if name in env:
            return self._call_fn(env[name], args, depth)
        raise DSLError(f"Unknown function: {name!r}")


def _format_value(val: Any) -> str:
    if isinstance(val, bool):
        return "True" if val else "False"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            raise DSLError("Non-finite result")
        if abs(val - round(val)) < 1e-9 and abs(val) < 1e15:
            return str(round(val))
        return f"{val:.10g}"
    if isinstance(val, list):
        return "[" + ", ".join(_format_value(v) for v in val) + "]"
    return str(val)


def safe_dsl(code: str) -> str:
    """Evaluate a coregrad-DSL expression safely and return a canonical string result."""
    src = code.strip()
    if not src or len(src) > 1024:
        raise DSLError("DSL expression must be 1..1024 characters")

    # Handle top-level `let ...; prove(...)` or `let ...; solve/diff/int(...)` cleanly
    parts = _split_top_level(src, ";")
    env: dict[str, Any] = {"pi": math.pi, "e": math.e}
    ev = _Evaluator()

    for idx, part in enumerate(parts):
        is_last = idx == len(parts) - 1
        # Check for prove(...) or SymPy symbolic call
        m_call = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$", part, re.DOTALL)
        if m_call:
            fn_name, raw_args = m_call.group(1), m_call.group(2)
            if fn_name == "prove":
                res = _eval_prove_call(raw_args)
                if is_last:
                    return res
                continue
            if fn_name in _SYMPY_FUNCS:
                res = _eval_sympy_call(fn_name, raw_args, env)
                if is_last:
                    return res
                continue

        toks = _tokenize(part)
        stmts = _Parser(toks, part).parse_program()
        val = ev.eval_program(stmts, env)
        if is_last:
            return _format_value(val)

    raise DSLError("No final expression evaluated")
