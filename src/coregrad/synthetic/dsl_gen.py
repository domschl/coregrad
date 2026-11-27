"""Progressive English-grounded & pure-symbolic curriculum generator for coregrad-DSL.

Builds deterministic, ToolExecutor-verified trajectories across 4 unified levels:
  Level 1 (Logic)     : English syllogism / Horn rules -> bind + calc(expr="prove(...)")
  Level 2 (Bindings)  : Multi-step word problems -> calc(expr="let a = ...; ...")
  Level 3 (Pipelines) : Series, filters, folds & bounded recursion -> calc(expr="1..n |> ...")
  Level 4 (Symbolic)  : Equations, systems, derivatives & integrals -> calc(expr="solve/diff/int(...)")
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from coregrad.tools.dsl import safe_dsl
from coregrad.tools.protocol import DEFAULT_SYSTEM_PROMPT, Turn, format_tool_call


@dataclass(frozen=True)
class VerifiableDSLTask:
    task_id: str
    level: str  # logic | bindings | pipelines | symbolic
    question: str
    expected_answer: str
    dsl_expr: str
    turns: list[Turn]


def _make_logic_dsl(rng: random.Random, idx: int) -> VerifiableDSLTask:
    ent = rng.choice(["Ada", "Bram", "Kora", "Tavin", "Soren", "Lyra"])
    syms = ["A", "B", "C", "D", "E", "F"]
    words = rng.sample(
        ["glimmer", "vortex", "beacon", "prism", "cipher", "nebula", "quartz", "helix", "vector", "lattice"],
        6,
    )
    depth = rng.randint(3, 5)
    mode = rng.choice(["pos", "neg", "unk", "conj"])
    binds = [f"{syms[i]}={words[i]}" for i in range(depth + 1)]
    clauses: list[str] = [f"x:{syms[0]}"]
    en_lines: list[str] = [f"{ent} is a {words[0]}."]

    if mode == "pos":
        for i in range(depth):
            clauses.append(f"{syms[i]}->{syms[i + 1]}")
            en_lines.append(f"Every {words[i]} is a {words[i + 1]}.")
    elif mode == "neg":
        for i in range(depth - 1):
            clauses.append(f"{syms[i]}->{syms[i + 1]}")
            en_lines.append(f"Every {words[i]} is a {words[i + 1]}.")
        clauses.append(f"{syms[depth - 1]}-/>{syms[depth]}")
        en_lines.append(f"No {words[depth - 1]} is a {words[depth]}.")
    elif mode == "unk":
        brk = rng.randint(1, depth - 1)
        for i in range(depth):
            if i == brk:
                clauses.append(f"{syms[i + 1]}->{syms[i]}")
                en_lines.append(f"Every {words[i + 1]} is a {words[i]}.")
            else:
                clauses.append(f"{syms[i]}->{syms[i + 1]}")
                en_lines.append(f"Every {words[i]} is a {words[i + 1]}.")
    else:  # conj
        extra_w = words[depth + 1] if depth + 1 < len(words) else "spire"
        binds.append(f"Z={extra_w}")
        clauses.append("x:Z")
        en_lines.append(f"{ent} is a {extra_w}.")
        join = max(1, depth // 2)
        for i in range(depth):
            if i == join:
                clauses.append(f"{syms[i]}&Z->{syms[i + 1]}")
                en_lines.append(f"Anything that is both a {words[i]} and a {extra_w} is a {words[i + 1]}.")
            else:
                clauses.append(f"{syms[i]}->{syms[i + 1]}")
                en_lines.append(f"Every {words[i]} is a {words[i + 1]}.")

    rng.shuffle(en_lines)
    target_w = words[depth]
    expr = f"prove([{', '.join(clauses)}], x:{syms[depth]})"
    ans = safe_dsl(expr)
    q = (
        f"Rules: {' '.join(en_lines)}\n"
        f"Based strictly on these rules, is {ent} a {target_w}? Answer True, False, or Unknown."
    )
    think = f"bind: x={ent}; {'; '.join(binds)}\nFormalize and verify via prove()."
    turns = [
        Turn("system", DEFAULT_SYSTEM_PROMPT),
        Turn("user", q),
        Turn("think", think),
        Turn("call", format_tool_call("calc", expr=expr)),
        Turn("result", ans),
        Turn("assistant", ans),
    ]
    return VerifiableDSLTask(f"dsl_logic_{idx}", "logic", q, ans, expr, turns)


def _make_binding_problem(rng: random.Random, idx: int) -> VerifiableDSLTask:
    n1 = rng.randint(4, 30)
    p1 = rng.randint(5, 60)
    n2 = rng.randint(3, 25)
    p2 = rng.randint(6, 50)
    disc = rng.choice([5, 10, 15, 20])
    tax = rng.choice([7, 8, 10, 15, 19])
    expr = f"let sub = {n1}*{p1} + {n2}*{p2}; let net = sub * (1 - {disc}/100); round(net * (1 + {tax}/100), 2)"
    ans = safe_dsl(expr)
    q = (
        f"A lab purchases {n1} sensors at ${p1} each and {n2} cables at ${p2} each. "
        f"After a {disc}% discount, a {tax}% tax is applied. What is the final cost?"
    )
    think = f"bind: n1={n1}, p1={p1}, n2={n2}, p2={p2}, disc={disc}%, tax={tax}%\nCompute subtotal -> discounted -> taxed."
    turns = [
        Turn("system", DEFAULT_SYSTEM_PROMPT),
        Turn("user", q),
        Turn("think", think),
        Turn("call", format_tool_call("calc", expr=expr)),
        Turn("result", ans),
        Turn("assistant", f"The final cost is ${ans}."),
    ]
    return VerifiableDSLTask(f"dsl_bind_{idx}", "bindings", q, ans, expr, turns)


def _make_pipeline_problem(rng: random.Random, idx: int) -> VerifiableDSLTask:
    kind = rng.choice(["sum_sq", "filter_div", "factorial", "mean_list"])
    if kind == "sum_sq":
        lo = rng.randint(1, 5)
        hi = rng.randint(lo + 4, 25)
        expr = f"{lo}..{hi} |> map(x -> x^2) |> sum"
        q = f"What is the sum of squares of all integers from {lo} to {hi} inclusive?"
        think = f"Map x -> x^2 over {lo}..{hi} and sum."
    elif kind == "filter_div":
        hi = rng.randint(20, 120)
        d = rng.choice([3, 4, 5, 6, 7])
        expr = f"1..{hi} |> filter(x -> x % {d} == 0) |> sum"
        q = f"Find the sum of all integers from 1 to {hi} that are divisible by {d}."
        think = f"Filter multiples of {d} in 1..{hi} and sum."
    elif kind == "factorial":
        n = rng.randint(5, 12)
        expr = f"let f(n) = if(n <= 1, 1, n * f(n - 1)); f({n})"
        q = f"Compute {n}! (the factorial of {n})."
        think = f"Define recursive factorial f(n) and evaluate f({n})."
    else:
        vals = [rng.randint(10, 99) for _ in range(rng.randint(4, 7))]
        arr = "[" + ", ".join(str(v) for v in vals) + "]"
        expr = f"round({arr} |> mean, 2)"
        q = f"What is the arithmetic mean of {', '.join(str(v) for v in vals)} (rounded to 2 decimals)?"
        think = "Pipe list into mean and round to 2 decimals."

    ans = safe_dsl(expr)
    turns = [
        Turn("system", DEFAULT_SYSTEM_PROMPT),
        Turn("user", q),
        Turn("think", think),
        Turn("call", format_tool_call("calc", expr=expr)),
        Turn("result", ans),
        Turn("assistant", f"The result is {ans}."),
    ]
    return VerifiableDSLTask(f"dsl_pipe_{idx}", "pipelines", q, ans, expr, turns)


def _make_symbolic_problem(rng: random.Random, idx: int) -> VerifiableDSLTask:
    kind = rng.choice(["linear", "quadratic", "system", "diff", "def_int", "factor"])
    if kind == "linear":
        a = rng.randint(2, 12)
        x_val = rng.randint(-15, 25)
        b = rng.randint(1, 30)
        c = a * x_val + b
        expr = f"solve({a}*x + {b} = {c}, x)"
        q = f"Solve for x: {a}x + {b} = {c}."
        think = "Solve linear equation symbolically with solve()."
    elif kind == "quadratic":
        r1 = rng.randint(-6, 6)
        r2 = rng.randint(-6, 6)
        while r2 == r1:
            r2 = rng.randint(-6, 6)
        b = -(r1 + r2)
        c = r1 * r2
        expr = f"solve(x^2 + ({b})*x + ({c}) = 0, x)"
        q = f"Find all real solutions x to x^2 + ({b})x + ({c}) = 0."
        think = "Solve quadratic equation via solve()."
    elif kind == "system":
        xv = rng.randint(1, 15)
        yv = rng.randint(1, 15)
        expr = f"solve([x + y = {xv + yv}, x - y = {xv - yv}], [x, y])"
        q = f"Solve the system: x + y = {xv + yv} and x - y = {xv - yv}."
        think = "Solve 2-variable linear system via solve()."
    elif kind == "diff":
        a = rng.randint(2, 9)
        b = rng.randint(1, 9)
        c = rng.randint(1, 12)
        expr = f"diff({a}*x^3 + {b}*x^2 + {c}*x, x)"
        q = f"What is the derivative of {a}x^3 + {b}x^2 + {c}x with respect to x?"
        think = "Differentiate polynomial with diff()."
    elif kind == "def_int":
        a = rng.choice([2, 3, 6])
        hi = rng.randint(2, 6)
        expr = f"int({a}*x^2, x, 0, {hi})"
        q = f"Evaluate the definite integral of {a}x^2 dx from 0 to {hi}."
        think = "Compute definite integral with int()."
    else:
        r = rng.randint(2, 12)
        expr = f"factor(x^2 - {r * r})"
        q = f"Factor the polynomial x^2 - {r * r}."
        think = "Factor expression via factor()."

    ans = safe_dsl(expr)
    turns = [
        Turn("system", DEFAULT_SYSTEM_PROMPT),
        Turn("user", q),
        Turn("think", think),
        Turn("call", format_tool_call("calc", expr=expr)),
        Turn("result", ans),
        Turn("assistant", f"The result is {ans}."),
    ]
    return VerifiableDSLTask(f"dsl_sym_{idx}", "symbolic", q, ans, expr, turns)


def generate_dsl_tasks(n_tasks: int, *, seed: int = 2026) -> list[VerifiableDSLTask]:
    """Generate a balanced 4-level curriculum of ToolExecutor-verified coregrad-DSL trajectories."""
    rng = random.Random(seed)
    makers = (_make_logic_dsl, _make_binding_problem, _make_pipeline_problem, _make_symbolic_problem)
    tasks = [makers[i % len(makers)](rng, i) for i in range(n_tasks)]
    rng.shuffle(tasks)
    return tasks
