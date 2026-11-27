"""Unit tests for coregrad-DSL (unified arithmetic, arrays/pipes, recursion, SymPy, logic prover)."""

from __future__ import annotations

import pytest

from coregrad.synthetic.dsl_gen import generate_dsl_tasks
from coregrad.synthetic.logic_gen import generate_logic_tasks
from coregrad.tools.dsl import DSLError, safe_dsl
from coregrad.tools.executor import ToolExecutor
from coregrad.tools.protocol import extract_tool_calls, render_trajectory


def test_dsl_arithmetic_variables_and_pipes() -> None:
    assert safe_dsl("2 + 3 * 4") == "14"
    assert safe_dsl("let sub = 8 * 26 + 10 * 42; round(sub * 1.19, 2)") == "747.32"
    assert safe_dsl("1..10 |> map(x -> x^2) |> sum") == "385"
    assert safe_dsl("1..20 |> filter(x -> x % 2 == 0) |> len") == "10"
    assert safe_dsl("fold(1..5, 1, (a, x) -> a * x)") == "120"
    assert safe_dsl("[10, 20, 30] |> mean") == "20"


def test_dsl_bounded_recursion_and_gas_limits() -> None:
    assert safe_dsl("let f(n) = if(n <= 1, 1, n * f(n - 1)); f(7)") == "5040"
    assert safe_dsl("let fib(n) = if(n <= 1, n, fib(n - 1) + fib(n - 2)); fib(10)") == "55"
    with pytest.raises(DSLError):
        safe_dsl("let loop(n) = loop(n + 1); loop(0)")
    with pytest.raises(DSLError):
        safe_dsl("2 ^ 1000")
    with pytest.raises(DSLError):
        safe_dsl("1..100000")


def test_dsl_sympy_algebra_and_calculus() -> None:
    assert safe_dsl("solve(3*x + 7 = 22, x)") == "5"
    assert safe_dsl("solve(x^2 - 5*x + 6 = 0, x)") == "[2, 3]"
    assert safe_dsl("solve([x + y = 10, x - y = 2], [x, y])") == "{x: 6, y: 4}"
    assert safe_dsl("diff(x^3 + 2*x^2, x)") == "3*x^2 + 4*x"
    assert safe_dsl("int(x^2, x, 0, 3)") == "9"
    assert safe_dsl("simplify((x^2 - 1)/(x - 1))") == "x + 1"
    assert safe_dsl("factor(x^2 - 9)") == "(x - 3)*(x + 3)"
    assert safe_dsl("limit(sin(x)/x, x, 0)") == "1"


def test_dsl_formal_logic_prover() -> None:
    assert safe_dsl("prove([x:A, A->B, B->C, C->D], x:D)") == "True"
    assert safe_dsl("prove([x:A, A->B, B-/>C], x:C)") == "False"
    # Affirming-the-consequent trap: A->B and C->B does NOT prove C
    assert safe_dsl("prove([x:A, A->B, C->B], x:C)") == "Unknown"
    assert safe_dsl("prove([x:A, x:B, A&B->C, C->D], x:D)") == "True"


def test_dsl_curriculum_generator_executes_via_tool_executor() -> None:
    tasks = generate_dsl_tasks(32, seed=123)
    assert {t.level for t in tasks} == {"logic", "bindings", "pipelines", "symbolic"}
    ex = ToolExecutor(None)
    for t in tasks:
        rendered = render_trajectory(t.turns)
        calls = extract_tool_calls(rendered)
        assert len(calls) == 1
        res = ex.execute_raw(calls[0])
        assert res.ok, f"Failed {t.task_id}: {res.output}"
        assert res.output == t.expected_answer


def test_logic_gen_dsl_mode_executes_exact_prover() -> None:
    tasks = generate_logic_tasks(24, depths=(2, 5, 10), split="eval", seed=77, notation="dsl", cot=False)
    ex = ToolExecutor(None)
    for t in tasks:
        rendered = render_trajectory(t.turns)
        calls = extract_tool_calls(rendered)
        assert len(calls) == 1
        res = ex.execute_raw(calls[0])
        assert res.ok, f"Failed {t.task_id}: {res.output}"
        assert res.output == t.expected_answer
