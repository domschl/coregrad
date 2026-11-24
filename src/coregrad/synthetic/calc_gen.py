"""Deterministic synthetic calculator & multi-step reasoning trajectory generator."""

from __future__ import annotations

import random
from dataclasses import dataclass

from coregrad.tools.calc import safe_calc
from coregrad.tools.protocol import DEFAULT_SYSTEM_PROMPT, Turn, format_tool_call


@dataclass(frozen=True)
class VerifiableCalcTask:
    task_id: str
    category: str
    question: str
    expected_answer: str
    turns: list[Turn]
    requires_tool: bool = True


def _make_year_diff(rng: random.Random, idx: int) -> VerifiableCalcTask:
    y1 = rng.randint(1520, 1985)
    y2 = y1 + rng.randint(7, 240)
    expr = f"{y2} - {y1}"
    ans = safe_calc(expr)
    templates = [
        f"How many years elapsed between {y1} and {y2}?",
        f"Calculate the number of years from {y1} to {y2}.",
        f"If an event happened in {y1} and another in {y2}, how many years apart were they?",
    ]
    q = rng.choice(templates)
    call = format_tool_call("calc", expr=expr)
    turns = [
        Turn("system", DEFAULT_SYSTEM_PROMPT),
        Turn("user", q),
        Turn("think", f"Subtract {y1} from {y2} using calc."),
        Turn("call", call),
        Turn("result", ans),
        Turn("assistant", f"{ans} years elapsed between {y1} and {y2}."),
    ]
    return VerifiableCalcTask(f"calc_ydiff_{idx}", "calc_1step", q, ans, turns)


def _make_percent_change(rng: random.Random, idx: int) -> VerifiableCalcTask:
    base = rng.randint(45, 2500)
    pct = rng.randint(3, 45)
    increase = rng.choice([True, False])
    factor = f"1 + {pct}/100" if increase else f"1 - {pct}/100"
    expr = f"round({base} * ({factor}), 2)"
    ans = safe_calc(expr)
    verb = "increased" if increase else "reduced"
    q = f"A quantity of {base} is {verb} by {pct}%. What is the new value (rounded to 2 decimal places)?"
    call = format_tool_call("calc", expr=expr)
    turns = [
        Turn("system", DEFAULT_SYSTEM_PROMPT),
        Turn("user", q),
        Turn("think", f"Multiply {base} by ({factor}) and round to 2 decimals."),
        Turn("call", call),
        Turn("result", ans),
        Turn("assistant", f"After being {verb} by {pct}%, the new value is {ans}."),
    ]
    return VerifiableCalcTask(f"calc_pct_{idx}", "calc_1step", q, ans, turns)


def _make_unit_conversion(rng: random.Random, idx: int) -> VerifiableCalcTask:
    kind = rng.choice(["c_to_k", "c_to_f", "km_to_mi", "hms_to_sec"])
    if kind == "c_to_k":
        c = round(rng.uniform(-40.0, 450.0), 1)
        expr = f"round({c} + 273.15, 2)"
        ans = safe_calc(expr)
        q = f"Convert {c} degrees Celsius to Kelvin."
        think = "Add 273.15 to convert Celsius to Kelvin."
        final = f"{c} °C equals {ans} K."
    elif kind == "c_to_f":
        c = rng.randint(-30, 150)
        expr = f"round({c} * 9 / 5 + 32, 2)"
        ans = safe_calc(expr)
        q = f"What is {c} degrees Celsius in Fahrenheit?"
        think = "Use F = C * 9/5 + 32."
        final = f"{c} °C equals {ans} °F."
    elif kind == "km_to_mi":
        km = rng.randint(12, 950)
        expr = f"round({km} * 0.621371, 2)"
        ans = safe_calc(expr)
        q = f"Convert {km} kilometers to miles (rounded to 2 decimal places)."
        think = "Multiply kilometers by 0.621371."
        final = f"{km} km is approximately {ans} miles."
    else:
        h = rng.randint(1, 12)
        m = rng.randint(1, 59)
        s = rng.randint(1, 59)
        expr = f"{h} * 3600 + {m} * 60 + {s}"
        ans = safe_calc(expr)
        q = f"How many seconds are in {h} hours, {m} minutes, and {s} seconds?"
        think = "Convert hours and minutes to seconds and sum them."
        final = f"There are {ans} seconds in {h}h {m}m {s}s."

    call = format_tool_call("calc", expr=expr)
    turns = [
        Turn("system", DEFAULT_SYSTEM_PROMPT),
        Turn("user", q),
        Turn("think", think),
        Turn("call", call),
        Turn("result", ans),
        Turn("assistant", final),
    ]
    return VerifiableCalcTask(f"calc_unit_{idx}", "calc_1step", q, ans, turns)


def _make_multistep_order(rng: random.Random, idx: int) -> VerifiableCalcTask:
    n1, p1 = rng.randint(3, 18), rng.randint(12, 95)
    n2, p2 = rng.randint(2, 15), rng.randint(8, 75)
    tax = rng.choice([5, 7, 8, 10, 12, 15, 19])
    expr1 = f"{n1} * {p1} + {n2} * {p2}"
    subtotal = safe_calc(expr1)
    expr2 = f"round({subtotal} * (1 + {tax}/100), 2)"
    total = safe_calc(expr2)

    q = (
        f"An archive orders {n1} hardcover books at ${p1} each and {n2} journals at ${p2} each. "
        f"With {tax}% tax applied to the subtotal, what is the final total cost?"
    )
    turns = [
        Turn("system", DEFAULT_SYSTEM_PROMPT),
        Turn("user", q),
        Turn("think", "First compute the subtotal for both items, then add tax."),
        Turn("call", format_tool_call("calc", expr=expr1)),
        Turn("result", subtotal),
        Turn("think", f"The subtotal is ${subtotal}. Now apply {tax}% tax."),
        Turn("call", format_tool_call("calc", expr=expr2)),
        Turn("result", total),
        Turn("assistant", f"The subtotal is ${subtotal}, and after {tax}% tax the final total is ${total}."),
    ]
    return VerifiableCalcTask(f"calc_order_{idx}", "calc_2step", q, total, turns)


def _make_multistep_geometry(rng: random.Random, idx: int) -> VerifiableCalcTask:
    a = rng.randint(5, 60)
    b = rng.randint(5, 60)
    expr1 = f"{a}**2 + {b}**2"
    sq_sum = safe_calc(expr1)
    expr2 = f"round(sqrt({sq_sum}), 2)"
    hyp = safe_calc(expr2)
    expr3 = f"round({a} + {b} + {hyp}, 2)"
    perim = safe_calc(expr3)

    q = (
        f"A right triangle has legs of length {a} cm and {b} cm. "
        f"Compute its hypotenuse (rounded to 2 decimal places) and then its total perimeter."
    )
    turns = [
        Turn("system", DEFAULT_SYSTEM_PROMPT),
        Turn("user", q),
        Turn("think", "Use the Pythagorean theorem: first sum the squares of the legs, take sqrt, then sum all three sides."),
        Turn("call", format_tool_call("calc", expr=expr1)),
        Turn("result", sq_sum),
        Turn("call", format_tool_call("calc", expr=expr2)),
        Turn("result", hyp),
        Turn("call", format_tool_call("calc", expr=expr3)),
        Turn("result", perim),
        Turn("assistant", f"The hypotenuse is {hyp} cm and the total perimeter is {perim} cm."),
    ]
    return VerifiableCalcTask(f"calc_geom_{idx}", "calc_3step", q, perim, turns)


def _make_compound_growth(rng: random.Random, idx: int) -> VerifiableCalcTask:
    principal = rng.randint(500, 12000)
    rate = rng.randint(3, 12)
    years = rng.randint(3, 15)
    expr1 = f"round({principal} * (1 + {rate}/100)**{years}, 2)"
    final_amt = safe_calc(expr1)
    expr2 = f"round({final_amt} - {principal}, 2)"
    gain = safe_calc(expr2)

    q = (
        f"If ${principal} grows at {rate}% annual compound interest for {years} years, "
        f"how much total interest is earned (rounded to 2 decimal places)?"
    )
    turns = [
        Turn("system", DEFAULT_SYSTEM_PROMPT),
        Turn("user", q),
        Turn("think", "Compute the final compounded balance first, then subtract the original principal."),
        Turn("call", format_tool_call("calc", expr=expr1)),
        Turn("result", final_amt),
        Turn("call", format_tool_call("calc", expr=expr2)),
        Turn("result", gain),
        Turn("assistant", f"The balance reaches ${final_amt}, earning ${gain} in total interest."),
    ]
    return VerifiableCalcTask(f"calc_comp_{idx}", "calc_2step", q, gain, turns)


def generate_calc_tasks(n: int, seed: int = 101) -> list[VerifiableCalcTask]:
    """Generate n deterministic, exact 1-to-3-step calculator trajectories."""
    rng = random.Random(seed)
    builders = [
        _make_year_diff,
        _make_percent_change,
        _make_unit_conversion,
        _make_multistep_order,
        _make_multistep_geometry,
        _make_compound_growth,
    ]
    return [rng.choice(builders)(rng, i) for i in range(n)]
