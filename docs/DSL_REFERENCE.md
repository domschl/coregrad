# `coregrad-DSL` Language Reference

> **Implementation:** [`src/coregrad/tools/dsl.py`](../src/coregrad/tools/dsl.py) · **Curriculum Generator:** [`src/coregrad/synthetic/dsl_gen.py`](../src/coregrad/synthetic/dsl_gen.py) · **Unit Tests:** [`tests/test_dsl.py`](../tests/test_dsl.py)

`coregrad-DSL` is a unified, 100% sandboxed, terse ASCII-functional language for **exact arithmetic, variable bindings, array pipelines, bounded recursion, symbolic algebra/calculus (SymPy), and formal forward-chaining logic (`prove`)**.

It is designed as a single universal coprocessor for compact (`25M–125M`) cognitive cores and is wired transparently behind the canonical `calc(expr="...")` tool in [`executor.py`](../src/coregrad/tools/executor.py) — preserving `coregrad`'s minimal 4-tool interface (`search`, `read`, `calc`, `list_docs`).

```python
from coregrad.tools.dsl import safe_dsl

safe_dsl("1..10 |> map(x -> x^2) |> sum")                    # "385"
safe_dsl("let f(n) = if(n <= 1, 1, n * f(n - 1)); f(7)")     # "5040"
safe_dsl("solve([x + y = 10, x - y = 2], [x, y])")           # "{x: 6, y: 4}"
safe_dsl("prove([ x : A, A -> B, B -/> C ], x : C)")         # "False"
```

---

## 1. Design Law: BPE-Aligned Operator Spacing

In byte-level BPE tokenizers, words inside English prose almost always begin with a leading space (`" treldon"`). Omitting spaces around DSL operators (`"treldon->plendor"`) strips that leading space and produces completely different token IDs — forcing small models to spell-translate unseen identifiers across BPE boundaries instead of copying token IDs directly.

**Recommended canonical formatting:** pad operators (`->`, `-/>`, `&`, `:`, `|>`, `=`, `+`, `-`, `*`, `/`) with single spaces, e.g.:

```text
prove([ x : quoril, quoril -> glorim, glorim -/> drelon ], x : drelon)
```

In our Phase 5.1 ablations ([`EXPERIMENTS.md`](EXPERIMENTS.md) §3), BPE-aligned spacing used *fewer* tokens than tight `a->b` while lifting unseen-vocabulary logic accuracy by **`+34%` ID / `+20%` OOD**. The parser accepts both compact and spaced forms identically.

---

## 2. Program Structure, Literals & Bindings

A `coregrad-DSL` program is one or more statements separated by `;`. Statements execute left-to-right in a shared environment, and the **value of the final statement** is formatted and returned.

### Literals & Built-in Constants
| Kind | Syntax | Example |
|---|---|---|
| Integer | digits | `42`, `0` |
| Float | decimal digits | `3.14`, `1.19` |
| Boolean | `True`, `False` | `True` |
| List | `[e1, e2, ...]` (trailing comma allowed) | `[10, 20, 30]` |
| Inclusive Range | `a..b` (step `+1` if `b >= a`, `-1` if `b < a`) | `1..5` → `[1, 2, 3, 4, 5]`; `3..1` → `[3, 2, 1]` |
| Built-in Constants | `pi`, `e` | `round(pi * 4^2, 2)` → `50.27` |

### Variable & Function Bindings (`let`)
- **Value binding:** `let name = expr`
  ```text
  let sub = 8 * 26 + 10 * 42; let net = sub * 0.9; round(net * 1.19, 2)
  ```
- **Function binding (supports recursion):** `let name(p1, p2, ...) = expr`
  ```text
  let fact(n) = if(n <= 1, 1, n * fact(n - 1)); fact(7)
  ```
  Named `let_fn` closures capture the enclosing environment by reference so self-recursion works naturally up to the recursion depth cap (`32`).
- **Anonymous lambdas:**
  - Single parameter: `x -> expr`
  - Multi-parameter: `(a, b) -> expr`
  ```text
  1..5 |> fold(1, (acc, x) -> acc * x)
  ```

---

## 3. Operators & Precedence

Listed from **lowest** to **highest** precedence:

| Precedence | Operator(s) | Associativity | Notes |
|---:|---|---|---|
| 1 (lowest) | `x -> e`, `(a, b) -> e` | Right | Lambda expression |
| 2 | `\|>` | Left | Pipeline: injects LHS as the **first argument** of RHS |
| 3 | `or` | Left | Short-circuit boolean OR |
| 4 | `and` | Left | Short-circuit boolean AND |
| 5 | `==`, `!=`, `<`, `<=`, `>`, `>=` | Left | Standard comparisons |
| 6 | `..` | Non-assoc | Inclusive integer range (`a..b`, max length `1000`) |
| 7 | `+`, `-` | Left | Addition, subtraction |
| 8 | `*`, `/`, `//`, `%` | Left | Multiplication, true div, floor div, modulo |
| 9 | unary `+`, `-`, `not` | Right | Unary sign and boolean negation |
| 10 (highest) | `^`, `**` | Right | Exponentiation (`abs(exp) <= 256`) |

Parentheses `( ... )` override precedence as usual.

---

## 4. Conditionals (`if`) & Short-Circuiting

Conditionals use a **functional, strictly short-circuiting expression**:

```text
if(cond, then_expr, else_expr)
```

- **Only the selected branch is evaluated.** This is mandatory for terminating recursion (e.g., `if(n <= 1, 1, n * f(n - 1))` never recurses once `n <= 1`) and avoiding division-by-zero guards (`if(x == 0, 0, 100 / x)`).
- **Multi-branch (`elif`) logic** nests cleanly as expressions:
  ```text
  let grade(s) = if(s >= 90, 4, if(s >= 80, 3, if(s >= 70, 2, 1))); grade(84)
  ```
- **Boolean operators `and` / `or`** also short-circuit left-to-right.

---

## 5. Pipelines (`|>`) & Built-in Function Reference

The pipe operator `lhs |> rhs` evaluates `lhs` and passes it as the **first argument** to `rhs`:
- `xs |> sum` is equivalent to `sum(xs)`
- `xs |> map(x -> x^2)` is equivalent to `map(xs, x -> x^2)`
- `xs |> fold(0, (a, x) -> a + x)` is equivalent to `fold(xs, 0, (a, x) -> a + x)`

### Higher-Order Array Primitives
| Function | Signature | Description |
|---|---|---|
| `map` | `map(list, fn)` | Apply `fn(x)` to each element |
| `filter` | `filter(list, pred)` | Keep elements where `pred(x)` is truthy |
| `fold` | `fold(list, init, fn)` | Left fold with accumulator `fn(acc, x)` |

### List & Aggregation Primitives
| Function | Signature | Description |
|---|---|---|
| `sum` | `sum(list)` or `sum(a, b, ...)` | Sum of elements |
| `prod` | `prod(list)` or `prod(a, b, ...)` | Product of elements |
| `mean` | `mean(list)` or `mean(a, b, ...)` | Arithmetic mean (errors on empty list) |
| `min`, `max` | `min(list)` / `max(list)` or variadic | Minimum / maximum element |
| `len` | `len(list)` | Length of list |
| `sort` | `sort(list)` | Ascending sorted copy |
| `rev` | `rev(list)` | Reversed copy |
| `uniq` | `uniq(list)` | Order-preserving deduplication |
| `take` | `take(list, n)` | First `n` elements (`list[:n]`) |
| `drop` | `drop(list, n)` | Elements after first `n` (`list[n:]`) |
| `range` | `range(n)` or `range(a, b)` | Python-style half-open integer range `[a, b)` |
| `any`, `all` | `any(list)` / `all(list)` | Boolean existential / universal check |

### Numeric & Math Primitives
| Function | Signature | Description |
|---|---|---|
| `round` | `round(x)` or `round(x, ndigits)` | Round to nearest integer or `ndigits` decimals |
| `abs` | `abs(x)` | Absolute value |
| `sqrt` | `sqrt(x)` | Square root |
| `floor`, `ceil` | `floor(x)` / `ceil(x)` | Floor / ceiling integer |
| `gcd`, `lcm` | `gcd(a, b, ...)` / `lcm(a, b, ...)` | Greatest common divisor / least common multiple |
| `log` | `log(x)` or `log(x, base)` | Natural log, or log in `base` |
| `exp` | `exp(x)` | $e^x$ |
| `sin`, `cos` | `sin(x)` / `cos(x)` | Trigonometric functions (radians) |

### Output Formatting
- Booleans format as `True` / `False`.
- Near-integer floats (`|v - round(v)| < 1e-9` and `|v| < 1e15`) format cleanly as integers (`20.0` → `20`).
- Other floats format with `:.10g`. Non-finite results (`NaN`, `Inf`) raise `DSLError`.

---

## 6. Symbolic Algebra & Calculus (SymPy Bridge)

Top-level symbolic calls delegate to an allowlisted SymPy parser (`^` is accepted as exponentiation and normalized back to `^` in output). Any numeric variables bound earlier via `let` in the same `;`-separated program are automatically substituted into the symbolic environment, and any other identifier is treated as a symbolic variable.

Allowed symbolic constants/functions: `sin`, `cos`, `tan`, `exp`, `log`, `ln`, `sqrt`, `abs`, `pi`, `e`, `oo` ($\infty$).

| Primitive | Signature | Example → Output |
|---|---|---|
| `solve` | `solve(eq, var)` | `solve(3*x + 7 = 22, x)` → `5` |
| `solve` (poly) | `solve(eq, var)` | `solve(x^2 - 5*x + 6 = 0, x)` → `[2, 3]` |
| `solve` (system) | `solve([eq1, eq2], [x, y])` | `solve([x + y = 10, x - y = 2], [x, y])` → `{x: 6, y: 4}` |
| `diff` | `diff(expr, var[, order])` (`1..5`) | `diff(x^3 + 2*x^2, x)` → `3*x^2 + 4*x` |
| `int` | `int(expr, var)` | `int(3*x^2, x)` → `x^3` |
| `int` (definite) | `int(expr, var, a, b)` | `int(x^2, x, 0, 3)` → `9` |
| `simplify` | `simplify(expr)` | `simplify((x^2 - 1)/(x - 1))` → `x + 1` |
| `expand` | `expand(expr)` | `expand((x + 2)^2)` → `x^2 + 4*x + 4` |
| `factor` | `factor(expr)` | `factor(x^2 - 9)` → `(x - 3)*(x + 3)` |
| `subs` | `subs(expr, var, val)` | `subs(x^2 + 2*x, x, 3)` → `15` |
| `limit` | `limit(expr, var, point)` | `limit(sin(x)/x, x, 0)` → `1` |

*Note:* Single-element solution lists from `solve` unwrap cleanly (`[5]` → `5`), while multi-root solutions remain `[r1, r2]` and systems return `{x: ..., y: ...}` sorted by variable name.

---

## 7. Formal Logic Prover (`prove`)

`prove(...)` runs a deterministic, sound forward-chaining fixpoint solver over unary implications, multi-antecedent conjunctive Horn rules, and symmetric exclusions — specifically designed to eliminate multi-hop reasoning drift and affirming-the-consequent fallacies.

### Syntax
```text
prove([ clause_1, clause_2, ... ], entity : TargetPredicate)
```

### Supported Clause Types
| Clause | Meaning |
|---|---|
| `x : A` | Entity `x` has predicate `A` (positive base fact) |
| `x : !A` | Entity `x` does **not** have predicate `A` (negative base fact) |
| `A -> B` | Every `A` is a `B` (unary implication) |
| `A & B -> C` | Anything that is both `A` and `B` is `C` (2+ antecedents supported) |
| `A -/> B` | No `A` is a `B` (symmetric exclusion: derives `!B` from `A` and `!A` from `B`) |

### Return Values
- `True` — `TargetPredicate` is provably entailed for `entity`.
- `False` — `TargetPredicate` is provably excluded/disproven for `entity`.
- `Unknown` — Neither entailed nor disproven (e.g., broken chain or affirming-the-consequent trap `prove([ x : A, A -> B, C -> B ], x : C)` → `Unknown`).
- `Contradiction` — Both positive and negative facts are derived for `TargetPredicate`.

---

## 8. Security Sandbox & Resource Limits

`coregrad-DSL` uses a hand-written lexer, recursive-descent parser, and tree-walking evaluator — **never** Python `eval`/`exec`, with zero import, attribute-access, or filesystem surface. Every failure raises `DSLError`, which `ToolExecutor` returns cleanly inside `<|result|>Error: ...<|/result|>`.

| Guard | Limit |
|---|---:|
| Max program source length | `1,024` chars |
| Max symbolic expression length | `256` chars |
| Max evaluator AST ticks (gas) | `2,500` steps |
| Max function recursion depth | `32` frames |
| Max list / range length | `1,000` elements |
| Max exponent magnitude | `256` |
| Max derivative order (`diff`) | `5` |
| Max `prove(...)` clauses / fixpoint iterations | `128` clauses / `256` iterations |

---

## 9. End-to-End Tool-Protocol Examples

How `coregrad` invokes `coregrad-DSL` inside the canonical tool protocol:

```text
<|user|>What is the sum of squares of even numbers from 1 to 20?<|/user|>
<|think|>Filter even numbers in 1..20, square them, and sum via calc.<|/think|>
<|call|>calc(expr="1..20 |> filter(x -> x % 2 == 0) |> map(x -> x^2) |> sum")<|/call|>
<|result|>1540<|/result|>
<|assistant|>The sum of squares of the even numbers from 1 to 20 is 1540.<|/assistant|>
```

```text
<|user|>Rules: Bram is a quoril. Every glorim is a kormel. Every quoril is a glorim. No kormel is a drelon.
Is Bram a drelon? Answer True, False, or Unknown.<|/user|>
<|call|>calc(expr="prove([ x : quoril, glorim -> kormel, quoril -> glorim, kormel -/> drelon ], x : drelon)")<|/call|>
<|result|>False<|/result|>
<|assistant|>False<|/assistant|>
```
