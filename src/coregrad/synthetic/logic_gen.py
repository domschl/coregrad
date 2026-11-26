"""Deterministic procedural multi-step syllogism & Horn-clause logic generator.

Designed as a zero-memorization cognitive-core benchmark testing how deeply compact dense
and looped Transformers can execute formal deductive rules across controllable depths D >= 2:
- Strictly disjoint fictional entity & predicate vocabularies between train and eval splits
- Shuffled rules with acyclic distractors (model cannot exploit linear rule order)
- Verified mathematically by an exact forward-chaining prover (True / False / Unknown)
- Includes affirming-the-consequent fallacy traps (reversed implication -> Unknown)
- Supports both explicit <|think|> proof chains (cot=True) and direct latent-depth answers (cot=False)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from coregrad.tools.protocol import DEFAULT_SYSTEM_PROMPT, Turn

_TRAIN_ENTITIES = (
    "Kora", "Tavin", "Miren", "Dalen", "Sora", "Vexel", "Nira", "Orin",
    "Luma", "Riven", "Talia", "Caden", "Zera", "Fenn", "Yara", "Jorin",
)
_EVAL_ENTITIES = (
    "Bram", "Selis", "Oren", "Vira", "Kael", "Novi", "Toren", "Ayla",
    "Drax", "Lyra", "Soren", "Mina", "Ronan", "Eris", "Pelin", "Cora",
)

# Strictly disjoint fictional category stems for train vs eval (zero world-knowledge shortcuts)
_TRAIN_PREDS = (
    "zorplin", "velmar", "krendal", "tovix", "marlen", "sylvok", "brenar", "calvix",
    "dornel", "fenrik", "galmor", "helvix", "jorven", "kalnor", "lorvix", "melkor",
    "norval", "orvix", "pelmar", "quendal", "relvix", "sorval", "talven", "ulvix",
    "vornal", "welvix", "xarlen", "yorval", "zelvix", "arvend", "belnor", "corvix",
)
_EVAL_PREDS = (
    "quoril", "tanvex", "brinor", "vaskel", "drelon", "mirvax", "solren", "kaltor",
    "nepvix", "glorim", "tharnel", "vextor", "primalon", "cenvix", "haldor", "zintor",
    "fralen", "mordix", "stelnor", "gravix", "plendor", "trixon", "brenvix", "yaldor",
    "kormel", "senvix", "droval", "larnix", "fenvor", "parvix", "treldon", "wexnor",
)


@dataclass(frozen=True)
class VerifiableLogicTask:
    task_id: str
    category: str  # syllogism_pos | syllogism_neg | syllogism_unknown | horn_conj
    depth: int
    question: str
    expected_answer: str  # "True" | "False" | "Unknown"
    turns: list[Turn]


def _prove_entailment(
    base_facts: set[str],
    unary_rules: list[tuple[str, str]],
    neg_rules: list[tuple[str, str]],
    conj_rules: list[tuple[str, str, str]],
    target: str,
) -> str:
    """Exact sound forward-chaining prover over unary, conjunctive Horn, and exclusion rules."""
    known_pos = set(base_facts)
    known_neg: set[str] = set()
    changed = True
    while changed:
        changed = False
        for a, b in unary_rules:
            if a in known_pos and b not in known_pos:
                known_pos.add(b)
                changed = True
        for a, b, c in conj_rules:
            if a in known_pos and b in known_pos and c not in known_pos:
                known_pos.add(c)
                changed = True
        for a, b in neg_rules:
            if a in known_pos and b not in known_neg:
                known_neg.add(b)
                changed = True
            if b in known_pos and a not in known_neg:
                known_neg.add(a)
                changed = True
    if target in known_pos and target in known_neg:
        raise ValueError(f"Contradictory rulebase for target {target!r}")
    if target in known_pos:
        return "True"
    if target in known_neg:
        return "False"
    return "Unknown"


def _render_unary(rng: random.Random, a: str, b: str) -> str:
    return rng.choice(
        [
            f"All {a}s are {b}s.",
            f"Every {a} is a {b}.",
            f"If something is a {a}, then it is a {b}.",
        ]
    )


def _render_neg(rng: random.Random, a: str, b: str) -> str:
    return rng.choice(
        [
            f"No {a} is a {b}.",
            f"Nothing that is a {a} can be a {b}.",
            f"If something is a {a}, then it is not a {b}.",
        ]
    )


def _render_conj(rng: random.Random, a: str, b: str, c: str) -> str:
    return rng.choice(
        [
            f"Anything that is both a {a} and a {b} is a {c}.",
            f"If something is a {a} and also a {b}, then it is a {c}.",
        ]
    )


def _build_single_task(
    rng: random.Random,
    *,
    idx: int,
    depth: int,
    category: str,
    entities: tuple[str, ...],
    preds: tuple[str, ...],
    n_distractors: int,
    cot: bool,
) -> VerifiableLogicTask:
    if depth < 2:
        raise ValueError(f"depth must be >= 2, got {depth}")
    entity = rng.choice(entities)

    # Reserve enough unique predicates for main chain + optional branch + distractors
    need = depth + 3 + max(2, n_distractors + 1)
    if need > len(preds):
        # Extend dynamically with deterministic suffixes if very large depth requested
        pool = list(preds) + [f"{p}{k}" for k in range(2, 6) for p in preds]
    else:
        pool = list(preds)
    chosen = rng.sample(pool, need)

    unary_rules: list[tuple[str, str]] = []
    neg_rules: list[tuple[str, str]] = []
    conj_rules: list[tuple[str, str, str]] = []
    base_facts: set[str] = set()
    think_steps: list[str] = []

    if category == "syllogism_pos":
        chain = chosen[: depth + 1]
        base_facts.add(chain[0])
        for i in range(depth):
            unary_rules.append((chain[i], chain[i + 1]))
            think_steps.append(
                f"Step {i + 1}: {entity} is a {chain[i]} and every {chain[i]} is a {chain[i + 1]}, so {entity} is a {chain[i + 1]}."
            )
        target = chain[depth]
        expected = "True"
        used_end = depth + 1

    elif category == "syllogism_neg":
        chain = chosen[: depth + 1]
        base_facts.add(chain[0])
        for i in range(depth - 1):
            unary_rules.append((chain[i], chain[i + 1]))
            think_steps.append(
                f"Step {i + 1}: {entity} is a {chain[i]} and every {chain[i]} is a {chain[i + 1]}, so {entity} is a {chain[i + 1]}."
            )
        neg_rules.append((chain[depth - 1], chain[depth]))
        think_steps.append(
            f"Step {depth}: {entity} is a {chain[depth - 1]} and no {chain[depth - 1]} is a {chain[depth]}, so {entity} is not a {chain[depth]}."
        )
        target = chain[depth]
        expected = "False"
        used_end = depth + 1

    elif category == "syllogism_unknown":
        chain = chosen[: depth + 1]
        base_facts.add(chain[0])
        # Break or reverse one link in the chain (affirming-the-consequent trap)
        break_at = rng.randint(max(1, depth // 2), depth - 1)
        for i in range(depth):
            if i == break_at:
                # Reversed implication: All chain[i+1] are chain[i] (does NOT imply chain[i] -> chain[i+1])
                unary_rules.append((chain[i + 1], chain[i]))
            else:
                unary_rules.append((chain[i], chain[i + 1]))
        for i in range(break_at):
            think_steps.append(
                f"Step {i + 1}: {entity} is a {chain[i]} and every {chain[i]} is a {chain[i + 1]}, so {entity} is a {chain[i + 1]}."
            )
        think_steps.append(
            f"Step {break_at + 1}: We know {entity} is a {chain[break_at]}, and every {chain[break_at + 1]} is a {chain[break_at]}, "
            f"but that does not prove {entity} is a {chain[break_at + 1]}. Therefore {chain[depth]} cannot be proven or disproven."
        )
        target = chain[depth]
        expected = "Unknown"
        used_end = depth + 1

    elif category == "horn_conj":
        # Two-branch chain joining at a conjunctive Horn rule (A_k AND B -> A_{k+1})
        chain = chosen[: depth + 1]
        side_p = chosen[depth + 1]
        join_at = max(1, depth // 2)
        base_facts.add(chain[0])
        base_facts.add(side_p)
        for i in range(depth):
            if i == join_at:
                conj_rules.append((chain[i], side_p, chain[i + 1]))
                think_steps.append(
                    f"Step {i + 1}: {entity} is both a {chain[i]} and a {side_p}, so {entity} is a {chain[i + 1]}."
                )
            else:
                unary_rules.append((chain[i], chain[i + 1]))
                think_steps.append(
                    f"Step {i + 1}: {entity} is a {chain[i]} and every {chain[i]} is a {chain[i + 1]}, so {entity} is a {chain[i + 1]}."
                )
        target = chain[depth]
        expected = "True"
        used_end = depth + 2
    else:
        raise ValueError(f"Unsupported category: {category!r}")

    # Add acyclic distractor rules using strictly disjoint predicates so target truth never changes
    dist_pool = chosen[used_end:]
    for j in range(min(n_distractors, len(dist_pool) - 1)):
        if j % 3 == 0:
            # Side branch off an intermediate chain node into an unused distractor node
            anchor = chosen[rng.randint(0, depth - 1)]
            unary_rules.append((anchor, dist_pool[j]))
        else:
            unary_rules.append((dist_pool[j], dist_pool[j + 1]))

    # Soundness check via exact forward-chaining prover
    verified = _prove_entailment(base_facts, unary_rules, neg_rules, conj_rules, target)
    assert verified == expected, f"Prover mismatch: expected={expected}, verified={verified}"

    # Render facts and shuffled rules
    rule_sentences = (
        [_render_unary(rng, a, b) for a, b in unary_rules]
        + [_render_neg(rng, a, b) for a, b in neg_rules]
        + [_render_conj(rng, a, b, c) for a, b, c in conj_rules]
    )
    rng.shuffle(rule_sentences)
    fact_sentences = [f"{entity} is a {f}." for f in sorted(base_facts)]
    rng.shuffle(fact_sentences)

    premise_block = " ".join(fact_sentences + rule_sentences)
    question = (
        f"Rules and facts: {premise_block}\n"
        f"Question: Based strictly on the rules above, is {entity} a {target}? "
        f"Answer with exactly one word: True, False, or Unknown."
    )

    turns: list[Turn] = [Turn("system", DEFAULT_SYSTEM_PROMPT), Turn("user", question)]
    if cot:
        turns.append(Turn("think", " ".join(think_steps) + f" Conclusion: {expected}."))
    turns.append(Turn("assistant", expected))

    return VerifiableLogicTask(
        task_id=f"logic_{category}_d{depth}_{idx}",
        category=category,
        depth=depth,
        question=question,
        expected_answer=expected,
        turns=turns,
    )


def generate_logic_tasks(
    n_tasks: int,
    *,
    depths: tuple[int, ...] = (2, 3, 4, 5),
    split: str = "train",
    seed: int = 42,
    n_distractors: int = 3,
    cot: bool = True,
) -> list[VerifiableLogicTask]:
    """Generate a balanced, prover-verified dataset of multi-step syllogism & Horn-clause tasks."""
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
    rng = random.Random(seed)
    entities = _TRAIN_ENTITIES if split == "train" else _EVAL_ENTITIES
    preds = _TRAIN_PREDS if split == "train" else _EVAL_PREDS
    categories = ("syllogism_pos", "syllogism_neg", "syllogism_unknown", "horn_conj")

    tasks: list[VerifiableLogicTask] = []
    for idx in range(n_tasks):
        depth = depths[idx % len(depths)]
        cat = categories[(idx // len(depths)) % len(categories)]
        tasks.append(
            _build_single_task(
                rng,
                idx=idx,
                depth=depth,
                category=cat,
                entities=entities,
                preds=preds,
                n_distractors=n_distractors,
                cot=cot,
            )
        )
    rng.shuffle(tasks)
    return tasks
