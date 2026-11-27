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

from coregrad.tools.protocol import DEFAULT_SYSTEM_PROMPT, Turn, format_tool_call

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


def _build_open_train_preds() -> tuple[str, ...]:
    """Build >2,000 pronounceable pseudowords strictly disjoint from _EVAL_PREDS.

    Prevents small models from memorizing a 32-word training vocabulary and forces
    general verbatim copying / singularization of unseen multi-subword predicates.
    """
    onsets = (
        "ba", "be", "bi", "bo", "bra", "ca", "ce", "co", "da", "de", "do", "dra",
        "fa", "fe", "fi", "fo", "ga", "ge", "go", "gra", "ha", "he", "ja", "jo",
        "ka", "ke", "ki", "ko", "kra", "la", "le", "li", "lo", "ma", "me", "mi",
        "mo", "na", "ne", "no", "pa", "pe", "po", "pra", "ra", "re", "ri", "ro",
        "sa", "se", "so", "ta", "te", "ti", "to", "tra", "va", "ve", "vi", "vo",
        "wa", "we", "za", "ze", "zo",
    )
    codas = (
        "bal", "ben", "bor", "dal", "den", "dor", "fal", "fen", "gal", "gen",
        "kal", "kel", "ken", "kor", "lan", "lar", "len", "lin", "lor", "mal",
        "mar", "mel", "men", "mor", "nal", "nel", "nor", "pal", "pel", "pen",
        "ral", "rel", "ren", "rik", "rin", "ron", "sal", "sel", "sen", "tal",
        "tel", "ten", "tor", "val", "vel", "ven", "vix", "vor", "wal", "zel",
    )
    eval_set = set(_EVAL_PREDS)
    words = [f"{o}{c}" for o in onsets for c in codas if f"{o}{c}" not in eval_set]
    return tuple(words)


_OPEN_TRAIN_PREDS = _build_open_train_preds()


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
    notation: str = "english",
) -> VerifiableLogicTask:
    if depth < 2:
        raise ValueError(f"depth must be >= 2, got {depth}")
    if notation not in ("english", "lot", "dsl"):
        raise ValueError(f"notation must be 'english', 'lot', or 'dsl', got {notation!r}")
    entity = rng.choice(entities)

    # Reserve enough unique predicates for main chain + optional branch + distractors
    need = depth + 3 + max(4, n_distractors + 3)
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
    # Compact single-token symbol mapping for Language-of-Thought (LoT) mode
    sym_map: dict[str, str] = {}
    used_chain_preds: list[str] = []

    def sym(p: str) -> str:
        if p not in sym_map:
            idx_s = len(sym_map)
            sym_map[p] = f"P{idx_s}"
            used_chain_preds.append(p)
        return sym_map[p]

    if category == "syllogism_pos":
        chain = chosen[: depth + 1]
        base_facts.add(chain[0])
        for i in range(depth):
            unary_rules.append((chain[i], chain[i + 1]))
            if notation == "lot":
                think_steps.append(f"{sym(chain[i])}->{sym(chain[i + 1])} => x:{sym(chain[i + 1])}")
            else:
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
            if notation == "lot":
                think_steps.append(f"{sym(chain[i])}->{sym(chain[i + 1])} => x:{sym(chain[i + 1])}")
            else:
                think_steps.append(
                    f"Step {i + 1}: {entity} is a {chain[i]} and every {chain[i]} is a {chain[i + 1]}, so {entity} is a {chain[i + 1]}."
                )
        neg_rules.append((chain[depth - 1], chain[depth]))
        if notation == "lot":
            think_steps.append(f"{sym(chain[depth - 1])}-/>{sym(chain[depth])} => !x:{sym(chain[depth])}")
        else:
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
                unary_rules.append((chain[i + 1], chain[i]))
            else:
                unary_rules.append((chain[i], chain[i + 1]))
        for i in range(break_at):
            if notation == "lot":
                think_steps.append(f"{sym(chain[i])}->{sym(chain[i + 1])} => x:{sym(chain[i + 1])}")
            else:
                think_steps.append(
                    f"Step {i + 1}: {entity} is a {chain[i]} and every {chain[i]} is a {chain[i + 1]}, so {entity} is a {chain[i + 1]}."
                )
        if notation == "lot":
            think_steps.append(
                f"rev({sym(chain[break_at + 1])}->{sym(chain[break_at])}) => stop; goal:{sym(chain[depth])}=?"
            )
        else:
            think_steps.append(
                f"Step {break_at + 1}: We know {entity} is a {chain[break_at]}, and every {chain[break_at + 1]} is a {chain[break_at]}, "
                f"but that does not prove {entity} is a {chain[break_at + 1]}. Therefore {chain[depth]} cannot be proven or disproven."
            )
        target = chain[depth]
        expected = "Unknown"
        used_end = depth + 1

    elif category == "horn_conj":
        chain = chosen[: depth + 1]
        side_p = chosen[depth + 1]
        join_at = max(1, depth // 2)
        base_facts.add(chain[0])
        base_facts.add(side_p)
        for i in range(depth):
            if i == join_at:
                conj_rules.append((chain[i], side_p, chain[i + 1]))
                if notation == "lot":
                    think_steps.append(
                        f"{sym(chain[i])}&{sym(side_p)}->{sym(chain[i + 1])} => x:{sym(chain[i + 1])}"
                    )
                else:
                    think_steps.append(
                        f"Step {i + 1}: {entity} is both a {chain[i]} and a {side_p}, so {entity} is a {chain[i + 1]}."
                    )
            else:
                unary_rules.append((chain[i], chain[i + 1]))
                if notation == "lot":
                    think_steps.append(f"{sym(chain[i])}->{sym(chain[i + 1])} => x:{sym(chain[i + 1])}")
                else:
                    think_steps.append(
                        f"Step {i + 1}: {entity} is a {chain[i]} and every {chain[i]} is a {chain[i + 1]}, so {entity} is a {chain[i + 1]}."
                    )
        target = chain[depth]
        expected = "True"
        used_end = depth + 2
    else:
        raise ValueError(f"Unsupported category: {category!r}")

    # Add acyclic distractor rules using strictly disjoint predicates so target truth never changes.
    # Always include at least one distractor negation rule and one distractor conjunctive rule so
    # surface keywords ("No ...", "both ... and ...") carry zero shortcut information.
    dist_pool = chosen[used_end:]
    if len(dist_pool) >= 3:
        neg_rules.append((dist_pool[0], dist_pool[1]))
        conj_rules.append((dist_pool[0], dist_pool[1], dist_pool[2]))
    for j in range(min(n_distractors, max(0, len(dist_pool) - 1))):
        if j % 3 == 0:
            # Side branch off an intermediate chain node into an unused distractor node
            anchor = chosen[rng.randint(0, depth - 1)]
            target_d = dist_pool[min(j + 2, len(dist_pool) - 1)]
            unary_rules.append((anchor, target_d))
        else:
            unary_rules.append((dist_pool[j], dist_pool[j + 1]))

    # Soundness check via exact forward-chaining prover
    verified = _prove_entailment(base_facts, unary_rules, neg_rules, conj_rules, target)
    assert verified == expected, f"Prover mismatch: expected={expected}, verified={verified}"

    # Render facts and shuffled rules (pair with BPE-aligned space-padded DSL clauses so shuffle order is bit-identical)
    rule_pairs = (
        [(_render_unary(rng, a, b), f"{a} -> {b}") for a, b in unary_rules]
        + [(_render_neg(rng, a, b), f"{a} -/> {b}") for a, b in neg_rules]
        + [(_render_conj(rng, a, b, c), f"{a} & {b} -> {c}") for a, b, c in conj_rules]
    )
    rng.shuffle(rule_pairs)
    fact_pairs = [(f"{entity} is a {f}.", f"x : {f}") for f in sorted(base_facts)]
    rng.shuffle(fact_pairs)

    rule_sentences = [s for s, _ in rule_pairs]
    fact_sentences = [s for s, _ in fact_pairs]
    premise_block = " ".join(fact_sentences + rule_sentences)
    question = (
        f"Rules and facts: {premise_block}\n"
        f"Question: Based strictly on the rules above, is {entity} a {target}? "
        f"Answer with exactly one word: True, False, or Unknown."
    )

    turns: list[Turn] = [Turn("system", DEFAULT_SYSTEM_PROMPT), Turn("user", question)]
    if notation == "dsl":
        # Pure left-to-right O(1)-lookahead semantic parse into BPE-aligned coregrad-DSL prove(...)
        clauses = [c for _, c in fact_pairs] + [c for _, c in rule_pairs]
        dsl_expr = f"prove([ {', '.join(clauses)} ], x : {target})"
        turns.append(Turn("call", format_tool_call("calc", expr=dsl_expr)))
        turns.append(Turn("result", expected))
    elif cot:
        if notation == "lot":
            bind_str = "; ".join(f"{sym_map[p]}={p}" for p in used_chain_preds)
            fact_str = ", ".join(f"x:{sym(f)}" for f in sorted(base_facts))
            think_body = f"bind: x={entity}; {bind_str}\nfact: {fact_str}\n" + "\n".join(think_steps) + f"\nans: {expected}"
        else:
            think_body = " ".join(think_steps) + f" Conclusion: {expected}."
        turns.append(Turn("think", think_body))
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
    notation: str = "english",
    open_vocab: bool = False,
) -> list[VerifiableLogicTask]:
    """Generate a balanced, prover-verified dataset of multi-step syllogism & Horn-clause tasks."""
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
    rng = random.Random(seed)
    entities = _TRAIN_ENTITIES if split == "train" else _EVAL_ENTITIES
    if split == "train":
        preds = _OPEN_TRAIN_PREDS if open_vocab else _TRAIN_PREDS
    else:
        preds = _EVAL_PREDS
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
                notation=notation,
            )
        )
    rng.shuffle(tasks)
    return tasks
