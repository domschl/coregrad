"""Unit tests for procedural multi-step syllogism & Horn-clause logic generator."""

from __future__ import annotations

from coregrad.synthetic.logic_gen import (
    _EVAL_ENTITIES,
    _EVAL_PREDS,
    _TRAIN_ENTITIES,
    _TRAIN_PREDS,
    generate_logic_tasks,
)


def test_logic_train_eval_vocab_strictly_disjoint() -> None:
    assert set(_TRAIN_ENTITIES).isdisjoint(_EVAL_ENTITIES)
    assert set(_TRAIN_PREDS).isdisjoint(_EVAL_PREDS)


def test_logic_generator_depths_and_soundness() -> None:
    tasks_cot = generate_logic_tasks(24, depths=(2, 6, 12), split="train", seed=10, cot=True)
    assert len(tasks_cot) == 24
    labels = {t.expected_answer for t in tasks_cot}
    assert labels == {"True", "False", "Unknown"}
    assert all(any(turn.role == "think" for turn in t.turns) for t in tasks_cot)

    tasks_direct = generate_logic_tasks(16, depths=(4, 10), split="eval", seed=20, cot=False)
    assert len(tasks_direct) == 16
    assert all(not any(turn.role == "think" for turn in t.turns) for t in tasks_direct)
    assert {t.depth for t in tasks_direct} == {4, 10}

    tasks_lot = generate_logic_tasks(16, depths=(3, 8), split="eval", seed=30, cot=True, notation="lot")
    assert len(tasks_lot) == 16
    for t in tasks_lot:
        think_turns = [turn.content for turn in t.turns if turn.role == "think"]
        assert len(think_turns) == 1
        assert "bind:" in think_turns[0] and "fact:" in think_turns[0] and "ans:" in think_turns[0]
