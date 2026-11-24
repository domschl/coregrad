"""Self-contained Gate 1 language probes (grammar minimal pairs, 4-way completions, qualitative suite)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from coregrad.model import CoreGradLM
from coregrad.tokenizer import CoreGradTokenizer

# 32 BLiMP-style grammatical minimal pairs: (grammatical, ungrammatical)
GRAMMAR_MINIMAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("The keys to the cabinet are on the table.", "The keys to the cabinet is on the table."),
    ("The author of those novels writes every morning.", "The author of those novels write every morning."),
    ("Every student in the lecture hall was listening carefully.", "Every student in the lecture hall were listening carefully."),
    ("The dogs that chased the cat across the yard were barking.", "The dogs that chased the cat across the yard was barking."),
    ("Neither the manager nor the engineers have approved the design.", "Neither the manager nor the engineers has approved the design."),
    ("The box of old photographs sits in the attic.", "The box of old photographs sit in the attic."),
    ("She reminded herself to check the calculations twice.", "She reminded himself to check the calculations twice."),
    ("The scientists prepared themselves for a long winter expedition.", "The scientists prepared itself for a long winter expedition."),
    ("He blamed himself for forgetting the measurement.", "He blamed herself for forgetting the measurement."),
    ("The committee reached its decision after three hours.", "The committee reached their decision after three hours."),
    ("These experiments require careful calibration.", "This experiments require careful calibration."),
    ("That hypothesis was tested in two separate laboratories.", "Those hypothesis was tested in two separate laboratories."),
    ("Many researchers have studied this phenomenon for decades.", "Much researchers have studied this phenomenon for decades."),
    ("There is very little water left in the reservoir.", "There is very few water left in the reservoir."),
    ("The bridge was built by engineers in 1937.", "The bridge was builded by engineers in 1937."),
    ("She had already written the final chapter before noon.", "She had already wrote the final chapter before noon."),
    ("The glass broke when it fell onto the stone floor.", "The glass breaked when it fell onto the stone floor."),
    ("They swam across the cold lake before sunrise.", "They swimmed across the cold lake before sunrise."),
    ("No one has ever seen that rare bird in this valley.", "Someone has ever seen that rare bird in this valley."),
    ("She did not buy any groceries on her way home.", "She bought any groceries on her way home."),
    ("What did the historian say that the letter contained?", "What did the historian wonder whether the letter contained?"),
    ("Who do you think solved the equation first?", "Who do you think that solved the equation first?"),
    ("The book that I borrowed from the library was fascinating.", "The book what I borrowed from the library was fascinating."),
    ("Although it was raining heavily, we walked to the station.", "Although it was raining heavily, but we walked to the station."),
    ("She is taller than her younger brother.", "She is more taller than her younger brother."),
    ("This algorithm runs much faster than the previous version.", "This algorithm runs much fast than the previous version."),
    ("The experiment could have succeeded with better insulation.", "The experiment could of succeeded with better insulation."),
    ("If he had known the answer, he would have spoken up.", "If he would have known the answer, he would have spoken up."),
    ("The children were playing quietly in the garden.", "The children was playing quietly in the garden."),
    ("Between you and me, the estimate seems too optimistic.", "Between you and I, the estimate seems too optimistic."),
    ("The telescope allows astronomers to observe distant galaxies.", "The telescope allows astronomers observing distant galaxies."),
    ("Having finished the report, Maria closed her laptop.", "Having finish the report, Maria closed her laptop."),
)


@dataclass(frozen=True)
class ChoiceProbe:
    category: str
    context: str
    choices: tuple[str, str, str, str]
    answer_idx: int


# 24 clean 4-way completion probes across science, reasoning, common sense, and Python
COMPLETION_PROBES: tuple[ChoiceProbe, ...] = (
    ChoiceProbe(
        "science",
        "When liquid water is cooled below 0 degrees Celsius at standard atmospheric pressure, it",
        (" freezes into solid ice.", " evaporates into plasma.", " turns into metallic sodium.", " doubles its atomic number."),
        0,
    ),
    ChoiceProbe(
        "science",
        "Plants convert sunlight, water, and carbon dioxide into glucose and oxygen through a process called",
        (" electrolysis.", " photosynthesis.", " fermentation.", " distillation."),
        1,
    ),
    ChoiceProbe(
        "science",
        "In a vacuum, electromagnetic waves such as visible light and radio waves all travel at the same",
        (" wavelength.", " frequency.", " speed.", " amplitude."),
        2,
    ),
    ChoiceProbe(
        "science",
        "Earth's seasons are primarily caused by the tilt of Earth's rotational axis relative to its",
        (" magnetic poles.", " ocean currents.", " lunar tide.", " orbital plane around the Sun."),
        3,
    ),
    ChoiceProbe(
        "science",
        "An object moving at constant velocity in the absence of net external force will",
        (" continue moving at that same velocity.", " immediately come to a stop.", " accelerate exponentially.", " reverse direction."),
        0,
    ),
    ChoiceProbe(
        "science",
        "Sound waves cannot travel through outer space because sound requires a physical",
        (" battery.", " medium.", " mirror.", " magnet."),
        1,
    ),
    ChoiceProbe(
        "logic",
        "All mammals are warm-blooded. Whales are mammals. Therefore, whales are",
        (" reptiles.", " cold-blooded.", " warm-blooded.", " amphibians."),
        2,
    ),
    ChoiceProbe(
        "logic",
        "Alice is taller than Bob, and Bob is taller than Carol. Therefore, Alice is",
        (" shorter than Carol.", " the same height as Carol.", " shorter than Bob.", " taller than Carol."),
        3,
    ),
    ChoiceProbe(
        "logic",
        "If it rains, the sidewalk gets wet. The sidewalk is completely dry. Therefore,",
        (" it did not rain.", " it must be raining.", " the sun disappeared.", " the sidewalk is flooded."),
        0,
    ),
    ChoiceProbe(
        "logic",
        "A train leaves at 14:00 and arrives at 16:30 on the same day. The journey took",
        (" one hour.", " two hours and thirty minutes.", " four hours.", " thirty minutes."),
        1,
    ),
    ChoiceProbe(
        "logic",
        "In the sequence 2, 4, 8, 16, 32, the next number is",
        (" 34.", " 48.", " 64.", " 128."),
        2,
    ),
    ChoiceProbe(
        "logic",
        "Every book on the top shelf is about history. This volume is on the top shelf, so it is about",
        (" cooking.", " astronomy.", " gardening.", " history."),
        3,
    ),
    ChoiceProbe(
        "commonsense",
        "After walking for ten miles in the desert without a canteen, the hiker was extremely",
        (" thirsty.", " damp.", " frozen.", " silent."),
        0,
    ),
    ChoiceProbe(
        "commonsense",
        "To keep the soup warm until dinner, she placed a tight lid on the",
        (" pillow.", " pot.", " bicycle.", " envelope."),
        1,
    ),
    ChoiceProbe(
        "commonsense",
        "Because the power went out across the neighborhood, the room suddenly became",
        (" noisy.", " colorful.", " dark.", " windy."),
        2,
    ),
    ChoiceProbe(
        "commonsense",
        "Before hammering the nail into the wooden board, the carpenter carefully aligned it with the",
        (" spoon.", " blanket.", " trumpet.", " pencil mark."),
        3,
    ),
    ChoiceProbe(
        "commonsense",
        "She forgot her umbrella at home, so her coat was soaked when she arrived in the",
        (" rain.", " library.", " oven.", " desert."),
        0,
    ),
    ChoiceProbe(
        "commonsense",
        "When the alarm clock rang at six in the morning, he rubbed his eyes and",
        (" fell into orbit.", " got out of bed.", " painted the ocean.", " drove a submarine."),
        1,
    ),
    ChoiceProbe(
        "code",
        "In Python, `len([10, 20, 30, 40])` evaluates to",
        (" 100", " 3", " 4", " 40"),
        2,
    ),
    ChoiceProbe(
        "code",
        "In Python, `{'a': 1, 'b': 2}['b']` returns",
        (" 'b'", " 1", " 3", " 2"),
        3,
    ),
    ChoiceProbe(
        "code",
        "In Python, a function that returns the square of x is written as `def square(x): return`",
        (" x * x", " x + x", " x - 2", " x / 2"),
        0,
    ),
    ChoiceProbe(
        "code",
        "In JSON, keys inside an object must be enclosed in",
        (" single parentheses.", " double quotes.", " angle brackets.", " square brackets."),
        1,
    ),
    ChoiceProbe(
        "code",
        "In Python, `sorted([3, 1, 2])` returns",
        (" [3, 2, 1]", " [2, 1, 3]", " [1, 2, 3]", " [1, 3, 2]"),
        2,
    ),
    ChoiceProbe(
        "code",
        "In binary, the decimal number 5 is written as",
        (" 111", " 100", " 010", " 101"),
        3,
    ),
)


# Fixed 20-prompt qualitative suite for checkpoint-by-checkpoint inspection
QUALITATIVE_PROMPTS: tuple[str, ...] = (
    "Explain in plain English why the sky appears blue during the day:",
    "A concise summary of the scientific method is that it begins with",
    "In 1969, engineers faced a difficult navigation problem when",
    "The main difference between a stack and a queue in computer science is that",
    "Here is a clean Python function to compute the greatest common divisor of two integers:\n```python\ndef gcd(a: int, b: int) -> int:\n",
    "Once upon a time in a quiet coastal town, an old clockmaker discovered",
    "Why is compound interest so powerful over long time horizons? Consider",
    "Step-by-step, to solve the linear equation 3x + 7 = 22, we first",
    "Three practical tips for writing maintainable software are:\n1.",
    "Photosynthesis is essential to life on Earth because",
    "What happens when you heat a sealed container of gas? According to the ideal gas law,",
    "In formal logic, a valid deductive argument guarantees that if the premises are true,",
    "Write a short paragraph comparing electric motors and internal combustion engines:",
    "A hash table provides average O(1) lookup time by",
    "The printing press transformed European society in the fifteenth century because",
    "Consider a library with 10,000 books. To find a passage quickly without reading every page, you need",
    "Explain what a prime number is and list the first five prime numbers:",
    "In a clean JSON object describing a book with title, author, and year:\n```json\n{\n",
    "The difference between correlation and causation is important in statistics because",
    "When designing a small language model for local research, the key advantage of external tools is",
)


@torch.no_grad()
def _sequence_log_prob(
    model: CoreGradLM,
    token_ids: list[int],
    start_idx: int,
    device: torch.device,
    use_bf16: bool = True,
) -> float:
    """Sum of log-probabilities of tokens at positions [start_idx .. len-1]."""
    if len(token_ids) < 2 or start_idx >= len(token_ids):
        return float("-inf")
    x = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)
    amp_enabled = use_bf16 and device.type == "cuda"
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
        logits, _ = model(x)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    tok_lp = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)[0]
    # Position i in tok_lp predicts token_ids[i + 1]
    lo = max(0, start_idx - 1)
    return float(tok_lp[lo:].sum().item())


@torch.no_grad()
def evaluate_grammar_pairs(
    model: CoreGradLM,
    tokenizer: CoreGradTokenizer,
    device: torch.device,
    use_bf16: bool = True,
) -> dict[str, float]:
    model.eval()
    correct = 0
    for good, bad in GRAMMAR_MINIMAL_PAIRS:
        g_ids = tokenizer.encode(good, add_bos=True)
        b_ids = tokenizer.encode(bad, add_bos=True)
        g_lp = _sequence_log_prob(model, g_ids, start_idx=1, device=device, use_bf16=use_bf16) / max(
            len(g_ids) - 1, 1
        )
        b_lp = _sequence_log_prob(model, b_ids, start_idx=1, device=device, use_bf16=use_bf16) / max(
            len(b_ids) - 1, 1
        )
        if g_lp > b_lp:
            correct += 1
    total = len(GRAMMAR_MINIMAL_PAIRS)
    return {"correct": float(correct), "total": float(total), "accuracy": correct / total}


@torch.no_grad()
def evaluate_completion_probes(
    model: CoreGradLM,
    tokenizer: CoreGradTokenizer,
    device: torch.device,
    use_bf16: bool = True,
) -> dict[str, object]:
    model.eval()
    by_cat: dict[str, list[int]] = {}
    correct_total = 0

    for probe in COMPLETION_PROBES:
        ctx_ids = tokenizer.encode(probe.context, add_bos=True)
        scores: list[float] = []
        for choice in probe.choices:
            full_ids = tokenizer.encode(probe.context + choice, add_bos=True)
            cont_len = max(len(full_ids) - len(ctx_ids), 1)
            lp = _sequence_log_prob(
                model, full_ids, start_idx=len(ctx_ids), device=device, use_bf16=use_bf16
            )
            scores.append(lp / cont_len)
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        hit = int(pred == probe.answer_idx)
        correct_total += hit
        by_cat.setdefault(probe.category, []).append(hit)

    total = len(COMPLETION_PROBES)
    return {
        "correct": correct_total,
        "total": total,
        "accuracy": correct_total / total,
        "by_category": {k: sum(v) / len(v) for k, v in by_cat.items()},
    }


@torch.no_grad()
def run_qualitative_suite(
    model: CoreGradLM,
    tokenizer: CoreGradTokenizer,
    device: torch.device,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_k: int = 40,
    limit: int | None = None,
) -> list[dict[str, str]]:
    model.eval()
    prompts = QUALITATIVE_PROMPTS[:limit] if limit is not None else QUALITATIVE_PROMPTS
    outputs: list[dict[str, str]] = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_bos=True)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        gen = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_id,
        )
        completion = tokenizer.decode(gen[0, len(ids) :].tolist(), skip_special_tokens=True)
        outputs.append({"prompt": prompt, "completion": completion})
    return outputs
