"""Deterministic fictional-world multi-hop retrieval, read, calc+search, and abstention generator.

All entities, cities, monographs, and dates are procedurally generated from disjoint
train/eval vocabularies so the model CANNOT answer from parametric memorization and
must genuinely use search(), read(), list_docs(), and calc().
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from coregrad.tools.calc import safe_calc
from coregrad.tools.protocol import DEFAULT_SYSTEM_PROMPT, Turn, format_tool_call
from coregrad.tools.retriever import Document, InMemoryBM25Retriever

TRAIN_SURNAMES = (
    "Alder", "Bramwell", "Carrow", "Denvir", "Elswick", "Fallow", "Garrick", "Halstead",
    "Iver", "Jarrow", "Kestrel", "Larkin", "Marrow", "Norwick", "Orwellis", "Pellham",
    "Quarry", "Riven", "Selwyn", "Tallow", "Ulric", "Vane", "Wickham", "Yarrow",
)
EVAL_SURNAMES = (
    "Ashford", "Belling", "Corwin", "Draper", "Eldon", "Fenwick", "Granger", "Hollis",
    "Ingram", "Jenson", "Kensal", "Lindon", "Mercer", "Nolan", "Oakhill", "Prescott",
)

TRAIN_GIVEN = (
    "Helena", "Tomas", "Maren", "Caleb", "Nadia", "Arthur", "Sora", "Victor",
    "Clara", "Felix", "Linnea", "Jonas", "Miriam", "Oskar", "Tessa", "Dorian",
)
EVAL_GIVEN = (
    "Beatrice", "Gideon", "Sylvia", "Rowan", "Astrid", "Edmund", "Freya", "Malcolm",
)

TRAIN_CITIES = (
    "Vaelen", "Oakhaven", "Keldor", "Brimstone", "Tarnwick", "Solisport", "Calderis",
    "Mirehaven", "Northbridge", "Crestfall", "Veridia", "Lindenberg", "Stonemere", "Aurelia",
)
EVAL_CITIES = (
    "Westmere", "Duskford", "Alverton", "Kestrelbay", "Highridge", "Silverton", "Redvale", "Corinthia",
)

DISCIPLINES = (
    "acoustics", "cartography", "hydrology", "metallurgy", "optics", "botany",
    "seismology", "astronomy", "thermodynamics", "mineralogy", "glaciology", "aerodynamics",
)

INSTRUMENTS = (
    "barograph", "chronometer", "spectrometer", "theodolite", "seismograph",
    "hygrometer", "interferometer", "magnetometer", "pyrometer", "astrolabe",
)


@dataclass(frozen=True)
class VerifiableRetrievalTask:
    task_id: str
    category: str
    question: str
    documents: list[Document]
    expected_answer: str
    expected_doc_ids: tuple[str, ...]
    should_abstain: bool
    turns: list[Turn]


def _build_fictional_world(
    rng: random.Random, split: str = "train", n_scholars: int = 8
) -> tuple[list[Document], list[dict[str, object]], list[dict[str, object]]]:
    surnames = list(TRAIN_SURNAMES if split == "train" else EVAL_SURNAMES)
    givens = list(TRAIN_GIVEN if split == "train" else EVAL_GIVEN)
    cities = list(TRAIN_CITIES if split == "train" else EVAL_CITIES)
    rng.shuffle(surnames)
    rng.shuffle(givens)
    rng.shuffle(cities)

    n_scholars = min(n_scholars, len(surnames), len(givens))
    scholars: list[dict[str, object]] = []
    labs: list[dict[str, object]] = []
    docs: list[Document] = []

    for i in range(n_scholars):
        full_name = f"{givens[i]} {surnames[i]}"
        disc = rng.choice(DISCIPLINES)
        inst = rng.choice(INSTRUMENTS)
        pub_year = rng.randint(1810, 1985)
        mass_g = rng.randint(120, 4800)
        lab_name = f"{cities[i % len(cities)]} {disc.title()} Institute"
        work_title = f"Treatise on {disc.title()} No. {rng.randint(1, 9)}"
        doc_id = f"d{i + 1:03d}"

        filler = (
            f"Archival records note that early experiments in {disc} faced seasonal drift "
            f"and instrument calibration challenges across regional observatories. "
            f"Field notebooks preserved in the archive document routine maintenance procedures "
            f"and comparative readings taken during winter and summer surveys. "
        )
        text = (
            f"# {full_name}\n\n"
            f"{full_name} was a specialist in {disc} affiliated with the {lab_name}. "
            f"In {pub_year}, {full_name} published the monograph '{work_title}', "
            f"introducing an improved {inst} weighing {mass_g} grams.\n\n"
            f"{filler}\n\n"
            f"Later correspondence confirmed that the reference {inst} prototype was archived "
            f"under catalog code {surnames[i][:3].upper()}-{pub_year % 100:02d}."
        )
        docs.append(Document(doc_id=doc_id, title=full_name, text=text, source=f"{doc_id}.md"))
        scholars.append(
            {
                "doc_id": doc_id,
                "name": full_name,
                "discipline": disc,
                "instrument": inst,
                "pub_year": pub_year,
                "mass_g": mass_g,
                "lab_name": lab_name,
                "work_title": work_title,
                "catalog_code": f"{surnames[i][:3].upper()}-{pub_year % 100:02d}",
            }
        )

    # Create institute documents for 2-hop lookups
    used_labs: set[str] = set()
    for j, sch in enumerate(scholars):
        lab_name = str(sch["lab_name"])
        if lab_name in used_labs:
            continue
        used_labs.add(lab_name)
        founded = rng.randint(1740, 1920)
        director = f"Dr. {rng.choice(givens)} {rng.choice(surnames)}"
        city = lab_name.split()[0]
        doc_id = f"d{n_scholars + j + 1:03d}"
        text = (
            f"# {lab_name}\n\n"
            f"The {lab_name} is located in the coastal district of {city}. "
            f"It was founded in {founded} to coordinate regional measurements in {sch['discipline']}. "
            f"During the archival survey period, its chief director was {director}."
        )
        docs.append(Document(doc_id=doc_id, title=lab_name, text=text, source=f"{doc_id}.md"))
        labs.append(
            {
                "doc_id": doc_id,
                "lab_name": lab_name,
                "city": city,
                "founded": founded,
                "director": director,
            }
        )

    return docs, scholars, labs


def _make_1hop_task(rng: random.Random, idx: int, split: str) -> VerifiableRetrievalTask:
    docs, scholars, _ = _build_fictional_world(rng, split=split, n_scholars=6)
    retriever = InMemoryBM25Retriever(docs)
    target = rng.choice(scholars)
    name = str(target["name"])
    doc_id = str(target["doc_id"])

    variant = rng.choice(["year", "instrument", "code"])
    if variant == "year":
        q = f"In what year did {name} publish '{target['work_title']}'?"
        query = f"{name} {target['work_title']} published"
        ans = str(target["pub_year"])
        final = f"{name} published '{target['work_title']}' in {ans} <|cite|>{doc_id}<|/cite|>."
    elif variant == "instrument":
        q = f"What instrument did {name} introduce, and what was its weight in grams?"
        query = f"{name} instrument grams"
        ans = f"{target['mass_g']}"
        final = (
            f"{name} introduced an improved {target['instrument']} weighing "
            f"{target['mass_g']} grams <|cite|>{doc_id}<|/cite|>."
        )
    else:
        q = f"What is the catalog code of the prototype created by {name}?"
        query = f"{name} catalog code prototype"
        ans = str(target["catalog_code"])
        final = f"The prototype created by {name} has catalog code {ans} <|cite|>{doc_id}<|/cite|>."

    hits = retriever.search(query, k=3)
    res_json = json.dumps([h.to_compact_dict() for h in hits], ensure_ascii=False)
    turns = [
        Turn("system", DEFAULT_SYSTEM_PROMPT),
        Turn("user", q),
        Turn("think", f"Search the local documents for {name}."),
        Turn("call", format_tool_call("search", query=query, k=3)),
        Turn("result", res_json),
        Turn("think", f"Passage {doc_id} states the answer is {ans}."),
        Turn("assistant", final),
    ]
    return VerifiableRetrievalTask(
        task_id=f"ret_1hop_{split}_{idx}",
        category="retrieval_1hop",
        question=q,
        documents=docs,
        expected_answer=ans,
        expected_doc_ids=(doc_id,),
        should_abstain=False,
        turns=turns,
    )


def _make_2hop_task(rng: random.Random, idx: int, split: str) -> VerifiableRetrievalTask:
    docs, scholars, labs = _build_fictional_world(rng, split=split, n_scholars=6)
    retriever = InMemoryBM25Retriever(docs)
    lab_by_name = {str(lab["lab_name"]): lab for lab in labs}
    target = rng.choice([s for s in scholars if str(s["lab_name"]) in lab_by_name])
    lab = lab_by_name[str(target["lab_name"])]

    name = str(target["name"])
    s_doc = str(target["doc_id"])
    l_name = str(lab["lab_name"])
    l_doc = str(lab["doc_id"])
    founded = str(lab["founded"])

    q = f"In what year was the institute affiliated with {name} founded?"
    q1 = f"{name} affiliated institute"
    hits1 = retriever.search(q1, k=2)
    res1 = json.dumps([h.to_compact_dict() for h in hits1], ensure_ascii=False)

    q2 = f"{l_name} founded"
    hits2 = retriever.search(q2, k=2)
    res2 = json.dumps([h.to_compact_dict() for h in hits2], ensure_ascii=False)

    turns = [
        Turn("system", DEFAULT_SYSTEM_PROMPT),
        Turn("user", q),
        Turn("think", f"First find which institute {name} was affiliated with."),
        Turn("call", format_tool_call("search", query=q1, k=2)),
        Turn("result", res1),
        Turn("think", f"{name} was affiliated with the {l_name} ({s_doc}). Now search for when it was founded."),
        Turn("call", format_tool_call("search", query=q2, k=2)),
        Turn("result", res2),
        Turn("think", f"The {l_name} was founded in {founded} ({l_doc})."),
        Turn(
            "assistant",
            f"{name} was affiliated with the {l_name} <|cite|>{s_doc}<|/cite|>, "
            f"which was founded in {founded} <|cite|>{l_doc}<|/cite|>.",
        ),
    ]
    return VerifiableRetrievalTask(
        task_id=f"ret_2hop_{split}_{idx}",
        category="retrieval_2hop",
        question=q,
        documents=docs,
        expected_answer=founded,
        expected_doc_ids=(s_doc, l_doc),
        should_abstain=False,
        turns=turns,
    )


def _make_retrieval_plus_calc_task(rng: random.Random, idx: int, split: str) -> VerifiableRetrievalTask:
    docs, scholars, _ = _build_fictional_world(rng, split=split, n_scholars=6)
    retriever = InMemoryBM25Retriever(docs)
    s1, s2 = rng.sample(scholars, 2)
    y1, y2 = int(s1["pub_year"]), int(s2["pub_year"])  # type: ignore[arg-type]
    hi, lo = (y2, y1) if y2 >= y1 else (y1, y2)
    expr = f"{hi} - {lo}"
    diff = safe_calc(expr)

    q = (
        f"How many years apart were the monographs by {s1['name']} and {s2['name']} published?"
    )
    q1 = f"{s1['name']} published monograph"
    q2 = f"{s2['name']} published monograph"
    res1 = json.dumps([h.to_compact_dict() for h in retriever.search(q1, k=2)], ensure_ascii=False)
    res2 = json.dumps([h.to_compact_dict() for h in retriever.search(q2, k=2)], ensure_ascii=False)

    d1, d2 = str(s1["doc_id"]), str(s2["doc_id"])
    turns = [
        Turn("system", DEFAULT_SYSTEM_PROMPT),
        Turn("user", q),
        Turn("think", f"Look up the publication years for both {s1['name']} and {s2['name']}, then subtract."),
        Turn("call", format_tool_call("search", query=q1, k=2)),
        Turn("result", res1),
        Turn("call", format_tool_call("search", query=q2, k=2)),
        Turn("result", res2),
        Turn("think", f"Publication years are {y1} ({d1}) and {y2} ({d2}). Compute the difference with calc."),
        Turn("call", format_tool_call("calc", expr=expr)),
        Turn("result", diff),
        Turn(
            "assistant",
            f"{s1['name']} published in {y1} <|cite|>{d1}<|/cite|> and {s2['name']} published in {y2} "
            f"<|cite|>{d2}<|/cite|>, a difference of {diff} years.",
        ),
    ]
    return VerifiableRetrievalTask(
        task_id=f"ret_calc_{split}_{idx}",
        category="retrieval_calc",
        question=q,
        documents=docs,
        expected_answer=diff,
        expected_doc_ids=(d1, d2),
        should_abstain=False,
        turns=turns,
    )


def _make_abstention_task(rng: random.Random, idx: int, split: str) -> VerifiableRetrievalTask:
    docs, _, _ = _build_fictional_world(rng, split=split, n_scholars=6)
    retriever = InMemoryBM25Retriever(docs)
    missing_name = f"Zephira {rng.choice(['Kovacs', 'Mirelle', 'Thornby', 'Valen', 'Ostrander'])}"
    missing_topic = rng.choice(["quantum telemetry", "deep-sea lichen", "lunar acoustics", "amber glassmaking"])
    q = f"In what year did {missing_name} publish their study on {missing_topic}?"
    query = f"{missing_name} {missing_topic}"
    res = json.dumps([h.to_compact_dict() for h in retriever.search(query, k=3)], ensure_ascii=False)

    turns = [
        Turn("system", DEFAULT_SYSTEM_PROMPT),
        Turn("user", q),
        Turn("think", f"Search the available documents for {missing_name} and {missing_topic}."),
        Turn("call", format_tool_call("search", query=query, k=3)),
        Turn("result", res),
        Turn("think", "None of the retrieved passages mention this author or study. I must abstain rather than guess."),
        Turn(
            "assistant",
            f"I could not find any information about {missing_name} or a study on {missing_topic} "
            "in the available documents.",
        ),
    ]
    return VerifiableRetrievalTask(
        task_id=f"ret_abstain_{split}_{idx}",
        category="abstention",
        question=q,
        documents=docs,
        expected_answer="could not find",
        expected_doc_ids=(),
        should_abstain=True,
        turns=turns,
    )


def generate_retrieval_tasks(
    n: int, seed: int = 202, split: str = "train"
) -> list[VerifiableRetrievalTask]:
    """Generate balanced 1-hop, 2-hop, retrieval+calc, and abstention tasks."""
    rng = random.Random(seed)
    builders = [
        _make_1hop_task,
        _make_2hop_task,
        _make_retrieval_plus_calc_task,
        _make_abstention_task,
    ]
    tasks: list[VerifiableRetrievalTask] = []
    for i in range(n):
        fn = builders[i % len(builders)]
        tasks.append(fn(rng, i, split))
    return tasks
