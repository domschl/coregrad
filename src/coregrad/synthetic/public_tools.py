"""Normalizers converting cached public tool-calling and chat SFT corpora into coregrad Turns."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from coregrad.config import REPO_ROOT, load_local_paths, load_yaml
from coregrad.tools.calc import safe_calc
from coregrad.tools.protocol import DEFAULT_SYSTEM_PROMPT, Turn, format_tool_call

_HERMES_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_HERMES_RESP_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
_GLAIVE_CALL_RE = re.compile(r"<functioncall>\s*(\{.*?\})\s*(?:<\|endoftext\|>|$)", re.DOTALL)
_GSM8K_CALC_RE = re.compile(r"<<([^<>]+)=([^<>]+)>>")
_QUERY_STOPWORDS = frozenset(
    {
        "what", "when", "where", "who", "why", "how", "which", "did", "does", "do",
        "is", "are", "was", "were", "the", "a", "an", "in", "of", "to", "for", "on",
        "at", "by", "with", "from", "as", "and", "or", "that", "this", "it",
    }
)


def _is_simple_args(args: Any) -> bool:
    if not isinstance(args, dict) or len(args) > 4:
        return False
    return all(isinstance(v, (str, int, float, bool)) for v in args.values())


def _compact_tool_list(raw_tools: Any, max_tools: int = 4) -> str:
    if isinstance(raw_tools, str):
        try:
            raw_tools = json.loads(raw_tools)
        except json.JSONDecodeError:
            return ""
    if not isinstance(raw_tools, list):
        return ""
    sigs: list[str] = []
    for item in raw_tools[:max_tools]:
        if not isinstance(item, dict):
            continue
        fn = item.get("function", item)
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.isidentifier():
            continue
        params = fn.get("parameters", {})
        props = params.get("properties", params) if isinstance(params, dict) else {}
        arg_names = list(props.keys())[:4] if isinstance(props, dict) else []
        sigs.append(f"{name}({', '.join(arg_names)})")
    return ", ".join(sigs)


def iter_xlam_trajectories(files: list[Path], limit: int | None = None) -> Iterator[list[Turn]]:
    count = 0
    for f in files:
        rows = json.loads(f.read_text(encoding="utf-8"))
        for row in rows:
            query = str(row.get("query", "")).strip()
            if not query:
                continue
            try:
                answers = json.loads(row.get("answers", "[]"))
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(answers, list) or not (1 <= len(answers) <= 2):
                continue

            calls: list[str] = []
            valid = True
            for ans in answers:
                name = ans.get("name") if isinstance(ans, dict) else None
                args = ans.get("arguments") if isinstance(ans, dict) else None
                if not isinstance(name, str) or not name.isidentifier() or not _is_simple_args(args):
                    valid = False
                    break
                calls.append(format_tool_call(name, **args))
            if not valid:
                continue

            tool_sigs = _compact_tool_list(row.get("tools", "[]"))
            sys_prompt = (
                f"You have access to these tools: {tool_sigs}. Call the appropriate tool(s)."
                if tool_sigs
                else "Call the appropriate tool to answer the user."
            )
            turns = [Turn("system", sys_prompt), Turn("user", query)]
            for c in calls:
                turns.append(Turn("call", c))
            yield turns
            count += 1
            if limit is not None and count >= limit:
                return


def iter_hermes_trajectories(files: list[Path], limit: int | None = None) -> Iterator[list[Turn]]:
    count = 0
    for f in files:
        rows = json.loads(f.read_text(encoding="utf-8"))
        for row in rows:
            conv = row.get("conversations")
            if not isinstance(conv, list) or len(conv) < 2:
                continue
            turns: list[Turn] = []
            valid = True
            has_assistant = False

            for msg in conv:
                role = msg.get("from", "")
                val = str(msg.get("value", "")).strip()
                if not val:
                    continue
                if role == "system":
                    sigs = _compact_tool_list(row.get("tools", "[]"))
                    turns.append(
                        Turn("system", f"Available tools: {sigs}." if sigs else val[:400])
                    )
                elif role == "human":
                    turns.append(Turn("user", val))
                elif role == "tool":
                    m = _HERMES_RESP_RE.search(val)
                    turns.append(Turn("result", (m.group(1) if m else val)[:1200]))
                elif role == "gpt":
                    matches = list(_HERMES_CALL_RE.finditer(val))
                    if matches:
                        for m in matches:
                            try:
                                obj = json.loads(m.group(1))
                            except json.JSONDecodeError:
                                valid = False
                                break
                            name = obj.get("name")
                            args = obj.get("arguments", {})
                            if not isinstance(name, str) or not name.isidentifier() or not _is_simple_args(args):
                                valid = False
                                break
                            turns.append(Turn("call", format_tool_call(name, **args)))
                            has_assistant = True
                        if not valid:
                            break
                    else:
                        turns.append(Turn("assistant", val[:1500]))
                        has_assistant = True

            if valid and has_assistant and len(turns) >= 2:
                yield turns
                count += 1
                if limit is not None and count >= limit:
                    return


def iter_glaive_trajectories(files: list[Path], limit: int | None = None) -> Iterator[list[Turn]]:
    count = 0
    for f in files:
        rows = json.loads(f.read_text(encoding="utf-8"))
        for row in rows:
            chat = str(row.get("chat", "")).strip()
            if not chat:
                continue
            sys_raw = str(row.get("system", "")).strip()
            turns: list[Turn] = [Turn("system", sys_raw[:400] if sys_raw else "Use tools only when required.")]
            # Split simple USER: / ASSISTANT: / FUNCTION RESPONSE: blocks
            parts = re.split(r"\n*(USER:|ASSISTANT:|FUNCTION RESPONSE:)\s*", chat)
            valid = True
            has_assistant = False
            i = 1
            while i + 1 < len(parts):
                tag = parts[i]
                body = parts[i + 1].replace("<|endoftext|>", "").strip()
                i += 2
                if not body:
                    continue
                if tag == "USER:":
                    turns.append(Turn("user", body[:1000]))
                elif tag == "FUNCTION RESPONSE:":
                    turns.append(Turn("result", body[:1200]))
                elif tag == "ASSISTANT:":
                    m = _GLAIVE_CALL_RE.search(body)
                    if m:
                        try:
                            raw_json = m.group(1).replace("'", '"')
                            obj = json.loads(raw_json)
                        except json.JSONDecodeError:
                            valid = False
                            break
                        name = obj.get("name")
                        args = obj.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                valid = False
                                break
                        if not isinstance(name, str) or not name.isidentifier() or not _is_simple_args(args):
                            valid = False
                            break
                        turns.append(Turn("call", format_tool_call(name, **args)))
                        has_assistant = True
                    else:
                        turns.append(Turn("assistant", body[:1500]))
                        has_assistant = True

            if valid and has_assistant and len(turns) >= 3:
                yield turns
                count += 1
                if limit is not None and count >= limit:
                    return


def iter_smoltalk_trajectories(
    files: list[Path],
    limit: int | None = None,
    use_default_system_prompt: bool = True,
) -> Iterator[list[Turn]]:
    count = 0
    for f in files:
        pf = pq.ParquetFile(f)
        if "messages" not in pf.schema_arrow.names:
            continue
        for batch in pf.iter_batches(batch_size=512, columns=["messages"]):
            for msgs in batch.column(0).to_pylist():
                if not isinstance(msgs, list) or len(msgs) < 2:
                    continue
                turns: list[Turn] = []
                total_chars = 0
                has_system = False
                for msg in msgs:
                    role = msg.get("role", "")
                    content = str(msg.get("content", "")).strip()
                    if not content or role not in ("system", "user", "assistant"):
                        continue
                    if role == "system":
                        has_system = True
                    turns.append(Turn(role, content))
                    total_chars += len(content)
                if len(turns) >= 2 and total_chars <= 5000 and any(t.role == "assistant" for t in turns):
                    # Align system prompt with cognitive-core default so the model learns
                    # to answer general chat / conceptual questions directly without calling tools.
                    if use_default_system_prompt and not has_system:
                        turns = [Turn("system", DEFAULT_SYSTEM_PROMPT), *turns]
                    yield turns
                    count += 1
                    if limit is not None and count >= limit:
                        return


def _build_search_query(title: str, question: str) -> str:
    clean_title = title.replace("_", " ").strip()
    words = [w for w in re.findall(r"[A-Za-z0-9]+", question) if w.lower() not in _QUERY_STOPWORDS]
    q_part = " ".join(words[:7])
    if clean_title and clean_title.lower() not in q_part.lower():
        return f"{clean_title} {q_part}".strip()
    return q_part or question[:80]


def iter_squad_v2_trajectories(
    files: list[Path],
    limit: int | None = None,
    seed: int = 2026,
) -> Iterator[list[Turn]]:
    """Convert SQuAD v2 real-passage QA & unanswerable questions into grounded search+cite trajectories."""
    rng = random.Random(seed)
    rows_pool: list[dict[str, Any]] = []
    for f in files:
        pf = pq.ParquetFile(f)
        tbl = pf.read(columns=["title", "context", "question", "answers"])
        rows_pool.extend(tbl.to_pylist())

    if not rows_pool:
        return

    # Group unique contexts by title for clean negative distractors from other topics
    by_title: dict[str, list[str]] = {}
    for row in rows_pool:
        title = str(row.get("title", "")).replace("_", " ").strip()
        ctx = str(row.get("context", "")).strip()
        if title and ctx:
            lst = by_title.setdefault(title, [])
            if len(lst) < 5 and ctx not in lst:
                lst.append(ctx)
    all_titles = list(by_title.keys())
    rng.shuffle(rows_pool)

    count = 0
    for row in rows_pool:
        title = str(row.get("title", "")).replace("_", " ").strip()
        context = str(row.get("context", "")).strip()
        question = str(row.get("question", "")).strip()
        answers_obj = row.get("answers") or {}
        ans_texts = answers_obj.get("text", []) if isinstance(answers_obj, dict) else []
        if not title or not context or not question or len(context) > 1600:
            continue

        target_id = f"d{rng.randint(1, 999):04d}"
        hits: list[dict[str, object]] = [
            {"doc_id": target_id, "title": title, "span": [0, len(context)], "text": context}
        ]
        # Add 1-2 distractor passages from different topics and shuffle hit ranks
        n_dist = rng.choice([1, 2])
        for _ in range(n_dist):
            other_t = rng.choice(all_titles)
            if other_t == title:
                continue
            other_ctx = rng.choice(by_title[other_t])[:900]
            dist_id = f"d{rng.randint(1, 999):04d}"
            if dist_id == target_id:
                dist_id = f"d{(int(target_id[1:]) % 998) + 1:04d}"
            hits.append(
                {"doc_id": dist_id, "title": other_t, "span": [0, len(other_ctx)], "text": other_ctx}
            )
        rng.shuffle(hits)

        query = _build_search_query(title, question)
        res_json = json.dumps(hits, ensure_ascii=False)

        if ans_texts:
            ans = str(ans_texts[0]).strip()
            if not ans:
                continue
            turns = [
                Turn("system", DEFAULT_SYSTEM_PROMPT),
                Turn("user", question),
                Turn("think", f"Search the documents for information about {title}."),
                Turn("call", format_tool_call("search", query=query, k=3)),
                Turn("result", res_json),
                Turn("think", f"Passage {target_id} ({title}) directly answers the question: {ans}."),
                Turn("assistant", f"According to the documents, {ans} <|cite|>{target_id}<|/cite|>."),
            ]
        else:
            turns = [
                Turn("system", DEFAULT_SYSTEM_PROMPT),
                Turn("user", question),
                Turn("think", f"Search the documents for information about {title}."),
                Turn("call", format_tool_call("search", query=query, k=3)),
                Turn("result", res_json),
                Turn("think", "The retrieved passages do not contain an answer to this question. I should abstain."),
                Turn("assistant", "I could not find the answer to that question in the available documents."),
            ]

        yield turns
        count += 1
        if limit is not None and count >= limit:
            return


def convert_gsm8k_row(question: str, answer_text: str) -> list[Turn] | None:
    """Convert one GSM8K solution with <<expr=val>> tags into a verified calc() trajectory."""
    if "####" not in answer_text:
        return None
    body, final_part = answer_text.rsplit("####", 1)
    final_num = final_part.strip().replace(",", "")
    if not final_num:
        return None

    calc_matches = list(_GSM8K_CALC_RE.finditer(body))
    if not calc_matches or len(calc_matches) > 5:
        return None

    turns: list[Turn] = [
        Turn("system", DEFAULT_SYSTEM_PROMPT),
        Turn("user", question.strip()),
    ]
    # Emit concise step-by-step think + verified calc() calls
    lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
    any_valid_call = False
    for line in lines:
        matches = list(_GSM8K_CALC_RE.finditer(line))
        clean_thought = _GSM8K_CALC_RE.sub(lambda m: m.group(2), line).strip()
        if clean_thought:
            turns.append(Turn("think", clean_thought))
        for m in matches:
            raw_expr = m.group(1).strip().replace(",", "").replace("x", "*").replace("X", "*")
            try:
                computed = safe_calc(raw_expr)
            except (ValueError, ArithmeticError, OverflowError):
                return None
            turns.append(Turn("call", format_tool_call("calc", expr=raw_expr)))
            turns.append(Turn("result", computed))
            any_valid_call = True

    if not any_valid_call:
        return None

    turns.append(Turn("assistant", f"The final answer is {final_num}."))
    return turns


def iter_gsm8k_trajectories(files: list[Path], limit: int | None = None) -> Iterator[list[Turn]]:
    count = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=512, columns=["question", "answer"]):
            qs = batch.column(0).to_pylist()
            ans_list = batch.column(1).to_pylist()
            for q, a in zip(qs, ans_list, strict=True):
                if not isinstance(q, str) or not isinstance(a, str):
                    continue
                turns = convert_gsm8k_row(q, a)
                if turns is not None:
                    yield turns
                    count += 1
                    if limit is not None and count >= limit:
                        return


def load_cached_public_tool_files(repo_root: Path = REPO_ROOT) -> dict[str, list[Path]]:
    paths = load_local_paths(repo_root)
    if paths.hf_cache_dir is None or not paths.hf_cache_dir.exists():
        return {}
    manifest = load_yaml(repo_root / "configs" / "datasets.yaml")
    out: dict[str, list[Path]] = {}
    for name, spec in manifest.get("tools_and_sft", {}).items():
        repo_dir = paths.hf_cache_dir / spec["repo"]
        out[name] = sorted(repo_dir.glob(spec["glob"])) if repo_dir.exists() else []
    return out
