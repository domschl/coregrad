"""Normalizers converting cached public tool-calling and chat SFT corpora into coregrad Turns."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from coregrad.config import REPO_ROOT, load_local_paths, load_yaml
from coregrad.tools.protocol import Turn, format_tool_call

_HERMES_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_HERMES_RESP_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
_GLAIVE_CALL_RE = re.compile(r"<functioncall>\s*(\{.*?\})\s*(?:<\|endoftext\|>|$)", re.DOTALL)


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


def iter_smoltalk_trajectories(files: list[Path], limit: int | None = None) -> Iterator[list[Turn]]:
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
                for msg in msgs:
                    role = msg.get("role", "")
                    content = str(msg.get("content", "")).strip()
                    if not content or role not in ("system", "user", "assistant"):
                        continue
                    turns.append(Turn(role, content))
                    total_chars += len(content)
                if len(turns) >= 2 and total_chars <= 6000 and any(t.role == "assistant" for t in turns):
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
