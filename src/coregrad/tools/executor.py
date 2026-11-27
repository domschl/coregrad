"""Unified tool execution dispatcher for coregrad's 4 cognitive primitives."""

from __future__ import annotations

import json
from dataclasses import dataclass

from coregrad.tools.calc import safe_calc
from coregrad.tools.dsl import DSLError, safe_dsl
from coregrad.tools.protocol import ToolCall, parse_tool_call, validate_tool_call
from coregrad.tools.retriever import Retriever


@dataclass(frozen=True)
class ExecutionResult:
    ok: bool
    tool_name: str
    output: str
    retrieved_doc_ids: tuple[str, ...] = ()


class ToolExecutor:
    """Executes validated search, read, list_docs, and calc calls safely."""

    def __init__(self, retriever: Retriever | None = None) -> None:
        self.retriever = retriever

    def execute_raw(self, call_str: str) -> ExecutionResult:
        try:
            call = parse_tool_call(call_str)
        except (ValueError, TypeError) as exc:
            return ExecutionResult(ok=False, tool_name="parse_error", output=json.dumps({"error": str(exc)}))
        return self.execute(call)

    def execute(self, call: ToolCall) -> ExecutionResult:
        valid, msg = validate_tool_call(call)
        if not valid:
            return ExecutionResult(ok=False, tool_name=call.name, output=json.dumps({"error": msg}))

        if call.name == "calc":
            expr = str(call.args["expr"])
            try:
                val = safe_calc(expr)
                return ExecutionResult(ok=True, tool_name="calc", output=val)
            except (ValueError, ArithmeticError, OverflowError):
                try:
                    val = safe_dsl(expr)
                    return ExecutionResult(ok=True, tool_name="calc", output=val)
                except (DSLError, ValueError, ArithmeticError, OverflowError) as exc:
                    return ExecutionResult(ok=False, tool_name="calc", output=json.dumps({"error": str(exc)}))

        if self.retriever is None:
            return ExecutionResult(
                ok=False,
                tool_name=call.name,
                output=json.dumps({"error": "No document retriever is mounted."}),
            )

        if call.name == "search":
            k = int(call.args.get("k", 3))
            hits = self.retriever.search(str(call.args["query"]), k=k)
            payload = [h.to_compact_dict() for h in hits]
            doc_ids = tuple(h.doc_id for h in hits)
            return ExecutionResult(
                ok=True,
                tool_name="search",
                output=json.dumps(payload, ensure_ascii=False),
                retrieved_doc_ids=doc_ids,
            )

        if call.name == "read":
            doc_id = str(call.args["doc_id"])
            start = int(call.args.get("start", 0))
            length = int(call.args.get("length", 800))
            res = self.retriever.read(doc_id, start=start, length=length)
            ok = "error" not in res
            return ExecutionResult(
                ok=ok,
                tool_name="read",
                output=json.dumps(res, ensure_ascii=False),
                retrieved_doc_ids=(doc_id,) if ok else (),
            )

        if call.name == "list_docs":
            filt = str(call.args.get("filter", ""))
            docs = self.retriever.list_docs(filter_text=filt, limit=20)
            return ExecutionResult(
                ok=True,
                tool_name="list_docs",
                output=json.dumps(docs, ensure_ascii=False),
            )

        return ExecutionResult(
            ok=False,
            tool_name=call.name,
            output=json.dumps({"error": f"Unhandled tool: {call.name}"}),
        )
