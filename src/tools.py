"""Tool definitions, validation, injection of *business-tool* faults, dispatch."""
from __future__ import annotations

from typing import Any
from .kb import KnowledgeBase

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search synthetic enterprise policy/product documents. Returns IDs, "
                "version dates, scopes and SHORT summaries only. For a cited answer "
                "you MUST call read_document. For ambiguous policies read all relevant docs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Contextualized search query; resolve follow-ups using conversation history."
                    },
                    "limit": {"type": "integer", "description": "Max hits (1 to 10)."}
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "Read original evidence paragraphs by document ID. "
                "Optionally provide paragraph_id to read one specific paragraph; "
                "omit it to read the full short document. "
                "Use returned text/version_date for citations; search summaries are not evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Exact document ID, e.g. D03"},
                    "paragraph_id": {"type": "string",
                                     "description": "Optional exact paragraph ID from search matched_paragraph_ids, e.g. p1"}
                },
                "required": ["doc_id"],
                "additionalProperties": False
            }
        }
    }
]


class FaultPolicy:
    """Deterministic failure injection for offline tests and live recovery tests."""
    def __init__(self, spec: str | None = None):
        self.tool = None
        self.kind = None
        self.count = 0
        if spec:
            parts = spec.split(":")
            if len(parts) != 2 or parts[0] not in {
                "search_documents", "read_document"
            } or parts[1] not in {"once", "always"}:
                raise ValueError("Expected --fail-tool name:once or name:always")
            self.tool, self.kind = parts

    def fail(self, name: str) -> bool:
        if self.tool != name:
            return False
        self.count += 1
        return self.kind == "always" or self.count == 1


class ToolRegistry:
    def __init__(self, kb: KnowledgeBase, fault: FaultPolicy | None = None):
        self.kb = kb
        self.fault = fault or FaultPolicy()

    @staticmethod
    def _error(code: str, message: str, retryable: bool = False) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {"code": code, "message": message},
            "retryable": retryable
        }

    def execute(self, name: str, arguments: Any) -> dict[str, Any]:
        if name not in {"search_documents", "read_document"}:
            return self._error("UNKNOWN_TOOL", f"Unknown tool: {name}")
        if not isinstance(arguments, dict):
            return self._error("INVALID_ARGUMENTS", "Arguments must be a JSON object")

        allowed = {"query", "limit"} if name == "search_documents" else {"doc_id", "paragraph_id"}
        if set(arguments) - allowed:
            return self._error("INVALID_ARGUMENTS", "Unexpected argument names")

        if name == "search_documents":
            query = arguments.get("query")
            limit = arguments.get("limit", 5)
            if not isinstance(query, str) or not query.strip() or len(query) > 300:
                return self._error("INVALID_ARGUMENTS", "query must have 1-300 chars")
            if type(limit) is not int or not 1 <= limit <= 10:
                return self._error("INVALID_ARGUMENTS", "limit must be an integer 1-10")
        else:
            doc_id = arguments.get("doc_id")
            if not isinstance(doc_id, str) or not 1 <= len(doc_id) <= 40:
                return self._error("INVALID_ARGUMENTS", "doc_id must be a nonempty string")
            paragraph_id = arguments.get("paragraph_id")
            if paragraph_id is not None and (
                not isinstance(paragraph_id, str)
                or not 1 <= len(paragraph_id) <= 40
            ):
                return self._error("INVALID_ARGUMENTS", "paragraph_id must be a nonempty string")

        if self.fault.fail(name):
            return self._error(
                "TOOL_TEMPORARY_FAILURE",
                f"Injected temporary failure in {name}",
                retryable=True
            )

        if name == "search_documents":
            results = self.kb.search(query, limit)
            return {
                "ok": True,
                "query": query,
                "count": len(results),
                "results": results,
                "note": "Search summaries are not original citations; read documents before answering."
            }

        paragraph_id = arguments.get("paragraph_id")
        if doc_id not in self.kb.by_id:
            return self._error("DOCUMENT_NOT_FOUND", f"No document: {doc_id}")
        document = self.kb.read(doc_id, paragraph_id=paragraph_id)
        if document is None:
            return self._error(
                "PARAGRAPH_NOT_FOUND",
                f"No paragraph {paragraph_id} in document {doc_id}"
            )
        return {"ok": True, "document": document}
