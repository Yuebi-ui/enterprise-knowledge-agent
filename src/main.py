"""CLI entry point. Offline is visibly labeled; live calls DeepSeek."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .agent import KnowledgeAgent
from .kb import KnowledgeBase
from .tools import ToolRegistry, FaultPolicy
from .offline_replay import offline_model, OFFLINE_CASES
from .provider import DeepSeekModel

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "documents.json"


def run_case(question: str, history: list[dict], data_path: str | Path,
             mode: str, case: str | None = None, fail_tool: str | None = None,
             max_steps: int = 8, max_tool_calls: int = 12) -> dict:
    kb = KnowledgeBase.load(data_path)
    if mode == "offline":
        if not case:
            raise ValueError("Offline mode requires --case")
        model = offline_model(case)
        if case == "tool_failure" and fail_tool is None:
            fail_tool = "search_documents:once"
    elif mode == "live":
        model = DeepSeekModel()
    else:
        raise ValueError("mode must be live or offline")

    return KnowledgeAgent(
        model,
        ToolRegistry(kb, FaultPolicy(fail_tool)),
        max_steps=max_steps,
        max_tool_calls=max_tool_calls
    ).run(question, history)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evidence-grounded enterprise knowledge QA Agent"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="JSON file containing question and optional history")
    source.add_argument("--question", help="Question as plain text")
    parser.add_argument("--mode", choices=["live", "offline"], default="live")
    parser.add_argument("--case", choices=OFFLINE_CASES, help="Required for offline mode")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Knowledge base JSON")
    parser.add_argument("--fail-tool", help="Fault injection, e.g. search_documents:once")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=12)
    parser.add_argument("--output", help="Write whole JSON result to this file")
    args = parser.parse_args(argv)

    try:
        if args.input:
            content = json.loads(Path(args.input).read_text(encoding="utf-8"))
            question = content["question"]
            history = content.get("history", [])
        else:
            question, history = args.question, []
        answer = run_case(
            question, history, args.data, args.mode,
            case=args.case, fail_tool=args.fail_tool,
            max_steps=args.max_steps, max_tool_calls=args.max_tool_calls
        )
        serialized = json.dumps(answer, ensure_ascii=False, indent=2)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
        return 0
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({
            "status": "setup_error",
            "message": str(exc)
        }, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
