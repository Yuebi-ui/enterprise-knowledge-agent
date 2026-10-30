"""Run six REAL DeepSeek cases and preserve every actual result; no fabricated logs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from src.main import run_case
from src.offline_replay import OFFLINE_CASES

ROOT = Path(__file__).resolve().parent.parent


def save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def checks(case: str, result: dict) -> dict:
    status = result["status"]
    citations = {c["doc_id"] for c in result.get("citations", [])}
    events = result["trace"]
    called = [x["tool"] for x in events if x["event"] == "tool_result"]
    tool_results = [x["observation"] for x in events if x["event"] == "tool_result"]
    has_failure = any(not x.get("ok", False) for x in tool_results)
    did_search = "search_documents" in called
    criteria = {
        "normal": status == "completed" and "D03" in citations and did_search,
        "conflict": (status == "needs_clarification" and {"D01", "D02"} <= citations
                     and called.count("read_document") >= 2),
        "out_of_scope": status in {"insufficient_evidence", "needs_clarification"} and did_search,
        "paraphrase": status == "completed" and "D03" in citations and did_search,
        "follow_up": (status == "completed" and any(
            c["doc_id"] == "D03" and c["paragraph_id"] == "p2"
            for c in result.get("citations", []))),
        "tool_failure": (
            has_failure
            and (status in {"tool_error", "incomplete", "insufficient_evidence"}
                 or (status == "completed" and bool(citations)))
        )
    }
    return {
        "passed": criteria[case],
        "status": status,
        "called_tools": called,
        "cited_docs": sorted(citations),
        "injected_tool_failure_observed": has_failure,
        "note": "Heuristic checks only. Inspect answer semantics and traces manually."
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=OFFLINE_CASES,
                        default=OFFLINE_CASES, help="Cases to run (default: all six)")
    parser.add_argument("--out", default="examples/live")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=12)
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("No DEEPSEEK_API_KEY. Configure .env before running live evaluation.", file=sys.stderr)
        return 2

    dest = ROOT / args.out
    outcomes = []
    for case in args.only:
        example = json.loads(
            (ROOT / f"examples/inputs/{case}.input.json").read_text(encoding="utf-8")
        )
        injected = "search_documents:once" if case == "tool_failure" else None
        try:
            result = run_case(
                example["question"], example.get("history", []),
                ROOT / "data/documents.json",
                mode="live", fail_tool=injected,
                max_steps=args.max_steps,
                max_tool_calls=args.max_tool_calls
            )
        except Exception as exc:
            result = {
                "mode": "live_deepseek",
                "status": "setup_error",
                "answer": f"Live case setup failed: {type(exc).__name__}",
                "citations": [], "trace": [], "metrics": {}
            }
        trace = result.pop("trace")
        score = checks(case, {**result, "trace": trace}) if result["status"] != "setup_error" else {
            "passed": False, "status": "setup_error", "note": "Setup failed"
        }
        save(dest / f"{case}.input.json", example)
        save(dest / f"{case}.output.json", result)
        save(dest / f"{case}.trace.json", trace)
        outcomes.append({"case": case, **score})
        print(f"{case:14s} {result['status']:22s} heuristic_pass={score['passed']}")

    save(dest / "manifest.json", {
        "mode": "live_deepseek", "model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        "data_source": "synthetic documents",
        "summary": {
            "total": len(outcomes), "heuristic_passed": sum(x["passed"] for x in outcomes)
        },
        "cases": outcomes,
        "notice": "Real provider call results. Scores are heuristic; manually inspect correctness."
    })
    print(f"Saved actual live outputs and traces to {dest}")
    return 0 if all(x["passed"] for x in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
