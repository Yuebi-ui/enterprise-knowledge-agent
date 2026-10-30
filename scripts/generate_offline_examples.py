"""Actually execute the offline scripted pipeline, then save outputs and traces."""
from __future__ import annotations

import json
from pathlib import Path
from src.main import run_case
from src.offline_replay import OFFLINE_CASES

ROOT = Path(__file__).resolve().parent.parent


def save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    summary = []
    for case in OFFLINE_CASES:
        input_data = json.loads(
            (ROOT / f"examples/inputs/{case}.input.json").read_text(encoding="utf-8")
        )
        result = run_case(
            input_data["question"], input_data.get("history", []),
            ROOT / "data/documents.json",
            mode="offline", case=case
        )
        trace = result.pop("trace")
        prefix = ROOT / "examples/offline" / case
        save(prefix.with_suffix(".output.json"), result)
        save(prefix.with_suffix(".trace.json"), trace)
        summary.append({
            "case": case, "mode": result["mode"],
            "status": result["status"], "model_calls": result["metrics"]["model_calls"],
            "tool_calls": result["metrics"]["tool_calls"]
        })
        print(f"{case:14s} {result['status']:22s} tools={result['metrics']['tool_calls']}")
    save(ROOT / "examples/offline/manifest.json", {
        "notice": "Scripted offline results. They do NOT prove model-driven ReAct.",
        "cases": summary
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
