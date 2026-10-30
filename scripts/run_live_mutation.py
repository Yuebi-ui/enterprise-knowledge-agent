"""Real-LLM, source-sensitivity test (two independent document mutations).

Run without API:
    python -m scripts.run_live_mutation --scenario all --dry-run
Real DeepSeek (4 separate complete agent runs for --scenario all):
    python -m scripts.run_live_mutation --scenario all

This script never edits data/documents.json. The original and mutated documents
and resulting model outputs/traces are archived separately.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys

from src.kb import KnowledgeBase
from src.main import run_case

ROOT = Path(__file__).resolve().parent.parent

SCENARIOS = {
    "leave": {
        "doc_id": "D03",
        "old_text": "3个工作日",
        "new_text": "5个工作日",
        "question": "年假申请需要提前几天？",
        "goal": "The source answer should change from 3 to 5 business days.",
    },
    "conflict": {
        "doc_id": "D02",
        "old_text": "15个自然日",
        "new_text": "30个自然日",
        "question": "员工报销期限是多少天？",
        "goal": "D01/D02 originally conflict; after mutation both specify 30 calendar days.",
    }
}


def save(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )


def change_corpus(base: dict, scenario: dict) -> dict:
    changed = copy.deepcopy(base)
    count = 0
    for doc in changed["documents"]:
        if doc["doc_id"] != scenario["doc_id"]:
            continue
        for para in doc["paragraphs"]:
            text = para["text"]
            if scenario["old_text"] in text:
                para["text"] = text.replace(scenario["old_text"], scenario["new_text"])
                count += 1
    if count != 1:
        raise ValueError(f"Mutation expected exactly one paragraph, found {count}")
    return changed


def verified_quote(result: dict, needle: str) -> bool:
    return any(
        needle in citation.get("quote","")
        for citation in result.get("citations", [])
    )


def answer_has_fact(answer: str, num: int, unit: str) -> bool:
    # whitespace / 个-variations are acceptable; do not match bare "30".
    return re.search(rf"(?<!\d){num}\s*(?:个)?\s*{re.escape(unit)}", answer) is not None


def meets_goal(case: str, before: dict, after: dict) -> dict:
    if case == "leave":
        checks = {
            "original_completed": before.get("status") == "completed",
            "modified_completed": after.get("status") == "completed",
            "original_citation_3_business_days": verified_quote(before,"3个工作日"),
            "modified_citation_5_business_days": verified_quote(after,"5个工作日"),
            "original_answer_3_business_days": answer_has_fact(before.get("answer",""),3,"工作日"),
            "modified_answer_5_business_days": answer_has_fact(after.get("answer",""),5,"工作日"),
        }
    else:
        before_conflicts = before.get("conflicts") or []
        after_conflicts = after.get("conflicts") or []
        checks = {
            "original_requires_clarification": before.get("status") == "needs_clarification",
            "original_mentions_conflicting_docs": any(
                {"D01","D02"} <= set(c.get("doc_ids",[])) for c in before_conflicts
            ),
            "original_cites_15_and_30": (
                verified_quote(before,"15个自然日")
                and verified_quote(before,"30个自然日")
            ),
            "modified_completed": after.get("status") == "completed",
            "modified_conflicts_empty": after_conflicts == [],
            "modified_answer_30_calendar_days": answer_has_fact(after.get("answer",""),30,"自然日"),
            "modified_quote_30_calendar_days": verified_quote(after,"30个自然日"),
        }
    return {
        "checks":checks,
        "pass":all(checks.values())
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["leave","conflict","all"],default="leave")
    parser.add_argument("--out",default="examples/live_mutation_v3")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check source mutation and deterministic conflicts only; no DeepSeek calls"
    )
    args=parser.parse_args(argv)
    scenarios=list(SCENARIOS) if args.scenario=="all" else [args.scenario]
    original_path=ROOT/"data/documents.json"
    base=json.loads(original_path.read_text(encoding="utf-8"))
    out_root=Path(args.out)
    if not out_root.is_absolute():
        out_root=ROOT/out_root

    if not args.dry_run:
        from dotenv import load_dotenv
        load_dotenv(ROOT/".env")
        if not os.getenv("DEEPSEEK_API_KEY"):
            print("Set DEEPSEEK_API_KEY in .env before live mutation",file=sys.stderr)
            return 2

    results={}
    for name in scenarios:
        scenario=SCENARIOS[name]
        dest=out_root/name
        changed=change_corpus(base,scenario)
        modified_path=dest/"modified_documents.json"
        save(modified_path,changed)
        original_kb=KnowledgeBase.load(original_path)
        changed_kb=KnowledgeBase.load(modified_path)
        original_conflicts=original_kb.conflicts_among(
            {"D01","D02"},question="报销期限"
        )
        modified_conflicts=changed_kb.conflicts_among(
            {"D01","D02"},question="报销期限"
        )
        if args.dry_run:
            original_doc=original_kb.read(scenario["doc_id"])
            modified_doc=changed_kb.read(scenario["doc_id"])
            original_paragraph = next(
                p for p in original_doc["paragraphs"]
                if scenario["old_text"] in p["text"]
            )
            changed_paragraph = next(
                p for p in modified_doc["paragraphs"]
                if scenario["new_text"] in p["text"]
            )
            local={
                "mode":"offline_source_check_NOT_a_model_run",
                "mutation":scenario,
                "original_source_paragraph":original_paragraph,
                "modified_source_paragraph":changed_paragraph,
                "original_reimbursement_conflicts":original_conflicts,
                "modified_reimbursement_conflicts":modified_conflicts,
                "source_changed":True
            }
            save(dest/"offline_source_check.json",local)
            results[name]=local
            print(
                f"[OFFLINE source only] {name}: "
                f"{scenario['doc_id']} {scenario['old_text']} -> {scenario['new_text']}; "
                f"D01/D02 conflicts {len(original_conflicts)} -> {len(modified_conflicts)}"
            )
            continue

        runs={}
        for label,source in [
            ("original",original_path),
            ("modified",modified_path)
        ]:
            result=run_case(scenario["question"],[],source,"live")
            trace=result.pop("trace")
            save(dest/f"{label}.output.json",result)
            save(dest/f"{label}.trace.json",trace)
            runs[label]=result
            print(
                f"[LIVE] {name}/{label}: {result['status']} "
                f"model={result['metrics']['model_calls']} "
                f"tools={result['metrics']['tool_calls']} "
                f"answer={result['answer'][:105]}"
            )
        check=meets_goal(name,runs["original"],runs["modified"])
        results[name]={
            "mode":"live_deepseek",
            "mutation":scenario,
            "original_status":runs["original"]["status"],
            "modified_status":runs["modified"]["status"],
            "original_reimbursement_conflicts":original_conflicts,
            "modified_reimbursement_conflicts":modified_conflicts,
            **check
        }
        save(dest/"manifest.json",results[name])
        print(f"[LIVE] {name} acceptance={check['pass']}; checks={check['checks']}")

    save(out_root/"summary.json",{
        "run_type":"offline_source_check" if args.dry_run else "live_deepseek",
        "warning":"Offline source checks are NOT model decision traces.",
        "scenarios":results,
        "all_pass": (
            all(r.get("pass") for r in results.values())
            if not args.dry_run else None
        ),
        "original_data_unchanged":True
    })
    if args.dry_run:
        return 0
    return 0 if all(r["pass"] for r in results.values()) else 1


if __name__=="__main__":
    raise SystemExit(main())
