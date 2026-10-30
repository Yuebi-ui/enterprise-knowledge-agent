"""Independent regression tests for mutable evidence and sourced assertions.

All tests run locally without an API key or a live model. They do not
constitute genuine model-driven ReAct evidence.
"""
from __future__ import annotations

import copy
import unittest

from src.agent import KnowledgeAgent
from src.kb import KnowledgeBase
from src.provider import ScriptedModel
from src.offline_replay import final, tool, D03_P1
from src.tools import ToolRegistry
from src.validation import validate_final
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def citation(doc: dict, paragraph_id: str = "p1") -> dict:
    paragraph = next(
        p for p in doc["paragraphs"] if p["paragraph_id"] == paragraph_id
    )
    return {
        "doc_id": doc["doc_id"],
        "paragraph_id": paragraph_id,
        "version_date": doc["version_date"],
        "quote": paragraph["text"],
    }


class MutableKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.original = KnowledgeBase.load(ROOT / "data" / "documents.json")

    def changed(self, doc_id: str, old: str, new: str) -> KnowledgeBase:
        docs = copy.deepcopy(self.original.documents)
        found = False
        for document in docs:
            if document["doc_id"] == doc_id:
                for paragraph in document["paragraphs"]:
                    if old in paragraph["text"]:
                        paragraph["text"] = paragraph["text"].replace(old, new)
                        found = True
        self.assertTrue(found, "Mutation target not found")
        return KnowledgeBase(docs)

    def test_conflict_reflects_current_source_text_not_metadata(self):
        docs = copy.deepcopy(self.original.documents)
        for doc in docs:
            if doc["doc_id"] == "D02":
                doc["fact_value"] = "15天"  # stale metadata must be ignored
                doc["paragraphs"][0]["text"] = (
                    "员工费用发生后，须在30个自然日内提交报销申请。"
                )
        kb = KnowledgeBase(docs)
        self.assertEqual(
            kb.conflicts_among({"D01", "D02"}, "员工报销期限是多少天？"),
            []
        )
        agreement = kb.agreeing_groups_among({"D01", "D02"}, "员工报销期限是多少天？")
        self.assertEqual(len(agreement), 1)
        self.assertEqual(agreement[0]["value"], ("30", "自然日"))

    def test_mutation_to_new_deadline_uses_new_fact(self):
        kb = self.changed("D02", "15个自然日", "40个自然日")
        conflicts = kb.conflicts_among({"D01","D02"}, "员工报销期限是多少天？")
        self.assertEqual(len(conflicts), 1)
        self.assertIn("40自然日", conflicts[0]["reason"])
        self.assertNotIn("15自然日", conflicts[0]["reason"])

    def test_change_unit_causes_policy_conflict(self):
        kb = self.changed("D02", "15个自然日", "30个工作日")
        self.assertEqual(
            len(kb.conflicts_among({"D01","D02"}, "报销期限")), 1
        )

    def test_unrelated_question_not_affected_by_reimbursement_conflict(self):
        self.assertEqual(
            self.original.conflicts_among(
                {"D01","D02","D03"}, "年假申请需要提前几天？"
            ), []
        )

    def test_agreeing_sources_allow_completion(self):
        kb = self.changed("D02", "15个自然日", "30个自然日")
        docs = {name: kb.read(name) for name in ("D01","D02")}
        result = validate_final(
            {
                "status": "completed",
                "answer": "两份相关制度均要求在30个自然日内提交报销申请。",
                "citations": [citation(docs["D01"]),citation(docs["D02"])],
                "conflicts": [],
                "missing_information": [],
            },
            kb, {"D01","D02"}, docs, 1,
            question="员工报销期限是多少天？"
        )
        self.assertEqual(result["status"], "completed")

    def test_spurious_conflict_after_mutation_is_rejected(self):
        kb = self.changed("D02", "15个自然日", "30个自然日")
        docs = {name: kb.read(name) for name in ("D01","D02")}
        result = validate_final(
            {
                "status": "needs_clarification",
                "answer": "两份制度存在期限冲突，暂无法判断。",
                "citations": [citation(docs["D01"]),citation(docs["D02"])],
                "conflicts": [{"topic":"报销期限","doc_ids":["D01","D02"],
                               "reason":"两份制度冲突"}],
                "missing_information": ["需要确认哪个版本"],
            },
            kb, {"D01","D02"}, docs, 1,
            question="员工报销期限是多少天？"
        )
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("spurious_conflict:D01,D02", result["validation_errors"])

    def test_no_static_fact_value_needed(self):
        self.assertTrue(all(
            "fact_value" not in doc for doc in self.original.documents
        ))
        self.assertEqual(
            len(self.original.conflicts_among(
                {"D01","D02"}, "报销期限"
            )), 1
        )

    def test_paragraph_scoped_read_and_search(self):
        matches = self.original.search("年假申请提前多久")
        d03 = next(d for d in matches if d["doc_id"] == "D03")
        self.assertIn("p1", d03["matched_paragraph_ids"])
        tools = ToolRegistry(self.original)
        result = tools.execute(
            "read_document", {"doc_id":"D03", "paragraph_id":"p1"}
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            [p["paragraph_id"] for p in result["document"]["paragraphs"]],
            ["p1"]
        )
        self.assertEqual(
            tools.execute(
                "read_document", {"doc_id":"D03", "paragraph_id":"p99"}
            )["error"]["code"], "PARAGRAPH_NOT_FOUND"
        )
        self.assertFalse(tools.execute(
            "read_document", {"doc_id":"D03", "paragraph_id":42}
        )["ok"])

    def test_unread_paragraph_cannot_be_cited(self):
        partial = self.original.read("D03",paragraph_id="p1")
        full = self.original.read("D03")
        result = validate_final(
            {
                "status": "completed",
                "answer": "由直属主管审批。",
                "citations": [citation(full,"p2")],
            }, self.original, {"D03"}, {"D03":partial}, 1,
            question="年假由谁审批"
        )
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("validation_errors", result)

    def test_multiple_precise_reads_preserve_both_observations(self):
        full = self.original.read("D03")
        plan = [
            tool("search_documents", query="年假 申请 审批"),
            tool("read_document", doc_id="D03", paragraph_id="p1"),
            tool("read_document", doc_id="D03", paragraph_id="p2"),
            final(
                "completed",
                "年假至少提前3个工作日申请，由直属主管审批。",
                [citation(full,"p1"),citation(full,"p2")]
            )
        ]
        result = KnowledgeAgent(
            ScriptedModel(plan), ToolRegistry(self.original)
        ).run("年假提前多久并由谁审批？")
        self.assertEqual(result["status"],"completed")
        self.assertEqual(
            {c["paragraph_id"] for c in result["citations"]},
            {"p1","p2"}
        )


class SemanticEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase.load(ROOT/"data"/"documents.json")

    def validate_answer(self, answer: str, doc_id: str, paragraphs: tuple[str,...]):
        doc = self.kb.read(doc_id)
        return validate_final(
            {
                "status":"completed", "answer":answer,
                "citations":[citation(doc,p) for p in paragraphs],
                "missing_information":[],
            },
            self.kb, {doc_id}, {doc_id:doc}, 1,
            question="请问制度是什么？"
        )

    def test_genuine_quote_but_false_approval_claim_blocked(self):
        out = self.validate_answer("年假不需要任何审批。", "D03", ("p1","p2"))
        self.assertEqual(out["status"],"incomplete")
        self.assertIn("contradictory_approval_requirement",out["validation_errors"])

    def test_wrong_approval_actor_blocked(self):
        out = self.validate_answer("年假由人力资源部审批。", "D03", ("p2",))
        self.assertEqual(out["status"],"incomplete")
        self.assertIn("unsupported_approval_actor",out["validation_errors"])

    def test_genuine_approval_actor_accepted(self):
        out = self.validate_answer("年假由直属主管审批。", "D03", ("p2",))
        self.assertEqual(out["status"],"completed")

    def test_feature_support_reversal_blocked(self):
        out = self.validate_answer("星河协作标准版支持离线编辑。", "D06", ("p2",))
        self.assertEqual(out["status"],"incomplete")
        self.assertIn("contradictory_feature_support",out["validation_errors"])

    def test_unsupported_feature_without_matching_quote_blocked(self):
        out = self.validate_answer("星河协作标准版支持离线编辑。", "D06", ("p1",))
        self.assertEqual(out["status"],"incomplete")
        self.assertIn("unsupported_feature_claim",out["validation_errors"])

    def test_genuine_feature_claim_accepted(self):
        out = self.validate_answer(
            "星河协作标准版暂不支持离线编辑。", "D06", ("p2",)
        )
        self.assertEqual(out["status"],"completed")

    def test_real_quote_from_unrelated_topic_cannot_support_answer(self):
        # A genuine reimbursement quote cannot support a claim about annual leave.
        out = validate_final(
            {
                "status": "completed",
                "answer": "年假应提前30个自然日申请。",
                "citations": [citation(self.kb.read("D01"))],
                "missing_information": [],
            },
            self.kb,
            {"D01", "D03"},
            {"D01":self.kb.read("D01")}, 1,
            question="年假申请需要提前几天？"
        )
        self.assertEqual(out["status"], "incomplete")
        self.assertIn("citation_topic_mismatch",out["validation_errors"])

    def test_valid_leave_number_and_approval_pass(self):
        out = self.validate_answer(
            "年假申请至少提前3个工作日，由直属主管审批。",
            "D03", ("p1","p2")
        )
        self.assertEqual(out["status"],"completed")


class HistoricalLiveTraceCompatibilityTests(unittest.TestCase):
    """Safeguard: code changes should not reject archived, previously valid V2 runs."""

    def test_archived_six_real_outputs_still_validate(self):
        import json
        root = ROOT
        kb = KnowledgeBase.load(root / "data" / "documents.json")
        expected = {
            "normal": "completed",
            "conflict": "needs_clarification",
            "out_of_scope": "insufficient_evidence",
            "paraphrase": "completed",
            "follow_up": "completed",
            "tool_failure": "completed",
        }
        for case, wanted in expected.items():
            with self.subTest(case=case):
                output = json.loads(
                    (root/"examples/live"/f"{case}.output.json").read_text(encoding="utf8")
                )
                trace = json.loads(
                    (root/"examples/live"/f"{case}.trace.json").read_text(encoding="utf8")
                )
                searched=set()
                read_docs={}
                success=0
                for event in trace:
                    if event.get("event") != "tool_result":
                        continue
                    ob=event.get("observation", {})
                    if event.get("tool")=="search_documents" and ob.get("ok"):
                        success+=1
                        searched.update(d["doc_id"] for d in ob.get("results",[]))
                    if event.get("tool")=="read_document" and ob.get("ok"):
                        doc=ob["document"]
                        read_docs[doc["doc_id"]]=doc
                checked=validate_final(
                    output,kb,searched,read_docs,success,question=output["question"]
                )
                self.assertEqual(checked["status"], wanted,
                                 f"Unexpected regression on archived {case}")



if __name__ == "__main__":
    unittest.main()
