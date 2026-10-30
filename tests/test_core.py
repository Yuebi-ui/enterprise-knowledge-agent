from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from src.agent import KnowledgeAgent
from src.kb import KnowledgeBase
from src.offline_replay import OFFLINE_CASES, offline_model, final, tool, D03_P1
from src.provider import ScriptedModel, DeepSeekModel, ModelReply, ToolCall
from src.tools import FaultPolicy, ToolRegistry
from src.validation import validate_final


ROOT = Path(__file__).resolve().parent.parent


class KnowledgeAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase.load(ROOT / "data/documents.json")

    def run_offline(self, case, max_steps=8, max_tools=12, fault=None, history=None):
        if case == "tool_failure" and fault is None:
            fault = "search_documents:once"
        return KnowledgeAgent(
            offline_model(case), ToolRegistry(self.kb, FaultPolicy(fault)),
            max_steps=max_steps, max_tool_calls=max_tools
        ).run("synthetic test question", history)



    def test_live_mode_validator_feedback_allows_model_repair(self):
        class FakeLive:
            mode = "live_deepseek"
            model_name = "no-network-test"
            def __init__(self):
                self.n = 0
            def complete(self, messages, tools):
                self.n += 1
                if self.n == 1:
                    return ModelReply(tool_calls=[
                        ToolCall("one", "search_documents", '{"query":"年假"}')])
                if self.n == 2:
                    return ModelReply(tool_calls=[
                        ToolCall("two", "read_document", '{"doc_id":"D03"}')])
                if self.n == 3:
                    return ModelReply(content='{"status":"completed","answer":"提前99个工作日",'
                                      '"citations":[{"doc_id":"D03","paragraph_id":"p1",'
                                      '"version_date":"2025-05-01","quote":'
                                      '"员工申请年假应至少提前3个工作日提交审批。"}]}')
                self.assert_message_contains_feedback(messages)
                return ModelReply(content='{"status":"completed","answer":"提前3个工作日",'
                                  '"citations":[{"doc_id":"D03","paragraph_id":"p1",'
                                  '"version_date":"2025-05-01","quote":'
                                  '"员工申请年假应至少提前3个工作日提交审批。"}]}')
            def assert_message_contains_feedback(self, messages):
                assert "未通过可验证性检查" in messages[-1]["content"]

        result = KnowledgeAgent(FakeLive(), ToolRegistry(self.kb)).run("年假提前多久")
        self.assertEqual(result["status"], "completed")
        self.assertTrue(any(x["event"] == "validation_feedback" for x in result["trace"]))

    def test_deepseek_adapter_protocol_without_network(self):
        seen = {}
        class FakeCompletion:
            def create(self, **kwargs):
                seen.update(kwargs)
                call = SimpleNamespace(
                    id="call_abc",
                    function=SimpleNamespace(name="search_documents",
                                             arguments='{"query":"年假"}')
                )
                msg = SimpleNamespace(content=None, tool_calls=[call])
                usage = SimpleNamespace(
                    prompt_tokens=10, completion_tokens=4, total_tokens=14
                )
                return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)

        # Avoid constructing real OpenAI client or using secrets/network.
        fake_model = object.__new__(DeepSeekModel)
        fake_model.model_name = "deepseek-flash"
        fake_model.client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletion())
        )
        answer = fake_model.complete([{"role": "user", "content": "test"}], [])
        self.assertEqual(answer.tool_calls[0].name, "search_documents")
        self.assertEqual(answer.usage["total_tokens"], 14)
        self.assertEqual(seen["tool_choice"], "auto")
        self.assertEqual(seen["extra_body"]["thinking"]["type"], "disabled")

    def test_trace_contains_observation_before_next_decision(self):
        out = self.run_offline("normal")
        decisions = [e for e in out["trace"] if e["event"] == "model_decision"]
        self.assertEqual(decisions[0]["observed_before_decision"], [])
        self.assertIn("D03", " ".join(decisions[1]["observed_before_decision"]))
        self.assertTrue(all("observed_before_decision" in e for e in decisions))

    def test_completed_and_missing_info_produces_actionable_feedback(self):
        result = validate_final(
            {"status": "completed",
             "answer": "员工申请年假应至少提前3个工作日提交审批。",
             "citations": [D03_P1],
             "missing_information": ["可能存在未知的特殊规定"]},
            self.kb, {"D03"}, {"D03": self.kb.read("D03")}, 1,
            question="年假申请需要提前几天？"
        )
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("missing_information", result["answer"])

    def test_real_world_leave_search_excludes_unrelated_reimbursements(self):
        matches = self.kb.search("年假申请 提前几天 提交时限")
        self.assertEqual([d["doc_id"] for d in matches], ["D03"])

    def test_unrelated_conflict_cannot_block_leave_answer_even_if_retrieved(self):
        # Regression for the user's actual DeepSeek trace: D01, D02 happened
        # to be retrieved while answering an unrelated annual leave question.
        result = validate_final(
            {"status": "completed",
             "answer": "员工申请年假应至少提前3个工作日提交审批。",
             "citations": [D03_P1],
             "missing_information": []},
            self.kb,
            {"D01", "D02", "D03"},
            {"D03": self.kb.read("D03")},
            1,
            question="年假申请需要提前几天？"
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["conflicts"], [])

    def test_reimbursement_topic_still_requires_both_conflicting_docs(self):
        result = validate_final(
            {"status": "completed", "answer": "报销期限是30个自然日",
             "citations": [{"doc_id": "D01", "paragraph_id": "p1",
                            "version_date": "2025-01-10",
                            "quote": "员工费用发生后，须在30个自然日内提交报销申请。"}], "missing_information": []},
            self.kb,
            {"D01", "D02"},
            {"D01": self.kb.read("D01")},
            1,
            question="报销期限是多少天？"
        )
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(result["conflicts"])

    def test_wrong_time_unit_rejected_even_if_number_matches(self):
        plan = [
            tool("search_documents", query="年假"),
            tool("read_document", doc_id="D03"),
            final("completed", "员工年假提前3个自然日申请。", [D03_P1])
        ]
        out = KnowledgeAgent(
            ScriptedModel(plan), ToolRegistry(self.kb)
        ).run("年假提前多久")
        self.assertEqual(out["status"], "incomplete")
        self.assertIn("validation_errors", out)

    def test_retrieval_hits_both_conflicting_policies(self):
        docs = {d["doc_id"] for d in self.kb.search("报销期限", limit=10)}
        self.assertTrue({"D01", "D02"} <= docs)

    def test_retrieval_normal_and_unknown(self):
        docs = {d["doc_id"] for d in self.kb.search("年假申请提前多久")}
        self.assertIn("D03", docs)
        self.assertEqual(self.kb.search("宠物托管补贴政策"), [])

    def test_tool_argument_validation(self):
        tools = ToolRegistry(self.kb)
        self.assertEqual(tools.execute("read_document", {"doc_id": 42})["error"]["code"], "INVALID_ARGUMENTS")
        self.assertEqual(tools.execute("search_documents", {"query": ""})["error"]["code"], "INVALID_ARGUMENTS")
        self.assertEqual(tools.execute("fake_tool", {})["error"]["code"], "UNKNOWN_TOOL")
        self.assertEqual(tools.execute("read_document", {"doc_id": "D99"})["error"]["code"], "DOCUMENT_NOT_FOUND")
        self.assertFalse(tools.execute("search_documents", {"query": "年假", "limit": True})["ok"])

    def test_once_fault_only_fails_first_call(self):
        tools = ToolRegistry(self.kb, FaultPolicy("search_documents:once"))
        self.assertFalse(tools.execute("search_documents", {"query": "年假"})["ok"])
        self.assertTrue(tools.execute("search_documents", {"query": "年假"})["ok"])

    def test_all_six_offline_scenarios(self):
        expected = {
            "normal": "completed",
            "conflict": "needs_clarification",
            "out_of_scope": "insufficient_evidence",
            "paraphrase": "completed",
            "follow_up": "completed",
            "tool_failure": "completed",
        }
        for name in OFFLINE_CASES:
            with self.subTest(case=name):
                result = self.run_offline(name)
                self.assertEqual(result["mode"], "offline_scripted")
                self.assertEqual(result["status"], expected[name])
                self.assertTrue(any(t["event"] == "model_decision" for t in result["trace"]))
                self.assertTrue(any(t["event"] == "tool_result" for t in result["trace"]))
                if result["status"] == "completed":
                    self.assertGreaterEqual(len(result["citations"]), 1)

    def test_conflict_evidence_contains_both_docs(self):
        out = self.run_offline("conflict")
        self.assertEqual({c["doc_id"] for c in out["citations"]}, {"D01", "D02"})
        self.assertTrue(out["conflicts"])
        self.assertTrue(out["follow_up_question"])

    def test_follow_up_references_approval_paragraph(self):
        out = self.run_offline("follow_up")
        self.assertEqual(out["citations"][0]["paragraph_id"], "p2")

    def test_tool_failure_retry_is_explicitly_visible(self):
        out = self.run_offline("tool_failure")
        events = [t for t in out["trace"] if t["event"] == "tool_result"]
        self.assertEqual(events[0]["observation"]["error"]["code"], "TOOL_TEMPORARY_FAILURE")
        self.assertTrue(events[1]["observation"]["ok"])
        self.assertEqual(out["status"], "completed")

    def test_read_failure_cannot_claim_completion(self):
        out = self.run_offline("normal", fault="read_document:always")
        self.assertEqual(out["status"], "incomplete")
        self.assertFalse(out["citations"])

    def test_step_limit_explicit_incomplete(self):
        out = self.run_offline("normal", max_steps=1)
        self.assertEqual(out["status"], "incomplete")
        self.assertIn("max_steps_exceeded", out["missing_information"])

    def test_tool_limit_explicit_incomplete(self):
        out = self.run_offline("normal", max_tools=1)
        self.assertEqual(out["status"], "incomplete")
        self.assertIn("max_tool_calls_exceeded", out["missing_information"])

    def test_invalid_citation_is_rejected(self):
        plan = [
            tool("search_documents", query="年假"),
            tool("read_document", doc_id="D03"),
            final("completed", "员工年假至少提前99天", [
                {**D03_P1, "quote": "员工年假至少提前99天"}
            ])
        ]
        out = KnowledgeAgent(
            ScriptedModel(plan), ToolRegistry(self.kb)
        ).run("年假怎么申请")
        self.assertEqual(out["status"], "incomplete")
        self.assertIn("validation_errors", out)


    def test_numeric_claim_without_evidence_is_rejected(self):
        plan = [
            tool("search_documents", query="年假"),
            tool("read_document", doc_id="D03"),
            final("completed", "年假要求提前99个工作日", [D03_P1])
        ]
        out = KnowledgeAgent(
            ScriptedModel(plan), ToolRegistry(self.kb)
        ).run("年假怎么申请")
        self.assertEqual(out["status"], "incomplete")
        self.assertIn("validation_errors", out)

    def test_answer_from_model_only_is_not_completed(self):
        plan = [final("completed", "年假提前3个工作日", [D03_P1])]
        out = KnowledgeAgent(
            ScriptedModel(plan), ToolRegistry(self.kb)
        ).run("年假怎么申请")
        self.assertEqual(out["status"], "incomplete")

    def test_unchecked_conflict_blocks_single_policy_answer(self):
        plan = [
            tool("search_documents", query="报销期限"),
            tool("read_document", doc_id="D01"),
            final("completed", "期限是30个自然日", [
                {"doc_id": "D01", "paragraph_id": "p1",
                 "version_date": "2025-01-10",
                 "quote": "员工费用发生后，须在30个自然日内提交报销申请。"}
            ])
        ]
        out = KnowledgeAgent(
            ScriptedModel(plan), ToolRegistry(self.kb)
        ).run("报销期限")
        self.assertEqual(out["status"], "incomplete")
        self.assertTrue(out["conflicts"])

    def test_replaced_document_invalidates_stale_citation(self):
        docs = copy.deepcopy(self.kb.documents)
        for doc in docs:
            if doc["doc_id"] == "D03":
                doc["paragraphs"][0]["text"] = "员工申请年假应至少提前5个工作日提交审批。"
        changed = KnowledgeBase(docs)
        result = validate_final(
            {"status": "completed", "answer": "提前3天", "citations": [D03_P1]},
            changed, {"D03"}, {"D03": changed.read("D03")}, 1
        )
        self.assertEqual(result["status"], "incomplete")
        valid = validate_final(
            {"status": "completed", "answer": "提前5个工作日",
             "citations": [{**D03_P1, "quote": "员工申请年假应至少提前5个工作日提交审批。"}]},
            changed, {"D03"}, {"D03": changed.read("D03")}, 1
        )
        self.assertEqual(valid["status"], "completed")

    def test_history_is_forwarded_to_model(self):
        class CapturingModel(ScriptedModel):
            def complete(self, messages, tools):
                self.received = messages
                return super().complete(messages, tools)

        model = CapturingModel([final(
            "insufficient_evidence", "资料不足", missing_information=["缺少证明"]
        )])
        KnowledgeAgent(model, ToolRegistry(self.kb)).run(
            "那是谁审批？", [{"role": "user", "content": "请问年假申请流程"},
                       {"role": "assistant", "content": "需要审批"}]
        )
        self.assertEqual(model.received[1]["role"], "user")
        self.assertEqual(model.received[-1]["content"], "那是谁审批？")

    def test_model_failure_does_not_claim_completion(self):
        out = KnowledgeAgent(
            ScriptedModel([]), ToolRegistry(self.kb)
        ).run("任何问题")
        self.assertEqual(out["status"], "tool_error")
        self.assertTrue(any(x["event"] == "model_error" for x in out["trace"]))


if __name__ == "__main__":
    unittest.main()
