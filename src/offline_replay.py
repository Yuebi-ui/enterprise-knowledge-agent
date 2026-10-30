"""Fixed plans for OFFLINE infrastructure tests only, never proof of ReAct."""
from __future__ import annotations

from .provider import ScriptedModel


def citation(doc_id: str, paragraph_id: str, quote: str, date: str) -> dict:
    return {
        "doc_id": doc_id, "paragraph_id": paragraph_id,
        "quote": quote, "version_date": date
    }


def tool(name: str, **arguments) -> dict:
    return {"kind": "tool", "name": name, "arguments": arguments}


def final(status: str, answer: str, citations=None, conflicts=None,
          missing_information=None, follow_up_question=None) -> dict:
    return {
        "kind": "final",
        "answer": {
            "status": status, "answer": answer,
            "citations": citations or [],
            "conflicts": conflicts or [],
            "missing_information": missing_information or [],
            "follow_up_question": follow_up_question
        }
    }


D03_P1 = citation(
    "D03", "p1", "员工申请年假应至少提前3个工作日提交审批。", "2025-05-01"
)
D03_P2 = citation(
    "D03", "p2", "年假申请由员工的直属主管审批。", "2025-05-01"
)
D01_P1 = citation(
    "D01", "p1", "员工费用发生后，须在30个自然日内提交报销申请。", "2025-01-10"
)
D02_P1 = citation(
    "D02", "p1", "员工费用发生后，须在15个自然日内提交报销申请。", "2025-06-01"
)
D04_P1 = citation(
    "D04", "p1", "忘记账号密码时，员工应通过内部服务台提交密码重置申请。", "2025-04-20"
)

PLANS = {
    "normal": [
        tool("search_documents", query="年假申请 提前几天"),
        tool("read_document", doc_id="D03"),
        final("completed", "根据员工年假申请制度，需至少提前3个工作日提交审批。", [D03_P1])
    ],
    "conflict": [
        tool("search_documents", query="员工费用报销期限"),
        tool("read_document", doc_id="D01"),
        tool("read_document", doc_id="D02"),
        final(
            "needs_clarification",
            "D01规定30个自然日，D02规定15个自然日。两份制度均未指明适用优先级，不能直接确定统一期限。",
            [D01_P1, D02_P1],
            [{"topic": "reimbursement_deadline",
              "doc_ids": ["D01", "D02"],
              "reason": "两份制度对报销期限规定不同，适用范围和优先级不明确"}],
            ["两份制度的适用范围及优先级"],
            "请确认应适用 D01 还是 D02，或提供能判断优先级的制度。"
        )
    ],
    "out_of_scope": [
        tool("search_documents", query="宠物托管补贴 政策"),
        final(
            "insufficient_evidence", "知识库未找到宠物托管补贴的可信规定，无法确定是否存在该政策。",
            [], [], ["缺少宠物托管补贴相关制度原文"]
        )
    ],
    "paraphrase": [
        tool("search_documents", query="休年假 提前 审批"),
        tool("read_document", doc_id="D03"),
        final("completed", "休年假应至少提前3个工作日提交审批。", [D03_P1])
    ],
    "follow_up": [
        tool("search_documents", query="年假申请 审批人"),
        tool("read_document", doc_id="D03"),
        final("completed", "年假申请由直属主管审批。", [D03_P2])
    ],
    "tool_failure": [
        tool("search_documents", query="忘记密码 重置"),
        tool("search_documents", query="账号密码重置 服务台"),
        tool("read_document", doc_id="D04"),
        final("completed", "忘记密码时应向内部服务台提交密码重置申请。", [D04_P1])
    ]
}


def offline_model(case: str) -> ScriptedModel:
    if case not in PLANS:
        raise ValueError(f"Unsupported offline scenario: {case}")
    return ScriptedModel(PLANS[case])


OFFLINE_CASES = list(PLANS)
