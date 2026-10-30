"""Evidence-level final answer validation. It does not replace model decisions."""
from __future__ import annotations

import json
import re
from typing import Any
from .kb import KnowledgeBase
from .evidence import check_explicit_claims

VALID_STATUSES = {
    "completed", "needs_clarification", "insufficient_evidence",
    "tool_error", "incomplete"
}


def parse_final(content: str | None) -> dict[str, Any] | None:
    if not content:
        return None
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def validate_final(
    candidate: dict[str, Any] | None,
    kb: KnowledgeBase,
    searched_ids: set[str],
    read_documents: dict[str, dict[str, Any]],
    successful_searches: int,
    question: str | None = None,
) -> dict[str, Any]:
    if candidate is None:
        return _fallback("模型最终回复不是可解析的 JSON 对象", "incomplete")

    status = candidate.get("status", "incomplete")
    if status not in VALID_STATUSES:
        return _fallback("模型输出了不支持的任务状态", "incomplete")

    answer = candidate.get("answer", "")
    if not isinstance(answer, str) or not answer.strip():
        return _fallback("模型未给出有效回答", "incomplete")
    answer = answer[:5000]
    raw_citations = candidate.get("citations", [])
    if not isinstance(raw_citations, list) or len(raw_citations) > 20:
        return _fallback("模型引用结构不合法", "incomplete")

    verified = []
    errors = []
    for cite in raw_citations:
        if not isinstance(cite, dict):
            errors.append("citation is not an object")
            continue
        doc_id = cite.get("doc_id")
        para_id = cite.get("paragraph_id")
        quote = cite.get("quote")
        date = cite.get("version_date")
        document = read_documents.get(doc_id)
        paragraph = next(
            (p for p in document.get("paragraphs", [])
             if p.get("paragraph_id") == para_id), None
        ) if document else None
        if (
            not document or not paragraph or
            not isinstance(quote, str) or not quote.strip() or
            quote not in paragraph.get("text", "") or
            date != document.get("version_date")
        ):
            errors.append(f"invalid or unread evidence: {doc_id}/{para_id}")
            continue
        normalized = {
            "doc_id": doc_id,
            "paragraph_id": para_id,
            "version_date": date,
            "quote": quote
        }
        if normalized not in verified:
            verified.append(normalized)

    if errors:
        return {
            **_fallback("引用校验不通过，禁止将回答标为已完成", "incomplete"),
            "validation_errors": errors
        }

    missing = candidate.get("missing_information", [])
    if not isinstance(missing, list):
        missing = []
    missing = [str(x)[:300] for x in missing[:12]]
    conflicts = candidate.get("conflicts", [])
    if not isinstance(conflicts, list):
        conflicts = []
    conflicts = [
        {
            "topic": str(c.get("topic", ""))[:100],
            "doc_ids": c.get("doc_ids", []) if isinstance(c.get("doc_ids", []), list) else [],
            "reason": str(c.get("reason", ""))[:500]
        }
        for c in conflicts[:10] if isinstance(c, dict)
    ]
    follow_up = candidate.get("follow_up_question")
    if follow_up is not None and not isinstance(follow_up, str):
        follow_up = None

    # Conservative factual consistency check: numeric claims with business units
    # must appear in cited original text. This is NOT full semantic entailment.
    if status in {"completed", "needs_clarification"} and verified:
        pattern = r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*(?:个)?\s*(自然日|工作日|天|日|小时|分钟|元|%)"
        stated_values = set(re.findall(pattern, answer))
        evidence_values = set(re.findall(pattern, " ".join(c["quote"] for c in verified)))
        unsupported = sorted(stated_values - evidence_values)
        if unsupported:
            return {
                **_fallback("回答包含原文引用中无法支持的数值，不能标记完成", "incomplete"),
                "validation_errors": ["unsupported_numeric_claims:" + ",".join(number + unit for number, unit in unsupported)]
            }

    # For a self-contained question with a strong lexical match, a citation
    # from a wholly unrelated source is not enough merely because its quote
    # exists and its numbers happen to match. Do not force this heuristic
    # for elliptical follow-ups; their subject may be in conversation history.
    if status == "completed" and verified and question:
        independently_ranked = kb.search(question, limit=10)
        if independently_ranked and independently_ranked[0]["score"] >= 8:
            topical_docs = {doc["doc_id"] for doc in independently_ranked}
            cited_docs = {cite["doc_id"] for cite in verified}
            if not (topical_docs & cited_docs):
                return {
                    **_fallback(
                        "回答引用虽然真实，但与当前问题的直接检索主题不匹配，需补读相关文档",
                        "incomplete"
                    ),
                    "validation_errors": ["citation_topic_mismatch"]
                }

    # These checks are *limited*, source-driven business assertion checks;
    # they do not claim to prove general natural-language entailment.
    if status == "completed" and verified:
        explicit_issues = check_explicit_claims(
            answer, [cite["quote"] for cite in verified]
        )
        if explicit_issues:
            return {
                **_fallback(
                    "最终回答包含与已引用原文相反或未获引文支持的明确事实，请核对后重新回答",
                    "incomplete"
                ),
                "validation_errors": explicit_issues
            }

    # Keep conflicts tied to the CURRENT source text, not stale fact_value.
    # This is a guardrail, NOT a branch that chooses what tool to use next.
    known_conflicts = kb.conflicts_among(searched_ids, question=question)

    # After a mutation, D01 and D02 may AGREE: an outdated model response
    # must not continue claiming a conflict merely because two docs exist.
    agreed_groups = kb.agreeing_groups_among(searched_ids, question=question)
    for agreed in agreed_groups:
        group_ids = set(agreed["doc_ids"])
        candidate_claims_group_conflict = any(
            group_ids.issubset(set(c["doc_ids"]))
            for c in conflicts
        )
        if candidate_claims_group_conflict or status == "needs_clarification":
            return {
                **_fallback(
                    "相关制度当前原文对该期限规定一致，不能仅凭旧版冲突标签或版本不同宣称冲突；请基于最新原文重新判断",
                    "incomplete"
                ),
                "validation_errors": [
                    "spurious_conflict:" + ",".join(sorted(group_ids))
                ]
            }
    if known_conflicts:
        conflict_docs = {d for c in known_conflicts for d in c["doc_ids"]}
        cited_docs = {c["doc_id"] for c in verified}
        if (
            not conflict_docs.issubset(read_documents)
            or not conflict_docs.issubset(cited_docs)
        ):
            return {
                "status": "incomplete",
                "answer": "发现可能存在冲突的制度，但尚未完成双方原文核查及引用，不能确定统一答案。",
                "citations": verified,
                "conflicts": known_conflicts,
                "missing_information": ["需要读取并引用所有冲突文档的原文及适用范围"],
                "follow_up_question": "请明确适用的制度范围或优先级。",
                "validation_errors": ["conflicting documents not all read and cited"]
            }
        if status != "needs_clarification":
            return {
                "status": "needs_clarification",
                "answer": "文档存在未解决的不同规定，不能任意选择一个期限。请先核对适用范围和制度优先级。",
                "citations": verified,
                "conflicts": known_conflicts,
                "missing_information": ["缺少适用范围或明确的制度优先级"],
                "follow_up_question": "当前应适用哪份制度？"
            }
        if not conflicts:
            conflicts = known_conflicts

    if status == "completed":
        if successful_searches == 0:
            return _fallback("缺少真实检索步骤，不能认为任务已经完成", "incomplete")
        if not verified:
            return _fallback("缺少经过原文核验的引用，不能认为回答已经完成", "incomplete")
        if missing:
            return _fallback("模型同时声明 completed 和 missing_information；若已完整回答当前问题，请移除无关或猜测性的缺口并将 missing_information 设为空列表；若缺口影响结论，请改为需澄清或证据不足状态", "incomplete")

    # Model can report uncertainty without citations, but cannot falsely claim completion.
    return {
        "status": status,
        "answer": answer,
        "citations": verified,
        "conflicts": conflicts,
        "missing_information": missing,
        "follow_up_question": follow_up[:500] if follow_up else None
    }


def _fallback(message: str, status: str) -> dict[str, Any]:
    return {
        "status": status, "answer": message, "citations": [],
        "conflicts": [], "missing_information": [message],
        "follow_up_question": None
    }
