"""Conservative, source-driven checks for a few high-risk business claims.

This is NOT a general natural-language inference or semantic entailment model.
It catches checkable, explicit reversals (approval, actor, feature support)
using only the actual paragraphs returned by read_document.
"""
from __future__ import annotations

import re


_APPROVAL_NEGATIVE = re.compile(
    r"(?:无需|无须|不需要|不需|不用|不必|免于|不要求|可以不|可不)"
    r"[^。；，、\n]{0,10}审批"
)
_APPROVAL_POSITIVE = re.compile(
    r"(?:必须|需要|须|应|要|由|提交)"
    r"[^。；，、\n]{0,30}审批"
)
_APPROVAL_ACTOR = re.compile(
    r"由\s*(?P<actor>[\u4e00-\u9fffA-Za-z]{2,22}?)"
    r"\s*(?:负责)?审批"
)
_FEATURE_CLAIM = re.compile(
    r"(?P<neg>暂不|不|未|不能|无法)?支持"
    r"(?P<feature>[\u4e00-\u9fff]{2,18})"
)
# Normalize generic role qualifiers; "员工的直属主管" and "直属主管" are equivalent.
_ACTOR_PREFIX = re.compile(r"^(?:员工的|该员工的|本人的|该|其)")


def _actor_normalize(actor: str) -> str:
    text = actor.strip()
    return _ACTOR_PREFIX.sub("", text)


def _feature_facts(text: str) -> list[tuple[str, bool]]:
    values = []
    for match in _FEATURE_CLAIM.finditer(text):
        feature = match.group("feature").strip()
        # Punctuation is excluded, but feature modifiers are kept for matching.
        values.append((feature, bool(match.group("neg"))))
    return values


def _similar_feature(a: str, b: str) -> bool:
    return a in b or b in a


def check_explicit_claims(answer: str, quotes: list[str]) -> list[str]:
    """Return machine-readable inconsistency reasons.

    Designed to stop obvious false completions, not to certify all sentences.
    A claim with no matching source is rejected only for the supported patterns.
    """
    evidence = "\n".join(quotes)
    issues: list[str] = []

    # Example: "年假不需要审批" is not entailed by "须提交审批".
    if _APPROVAL_NEGATIVE.search(answer):
        if _APPROVAL_POSITIVE.search(evidence) and not _APPROVAL_NEGATIVE.search(evidence):
            issues.append("contradictory_approval_requirement")
    if _APPROVAL_POSITIVE.search(answer):
        if _APPROVAL_NEGATIVE.search(evidence) and not _APPROVAL_POSITIVE.search(evidence):
            issues.append("contradictory_approval_requirement")

    # Example: "由人力资源部审批" cannot cite "由直属主管审批".
    actors_in_answer = [_actor_normalize(m.group("actor"))
                        for m in _APPROVAL_ACTOR.finditer(answer)]
    actors_in_evidence = [_actor_normalize(m.group("actor"))
                          for m in _APPROVAL_ACTOR.finditer(evidence)]
    for actor in actors_in_answer:
        if not any(actor in original or original in actor
                   for original in actors_in_evidence):
            issues.append("unsupported_approval_actor")
            break

    # Example: "支持离线编辑" cannot cite "暂不支持离线编辑".
    claim_features = _feature_facts(answer)
    source_features = _feature_facts(evidence)
    for feature, polarity in claim_features:
        aligned = [
            source_polarity for source_feature, source_polarity in source_features
            if _similar_feature(feature, source_feature)
        ]
        if not aligned:
            issues.append("unsupported_feature_claim")
            break
        if polarity not in aligned:
            issues.append("contradictory_feature_support")
            break

    return list(dict.fromkeys(issues))
