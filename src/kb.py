"""Local synthetic enterprise knowledge base; deterministic, testable retrieval."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _chunks(text: str) -> set[str]:
    chunks: set[str] = set()
    for part in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", text.lower()):
        if re.search(r"[\u4e00-\u9fff]", part):
            chunks.update(part[i:i + 2] for i in range(len(part) - 1))
        elif len(part) >= 2:
            chunks.add(part)
    return chunks


def _matched_paragraph_ids(doc: dict[str, Any], query_chunks: set[str]) -> list[str]:
    """Return likely original paragraph IDs; no paragraph content is exposed by search."""
    ranked = []
    for paragraph in doc["paragraphs"]:
        overlap = len(query_chunks & _chunks(paragraph["text"]))
        if overlap >= 2:
            ranked.append((overlap, paragraph["paragraph_id"]))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [para_id for _, para_id in ranked]


_DEADLINE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?P<number>\d+(?:\.\d+)?)\s*(?:个)?\s*"
    r"(?P<unit>自然日|工作日|天|日)"
)


def _source_deadline(doc: dict[str, Any], topic: str) -> tuple[str, str] | None:
    """Read deadline facts from the CURRENT source text, never from fact_value.

    This small synthetic KB only has a curated reimbursement conflict_group.
    Unknown groups remain unsupported rather than generating fake equivalences.
    """
    if topic != "reimbursement_deadline":
        return None
    facts = set()
    for paragraph in doc["paragraphs"]:
        source = paragraph.get("text", "")
        if "报销" not in source:
            continue
        for match in _DEADLINE_PATTERN.finditer(source):
            context_before = source[max(0, match.start() - 45):match.start()]
            context_after = source[match.end():match.end() + 35]
            if (
                any(term in context_before for term in ("费用", "报销", "提交", "期限", "时限"))
                and any(term in context_after for term in ("内", "提交", "报销", "期限", "时限"))
            ):
                facts.add((match.group("number"), match.group("unit")))
    # Do not pick one arbitrarily if a source text itself is ambiguous.
    return next(iter(facts)) if len(facts) == 1 else None


class KnowledgeBase:
    def __init__(self, documents: list[dict[str, Any]]):
        if not isinstance(documents, list) or not documents:
            raise ValueError("Knowledge base must be a nonempty list of documents")
        self.documents = documents
        self.by_id: dict[str, dict[str, Any]] = {}
        for doc in documents:
            doc_id = doc.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id or doc_id in self.by_id:
                raise ValueError("Document IDs must be present and unique")
            if not isinstance(doc.get("paragraphs"), list):
                raise ValueError(f"Document {doc_id} has no paragraphs list")
            paragraph_ids = [p.get("paragraph_id") for p in doc["paragraphs"]]
            if not paragraph_ids or len(paragraph_ids) != len(set(paragraph_ids)):
                raise ValueError(f"Invalid paragraph IDs for {doc_id}")
            self.by_id[doc_id] = doc

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeBase":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(raw["documents"])

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Rank by lexical evidence, then suppress weak, off-topic hits.

        The relative cutoff is intentionally simple, transparent and testable.
        It is a small synthetic-corpus retriever, not production semantic search.
        """
        query = query.strip()
        query_chunks = _chunks(query)
        candidates = []
        for doc in self.documents:
            terms = doc.get("keywords", [])
            haystack = " ".join(
                [doc["title"], *terms, doc.get("summary", "")]
                + [p["text"] for p in doc["paragraphs"]]
            )
            overlap = len(query_chunks & _chunks(haystack))
            exact = sum(
                5 for term in terms
                if len(term) >= 2 and term.lower() in query.lower()
            )
            score = min(12, overlap) + exact
            if score < 4:
                continue
            candidates.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "version_date": doc["version_date"],
                "scope": doc.get("scope", "未注明"),
                "priority": doc.get("priority", "未注明"),
                "summary": doc.get("summary", ""),
                "score": score,
                "matched_paragraph_ids": _matched_paragraph_ids(doc, query_chunks),
            })
        if not candidates:
            return []
        best_score = max(d["score"] for d in candidates)
        cutoff = max(4, best_score * 0.40)
        found = [d for d in candidates if d["score"] >= cutoff]
        found.sort(key=lambda d: (-d["score"], d["doc_id"]))
        return found[:limit]

    def read(
        self, doc_id: str, paragraph_id: str | None = None
    ) -> dict[str, Any] | None:
        """Return a document or an exact evidence paragraph, never a search summary."""
        doc = self.by_id.get(doc_id)
        if doc is None:
            return None
        paragraphs = doc["paragraphs"]
        if paragraph_id is not None:
            paragraphs = [
                p for p in paragraphs if p.get("paragraph_id") == paragraph_id
            ]
            if not paragraphs:
                return None
        return {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "version_date": doc["version_date"],
            "scope": doc.get("scope", "未注明"),
            "priority": doc.get("priority", "未注明"),
            "paragraphs": [dict(p) for p in paragraphs],
        }

    def grouped_policy_facts(
        self, doc_ids: set[str], question: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """Collect relevant source-text policy values for transparent guardrails.

        A group labels a topic. It never supplies the fact value itself.
        """
        groups: dict[str, list[dict[str, Any]]] = {}
        for doc_id in sorted(doc_ids):
            doc = self.by_id.get(doc_id)
            if doc and doc.get("conflict_group"):
                groups.setdefault(doc["conflict_group"], []).append(doc)

        normalized_question = question.lower().strip() if question is not None else None
        relevant_groups = {}
        for group, docs in groups.items():
            if normalized_question is not None:
                topic_terms = {
                    term.lower()
                    for doc in docs
                    for term in doc.get("keywords", [])
                    if isinstance(term, str) and len(term) >= 2
                }
                if not any(term in normalized_question for term in topic_terms):
                    continue
            facts = {
                doc["doc_id"]: _source_deadline(doc, group) for doc in docs
            }
            relevant_groups[group] = {"facts": facts, "doc_ids": sorted(facts)}
        return relevant_groups

    def conflicts_among(
        self, doc_ids: set[str], question: str | None = None
    ) -> list[dict[str, Any]]:
        """Only contradictions in CURRENT, question-related paragraph text count."""
        conflicts = []
        for group, info in self.grouped_policy_facts(doc_ids, question).items():
            facts = info["facts"]
            # Only claim a confirmed conflict when both sides can be parsed.
            if len(facts) < 2 or any(value is None for value in facts.values()):
                continue
            if len(set(facts.values())) > 1:
                conflicts.append({
                    "topic": group,
                    "doc_ids": info["doc_ids"],
                    "reason": (
                        "同一问题相关的制度原文存在不同规定："
                        + "；".join(
                            f"{doc_id} 为 {value[0]}{value[1]}"
                            for doc_id, value in sorted(facts.items())
                        )
                        + "；暂无明确的适用范围或优先级依据"
                    )
                })
        return conflicts

    def agreeing_groups_among(
        self, doc_ids: set[str], question: str | None = None
    ) -> list[dict[str, Any]]:
        """Detect stale/hallucinated conflict reports after a text mutation."""
        agreed = []
        for group, info in self.grouped_policy_facts(doc_ids, question).items():
            facts = info["facts"]
            if (
                len(facts) >= 2
                and all(value is not None for value in facts.values())
                and len(set(facts.values())) == 1
            ):
                agreed.append({
                    "topic": group,
                    "doc_ids": info["doc_ids"],
                    "value": next(iter(facts.values()))
                })
        return agreed
