"""Model-directed tool loop; deterministic code handles limits, IO and evidence safety."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

from .prompts import SYSTEM_PROMPT
from .tools import TOOL_SCHEMAS, ToolRegistry
from .validation import parse_final, validate_final


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class KnowledgeAgent:
    def __init__(
        self,
        model: Any,
        registry: ToolRegistry,
        max_steps: int = 8,
        max_tool_calls: int = 12
    ):
        if max_steps <= 0 or max_tool_calls <= 0:
            raise ValueError("max_steps and max_tool_calls must be positive")
        self.model = model
        self.registry = registry
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls

    def run(self, question: str, history: list[dict] | None = None) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question cannot be blank")
        if len(question) > 1000:
            raise ValueError("question must not exceed 1000 characters")
        history = history or []
        if not isinstance(history, list) or len(history) > 20:
            raise ValueError("history must be a list of at most 20 turns")
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in history:
            if not isinstance(item, dict) or item.get("role") not in {
                "user", "assistant"
            } or not isinstance(item.get("content"), str):
                raise ValueError("history entries must have user/assistant role and text")
            if len(item["content"]) > 1500:
                raise ValueError("history content exceeds 1500 chars")
            messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": question})

        trace: list[dict[str, Any]] = []
        searched_ids: set[str] = set()
        read_documents: dict[str, dict] = {}
        successful_searches = 0
        tool_calls = 0
        model_calls = 0
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        last_observations: list[str] = []
        validation_retries = 0

        def summarize_tool(name: str, value: dict) -> str:
            if not value.get("ok"):
                return f"{name} returned error: {value.get('error', {}).get('code', 'UNKNOWN')}"
            if name == "search_documents":
                ids = [d["doc_id"] for d in value.get("results", [])]
                return f"search found {value.get('count', 0)} candidate docs: {ids}"
            if name == "read_document":
                doc = value.get("document", {})
                return f"read {doc.get('doc_id')} ({len(doc.get('paragraphs', []))} paragraphs)"
            return f"{name}: ok"

        def result(status: str, answer: str, citations=None, conflicts=None,
                   missing_information=None, follow_up_question=None,
                   validation_errors=None) -> dict[str, Any]:
            data = {
                "mode": getattr(self.model, "mode", "unknown"),
                "model": getattr(self.model, "model_name", "scripted-offline"),
                "question": question,
                "status": status,
                "answer": answer,
                "citations": citations or [],
                "conflicts": conflicts or [],
                "missing_information": missing_information or [],
                "follow_up_question": follow_up_question,
                "metrics": {
                    "model_calls": model_calls,
                    "tool_calls": tool_calls,
                    "token_usage": token_usage
                },
                "trace": trace
            }
            if validation_errors:
                data["validation_errors"] = validation_errors
            return data

        for step in range(1, self.max_steps + 1):
            try:
                response = self.model.complete(messages, TOOL_SCHEMAS)
                model_calls += 1
            except Exception as exc:
                # Do not expose credentials or raw HTTP request details.
                kind = type(exc).__name__
                trace.append({
                    "timestamp": utc_now(), "event": "model_error",
                    "step": step, "error_type": kind
                })
                return result(
                    "tool_error", f"模型调用失败（{kind}），未完成问答。",
                    missing_information=["模型 API 不可用或调用异常"]
                )

            for key in token_usage:
                token_usage[key] += response.usage.get(key, 0)

            calls = response.tool_calls or []
            trace.append({
                "timestamp": utc_now(),
                "event": "model_decision",
                "step": step,
                "action": "tool_calls" if calls else "final",
                "tools": [c.name for c in calls],
                "observed_before_decision": list(last_observations),
                "public_basis_note": "可观察工具反馈供模型本轮决策参考，不是模型内部思维过程",
                "final_preview": (response.content or "")[:600] if not calls else None
            })

            if calls:
                # Include EXACT matching assistant tool_calls and tool messages.
                # Crucial for DeepSeek/OpenAI-compatible chat-completions protocol.
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": c.arguments
                            }
                        } for c in calls
                    ]
                })
                new_observations: list[str] = []
                for call in calls:
                    if tool_calls >= self.max_tool_calls:
                        trace.append({
                            "timestamp": utc_now(), "event": "limit_reached",
                            "limit": "max_tool_calls", "value": self.max_tool_calls
                        })
                        return result(
                            "incomplete",
                            "已达到最大工具调用次数，任务尚未完成。",
                            missing_information=["max_tool_calls_exceeded"]
                        )
                    tool_calls += 1
                    try:
                        args = json.loads(call.arguments)
                    except (ValueError, TypeError):
                        args = None
                    observation = self.registry.execute(call.name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(observation, ensure_ascii=False)
                    })
                    new_observations.append(summarize_tool(call.name, observation))
                    trace.append({
                        "timestamp": utc_now(), "event": "tool_result",
                        "step": step, "call_id": call.id,
                        "tool": call.name, "arguments": args,
                        "observation_summary": summarize_tool(call.name, observation),
                        "observation": observation
                    })
                    if observation.get("ok") and call.name == "search_documents":
                        successful_searches += 1
                        searched_ids.update(
                            d["doc_id"] for d in observation.get("results", [])
                        )
                    if observation.get("ok") and call.name == "read_document":
                        doc = observation["document"]
                        existing = read_documents.get(doc["doc_id"])
                        if existing and existing.get("version_date") == doc.get("version_date"):
                            # Multiple paragraph-scoped reads of the same document
                            # must retain ALL actually observed evidence paragraphs.
                            merged = {para["paragraph_id"]: para
                                      for para in existing["paragraphs"]}
                            merged.update({para["paragraph_id"]: para
                                           for para in doc["paragraphs"]})
                            read_documents[doc["doc_id"]] = {
                                **doc, "paragraphs": list(merged.values())
                            }
                        else:
                            read_documents[doc["doc_id"]] = doc
                last_observations = new_observations
                continue

            candidate = parse_final(response.content)
            checked = validate_final(
                candidate,
                self.registry.kb, searched_ids,
                read_documents, successful_searches,
                question=question
            )
            # Only in actual live mode: offer a bounded correction opportunity.
            # This is format/evidence feedback, NOT a hard-coded business-tool decision.
            needs_repair = (
                candidate is None
                or bool(checked.get("validation_errors"))
                or (
                    candidate.get("status") == "completed"
                    and checked["status"] == "incomplete"
                )
            )
            if (
                getattr(self.model, "mode", "") == "live_deepseek"
                and needs_repair and validation_retries < 2
                and step < self.max_steps
            ):
                validation_retries += 1
                reason = (
                    checked.get("validation_errors")
                    or checked.get("missing_information")
                    or ["final_format_invalid"]
                )
                feedback = (
                    "你上一轮最终回答未通过可验证性检查：" +
                    json.dumps(reason, ensure_ascii=False) +
                    "。请根据已经读取的证据纠正引文，或自行决定再调用"
                    "必要的工具补充证据。不得编造引用；仍无法解决时"
                    "请明确返回 incomplete/needs_clarification。最终格式必须是 JSON。"
                )
                messages.append({
                    "role": "assistant", "content": response.content or ""
                })
                messages.append({"role": "user", "content": feedback})
                trace.append({
                    "timestamp": utc_now(), "event": "validation_feedback",
                    "step": step, "issues": reason,
                    "next_action": "model_reconsideration"
                })
                last_observations = ["validation feedback: " + str(reason)[:200]]
                continue
            return result(**checked)

        trace.append({
            "timestamp": utc_now(), "event": "limit_reached",
            "limit": "max_steps", "value": self.max_steps
        })
        return result(
            "incomplete", "已达到最大模型决策轮数，任务尚未完成。",
            missing_information=["max_steps_exceeded"]
        )
