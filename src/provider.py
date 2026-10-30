"""DeepSeek OpenAI-compatible Chat Completions adapter and offline scripted adapter."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ModelReply:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class DeepSeekModel:
    """Non-thinking mode avoids reasoning_content roundtrip requirements."""
    mode = "live_deepseek"

    def __init__(self):
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("Set DEEPSEEK_API_KEY in .env or your environment")
        self.model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        self.client = OpenAI(
            api_key=key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "45")),
            max_retries=0,
        )

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            extra_body={"thinking": {"type": "disabled"}},
            max_tokens=1600,
        )
        message = response.choices[0].message
        calls = [
            ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments)
            for call in (message.tool_calls or [])
        ]
        u = response.usage
        usage = {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            "total_tokens": getattr(u, "total_tokens", 0) or 0
        }
        return ModelReply(content=message.content, tool_calls=calls, usage=usage)


class ScriptedModel:
    """OFFLINE ONLY. Never represents genuine model-driven decisions."""
    mode = "offline_scripted"

    def __init__(self, actions: list[dict]):
        self.actions = actions
        self.position = 0

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        if self.position >= len(self.actions):
            raise RuntimeError("Scripted offline plan exhausted")
        action = self.actions[self.position]
        self.position += 1
        if action["kind"] == "tool":
            return ModelReply(
                tool_calls=[ToolCall(
                    id=f"offline_call_{self.position}",
                    name=action["name"],
                    arguments=json.dumps(action["arguments"], ensure_ascii=False)
                )]
            )
        if action["kind"] == "final":
            return ModelReply(content=json.dumps(action["answer"], ensure_ascii=False))
        raise ValueError(f"Unsupported offline action: {action['kind']}")
