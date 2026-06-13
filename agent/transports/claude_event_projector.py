"""Turn Claude Agent SDK messages into Hermes' messages list.

This lets Hermes' memory and skill review keep working under the claude_agent
runtime. It converts the SDK's typed message stream (AssistantMessage,
UserMessage, ResultMessage) into the OpenAI-shaped
`{role, content, tool_calls, tool_call_id}` entries that `agent/curator.py`
already reads. It matches agent/transports/codex_event_projector.py, which
does the same job for the codex_app_server runtime.

The SDK stream for one turn looks like this:
  SystemMessage(init)            ignored
  AssistantMessage               text blocks, thinking blocks, tool_use blocks
  UserMessage(ToolResultBlock)   tool results for the prior tool_use blocks
  ... repeat per tool call ...
  ResultMessage                  end of turn, read by the session not here

Mapping:
  - TextBlock        -> {role: "assistant", content}, last one is final_text
  - ThinkingBlock    -> kept in the next assistant message's "reasoning" field
  - ToolUseBlock     -> assistant tool_call. id is the SDK's toolu_* id, which
                        is stable across replay (see AGENTS.md Pitfall #16)
  - ToolResultBlock  -> {role: "tool", tool_call_id, content}, ticks
                        tool_iterations

Each tool_use/tool_result pair keeps Hermes' message order rules the same way
the codex projector does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

# Cap tool-result content the same way the codex projector caps MCP results.
# Tool outputs can be large (file reads, page extracts), and on this runtime
# the messages list feeds memory review, not the model.
_TOOL_RESULT_MAX_CHARS = 4000


@dataclass
class ProjectionResult:
    """Output of projecting one SDK message.

    `messages` is a list because one AssistantMessage can produce several
    entries (a text message plus a tool_calls message). Empty list means the
    message is ignored (SystemMessage, ResultMessage, partial events)."""

    messages: list[dict] = field(default_factory=list)
    is_tool_iteration: bool = False
    final_text: Optional[str] = None  # Set when assistant text completes


def _flatten_tool_result_content(content: Any) -> str:
    """ToolResultBlock.content is str, list[dict], or None. Flatten to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif item.get("type") == "image":
                    parts.append("[image]")
                else:
                    try:
                        parts.append(json.dumps(item, ensure_ascii=False))
                    except (TypeError, ValueError):
                        parts.append(repr(item))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


class ClaudeEventProjector:
    """Projects SDK messages in arrival order, keeping some state.

    It holds the in-progress thinking content. The SDK sends thinking as
    blocks inside AssistantMessage, but Hermes keeps it on the assistant
    entry, the same way the codex projector handles reasoning items."""

    def __init__(self) -> None:
        self._pending_reasoning: list[str] = []

    def project(self, message: Any) -> ProjectionResult:
        if isinstance(message, AssistantMessage):
            return self._project_assistant(message)
        if isinstance(message, UserMessage):
            return self._project_user(message)
        # SystemMessage (init/status) and ResultMessage (end of turn, the
        # session reads usage/cost/session_id off it directly) do not produce
        # messages.
        if isinstance(message, (SystemMessage, ResultMessage)):
            return ProjectionResult()
        return ProjectionResult()

    # ---------- per-type projections ----------

    def _project_assistant(self, message: AssistantMessage) -> ProjectionResult:
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in message.content or []:
            if isinstance(block, TextBlock):
                if block.text:
                    text_parts.append(block.text)
            elif isinstance(block, ThinkingBlock):
                if block.thinking:
                    self._pending_reasoning.append(block.thinking)
            elif isinstance(block, ToolUseBlock):
                args = block.input if isinstance(block.input, dict) else {}
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(
                                args, ensure_ascii=False, sort_keys=True
                            ),
                        },
                    }
                )

        result = ProjectionResult()
        text = "\n".join(text_parts).strip()
        if text:
            msg: dict[str, Any] = {"role": "assistant", "content": text}
            if self._pending_reasoning:
                msg["reasoning"] = "\n".join(self._pending_reasoning)
                self._pending_reasoning = []
            result.messages.append(msg)
            result.final_text = text
        if tool_calls:
            call_msg: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            }
            if self._pending_reasoning:
                call_msg["reasoning"] = "\n".join(self._pending_reasoning)
                self._pending_reasoning = []
            result.messages.append(call_msg)
        return result

    def _project_user(self, message: UserMessage) -> ProjectionResult:
        """A UserMessage in the middle of a turn carries tool results. (The
        real user prompt is added to `messages` by run_conversation() before
        the runtime runs, so plain text here would be a duplicate. Skip it.)"""
        result = ProjectionResult()
        content = message.content
        if not isinstance(content, list):
            return result
        for block in content:
            if isinstance(block, ToolResultBlock):
                output = _flatten_tool_result_content(block.content)
                if block.is_error:
                    output = f"[error] {output}"
                result.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": output[:_TOOL_RESULT_MAX_CHARS],
                    }
                )
                result.is_tool_iteration = True
        return result
