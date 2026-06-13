"""Claude Agent runtime: hand turns to Claude Code via the official
claude-agent-sdk.

This matches the app-server path in agent/codex_runtime.py. Each function
takes the parent ``AIAgent`` as its first argument (``agent``). AIAgent keeps
a thin ``_run_claude_agent_turn`` forwarder for the conversation-loop dispatch.

* ``run_claude_agent_turn`` drives one turn through a ``ClaudeAgentSession``
  (used when api_mode == "claude_agent", which the user turns on with
  ``/claude-runtime`` while on provider anthropic).

Auth is the user's own Claude Code login (the `claude` CLI / `/login`). Hermes
never sees or stores Anthropic credentials on this runtime.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# Names the claude CLI accepts for --model. Anything else (for example a stale
# nemotron default) is dropped so the CLI uses the user's own configured
# default model.
_CLAUDE_MODEL_ALIASES = {"sonnet", "opus", "haiku"}


def _claude_model_for(model: Any) -> str | None:
    name = str(model or "").strip().lower()
    if not name:
        return None
    if name in _CLAUDE_MODEL_ALIASES or name.startswith("claude"):
        return name
    return None


def _coerce_usage_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(value), 0)
        except ValueError:
            return 0
    return 0


def _record_claude_agent_usage(agent, turn) -> dict[str, Any]:
    """Turn Claude Code token usage into Hermes accounting.

    The SDK's ResultMessage.usage has input_tokens, output_tokens,
    cache_read_input_tokens, and cache_creation_input_tokens. Unlike the codex
    app-server protocol, cache-write tokens are reported here.

    Cost: on a subscription login the turn is part of the user's plan, so
    cost_status is "included" with no dollar amount. The CLI also reports
    total_cost_usd, but at API list rates whatever the billing mode, so we
    only treat it as real cost when an ANTHROPIC_API_KEY is set (see below).
    """
    agent.session_api_calls += 1

    usage = getattr(turn, "token_usage_last", None)
    if not isinstance(usage, dict) or not usage:
        if agent._session_db and agent.session_id:
            try:
                if not agent._session_db_created:
                    agent._ensure_db_session()
                agent._session_db.update_token_counts(
                    agent.session_id,
                    model=agent.model,
                    api_call_count=1,
                )
            except Exception as exc:
                logger.debug(
                    "Claude agent api-call persistence failed (session=%s): %s",
                    agent.session_id, exc,
                )
        return {}

    from agent.usage_pricing import CanonicalUsage

    canonical_usage = CanonicalUsage(
        input_tokens=_coerce_usage_int(usage.get("input_tokens")),
        output_tokens=_coerce_usage_int(usage.get("output_tokens")),
        cache_read_tokens=_coerce_usage_int(usage.get("cache_read_input_tokens")),
        cache_write_tokens=_coerce_usage_int(usage.get("cache_creation_input_tokens")),
        reasoning_tokens=0,
        raw_usage=usage,
    )
    prompt_tokens = canonical_usage.prompt_tokens
    completion_tokens = canonical_usage.output_tokens
    total_tokens = canonical_usage.total_tokens
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_tokens": canonical_usage.input_tokens,
        "output_tokens": canonical_usage.output_tokens,
        "cache_read_tokens": canonical_usage.cache_read_tokens,
        "cache_write_tokens": canonical_usage.cache_write_tokens,
        "reasoning_tokens": 0,
    }

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        try:
            compressor.update_from_response(usage_dict)
        except Exception:
            logger.debug("claude agent usage update failed", exc_info=True)

    agent.session_prompt_tokens += prompt_tokens
    agent.session_completion_tokens += completion_tokens
    agent.session_total_tokens += total_tokens
    agent.session_input_tokens += canonical_usage.input_tokens
    agent.session_output_tokens += canonical_usage.output_tokens
    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens

    # The CLI reports total_cost_usd at API list rates no matter the billing
    # mode, so the number alone cannot tell subscription from API-key usage.
    # Rule of thumb: an ANTHROPIC_API_KEY in the environment means the CLI
    # bills against it (cost is real). Otherwise the turn rides the user's
    # subscription login and is part of their plan.
    reported_cost = getattr(turn, "total_cost_usd", None)
    if reported_cost and os.environ.get("ANTHROPIC_API_KEY"):
        agent.session_estimated_cost_usd += float(reported_cost)
        cost_status = "estimated"
        cost_source = "claude_agent_cli"
    else:
        reported_cost = None
        cost_status = "included"
        cost_source = "claude_agent_subscription"
    agent.session_cost_status = cost_status
    agent.session_cost_source = cost_source

    if agent._session_db and agent.session_id:
        try:
            if not agent._session_db_created:
                agent._ensure_db_session()
            agent._session_db.update_token_counts(
                agent.session_id,
                input_tokens=canonical_usage.input_tokens,
                output_tokens=canonical_usage.output_tokens,
                cache_read_tokens=canonical_usage.cache_read_tokens,
                cache_write_tokens=canonical_usage.cache_write_tokens,
                estimated_cost_usd=float(reported_cost) if reported_cost else None,
                cost_status=cost_status,
                cost_source=cost_source,
                billing_provider=agent.provider,
                billing_base_url=agent.base_url,
                billing_mode="subscription_included"
                if cost_status == "included" else None,
                model=agent.model,
                api_call_count=1,
            )
        except Exception as exc:
            logger.debug(
                "Claude agent token persistence failed (session=%s, tokens=%d): %s",
                agent.session_id, total_tokens, exc,
            )

    return {
        **usage_dict,
        "last_prompt_tokens": prompt_tokens,
        "estimated_cost_usd": float(reported_cost) if reported_cost else None,
        "cost_status": cost_status,
        "cost_source": cost_source,
    }


def run_claude_agent_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """Claude agent runtime path. Hands the whole turn to Claude Code via the
    claude-agent-sdk and projects its messages back into Hermes' messages list
    so memory review keeps working.

    Called from run_conversation() when agent.api_mode == "claude_agent".
    Returns the same dict shape as the chat_completions path.
    """
    from agent.transports.claude_agent_session import ClaudeAgentSession

    # Lazy session: one ClaudeAgentSession per AIAgent instance. Spawned on the
    # first turn, reused across turns (Claude Code keeps the conversation state
    # inside its own session), retired on a stuck process or auth failure.
    if not hasattr(agent, "_claude_session") or agent._claude_session is None:
        cwd = getattr(agent, "session_cwd", None) or os.getcwd()
        # Approval callback: use Hermes' standard prompt flow if a CLI thread
        # set one. Gateway and cron contexts get the permission-mode default
        # (see ClaudeAgentSession._can_use_tool).
        try:
            from tools.terminal_tool import _get_approval_callback
            approval_callback = _get_approval_callback()
        except Exception:
            approval_callback = None
        agent._claude_session = ClaudeAgentSession(
            cwd=cwd,
            model=_claude_model_for(getattr(agent, "model", None)),
            approval_callback=approval_callback,
        )

    # NOTE: the user message is already added to messages by the standard
    # run_conversation() flow before the early return reaches us. Do not add it
    # again or it will be duplicated. (Same contract as codex_runtime.)

    try:
        turn = agent._claude_session.run_turn(user_input=user_message)
    except Exception as exc:
        logger.exception("claude agent turn failed")
        try:
            agent._claude_session.close()
        except Exception:
            pass
        agent._claude_session = None
        return {
            "final_response": (
                f"Claude agent turn failed: {exc}. "
                f"Switch back to the default runtime with `/claude-runtime auto`."
            ),
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": str(exc),
        }

    # The turn says the subprocess is stuck or its login broke. Retire the
    # session so the next turn starts a fresh claude instead of reusing a
    # broken one. Same contract as the codex runtime.
    if getattr(turn, "should_retire", False):
        logger.warning(
            "claude agent session retired (turn error: %s)", turn.error
        )
        try:
            agent._claude_session.close()
        except Exception:
            pass
        agent._claude_session = None

    # Add the projected messages to the conversation. They are standard
    # {role, content, tool_calls, tool_call_id} entries for curator.py and the
    # sessions DB.
    if turn.projected_messages:
        messages.extend(turn.projected_messages)

    # _turns_since_memory and _user_turn_count are already incremented in the
    # run_conversation() pre-loop block. Only _iters_since_skill needs a bump
    # here (the chat_completions loop does it per tool call, and that loop is
    # skipped on this path).
    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )
    usage_result = _record_claude_agent_usage(agent, turn)
    api_calls = 1

    should_review_skills = False
    if (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in agent.valid_tool_names
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    # External memory provider sync. Skipped on interrupt or error so we do not
    # feed a partial transcript to memory.
    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)

    # Background review is skipped on this runtime. The review fork needs
    # agent-loop tools (memory, skill_manage) on an API-backed api_mode. The
    # codex runtime downgrades its fork to codex_responses (same credentials,
    # different transport). There is no API fallback for a subscription-only
    # Claude login, so the review triggers stay set for a later turn on a
    # default-runtime session instead of failing here.
    _ = should_review_memory, should_review_skills

    return {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": api_calls,
        "completed": not turn.interrupted and turn.error is None,
        "partial": turn.interrupted or turn.error is not None,
        "error": turn.error,
        "claude_session_id": turn.thread_id,
        **usage_result,
    }


__all__ = ["run_claude_agent_turn"]
