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

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Where we remember which Claude Code session belongs to each Hermes thread.
# Claude Code keeps the conversation inside its own session and writes the
# transcript to disk; the SDK can reload it with resume=<id>. Hermes only needs
# to remember the id per thread so a fresh agent (after a gateway restart or an
# agent-cache eviction) continues the same conversation instead of starting
# blank. Map shape: {hermes_session_id: {"claude_session_id", "cwd", "ts"}}.
_RESUME_STORE_NAME = "claude_runtime_sessions.json"


def _resume_store_path() -> Optional[str]:
    try:
        from hermes_constants import get_hermes_home
        return str(get_hermes_home() / _RESUME_STORE_NAME)
    except Exception:
        logger.debug("could not resolve hermes home for resume store", exc_info=True)
        return None


def _load_resume_map() -> Dict[str, Any]:
    path = _resume_store_path()
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.debug("could not read resume store %s", path, exc_info=True)
        return {}


def _lookup_resume_id(hermes_session_id: Optional[str], cwd: str) -> Optional[str]:
    """Return the saved Claude session id for this Hermes thread, but only if it
    was saved for the same cwd. The transcript is stored per project dir, so a
    resume id from another cwd would not load."""
    if not hermes_session_id:
        return None
    entry = _load_resume_map().get(hermes_session_id)
    if not isinstance(entry, dict):
        return None
    if entry.get("cwd") != cwd:
        return None
    sid = entry.get("claude_session_id")
    return sid or None


def _save_resume_id(
    hermes_session_id: Optional[str], claude_session_id: Optional[str], cwd: str
) -> None:
    if not hermes_session_id or not claude_session_id:
        return
    path = _resume_store_path()
    if not path:
        return
    try:
        data = _load_resume_map()
        data[hermes_session_id] = {
            "claude_session_id": claude_session_id,
            "cwd": cwd,
            "ts": int(time.time()),
        }
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
    except Exception:
        logger.debug("could not save resume id for %s", hermes_session_id, exc_info=True)


def _forget_resume_id(hermes_session_id: Optional[str]) -> None:
    """Drop a saved id, used when a resume failed (the transcript is gone)."""
    if not hermes_session_id:
        return
    path = _resume_store_path()
    if not path:
        return
    try:
        data = _load_resume_map()
        if hermes_session_id in data:
            del data[hermes_session_id]
            tmp = f"{path}.tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp, path)
    except Exception:
        logger.debug("could not forget resume id for %s", hermes_session_id, exc_info=True)


# Names the claude CLI accepts for --model. Anything else (for example a stale
# nemotron default) is dropped so the CLI uses the user's own configured
# default model.
_CLAUDE_MODEL_ALIASES = {"sonnet", "opus", "haiku"}


# Claude Code's own scheduling tools. They make Claude Code routines or wake-ups,
# which are separate from Hermes cron. We turn them off so the model uses Hermes'
# cron instead of silently scheduling something that does nothing for the user.
# This is the bug we saw: asked about cron, the model reached for these.
_DISALLOWED_CLAUDE_TOOLS = (
    "CronCreate",
    "CronDelete",
    "CronList",
    "RemoteTrigger",
    "ScheduleWakeup",
)


# Runtime note added on top of Hermes' own identity. It explains the one thing
# the model cannot work out on its own: that it is the engine behind Hermes,
# and how this runtime is wired. It is kept to principles, not a tool list, so
# it does not go stale when Hermes adds or renames tools (the real tool list is
# self-documenting through the hermes-tools MCP server).
_RUNTIME_BRIDGE_NOTE = """\
How this runtime works:
- You are Claude Code, running as the engine behind Hermes. The user talks to \
you through Hermes, usually from a chat app (Discord, Telegram, and so on), \
not from a terminal. Present yourself as Hermes, using the identity above.
- You keep all your own tools. You also have Hermes' own tools through an MCP \
server named "hermes-tools" (their names start with mcp__hermes-tools__). Use \
those for Hermes features.
- Hermes runs its own cron jobs through the hermes-tools cron tool. Use that \
to list, add, or change scheduled jobs. Do not use your own Cron, Routine, \
scheduled-task tools or scheduling skills (like /schedule): those make Claude \
Code routines, which are separate from Hermes and will not do what the user \
wants.
- Hermes memory is available here, through the hermes-tools memory tool. The \
memory shown above is what Hermes already knows. When you learn a durable fact \
(a user preference, a detail about their setup, a project convention), save it \
with that tool. Edit or merge against what is already there instead of adding \
duplicates. Do not use Claude Code's own memory or CLAUDE.md files for this; \
they are separate from Hermes memory.
- Some Hermes features are still not available on this runtime, because they \
need the live Hermes process: delegate_task and session search. You also cannot \
send messages to other chats from here; your final reply is delivered to the \
current chat automatically. If the user asks for one of these, say plainly that \
it is not available on this runtime, instead of guessing or using a Claude Code \
feature as a stand-in."""


# Appended to the turn prompt every _memory_nudge_interval turns (Hermes counts
# the turns and sets should_review_memory). The default runtime spawns a
# background review here, but that path needs an API-backed model, which a
# subscription Claude login does not have. So instead we ask the model, which is
# already running the turn, to review and save in band. Cheap: a few tokens, no
# extra round trip. It is only added to the prompt sent to the model, not to the
# stored transcript.
_MEMORY_REVIEW_NUDGE = (
    "[Memory check: if anything durable came up recently (a user preference, a "
    "fact about their setup, a project convention), save or update it now with "
    "the hermes-tools memory tool, merging against what is already saved. Skip "
    "if nothing is worth keeping.]"
)


def _memory_context_block(agent) -> str:
    """Load Hermes memory (MEMORY.md + USER.md) and render it for the system
    prompt, the same frozen-snapshot blocks the default runtime injects.

    Without this the Claude session cannot see what Hermes already knows, so it
    would answer blind and (once it can write memory) risk saving duplicates.
    Reuses the store the agent already loaded at init, and falls back to building
    one from config so the read-in still works if that store is absent.
    """
    store = getattr(agent, "_memory_store", None)
    if store is None:
        try:
            from hermes_cli.config import load_config

            mem_cfg = (load_config() or {}).get("memory") or {}
            if mem_cfg.get("memory_enabled") or mem_cfg.get("user_profile_enabled"):
                from tools.memory_tool import MemoryStore

                store = MemoryStore(
                    memory_char_limit=mem_cfg.get("memory_char_limit", 2200),
                    user_char_limit=mem_cfg.get("user_char_limit", 1375),
                )
                store.load_from_disk()
        except Exception:
            logger.debug("could not load Hermes memory for read-in", exc_info=True)
            store = None
    if store is None:
        return ""

    blocks: list[str] = []
    for target in ("user", "memory"):
        try:
            block = store.format_for_system_prompt(target)
        except Exception:
            block = None
        if block:
            blocks.append(block)
    return "\n\n".join(blocks)


def _build_runtime_context(agent) -> str:
    """Build the text added to Claude Code's system prompt so the session knows
    it is running as Hermes.

    Reuses Hermes' own identity (SOUL.md or the default identity), Hermes memory,
    and the platform hint so this stays in sync with Hermes. Only the runtime
    note is written here, because it describes this runtime, which Hermes itself
    does not.
    """
    parts: list[str] = []

    # 1. Hermes identity. Same source the normal runtime uses.
    soul = None
    try:
        import run_agent

        soul = run_agent.load_soul_md()
    except Exception:
        soul = None
    if soul:
        parts.append(soul)
    else:
        try:
            from agent.prompt_builder import DEFAULT_AGENT_IDENTITY

            parts.append(DEFAULT_AGENT_IDENTITY)
        except Exception:
            pass

    # 2. Platform hint (how the user reaches the agent). Reused from Hermes.
    platform_key = (getattr(agent, "platform", "") or "").lower().strip()
    if platform_key:
        try:
            from agent.prompt_builder import PLATFORM_HINTS

            if platform_key in PLATFORM_HINTS:
                parts.append(PLATFORM_HINTS[platform_key])
        except Exception:
            pass

    # 3. Hermes memory (read side). Same frozen snapshot the default runtime
    # shows the model, so the session knows what Hermes already remembers.
    mem_block = _memory_context_block(agent)
    if mem_block:
        parts.append(mem_block)

    # 4. Runtime note. Written here because it is specific to this runtime.
    parts.append(_RUNTIME_BRIDGE_NOTE)

    return "\n\n".join(p for p in parts if p)


# Session env keys that carry the chat origin. We read them once when the
# session is created (a Hermes session maps to one chat, so they are stable
# for its lifetime) and pass them to the MCP subprocess, so cron jobs created
# from chat post their results back to the right place.
_ORIGIN_ENV_KEYS = (
    "HERMES_SESSION_PLATFORM",
    "HERMES_SESSION_CHAT_ID",
    "HERMES_SESSION_CHAT_NAME",
    "HERMES_SESSION_THREAD_ID",
)


def _build_mcp_env_extra() -> dict[str, str]:
    """Extra env for the hermes-tools MCP subprocess.

    Sets HERMES_GATEWAY_SESSION so the cron tool is available (the cron toolset
    needs a gateway or interactive flag to turn on), and forwards the chat
    origin so cron jobs deliver back to the originating chat. Origin is read
    through gateway.session_context, which falls back to os.environ outside a
    gateway, so this is safe in CLI and cron contexts too."""
    extra: dict[str, str] = {"HERMES_GATEWAY_SESSION": "1"}
    try:
        from gateway.session_context import get_session_env

        for key in _ORIGIN_ENV_KEYS:
            value = get_session_env(key, "")
            if value:
                extra[key] = str(value)
    except Exception:
        logger.debug("could not read session origin for cron", exc_info=True)
    return extra


def _is_truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "on"}


def _runtime_surface(config: dict) -> dict[str, Any]:
    """Decide how much of the user's personal Claude Code setup the session
    inherits.

    Default is a predictable surface: only the MCP servers Hermes passes
    (strict_mcp_config=True), and only project settings, so the user's global
    Claude Code skills (like /schedule) do not bleed in. This keeps behavior the
    same for every user instead of depending on whatever is in their ~/.claude.

    Set model.claude_runtime_inherit_user_config: true to inherit the user's own
    MCP servers and skills, for users who want that."""
    model_cfg = (config or {}).get("model") or {}
    if _is_truthy(model_cfg.get("claude_runtime_inherit_user_config")):
        return {"strict_mcp_config": False, "setting_sources": ["user", "project"]}
    return {"strict_mcp_config": True, "setting_sources": ["project"]}


def _hermes_mcp_servers_for_sdk(config: dict) -> dict[str, Any]:
    """Translate Hermes' own configured MCP servers (config.yaml mcp_servers)
    into the claude-agent-sdk format, so they work on this runtime the same way
    they work on the default runtime. This is the consistent way for a user to
    add an MCP server (like Gmail) that works on every runtime, instead of
    relying on it being in their personal ~/.claude. Mirrors what the codex
    runtime migrates into ~/.codex/config.toml. Skips servers with no command
    or url, and ones marked enabled: false."""
    servers = (config or {}).get("mcp_servers")
    if not isinstance(servers, dict):
        return {}
    out: dict[str, Any] = {}
    for name, cfg in servers.items():
        if not isinstance(cfg, dict) or cfg.get("enabled") is False:
            continue
        command = cfg.get("command")
        url = cfg.get("url")
        if command:
            entry: dict[str, Any] = {"type": "stdio", "command": str(command)}
            args = cfg.get("args") or []
            if args:
                entry["args"] = [str(a) for a in args]
            env = cfg.get("env") or {}
            if env:
                entry["env"] = {str(k): str(v) for k, v in env.items()}
            out[str(name)] = entry
        elif url:
            transport = "sse" if cfg.get("transport") == "sse" else "http"
            entry = {"type": transport, "url": str(url)}
            headers = cfg.get("headers") or {}
            if headers:
                entry["headers"] = {str(k): str(v) for k, v in headers.items()}
            out[str(name)] = entry
    return out


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


def _build_claude_session(agent, cwd: str, *, resume: Optional[str] = None):
    """Build a ClaudeAgentSession for this agent. Factored out so the turn can
    rebuild it without resume if a resume fails. resume continues a saved Claude
    session (see the resume-store helpers above)."""
    from agent.transports.claude_agent_session import ClaudeAgentSession

    # Approval callback: use Hermes' standard prompt flow if a CLI thread set
    # one. Gateway and cron contexts get the permission-mode default (see
    # ClaudeAgentSession._can_use_tool).
    try:
        from tools.terminal_tool import _get_approval_callback
        approval_callback = _get_approval_callback()
    except Exception:
        approval_callback = None
    try:
        from hermes_cli.config import load_config
        config = load_config()
    except Exception:
        config = {}
    surface = _runtime_surface(config)
    return ClaudeAgentSession(
        cwd=cwd,
        model=_claude_model_for(getattr(agent, "model", None)),
        approval_callback=approval_callback,
        system_prompt_append=_build_runtime_context(agent),
        disallowed_tools=list(_DISALLOWED_CLAUDE_TOOLS),
        mcp_env_extra=_build_mcp_env_extra(),
        extra_mcp_servers=_hermes_mcp_servers_for_sdk(config) or None,
        strict_mcp_config=surface["strict_mcp_config"],
        setting_sources=surface["setting_sources"],
        resume=resume,
    )


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
    # Lazy session: one ClaudeAgentSession per AIAgent instance. Spawned on the
    # first turn, reused across turns (Claude Code keeps the conversation state
    # inside its own session), retired on a stuck process or auth failure.
    #
    # A fresh agent (after a gateway restart or an agent-cache eviction) has no
    # live session, so we resume the Claude session saved for this Hermes thread.
    # That reloads the transcript from disk and keeps the conversation going.
    # resumed_id stays set only for a fresh resume, so the error path below can
    # retry without it if the saved transcript is gone.
    cwd = getattr(agent, "session_cwd", None) or os.getcwd()
    resumed_id: Optional[str] = None
    if not hasattr(agent, "_claude_session") or agent._claude_session is None:
        resumed_id = _lookup_resume_id(getattr(agent, "session_id", None), cwd)
        agent._claude_session = _build_claude_session(agent, cwd, resume=resumed_id)

    # NOTE: the user message is already added to messages by the standard
    # run_conversation() flow before the early return reaches us. Do not add it
    # again or it will be duplicated. (Same contract as codex_runtime.)

    # Periodic memory review: when Hermes' turn counter says it is due, ask the
    # model to save durable facts as part of this turn. Added to the prompt only,
    # not to the stored messages.
    turn_input = user_message
    if should_review_memory:
        turn_input = f"{user_message}\n\n{_MEMORY_REVIEW_NUDGE}"

    try:
        turn = agent._claude_session.run_turn(user_input=turn_input)
    except Exception as exc:
        logger.exception("claude agent turn failed")
        try:
            agent._claude_session.close()
        except Exception:
            pass
        agent._claude_session = None
        # If this was a resume, the saved transcript may be gone (deleted, or on
        # another machine). Forget the stale id and retry once with a fresh
        # Claude session so the user still gets an answer this turn.
        if resumed_id:
            logger.warning(
                "claude resume failed for session %s, retrying fresh",
                getattr(agent, "session_id", None),
            )
            _forget_resume_id(getattr(agent, "session_id", None))
            try:
                agent._claude_session = _build_claude_session(agent, cwd, resume=None)
                turn = agent._claude_session.run_turn(user_input=turn_input)
            except Exception as exc2:
                logger.exception("claude agent turn failed after resume retry")
                try:
                    agent._claude_session.close()
                except Exception:
                    pass
                agent._claude_session = None
                exc = exc2
                turn = None
        else:
            turn = None
        if turn is None:
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
        # Do not resume a session we just retired as broken; forget its id.
        _forget_resume_id(getattr(agent, "session_id", None))
    else:
        # Remember this Claude session id for this Hermes thread, so a fresh
        # agent later (after a restart or cache eviction) continues it.
        _save_resume_id(
            getattr(agent, "session_id", None),
            getattr(turn, "thread_id", None),
            cwd,
        )

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
    _ = should_review_skills

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
