"""Session adapter for the claude_agent runtime.

Owns one Claude Code session per Hermes session, driven through the official
Python `claude-agent-sdk`. The SDK spawns and talks to the `claude` CLI under
the user's own Claude login, so no API key is used. This file matches
agent/transports/codex_app_server_session.py, with one big simplification:
the SDK already implements the wire client (subprocess spawn, stream framing,
interrupts), so there is no claude_agent version of codex_app_server.py.

Lifecycle:
    session = ClaudeAgentSession(cwd="/home/x/proj")
    session.ensure_started()                          # spawns claude + connects
    result = session.run_turn(user_input="hello")     # blocks until the turn ends
    # result.final_text          -> assistant text returned to caller
    # result.projected_messages  -> list of {role, content, ...} for messages list
    # result.tool_iterations     -> completed tool calls (skill nudge counter)
    # result.interrupted         -> True if an interrupt fired mid-turn
    session.close()                                   # tears down subprocess

Threading: the SDK is asyncio-native but AIAgent's loop is synchronous, so the
session owns its own event-loop thread and bridges with
run_coroutine_threadsafe. run_turn() is synchronous to the caller, like
CodexAppServerSession.run_turn(). The caller thread runs the watchdogs (turn
deadline, post-tool quiet timeout) and the interrupt check while the loop
thread reads the SDK stream.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ClaudeSDKError,
    ResultMessage,
)
from claude_agent_sdk.types import (
    PermissionResultAllow,
    PermissionResultDeny,
)

from agent.transports.claude_event_projector import ClaudeEventProjector

logger = logging.getLogger(__name__)


# Hermes' tools.terminal.security_mode maps to a Claude Code permission mode.
# This matches _HERMES_TO_CODEX_PERMISSION_PROFILE in
# codex_app_server_session.py:
#   auto              -> acceptEdits (file edits auto-approved; Bash and other
#                        gated tools go through can_use_tool, which calls
#                        Hermes' approval flow)
#   approval-required -> default (everything gated goes through can_use_tool)
#   unrestricted/yolo -> bypassPermissions (no prompts at all)
_HERMES_TO_CLAUDE_PERMISSION_MODE = {
    "auto": "acceptEdits",
    "approval-required": "default",
    "unrestricted": "bypassPermissions",
    "yolo": "bypassPermissions",
}

# Minimum Claude Code CLI version known to work with claude-agent-sdk 0.2.x.
_MIN_CLAUDE_VERSION = (2, 0, 0)


@dataclass
class TurnResult:
    """Result of one user/assistant/tool turn through Claude Code.

    The fields match codex_app_server_session.TurnResult, so
    agent/claude_runtime.py can mirror agent/codex_runtime.py closely."""

    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None
    turn_id: Optional[str] = None
    thread_id: Optional[str] = None  # the Claude Code session id
    token_usage_last: Optional[dict[str, Any]] = None
    total_cost_usd: Optional[float] = None
    # Tells the caller the claude subprocess is probably stuck, or its login
    # broke. The caller should retire the session so the next turn starts a
    # fresh one. Same contract as the codex adapter.
    should_retire: bool = False


# Substrings in SDK/CLI error text that point to a missing or expired Claude
# login. Kept short on purpose: only send the user to re-login when the signal
# is clear, otherwise show the original error as is. Matches
# _OAUTH_REFRESH_FAILURE_HINTS in the codex adapter.
_AUTH_FAILURE_HINTS = (
    "invalid api key",
    "api key not found",
    "please run /login",
    "please log in",
    "please login",
    "not logged in",
    "not authenticated",
    "unauthenticated",
    "unauthorized",
    "401",
    "authentication_error",
    "oauth token has expired",
    "token has expired",
    "expired_token",
    "credential",
    "re-authenticate",
)


def _classify_auth_failure(*parts: str) -> Optional[str]:
    """Return a re-login hint if any of the given strings look like a Claude
    Code login failure. Otherwise return None."""
    haystack = " ".join(p for p in parts if p).lower()
    if not haystack:
        return None
    for needle in _AUTH_FAILURE_HINTS:
        if needle in haystack:
            return (
                "Claude authentication failed. Your Claude Code login looks "
                "expired or missing. Run `claude` and use `/login` to log in "
                "again, then retry. (Or switch back to the default runtime "
                "with `/claude-runtime auto`.)"
            )
    return None


def check_claude_cli(claude_bin: str = "claude") -> tuple[bool, Optional[str]]:
    """Check that the Claude Code CLI is installed at an acceptable version.
    Returns (ok, version_or_message). This is the counterpart of
    codex_app_server.check_codex_binary."""
    path = shutil.which(claude_bin)
    if not path:
        # The SDK ships a bundled CLI and prefers it over PATH, so a missing
        # system install is fine if the bundle is there.
        try:
            import claude_agent_sdk

            bundled = (
                os.path.dirname(claude_agent_sdk.__file__)
                + "/_bundled/claude"
            )
            if os.path.exists(bundled):
                path = bundled
        except Exception:
            path = None
    if not path:
        return False, (
            f"`{claude_bin}` not found on PATH and no bundled CLI in "
            "claude-agent-sdk. Install with: npm i -g @anthropic-ai/claude-code"
        )
    try:
        proc = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return False, f"`{claude_bin} --version` failed: {exc}"
    raw = (proc.stdout or proc.stderr or "").strip()
    if proc.returncode != 0:
        return False, f"`{claude_bin} --version` exited {proc.returncode}: {raw}"
    version_token = raw.split()[0] if raw else ""
    try:
        parts = tuple(int(p) for p in version_token.split(".")[:3])
        if parts < _MIN_CLAUDE_VERSION:
            return False, (
                f"Claude Code {version_token} is older than the minimum "
                f"{'.'.join(str(p) for p in _MIN_CLAUDE_VERSION)}. "
                "Update with: npm i -g @anthropic-ai/claude-code"
            )
    except ValueError:
        pass  # Cannot parse the version string. Accept it and let the SDK decide.
    return True, raw or "unknown version"


def _build_hermes_tools_mcp_config() -> dict[str, Any]:
    """Stdio MCP config that gives the Claude Code subprocess Hermes' tools
    (web_search, browser_*, vision, skills, kanban_*, and so on).

    It reuses agent/transports/hermes_tools_mcp_server.py with no changes. It
    is the same server the codex runtime registers in ~/.codex/config.toml.
    Here the SDK passes it per session via ClaudeAgentOptions.mcp_servers, so
    nothing is written to disk. The env passthrough matches
    codex_runtime_plugin_migration._build_hermes_tools_mcp_entry()."""
    env: dict[str, str] = {
        "HERMES_QUIET": "1",
        "HERMES_REDACT_SECRETS": os.environ.get("HERMES_REDACT_SECRETS", "true"),
    }
    hermes_home = os.environ.get("HERMES_HOME") or ""
    if hermes_home:
        env["HERMES_HOME"] = hermes_home
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    # Kanban workers tell the kanban_* tools who they are through this env var
    # (the dispatcher sets it when it spawns the worker process).
    kanban_task = os.environ.get("HERMES_KANBAN_TASK")
    if kanban_task:
        env["HERMES_KANBAN_TASK"] = kanban_task
    return {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "agent.transports.hermes_tools_mcp_server"],
        "env": env,
    }


class ClaudeAgentSession:
    """One Claude Code session per Hermes session, owned by AIAgent.

    Not thread-safe. One caller drives it at a time, the same way AIAgent's
    run_conversation() loop works today (same contract as
    CodexAppServerSession)."""

    def __init__(
        self,
        *,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        claude_bin: Optional[str] = None,
        permission_mode: Optional[str] = None,
        approval_callback: Optional[Callable[..., str]] = None,
        on_event: Optional[Callable[[Any], None]] = None,
        expose_hermes_tools: bool = True,
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._model = model
        self._claude_bin = claude_bin
        self._permission_mode = (
            permission_mode
            or _HERMES_TO_CLAUDE_PERMISSION_MODE.get(
                os.environ.get("HERMES_TERMINAL_SECURITY_MODE", "auto"),
                "acceptEdits",
            )
        )
        self._approval_callback = approval_callback
        self._on_event = on_event  # Display hook, same slot as the codex adapter
        self._expose_hermes_tools = expose_hermes_tools

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._client: Optional[ClaudeSDKClient] = None
        self._session_id: Optional[str] = None
        self._interrupt_event = threading.Event()
        # Tools the user approved with "session" or "always" via the approval
        # callback. Bash entries store the exact command string; other tools
        # store the tool name (an "always allow Bash" rule would be too broad).
        # The codex version is acceptForSession, which codex tracks on its side.
        self._session_approved: set[str] = set()
        self._stderr_tail: list[str] = []
        self._closed = False

    # ---------- lifecycle ----------

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or not (
            self._loop_thread and self._loop_thread.is_alive()
        ):
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._loop.run_forever,
                name="claude-agent-session",
                daemon=True,
            )
            self._loop_thread.start()
        return self._loop

    def _call(self, coro, timeout: float):
        """Run a coroutine on the session's loop thread and block for the result."""
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout)

    def ensure_started(self) -> str:
        """Spawn the claude subprocess and connect. Returns the Claude Code
        session id once known (empty string before the first turn finishes).
        Safe to call again. Repeated calls reuse the live client."""
        if self._client is not None:
            return self._session_id or ""
        self._call(self._async_start(), timeout=60)
        return self._session_id or ""

    async def _async_start(self) -> None:
        mcp_servers: dict[str, Any] = {}
        if self._expose_hermes_tools:
            mcp_servers["hermes-tools"] = _build_hermes_tools_mcp_config()
        options = ClaudeAgentOptions(
            cwd=self._cwd,
            model=self._model,
            permission_mode=self._permission_mode,
            mcp_servers=mcp_servers,
            can_use_tool=self._can_use_tool,
            cli_path=self._claude_bin,
            stderr=self._collect_stderr,
        )
        client = ClaudeSDKClient(options=options)
        await client.connect()
        self._client = client
        logger.info(
            "claude agent session started: model=%s mode=%s cwd=%s",
            self._model or "(cli default)",
            self._permission_mode,
            self._cwd,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._client is not None and self._loop is not None:
            try:
                self._call(self._async_close(), timeout=15)
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        self._client = None
        self._session_id = None
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:  # pragma: no cover
                pass
            self._loop = None
            self._loop_thread = None

    async def _async_close(self) -> None:
        if self._client is not None:
            await self._client.disconnect()

    def __enter__(self) -> "ClaudeAgentSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- interrupt ----------

    def request_interrupt(self) -> None:
        """Safe to call again. Tells the active turn loop to interrupt the
        claude turn and stop. Same contract as CodexAppServerSession."""
        self._interrupt_event.set()

    # ---------- diagnostics ----------

    def _collect_stderr(self, line: str) -> None:
        self._stderr_tail.append(line)
        if len(self._stderr_tail) > 200:
            del self._stderr_tail[:100]

    def stderr_tail(self, n: int = 12) -> list[str]:
        return self._stderr_tail[-n:]

    def _format_error_with_stderr(self, prefix: str, exc: Any = "") -> str:
        exc_str = str(exc) if exc != "" and exc is not None else ""
        base = f"{prefix}: {exc_str}" if exc_str else prefix
        tail = self.stderr_tail()
        if not tail:
            return base
        joined = "\n".join(line.rstrip() for line in tail if line).strip()
        if not joined:
            return base
        try:
            from agent.redact import redact_sensitive_text

            joined = redact_sensitive_text(joined, force=True)
        except Exception:  # pragma: no cover - redaction best-effort
            pass
        return f"{base}\nclaude stderr (last {len(tail)} lines):\n{joined}"

    # ---------- approval bridge ----------

    async def _can_use_tool(self, tool_name: str, input_data: dict, context: Any):
        """SDK permission callback, wired to Hermes' approval flow.

        This is the claude version of the codex adapter's
        _handle_server_request. A gated tool call becomes a Hermes approval
        prompt when an interactive callback is set. With no callback (gateway
        or cron), fall back to the permission mode's intent: acceptEdits
        ("auto") allows, default ("approval-required") denies."""
        # Hermes' own tools were already approved when the user enabled the
        # runtime. Same reasoning as the codex adapter auto-accepting
        # hermes-tools MCP elicitations.
        if tool_name.startswith("mcp__hermes-tools__"):
            return PermissionResultAllow()

        approval_key = (
            str(input_data.get("command") or "")[:500]
            if tool_name == "Bash"
            else tool_name
        )
        if approval_key and approval_key in self._session_approved:
            return PermissionResultAllow()

        if self._approval_callback is None:
            if self._permission_mode == "default":
                return PermissionResultDeny(
                    message=(
                        f"{tool_name} needs approval, but there is no "
                        "interactive approval channel in this context."
                    )
                )
            return PermissionResultAllow()

        if tool_name == "Bash":
            command_label = str(input_data.get("command") or "")
            description = f"Claude wants to run a command in {self._cwd}"
            reason = input_data.get("description")
            if reason:
                description += f" - {reason}"
        else:
            try:
                args_preview = json.dumps(input_data, ensure_ascii=False)[:300]
            except (TypeError, ValueError):
                args_preview = repr(input_data)[:300]
            command_label = f"{tool_name}: {args_preview}"
            description = f"Claude wants to use the {tool_name} tool"

        loop = asyncio.get_running_loop()
        try:
            choice = await loop.run_in_executor(
                None,
                lambda: self._approval_callback(
                    command_label, description, allow_permanent=False
                ),
            )
        except Exception:
            logger.exception("approval_callback raised on %s request", tool_name)
            return PermissionResultDeny(message="approval prompt failed")

        if choice in {"session", "always"} and approval_key:
            self._session_approved.add(approval_key)
        if choice in {"once", "session", "always"}:
            return PermissionResultAllow()
        return PermissionResultDeny(message="User denied the request")

    # ---------- per-turn ----------

    def run_turn(
        self,
        user_input: Any,
        *,
        turn_timeout: float = 600.0,
        poll_interval: float = 0.25,
        post_tool_quiet_timeout: float = 90.0,
    ) -> TurnResult:
        """Send a user message and block until the turn ends, projecting SDK
        messages into Hermes' messages shape as they arrive.

        post_tool_quiet_timeout: if claude finishes a tool and then goes quiet
        for this many seconds with no more stream activity, stop early and mark
        the session for retirement. This is the same watchdog the codex adapter
        runs (openclaw beta.8's post-tool completion watchdog)."""
        result = TurnResult()
        try:
            self.ensure_started()
        except Exception as exc:
            hint = _classify_auth_failure(str(exc), "\n".join(self.stderr_tail(40)))
            result.error = hint or self._format_error_with_stderr(
                "claude agent startup failed", exc
            )
            result.should_retire = True
            return result
        assert self._client is not None and self._loop is not None
        result.thread_id = self._session_id

        self._interrupt_event.clear()
        # last_tool_completion is written by the loop thread and read by the
        # caller thread. A plain dict of floats is fine for a watchdog (worst
        # case it fires one poll late).
        state: dict[str, Any] = {"last_tool_completion": None}

        future = asyncio.run_coroutine_threadsafe(
            self._async_run_turn(user_input, result, state), self._loop
        )

        deadline = time.monotonic() + turn_timeout
        interrupt_sent = False
        interrupt_deadline: Optional[float] = None

        while True:
            try:
                future.result(timeout=poll_interval)
                break
            except concurrent.futures.TimeoutError:
                pass
            except Exception as exc:  # the coroutine itself sets result.error
                logger.debug("claude turn future raised: %s", exc, exc_info=True)
                break

            now = time.monotonic()

            if interrupt_sent:
                # Give the SDK a short window to stop after the interrupt. If
                # it does not, drop the future and retire the session.
                if interrupt_deadline is not None and now > interrupt_deadline:
                    future.cancel()
                    result.should_retire = True
                    if not result.error:
                        result.error = (
                            "claude did not stop within 15s of an interrupt; "
                            "retiring the session."
                        )
                    break
                continue

            if self._interrupt_event.is_set():
                result.interrupted = True
                interrupt_sent = True
                interrupt_deadline = now + 15
                self._issue_interrupt()
                continue

            last_tool = state.get("last_tool_completion")
            if (
                last_tool is not None
                and (now - last_tool) > post_tool_quiet_timeout
            ):
                result.interrupted = True
                result.error = (
                    f"claude went silent for {post_tool_quiet_timeout:.0f}s "
                    "after a tool result; retiring the session."
                )
                result.should_retire = True
                interrupt_sent = True
                interrupt_deadline = now + 15
                self._issue_interrupt()
                continue

            if now > deadline:
                result.interrupted = True
                if not result.error:
                    result.error = self._format_error_with_stderr(
                        f"turn timed out after {turn_timeout:.0f}s"
                    )
                result.should_retire = True
                interrupt_sent = True
                interrupt_deadline = now + 15
                self._issue_interrupt()
                continue

        if result.thread_id is None:
            result.thread_id = self._session_id
        return result

    def _issue_interrupt(self) -> None:
        if self._client is None or self._loop is None:
            return

        async def _interrupt() -> None:
            try:
                await self._client.interrupt()
            except Exception as exc:
                # "no active turn" and similar mean it is already done. Not fatal.
                logger.debug("claude interrupt non-fatal: %s", exc)

        asyncio.run_coroutine_threadsafe(_interrupt(), self._loop)

    async def _async_run_turn(
        self, user_input: Any, result: TurnResult, state: dict
    ) -> None:
        """The loop-thread half of run_turn: send the prompt, read the stream,
        and fill `result` in place. The caller thread reads it only after the
        future resolves or is dropped."""
        projector = ClaudeEventProjector()
        text = user_input if isinstance(user_input, str) else str(user_input)
        try:
            await self._client.query(text)
            async for message in self._client.receive_response():
                if self._on_event is not None:
                    try:
                        self._on_event(message)
                    except Exception:  # pragma: no cover - display callback
                        logger.debug("on_event callback raised", exc_info=True)

                projection = projector.project(message)
                if projection.messages:
                    result.projected_messages.extend(projection.messages)
                if projection.is_tool_iteration:
                    result.tool_iterations += 1
                    state["last_tool_completion"] = time.monotonic()
                elif projection.messages or projection.final_text is not None:
                    # Any non-tool activity means claude is still producing
                    # output, so clear the quiet timer.
                    state["last_tool_completion"] = None
                if projection.final_text is not None:
                    result.final_text = projection.final_text

                if isinstance(message, ResultMessage):
                    self._session_id = message.session_id or self._session_id
                    result.thread_id = self._session_id
                    result.turn_id = message.uuid
                    if isinstance(message.usage, dict):
                        result.token_usage_last = dict(message.usage)
                    if message.total_cost_usd is not None:
                        result.total_cost_usd = float(message.total_cost_usd)
                    if message.is_error and not result.error:
                        err_text = message.result or f"subtype={message.subtype}"
                        hint = _classify_auth_failure(
                            err_text, "\n".join(self.stderr_tail(40))
                        )
                        if hint is not None:
                            result.error = hint
                            result.should_retire = True
                        else:
                            result.error = self._format_error_with_stderr(
                                "claude turn ended in error", err_text
                            )
                    # Use the CLI's final result text when no assistant text
                    # block was projected (for example a zero-output edge case).
                    if not result.final_text and message.result and not message.is_error:
                        result.final_text = message.result
        except ClaudeSDKError as exc:
            hint = _classify_auth_failure(str(exc), "\n".join(self.stderr_tail(40)))
            result.error = hint or self._format_error_with_stderr(
                "claude agent turn failed", exc
            )
            result.should_retire = True
        except Exception as exc:
            result.error = self._format_error_with_stderr(
                "claude agent turn failed", exc
            )
            result.should_retire = True
