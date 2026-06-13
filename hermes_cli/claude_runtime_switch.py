"""Shared logic for the /claude-runtime slash command.

Toggles `model.anthropic_runtime` between "auto" (Hermes' default runtime) and
"claude_agent" (hand turns to Claude Code via the claude-agent-sdk, logged in
with the user's own `claude` login, no API key).

This matches hermes_cli/codex_runtime_switch.py. It has one extra job the codex
switch does not: codex users already run provider "openai-codex", but a
Claude-subscription user may be on any provider (nous, openrouter, and so on).
So enabling stores the prior model.provider and model.default, then switches to
provider "anthropic". Disabling restores them.

Both CLI (cli.py) and gateway call into this module so the behavior is the same
on both. The real runtime selection happens in hermes_cli.runtime_provider's
claude_agent short-circuit, which reads the saved config value. This module
just saves the value and reports the result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


VALID_RUNTIMES = ("auto", "claude_agent")

# Config keys (under model:) that remember what to restore on disable.
_PRIOR_PROVIDER_KEY = "claude_agent_prior_provider"
_PRIOR_MODEL_KEY = "claude_agent_prior_model"

_DEFAULT_CLAUDE_MODEL = "sonnet"


@dataclass
class ClaudeRuntimeStatus:
    """Result of a /claude-runtime call. Callers render it however fits their
    surface (CLI uses Rich panels, gateway sends a text message)."""

    success: bool
    new_value: Optional[str] = None
    old_value: Optional[str] = None
    message: str = ""
    requires_new_session: bool = False
    claude_binary_ok: bool = True
    claude_version: Optional[str] = None


def parse_args(arg_string: str) -> tuple[Optional[str], list[str]]:
    """Parse the slash-command argument string. Returns (value, errors).

    No args         -> return current state (value=None)
    'auto' / 'claude_agent' / 'on' / 'off' -> return that value
    anything else   -> error
    """
    raw = (arg_string or "").strip().lower()
    if not raw:
        return None, []
    if raw in {"on", "claude", "claude_code", "claude-code", "enable"}:
        return "claude_agent", []
    if raw in {"off", "default", "disable", "hermes"}:
        return "auto", []
    if raw in VALID_RUNTIMES:
        return raw, []
    return None, [
        f"Unknown runtime {raw!r}. Use one of: auto, claude_agent, on, off"
    ]


def get_current_runtime(config: dict) -> str:
    """Read the current `model.anthropic_runtime` value from a config dict.
    Returns 'auto' for an unset, empty, or unknown value."""
    if not isinstance(config, dict):
        return "auto"
    model_cfg = config.get("model") or {}
    if not isinstance(model_cfg, dict):
        return "auto"
    value = str(model_cfg.get("anthropic_runtime") or "").strip().lower()
    if value in VALID_RUNTIMES:
        return value
    return "auto"


def _looks_like_claude_model(model: object) -> bool:
    name = str(model or "").strip().lower()
    return bool(name) and (
        name in {"sonnet", "opus", "haiku"} or name.startswith("claude")
    )


def set_runtime(config: dict, new_value: str) -> str:
    """Change the config dict in place to save the new runtime value, and
    switch or restore model.provider and model.default as needed. Returns the
    previous value for callers that want to report the change."""
    if new_value not in VALID_RUNTIMES:
        raise ValueError(
            f"invalid runtime {new_value!r}; must be one of {VALID_RUNTIMES}"
        )
    old = get_current_runtime(config)
    if not isinstance(config.get("model"), dict):
        config["model"] = {}
    model_cfg = config["model"]

    if new_value == "claude_agent" and old != "claude_agent":
        prior_provider = str(model_cfg.get("provider") or "").strip()
        if prior_provider and prior_provider.lower() != "anthropic":
            model_cfg[_PRIOR_PROVIDER_KEY] = prior_provider
        model_cfg["provider"] = "anthropic"
        if not _looks_like_claude_model(model_cfg.get("default")):
            prior_model = str(model_cfg.get("default") or "").strip()
            if prior_model:
                model_cfg[_PRIOR_MODEL_KEY] = prior_model
            model_cfg["default"] = _DEFAULT_CLAUDE_MODEL
    elif new_value == "auto" and old == "claude_agent":
        prior_provider = model_cfg.pop(_PRIOR_PROVIDER_KEY, None)
        if prior_provider:
            model_cfg["provider"] = prior_provider
        prior_model = model_cfg.pop(_PRIOR_MODEL_KEY, None)
        if prior_model:
            model_cfg["default"] = prior_model

    model_cfg["anthropic_runtime"] = new_value
    return old


def check_claude_binary_ok() -> tuple[bool, Optional[str]]:
    """Check that the Claude Code CLI is installed at an acceptable version.
    Returns (ok, version_or_message)."""
    try:
        from agent.transports.claude_agent_session import check_claude_cli

        return check_claude_cli()
    except Exception as exc:  # pragma: no cover
        return False, f"claude check failed: {exc}"


def apply(
    config: dict,
    new_value: Optional[str],
    *,
    persist_callback=None,
) -> ClaudeRuntimeStatus:
    """Top-level entry point used by both CLI and gateway handlers.

    Args:
        config: in-memory config dict (changed in place when new_value is set)
        new_value: the runtime to set; None means "show current state only"
        persist_callback: optional callable that takes the changed config dict
            and saves it to disk. Skipped when None (used by tests).

    Returns: ClaudeRuntimeStatus describing the result.
    """
    current = get_current_runtime(config)

    _binary_check: Optional[tuple[bool, Optional[str]]] = None

    def _check_binary_cached() -> tuple[bool, Optional[str]]:
        nonlocal _binary_check
        if _binary_check is None:
            _binary_check = check_claude_binary_ok()
        return _binary_check

    # Read-only call: just report state.
    if new_value is None:
        ok, ver = _check_binary_cached()
        msg = (
            f"anthropic_runtime: {current}\n"
            f"claude CLI: {'OK ' + ver if ok else 'not available, ' + (ver or 'install with `npm i -g @anthropic-ai/claude-code`')}"
        )
        return ClaudeRuntimeStatus(
            success=True,
            new_value=current,
            old_value=current,
            message=msg,
            claude_binary_ok=ok,
            claude_version=ver if ok else None,
        )

    # No change requested.
    if new_value == current:
        return ClaudeRuntimeStatus(
            success=True,
            new_value=current,
            old_value=current,
            message=f"anthropic_runtime already set to {current}",
        )

    # When switching on, check the claude CLI is installed before saving. An
    # opt-in toggle that fails silently on the first turn is bad, so block here
    # with a clear install hint.
    if new_value == "claude_agent":
        ok, ver_or_msg = _check_binary_cached()
        if not ok:
            return ClaudeRuntimeStatus(
                success=False,
                new_value=None,
                old_value=current,
                message=(
                    "Cannot enable claude_agent runtime: "
                    f"{ver_or_msg or 'claude CLI not available'}\n"
                    "Install with: npm i -g @anthropic-ai/claude-code"
                ),
                claude_binary_ok=False,
                claude_version=None,
            )

    set_runtime(config, new_value)
    if persist_callback is not None:
        try:
            persist_callback(config)
        except Exception as exc:
            logger.exception("failed to save anthropic_runtime change")
            return ClaudeRuntimeStatus(
                success=False,
                new_value=new_value,
                old_value=current,
                message=f"updated config in memory but save failed: {exc}",
            )

    msg_lines = [
        f"anthropic_runtime: {current} -> {new_value}",
    ]
    model_cfg = config.get("model") or {}
    if new_value == "claude_agent":
        ok, ver = _check_binary_cached()
        if ok:
            msg_lines.append(f"claude CLI: {ver}")
        msg_lines.append(
            f"provider: anthropic, model: {model_cfg.get('default', '?')}"
        )
        msg_lines.append(
            "Turns now run through Claude Code (terminal and file ops run "
            "inside Claude; Hermes tools are available through the "
            "hermes-tools MCP server, registered per session, nothing written "
            "to disk)."
        )
        msg_lines.append(
            "Auth is your own `claude` login. No Anthropic API key is used."
        )
        msg_lines.append(
            "  (delegate_task, memory, session_search, todo run only on the "
            "default Hermes runtime. They need the agent loop context.)"
        )
    else:
        restored_provider = model_cfg.get("provider", "?")
        restored_model = model_cfg.get("default", "?")
        msg_lines.append(
            f"Restored provider: {restored_provider}, model: {restored_model}"
        )
        msg_lines.append("Turns will use the default Hermes runtime.")
    msg_lines.append(
        "Takes effect on the next session. The current cached agent keeps the "
        "old runtime to preserve the prompt cache."
    )
    return ClaudeRuntimeStatus(
        success=True,
        new_value=new_value,
        old_value=current,
        message="\n".join(msg_lines),
        requires_new_session=True,
    )
