"""Hermes-tools-as-MCP server for the codex_app_server runtime.

When the user runs `openai/*` turns through the codex app-server, codex
owns the loop and builds its own tool list. By default, that means
Hermes' richer tool surface — web search, browser automation,
delegate_task subagents, vision analysis, persistent memory, skills,
cross-session search, image generation, TTS — is unreachable.

This module exposes a curated subset of those Hermes tools to the
spawned codex subprocess via stdio MCP. Codex registers it as a normal
MCP server (per `~/.codex/config.toml [mcp_servers.hermes-tools]`) and
the user gets full Hermes capability inside a Codex turn.

Scope (what we expose):
  - web_search, web_extract              — Firecrawl, no codex equivalent
  - browser_navigate / _click / _type /  — Camofox/Browserbase automation
    _snapshot / _scroll / _back / _press /
    _get_images / _console / _vision
  - vision_analyze                       — image inspection by vision model
  - image_generate                       — image generation
  - skill_view, skills_list              — Hermes' skill library
  - text_to_speech                       — TTS
  - kanban_* (complete/block/comment/    — kanban worker + orchestrator
    heartbeat/show/list/create/            handoff (stateless: read env var,
    unblock/link)                          write ~/.hermes/kanban.db)

What we DO NOT expose:
  - terminal / shell                     — codex's own shell tool
  - read_file / write_file / patch       — codex's apply_patch + shell
  - search_files / process               — codex's shell
  - clarify                              — codex's own UX
  - delegate_task / memory /             — `_AGENT_LOOP_TOOLS` in Hermes
    session_search / todo                  (model_tools.py). They require
                                           the running AIAgent context to
                                           dispatch (mid-loop state), so a
                                           stateless MCP callback can't
                                           drive them. See the inline
                                           comment on EXPOSED_TOOLS below.

Run with: python -m agent.transports.hermes_tools_mcp_server
Spawned by: CodexAppServerSession.ensure_started() when the runtime is
            active and config opts in.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Map JSON Schema types to Python types so FastMCP can build a real input
# schema from Hermes' tool definitions. Anything not listed becomes Any.
_JSON_TO_PY: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _signature_from_schema(schema: dict) -> Optional[inspect.Signature]:
    """Build a function signature from a JSON Schema's properties.

    The dispatch closure is written as ``_dispatch(**kwargs)``, which makes
    FastMCP advertise a single ``kwargs`` parameter instead of the tool's real
    parameters. The model then wraps its arguments in a ``kwargs`` field and
    the tool never sees them. Setting ``__signature__`` with the real
    parameter names (and types) fixes this for every exposed tool. Returns
    None when the schema has no usable properties, so the caller keeps the
    open ``**kwargs`` form.
    """
    props = (schema or {}).get("properties")
    if not isinstance(props, dict) or not props:
        return None
    required = set((schema or {}).get("required") or [])
    req_params: list[inspect.Parameter] = []
    opt_params: list[inspect.Parameter] = []
    for pname, pspec in props.items():
        if not isinstance(pname, str) or not pname.isidentifier():
            # A property we cannot turn into a clean parameter. Give up and
            # keep the open form rather than build a half-broken signature.
            return None
        py_type = _JSON_TO_PY.get((pspec or {}).get("type"), Any)
        if pname in required:
            req_params.append(
                inspect.Parameter(
                    pname,
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=py_type,
                )
            )
        else:
            opt_params.append(
                inspect.Parameter(
                    pname,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=(Optional[py_type] if py_type is not Any else Any),
                )
            )
    return inspect.Signature(req_params + opt_params)


# Tools we expose. Each name MUST match a registered Hermes tool that
# `model_tools.handle_function_call()` can dispatch.
#
# What we deliberately DO NOT expose:
#   - terminal / shell / read_file / write_file / patch / search_files /
#     process — codex's built-ins cover these and approval routes through
#     codex's own UI.
#   - delegate_task / memory / session_search / todo — these are
#     `_AGENT_LOOP_TOOLS` in Hermes (model_tools.py:493). They require
#     the running AIAgent context to dispatch (mid-loop state), so a
#     stateless MCP callback can't drive them. Hermes' default runtime
#     keeps these working; the codex_app_server runtime cannot.
EXPOSED_TOOLS: tuple[str, ...] = (
    "web_search",
    "web_extract",
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_press",
    "browser_snapshot",
    "browser_scroll",
    "browser_back",
    "browser_get_images",
    "browser_console",
    "browser_vision",
    "vision_analyze",
    "image_generate",
    "skill_view",
    "skills_list",
    "text_to_speech",
    # Kanban worker handoff tools — gated on HERMES_KANBAN_TASK env var
    # (set by the kanban dispatcher when spawning a worker). Without these
    # in the callback, a worker spawned with openai_runtime=codex_app_server
    # could do the work but couldn't report completion back to the kernel,
    # making it hang until timeout. Stateless dispatch — they just read
    # the env var and write to ~/.hermes/kanban.db.
    "kanban_complete",
    "kanban_block",
    "kanban_comment",
    "kanban_heartbeat",
    "kanban_show",
    "kanban_list",
    # NOTE: kanban_create / kanban_unblock / kanban_link are orchestrator-
    # only — the kanban tool gates them on HERMES_KANBAN_TASK being unset.
    # They're exposed here for orchestrator agents running on the codex
    # runtime that need to dispatch new tasks.
    "kanban_create",
    "kanban_unblock",
    "kanban_link",
    # Hermes cron. Claude Code has no Hermes cron of its own (only its own
    # routines, which are separate), so the model needs this to manage Hermes
    # scheduled jobs from chat. The tool only registers when the cron toolset's
    # requirement passes, which needs a gateway or interactive env flag. The
    # claude_agent runtime sets that flag in its MCP env; the codex runtime
    # does not, so this is a no-op there until they opt in the same way.
    "cronjob",
)


def _build_server() -> Any:
    """Create the FastMCP server with Hermes tools attached. Lazy imports
    so the module can be imported without the mcp package installed
    (we degrade to a clear error only when actually run)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - install hint
        raise ImportError(
            f"hermes-tools MCP server requires the 'mcp' package: {exc}"
        ) from exc

    # Discover Hermes tools so dispatch works.
    from model_tools import (
        get_tool_definitions,
        handle_function_call,
    )

    mcp = FastMCP(
        "hermes-tools",
        instructions=(
            "Hermes Agent's tool surface, exposed for use inside a Codex "
            "session. Use these for capabilities Codex's built-in toolset "
            "doesn't cover: web search/extract, browser automation, "
            "subagent delegation, vision, image generation, persistent "
            "memory, skills, and cross-session search."
        ),
    )

    # Pull authoritative Hermes tool schemas for the ones we expose, so
    # MCP clients see the same parameter docs Hermes gives the model.
    all_defs = {
        td["function"]["name"]: td["function"]
        for td in (get_tool_definitions(quiet_mode=True) or [])
        if isinstance(td, dict) and td.get("type") == "function"
    }

    exposed_count = 0

    for name in EXPOSED_TOOLS:
        spec = all_defs.get(name)
        if spec is None:
            logger.debug(
                "skipping %s — not registered in this Hermes process", name
            )
            continue

        description = spec.get("description") or f"Hermes {name} tool"
        params_schema = spec.get("parameters") or {"type": "object", "properties": {}}

        # FastMCP wants a Python callable. Build a closure that takes the
        # arguments dict, dispatches via handle_function_call, and returns
        # the result string. We set the closure's __signature__ from Hermes'
        # own parameter schema so FastMCP advertises the real parameters (not
        # a single opaque kwargs field). None values are dropped: they are
        # optional parameters the model left out.
        def _make_handler(tool_name: str, tool_schema: dict):
            def _dispatch(**kwargs: Any) -> str:
                args = {k: v for k, v in (kwargs or {}).items() if v is not None}
                try:
                    return handle_function_call(tool_name, args)
                except Exception as exc:
                    logger.exception("tool %s raised", tool_name)
                    return json.dumps({"error": str(exc), "tool": tool_name})
            _dispatch.__name__ = tool_name
            _dispatch.__doc__ = description
            sig = _signature_from_schema(tool_schema)
            if sig is not None:
                _dispatch.__signature__ = sig
            return _dispatch

        try:
            mcp.add_tool(
                _make_handler(name, params_schema),
                name=name,
                description=description,
            )
        except TypeError:
            # Older mcp SDK signature: fall back to decorator-style.
            handler = _make_handler(name, params_schema)
            handler = mcp.tool(name=name, description=description)(handler)

        exposed_count += 1

    logger.info(
        "hermes-tools MCP server registered %d/%d tools",
        exposed_count,
        len(EXPOSED_TOOLS),
    )
    return mcp


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for `python -m agent.transports.hermes_tools_mcp_server`."""
    argv = argv or sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv

    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        stream=sys.stderr,  # MCP uses stdio for protocol — logs MUST go to stderr
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Quiet mode: keep Hermes' own banners off stdout (which is the MCP wire).
    os.environ.setdefault("HERMES_QUIET", "1")
    os.environ.setdefault("HERMES_REDACT_SECRETS", "true")

    try:
        server = _build_server()
    except ImportError as exc:
        sys.stderr.write(f"hermes-tools MCP server cannot start: {exc}\n")
        return 2

    # FastMCP runs with stdio transport by default when launched as a
    # subprocess.
    try:
        server.run()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logger.exception("hermes-tools MCP server crashed")
        sys.stderr.write(f"hermes-tools MCP server error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
