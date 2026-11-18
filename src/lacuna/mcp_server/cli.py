"""`lacuna mcp serve` (stdio MCP server) and `lacuna mcp tools` (print the tool list)."""

from __future__ import annotations

import asyncio
import logging
import sys


def cmd_serve(args) -> None:
    # stdout belongs to the protocol: log to stderr only.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from lacuna.mcp_server.server import build_server

    server = build_server()
    state = server.lacuna_state
    state.startup()
    logging.getLogger("lacuna.mcp").info("serving over stdio, session %s", state.session)
    try:
        server.run("stdio")
    finally:
        state.close()


def cmd_tools(args) -> None:
    from lacuna.mcp_server.server import build_server

    server = build_server()
    for t in asyncio.run(server.list_tools()):
        first = (t.description or "").split(". ")[0].strip()
        print(f"{t.name}\n    {first}")
    for p in asyncio.run(server.list_prompts()):
        print(f"[prompt] {p.name}\n    {p.description}")


def register(sub) -> None:
    p = sub.add_parser("mcp", help="Lacuna MCP server (Mode A)")
    s = p.add_subparsers(dest="mcp_cmd", required=True)
    serve = s.add_parser("serve", help="run the MCP server on stdio")
    serve.set_defaults(fn=cmd_serve)
    tools = s.add_parser("tools", help="print the tool and prompt list")
    tools.set_defaults(fn=cmd_tools)
