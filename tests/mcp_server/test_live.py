"""Live stdio test: spawn `uv run lacuna mcp serve`, talk MCP, and prove stdout stays protocol-only."""

import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parents[2]
TOPIC = "t_d63691d8df"


def _services_up() -> bool:
    try:
        from lacuna.db import connect

        with connect() as c:
            return bool(c.execute("SELECT 1 FROM topics WHERE topic_id = %s", (TOPIC,)).fetchone())
    except Exception:
        return False


@pytest.mark.skipif(not _services_up(), reason="needs Postgres with topic " + TOPIC)
def test_stdio_roundtrip_and_clean_stdout():
    import anyio
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(command="uv", args=["run", "lacuna", "mcp", "serve"], cwd=str(ROOT))
    seen = {}

    async def run():
        with anyio.fail_after(240):
            async with Client(params) as client:
                tools = await client.list_tools()
                seen["tools"] = {t.name for t in tools.tools}
                ov = await client.call_tool("topic_overview", {"topic_id": TOPIC})
                seen["overview"] = ov.structured_content or json.loads(ov.content[0].text)
                ask = await client.call_tool("ask", {"topic_id": TOPIC, "question": "token pruning", "k": 3})
                seen["ask"] = ask.structured_content or json.loads(ask.content[0].text)
                ids = [e["chunk_id"] for e in seen["ask"]["evidence"]]
                v = await client.call_tool("validate_citations", {"chunk_ids": ids, "topic_id": TOPIC})
                seen["validate"] = v.structured_content or json.loads(v.content[0].text)

    anyio.run(run)
    assert "ask" in seen["tools"] and len(seen["tools"]) == 10
    assert seen["overview"].get("n_papers", 0) > 0, seen["overview"]
    assert seen["ask"]["evidence"], seen["ask"]
    assert seen["validate"]["ok"] is True
    # The SDK client parses every stdout line as JSON-RPC: a stray print or progress bar would have
    # raised above. Keep an explicit guard on the raw subprocess too.
    import subprocess

    p = subprocess.run(
        ["uv", "run", "lacuna", "mcp", "serve"], cwd=ROOT, input=b"", capture_output=True, timeout=120
    )
    assert p.stdout == b"", p.stdout[:200]
    assert sys.platform  # keep the import used
