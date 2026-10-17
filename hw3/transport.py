"""Standalone copies of the existing connector/MCP; only executor gets discovery."""

import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "printer_status",
    "preview_sequence",
    "preview_position",
    "execute_sequence",
    "measure_binary_pattern",
}


class Hardware:
    def __init__(self, mode="mock"):
        if mode not in ("mock", "real"):
            raise ValueError("Invalid mode")
        self.mode = mode
        self.stack = AsyncExitStack()
        self.clients = {}
        self.definitions = []

    async def __aenter__(self):
        try:
            env = {
                "PRINTER_MODE": self.mode,
                "PRINTER_ALLOW_MOTION": os.getenv("PRINTER_ALLOW_MOTION", "0"),
                "FASTMCP_SHOW_SERVER_BANNER": "false",
                "FASTMCP_LOG_LEVEL": "ERROR",
            }
            for key in ("SSH_HOST", "SSH_CONTROL_PATH"):
                if key in os.environ:
                    env[key] = os.environ[key]
            for source, script in [
                ("printer", ROOT / "vendor/printer/fastmcp_server.py"),
                ("bench", ROOT / "bench_server.py"),
            ]:
                client = await self.stack.enter_async_context(
                    Client(
                        PythonStdioTransport(
                            script,
                            python_cmd=sys.executable,
                            cwd=str(script.parent),
                            env=env,
                        ),
                        timeout=120,
                    )
                )
                for tool in await client.list_tools():
                    if tool.name not in ALLOWED:
                        continue
                    self.clients[tool.name] = client
                    self.definitions.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description or tool.name,
                                "parameters": tool.input_schema,
                            },
                        }
                    )
            return self
        except BaseException:
            await self.stack.aclose()
            raise

    async def __aexit__(self, *args):
        await self.stack.aclose()

    async def call(self, name, arguments):
        if name not in self.clients:
            raise ValueError("Tool not exposed")
        reply = await self.clients[name].call_tool(name, arguments)
        data = reply.data
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        if not isinstance(data, dict):
            data = reply.structured_content
        if not isinstance(data, dict):
            raise ValueError("MCP did not return an object")
        if data.get("ok") is False:
            raise ValueError(str(data))
        return data
