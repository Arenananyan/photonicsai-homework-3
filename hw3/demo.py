"""Deterministic transport demonstration, not evidence of LLM understanding."""

import json
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage


class DemoModel:
    def __init__(self, scenario="status"):
        self.scenario = scenario

    async def respond(self, role, messages, tools=None):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if role == "orchestrator":
            if not any(m.name == "scientist" for m in messages):
                return dict(
                    next="scientist", message="Plan the requested instrument check."
                )
            if not results:
                return dict(
                    next="executor", message="Read measured evidence through MCP."
                )
            return dict(
                next="finish",
                message="The measured tool result is in the shared dialogue; scripted run finished.",
            )
        if role == "scientist":
            return AIMessage(
                content="Read status. For movement, preview and ask approval, then read status again.",
                name=role,
            )

        def call(name, args):
            return AIMessage(
                content="",
                name="executor",
                tool_calls=[
                    dict(id=uuid4().hex, name=name, args=args, type="tool_call")
                ],
            )

        if self.scenario == "bench":
            if not results:
                return call("measure_binary_pattern", dict(on_indices=list(range(256))))
        elif not results:
            return call("printer_status", {})
        elif self.scenario == "move":
            if len(results) == 1:
                x, y, z = json.loads(results[-1].content)["state"]["toolhead"][
                    "position"
                ][:3]
                steps = []
                if z != 100:
                    steps.append(dict(axis="z", distance_mm=100 - z))
                if (x, y) != (110, 110):
                    steps.append(dict(axis="xy", distance_mm=[110 - x, 110 - y]))
                return call(
                    "preview_sequence",
                    dict(steps=steps, speed_mm_s=10.0, z_speed_mm_s=10.0),
                )
            if results[-1].name == "preview_sequence":
                return call(
                    "execute_sequence",
                    dict(
                        token=json.loads(results[-1].content)["token"],
                        operator_confirmed=False,
                    ),
                )
            if (
                results[-1].name == "execute_sequence"
                and json.loads(results[-1].content).get("ok") is not False
            ):
                return call("printer_status", {})
        return AIMessage(
            content="Read the tool response above. This is a synthetic demo; no real instrument success is claimed.",
            name="executor",
        )
