import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, ToolMessage

from hw3.model import Model, dialogue


class RoleProtocol(unittest.IsolatedAsyncioTestCase):
    async def test_scientist_returns_json_text_without_tool_protocol(self):
        model = Model.__new__(Model)
        model.model = "test"
        create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"message":"Ask executor to read status then preview."}',
                            tool_calls=[],
                        )
                    )
                ]
            )
        )
        model.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        reply = await model.respond("scientist", [])
        self.assertFalse(reply.tool_calls)
        request = create.call_args.kwargs
        self.assertNotIn("tools", request)
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertIn("executor", reply.content)

    async def test_reading_roles_get_recorded_tool_evidence_as_text(self):
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "a",
                        "name": "printer_status",
                        "args": {},
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content='{"simulated":true}', tool_call_id="a", name="printer_status"
            ),
        ]
        recorded = dialogue(messages, native_tools=False)
        self.assertTrue(
            all(m["role"] != "tool" and "tool_calls" not in m for m in recorded)
        )
        self.assertIn("simulated", recorded[-1]["content"])
        self.assertEqual(dialogue(messages)[-1]["role"], "tool")


class CompleteProviderProtocol(unittest.IsolatedAsyncioTestCase):
    async def test_json_provider_through_real_mcp_and_human_pause(self):
        import json
        import tempfile
        from pathlib import Path

        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command

        from hw3.graph import Audit, build_graph
        from hw3.transport import Hardware

        requests = []

        async def provider(**kwargs):
            requests.append(kwargs)
            self.assertNotIn("tools", kwargs)
            context = json.loads(kwargs["messages"][1]["content"])
            role = (
                kwargs["messages"][0]["content"]
                .split("CURRENT ROLE: ")[1]
                .split("\n")[0]
            )
            if role == "orchestrator":
                next_role = (
                    "scientist"
                    if not context["latest_plan"]
                    else "finish"
                    if context["executor_answer"]
                    else "executor"
                )
                value = {
                    "next": next_role,
                    "message": "Continue from measured evidence",
                }
            elif role == "scientist":
                value = {
                    "message": "Read simulated status; lift, traverse, preview, approval, execute, read status."
                }
            elif context["last_tool"] is None:
                value = {"tool": "printer_status", "arguments": {}}
            elif (
                context["last_tool"] == "printer_status"
                and "execute_sequence" in context["tool_order"]
            ):
                value = {
                    "message": "Simulated measured final position: X110 Y110 Z100."
                }
            elif context["last_tool"] == "printer_status":
                value = {
                    "tool": "preview_sequence",
                    "arguments": {
                        "steps": [
                            {"axis": "z", "distance_mm": 95},
                            {"axis": "xy", "distance_mm": [10, 10]},
                        ],
                        "speed_mm_s": 10,
                        "z_speed_mm_s": 10,
                    },
                }
            elif context["last_tool"] == "preview_sequence":
                preview = next(
                    r["result"]
                    for r in context["latest_results"]
                    if r["tool"] == "preview_sequence"
                )
                value = {
                    "tool": "execute_sequence",
                    "arguments": {
                        "token": preview["token"],
                        "operator_confirmed": False,
                    },
                }
            else:
                value = {"tool": "printer_status", "arguments": {}}
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(value), tool_calls=[]
                        )
                    )
                ],
                usage=None,
            )

        model = Model.__new__(Model)
        model.model = "openai/gpt-oss-120b"
        model.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=provider))
        )
        with tempfile.TemporaryDirectory() as tmp:
            async with Hardware("mock") as hardware:
                graph = build_graph(
                    model, hardware, InMemorySaver(), Audit(Path(tmp) / "run")
                )
                config = {"configurable": {"thread_id": "provider"}}
                state = await graph.ainvoke(
                    {
                        "messages": [
                            HumanMessage(
                                content="Plan once then move X110 Y110 Z100 at 10mm/s and measure."
                            )
                        ]
                    },
                    config,
                )
                stop = graph.get_state(config).interrupts[0].value
                before = [
                    m.name for m in state["messages"] if isinstance(m, ToolMessage)
                ]
                self.assertEqual(before, ["printer_status", "preview_sequence"])
                state = await graph.ainvoke(
                    Command(resume={"approved": True, "call_id": stop["call_id"]}),
                    config,
                )
                self.assertFalse(state.get("failed"))
                self.assertEqual(
                    [m.name for m in state["messages"] if isinstance(m, ToolMessage)],
                    [
                        "printer_status",
                        "preview_sequence",
                        "execute_sequence",
                        "printer_status",
                    ],
                )
                self.assertEqual(len(requests), 9)
                self.assertTrue(all(len(r["messages"]) == 2 for r in requests))
                self.assertTrue(all(r["reasoning_effort"] == "low" for r in requests))
