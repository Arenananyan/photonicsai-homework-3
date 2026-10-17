import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from hw3.demo import DemoModel
from hw3.graph import Audit, build_graph
from hw3.transport import Hardware


class Integration(unittest.IsolatedAsyncioTestCase):
    async def test_real_mcp_status_bench_and_approved_mock_motion(self):
        async with Hardware("mock") as tools:
            for scenario in ("status", "bench", "move"):
                with tempfile.TemporaryDirectory() as tmp:
                    audit = Audit(Path(tmp) / "run")
                    model = DemoModel(scenario)
                    graph = build_graph(model, tools, InMemorySaver(), audit)
                    config = {
                        "configurable": {"thread_id": scenario},
                        "recursion_limit": 100,
                    }
                    state = await graph.ainvoke(
                        dict(
                            messages=[HumanMessage(content="Test " + scenario)],
                            count=0,
                            previews={},
                        ),
                        config,
                    )
                    if scenario == "move":
                        stop = graph.get_state(config).interrupts[0].value
                        self.assertEqual(stop["kind"], "motion_approval")
                        self.assertFalse(
                            any(
                                m.name == "execute_sequence"
                                for m in state["messages"]
                                if isinstance(m, ToolMessage)
                            )
                        )
                        state = await graph.ainvoke(
                            Command(
                                resume=dict(approved=True, call_id=stop["call_id"])
                            ),
                            config,
                        )
                    self.assertFalse(state.get("failed"))
                    results = [
                        m for m in state["messages"] if isinstance(m, ToolMessage)
                    ]
                    self.assertTrue(results)
                    if scenario == "move":
                        self.assertEqual(
                            [m.name for m in results],
                            [
                                "printer_status",
                                "preview_sequence",
                                "execute_sequence",
                                "printer_status",
                            ],
                        )
                        position = json.loads(results[-1].content)["state"]["toolhead"][
                            "position"
                        ][:3]
                        self.assertEqual(position, [110, 110, 100])
                    self.assertTrue(
                        any(
                            m.name == "orchestrator"
                            for m in state["messages"][
                                state["messages"].index(results[-1]) + 1 :
                            ]
                        )
                    )


class FakeHardware:
    mode = "mock"
    definitions = [
        dict(
            type="function",
            function=dict(name="preview_position", parameters={"type": "object"}),
        ),
        dict(
            type="function",
            function=dict(name="execute_sequence", parameters={"type": "object"}),
        ),
    ]

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    async def call(self, name, args):
        self.calls.append((name, args))
        if name == "preview_position":
            return dict(
                token="t",
                expires_in_seconds=120,
                simulated=True,
                start=[0, 0, 100],
                end=[10, 10, 100],
                targets=[[10, 10, 100]],
                speed_mm_s=10,
            )
        if self.fail:
            raise RuntimeError("uncertain network outcome")
        return dict(ok=True, simulated=True)


class Script:
    def __init__(self, unknown=False):
        self.unknown = unknown
        self.seen = []

    async def respond(self, role, messages, tools=None):
        self.seen.append((role, tools, messages[:]))
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if role == "orchestrator":
            return dict(
                next="finish" if results else "executor",
                message="Route based on evidence",
            )
        if not results:
            name, args = (
                "preview_position",
                dict(x=10.0, y=10.0, z=100.0, speed_mm_s=10.0),
            )
        elif results[-1].name == "preview_position":
            name, args = (
                "execute_sequence",
                dict(token="fake" if self.unknown else "t", operator_confirmed=True),
            )
        else:
            return AIMessage(content="The tool result was read.", name="executor")
        return AIMessage(
            content="",
            name="executor",
            tool_calls=[
                dict(id="call-" + name, name=name, args=args, type="tool_call")
            ],
        )


class Guards(unittest.IsolatedAsyncioTestCase):
    async def setup_flow(self, tmp, model=None, tools=None):
        model = model or Script()
        tools = tools or FakeHardware()
        audit = Audit(Path(tmp) / "run")
        graph = build_graph(model, tools, InMemorySaver(), audit)
        config = {"configurable": {"thread_id": "x"}, "recursion_limit": 100}
        await graph.ainvoke(
            dict(messages=[HumanMessage(content="Move")], count=0, previews={}), config
        )
        return graph, config, audit, model, tools

    async def test_refusal_wrong_id_and_expiry_do_not_dispatch(self):
        for mode in ("refuse", "wrong", "expired"):
            with tempfile.TemporaryDirectory() as tmp:
                graph, config, audit, model, tools = await self.setup_flow(tmp)
                value = graph.get_state(config).interrupts[0].value
                if mode == "expired":
                    state = graph.get_state(config).values
                    previews = state["previews"]
                    previews["t"]["deadline"] = 0
                    graph.update_state(config, dict(previews=previews))
                await graph.ainvoke(
                    Command(
                        resume=dict(
                            approved=mode != "refuse",
                            call_id="wrong" if mode == "wrong" else value["call_id"],
                        )
                    ),
                    config,
                )
                self.assertEqual(
                    [name for name, _ in tools.calls], ["preview_position"]
                )

    async def test_uncertain_dispatch_stops_no_retry_and_is_single_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph, config, audit, model, tools = await self.setup_flow(
                tmp, tools=FakeHardware(fail=True)
            )
            value = graph.get_state(config).interrupts[0].value
            state = await graph.ainvoke(
                Command(resume=dict(approved=True, call_id=value["call_id"])), config
            )
            self.assertTrue(state["failed"])
            self.assertFalse(graph.get_state(config).next)
            self.assertEqual(sum(n == "execute_sequence" for n, a in tools.calls), 1)
            with self.assertRaises(FileExistsError):
                audit.claim("t")
            self.assertEqual(state["previews"], {})
            result = [m for m in state["messages"] if isinstance(m, ToolMessage)][-1]
            self.assertTrue(json.loads(result.content)["uncertain"])

    async def test_unknown_token_rejected_and_only_executor_gets_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools = FakeHardware()
            model = Script(unknown=True)
            with self.assertRaisesRegex(ValueError, "Unknown"):
                await self.setup_flow(tmp, model, tools)
            self.assertEqual([n for n, a in tools.calls], ["preview_position"])
            for role, definitions, _ in model.seen:
                self.assertEqual(definitions is not None, role == "executor")

    async def test_clarification_resumes_same_dialogue(self):
        class Clarify:
            async def respond(self, role, messages, tools=None):
                return dict(
                    next="finish"
                    if len([m for m in messages if m.type == "human"]) > 1
                    else "human",
                    message="Which XYZ?",
                )

        with tempfile.TemporaryDirectory() as tmp:
            graph = build_graph(
                Clarify(), FakeHardware(), InMemorySaver(), Audit(Path(tmp) / "run")
            )
            config = {"configurable": {"thread_id": "same"}}
            await graph.ainvoke(
                dict(messages=[HumanMessage(content="Go there")]), config
            )
            state = await graph.ainvoke(Command(resume="X10 Y10 Z100"), config)
            self.assertEqual(
                [m.content for m in state["messages"] if m.type == "human"],
                ["Go there", "X10 Y10 Z100"],
            )

    async def test_successful_motion_cannot_finish_without_new_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph, config, audit, model, tools = await self.setup_flow(tmp)
            value = graph.get_state(config).interrupts[0].value
            # This scripted executor tries to summarize immediately after motion.
            with self.assertRaisesRegex(ValueError, "fresh printer_status"):
                await graph.ainvoke(
                    Command(resume=dict(approved=True, call_id=value["call_id"])),
                    config,
                )
            self.assertEqual(
                sum(name == "execute_sequence" for name, args in tools.calls), 1
            )
            self.assertEqual(graph.get_state(config).values["previews"], {})


class AdditionalGuards(unittest.IsolatedAsyncioTestCase):
    async def test_unsafe_preview_stops_before_approval(self):
        class Unsafe(FakeHardware):
            async def call(self, name, args):
                result = await super().call(name, args)
                result["start"] = [0, 0, 5]
                return result

        with tempfile.TemporaryDirectory() as tmp:
            tools = Unsafe()
            graph = build_graph(
                Script(), tools, InMemorySaver(), Audit(Path(tmp) / "run")
            )
            config = {"configurable": {"thread_id": "unsafe"}}
            state = await graph.ainvoke(
                {"messages": [HumanMessage(content="Move")]}, config
            )
            self.assertTrue(state["failed"])
            self.assertFalse(graph.get_state(config).interrupts)
            self.assertEqual(state["previews"], {})
            self.assertEqual([name for name, _ in tools.calls], ["preview_position"])

    async def test_tool_loop_stops_at_budget(self):
        class Loop:
            async def respond(self, role, messages, tools=None):
                if role == "orchestrator":
                    return {"next": "executor", "message": "Read"}
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": str(len(messages)),
                            "name": "preview_position",
                            "args": {},
                            "type": "tool_call",
                        }
                    ],
                )

        with tempfile.TemporaryDirectory() as tmp:
            tools = FakeHardware()
            graph = build_graph(
                Loop(), tools, InMemorySaver(), Audit(Path(tmp) / "run")
            )
            state = await graph.ainvoke(
                {"messages": [HumanMessage(content="Read")]},
                {"configurable": {"thread_id": "loop"}, "recursion_limit": 100},
            )
            self.assertTrue(state["failed"])
            self.assertEqual(len(tools.calls), 24)


class ZeroStepRegression(unittest.IsolatedAsyncioTestCase):
    async def test_zero_final_z_is_removed_before_mcp_preview(self):
        class RedundantZ(DemoModel):
            async def respond(self, role, messages, tools=None):
                reply = await super().respond(role, messages, tools)
                if isinstance(reply, AIMessage) and reply.tool_calls:
                    call = reply.tool_calls[0]
                    if call["name"] == "preview_sequence":
                        call["args"]["steps"].append({"axis": "z", "distance_mm": 0})
                return reply

        with tempfile.TemporaryDirectory() as tmp:
            async with Hardware("mock") as tools:
                audit = Audit(Path(tmp) / "run")
                graph = build_graph(RedundantZ("move"), tools, InMemorySaver(), audit)
                config = {"configurable": {"thread_id": "zero"}}
                state = await graph.ainvoke(
                    {"messages": [HumanMessage(content="Move")]}, config
                )
                self.assertFalse(state.get("failed"))
                stop = graph.get_state(config).interrupts[0].value
                self.assertEqual(stop["kind"], "motion_approval")
                self.assertEqual(
                    stop["preview"]["targets"], [[100, 100, 100], [110, 110, 100]]
                )
                state = await graph.ainvoke(
                    Command(resume={"approved": True, "call_id": stop["call_id"]}),
                    config,
                )
                self.assertFalse(state.get("failed"))
                self.assertIn(
                    "preview_normalized", (audit.directory / "audit.jsonl").read_text()
                )
