"""Dynamic four-role graph, resumable human turn, explicit hardware turn."""

import json
import os
import time
from pathlib import Path
from typing import Annotated, TypedDict

from jsonschema import validate
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from .safety import validate_preview


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    next: str
    count: int
    pending: dict
    previews: dict
    approved: str
    failed: bool
    question: str
    tool_count: int


class Audit:
    def __init__(self, directory, live=False):
        self.live = live
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)

    def write(self, kind, **data):
        if self.live:
            if kind == "model_route":
                print(
                    f"\n[LangGraph → {data['proposal']['next']}] {data['proposal']['message']}",
                    flush=True,
                )
            elif kind == "model_answer":
                print(f"[{data['role']}] {data['message']}", flush=True)
            elif kind == "executor_proposal" and data["message"]:
                print(f"[executor] {data['message']}", flush=True)
            elif kind == "tool_request":
                print(
                    f"[MCP request] {data['tool']} {json.dumps(data['arguments'])}",
                    flush=True,
                )
            elif kind == "tool_result":
                result = data["result"]
                if data["tool"] == "printer_status" and "state" in result:
                    result = {
                        "simulated": result.get("simulated"),
                        "position": result["state"]["toolhead"]["position"][:3],
                        "ready": result["state"]["webhooks"],
                    }
                print(f"[MCP result] {data['tool']} {json.dumps(result)}", flush=True)
            elif kind == "human_approval":
                print(f"[Human → LangGraph] approved={data['approved']}", flush=True)
        with (self.directory / "audit.jsonl").open("a") as out:
            out.write(
                json.dumps(
                    dict(time=time.time(), kind=kind, **data), ensure_ascii=False
                )
                + "\n"
            )

    def claim(self, token):
        # Claim BEFORE dispatch. A crash can leave an uncertain outcome; never
        # repeat it automatically, including checkpoint/node replay.
        import hashlib

        path = self.directory / (
            "dispatch-" + hashlib.sha256(token.encode()).hexdigest() + ".claim"
        )
        with path.open("x") as out:
            out.write("One dispatch attempt; inspect actual hardware on failure.\n")


def build_graph(model, hardware, checkpointer, audit):
    builder = StateGraph(State)

    async def orchestrator(state):
        if state.get("count", 0) >= 16:
            return dict(
                next="finish",
                failed=True,
                messages=[
                    AIMessage(
                        content="Decision limit reached; task incomplete.",
                        name="system_status",
                    )
                ],
            )
        decision = await model.respond("orchestrator", state["messages"])
        if decision["next"] not in (
            "scientist",
            "general",
            "executor",
            "human",
            "finish",
        ):
            raise ValueError("Invalid next role")
        audit.write("model_route", proposal=decision)
        return dict(
            next=decision["next"],
            count=state.get("count", 0) + 1,
            question=decision["message"],
            messages=[AIMessage(content=decision["message"], name="orchestrator")],
        )

    def specialist(role):
        async def invoke(state):
            reply = await model.respond(role, state["messages"])
            if reply.tool_calls:
                raise ValueError("Specialists cannot hold hardware tools")
            audit.write("model_answer", role=role, message=reply.content)
            return dict(messages=[reply])

        return invoke

    async def executor(state):
        if state.get("tool_count", 0) >= 24:
            return dict(
                next="orchestrator",
                failed=True,
                messages=[
                    AIMessage(
                        content="Tool limit reached; task incomplete.",
                        name="system_status",
                    )
                ],
            )
        reply = await model.respond("executor", state["messages"], hardware.definitions)
        audit.write("executor_proposal", message=reply.content, calls=reply.tool_calls)
        results = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        needs_measurement = bool(
            results
            and results[-1].name == "execute_sequence"
            and json.loads(results[-1].content).get("ok") is not False
        )
        if needs_measurement and (
            len(reply.tool_calls) != 1
            or reply.tool_calls[0]["name"] != "printer_status"
        ):
            raise ValueError(
                "A fresh printer_status measurement is required after execution before any next action or conclusion"
            )
        if not reply.tool_calls:
            return dict(messages=[reply], next="orchestrator")
        if len(reply.tool_calls) != 1:
            raise ValueError("One hardware call at a time")
        call = reply.tool_calls[0]
        definitions = {
            t["function"]["name"]: t["function"]["parameters"]
            for t in hardware.definitions
        }
        if call["name"] not in definitions:
            raise ValueError("Unexposed hardware tool")
        validate(call["args"], definitions[call["name"]])
        if call["name"] == "execute_sequence":
            preview = state.get("previews", {}).get(call["args"]["token"])
            if not preview or time.time() > preview["deadline"]:
                raise ValueError("Unknown, consumed or expired preview")
            next_node = "approval"
        else:
            next_node = "hardware"
        return dict(messages=[reply], pending=call, approved="", next=next_node)

    def approval(state):
        call = state["pending"]
        token = call["args"]["token"]
        preview = state["previews"][token]
        reply = interrupt(
            dict(
                kind="motion_approval",
                call_id=call["id"],
                question="Review the full path and physical clearance. Approve this one dispatch?",
                preview=preview["result"],
            )
        )
        accepted = (
            isinstance(reply, dict)
            and reply.get("approved") is True
            and reply.get("call_id") == call["id"]
        )
        audit.write("human_approval", call_id=call["id"], approved=accepted)
        if accepted and time.time() <= preview["deadline"]:
            return dict(approved=call["id"], next="hardware")
        previews = dict(state["previews"])
        previews.pop(token, None)
        return dict(
            approved="",
            next="orchestrator",
            previews=previews,
            messages=[
                ToolMessage(
                    content=json.dumps(
                        dict(
                            ok=False,
                            error="Operator declined or preview expired; no movement.",
                        )
                    ),
                    tool_call_id=call["id"],
                    name=call["name"],
                ),
                HumanMessage(
                    content="Declined or expired approval for " + call["id"],
                    name="human",
                ),
            ],
        )

    async def hardware_turn(state):
        call = state["pending"]
        name = call["name"]
        args = dict(call["args"])
        previews = dict(state.get("previews", {}))
        audit.write("tool_request", tool=name, arguments=args, call_id=call["id"])
        try:
            if name == "preview_sequence":
                steps = args.get("steps", [])
                effective = [
                    step
                    for step in steps
                    if step.get("distance_mm") != 0
                    and step.get("distance_mm") != [0, 0]
                ]
                if not effective:
                    raise ValueError("The proposed path has no movement")
                if len(effective) != len(steps):
                    args["steps"] = effective
                    audit.write(
                        "preview_normalized",
                        call_id=call["id"],
                        reason="Removed zero-distance no-op steps",
                        arguments=args,
                    )
            if name == "execute_sequence":
                token = args["token"]
                p = previews.pop(token, None)
                if (
                    state.get("approved") != call["id"]
                    or not p
                    or time.time() > p["deadline"]
                ):
                    raise ValueError("No matching live operator approval")
                if hardware.mode == "real" and os.getenv("PRINTER_ALLOW_MOTION") != "1":
                    raise ValueError("Real motion disabled")
                audit.claim(token)
                args["operator_confirmed"] = True
            result = await hardware.call(name, args)
            if name != "measure_binary_pattern" and result.get("simulated") != (
                hardware.mode == "mock"
            ):
                raise ValueError("Printer mode mismatch")
            if name in ("preview_sequence", "preview_position"):
                validate_preview(result)
                previews[result["token"]] = dict(
                    result=result,
                    deadline=time.time() + min(110, result["expires_in_seconds"]),
                )
            failed = False
        except Exception as exc:
            result = dict(
                ok=False,
                error=str(exc),
                uncertain=name == "execute_sequence",
                automatic_retry=False,
            )
            failed = True
        audit.write("tool_result", tool=name, result=result, call_id=call["id"])
        messages = [
            ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                tool_call_id=call["id"],
                name=name,
            )
        ]
        if name == "execute_sequence" and not failed:
            messages.append(
                HumanMessage(
                    content="Operator approved exactly the preview for call "
                    + call["id"]
                    + ". This is not permission to repeat it.",
                    name="human",
                )
            )
        return dict(
            messages=messages,
            previews=previews,
            approved="",
            failed=failed,
            tool_count=state.get("tool_count", 0) + 1,
        )

    def human(state):
        reply = interrupt(dict(kind="clarification", question=state["question"]))
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("A nonempty reply is required")
        audit.write("human_answer", message=reply)
        return dict(messages=[HumanMessage(content=reply, name="human")])

    for name, node in [
        ("orchestrator", orchestrator),
        ("scientist", specialist("scientist")),
        ("general", specialist("general")),
        ("executor", executor),
        ("approval", approval),
        ("human", human),
        ("hardware", hardware_turn),
    ]:
        builder.add_node(name, node)
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges(
        "orchestrator",
        lambda s: s["next"],
        dict(
            scientist="scientist",
            general="general",
            executor="executor",
            human="human",
            finish=END,
        ),
    )
    for role in ("scientist", "general", "human"):
        builder.add_edge(role, "orchestrator")
    builder.add_conditional_edges(
        "executor",
        lambda s: "stop" if s.get("failed") else s["next"],
        dict(
            stop=END,
            approval="approval",
            hardware="hardware",
            orchestrator="orchestrator",
        ),
    )
    builder.add_conditional_edges(
        "approval",
        lambda s: s["next"],
        dict(hardware="hardware", orchestrator="orchestrator"),
    )
    builder.add_conditional_edges(
        "hardware",
        lambda s: "stop" if s["failed"] else "continue",
        {"stop": END, "continue": "executor"},
    )
    return builder.compile(checkpointer=checkpointer)
