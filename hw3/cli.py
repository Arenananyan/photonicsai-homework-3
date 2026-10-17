import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values, load_dotenv
from groq import APIError, APIStatusError
from langchain_core.messages import HumanMessage, messages_to_dict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from .graph import Audit, build_graph
from .model import Model
from .transport import ROOT, Hardware


async def run(args):
    load_dotenv(args.env_file)
    # An empty shell variable should not hide a configured local file value.
    if not os.getenv("GROQ_API_KEY", "").strip():
        configured_key = dotenv_values(args.env_file).get("GROQ_API_KEY")
        if configured_key:
            os.environ["GROQ_API_KEY"] = configured_key.strip()
    directory = (
        ROOT
        / "results"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    audit = Audit(directory, live=True)
    print(
        f"\nLANGGRAPH LAB | printer={args.mode} | model={'SCRIPTED DEMO' if args.demo else 'LIVE GROQ'}",
        flush=True,
    )
    print(
        "request → coordinator → scientist/executor → MCP → measured result\n                     motion → HUMAN APPROVAL → execution → status",
        flush=True,
    )
    audit.write(
        "run",
        mode=args.mode,
        prompt=args.prompt,
        model="scripted" if args.demo else "Groq",
        hardware_identity="Operator must verify exact printer make/model for the report",
    )
    graph = None
    config = None
    if args.demo:
        if args.mode != "mock":
            raise ValueError("Scripted demo is mock only")
        from .demo import DemoModel

        model = DemoModel(args.scenario)
    else:
        model = Model()
    try:
        async with Hardware(args.mode) as hardware:
            graph = build_graph(model, hardware, InMemorySaver(), audit)
            config = {
                "configurable": {"thread_id": uuid4().hex},
                "recursion_limit": 100,
            }
            update = {
                "messages": [HumanMessage(content=args.prompt)],
                "count": 0,
                "previews": {},
            }
            while True:
                state = await graph.ainvoke(update, config)
                for stop in state.get("__interrupt__", []):
                    print(json.dumps(stop.value, indent=2, ensure_ascii=False))
                snapshot = graph.get_state(config)
                with (directory / "dialogue.json").open("w") as out:
                    json.dump(
                        messages_to_dict(snapshot.values["messages"]),
                        out,
                        indent=2,
                        ensure_ascii=False,
                    )
                if not snapshot.interrupts:
                    break
                interrupt = snapshot.interrupts[0].value
                answer = input(
                    "APPROVE to execute this path; anything else declines: "
                    if interrupt["kind"] == "motion_approval"
                    else "Your answer: "
                )
                value = (
                    {"approved": answer == "APPROVE", "call_id": interrupt["call_id"]}
                    if interrupt["kind"] == "motion_approval"
                    else answer
                )
                update = Command(resume=value)
            print(
                "FAILED / inspect before retry"
                if snapshot.values.get("failed")
                else "Graph finished; inspect actual evidence in the dialogue."
            )
            return 1 if snapshot.values.get("failed") else 0
    except Exception as exc:
        audit.write("run_error", error=str(exc), automatic_retry=False)
        raise
    finally:
        if hasattr(model, "aclose"):
            await model.aclose()
        if getattr(model, "usage", None):
            audit.write("model_usage", requests=model.usage)
        if graph is not None:
            snapshot = graph.get_state(config)
            (directory / "dialogue.json").write_text(
                json.dumps(
                    messages_to_dict(snapshot.values.get("messages", [])),
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        print("Evidence:", directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--scenario", choices=["status", "move", "bench"], default="status"
    )
    try:
        exit_code = asyncio.run(run(parser.parse_args()))
    except ValueError as exc:
        parser.exit(1, f"Configuration or validation error: {exc}\n")
    except APIStatusError as exc:
        parser.exit(
            1,
            f"Groq request failed (HTTP {exc.status_code}); run stopped, no automatic retry. See the saved evidence.\n",
        )
    except APIError:
        parser.exit(
            1,
            "Groq connection failed; run stopped, no automatic retry. See the saved evidence.\n",
        )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
