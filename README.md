# Homework 3 — Hardware in the loop

Aren Ananyan · PhotonicsAI

A standalone LangGraph application with coordinator, scientist, general and
executor roles, MCP instrument tools, shared measured results and a human approval
interrupt before motion. Python 3.12+ is required.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
```

`requirements.txt` lists direct dependencies; `requirements.lock.txt` pins the
tested environment. Development tools are in `requirements-dev.txt`.

## Reproduce the recorded simulation

```bash
.venv/bin/python -m hw3.cli \
  "Plan a simulated move to X110 Y110 Z100, preview it, wait for approval, execute once and measure the final position." \
  --mode mock --demo --scenario move
```

The program prints the graph transitions and MCP requests/results as they occur.
It first reads [100, 100, 5], previews a vertical lift to Z100 and XY travel to
[110, 110, 100], then pauses. Type `APPROVE` to execute that exact preview once;
any other reply declines. A fresh status read verifies the final mock position.
Approval expires; do not leave the preview waiting for more than 110 seconds.

`--demo` uses a scripted decision model with the actual LangGraph, stdio MCP
servers and mock printer. It demonstrates the graph and approval mechanism;
it does not demonstrate live LLM reasoning or physical motion.

Other scripted scenarios:

```bash
.venv/bin/python -m hw3.cli "Read simulated status" --demo --scenario status
.venv/bin/python -m hw3.cli "Measure binary optical pattern" --demo --scenario bench
```

The optical bench uses binary on/off macropixels and returns simulated
measurements.

## Live Groq mode

Copy `.env.example` to `.env` and configure `GROQ_API_KEY` locally, or export the
variable in the current terminal. Never publish the key. Then omit `--demo`:

```bash
.venv/bin/python -m hw3.cli "Read the simulated printer status and report its position" --mode mock
```

Groq returns role decisions as JSON. Only executor receives discovered MCP tool
schemas and proposes a tool name and arguments. The host validates the proposal,
creates the LangGraph tool call and invokes MCP. Measured results return to the
shared dialogue before the next model decision. Full transcripts are preserved
while model requests use a compact measured-state context.

**Verification limit:** the included live-Groq evidence is a mock-status read
from an earlier run. The live movement scenario has encountered repeated planning
and provider rate limits; it is not an accepted end-to-end demonstration. The
recorded complete movement scenario uses `--demo`. Provider errors stop the run
without automatic retries or a hidden scripted fallback.

## Real printer mode

The connector uses an already authenticated shared SSH session. Configure the
current `SSH_HOST` and `SSH_CONTROL_PATH` locally, then use `--mode real` for
operator-supervised operation. The exact physical printer make/model must be
provided by the operator.

Real motion additionally requires `PRINTER_ALLOW_MOTION=1` and approval of the
concrete preview. Bounds, reported position and watchdog policies are checked
again by the printer MCP. The host rejects diagonal XYZ travel and XY travel
below `max(100, start Z, final Z)` mm. This geometric guard does not detect obstacles.
A consumed token or uncertain execution is never automatically retried. Checkpoints
support human pause/resume in the same process; a restart does not restore a job.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

Optional code checks:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/ruff check hw3 tests bench_server.py
.venv/bin/ruff format --check hw3 tests bench_server.py
```

## Repository contents

| Path | Purpose |
| --- | --- |
| `hw3/` | LangGraph, model adapter, CLI, transport and motion checks |
| `bench_server.py` | Simulated optical-bench MCP server |
| `vendor/` | Required printer connector/MCP modules and optical simulator |
| `tests/` | Protocol, integration and safety regression tests |
| `docs/diagrams.md` | Architecture and sequence diagrams |
| `docs/evidence/` | Selected transcripts with explicit model/hardware modes |
| `docs/validation.md` | Verified behavior and remaining limitations |
| `videos/` | Supplied demonstration recordings and their descriptions |

Each new run writes audit JSONL and dialogue JSON to local `results/`, ignored
by Git. The local virtual environment and credentials are ignored too.

The printer source license is preserved at `vendor/printer/LICENSE`.
`docs/source-manifest.json` records the origins and hashes of reused modules.
The final LaTeX report and repository publication are pending review.
