# Architecture and sequence diagrams

```mermaid
flowchart TD
 U[Human request] --> O[Orchestrator LLM]
 O --> S[Scientist LLM: experiment plan]
 O --> G[General LLM: explanation]
 S --> O
 G --> O
 O --> E[Executor LLM: exclusive MCP tools]
 E --> V[Validate proposed tool and arguments]
 V --> P{Motion dispatch?}
 P -->|yes| H[LangGraph human interrupt: exact preview approval]
 H -->|approved| T[Hardware turn]
 H -->|declined| D[Shared dialogue: refusal]
 P -->|no| T
 T --> M[Printer MCP server]
 M --> C[Existing SSH connector + policies + watchdog]
 C --> R[Real printer OR explicit mock]
 T --> B[Optical-bench MCP]
 B --> SIM[Homework 2 simulated bench]
 R --> TM[Native ToolMessage: measured result]
 SIM --> TM
 TM --> E
 E --> O
 D --> O
 O --> Q[Human clarification interrupt]
 Q --> O
```

```mermaid
sequenceDiagram
 participant U as Human
 participant O as Orchestrator
 participant S as Scientist
 participant E as Executor LLM
 participant H as Hardware node
 participant M as Printer MCP + connector
 participant P as Printer (mock or real)
 U->>O: Request a measured movement experiment
 O->>S: Ask for a plan
 S-->>O: Plan and measurement requirements
 O->>E: Operate instrument
 E->>H: Model-selected printer_status
 H->>M: MCP call
 M->>P: Read state
 P-->>M: Firmware telemetry
 M-->>H: Status object
 H-->>E: Shared ToolMessage
 E->>H: Model-selected preview_sequence
 H->>M: Validate complete proposed path
 M-->>E: ToolMessage: token, waypoints, speed, bounds
 E->>U: LangGraph interrupt: review exact preview
 U-->>E: Approve this call (or refuse)
 E->>H: Approved execute_sequence
 H->>M: Claim token once, execute
 M->>P: Validated motion with watchdog
 P-->>M: Execution and telemetry
 M-->>E: Shared ToolMessage
 E->>H: Model-selected printer_status
 H->>M: Read again
 M-->>E: Shared measured post-motion status
 E-->>O: Explain actual results
 O-->>U: Next decision based on measurement
```

The architecture supports these branches; actual run order is recorded in each
audit file. Scripted examples demonstrate transport/graph behavior only. A real
LLM/real-hardware sequence and video still need operator capture.
