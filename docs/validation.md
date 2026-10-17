# Validation

The repository passed 15 tests on 18 October 2026 (Asia/Yerevan), plus Ruff
lint/format checks and `pip check`. Tests use mocked model responses or the
scripted model; MCP integration tests run actual stdio servers with mock hardware.

The recording command was run separately with approval and refusal. Approval
produced [110,110,100] and a fresh post-execution status; refusal dispatched no
motion. The user subsequently supplied the successful recorded-run transcript,
retained under `evidence/simulation-approved/`.

Covered behavior includes same-thread clarification, matching approval call IDs,
expired/unknown tokens, single-use dispatch, uncertain-execution stop, mandatory
post-execution status, geometric/speed guards, bounded tool loops, zero-distance
step removal and the JSON model-proposal protocol.

## Limits

- The complete live-Groq movement workflow is not verified: attempts encountered
  repeated planning and provider rate limits. No claim of successful live movement
  should be inferred from the passing tests or the scripted recording.
- An earlier live-Groq/mock-printer status read is included as limited evidence.
- The new graph has not been demonstrated on a physical printer.
- Physical printer identity and provenance of existing footage remain operator
  supplied details for the final report.
- The graph adapts the earlier four-role architecture; it does not import the
  previous Silverbox scientist/training executor unchanged.

The final report is authored in LaTeX at `report/main.tex`, with the compiled
PDF at `report/PhotonicsAI_Homework3_Report.pdf`. Both desktop compilation and
Tectonic compilation succeeded; all five rendered pages were visually reviewed.
Outdated report drafts and failed local runs were removed from this directory.
