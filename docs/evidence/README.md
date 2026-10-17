# Selected run evidence

These are original recorded tool responses and dialogue, not reconstructed results.
Timestamps in records are UTC Unix timestamps; recording filenames use local time.

| Folder | Model | Instrument | Scope |
| --- | --- | --- | --- |
| `simulation-approved` | Scripted | Mock printer | The user-recorded run: preview → approval → one execution → final status [110,110,100]. Original run `20261017T200720Z-d3d5fce7`. |
| `simulation-declined` | Scripted | Mock printer | Refused approval; no execute call. Original run `20261017T200227Z-18340048`. |
| `simulated-optical-bench` | Scripted | Simulated optical bench | Binary pattern measurement. Original run `20261017T193520Z-44621c2f`. |
| `live-groq-mock-status` | Live Groq | Mock printer | Earlier read-only status run, not a completed live movement run. Original run `20261017T180358Z-0ab2b52f`. |

The full live-Groq movement workflow has not passed acceptance. These records do
not demonstrate that the new graph has controlled a physical instrument.
