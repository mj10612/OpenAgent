# Issue resolutions: 2026-10-13

The open backlog contained 20 issues and no pull requests. Existing upstream
redirect and blank-query fixes were retained before implementing this batch.

| Issues | Resolution | Regression evidence |
| --- | --- | --- |
| #81, #89 | Compare actual provider identity; preserve same-provider credentials, headers, gateway and custom mappings | REPL vendor switch and existing private gateway/custom provider tests |
| #82 | Reject empty/duplicate tool IDs before any execution | Invalid batches execute zero tools |
| #83 | Preserve original Gemini Parts and thoughtSignature associations in message metadata | Streaming, parallel calls, signature-only chunks, session round trip and replay |
| #84 | Store streamed call ID before ToolCallStart | Start event contains the wire call ID |
| #85 | Track ignored HTML ancestor tags | Nested noscript/SVG contents remain hidden |
| #86 | Decode gzip, deflate and Brotli incrementally with encoded and decoded limits | Normal content and compression-bomb rejection for all three encodings |
| #87 | Replace invalid UTF-8 bytes rather than omit the entire file | Search finds an ASCII query in legacy text |
| #88 | Fall back to exclusive copies where hard links are unavailable; retain no-overwrite behavior | Cross-device failure simulation and existing-destination protection |
| #90 | Add local /history, /messages and /prompt inspection | Inspection preserves history and literal prompt text |
| #91 | Add confirmation-gated unified-diff apply_patch for existing files | Multi-hunk edits, CRLF/LF, mismatch atomicity and sandbox rejection |
| #92, #93, #94 | Pin summaries and active user requests; summarize completed atomic tool groups; bound summaries | Tight-budget convergence and a 20-step tool loop through runner with failed LLM summarization |
| #95 | Match complete relative path components, with recursive ** semantics | Root, recursive and path-qualified patterns |
| #97 | Use a configurable bounded worker pool; await cancellation cleanup | 300-call batches at limits 1/4/8, ordered results and cancellation |
| #98 | Append complete JSONL batches and cache prior metadata; rewrite only changed history | Prior bytes unchanged, torn-batch recovery, edits and legacy-format export |
| #99 | Recognize -- before scanning options for implicit run | Dash-prefixed prompts and subcommand-like literal prompts |
| #100 | Remove unused helpers and duplicate permission enum; use heuristic API for actual fallback | Existing permission tests and runner-level summarizer failure test |
| #101 | Reconcile design with actual modules, routing, supported config and MCP transports | All documented module paths exist and the TOML example loads |

`ToolSpec.to_gemini_schema` and MCP reconnect callbacks were already used and
remain in place; removing them based on the original issue description would
break working behavior. `compact_heuristic` now serves the production fallback.

Session batches are readable by the current implementation alongside old
metadata/message rows. Before downgrading, stop agents and export sessions using
`SessionStore(storage_dir, incremental=False)`. Concurrent old/new writers on the
same session are unsupported. No user's sessions were migrated during validation.

Verification uses offline mocks. Live vendor API acceptance was not exercised.
The CI matrix additionally runs lint, typing, tests and packaging on Windows and
Linux with Python 3.11, 3.12 and 3.13.
