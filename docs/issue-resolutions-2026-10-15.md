# Issue resolutions: 2026-10-15

All open issues #135–#160 were addressed with regression coverage in four
`tests/test_issue_*_fixes.py` suites.

| Issues | Resolution | Regression suite |
| --- | --- | --- |
| #135 | Export redaction reconstructs raw tool arguments from masked values. | session |
| #136–#137 | Count both memory snapshots; stream large undo snapshots to bounded temporary disk storage. | tool |
| #138 | Preserve repeated tool policies across root and subcommand flags, including duplicate rejection. | CLI |
| #139 | Share offline preset routing and reject unavailable providers in doctor. | CLI |
| #140 | Serialize preset discovery as versioned JSON when requested. | CLI |
| #141 | Persist the configured session resume target. | CLI |
| #142–#143 | Reject oversized edits before publication and show empty-file existence changes. | tool |
| #144 | Track partial undo progress so restoration remains retryable without treating restored files as external edits. | tool |
| #145 | Split unified diffs on LF and preserve embedded control/Unicode separators, CRLF and missing final newlines. | tool |
| #146 | Discard outdated MCP discovery/reconnect results after stop. | tool |
| #147 | Render embedded text resources with URI/MIME information and bounded output. | tool |
| #148–#149 | Consistently select choice zero and surface explicit streamed provider errors. | provider |
| #150 | Include Gemini thoughts in completion totals and preserve cached-token snapshots. | provider |
| #151–#152 | Follow native discovery pages with cursor guards and use authoritative Anthropic token limits. | provider |
| #153–#156 | Validate result IDs, role/call compatibility, finite timestamps and imported argument JSON; retain later valid history and skip unsafe age cleanup. | session |
| #157 | Reject overlapping runner turns before modifying shared history; release the guard on cancellation. | session |
| #158–#160 | Apply Custom output defaults, send effective Ollama context size and map Gemini JSON response mode. | provider |

## Compatibility decisions

- Existing public tool and provider interfaces remain unchanged.
- Large-file checkpoints retain at most 1 GiB of disk backups; older completed
  turns may be evicted. A mutation exceeding the disk cap is refused before
  execution. Small snapshots retain the existing 40 MiB memory limit.
- OpenAI multi-choice requests retain choice zero; usage remains the provider's
  aggregate usage for the request.
- Ollama's effective context window takes precedence over a conflicting
  `extra_body.options.num_ctx`, while other options are retained.
- Invalid session age metadata prevents age-based deletion; explicit deletion
  remains available.

## Verification

CI gates: `python -m pytest -q`, `python -m ruff check src tests`,
`python -m mypy src tests`, and `python -m build`.

Each issue receives the pushed commit and final gate results before closure.

Local verification on Windows / Python 3.11: **516 passed, 2 skipped**; Ruff
passes, mypy reports no issues in 71 files, and source/wheel builds succeed.
An independent read-only review found no remaining substantive defects.
