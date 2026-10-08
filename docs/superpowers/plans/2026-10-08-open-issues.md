# Resolve all open issues

Goal: Resolve the 57 initially open issues plus newly opened #80 on 2026-10-08, verify the integrated changes, push a commit, and comment on and close each resolved issue.

Architecture: Fix each defect at its owning layer. Preserve the public provider, runner, and tool interfaces where possible. Add regression tests before implementation; use offline transports for provider tests.

## Tasks and ownership

1. Providers, routing, canonical types, HTTP/SSE utilities: #9, #11, #13, #14, #25, #26, #27, #28, #30, #34, #35, #38, #61, #67, #72, #74. Verify independent arguments, token accounting, signed reasoning, routing, transport failure handling and redirects.
2. Tools and MCP: #10, #12, #31, #32, #33, #41, #42, #44, #45, #46, #47, #48, #49, #52, #53, #54, #70, #71, #75, #76, #78. Verify sandbox boundaries, bounded I/O, confirmation ordering, nested schemas, connection recovery, and new delete/move tools.
3. Runner, context, sessions and prompts: #23, #24, #37, #39, #40, #43, #55, #57, #65, #68, #69, #73, #77, #79. Verify cancellation rollback, active-turn retention, tool pairing, persistence before completion, safe session management, and instruction discovery.
4. Configuration, CLI, TUI, CI and documentation: #29, #36, #50, #56, #58, #59, plus CLI portions of #39/#77 and REPL portion of #43. Verify Unicode/config round trips, config-relative paths, atomic owner-only writes, literal untrusted rendering and session commands.
5. REPL commands (#80): preserve the conversation while switching the provider, keep explicit output limits, track terminal usage once, render literal tool tables, and collect triple-quote multiline input. Verify these behaviors through offline interactive-loop tests.

Each task: inspect issue and caller; write and run failing regression; implement minimal fix; run focused tests; report issue-to-test coverage. Integrate shared interfaces explicitly (ToolSpec schema and session management API).

## Review focus

- Cancellation after tool side effects must leave a valid conversation.
- Tight context budgets must reject safely rather than discard the active request.
- Remote schemas, descriptions, URLs and authentication are untrusted.
- Windows newline/encoding/path behavior must be covered alongside POSIX permissions.
- Existing and new session identifiers must resolve safely and unambiguously.

## Final gates

Run pytest, ruff check, mypy, wheel build and CLI smoke checks. Review the integrated diff, confirm each captured issue has a fix and test or documented supported behavior, and check for newly opened issues. Commit and push only after passing checks. Then post a per-issue explanation with commit and validation evidence and close it; verify remote commit and zero remaining open issues.
