# Open issue backlog implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development to implement independent tasks with regression tests and final review.

**Goal:** Resolve all 26 open issues (#135 through #160), commit and push to origin/main, comment with evidence and close each issue.

**Architecture:** Fix defects at the owning boundary while retaining existing public APIs and default safety policies. Split provider, tool, session/runner, and CLI ownership to prevent conflicting edits.

**Tech Stack:** Python 3.11+, pytest/pytest-asyncio, httpx, MCP, Ruff, mypy.

**Spec:** GitHub issues https://github.com/mj10612/OpenAgent/issues/135 through /160; each issue contains reproduction and expected behavior.

## Global constraints

- Preserve existing user changes; starting tree is clean at d3fa2b5.
- No live LLM requests or real credentials in tests.
- Add failing regression tests before implementing fixes.
- Preserve bounded tool output, workspace confinement, atomic session writes, and explicit deny policies.
- User has authorized commit, push, issue comments, and closure; proceed after verification.

## Review focus

- Checkpoint memory includes both snapshots, and failed undo remains retryable.
- Large file moves/deletes work without allocating unbounded in-memory snapshots.
- Malformed session records cannot erase valid later history or trigger age deletion.
- Provider streams distinguish explicit errors and separate candidate choices.
- Repeated CLI policies retain deny regardless of flag location.

### Task 1: Providers (#148-152, #158-160)

**Files:** providers/openai_compat.py, gemini.py, anthropic.py, custom.py, ollama.py; tests/test_issue_provider_fixes.py.
- [ ] Add regression tests for choice isolation, in-band errors, Gemini usage, pagination and discovery metadata, default output cap, Ollama num_ctx, Gemini JSON mode; verify failures.
- [ ] Implement adapter fixes and run provider test suites.

### Task 2: Tools (#136-137, #142-147)

**Files:** tools/changes.py, fs.py, patch.py, mcp/manager.py, mcp/client.py; tests/test_issue_tool_fixes.py.
- [ ] Add failing tests for complete checkpoint accounting, large-file mutations, oversized edits, empty-file diffs, undo failure retry, LF-only patch splitting, lifecycle races, embedded resources.
- [ ] Implement bounded checkpoints and transactional/retryable undo; preserve confinement and permissions; run tool suites.

### Task 3: Sessions and runner (#135, #153-157)

**Files:** session/store.py, context/messages.py if needed, runner.py; tests/test_issue_session_fixes.py.
- [ ] Add failing tests for redacted raw arguments, invalid result IDs/roles/arguments, finite metadata, overlapping turns.
- [ ] Validate import/load boundaries and reject overlapping turns before mutation; run session/runner/context suites.

### Task 4: CLI/config/doctor (#138-141)

**Files:** cli.py, config.py, doctor.py, core/router.py if needed; tests/test_issue_cli_fixes.py.
- [ ] Add failing tests for policies before/after subcommands, diagnostic routing, preset JSON output, session_id round-trip.
- [ ] Preserve CLI contract and diagnose real router configuration without network requests.

### Task 5: Integration and delivery

- [ ] Review all diffs and cross-boundary behavior; fix remaining findings.
- [ ] Run full pytest, Ruff and mypy as CI requires; verify every issue has regression coverage.
- [ ] Commit fixes, push origin/main, verify remote commit.
- [ ] Comment on all resolved issues with commit and checks; close and verify no open issues remain.
