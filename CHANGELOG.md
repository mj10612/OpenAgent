# Changelog

## Unreleased

- Fix provider routing, streaming tool arguments, token accounting, reasoning replay, and HTTP failure handling.
- Preserve complete MCP input schemas, isolate external tools, serialize approvals and mutations, and recover disconnected servers.
- Bound filesystem and network reads, preserve file newlines, and add sandboxed file deletion and moving.
- Validate shell timeouts, scrub inherited credentials, close subprocess stdin, and decode Windows output correctly.
- Preserve active context and tool pairs; recover malformed session records and canceled turns; save sessions before completion events.
- Add session deletion/pruning and discover workspace instruction files.
- Fix Unicode and MCP configuration round trips, config-relative paths, secure atomic configuration writes, and literal terminal rendering.
- Add Linux/Windows CI for Python 3.11–3.13 and correct package repository links.

## 0.1.0

- Initial model-agnostic terminal agent with native and compatible providers, MCP tools, session persistence, and context compaction.
