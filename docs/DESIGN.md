# OpenAgent implementation design

This document describes the implemented architecture. Azure and Bedrock adapters,
Streamable HTTP MCP, and a web search tool are not implemented. Their identifiers
must not be interpreted as working native integrations.

## Execution flow

`cli.py` loads configuration, resolves the provider, creates sandboxed tools and
MCP connections, then constructs `AgentRunner`. `tui/app.py` consumes normalized
stream events or drives the interactive REPL. The runner owns conversational turns,
context budgeting, tool dispatch, and session persistence.

The provider paths are:

- Native adapters: Anthropic Messages, Gemini Generate Content, and Ollama.
- OpenAI-compatible gateways: presets select endpoint, credentials, defaults,
  and tool protocol for `OpenAICompatProvider`.
- Custom HTTP JSON APIs: `CustomJsonPathProvider` uses configurable JSONPath
  response mappings. This is an escape hatch, not a separate registry.

## Implemented modules

Paths below are relative to `src/openagent/`.

| File | Responsibility |
| --- | --- |
| `cli.py` | Argument parsing, run/chat/models/sessions/config commands |
| `config.py` | Layered TOML, environment and CLI configuration |
| `runner.py` | Agent turn loop and session transactions |
| `core/types.py` | Message, content parts, tool schemas, requests and responses |
| `core/events.py` | Normalized streaming event types |
| `core/provider.py` | ChatProvider contract and typed provider errors |
| `core/presets.py` | Provider defaults and credentials |
| `core/router.py` | ModelReference parsing and ProviderRouter resolution |
| `providers/openai_compat.py` | OpenAI-compatible streaming adapter |
| `providers/anthropic.py` | Native Anthropic adapter |
| `providers/gemini.py` | Native Gemini adapter and signed response Part replay |
| `providers/ollama.py` | Native Ollama adapter |
| `providers/custom.py` | Custom JSONPath response adapter |
| `providers/text_protocol.py` | Text tool-call codec |
| `context/estimator.py` | Input, output reserve and schema token estimation |
| `context/messages.py` | History and atomic tool-call/result grouping |
| `context/compactor.py` | LLM summarization and bounded heuristic fallback |
| `prompts/base.py` | System instructions, environment and workspace rule loading |
| `tools/base.py` | Tool interface and ToolResult |
| `tools/registry.py` | Schema validation, PermissionAction and bounded dispatch |
| `tools/fs.py` | Sandboxed file operations, glob and text search |
| `tools/patch.py` | Atomic multi-hunk unified-diff editing |
| `tools/shell.py` | Bounded subprocess output and process-tree termination |
| `tools/agent_tools.py` | ThinkTool, TodoTool and WebFetchTool |
| `tools/mcp/client.py` | MCP stdio/SSE connections and tool adapters |
| `tools/mcp/manager.py` | Namespaced tools, lifecycle and reconnect preparation |
| `session/store.py` | JSONL session transactions and recovery |
| `tui/app.py` | Rich output, approval prompts and REPL |
| `utils/http.py` | HTTP transport and error mapping |
| `utils/sse.py` | SSE decoding |

There is no ProviderRegistry or automatic Normalizer repair stage. Router factory
branches create providers; tools are registered in ToolRegistry. Prompts live in
Python rather than a separate assets directory.

## Messages and protocol fidelity

`Message.content` is a list of content parts. `ContentPart` is a union alias,
not a constructible class. Use `Message.user("request")`,
`Message.assistant("response")`, or `TextPart(text="...")`.

Providers normalize responses to StreamEvent values. The runner retains the final
DoneEvent message so vendor metadata survives. Gemini stores original response
Parts, including thoughtSignature associations, in message metadata and replays
those Parts when continuing tool calls. Anthropic signed thinking blocks are
retained as ThinkingPart values. Session serialization preserves this information.

An assistant tool-call message and its matching tool-result messages form one
atomic group. Call IDs must be nonempty and unique before tools execute. Results
must correspond exactly afterward. Invalid or orphaned groups are never sent to
providers. Failed or cancelled turns restore prior conversation history; external
tool side effects cannot be undone by restoring messages.

## Context management

The input budget subtracts reserved output tokens, tool schemas and request
framing from the configured context window. Compaction pins the active user
request and retains recent complete atomic groups. Earlier completed tool steps,
including steps in the same user turn, can be summarized. Summaries have a token
cap and stay pinned during windowing. When the active request and newest atomic
group cannot fit even with a minimal summary, the runner reports a clear error.

`Compactor.compact` uses LLM summaries when available and calls
`compact_heuristic` on failure. Previous heuristic summaries are bounded rather
than nested indefinitely. Token counts are estimates, not vendor tokenizer counts.

## Permissions and tool limits

PermissionAction in `tools/registry.py` is the authoritative allow/ask/deny enum.
Read-only built-ins default to allow; network, write and execute default to ask.
MCP annotations do not lower execute permissions. Write and execute operations
are serialized. ToolRegistry runs at most eight workers by default; callers can
set `max_concurrent_tools` when constructing the registry. Cancellation awaits
worker cleanup and result ordering follows call ordering.

Filesystem paths are resolved under the workspace and checked for containment.
Search traversal prunes dependency directories and symlinks. `*.py` matches direct
children; `**/*.py` matches root and nested paths. Unified patches validate every
hunk before atomic publication and preserve CRLF/LF endings. File moves refuse
existing destinations and use exclusive copies where hard links are unsupported.

WebFetchTool validates and pins public HTTP(S) addresses, rejects unsafe redirects,
and limits both encoded and decoded bodies to 2 MB. gzip, deflate and Brotli
responses use bounded streaming decompression. Shell tools still run with the
process account's filesystem access; workspace path checks are not an OS sandbox.

## Configuration example

```toml
model = "custom/internal-model"
base_url = "https://gateway.example/v2/generate"
workspace = "."
max_tokens = 4096
context_window = 64000

[custom_provider]
chat_endpoint = ""
jsonpath_text = "response.output.text"
jsonpath_tool_calls = "response.output.actions[*]"

[danger_policy]
none = "allow"
network = "ask"
write = "ask"
execute = "ask"

[[mcp_servers]]
name = "local"
transport = "stdio"
command = "example-mcp-server"
args = []
```

Supported MCP transports are `stdio` and `sse`. A remote server uses
`transport = "sse"` and `url = "https://example.com/sse"`. Arbitrary
`[providers.mygateway]` tables are not supported; use top-level endpoint settings
and `[custom_provider]` instead. Paths resolve relative to the containing TOML.
Credentials should normally come from environment variables.

## Sessions and interactive inspection

The initial session record and history replacements are written by atomic file
replacement. Incremental saves append one JSONL batch containing new messages
and current metadata; readers accept only complete batches, so torn writes do
not expose partial turns. The store caches previous serialized rows and metadata
while the file remains unchanged. Compaction or changed prior messages take the
full rewrite path. Legacy metadata/message rows remain readable.
Before downgrading, close all agents and use a
`SessionStore(storage_dir, incremental=False)` to load and save each session with
its metadata; this exports the original metadata/message-only format without
dropping message metadata. Old and new agent versions must not concurrently
write the same session. No existing sessions are converted until a save occurs.

The REPL supports /model, /tokens (/usage), /tools, /history (/messages), /prompt,
/clear, /help and /exit (/quit). Inspection commands do not call the provider.
Same-provider model changes preserve configured credentials and gateway settings;
cross-provider changes resolve the target provider independently.

## Extension and verification

For an OpenAI-compatible service, add a ProviderPreset in `core/presets.py`.
For a new protocol, implement ChatProvider in `providers/` and add an instantiation
branch in `core/router.py`. A native adapter must round-trip tool calls, results
and vendor reasoning metadata. Register new built-in tools in the relevant
factory, with an explicit danger level and input schema.

Run `python -m ruff check src tests`, `python -m mypy src tests`,
`python -m pytest -q`, and `python -m build`. CI checks Windows and Linux with
Python 3.11, 3.12 and 3.13. Tests use offline mocks; live vendor acceptance still
requires valid credentials and compatible endpoints.
