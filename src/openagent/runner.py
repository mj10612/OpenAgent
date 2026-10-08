"""The Core Agent Execution Loop.

Orchestrates multi-step agent reasoning, tool call execution, dynamic prompts,
context window management, streaming event delivery, and session persistence.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from pathlib import Path

from openagent.context.compactor import Compactor
from openagent.context.messages import MessageManager
from openagent.core.events import (
    DoneEvent,
    ErrorEvent,
    StartEvent,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResultEvent,
    UsageEvent,
)
from openagent.core.provider import ChatProvider
from openagent.core.types import (
    ChatRequest,
    FinishReason,
    Message,
    TextPart,
    ToolCall,
    Usage,
)
from openagent.prompts.base import PromptBuilder
from openagent.session.store import SessionStore
from openagent.tools.registry import AskCallback, ToolRegistry

__all__ = ["AgentRunner"]


class AgentRunner:
    """Core orchestrator driving the turn-based agent execution loop."""

    def __init__(
        self,
        provider: ChatProvider,
        *,
        model: str | None = None,
        tools: ToolRegistry | None = None,
        messages: MessageManager | None = None,
        prompt_builder: PromptBuilder | None = None,
        compactor: Compactor | None = None,
        session_store: SessionStore | None = None,
        session_id: str | None = None,
        max_tool_iterations: int = 25,
        max_tokens: int | None = None,
        context_window: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        workspace_root: str | Path | None = None,
        extra_instructions: str | Sequence[str] | None = None,
    ) -> None:
        self.provider = provider
        self.model = (
            model
            or getattr(provider, "model", None)
            or getattr(provider, "default_model", None)
            or "gpt-4o"
        )
        self.tools = tools if tools is not None else ToolRegistry()
        self.prompt_builder = prompt_builder if prompt_builder is not None else PromptBuilder()
        self.compactor = compactor if compactor is not None else Compactor()
        self.session_store = session_store
        self.session_id = session_id
        self.max_tool_iterations = max_tool_iterations
        self.explicit_max_tokens = max_tokens
        self.max_tokens = (
            max_tokens if max_tokens is not None else getattr(provider, "default_max_tokens", None)
        )
        self.context_window: int = (
            context_window
            if context_window is not None
            else getattr(provider, "context_window", None)
            or getattr(provider, "default_context_window", 128_000)
            or 128_000
        )
        if self.context_window <= 0 or (self.max_tokens is not None and self.max_tokens <= 0):
            raise ValueError("Context window and output token limit must be positive.")
        self.temperature = temperature
        self.top_p = top_p
        self.workspace_root = workspace_root
        self.extra_instructions = extra_instructions
        self._resuming_session = (
            messages is None and session_store is not None and session_id is not None
        )

        if messages is not None:
            self.messages = messages
        elif self.session_store is not None and self.session_id is not None:
            self.session_id = self.session_store.resolve_session_id(self.session_id)
            _, loaded_msgs = self.session_store.load_session(self.session_id)
            self.messages = MessageManager(loaded_msgs)
        else:
            self.messages = MessageManager()

        self._ensure_system_prompt(refresh=True)

    def _ensure_system_prompt(self, *, refresh: bool = False) -> None:
        """Ensure an initial system prompt is present in messages."""
        existing = self.messages.system_message
        if self.prompt_builder is not None and (
            existing is None
            or (
                refresh
                and (self._resuming_session or existing.metadata.get("openagent_generated_prompt"))
            )
        ):
            sys_text = self.prompt_builder.build_system_prompt(
                workspace_root=self.workspace_root,
                tools=self.tools,
                extra_instructions=self.extra_instructions,
            )
            prompt = self.messages.add_system(sys_text)
            prompt.metadata["openagent_generated_prompt"] = True

    def _persist_session(self) -> None:
        """Persist current conversation history to SessionStore if configured."""
        if self.session_store is not None:
            if self.session_id is None:
                self.session_id = self.session_store.create_session(model=self.model)
            self.session_store.save_messages(self.session_id, self.messages.messages)

    async def run_turn(
        self,
        user_input: str | Message,
        ask_callback: AskCallback | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Execute a transaction, restoring history on errors or interrupted consumption."""
        self._ensure_system_prompt()
        initial_messages = self.messages.copy().messages
        initial_session_id = self.session_id
        completed = False
        stream = self._run_turn(user_input, ask_callback)
        try:
            async for event in stream:
                if isinstance(event, ErrorEvent):
                    self.messages.clear()
                    self.messages.extend(initial_messages)
                if isinstance(event, DoneEvent):
                    completed = True
                yield event
        except Exception as exc:
            self.messages.clear()
            self.messages.extend(initial_messages)
            yield ErrorEvent(error=exc)
        finally:
            await stream.aclose()
            if not completed:
                self.messages.clear()
                self.messages.extend(initial_messages)
                self.session_id = initial_session_id

    async def _run_turn(
        self,
        user_input: str | Message,
        ask_callback: AskCallback | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute a conversational turn across streaming model calls and tools."""
        self._ensure_system_prompt()
        initial_messages = self.messages.copy().messages

        user_msg = Message.user(user_input) if isinstance(user_input, str) else user_input
        self.messages.append(user_msg)

        cumulative_usage = Usage()

        for _ in range(self.max_tool_iterations):
            tool_specs = self.tools.list_specs() if self.tools else []
            estimator = self.messages.estimator
            input_budget = (
                self.context_window
                - (self.max_tokens or 0)
                - estimator.estimate_tools(tool_specs)
                - estimator.request_overhead
            )
            system_message = self.messages.system_message
            system_tokens = estimator.estimate_message(system_message) if system_message else 0
            if input_budget <= system_tokens:
                self.messages.clear()
                self.messages.extend(initial_messages)
                yield ErrorEvent(
                    error=ValueError(
                        "Context window is too small for system, tools and output budget."
                    )
                )
                return
            # Check token budget and invoke Compactor if budget is exceeded
            if self.compactor is not None and self.messages.total_tokens() > input_budget:
                compacted = await self.compactor.compact(
                    self.messages.messages,
                    input_budget,
                    provider=self.provider,
                )
                self.messages.clear()
                self.messages.extend(compacted)

            # Window messages maintaining Atomic Tool Pair Invariant
            req_messages = self.messages.window(input_budget)
            if user_msg not in req_messages:
                raise ValueError("Context budget cannot fit the complete active user turn.")

            request = ChatRequest(
                messages=req_messages,
                model=self.model,
                tools=tool_specs,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            calls: list[ToolCall] = []
            done_event: DoneEvent | None = None
            error_encountered: ErrorEvent | None = None
            step_usage = Usage()

            try:
                async for event in self.provider.stream(request):
                    match event:
                        case TextDelta(text=t):
                            text_parts.append(t)
                            yield event
                        case ThinkingDelta(text=t):
                            reasoning_parts.append(t)
                            yield event
                        case ToolCallStart() | ToolCallDelta():
                            yield event
                        case ToolCallEnd(call=c):
                            if c is not None:
                                calls.append(c)
                            yield event
                        case UsageEvent(usage=u):
                            step_usage = step_usage + u
                            yield event
                        case DoneEvent() as done:
                            done_event = done
                        case ErrorEvent() as err:
                            error_encountered = err
                            yield event
                        case StartEvent():
                            yield event
                        case _:
                            yield event
            except Exception as exc:
                err_ev = ErrorEvent(error=exc)
                self.messages.clear()
                self.messages.extend(initial_messages)
                yield err_ev
                return

            if error_encountered is not None:
                self.messages.clear()
                self.messages.extend(initial_messages)
                return

            if done_event is not None and done_event.usage.total_tokens > 0:
                cumulative_usage = cumulative_usage + done_event.usage
            else:
                cumulative_usage = cumulative_usage + step_usage

            # Record assistant turn in messages
            if done_event is not None and done_event.message is not None:
                assistant_msg = done_event.message
                if calls and not assistant_msg.tool_calls:
                    assistant_msg.tool_calls = list(calls)
                if not assistant_msg.text and text_parts:
                    assistant_msg.content = [TextPart("".join(text_parts))]
                if not assistant_msg.reasoning and reasoning_parts:
                    assistant_msg.reasoning = "".join(reasoning_parts)
            else:
                assistant_msg = Message.assistant(
                    text="".join(text_parts) if text_parts else None,
                    tool_calls=calls,
                    reasoning="".join(reasoning_parts) if reasoning_parts else None,
                )

            self.messages.append(assistant_msg)

            # If no tool calls: yield DoneEvent, persist session, and end turn
            if not assistant_msg.tool_calls:
                final_done = (
                    done_event
                    if done_event is not None
                    else DoneEvent(
                        finish_reason=FinishReason.STOP,
                        message=assistant_msg,
                        usage=cumulative_usage,
                    )
                )
                final_done.message = assistant_msg
                final_done.usage = cumulative_usage
                self._persist_session()
                yield final_done
                return

            # Tool calls present: execute tools concurrently via ToolRegistry
            tool_results = await self.tools.execute_parallel(
                assistant_msg.tool_calls,
                ask_callback=ask_callback,
            )
            expected_ids = [call.id for call in assistant_msg.tool_calls]
            result_ids = [result.call_id for result in tool_results]
            if (
                not all(expected_ids)
                or len(set(expected_ids)) != len(expected_ids)
                or result_ids != expected_ids
            ):
                raise ValueError(
                    "Tool results do not correspond exactly to the assistant tool calls."
                )

            for call, result in zip(assistant_msg.tool_calls, tool_results, strict=True):
                yield ToolResultEvent(
                    call_id=result.call_id,
                    tool_name=call.name,
                    output=result.output,
                    is_error=result.is_error,
                )
                tool_msg = result.to_tool_message(name=call.name)
                self.messages.append(tool_msg)

        else:
            # Reached max_tool_iterations without model concluding
            terminal_event = DoneEvent(
                finish_reason=FinishReason.LENGTH,
                message=Message.assistant(
                    "Reached maximum tool call iterations without concluding."
                ),
                usage=cumulative_usage,
            )
            assert terminal_event.message is not None
            self.messages.append(terminal_event.message)
            self._persist_session()
            yield terminal_event

    async def run_turn_to_completion(
        self,
        user_input: str | Message,
        ask_callback: AskCallback | None = None,
    ) -> str:
        """Run a turn to completion and return the final text response."""
        text_chunks: list[str] = []
        done_text: str = ""
        error: BaseException | None = None
        async for event in self.run_turn(user_input, ask_callback=ask_callback):
            if isinstance(event, TextDelta):
                text_chunks.append(event.text)
            elif isinstance(event, DoneEvent) and event.message and event.message.text:
                done_text = event.message.text
            elif isinstance(event, ErrorEvent):
                error = (
                    event.error
                    if isinstance(event.error, BaseException)
                    else RuntimeError(str(event.error))
                )

        if error is not None:
            raise error

        if done_text.strip():
            return done_text
        return "".join(text_chunks)

    def reset(self) -> None:
        """Clear active turn state while preserving or re-initializing configuration."""
        self.messages.clear()
        self.session_id = None
        self._ensure_system_prompt()
