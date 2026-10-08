"""Conversation message history manager enforcing the Atomic Tool Pair Invariant.

Guarantees that an assistant message with tool_calls is never separated from
its subsequent tool result messages during truncation or sliding windowing.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from openagent.context.estimator import TokenEstimator
from openagent.core.types import ImagePart, Message, ToolCall


class MessageManager:
    """Manages conversation message history with turn-boundary invariants."""

    def __init__(
        self,
        messages: Sequence[Message] | None = None,
        *,
        estimator: TokenEstimator | None = None,
    ) -> None:
        self._messages: list[Message] = list(messages or [])
        self.estimator: TokenEstimator = estimator or TokenEstimator()

    # -- Message Accessors ------------------------------------------------- #

    @property
    def messages(self) -> list[Message]:
        """Return a copy of the current message history."""
        return list(self._messages)

    @property
    def system_message(self) -> Message | None:
        """Return the initial system prompt message if present."""
        if self._messages and self._messages[0].role == "system":
            return self._messages[0]
        return None

    @property
    def last_message(self) -> Message | None:
        """Return the latest message in history."""
        return self._messages[-1] if self._messages else None

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self) -> Iterator[Message]:
        return iter(self._messages)

    def __getitem__(self, item: Any) -> Any:
        return self._messages[item]

    # -- Mutation Methods -------------------------------------------------- #

    def add(self, message: Message) -> None:
        """Append a message to history."""
        self._messages.append(message)

    def append(self, message: Message) -> None:
        """Alias for add."""
        self._messages.append(message)

    def extend(self, messages: Iterable[Message]) -> None:
        """Append multiple messages to history."""
        self._messages.extend(messages)

    def pop(self, index: int = -1) -> Message:
        """Remove and return message at index."""
        return self._messages.pop(index)

    def clear(self) -> None:
        """Clear all messages from history."""
        self._messages.clear()

    def add_user(self, text: str, images: Sequence[ImagePart] | None = None) -> Message:
        """Create and append a user message."""
        msg = Message.user(text, images=images)
        self.add(msg)
        return msg

    def add_assistant(
        self,
        text: str | None = None,
        *,
        tool_calls: Sequence[ToolCall] | None = None,
        reasoning: str | None = None,
    ) -> Message:
        """Create and append an assistant message."""
        msg = Message.assistant(text, tool_calls=tool_calls, reasoning=reasoning)
        self.add(msg)
        return msg

    def add_tool_result(
        self,
        call_id: str,
        name: str,
        content: str,
        *,
        is_error: bool = False,
    ) -> Message:
        """Create and append a tool result message."""
        msg = Message.tool_result(call_id, name, content, is_error=is_error)
        self.add(msg)
        return msg

    def add_system(self, text: str) -> Message:
        """Create and prepend or append a system message."""
        msg = Message.system(text)
        if self._messages and self._messages[0].role == "system":
            self._messages[0] = msg
        else:
            self._messages.insert(0, msg)
        return msg

    def total_tokens(self, estimator: TokenEstimator | None = None) -> int:
        """Calculate total tokens across all messages in history."""
        est = estimator or self.estimator
        return est.estimate_messages(self._messages)

    # -- Atomic Group Partitioning ----------------------------------------- #

    @staticmethod
    def get_atomic_groups(messages: Sequence[Message]) -> list[list[Message]]:
        """Partition messages into atomic conversational units.

        An assistant message with tool calls and all consecutive tool responses that
        answer those calls form a single atomic group that cannot be broken.
        Normal user, system, or plain assistant messages form single-message groups.
        Orphaned tool messages (not preceded by assistant with tool calls) form their
        own groups.
        """
        groups: list[list[Message]] = []
        i = 0
        n = len(messages)
        while i < n:
            msg = messages[i]
            if msg.role == "assistant" and msg.tool_calls:
                group = [msg]
                i += 1
                while i < n and messages[i].role == "tool":
                    group.append(messages[i])
                    i += 1
                groups.append(group)
            else:
                groups.append([msg])
                i += 1
        return groups

    # -- Sliding Window ---------------------------------------------------- #

    @staticmethod
    def valid_tool_group(group: Sequence[Message]) -> bool:
        """Require exactly one result for each uniquely identified tool call."""
        first = group[0]
        if first.role == "tool":
            return False
        if first.role != "assistant" or not first.tool_calls:
            return True
        expected = [call.id for call in first.tool_calls]
        actual = [message.tool_call_id for message in group[1:]]
        return (
            all(expected)
            and len(set(expected)) == len(expected)
            and len(actual) == len(expected)
            and set(actual) == set(expected)
        )

    def window(
        self,
        max_tokens: int,
        estimator: TokenEstimator | None = None,
    ) -> list[Message]:
        """Retain initial system prompt and recent messages within max_tokens budget.

        Strictly enforces the Atomic Tool Pair Invariant:
        - Never cuts between an assistant tool call and its tool results.
        - Never leaves an orphaned tool response without its assistant tool call.
        - Never leaves an assistant tool call without its corresponding tool results.
        """
        est = estimator or self.estimator
        if not self._messages or max_tokens <= 0:
            return []

        # Retain the initial system prompt if present
        has_system = self._messages[0].role == "system"
        system_msg = self._messages[0] if has_system else None
        conv_messages = self._messages[1:] if has_system else self._messages

        system_tokens = est.estimate_message(system_msg) if system_msg else 0
        if system_tokens >= max_tokens:
            if conv_messages:
                raise ValueError("Context budget cannot fit the active turn and system prompt.")
            return [system_msg] if system_msg else []

        remaining_budget = max_tokens - system_tokens

        # Partition conversation messages into atomic groups
        groups = self.get_atomic_groups(conv_messages)
        selected_groups: list[list[Message]] = []

        for group in reversed(groups):
            first_msg = group[0]

            if not self.valid_tool_group(group):
                continue

            # Enforce Atomic Tool Pair Invariant:
            # 1. An orphaned tool message without an assistant must never be included.
            if first_msg.role == "tool":
                continue

            # 2. An assistant message with tool_calls must have its tool responses preserved.
            # If this assistant turn has tool_calls, but no tool responses followed it in history
            # (and it is not the latest active turn awaiting tool execution), skip it.
            if (
                first_msg.role == "assistant"
                and first_msg.tool_calls
                and len(group) == 1
                and conv_messages
                and first_msg is not conv_messages[-1]
            ):
                continue

            group_tokens = sum(est.estimate_message(m) for m in group)
            if group_tokens <= remaining_budget:
                selected_groups.append(group)
                remaining_budget -= group_tokens
            else:
                if not selected_groups:
                    raise ValueError(
                        f"The newest turn needs approximately {group_tokens} tokens, "
                        f"but only {remaining_budget} are available."
                    )
                # Cannot fit this entire atomic group; stop adding older turns.
                break

        selected_groups.reverse()
        result: list[Message] = []
        if system_msg:
            result.append(system_msg)
        for g in selected_groups:
            result.extend(g)
        return result

    def truncate_to_window(
        self,
        max_tokens: int,
        estimator: TokenEstimator | None = None,
    ) -> list[Message]:
        """Apply sliding window in place and return the updated messages."""
        self._messages = self.window(max_tokens, estimator=estimator)
        return list(self._messages)

    def copy(self) -> MessageManager:
        """Create a deep copy of this MessageManager."""
        return MessageManager(
            messages=[Message.from_dict(m.to_dict()) for m in self._messages],
            estimator=self.estimator,
        )

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialize message history to a list of dictionaries."""
        return [m.to_dict() for m in self._messages]

    @classmethod
    def from_dict(
        cls,
        data: Sequence[dict[str, Any]],
        *,
        estimator: TokenEstimator | None = None,
    ) -> MessageManager:
        """Deserialize message history from a list of dictionaries."""
        messages = [Message.from_dict(d) for d in data]
        return cls(messages=messages, estimator=estimator)
