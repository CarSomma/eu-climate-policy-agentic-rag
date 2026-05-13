"""Helpers for replaying OpenAI Responses-style agent turns in tests."""

import json
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

from eu_climate_policy_rag.core.agent_loop import OpenAIResponsesToolLoop


def message(text: str) -> SimpleNamespace:
    """Create a Responses API-like message item."""

    return SimpleNamespace(type="message", content=[SimpleNamespace(text=text)])


def function_call(
    name: str,
    arguments: dict[str, Any],
    call_id: str,
) -> SimpleNamespace:
    """Create a Responses API-like function_call item."""

    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=json.dumps(arguments),
        call_id=call_id,
    )


def response(*output: SimpleNamespace) -> SimpleNamespace:
    """Create a Responses API-like response object."""

    return SimpleNamespace(output=list(output))


def function_call_output(
    call_id: str,
    output: str,
) -> dict[str, str]:
    """Create a Responses API-like function_call_output item."""

    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }


class ResponsesReplay:
    """Replay a finite sequence of Responses-style outputs through the loop."""

    def __init__(
        self,
        outputs: Iterable[SimpleNamespace],
        *,
        tool_outputs: dict[str, dict[str, str]] | None = None,
        max_turns: int = 3,
    ) -> None:
        self._responses = list(outputs)
        self.tool_outputs = tool_outputs or {}
        self.max_turns = max_turns
        self.seen_histories: list[list[Any]] = []

    def run(self, *, query: str = "Question", instructions: str = "Use tools.") -> tuple[str, list[Any]]:
        """Run the replay through the sync Responses tool loop."""

        loop = OpenAIResponsesToolLoop(
            create_response=self.create_response,
            execute_tool_call=self.execute_tool_call,
            max_turns=self.max_turns,
        )
        return loop.run(query=query, instructions=instructions)

    def create_response(self, history: list[Any]) -> SimpleNamespace:
        """Return the next response and record the incoming message history."""

        self.seen_histories.append(list(history))
        if not self._responses:
            msg = "Responses replay is exhausted."
            raise AssertionError(msg)
        return self._responses.pop(0)

    def execute_tool_call(self, call: Any) -> dict[str, str]:
        """Return the configured output for a replayed function call."""

        try:
            return self.tool_outputs[call.call_id]
        except KeyError as exc:
            msg = f"No replayed tool output for call_id {call.call_id}."
            raise AssertionError(msg) from exc
