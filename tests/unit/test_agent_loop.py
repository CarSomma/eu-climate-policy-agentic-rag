"""Unit tests for the shared OpenAI Responses tool-call loop."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from pydantic import BaseModel

from eu_climate_policy_rag.core.agent import AbstractAgent
from eu_climate_policy_rag.core.tooling import OpenAIFunctionTool, ToolRegistry


class EchoInput(BaseModel):
    """Arguments for a small loop test tool."""

    text: str


def echo_handler(text: str) -> dict[str, str]:
    """Return a predictable structured payload."""

    return {"echo": text}


class LoopTestAgent(AbstractAgent):
    """Concrete agent used to exercise the abstract loop."""

    def run(self, query: str) -> Any:
        return self._run_loop(query)


def make_message(text: str) -> SimpleNamespace:
    """Create a Responses API-like message item."""

    return SimpleNamespace(type="message", content=[SimpleNamespace(text=text)])


def make_tool_call(
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


def make_response(*output: SimpleNamespace) -> SimpleNamespace:
    """Create a Responses API-like response object."""

    return SimpleNamespace(output=list(output))


def build_agent(mock_client: MagicMock) -> LoopTestAgent:
    """Build a loop test agent with one local function tool."""

    tool = OpenAIFunctionTool(
        name="echo",
        description="Echo text",
        input_model=EchoInput,
        handler=echo_handler,
    )
    return LoopTestAgent(
        openai_client=mock_client,
        model="test-model",
        instructions="Use tools when useful.",
        tools=ToolRegistry(function_tools=[tool]),
        max_turns=3,
    )


def test_run_loop_appends_function_call_output_and_continues() -> None:
    """The loop should send model tool calls and tool outputs back next turn."""

    tool_call = make_tool_call("echo", {"text": "hello"}, "call_echo")
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = [
        make_response(tool_call),
        make_response(make_message("final answer")),
    ]
    agent = build_agent(mock_client)

    final_answer, history = agent.run("Say hello")

    assert final_answer == "final answer"
    assert mock_client.responses.create.call_count == 2
    assert history == [
        {"role": "system", "content": "Use tools when useful."},
        {"role": "user", "content": "Say hello"},
        tool_call,
        {
            "type": "function_call_output",
            "call_id": "call_echo",
            "output": json.dumps({"echo": "hello"}),
        },
        make_message("final answer"),
    ]


def test_run_loop_handles_multiple_function_calls_before_next_model_turn() -> None:
    """All function calls in one response should receive matching outputs."""

    first_call = make_tool_call("echo", {"text": "one"}, "call_one")
    second_call = make_tool_call("echo", {"text": "two"}, "call_two")
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = [
        make_response(first_call, second_call),
        make_response(make_message("done")),
    ]
    agent = build_agent(mock_client)

    _, history = agent.run("Use the echo tool twice")

    assert history[2:] == [
        first_call,
        second_call,
        {
            "type": "function_call_output",
            "call_id": "call_one",
            "output": json.dumps({"echo": "one"}),
        },
        {
            "type": "function_call_output",
            "call_id": "call_two",
            "output": json.dumps({"echo": "two"}),
        },
        make_message("done"),
    ]
