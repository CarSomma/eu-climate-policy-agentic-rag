"""Unit tests for the shared OpenAI Responses tool-call loop."""

import json
from typing import Any
from unittest.mock import MagicMock

from pydantic import BaseModel

from eu_climate_policy_rag.core.agent import AbstractAgent
from eu_climate_policy_rag.core.agent_loop import OpenAIResponsesToolLoop
from eu_climate_policy_rag.core.tools import (
    FunctionTool,
    PydanticSchemaProvider,
    ToolRegistry as NativeToolRegistry,
)
from eu_climate_policy_rag.core.tools.adapters import OpenAIResponsesToolAdapter
from tests.helpers.responses_replay import (
    ResponsesReplay,
    function_call,
    function_call_output,
    message,
    response,
)


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


def build_agent(mock_client: MagicMock) -> LoopTestAgent:
    """Build a loop test agent with one local function tool."""

    tool = FunctionTool(
        name="echo",
        description="Echo text",
        schema_provider=PydanticSchemaProvider(EchoInput),
        handler=echo_handler,
    )
    return LoopTestAgent(
        openai_client=mock_client,
        model="test-model",
        instructions="Use tools when useful.",
        tools=NativeToolRegistry(function_tools=[tool]),
        max_turns=3,
    )


def build_native_agent(mock_client: MagicMock) -> LoopTestAgent:
    """Build a loop test agent with a native provider-neutral registry."""

    tool = FunctionTool(
        name="echo",
        description="Echo text",
        schema_provider=PydanticSchemaProvider(EchoInput),
        handler=echo_handler,
    )
    return LoopTestAgent(
        openai_client=mock_client,
        model="test-model",
        instructions="Use tools when useful.",
        tools=NativeToolRegistry(function_tools=[tool]),
        max_turns=3,
    )


def test_run_loop_appends_function_call_output_and_continues() -> None:
    """The loop should send model tool calls and tool outputs back next turn."""

    tool_call = function_call("echo", {"text": "hello"}, "call_echo")
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = [
        response(tool_call),
        response(message("final answer")),
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
            "output": json.dumps({"ok": True, "data": {"echo": "hello"}}),
        },
        message("final answer"),
    ]


def test_agent_create_response_uses_openai_responses_adapter_tools() -> None:
    """Agents should export model-visible tools through the provider adapter."""

    mock_client = MagicMock()
    mock_client.responses.create.return_value = response(message("done"))
    agent = build_agent(mock_client)

    assert isinstance(agent.tool_adapter, OpenAIResponsesToolAdapter)

    agent.run("Say hello")

    _, kwargs = mock_client.responses.create.call_args
    assert kwargs["tools"] == agent.tool_adapter.tools


def test_base_agent_converts_tool_result_through_responses_adapter() -> None:
    """Base agent dispatch should let the adapter build function_call_output."""

    tool_call = function_call("echo", {"text": "hello"}, "call_echo")
    mock_client = MagicMock()
    agent = build_agent(mock_client)
    original_converter = agent.tool_adapter.to_function_call_output
    agent.tool_adapter.to_function_call_output = MagicMock(wraps=original_converter)

    output = agent._execute_tool_call(tool_call)

    agent.tool_adapter.to_function_call_output.assert_called_once()
    assert output == {
        "type": "function_call_output",
        "call_id": "call_echo",
        "output": json.dumps({"ok": True, "data": {"echo": "hello"}}),
    }


def test_agent_accepts_native_tool_registry() -> None:
    """Agents should accept provider-neutral registries at the boundary."""

    tool_call = function_call("echo", {"text": "native"}, "call_echo")
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = [
        response(tool_call),
        response(message("done")),
    ]
    agent = build_native_agent(mock_client)

    final_answer, history = agent.run("Say hello")

    assert final_answer == "done"
    assert history[3] == {
        "type": "function_call_output",
        "call_id": "call_echo",
        "output": json.dumps({"ok": True, "data": {"echo": "native"}}),
    }


def test_agent_adapter_receives_provider_neutral_registry_for_native_tools() -> None:
    """Native registries should be passed directly to the provider adapter."""

    mock_client = MagicMock()
    agent = build_native_agent(mock_client)

    assert isinstance(agent.tools, NativeToolRegistry)
    assert agent.tool_adapter.registry is agent.tools


def test_run_loop_handles_multiple_function_calls_before_next_model_turn() -> None:
    """All function calls in one response should receive matching outputs."""

    first_call = function_call("echo", {"text": "one"}, "call_one")
    second_call = function_call("echo", {"text": "two"}, "call_two")
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = [
        response(first_call, second_call),
        response(message("done")),
    ]
    agent = build_agent(mock_client)

    _, history = agent.run("Use the echo tool twice")

    assert history[2:] == [
        first_call,
        second_call,
        {
            "type": "function_call_output",
            "call_id": "call_one",
            "output": json.dumps({"ok": True, "data": {"echo": "one"}}),
        },
        {
            "type": "function_call_output",
            "call_id": "call_two",
            "output": json.dumps({"ok": True, "data": {"echo": "two"}}),
        },
        message("done"),
    ]


def test_base_agent_unknown_tool_returns_structured_model_visible_error() -> None:
    """Unknown tools should become structured function_call_output failures."""

    tool_call = function_call("missing_tool", {}, "call_missing")
    mock_client = MagicMock()
    agent = build_agent(mock_client)

    output = agent._execute_tool_call(tool_call)

    payload = json.loads(output["output"])
    assert output["type"] == "function_call_output"
    assert output["call_id"] == "call_missing"
    assert payload["ok"] is False
    assert payload["error"]["type"] == "UnknownToolError"
    assert payload["error"]["message"] == "Unknown tool: missing_tool"


def test_responses_tool_loop_runs_with_callbacks() -> None:
    """The reusable loop should own Responses turn mechanics."""

    tool_call = function_call("echo", {"text": "hello"}, "call_echo")
    responses = [response(tool_call), response(message("done"))]

    def create_response(history: list[Any]) -> Any:
        return responses.pop(0)

    def execute_tool_call(call: Any) -> dict[str, str]:
        assert call is tool_call
        return {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": "hello",
        }

    loop = OpenAIResponsesToolLoop(
        create_response=create_response,
        execute_tool_call=execute_tool_call,
        max_turns=3,
    )

    final_answer, history = loop.run(
        query="Say hello",
        instructions="Use tools when useful.",
    )

    assert final_answer == "done"
    assert history == [
        {"role": "system", "content": "Use tools when useful."},
        {"role": "user", "content": "Say hello"},
        tool_call,
        {
            "type": "function_call_output",
            "call_id": "call_echo",
            "output": "hello",
        },
        message("done"),
    ]


async def test_responses_tool_loop_runs_async_tool_callbacks() -> None:
    """The reusable loop should support async tool dispatch callbacks."""

    tool_call = function_call("echo", {"text": "async"}, "call_async")
    responses = [response(tool_call), response(message("done"))]

    def create_response(history: list[Any]) -> Any:
        return responses.pop(0)

    async def execute_tool_call(call: Any) -> dict[str, str]:
        return {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": "async",
        }

    loop = OpenAIResponsesToolLoop(
        create_response=create_response,
        execute_tool_call=execute_tool_call,
        max_turns=3,
    )

    final_answer, history = await loop.run_async(
        query="Say hello",
        instructions="Use tools when useful.",
    )

    assert final_answer == "done"
    assert history[3] == {
        "type": "function_call_output",
        "call_id": "call_async",
        "output": "async",
    }


def test_responses_replay_fixture_covers_multiple_function_calls() -> None:
    """Replay helpers should make same-turn tool-call histories easy to assert."""

    first_call = function_call("echo", {"text": "one"}, "call_one")
    second_call = function_call("echo", {"text": "two"}, "call_two")
    first_output = function_call_output("call_one", "one")
    second_output = function_call_output("call_two", "two")
    replay = ResponsesReplay(
        [
            response(first_call, second_call),
            response(message("done")),
        ],
        tool_outputs={
            "call_one": first_output,
            "call_two": second_output,
        },
    )

    final_answer, history = replay.run(
        query="Use the tool twice",
        instructions="Use tools.",
    )

    assert final_answer == "done"
    assert history == [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Use the tool twice"},
        first_call,
        second_call,
        first_output,
        second_output,
        message("done"),
    ]
    assert replay.seen_histories[1] == history[:6]


def test_responses_replay_fixture_covers_unknown_tool_output() -> None:
    """Replay helpers should preserve structured unknown-tool outputs."""

    missing_call = function_call("missing_tool", {}, "call_missing")
    error_output = function_call_output(
        "call_missing",
        json.dumps(
            {
                "ok": False,
                "error": {
                    "type": "UnknownToolError",
                    "message": "Unknown tool: missing_tool",
                },
            },
        ),
    )
    replay = ResponsesReplay(
        [
            response(missing_call),
            response(message("done")),
        ],
        tool_outputs={"call_missing": error_output},
    )

    _, history = replay.run()

    assert history[2:4] == [missing_call, error_output]


def test_responses_replay_fixture_covers_max_turn_behavior() -> None:
    """Replay helpers should make max-turn regressions straightforward."""

    first_call = function_call("echo", {"text": "again"}, "call_one")
    second_call = function_call("echo", {"text": "again"}, "call_two")
    replay = ResponsesReplay(
        [
            response(first_call),
            response(second_call),
        ],
        tool_outputs={
            "call_one": function_call_output("call_one", "one"),
            "call_two": function_call_output("call_two", "two"),
        },
        max_turns=2,
    )

    final_answer, history = replay.run()

    assert final_answer == "Agent reached max turns without a final answer."
    assert history[2:] == [
        first_call,
        function_call_output("call_one", "one"),
        second_call,
        function_call_output("call_two", "two"),
    ]
