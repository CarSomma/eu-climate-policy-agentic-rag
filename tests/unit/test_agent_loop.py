"""Unit tests for the shared OpenAI Responses tool-call loop."""

import json
from types import SimpleNamespace
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
            "output": json.dumps({"ok": True, "data": {"echo": "hello"}}),
        },
        make_message("final answer"),
    ]


def test_agent_create_response_uses_openai_responses_adapter_tools() -> None:
    """Agents should export model-visible tools through the provider adapter."""

    mock_client = MagicMock()
    mock_client.responses.create.return_value = make_response(make_message("done"))
    agent = build_agent(mock_client)

    assert isinstance(agent.tool_adapter, OpenAIResponsesToolAdapter)

    agent.run("Say hello")

    _, kwargs = mock_client.responses.create.call_args
    assert kwargs["tools"] == agent.tool_adapter.tools


def test_base_agent_converts_tool_result_through_responses_adapter() -> None:
    """Base agent dispatch should let the adapter build function_call_output."""

    tool_call = make_tool_call("echo", {"text": "hello"}, "call_echo")
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

    tool_call = make_tool_call("echo", {"text": "native"}, "call_echo")
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = [
        make_response(tool_call),
        make_response(make_message("done")),
    ]
    agent = build_native_agent(mock_client)

    final_answer, history = agent.run("Say hello")

    assert final_answer == "done"
    assert history[3] == {
        "type": "function_call_output",
        "call_id": "call_echo",
        "output": json.dumps({"ok": True, "data": {"echo": "native"}}),
    }


def test_agent_adapter_receives_provider_neutral_registry_for_legacy_tools() -> None:
    """Legacy registries should be normalized before adapter construction."""

    mock_client = MagicMock()
    agent = build_agent(mock_client)

    assert isinstance(agent.tools, ToolRegistry)
    assert agent.tool_adapter.registry is agent.tools.base_registry


def test_agent_adapter_receives_provider_neutral_registry_for_native_tools() -> None:
    """Native registries should be passed directly to the provider adapter."""

    mock_client = MagicMock()
    agent = build_native_agent(mock_client)

    assert isinstance(agent.tools, NativeToolRegistry)
    assert agent.tool_adapter.registry is agent.tools


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
            "output": json.dumps({"ok": True, "data": {"echo": "one"}}),
        },
        {
            "type": "function_call_output",
            "call_id": "call_two",
            "output": json.dumps({"ok": True, "data": {"echo": "two"}}),
        },
        make_message("done"),
    ]


def test_base_agent_unknown_tool_returns_structured_model_visible_error() -> None:
    """Unknown tools should become structured function_call_output failures."""

    tool_call = make_tool_call("missing_tool", {}, "call_missing")
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

    tool_call = make_tool_call("echo", {"text": "hello"}, "call_echo")
    responses = [make_response(tool_call), make_response(make_message("done"))]

    def create_response(history: list[Any]) -> SimpleNamespace:
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
        make_message("done"),
    ]


async def test_responses_tool_loop_runs_async_tool_callbacks() -> None:
    """The reusable loop should support async tool dispatch callbacks."""

    tool_call = make_tool_call("echo", {"text": "async"}, "call_async")
    responses = [make_response(tool_call), make_response(make_message("done"))]

    def create_response(history: list[Any]) -> SimpleNamespace:
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
