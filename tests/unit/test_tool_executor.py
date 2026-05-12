"""Unit tests for tool execution, results, and structured errors."""

import json

import pytest
from pydantic import BaseModel, Field

from eu_climate_policy_rag.core.tools import (
    FunctionTool,
    PydanticSchemaProvider,
    ToolExecutor,
    ToolRegistry,
)
from eu_climate_policy_rag.core.tools.errors import (
    ToolValidationError,
    UnknownToolError,
)


class AddInput(BaseModel):
    """Arguments for a small arithmetic tool."""

    left: int = Field(ge=0)
    right: int = Field(ge=0)


def add_handler(left: int, right: int) -> dict[str, int]:
    """Add two integers."""

    return {"sum": left + right}


def build_executor() -> ToolExecutor:
    """Build an executor with one test function tool."""

    tool = FunctionTool(
        name="add_numbers",
        description="Add two non-negative integers",
        schema_provider=PydanticSchemaProvider(AddInput),
        handler=add_handler,
    )
    return ToolExecutor(ToolRegistry(function_tools=[tool]))


def test_tool_executor_run_sync_returns_structured_success_result() -> None:
    """Sync execution should validate, call the handler, and serialize output."""

    executor = build_executor()

    result = executor.run_sync("add_numbers", {"left": 2, "right": 3}, call_id="call_1")

    assert result.ok is True
    assert result.tool_name == "add_numbers"
    assert result.call_id == "call_1"
    assert result.value == {"sum": 5}
    assert json.loads(result.output) == {"ok": True, "data": {"sum": 5}}
    assert result.to_responses_output() == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": result.output,
    }


def test_tool_executor_run_sync_returns_structured_unknown_tool_error() -> None:
    """Unknown tools should be model-visible in return-error mode."""

    executor = build_executor()

    result = executor.run_sync("missing_tool", {}, call_id="call_2")

    assert result.ok is False
    assert result.error is not None
    assert result.error.type == "UnknownToolError"
    assert result.error.retryable is False
    assert "missing_tool" in result.output


def test_tool_executor_can_raise_unknown_tool_errors() -> None:
    """Direct developer calls should be able to raise instead of serialize."""

    executor = build_executor()

    with pytest.raises(UnknownToolError):
        executor.run_sync("missing_tool", {}, error_mode="raise")


def test_tool_executor_wraps_validation_errors() -> None:
    """Invalid arguments should become ToolValidationError payloads."""

    executor = build_executor()

    result = executor.run_sync("add_numbers", {"left": -1, "right": 3})

    assert result.ok is False
    assert result.error is not None
    assert result.error.type == "ToolValidationError"
    assert "add_numbers" in result.error.message


def test_tool_executor_can_raise_validation_errors() -> None:
    """Validation errors should raise in raise-error mode."""

    executor = build_executor()

    with pytest.raises(ToolValidationError):
        executor.run_sync("add_numbers", {"left": -1, "right": 3}, error_mode="raise")

