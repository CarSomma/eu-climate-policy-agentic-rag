"""Unit tests for tool execution, results, and structured errors."""

import asyncio
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
    ToolExecutionError,
    ToolValidationError,
    UnknownToolError,
)


class AddInput(BaseModel):
    """Arguments for a small arithmetic tool."""

    left: int = Field(ge=0)
    right: int = Field(ge=0)


class AddResult(BaseModel):
    """Pydantic result model for serialization tests."""

    total: int


def add_handler(left: int, right: int) -> dict[str, int]:
    """Add two integers."""

    return {"sum": left + right}


async def async_add_handler(left: int, right: int) -> dict[str, int]:
    """Async variant of the arithmetic tool."""

    return {"sum": left + right}


async def slow_add_handler(left: int, right: int) -> dict[str, int]:
    """Async tool that exceeds a tiny timeout."""

    await asyncio.sleep(0.05)
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


def test_tool_executor_run_sync_rejects_async_handlers() -> None:
    """Sync execution should not silently run async handlers."""

    tool = FunctionTool(
        name="async_add_numbers",
        description="Add two non-negative integers asynchronously",
        schema_provider=PydanticSchemaProvider(AddInput),
        handler=async_add_handler,
    )
    executor = ToolExecutor(ToolRegistry(function_tools=[tool]))

    result = executor.run_sync("async_add_numbers", {"left": 2, "right": 3})

    assert result.ok is False
    assert result.error is not None
    assert result.error.type == "ToolExecutionError"


def test_tool_executor_serializes_pydantic_result_models() -> None:
    """ToolResult should serialize Pydantic handler results safely."""

    tool = FunctionTool(
        name="add_model",
        description="Add two non-negative integers",
        schema_provider=PydanticSchemaProvider(AddInput),
        handler=lambda left, right: AddResult(total=left + right),
    )
    executor = ToolExecutor(ToolRegistry(function_tools=[tool]))

    result = executor.run_sync("add_model", {"left": 2, "right": 8})

    assert result.ok is True
    assert isinstance(result.value, AddResult)
    assert json.loads(result.output) == {"ok": True, "data": {"total": 10}}


@pytest.mark.asyncio
async def test_tool_executor_run_awaits_async_handlers() -> None:
    """Async execution should await async tool handlers."""

    tool = FunctionTool(
        name="async_add_numbers",
        description="Add two non-negative integers asynchronously",
        schema_provider=PydanticSchemaProvider(AddInput),
        handler=async_add_handler,
    )
    executor = ToolExecutor(ToolRegistry(function_tools=[tool]))

    result = await executor.run(
        "async_add_numbers",
        {"left": 2, "right": 4},
        call_id="call_async",
    )

    assert result.ok is True
    assert result.value == {"sum": 6}
    assert result.to_responses_output()["call_id"] == "call_async"


@pytest.mark.asyncio
async def test_tool_executor_run_supports_sync_handlers() -> None:
    """Async execution should support existing synchronous handlers."""

    executor = build_executor()

    result = await executor.run("add_numbers", {"left": 4, "right": 5})

    assert result.ok is True
    assert result.value == {"sum": 9}


@pytest.mark.asyncio
async def test_tool_executor_run_returns_validation_errors() -> None:
    """Async execution should preserve structured validation errors."""

    executor = build_executor()

    result = await executor.run("add_numbers", {"left": -1, "right": 5})

    assert result.ok is False
    assert result.error is not None
    assert result.error.type == "ToolValidationError"


@pytest.mark.asyncio
async def test_tool_executor_run_can_raise_unknown_tool_errors() -> None:
    """Async execution should support raise-error mode."""

    executor = build_executor()

    with pytest.raises(UnknownToolError):
        await executor.run("missing_tool", {}, error_mode="raise")


@pytest.mark.asyncio
async def test_tool_executor_run_returns_timeout_error() -> None:
    """Async execution should return structured timeout errors."""

    tool = FunctionTool(
        name="slow_add_numbers",
        description="Slowly add two non-negative integers",
        schema_provider=PydanticSchemaProvider(AddInput),
        handler=slow_add_handler,
    )
    executor = ToolExecutor(ToolRegistry(function_tools=[tool]), timeout_seconds=0.001)

    result = await executor.run("slow_add_numbers", {"left": 1, "right": 2})

    assert result.ok is False
    assert result.error is not None
    assert result.error.type == "ToolExecutionError"
    assert "timed out" in result.error.message


@pytest.mark.asyncio
async def test_tool_executor_run_can_raise_timeout_errors() -> None:
    """Async execution should raise timeout errors in raise-error mode."""

    tool = FunctionTool(
        name="slow_add_numbers",
        description="Slowly add two non-negative integers",
        schema_provider=PydanticSchemaProvider(AddInput),
        handler=slow_add_handler,
    )
    executor = ToolExecutor(ToolRegistry(function_tools=[tool]), timeout_seconds=0.001)

    with pytest.raises(ToolExecutionError, match="timed out"):
        await executor.run(
            "slow_add_numbers",
            {"left": 1, "right": 2},
            error_mode="raise",
        )
