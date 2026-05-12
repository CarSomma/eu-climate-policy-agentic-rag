"""Unit tests for tool execution, results, and structured errors."""

import asyncio
import json

import pytest
from pydantic import BaseModel, Field

from eu_climate_policy_rag.core.tools import (
    FunctionTool,
    PydanticSchemaProvider,
    ToolExecutionConfig,
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


async def cancellable_add_handler(left: int, right: int) -> dict[str, int]:
    """Async tool that can be cancelled while sleeping."""

    await asyncio.sleep(1)
    return {"sum": left + right}


class TransientToolFailure(RuntimeError):
    """Test exception for retryable handler failures."""


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


@pytest.mark.asyncio
async def test_tool_executor_run_retries_async_handler_failures() -> None:
    """Async execution should retry opt-in handler failures."""

    state = {"attempts": 0}

    async def flaky_add_handler(left: int, right: int) -> dict[str, int]:
        state["attempts"] += 1
        if state["attempts"] == 1:
            raise TransientToolFailure("temporary failure")
        return {"sum": left + right}

    tool = FunctionTool(
        name="flaky_add_numbers",
        description="Flakily add two non-negative integers",
        schema_provider=PydanticSchemaProvider(AddInput),
        handler=flaky_add_handler,
    )
    executor = ToolExecutor(ToolRegistry(function_tools=[tool]), max_retries=1)

    result = await executor.run("flaky_add_numbers", {"left": 5, "right": 6})

    assert result.ok is True
    assert result.value == {"sum": 11}
    assert state["attempts"] == 2


@pytest.mark.asyncio
async def test_tool_executor_run_uses_per_tool_retry_configuration() -> None:
    """Per-tool retry configuration should override the executor default."""

    state = {"attempts": 0}

    async def flaky_add_handler(left: int, right: int) -> dict[str, int]:
        state["attempts"] += 1
        if state["attempts"] == 1:
            raise TransientToolFailure("temporary failure")
        return {"sum": left + right}

    tool = FunctionTool(
        name="flaky_add_numbers",
        description="Flakily add two non-negative integers",
        schema_provider=PydanticSchemaProvider(AddInput),
        handler=flaky_add_handler,
        execution=ToolExecutionConfig(max_retries=1),
    )
    executor = ToolExecutor(ToolRegistry(function_tools=[tool]), max_retries=0)

    result = await executor.run("flaky_add_numbers", {"left": 5, "right": 6})

    assert result.ok is True
    assert result.value == {"sum": 11}
    assert state["attempts"] == 2


@pytest.mark.asyncio
async def test_tool_executor_run_returns_error_after_retry_exhaustion() -> None:
    """Async execution should return a structured error after retries are exhausted."""

    state = {"attempts": 0}

    async def always_fails_handler(left: int, right: int) -> dict[str, int]:
        state["attempts"] += 1
        raise TransientToolFailure("still failing")

    tool = FunctionTool(
        name="failing_add_numbers",
        description="Always fail while adding",
        schema_provider=PydanticSchemaProvider(AddInput),
        handler=always_fails_handler,
    )
    executor = ToolExecutor(ToolRegistry(function_tools=[tool]), max_retries=2)

    result = await executor.run("failing_add_numbers", {"left": 5, "right": 6})

    assert result.ok is False
    assert result.error is not None
    assert result.error.type == "ToolExecutionError"
    assert state["attempts"] == 3


@pytest.mark.asyncio
async def test_tool_executor_run_does_not_retry_validation_errors() -> None:
    """Validation failures should not consume retry attempts."""

    state = {"attempts": 0}

    async def counted_add_handler(left: int, right: int) -> dict[str, int]:
        state["attempts"] += 1
        return {"sum": left + right}

    tool = FunctionTool(
        name="counted_add_numbers",
        description="Count attempts while adding",
        schema_provider=PydanticSchemaProvider(AddInput),
        handler=counted_add_handler,
    )
    executor = ToolExecutor(ToolRegistry(function_tools=[tool]), max_retries=2)

    result = await executor.run("counted_add_numbers", {"left": -1, "right": 6})

    assert result.ok is False
    assert result.error is not None
    assert result.error.type == "ToolValidationError"
    assert state["attempts"] == 0


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
async def test_tool_executor_run_uses_per_tool_timeout_configuration() -> None:
    """Per-tool timeout configuration should override the executor default."""

    tool = FunctionTool(
        name="slow_add_numbers",
        description="Slowly add two non-negative integers",
        schema_provider=PydanticSchemaProvider(AddInput),
        handler=slow_add_handler,
        execution=ToolExecutionConfig(timeout_seconds=0.001),
    )
    executor = ToolExecutor(ToolRegistry(function_tools=[tool]), timeout_seconds=1)

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


@pytest.mark.asyncio
async def test_tool_executor_run_honors_async_concurrency_limit() -> None:
    """Async execution should limit concurrent handler calls."""

    state = {"active": 0, "max_active": 0}

    async def tracked_add_handler(left: int, right: int) -> dict[str, int]:
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        return {"sum": left + right}

    tool = FunctionTool(
        name="tracked_add_numbers",
        description="Track concurrent arithmetic calls",
        schema_provider=PydanticSchemaProvider(AddInput),
        handler=tracked_add_handler,
    )
    executor = ToolExecutor(
        ToolRegistry(function_tools=[tool]),
        max_concurrency=1,
    )

    results = await asyncio.gather(
        executor.run("tracked_add_numbers", {"left": 1, "right": 2}),
        executor.run("tracked_add_numbers", {"left": 3, "right": 4}),
    )

    assert [result.value for result in results] == [{"sum": 3}, {"sum": 7}]
    assert state["max_active"] == 1


@pytest.mark.asyncio
async def test_tool_executor_run_honors_per_tool_concurrency_limit() -> None:
    """Per-tool concurrency configuration should serialize calls to that tool."""

    state = {"active": 0, "max_active": 0}

    async def tracked_add_handler(left: int, right: int) -> dict[str, int]:
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        return {"sum": left + right}

    tool = FunctionTool(
        name="tracked_add_numbers",
        description="Track concurrent arithmetic calls",
        schema_provider=PydanticSchemaProvider(AddInput),
        handler=tracked_add_handler,
        execution=ToolExecutionConfig(max_concurrency=1),
    )
    executor = ToolExecutor(ToolRegistry(function_tools=[tool]))

    results = await asyncio.gather(
        executor.run("tracked_add_numbers", {"left": 1, "right": 2}),
        executor.run("tracked_add_numbers", {"left": 3, "right": 4}),
    )

    assert [result.value for result in results] == [{"sum": 3}, {"sum": 7}]
    assert state["max_active"] == 1


@pytest.mark.asyncio
async def test_tool_executor_run_propagates_cancellation() -> None:
    """Async cancellation should not be converted into a tool error result."""

    tool = FunctionTool(
        name="cancellable_add_numbers",
        description="Cancellable arithmetic call",
        schema_provider=PydanticSchemaProvider(AddInput),
        handler=cancellable_add_handler,
    )
    executor = ToolExecutor(ToolRegistry(function_tools=[tool]))

    task = asyncio.create_task(
        executor.run("cancellable_add_numbers", {"left": 1, "right": 2})
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_tool_executor_releases_concurrency_limit_after_cancellation() -> None:
    """Cancelled calls should release executor concurrency slots."""

    tool = FunctionTool(
        name="cancellable_add_numbers",
        description="Cancellable arithmetic call",
        schema_provider=PydanticSchemaProvider(AddInput),
        handler=cancellable_add_handler,
    )
    executor = ToolExecutor(
        ToolRegistry(function_tools=[tool]),
        max_concurrency=1,
    )

    task = asyncio.create_task(
        executor.run("cancellable_add_numbers", {"left": 1, "right": 2})
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    result = await executor.run("cancellable_add_numbers", {"left": 3, "right": 4}, timeout_seconds=0.001)

    assert result.ok is False
    assert result.error is not None
    assert result.error.type == "ToolExecutionError"
