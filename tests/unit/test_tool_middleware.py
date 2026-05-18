"""Unit tests for tool executor middleware hooks."""

from collections.abc import Mapping

import pytest
from pydantic import BaseModel, Field

from eu_climate_policy_rag.core.tools import (
    FunctionTool,
    PydanticSchemaProvider,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
)
from eu_climate_policy_rag.core.tools.middleware import ToolMiddleware
from eu_climate_policy_rag.core.tools.middleware import ToolMetricsMiddleware


class AddInput(BaseModel):
    """Arguments for a small arithmetic tool."""

    left: int = Field(ge=0)
    right: int = Field(ge=0)


def add_handler(left: int, right: int) -> dict[str, int]:
    """Add two integers."""

    return {"sum": left + right}


def build_tool() -> FunctionTool[AddInput, dict[str, int]]:
    """Build one test function tool."""

    return FunctionTool(
        name="add_numbers",
        description="Add two non-negative integers",
        schema_provider=PydanticSchemaProvider(AddInput),
        handler=add_handler,
    )


class RecordingMiddleware(ToolMiddleware):
    """Middleware that records sync lifecycle events."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def before_validate(
        self,
        context: ToolContext,
        args: Mapping[str, object],
    ) -> Mapping[str, object]:
        self.events.append(f"before_validate:{context.tool_name}")
        return args

    def after_validate(self, context: ToolContext, validated: object) -> object:
        self.events.append(f"after_validate:{context.tool_name}")
        return validated

    def before_call(
        self,
        context: ToolContext,
        call_args: Mapping[str, object],
    ) -> Mapping[str, object]:
        self.events.append(f"before_call:{context.tool_name}")
        return call_args

    def after_call(self, context: ToolContext, result: object) -> object:
        self.events.append(f"after_call:{context.tool_name}")
        context.metadata["observed"] = True
        return result


class ArgumentInjectionMiddleware(ToolMiddleware):
    """Middleware that can inject missing arguments before validation."""

    def before_validate(
        self,
        context: ToolContext,
        args: Mapping[str, object],
    ) -> Mapping[str, object]:
        updated = dict(args)
        updated.setdefault("right", 10)
        return updated


def test_tool_executor_runs_sync_middleware_hooks_in_order() -> None:
    """Sync execution should run middleware around validation and calls."""

    middleware = RecordingMiddleware()
    executor = ToolExecutor(
        ToolRegistry(function_tools=[build_tool()]),
        middleware=[middleware],
    )

    result = executor.run_sync("add_numbers", {"left": 2, "right": 3})

    assert result.ok is True
    assert result.value == {"sum": 5}
    assert result.metadata["observed"] is True
    assert middleware.events == [
        "before_validate:add_numbers",
        "after_validate:add_numbers",
        "before_call:add_numbers",
        "after_call:add_numbers",
    ]


def test_tool_executor_middleware_can_inject_arguments_before_validation() -> None:
    """Middleware should support repo use cases like fetch argument injection."""

    executor = ToolExecutor(
        ToolRegistry(function_tools=[build_tool()]),
        middleware=[ArgumentInjectionMiddleware()],
    )

    result = executor.run_sync("add_numbers", {"left": 5})

    assert result.ok is True
    assert result.value == {"sum": 15}


async def async_add_handler(left: int, right: int) -> dict[str, int]:
    """Async arithmetic handler for middleware tests."""

    return {"sum": left + right}


@pytest.mark.asyncio
async def test_tool_executor_runs_middleware_for_async_execution() -> None:
    """Async execution should use the same middleware lifecycle."""

    middleware = RecordingMiddleware()
    tool = FunctionTool(
        name="async_add_numbers",
        description="Add two non-negative integers asynchronously",
        schema_provider=PydanticSchemaProvider(AddInput),
        handler=async_add_handler,
    )
    executor = ToolExecutor(
        ToolRegistry(function_tools=[tool]),
        middleware=[middleware],
    )

    result = await executor.run("async_add_numbers", {"left": 1, "right": 6})

    assert result.ok is True
    assert result.value == {"sum": 7}
    assert result.metadata["observed"] is True
    assert middleware.events == [
        "before_validate:async_add_numbers",
        "after_validate:async_add_numbers",
        "before_call:async_add_numbers",
        "after_call:async_add_numbers",
    ]


def test_tool_metrics_middleware_records_success_metadata() -> None:
    """Metrics middleware should record successful tool execution metadata."""

    events = []
    middleware = ToolMetricsMiddleware(events.append)
    executor = ToolExecutor(
        ToolRegistry(function_tools=[build_tool()]),
        middleware=[middleware],
    )

    result = executor.run_sync("add_numbers", {"left": 2, "right": 3})

    assert result.ok is True
    assert result.metadata["metrics"]["tool_name"] == "add_numbers"
    assert result.metadata["metrics"]["attempt"] == 1
    assert result.metadata["metrics"]["ok"] is True
    assert result.metadata["metrics"]["duration_ms"] >= 0
    assert events == [result.metadata["metrics"]]


def test_tool_metrics_middleware_records_validation_errors() -> None:
    """Metrics middleware should observe validation failures."""

    events = []
    middleware = ToolMetricsMiddleware(events.append)
    executor = ToolExecutor(
        ToolRegistry(function_tools=[build_tool()]),
        middleware=[middleware],
    )

    result = executor.run_sync("add_numbers", {"left": -1, "right": 3})

    assert result.ok is False
    assert result.metadata["metrics"]["tool_name"] == "add_numbers"
    assert result.metadata["metrics"]["ok"] is False
    assert result.metadata["metrics"]["error_type"] == "ToolValidationError"
    assert events == [result.metadata["metrics"]]
