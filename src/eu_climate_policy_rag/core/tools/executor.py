"""Tool execution pipeline."""

import asyncio
from collections.abc import Mapping
from inspect import isawaitable, iscoroutinefunction

from pydantic import ValidationError

from eu_climate_policy_rag.core.tools.context import ToolContext
from eu_climate_policy_rag.core.tools.errors import (
    ToolExecutionError,
    ToolValidationError,
    UnknownToolError,
)
from eu_climate_policy_rag.core.tools.function import FunctionTool
from eu_climate_policy_rag.core.tools.middleware import ToolMiddleware
from eu_climate_policy_rag.core.tools.registry import ToolRegistry
from eu_climate_policy_rag.core.tools.result import ToolResult

ErrorMode = str


class ToolExecutor:
    """Execute registered local function tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        middleware: list[ToolMiddleware] | None = None,
    ) -> None:
        self.registry = registry
        self.middleware = middleware or []

    async def run(
        self,
        name: str,
        args: Mapping[str, object],
        *,
        call_id: str | None = None,
        error_mode: ErrorMode = "return",
    ) -> ToolResult[object]:
        """Validate and execute a registered function tool asynchronously."""

        context = ToolContext(tool_name=name, raw_arguments=args, call_id=call_id)
        tool = self.registry.get_function(name)
        if tool is None:
            return self._handle_error(
                UnknownToolError(f"Unknown tool: {name}"),
                context,
                error_mode,
            )

        dumped_args = self._validate_arguments(tool, args, context, error_mode)
        if isinstance(dumped_args, ToolResult):
            return dumped_args

        try:
            call_args = self._before_call(context, dumped_args)
            if iscoroutinefunction(tool.handler):
                value = tool.handler(**call_args)
            else:
                value = await asyncio.to_thread(tool.handler, **call_args)
            if isawaitable(value):
                value = await value
            value = self._after_call(context, value)
        except Exception as exc:
            return self._handle_error(
                ToolExecutionError(f"Tool execution failed for {name}."),
                context,
                error_mode,
                cause=exc,
            )

        return ToolResult.success(
            tool_name=name,
            value=value,
            call_id=call_id,
            metadata=context.metadata,
        )

    def run_sync(
        self,
        name: str,
        args: Mapping[str, object],
        *,
        call_id: str | None = None,
        error_mode: ErrorMode = "return",
    ) -> ToolResult[object]:
        """Validate and execute a registered function tool synchronously."""

        context = ToolContext(tool_name=name, raw_arguments=args, call_id=call_id)
        tool = self.registry.get_function(name)
        if tool is None:
            return self._handle_error(
                UnknownToolError(f"Unknown tool: {name}"),
                context,
                error_mode,
            )

        dumped_args = self._validate_arguments(tool, args, context, error_mode)
        if isinstance(dumped_args, ToolResult):
            return dumped_args

        try:
            call_args = self._before_call(context, dumped_args)
            if iscoroutinefunction(tool.handler):
                msg = f"Tool {name} cannot run an async handler in sync execution."
                raise ToolExecutionError(msg)
            value = tool.handler(**call_args)
            if isawaitable(value):
                msg = f"Tool {name} returned an awaitable in sync execution."
                raise ToolExecutionError(msg)
            value = self._after_call(context, value)
        except Exception as exc:
            return self._handle_error(
                ToolExecutionError(f"Tool execution failed for {name}."),
                context,
                error_mode,
                cause=exc,
            )

        return ToolResult.success(
            tool_name=name,
            value=value,
            call_id=call_id,
            metadata=context.metadata,
        )

    def _validate_arguments(
        self,
        tool: FunctionTool[object, object],
        args: Mapping[str, object],
        context: ToolContext,
        error_mode: ErrorMode,
    ) -> Mapping[str, object] | ToolResult[object]:
        try:
            current_args = args
            for item in self.middleware:
                current_args = item.before_validate(context, current_args)
            validated = tool.validate_arguments(current_args)
            current_validated: object = validated
            for item in self.middleware:
                current_validated = item.after_validate(context, current_validated)
            return tool.schema_provider.dump_validated(current_validated)
        except ValidationError as exc:
            return self._handle_error(
                ToolValidationError(
                    f"Invalid arguments for {context.tool_name}.",
                    details={"errors": exc.errors()},
                ),
                context,
                error_mode,
            )
        except Exception as exc:
            return self._handle_error(
                ToolValidationError(f"Invalid arguments for {context.tool_name}."),
                context,
                error_mode,
                cause=exc,
            )

    def _before_call(
        self,
        context: ToolContext,
        args: Mapping[str, object],
    ) -> Mapping[str, object]:
        current_args = args
        for item in self.middleware:
            current_args = item.before_call(context, current_args)
        return current_args

    def _after_call(self, context: ToolContext, value: object) -> object:
        current_value = value
        for item in self.middleware:
            current_value = item.after_call(context, current_value)
        return current_value

    @staticmethod
    def _handle_error(
        error: Exception,
        context: ToolContext,
        error_mode: ErrorMode,
        *,
        cause: Exception | None = None,
    ) -> ToolResult[object]:
        if error_mode == "raise":
            if cause is not None:
                raise error from cause
            raise error
        return ToolResult.failure(
            tool_name=context.tool_name,
            error=error,
            call_id=context.call_id,
        )
