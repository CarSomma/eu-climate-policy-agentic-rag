"""Tool execution pipeline."""

from collections.abc import Mapping

from pydantic import ValidationError

from eu_climate_policy_rag.core.tools.context import ToolContext
from eu_climate_policy_rag.core.tools.errors import (
    ToolExecutionError,
    ToolValidationError,
    UnknownToolError,
)
from eu_climate_policy_rag.core.tools.registry import ToolRegistry
from eu_climate_policy_rag.core.tools.result import ToolResult

ErrorMode = str


class ToolExecutor:
    """Execute registered local function tools."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

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

        try:
            validated = tool.validate_arguments(args)
            dumped_args = tool.schema_provider.dump_validated(validated)
        except ValidationError as exc:
            return self._handle_error(
                ToolValidationError(
                    f"Invalid arguments for {name}.",
                    details={"errors": exc.errors()},
                ),
                context,
                error_mode,
            )
        except Exception as exc:
            return self._handle_error(
                ToolValidationError(f"Invalid arguments for {name}."),
                context,
                error_mode,
                cause=exc,
            )

        try:
            value = tool.handler(**dumped_args)
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
        )

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
