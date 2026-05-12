"""Compatibility helpers for OpenAI function-tool definitions."""

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel

from eu_climate_policy_rag.core.tools import (
    FunctionTool,
    PydanticSchemaProvider,
    ToolExecutor,
    ToolMiddleware,
    ToolRegistry as BaseToolRegistry,
)


class OpenAIFunctionTool(FunctionTool[BaseModel, Any]):
    """Bind an OpenAI function-tool schema to a Python handler.

    This class preserves the historical `core.tooling` API while delegating
    schema export and validation to the provider-neutral tool framework.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        handler: Callable[..., Any],
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            schema_provider=PydanticSchemaProvider(input_model),
            handler=handler,
        )
        self.input_model = input_model

    @property
    def schema(self) -> dict[str, Any]:
        """Return the OpenAI Responses API tool schema."""

        return self.to_openai_tool()

    async def run(self, args: dict[str, Any]) -> Any:
        """Validate arguments and call the bound handler."""

        registry = BaseToolRegistry(function_tools=[self])
        result = await ToolExecutor(registry).run(self.name, args, error_mode="raise")
        return result.value

    def run_sync(self, args: dict[str, Any]) -> Any:
        """Validate arguments and call a synchronous bound handler."""

        registry = BaseToolRegistry(function_tools=[self])
        result = ToolExecutor(registry).run_sync(self.name, args, error_mode="raise")
        return result.value


class ToolRegistry:
    """Lookup and dispatch a set of named function tools.

    Supports both custom function tools and built-in OpenAI tools
    (web_search, code_interpreter) for the Responses API.
    """

    def __init__(
        self,
        tools: Sequence[OpenAIFunctionTool] | None = None,
        function_tools: Sequence[OpenAIFunctionTool] | None = None,
        builtin_tools: list[dict[str, Any]] | None = None,
        middleware: Sequence[ToolMiddleware] | None = None,
        timeout_seconds: float | None = None,
        max_concurrency: int | None = None,
        max_retries: int = 0,
    ) -> None:
        """Initialize with custom function tools and optional built-in tools.

        Args:
            tools: Legacy parameter for backward compatibility
            function_tools: Custom OpenAIFunctionTool instances
            builtin_tools: Built-in tool schemas (web_search, code_interpreter)
        """
        # Backward compatibility: support old ToolRegistry([tool1, tool2])
        if tools is not None and function_tools is None:
            function_tools = tools

        function_tools = function_tools or []
        self._registry = BaseToolRegistry(
            function_tools=function_tools,
            builtin_tools=builtin_tools or [],
        )
        self._executor = ToolExecutor(
            self._registry,
            middleware=list(middleware or []),
            timeout_seconds=timeout_seconds,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
        )

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """Return schemas for both function and built-in tools."""
        return self._registry.schemas

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        """Return schemas for both function and built-in tools."""

        return self._registry.openai_tools

    def get(self, name: str) -> OpenAIFunctionTool | None:
        """Return a custom function tool by name if registered."""
        tool = self._registry.get(name)
        return tool if isinstance(tool, OpenAIFunctionTool) else None

    def is_builtin(self, name: str) -> bool:
        """Check if a tool name refers to a built-in tool."""
        return self._registry.is_builtin(name)

    async def run(self, name: str, args: dict[str, Any]) -> Any:
        """Run a registered tool by name."""
        if self.get(name) is None:
            return {"error": f"Unknown tool: {name}"}
        result = await self._executor.run(name, args, error_mode="raise")
        return result.value

    def run_sync(self, name: str, args: dict[str, Any]) -> Any:
        """Run a registered synchronous tool by name."""
        if self.get(name) is None:
            return {"error": f"Unknown tool: {name}"}
        result = self._executor.run_sync(name, args, error_mode="raise")
        return result.value
