"""Small helpers for OpenAI function-tool definitions."""

from collections.abc import Callable, Sequence
from inspect import isawaitable
from typing import Any

from pydantic import BaseModel


class OpenAIFunctionTool:
    """Bind an OpenAI function-tool schema to a Python handler."""

    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        handler: Callable[..., Any],
    ) -> None:
        self.name = name
        self.description = description
        self.input_model = input_model
        self.handler = handler

    @property
    def schema(self) -> dict[str, Any]:
        """Return the OpenAI Responses API tool schema."""

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.input_model.model_json_schema(),
        }

    async def run(self, args: dict[str, Any]) -> Any:
        """Validate arguments and call the bound handler."""

        validated_args = self.input_model.model_validate(args).model_dump()
        result = self.handler(**validated_args)
        if isawaitable(result):
            return await result
        return result

    def run_sync(self, args: dict[str, Any]) -> Any:
        """Validate arguments and call a synchronous bound handler."""

        validated_args = self.input_model.model_validate(args).model_dump()
        return self.handler(**validated_args)


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
        self._tools = {tool.name: tool for tool in function_tools}
        self._builtin_schemas = builtin_tools or []

        # Extract built-in tool names for quick lookup
        self._builtin_names = {
            tool.get("type", tool.get("name", ""))
            for tool in self._builtin_schemas
        }

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """Return schemas for both function and built-in tools."""
        return [
            *[tool.schema for tool in self._tools.values()],
            *self._builtin_schemas,
        ]

    def get(self, name: str) -> OpenAIFunctionTool | None:
        """Return a custom function tool by name if registered."""
        return self._tools.get(name)

    def is_builtin(self, name: str) -> bool:
        """Check if a tool name refers to a built-in tool."""
        return name in self._builtin_names

    async def run(self, name: str, args: dict[str, Any]) -> Any:
        """Run a registered tool by name."""
        tool = self.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}
        return await tool.run(args)

    def run_sync(self, name: str, args: dict[str, Any]) -> Any:
        """Run a registered synchronous tool by name."""
        tool = self.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}
        return tool.run_sync(args)
