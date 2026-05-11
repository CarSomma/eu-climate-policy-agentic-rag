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
    """Lookup and dispatch a set of named function tools."""

    def __init__(self, tools: Sequence[OpenAIFunctionTool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """Return schemas for every registered tool."""

        return [tool.schema for tool in self._tools.values()]

    def get(self, name: str) -> OpenAIFunctionTool | None:
        """Return a tool by name if registered."""

        return self._tools.get(name)

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
