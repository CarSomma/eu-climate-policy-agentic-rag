"""Tool registry for model-visible tools."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field

from eu_climate_policy_rag.core.tools.builtin import BuiltinTool
from eu_climate_policy_rag.core.tools.function import FunctionTool
from eu_climate_policy_rag.core.tools.middleware import ToolMiddleware


@dataclass(frozen=True)
class ToolRegistry:
    """Immutable catalog of local function tools and provider built-ins."""

    function_tools: Sequence[FunctionTool[object, object]] = field(default_factory=tuple)
    builtin_tools: Sequence[BuiltinTool | Mapping[str, object]] = field(default_factory=tuple)
    middleware: Sequence[ToolMiddleware] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        function_map = {tool.name: tool for tool in self.function_tools}
        if len(function_map) != len(self.function_tools):
            msg = "Function tool names must be unique."
            raise ValueError(msg)

        builtins = tuple(
            tool if isinstance(tool, BuiltinTool) else BuiltinTool.from_mapping(tool)
            for tool in self.builtin_tools
        )

        object.__setattr__(self, "_function_map", function_map)
        object.__setattr__(self, "_builtins", builtins)
        object.__setattr__(self, "_builtin_types", {tool.type for tool in builtins})

    @property
    def openai_tools(self) -> list[dict[str, object]]:
        """Return all tools in OpenAI Responses API format."""

        return [
            *[tool.to_openai_tool() for tool in self.function_tools],
            *[tool.to_openai_tool() for tool in self._builtins],
        ]

    @property
    def schemas(self) -> list[dict[str, object]]:
        """Compatibility alias for OpenAI Responses API tool schemas."""

        return self.openai_tools

    @property
    def function_names(self) -> tuple[str, ...]:
        """Return registered local function tool names."""

        return tuple(self._function_map)

    @property
    def builtin_types(self) -> tuple[str, ...]:
        """Return registered provider built-in tool types."""

        return tuple(self._builtin_types)

    @property
    def builtins(self) -> tuple[BuiltinTool, ...]:
        """Return normalized provider built-in tool configs."""

        return tuple(self._builtins)

    def get_function(self, name: str) -> FunctionTool[object, object] | None:
        """Return a local function tool by name."""

        return self._function_map.get(name)

    def get(self, name: str) -> FunctionTool[object, object] | None:
        """Compatibility alias for local function lookup."""

        return self.get_function(name)

    def is_builtin(self, name: str) -> bool:
        """Return whether a tool type is registered as provider-managed."""

        return name in self._builtin_types

    def describe(self) -> dict[str, list[dict[str, object]]]:
        """Return a serializable provider-neutral summary of registered tools."""

        return {
            "function_tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "strict": tool.strict,
                }
                for tool in self.function_tools
            ],
            "builtin_tools": [
                {
                    "type": tool.type,
                    "config": deepcopy(dict(tool.config)),
                }
                for tool in self._builtins
            ],
        }
