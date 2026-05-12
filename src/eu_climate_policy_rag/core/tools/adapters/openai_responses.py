"""OpenAI Responses API adapter for provider-neutral tool registries."""

from eu_climate_policy_rag.core.tools.registry import ToolRegistry
from eu_climate_policy_rag.core.tools.result import ToolResult


class OpenAIResponsesToolAdapter:
    """Compile registry tools and results into OpenAI Responses API shapes."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    @property
    def tools(self) -> list[dict[str, object]]:
        """Return all registry tools in OpenAI Responses API format."""

        return [
            *[tool.to_openai_tool() for tool in self.registry.function_tools],
            *[tool.to_openai_tool() for tool in self.registry.builtins],
        ]

    @property
    def openai_tools(self) -> list[dict[str, object]]:
        """Compatibility alias for OpenAI Responses API tool configs."""

        return self.tools

    @property
    def schemas(self) -> list[dict[str, object]]:
        """Compatibility alias for OpenAI Responses API tool configs."""

        return self.tools

    @staticmethod
    def to_function_call_output(result: ToolResult[object]) -> dict[str, object]:
        """Convert a structured tool result to a Responses function_call_output."""

        return result.to_responses_output()
