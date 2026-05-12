"""OpenAI Responses API adapter for provider-neutral tool registries."""

from eu_climate_policy_rag.core.tools.function import FunctionTool
from eu_climate_policy_rag.core.tools.registry import ToolRegistry
from eu_climate_policy_rag.core.tools.result import ToolResult
from eu_climate_policy_rag.core.tools.schema import normalize_openai_schema


class OpenAIResponsesSchemaCompiler:
    """Compile provider-neutral tools into OpenAI Responses API schemas."""

    def compile_function_tool(
        self,
        tool: FunctionTool[object, object],
    ) -> dict[str, object]:
        """Return a function tool config in OpenAI Responses API format."""

        parameters = normalize_openai_schema(tool.schema_provider.json_schema())
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
            "strict": tool.strict,
        }


class OpenAIResponsesToolAdapter:
    """Compile registry tools and results into OpenAI Responses API shapes."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        schema_compiler: OpenAIResponsesSchemaCompiler | None = None,
    ) -> None:
        self.registry = registry
        self.schema_compiler = schema_compiler or OpenAIResponsesSchemaCompiler()

    @property
    def tools(self) -> list[dict[str, object]]:
        """Return all registry tools in OpenAI Responses API format."""

        return [
            *[
                self.schema_compiler.compile_function_tool(tool)
                for tool in self.registry.function_tools
            ],
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
