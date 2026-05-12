"""Local function-tool definitions."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from eu_climate_policy_rag.core.tools.providers import SchemaProvider
from eu_climate_policy_rag.core.tools.schema import normalize_openai_schema

InputT = TypeVar("InputT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class FunctionTool(Generic[InputT, ResultT]):
    """A locally executable function exposed to a model as a tool."""

    name: str
    description: str
    schema_provider: SchemaProvider[InputT]
    handler: Callable[..., ResultT]
    strict: bool = True

    def to_openai_tool(self) -> dict[str, object]:
        """Return this function tool in OpenAI Responses API format."""

        parameters = normalize_openai_schema(self.schema_provider.json_schema())
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
            "strict": self.strict,
        }

    @property
    def schema(self) -> dict[str, object]:
        """Compatibility alias for OpenAI Responses API schema export."""

        return self.to_openai_tool()

    def validate_arguments(self, args: Mapping[str, object]) -> InputT:
        """Validate raw tool arguments through the configured schema provider."""

        return self.schema_provider.validate(args)
