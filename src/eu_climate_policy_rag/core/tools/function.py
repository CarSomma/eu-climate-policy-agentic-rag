"""Local function-tool definitions."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from eu_climate_policy_rag.core.tools.providers import SchemaProvider
from eu_climate_policy_rag.core.tools.schema import normalize_openai_schema

InputT = TypeVar("InputT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class ToolExecutionConfig:
    """Optional per-tool execution policy."""

    timeout_seconds: float | None = None
    max_concurrency: int | None = None
    max_retries: int | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            msg = "timeout_seconds must be greater than 0."
            raise ValueError(msg)
        if self.max_concurrency is not None and self.max_concurrency < 1:
            msg = "max_concurrency must be at least 1."
            raise ValueError(msg)
        if self.max_retries is not None and self.max_retries < 0:
            msg = "max_retries must be at least 0."
            raise ValueError(msg)


@dataclass(frozen=True)
class FunctionTool(Generic[InputT, ResultT]):
    """A locally executable function exposed to a model as a tool."""

    name: str
    description: str
    schema_provider: SchemaProvider[InputT]
    handler: Callable[..., ResultT]
    strict: bool = True
    execution: ToolExecutionConfig = field(default_factory=ToolExecutionConfig)

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
