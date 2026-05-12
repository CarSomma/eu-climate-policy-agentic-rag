"""Schema provider abstractions for tool inputs."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

InputT = TypeVar("InputT")


@runtime_checkable
class SchemaProvider(Protocol[InputT]):
    """Expose an input schema and runtime argument validation."""

    def json_schema(self) -> Mapping[str, object]:
        """Return a JSON Schema-like mapping for the tool input."""

    def validate(self, raw_args: Mapping[str, object]) -> InputT:
        """Validate raw tool arguments."""

    def dump_validated(self, input_obj: InputT) -> Mapping[str, object]:
        """Dump a validated input object into a mapping."""


class PydanticSchemaProvider(Generic[InputT]):
    """Schema provider backed by a Pydantic v2 model."""

    def __init__(self, model: type[BaseModel]) -> None:
        self.model = model

    def json_schema(self) -> Mapping[str, object]:
        """Return the Pydantic validation JSON Schema."""

        return self.model.model_json_schema(mode="validation")

    def validate(self, raw_args: Mapping[str, object]) -> InputT:
        """Validate arguments with the configured Pydantic model."""

        return self.model.model_validate(dict(raw_args))  # type: ignore[return-value]

    def dump_validated(self, input_obj: InputT) -> Mapping[str, object]:
        """Dump a validated Pydantic object into a plain mapping."""

        if isinstance(input_obj, BaseModel):
            return input_obj.model_dump()
        return dict(input_obj)  # type: ignore[arg-type]


class RawJsonSchemaProvider:
    """Schema provider for non-Pydantic tool definitions."""

    def __init__(
        self,
        schema: Mapping[str, object],
        *,
        validator: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    ) -> None:
        self._schema = deepcopy(dict(schema))
        self._validator = validator

    def json_schema(self) -> Mapping[str, object]:
        """Return the configured raw JSON Schema."""

        return deepcopy(self._schema)

    def validate(self, raw_args: Mapping[str, object]) -> Mapping[str, object]:
        """Validate raw arguments with the optional custom validator."""

        if self._validator is None:
            return dict(raw_args)
        return self._validator(raw_args)

    def dump_validated(
        self,
        input_obj: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Dump validated raw arguments."""

        return dict(input_obj)
