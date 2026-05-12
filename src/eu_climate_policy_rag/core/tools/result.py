"""Tool execution results and serialization helpers."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel

from eu_climate_policy_rag.core.tools.errors import ToolErrorPayload

ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class ToolResult(Generic[ResultT]):
    """Structured result of one tool execution."""

    ok: bool
    tool_name: str
    output: str
    call_id: str | None = None
    value: ResultT | None = None
    error: ToolErrorPayload | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    duration_ms: float | None = None

    @classmethod
    def success(
        cls,
        *,
        tool_name: str,
        value: ResultT,
        call_id: str | None = None,
    ) -> "ToolResult[ResultT]":
        """Create a successful JSON-mode tool result."""

        output = json.dumps({"ok": True, "data": _to_jsonable(value)})
        return cls(
            ok=True,
            tool_name=tool_name,
            call_id=call_id,
            value=value,
            output=output,
        )

    @classmethod
    def failure(
        cls,
        *,
        tool_name: str,
        error: BaseException,
        call_id: str | None = None,
    ) -> "ToolResult[object]":
        """Create a failed JSON-mode tool result."""

        payload = ToolErrorPayload.from_exception(error)
        output = json.dumps({"ok": False, "error": payload.to_dict()})
        return cls(
            ok=False,
            tool_name=tool_name,
            call_id=call_id,
            error=payload,
            output=output,
        )

    def to_responses_output(self) -> dict[str, object]:
        """Return an OpenAI Responses `function_call_output` item."""

        if self.call_id is None:
            msg = "call_id is required to build a Responses function_call_output."
            raise ValueError(msg)
        return {
            "type": "function_call_output",
            "call_id": self.call_id,
            "output": self.output,
        }


def _to_jsonable(value: object) -> object:
    """Convert common structured Python values into JSON-serializable values."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_to_jsonable(item) for item in value]
    return value
