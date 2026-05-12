"""Per-call tool execution context."""

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass
class ToolContext:
    """Metadata for one tool execution."""

    tool_name: str
    raw_arguments: Mapping[str, object]
    call_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    attempt: int = 1
