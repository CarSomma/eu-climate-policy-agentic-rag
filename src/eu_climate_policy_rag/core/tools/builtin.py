"""OpenAI built-in tool configuration objects."""

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BuiltinTool:
    """Model-visible tool executed by the LLM provider, not local Python."""

    type: str
    config: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def web_search(
        cls,
        *,
        user_location: Mapping[str, str] | None = None,
        external_web_access: bool | None = None,
    ) -> "BuiltinTool":
        """Create an OpenAI Responses `web_search` tool configuration."""

        config: dict[str, object] = {}
        if user_location is not None:
            config["user_location"] = {"type": "approximate", **dict(user_location)}
        if external_web_access is not None:
            config["external_web_access"] = external_web_access
        return cls(type="web_search", config=config)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "BuiltinTool":
        """Create a built-in tool from an existing provider config mapping."""

        tool_type = value.get("type")
        if not isinstance(tool_type, str) or not tool_type:
            msg = "Built-in tool mappings must include a non-empty string 'type'."
            raise ValueError(msg)
        return cls(
            type=tool_type,
            config={key: item for key, item in value.items() if key != "type"},
        )

    def to_openai_tool(self) -> dict[str, object]:
        """Return this built-in tool in OpenAI Responses API format."""

        return {"type": self.type, **dict(self.config)}
