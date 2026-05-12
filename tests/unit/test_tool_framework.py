"""Unit tests for the provider-neutral tool framework foundation."""

from collections.abc import Mapping

from pydantic import BaseModel, Field

from eu_climate_policy_rag.core.tools import (
    BuiltinTool,
    FunctionTool,
    PydanticSchemaProvider,
    RawJsonSchemaProvider,
    ToolRegistry,
)


class SearchInput(BaseModel):
    """Input model used by the test function tool."""

    query: str = Field(min_length=1, description="Search query")
    limit: int | None = Field(default=None, ge=1, le=10)


def search_handler(query: str, limit: int | None = None) -> dict[str, object]:
    """Return a small structured search result."""

    return {"query": query, "limit": limit}


def test_function_tool_exports_openai_responses_schema_from_pydantic_provider() -> None:
    """Function tools should export strict OpenAI Responses-compatible schemas."""

    tool = FunctionTool(
        name="search_documents",
        description="Search local documents",
        schema_provider=PydanticSchemaProvider(SearchInput),
        handler=search_handler,
    )

    schema = tool.to_openai_tool()

    assert schema["type"] == "function"
    assert schema["name"] == "search_documents"
    assert schema["description"] == "Search local documents"
    assert schema["strict"] is True
    assert schema["parameters"]["type"] == "object"
    assert schema["parameters"]["additionalProperties"] is False
    assert set(schema["parameters"]["required"]) == {"query", "limit"}
    assert schema["parameters"]["properties"]["query"]["type"] == "string"
    assert schema["parameters"]["properties"]["limit"]["anyOf"] == [
        {"type": "integer", "minimum": 1, "maximum": 10},
        {"type": "null"},
    ]


def test_function_tool_can_use_non_pydantic_schema_provider() -> None:
    """The core FunctionTool should not require Pydantic as the schema source."""

    raw_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to inspect"},
        },
    }

    def validate(args: Mapping[str, object]) -> Mapping[str, object]:
        return {"url": str(args["url"])}

    tool = FunctionTool(
        name="inspect_url",
        description="Inspect a URL",
        schema_provider=RawJsonSchemaProvider(raw_schema, validator=validate),
        handler=lambda url: {"url": url},
    )

    schema = tool.to_openai_tool()
    validated = tool.validate_arguments({"url": "https://example.test"})

    assert schema["parameters"]["additionalProperties"] is False
    assert schema["parameters"]["required"] == ["url"]
    assert validated == {"url": "https://example.test"}


def test_tool_registry_exposes_openai_tools_and_legacy_schemas_alias() -> None:
    """Registry should expose provider-neutral tools through OpenAI export."""

    function_tool = FunctionTool(
        name="search_documents",
        description="Search local documents",
        schema_provider=PydanticSchemaProvider(SearchInput),
        handler=search_handler,
    )
    web_search = BuiltinTool.web_search(user_location={"country": "IT", "city": "Turin"})

    registry = ToolRegistry(function_tools=[function_tool], builtin_tools=[web_search])

    assert registry.openai_tools == registry.schemas
    assert registry.openai_tools[0]["type"] == "function"
    assert registry.openai_tools[0]["strict"] is True
    assert registry.openai_tools[1] == {
        "type": "web_search",
        "user_location": {
            "type": "approximate",
            "country": "IT",
            "city": "Turin",
        },
    }
    assert registry.get_function("search_documents") is function_tool
    assert registry.get("search_documents") is function_tool
    assert registry.get("web_search") is None
    assert registry.is_builtin("web_search") is True

