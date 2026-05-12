"""Unit tests for ToolRegistry and OpenAIFunctionTool."""

import pytest
from pydantic import BaseModel, Field

from eu_climate_policy_rag.core.tools import ToolExecutionConfig
from eu_climate_policy_rag.core.tooling import OpenAIFunctionTool, ToolRegistry


class MockSearchInput(BaseModel):
    """Mock input model for testing."""

    query: str = Field(min_length=1)


def mock_search_handler(query: str) -> str:
    """Mock search function."""
    return f"Results for: {query}"


def test_tool_registry_accepts_builtin_tools() -> None:
    """ToolRegistry should accept built-in tool schemas (web_search, code_interpreter)."""
    custom_tool = OpenAIFunctionTool(
        name="search_documents",
        description="Search local documents",
        input_model=MockSearchInput,
        handler=mock_search_handler,
    )

    builtin_tools = [
        {
            "type": "web_search",
            "user_location": {
                "type": "approximate",
                "country": "GB",
                "city": "London",
            },
        }
    ]

    registry = ToolRegistry(function_tools=[custom_tool], builtin_tools=builtin_tools)

    assert len(registry.schemas) == 2
    assert registry.schemas[0]["type"] == "function"
    assert registry.schemas[0]["name"] == "search_documents"
    assert registry.schemas[1]["type"] == "web_search"
    assert registry.schemas[1]["user_location"]["country"] == "GB"


def test_tool_registry_builtin_tools_optional() -> None:
    """ToolRegistry should work without built-in tools (backward compatible)."""
    custom_tool = OpenAIFunctionTool(
        name="search_documents",
        description="Search local documents",
        input_model=MockSearchInput,
        handler=mock_search_handler,
    )

    # Should work with old API (no builtin_tools parameter)
    registry = ToolRegistry(function_tools=[custom_tool])

    assert len(registry.schemas) == 1
    assert registry.schemas[0]["type"] == "function"


def test_tool_registry_backward_compatible_constructor() -> None:
    """ToolRegistry should accept old constructor signature for backward compatibility."""
    custom_tool = OpenAIFunctionTool(
        name="search_documents",
        description="Search local documents",
        input_model=MockSearchInput,
        handler=mock_search_handler,
    )

    # Old API: ToolRegistry([tool1, tool2])
    registry = ToolRegistry([custom_tool])

    assert len(registry.schemas) == 1
    assert registry.schemas[0]["name"] == "search_documents"


def test_tool_registry_checks_if_tool_is_builtin() -> None:
    """ToolRegistry should distinguish between custom and built-in tools."""
    custom_tool = OpenAIFunctionTool(
        name="search_documents",
        description="Search local documents",
        input_model=MockSearchInput,
        handler=mock_search_handler,
    )

    builtin_tools = [{"type": "web_search"}]

    registry = ToolRegistry(function_tools=[custom_tool], builtin_tools=builtin_tools)

    # Custom function tools can be retrieved
    assert registry.get("search_documents") is not None
    assert registry.is_builtin("search_documents") is False

    # Built-in tools are not in function tools registry
    assert registry.get("web_search") is None
    assert registry.is_builtin("web_search") is True

    # Unknown tools
    assert registry.get("unknown") is None
    assert registry.is_builtin("unknown") is False


def test_openai_function_tool_schema_format() -> None:
    """OpenAIFunctionTool should generate correct schema format."""
    tool = OpenAIFunctionTool(
        name="search_documents",
        description="Search local documents",
        input_model=MockSearchInput,
        handler=mock_search_handler,
    )

    schema = tool.schema

    assert schema["type"] == "function"
    assert schema["name"] == "search_documents"
    assert schema["description"] == "Search local documents"
    assert "parameters" in schema
    assert schema["parameters"]["properties"]["query"]["type"] == "string"


def test_openai_function_tool_schema_uses_strict_new_framework_export() -> None:
    """Compatibility wrapper should emit strict Responses API schemas."""

    tool = OpenAIFunctionTool(
        name="search_documents",
        description="Search local documents",
        input_model=MockSearchInput,
        handler=mock_search_handler,
    )

    schema = tool.schema

    assert schema["strict"] is True
    assert schema["parameters"]["additionalProperties"] is False
    assert schema["parameters"]["required"] == ["query"]


def test_tool_registry_exposes_openai_tools_alias() -> None:
    """Legacy registry should expose the new preferred OpenAI tools property."""

    custom_tool = OpenAIFunctionTool(
        name="search_documents",
        description="Search local documents",
        input_model=MockSearchInput,
        handler=mock_search_handler,
    )
    registry = ToolRegistry(function_tools=[custom_tool])

    assert registry.openai_tools == registry.schemas


@pytest.mark.asyncio
async def test_tool_registry_async_run_preserves_legacy_result_shape() -> None:
    """Legacy async dispatch should return handler values, not ToolResult objects."""

    custom_tool = OpenAIFunctionTool(
        name="search_documents",
        description="Search local documents",
        input_model=MockSearchInput,
        handler=mock_search_handler,
    )
    registry = ToolRegistry(function_tools=[custom_tool])

    result = await registry.run("search_documents", {"query": "climate"})

    assert result == "Results for: climate"


def test_tool_registry_sync_run_preserves_legacy_result_shape() -> None:
    """Legacy sync dispatch should return handler values, not ToolResult objects."""

    custom_tool = OpenAIFunctionTool(
        name="search_documents",
        description="Search local documents",
        input_model=MockSearchInput,
        handler=mock_search_handler,
    )
    registry = ToolRegistry(function_tools=[custom_tool])

    result = registry.run_sync("search_documents", {"query": "climate"})

    assert result == "Results for: climate"


def test_openai_function_tool_accepts_execution_configuration() -> None:
    """Legacy function tools should expose provider-neutral execution policy."""

    tool = OpenAIFunctionTool(
        name="search_documents",
        description="Search local documents",
        input_model=MockSearchInput,
        handler=mock_search_handler,
        execution=ToolExecutionConfig(max_retries=1),
    )

    assert tool.execution.max_retries == 1
