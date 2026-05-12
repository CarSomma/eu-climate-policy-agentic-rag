import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from eu_climate_policy_rag.collection.fetching.content_cache import ContentCache
from eu_climate_policy_rag.collection.fetching.fetch_agent import DocumentFetchAgent
from eu_climate_policy_rag.collection.fetching.fetch_tools import build_fetch_tools
from eu_climate_policy_rag.core.tooling import OpenAIFunctionTool
from eu_climate_policy_rag.core.tools import FunctionTool, ToolExecutor


async def test_run_tool_forces_agent_output_directory(
    tmp_path,
    climate_markdown: str,
) -> None:
    cache = ContentCache()
    markdown_id = cache.add({"markdown": climate_markdown, "title": "Climate law"})
    agent = DocumentFetchAgent(cache=cache, output_directory=tmp_path)

    result = await agent.run_tool(
        "save_content_to_file",
        {
            "markdown_id": markdown_id,
            "filename": "saved.md",
            "directory": ".",
        },
    )
    payload = json.loads(result)

    assert payload["saved"]
    assert (tmp_path / "saved.md").exists()
    assert payload["path"] == str(tmp_path / "saved.md")


async def test_fetch_tools_registry_forces_output_directory_with_middleware(
    tmp_path,
) -> None:
    mock_toolbox = SimpleNamespace(
        get_page_snapshot=AsyncMock(),
        click_and_capture=AsyncMock(),
        click_download_button=AsyncMock(),
        convert_to_markdown=MagicMock(),
        output_directory=tmp_path,
    )
    mock_toolbox.save_content_to_file = MagicMock(
        return_value={"directory": str(tmp_path)}
    )
    registry = build_fetch_tools(mock_toolbox)

    result = await ToolExecutor(registry).run(
        "save_content_to_file",
        {
            "markdown_id": "md_1",
            "filename": "saved.md",
            "directory": ".",
        },
        error_mode="raise",
    )

    assert result.value == {"directory": str(tmp_path)}
    mock_toolbox.save_content_to_file.assert_called_once_with(
        markdown_id="md_1",
        filename="saved.md",
        directory=str(tmp_path),
    )


def test_fetch_tools_are_native_function_tools(tmp_path) -> None:
    mock_toolbox = SimpleNamespace(
        get_page_snapshot=AsyncMock(),
        click_and_capture=AsyncMock(),
        click_download_button=AsyncMock(),
        convert_to_markdown=MagicMock(),
        save_content_to_file=MagicMock(),
        output_directory=tmp_path,
    )
    registry = build_fetch_tools(mock_toolbox)

    assert registry.function_tools
    assert all(
        isinstance(tool, FunctionTool)
        and not isinstance(tool, OpenAIFunctionTool)
        for tool in registry.function_tools
    )


def test_fetch_agent_exposes_class_backed_tool_schemas() -> None:
    agent = DocumentFetchAgent(cache=ContentCache())

    tool_names = {tool["name"] for tool in agent.tools.schemas}

    assert tool_names == {
        "get_page_snapshot",
        "click_and_capture",
        "click_download_button",
        "convert_to_markdown",
        "save_content_to_file",
    }


def test_agent_properties_delegate_to_toolbox(tmp_path) -> None:
    agent = DocumentFetchAgent(output_directory=tmp_path)

    assert agent.cache is agent.toolbox.cache
    assert agent.quality_check is agent.toolbox.quality_check
    assert agent.output_directory == tmp_path

    new_dir = tmp_path / "new"
    agent.output_directory = new_dir
    assert agent.toolbox.output_directory == new_dir


async def test_agent_delegates_get_page_snapshot_to_toolbox() -> None:
    expected = {"title": "Test", "url": "https://example.test", "links": []}
    mock_toolbox = SimpleNamespace(
        get_page_snapshot=AsyncMock(return_value=expected),
        click_and_capture=AsyncMock(),
        click_download_button=AsyncMock(),
        convert_to_markdown=MagicMock(),
        save_content_to_file=MagicMock(),
        output_directory=Path("climate_policy_docs"),
    )

    agent = DocumentFetchAgent(toolbox=mock_toolbox, openai_client=object())
    result = await agent.get_page_snapshot("https://example.test")

    mock_toolbox.get_page_snapshot.assert_called_once_with("https://example.test")
    assert result == expected


async def test_fetch_agent_run_tool_returns_error_for_unknown_tool() -> None:
    agent = DocumentFetchAgent(cache=ContentCache(), openai_client=object())

    result_json = await agent.run_tool("nonexistent_tool", {})
    result = json.loads(result_json)

    assert "error" in result
    assert "nonexistent_tool" in result["error"]


async def test_fetch_document_returns_final_answer_with_no_tool_calls(
    make_message_response,
) -> None:
    mock_client = MagicMock()
    mock_client.responses.create.return_value = make_message_response(
        "Document saved successfully."
    )
    agent = DocumentFetchAgent(openai_client=mock_client)

    result = await agent.fetch_document("https://example.test/doc", max_turns=3)

    assert result == "Document saved successfully."
    mock_client.responses.create.assert_called_once()


async def test_fetch_document_reaches_max_turns_returns_fallback_message(
    make_tool_call_response,
) -> None:
    mock_client = MagicMock()
    mock_client.responses.create.return_value = make_tool_call_response(
        "unknown_tool",
        {"url": "https://example.test"},
    )
    agent = DocumentFetchAgent(openai_client=mock_client)

    result = await agent.fetch_document("https://example.test/doc", max_turns=2)

    assert "max turns" in result
    assert mock_client.responses.create.call_count == 2
