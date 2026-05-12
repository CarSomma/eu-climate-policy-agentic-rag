from unittest.mock import MagicMock

from eu_climate_policy_rag.collection.cleaning.cleaning_agent import (
    CleaningCurationAgent,
)
from eu_climate_policy_rag.collection.cleaning.cleaning_toolbox import CleaningToolbox
from eu_climate_policy_rag.collection.cleaning.cleaning_tools import (
    CleaningToolMetricsMiddleware,
    build_cleaning_tools,
)
from eu_climate_policy_rag.core.tooling import OpenAIFunctionTool
from eu_climate_policy_rag.core.tools import FunctionTool, ToolExecutor


def test_cleaning_agent_exposes_class_backed_tool_schemas() -> None:
    agent = CleaningCurationAgent(openai_client=object())

    tool_names = {tool["name"] for tool in agent.tools.schemas}

    assert tool_names == {
        "list_documents",
        "inspect_document",
        "save_cleaned_document",
        "skip_document",
        "finalize",
    }


def test_cleaning_tools_are_native_function_tools(tmp_path) -> None:
    toolbox = CleaningToolbox(tmp_path, tmp_path / "out.json")
    registry = build_cleaning_tools(toolbox)

    assert registry.function_tools
    assert all(
        isinstance(tool, FunctionTool)
        and not isinstance(tool, OpenAIFunctionTool)
        for tool in registry.function_tools
    )


def test_cleaning_metrics_middleware_observes_mutating_tools_without_changing_results(
    tmp_path,
) -> None:
    document_path = tmp_path / "climate-law.md"
    document_path.write_text(
        (
            "European Climate Law sets EU climate targets, emissions reductions, "
            "adaptation policy, and climate neutrality obligations. "
        )
        * 8,
        encoding="utf-8",
    )
    toolbox = CleaningToolbox(tmp_path, tmp_path / "out.json")
    observer = CleaningToolMetricsMiddleware()
    registry = build_cleaning_tools(toolbox, middleware=[observer])

    executor = ToolExecutor(registry)

    save_result = executor.run_sync(
        "save_cleaned_document",
        {"path": str(document_path)},
        error_mode="raise",
    )
    skip_result = executor.run_sync(
        "skip_document",
        {"path": "missing.md", "reason": "not relevant"},
        error_mode="raise",
    )
    finalize_result = executor.run_sync("finalize", {}, error_mode="raise")

    assert save_result.value == {
        "saved": True,
        "path": str(document_path),
        "records": 1,
    }
    assert skip_result.value == {
        "skipped": True,
        "path": "missing.md",
        "reason": "not relevant",
    }
    assert finalize_result.value == {
        "output_path": str(tmp_path / "out.json"),
        "record_count": 1,
        "skipped_count": 1,
        "skipped": [{"path": "missing.md", "reason": "not relevant"}],
    }
    assert observer.events == [
        {
            "tool_name": "save_cleaned_document",
            "path": str(document_path),
            "saved": True,
        },
        {
            "tool_name": "skip_document",
            "path": "missing.md",
            "skipped": True,
        },
        {
            "tool_name": "finalize",
            "record_count": 1,
            "skipped_count": 1,
        },
    ]


def test_run_tool_returns_error_for_unknown_tool() -> None:
    agent = CleaningCurationAgent(openai_client=object())

    result = agent.run_tool("nonexistent_tool", {})

    assert "error" in result
    assert "nonexistent_tool" in result["error"]


def test_run_tool_dispatches_known_tool(tmp_path) -> None:
    input_directory = tmp_path / "docs"
    input_directory.mkdir()
    toolbox = CleaningToolbox(input_directory, tmp_path / "out.json")
    agent = CleaningCurationAgent(openai_client=object(), toolbox=toolbox)

    result = agent.run_tool("list_documents", {})

    assert "documents" in result
    assert result["count"] == 0


def test_run_returns_text_immediately_when_no_tool_calls(
    tmp_path,
    make_message_response,
) -> None:
    mock_client = MagicMock()
    mock_client.responses.create.return_value = make_message_response(
        "All documents cleaned."
    )
    toolbox = CleaningToolbox(tmp_path, tmp_path / "out.json")
    agent = CleaningCurationAgent(openai_client=mock_client, toolbox=toolbox)

    result = agent.run(max_turns=5)

    assert result == "All documents cleaned."
    assert mock_client.responses.create.call_count == 1


def test_run_dispatches_tool_call_then_returns_final_answer(
    tmp_path,
    make_message_response,
    make_tool_call_response,
) -> None:
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = [
        make_tool_call_response("list_documents"),
        make_message_response("Cleaned 0 documents."),
    ]
    toolbox = CleaningToolbox(tmp_path, tmp_path / "out.json")
    agent = CleaningCurationAgent(openai_client=mock_client, toolbox=toolbox)

    result = agent.run(max_turns=5)

    assert result == "Cleaned 0 documents."
    assert mock_client.responses.create.call_count == 2


def test_run_returns_max_turns_message_when_limit_exhausted(
    tmp_path,
    make_tool_call_response,
) -> None:
    mock_client = MagicMock()
    mock_client.responses.create.return_value = make_tool_call_response(
        "list_documents",
        call_id="call_x",
    )
    toolbox = CleaningToolbox(tmp_path, tmp_path / "out.json")
    agent = CleaningCurationAgent(openai_client=mock_client, toolbox=toolbox)

    result = agent.run(max_turns=2)

    assert "max turns" in result
    assert mock_client.responses.create.call_count == 2
