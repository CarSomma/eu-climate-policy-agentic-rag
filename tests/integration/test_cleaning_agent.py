from unittest.mock import MagicMock

from eu_climate_policy_rag.collection.cleaning.cleaning_agent import (
    CleaningCurationAgent,
)
from eu_climate_policy_rag.collection.cleaning.cleaning_toolbox import CleaningToolbox


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
