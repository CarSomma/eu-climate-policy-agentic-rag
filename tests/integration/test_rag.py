import json
from unittest.mock import MagicMock

from eu_climate_policy_rag.core.tools import FunctionTool
from eu_climate_policy_rag.qa.rag import ClimatePolicyAgent, format_context_item
from eu_climate_policy_rag.qa.tools import SearchDocumentsTool


def test_format_context_item(sample_document: dict[str, str]) -> None:
    assert format_context_item(sample_document) == (
        "[European Climate Law | Article 4 | topic: climate_law]\n"
        "The Union-wide 2030 climate target is binding."
    )


def test_search_documents_tool(sample_document: dict[str, str]) -> None:
    tool = SearchDocumentsTool([sample_document], num_results=1)

    result = tool.run(" 2030 target ")

    assert result.query == "2030 target"
    assert len(result.sources) == 1
    assert result.sources[0] == "European Climate Law"
    assert "Article 4" in result.context
    assert "binding" in result.context.lower()
    assert tool.schema["name"] == "search_documents"


def test_search_documents_tool_uses_native_function_tool(
    sample_document: dict[str, str],
) -> None:
    tool = SearchDocumentsTool([sample_document], num_results=1)

    assert isinstance(tool.function_tool, FunctionTool)
    assert type(tool.function_tool) is FunctionTool


def test_climate_policy_agent_uses_shared_tool_registry(
    sample_document: dict[str, str],
) -> None:
    agent = ClimatePolicyAgent([sample_document], openai_client=object())

    assert [tool["name"] for tool in agent.tools.schemas] == ["search_documents"]


def test_climate_policy_agent_accepts_injected_search_tool(
    sample_document: dict[str, str],
) -> None:
    search_tool = SearchDocumentsTool(
        [sample_document],
        num_results=1,
        max_chars_per_doc=500,
    )

    agent = ClimatePolicyAgent(
        [sample_document],
        openai_client=object(),
        search_tool=search_tool,
    )

    assert agent.search_tool is search_tool
    assert [tool["name"] for tool in agent.tools.schemas] == ["search_documents"]


def test_from_json_loads_documents(tmp_path, sample_document: dict[str, str]) -> None:
    path = tmp_path / "records.json"
    path.write_text(json.dumps([sample_document]), encoding="utf-8")

    agent = ClimatePolicyAgent.from_json(path, openai_client=object())

    assert len(agent.documents) == 1
    assert agent.documents[0]["source"] == "European Climate Law"


def test_execute_tool_call_dispatches_search(
    sample_document: dict[str, str],
    make_tool_call,
) -> None:
    agent = ClimatePolicyAgent([sample_document], openai_client=object())
    tool_call = make_tool_call("search_documents", {"query": "2030 target"})

    output = agent._execute_tool_call(tool_call)

    assert output["type"] == "function_call_output"
    assert output["call_id"] == "call_001"
    assert "Article 4" in output["output"]
    assert "European Climate Law" in output["output"]


def test_execute_tool_call_handles_unknown_tool(
    sample_document: dict[str, str],
    make_tool_call,
) -> None:
    agent = ClimatePolicyAgent([sample_document], openai_client=object())
    tool_call = make_tool_call("unknown_tool", {}, call_id="call_002")

    output = agent._execute_tool_call(tool_call)

    assert "Unknown tool" in output["output"]


def test_answer_returns_rag_answer_model_on_no_tool_calls(
    sample_document: dict[str, str],
    make_message_response,
) -> None:
    mock_client = MagicMock()
    mock_client.responses.create.return_value = make_message_response(
        "The 2030 target is -55%."
    )
    agent = ClimatePolicyAgent([sample_document], openai_client=mock_client)

    result = agent.answer("What is the 2030 target?")

    assert result.query == "What is the 2030 target?"
    assert result.answer == "The 2030 target is -55%."
    assert result.sources == []


def test_answer_iterates_tool_calls_then_returns_answer(
    sample_document: dict[str, str],
    make_message_response,
    make_tool_call_response,
) -> None:
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = [
        make_tool_call_response("search_documents", {"query": "2030 target"}),
        make_message_response("The 2030 target is -55%."),
    ]
    agent = ClimatePolicyAgent([sample_document], openai_client=mock_client)

    result = agent.answer("What is the 2030 target?")

    assert result.answer == "The 2030 target is -55%."
    assert mock_client.responses.create.call_count == 2


def test_answer_returns_max_turns_message_when_limit_exhausted(
    sample_document: dict[str, str],
    make_tool_call_response,
) -> None:
    mock_client = MagicMock()
    mock_client.responses.create.return_value = make_tool_call_response(
        "search_documents",
        {"query": "2030 target"},
    )
    agent = ClimatePolicyAgent(
        [sample_document],
        openai_client=mock_client,
        max_turns=2,
    )

    result = agent.answer("What is the 2030 target?")

    assert "max turns" in result.answer
    assert mock_client.responses.create.call_count == 2


def test_format_context_item_handles_missing_article() -> None:
    incomplete = {
        "source": "European Climate Law",
        "topic": "climate_law",
        "text": "Content without article field.",
        "file_path": "law.md",
        "content_hash": "xyz789",
    }

    result = format_context_item(incomplete)

    assert "European Climate Law" in result
    assert "unknown article" in result  # Default article value when missing
    assert "Content without article field." in result


def test_search_documents_tool_returns_empty_on_no_matches() -> None:
    documents = [
        {
            "source": "European Climate Law",
            "article": "Article 4",
            "topic": "climate_law",
            "text": "The Union-wide 2030 climate target is binding.",
            "file_path": "law.md",
            "content_hash": "abc123",
        }
    ]
    tool = SearchDocumentsTool(documents, num_results=5)

    result = tool.run("quantum physics")

    assert result.query == "quantum physics"
    assert len(result.sources) == 0 or result.context == ""


def test_climate_policy_agent_with_empty_document_list() -> None:
    agent = ClimatePolicyAgent([], openai_client=object())

    assert len(agent.documents) == 0
    assert agent.search_tool is not None


def test_answer_tracks_sources_from_tool_calls(
    sample_document: dict[str, str],
    make_message_response,
    make_tool_call_response,
) -> None:
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = [
        make_tool_call_response("search_documents", {"query": "2030 target"}),
        make_message_response("The 2030 target is -55%."),
    ]
    agent = ClimatePolicyAgent([sample_document], openai_client=mock_client)

    result = agent.answer("What is the 2030 target?")

    assert "European Climate Law" in result.sources
    assert len(result.sources) == 1


def test_search_documents_registry_middleware_collects_sources(
    sample_document: dict[str, str],
) -> None:
    agent = ClimatePolicyAgent([sample_document], openai_client=object())

    result = agent._tool_executor.run_sync(
        "search_documents",
        {"query": "2030 target"},
        error_mode="raise",
    )

    assert isinstance(result.value, str)
    assert "Article 4" in result.value
    assert agent._current_run_sources == ["European Climate Law"]
