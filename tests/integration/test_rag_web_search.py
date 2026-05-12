"""Integration tests for web_search tool support."""

from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from eu_climate_policy_rag.qa.rag import ClimatePolicyAgent, app


def test_climate_policy_agent_supports_web_search_tool(sample_document):
    """Agent should support web_search as a built-in tool."""
    agent = ClimatePolicyAgent(
        [sample_document],
        openai_client=MagicMock(),
        enable_web_search=True,
        web_search_location={"country": "IT", "city": "Turin"},
    )

    # Should have both search_documents (custom) and web_search (built-in)
    assert len(agent.tools.schemas) == 2
    assert agent.tools.schemas[0]["type"] == "function"
    assert agent.tools.schemas[0]["name"] == "search_documents"
    assert agent.tools.schemas[1]["type"] == "web_search"
    assert agent.tools.schemas[1]["user_location"]["city"] == "Turin"


def test_climate_policy_agent_web_search_disabled_by_default(sample_document):
    """Agent should not include web_search by default."""
    agent = ClimatePolicyAgent(
        [sample_document],
        openai_client=MagicMock(),
    )

    # Should only have search_documents
    assert len(agent.tools.schemas) == 1
    assert agent.tools.schemas[0]["name"] == "search_documents"


def test_climate_policy_agent_web_search_without_location(sample_document):
    """Agent should support web_search without explicit location."""
    agent = ClimatePolicyAgent(
        [sample_document],
        openai_client=MagicMock(),
        enable_web_search=True,
    )

    # Should have web_search without user_location
    assert len(agent.tools.schemas) == 2
    assert agent.tools.schemas[1]["type"] == "web_search"
    assert "user_location" not in agent.tools.schemas[1]


def test_cli_accepts_web_search_flag(tmp_path):
    """CLI should accept --enable-web-search flag."""
    runner = CliRunner()
    test_data = tmp_path / "test.json"
    test_data.write_text(
        '[{"source": "test", "topic": "climate_law", "article": "document", '
        '"text": "content", "file_path": "test.md", "content_hash": "abc"}]'
    )

    result = runner.invoke(
        app,
        [
            "What is the climate plan?",
            "--data", str(test_data),
            "--enable-web-search",
        ],
    )

    # Flag should be recognized (no "no such option" error)
    assert "no such option" not in result.output.lower()
    assert "enable-web-search" not in result.output.lower() or result.exit_code == 0


def test_cli_accepts_web_search_location(tmp_path):
    """CLI should accept --web-search-city and --web-search-country flags."""
    runner = CliRunner()
    test_data = tmp_path / "test.json"
    test_data.write_text(
        '[{"source": "test", "topic": "climate_law", "article": "document", '
        '"text": "content", "file_path": "test.md", "content_hash": "abc"}]'
    )

    result = runner.invoke(
        app,
        [
            "What is the climate plan?",
            "--data", str(test_data),
            "--enable-web-search",
            "--web-search-city", "Turin",
            "--web-search-country", "IT",
        ],
    )

    # Flags should be recognized
    assert "no such option" not in result.output.lower()


@pytest.mark.skip(reason="Requires real OpenAI API call with web_search support")
def test_climate_policy_agent_uses_web_search_for_unknown_queries():
    """Agent should fall back to web_search when local docs don't have info."""
    # This would require a real API call or more sophisticated mocking
    # Skipping for now, but shows the intended behavior
    pass
