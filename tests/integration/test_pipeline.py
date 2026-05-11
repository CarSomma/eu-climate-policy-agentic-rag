import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from eu_climate_policy_rag.collection import pipeline as pipeline_module
from eu_climate_policy_rag.collection.fetch_pipeline_steps import (
    count_fetched,
    fetch_document_candidate,
)
from eu_climate_policy_rag.collection.pipeline import (
    discover_documents,
    run_cli,
    run_fetch_pipeline,
)


def document_candidate() -> dict[str, str]:
    return {
        "title": "European Climate Law",
        "url": "https://example.test/law",
        "section": "Climate law",
    }


def scraper_with_links(*hrefs: str) -> SimpleNamespace:
    return SimpleNamespace(
        get_doc_links_by_section=AsyncMock(
            return_value={
                "Climate law": [
                    {"text": f"Doc {index}", "href": href}
                    for index, href in enumerate(hrefs, start=1)
                ]
            }
        )
    )


def test_run_cli_signature_accepts_output_directory() -> None:
    signature = inspect.signature(run_cli)

    assert "output_directory" in signature.parameters
    assert signature.parameters["output_directory"].default == Path("climate_policy_docs")


def test_run_cli_signature_accepts_model() -> None:
    signature = inspect.signature(run_cli)

    assert "model" in signature.parameters
    assert signature.parameters["model"].default == "gpt-5.4-mini"


async def test_fetch_document_candidate_returns_fetched_status() -> None:
    fetch_agent = SimpleNamespace(fetch_document=AsyncMock(return_value="Saved law.md"))

    result = await fetch_document_candidate(document_candidate(), fetch_agent, max_turns=3)

    fetch_agent.fetch_document.assert_called_once_with(
        "https://example.test/law",
        max_turns=3,
    )
    assert result["status"] == "fetched"
    assert result["message"] == "Saved law.md"


async def test_fetch_document_candidate_verifies_markdown_was_written(tmp_path) -> None:
    async def write_markdown(*_: object, **__: object) -> str:
        (tmp_path / "law.md").write_text("Climate law", encoding="utf-8")
        return "Saved law.md"

    fetch_agent = SimpleNamespace(
        output_directory=tmp_path,
        fetch_document=AsyncMock(side_effect=write_markdown),
    )

    result = await fetch_document_candidate(document_candidate(), fetch_agent, max_turns=3)

    assert result["status"] == "fetched"
    assert result["message"] == "Saved law.md"


async def test_fetch_document_candidate_fails_when_no_markdown_was_written(
    tmp_path,
) -> None:
    fetch_agent = SimpleNamespace(
        output_directory=tmp_path,
        fetch_document=AsyncMock(return_value="I could not save this document."),
    )

    result = await fetch_document_candidate(document_candidate(), fetch_agent, max_turns=3)

    assert result["status"] == "failed"
    assert "No Markdown file was written" in result["message"]


async def test_fetch_document_candidate_returns_failed_status() -> None:
    fetch_agent = SimpleNamespace(
        fetch_document=AsyncMock(side_effect=RuntimeError("network error"))
    )

    result = await fetch_document_candidate(document_candidate(), fetch_agent, max_turns=3)

    assert result["status"] == "failed"
    assert "network error" in result["message"]


def test_count_fetched_counts_successful_records() -> None:
    assert count_fetched(
        [
            {"status": "fetched"},
            {"status": "failed"},
            {"status": "fetched"},
        ]
    ) == 2


async def test_run_fetch_pipeline_no_documents_selected() -> None:
    scraper = SimpleNamespace(
        get_doc_links_by_section=AsyncMock(
            return_value={
                "Off-topic": [
                    {"text": "Single market report", "href": "https://example.test/a"}
                ]
            }
        )
    )
    fetch_agent = SimpleNamespace(fetch_document=AsyncMock())

    result = await run_fetch_pipeline(
        source_url="https://example.test",
        scraper=scraper,
        fetch_agent=fetch_agent,
        document_filter=lambda _: False,
    )

    assert result.discovered_count == 1
    assert result.selected_count == 0
    assert result.fetched_count == 0
    assert result.skipped_count == 1
    fetch_agent.fetch_document.assert_not_called()


async def test_run_fetch_pipeline_fetches_selected_documents() -> None:
    scraper = SimpleNamespace(
        get_doc_links_by_section=AsyncMock(
            return_value={
                "Climate law": [
                    {"text": "European Climate Law", "href": "https://example.test/law"}
                ]
            }
        )
    )
    fetch_agent = SimpleNamespace(
        fetch_document=AsyncMock(return_value="Saved climate-law.md")
    )

    result = await run_fetch_pipeline(
        source_url="https://example.test",
        scraper=scraper,
        fetch_agent=fetch_agent,
        document_filter=lambda _: True,
    )

    assert result.fetched_count == 1
    assert result.results[0]["status"] == "fetched"


async def test_run_fetch_pipeline_handles_fetch_failure() -> None:
    scraper = scraper_with_links("https://example.test/law")
    fetch_agent = SimpleNamespace(
        fetch_document=AsyncMock(side_effect=RuntimeError("network error"))
    )

    result = await run_fetch_pipeline(
        source_url="https://example.test",
        scraper=scraper,
        fetch_agent=fetch_agent,
        document_filter=lambda _: True,
    )

    assert result.fetched_count == 0
    assert result.results[0]["status"] == "failed"
    assert "network error" in result.results[0]["message"]


async def test_run_fetch_pipeline_applies_limit_to_first_selected_candidates() -> None:
    scraper = scraper_with_links(*(f"https://example.test/doc-{i}" for i in range(5)))
    fetch_agent = SimpleNamespace(fetch_document=AsyncMock(return_value="ok"))

    result = await run_fetch_pipeline(
        source_url="https://example.test",
        scraper=scraper,
        fetch_agent=fetch_agent,
        document_filter=lambda _: True,
        limit=1,
    )

    assert result.selected_count == 1
    fetch_agent.fetch_document.assert_called_once_with(
        "https://example.test/doc-0",
        max_turns=12,
    )


async def test_run_fetch_pipeline_passes_model_to_default_fetch_agent(monkeypatch) -> None:
    scraper = scraper_with_links("https://example.test/law")
    fetch_agent = SimpleNamespace(fetch_document=AsyncMock(return_value="ok"))
    fetch_agent_class = MagicMock(return_value=fetch_agent)
    monkeypatch.setattr(pipeline_module, "DocumentFetchAgent", fetch_agent_class)

    result = await run_fetch_pipeline(
        source_url="https://example.test",
        scraper=scraper,
        document_filter=lambda _: True,
        model="gpt-4.1-mini",
        output_directory=Path("docs"),
    )

    fetch_agent_class.assert_called_once_with(
        output_directory=Path("docs"),
        model="gpt-4.1-mini",
    )
    assert result.fetched_count == 1


async def test_discover_documents_returns_flat_list() -> None:
    scraper = SimpleNamespace(
        get_doc_links_by_section=AsyncMock(
            return_value={
                "Section A": [
                    {"text": "Doc 1", "href": "https://example.test/1"},
                    {"text": "Doc 2", "href": "https://example.test/2"},
                ]
            }
        )
    )

    documents = await discover_documents(
        source_url="https://example.test",
        scraper=scraper,
    )

    assert len(documents) == 2
    assert documents[0]["url"] == "https://example.test/1"
