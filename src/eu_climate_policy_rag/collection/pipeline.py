"""End-to-end pipeline for discovering, fetching, and saving source documents."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from eu_climate_policy_rag.collection.discovery.document_link_scraper import (
    DOCUMENTATION_URL,
    DocumentLinkScraper,
)
from eu_climate_policy_rag.collection.fetching.fetch_agent import DocumentFetchAgent
from eu_climate_policy_rag.collection.discovery.candidate_discovery import (
    discover_document_candidates,
)
from eu_climate_policy_rag.collection.fetch_pipeline_steps import (
    count_fetched,
    fetch_selected_documents,
)
from eu_climate_policy_rag.collection.discovery.candidate_utils import (
    DocumentFilter,
    is_relevant_document,
    select_documents,
)
from eu_climate_policy_rag.core.logging_utils import get_logger
from eu_climate_policy_rag.core.models import PipelineConfigModel, PipelineResultModel
from eu_climate_policy_rag.core.types import DocumentCandidate


LOGGER = get_logger(__name__)


async def run_fetch_pipeline(
    source_url: str = DOCUMENTATION_URL,
    *,
    scraper: DocumentLinkScraper | None = None,
    fetch_agent: DocumentFetchAgent | None = None,
    document_filter: DocumentFilter | None = None,
    limit: int | None = None,
    model: str = "gpt-5.4-mini",
    max_turns: int = 12,
    output_directory: str | Path = "climate_policy_docs",
) -> PipelineResultModel:
    """Discover document links, fetch selected URLs, and save Markdown.

    The default filter keeps only document titles that look relevant to the EU
    climate policy project. Pass ``document_filter=lambda _: True`` to fetch everything.
    """
    config = PipelineConfigModel(
        source_url=source_url,
        limit=limit,
        model=model,
        max_turns=max_turns,
        output_directory=Path(output_directory),
        fetch_all=document_filter is None,
    )

    scraper = scraper or DocumentLinkScraper()
    fetch_agent = fetch_agent or DocumentFetchAgent(
        output_directory=config.output_directory,
        model=config.model,
    )
    document_filter = document_filter or is_relevant_document

    documents = await discover_document_candidates(str(config.source_url), scraper)
    selected_documents = select_documents(documents, document_filter, config.limit)
    if config.limit is not None:
        LOGGER.info("Applied fetch limit: %s", config.limit)
    LOGGER.info(
        "Selected %s documents; skipped %s by title preselection",
        len(selected_documents),
        len(documents) - len(selected_documents),
    )

    results = await fetch_selected_documents(
        selected_documents,
        fetch_agent,
        config.max_turns,
    )
    fetched_count = count_fetched(results)
    LOGGER.info(
        "Pipeline finished: %s fetched, %s failed",
        fetched_count,
        len(results) - fetched_count,
    )
    return PipelineResultModel(
        discovered_count=len(documents),
        selected_count=len(selected_documents),
        fetched_count=fetched_count,
        skipped_count=len(documents) - len(selected_documents),
        results=results,
    )


async def discover_documents(
    source_url: str = DOCUMENTATION_URL,
    scraper: DocumentLinkScraper | None = None,
) -> list[DocumentCandidate]:
    """Return deduplicated discovered document candidates without fetching content."""

    return await discover_document_candidates(source_url, scraper)


def run_cli(
    source_url: Annotated[
        str,
        typer.Option(
            "--source-url",
            help="Documentation page URL to scrape.",
            show_default=True,
        ),
    ] = DOCUMENTATION_URL,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            help="Maximum number of selected documents to fetch.",
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="OpenAI model to use for document fetching.",
            show_default=True,
        ),
    ] = "gpt-5.4-mini",
    max_turns: Annotated[
        int,
        typer.Option(
            "--max-turns",
            help="Maximum LLM tool-loop turns per document.",
            show_default=True,
        ),
    ] = 12,
    output_directory: Annotated[
        Path,
        typer.Option(
            "--output-directory",
            help="Directory where fetched Markdown files are saved.",
            show_default=True,
        ),
    ] = Path("climate_policy_docs"),
    fetch_all: Annotated[
        bool,
        typer.Option(
            "--fetch-all/--no-fetch-all",
            help="Disable title preselection and attempt every discovered document.",
            show_default=True,
        ),
    ] = False,
) -> None:
    """Discover, preselect, fetch, and save EU climate policy documents from the CLI."""

    document_filter = (lambda _: True) if fetch_all else None
    result = asyncio.run(
        run_fetch_pipeline(
            source_url=source_url,
            document_filter=document_filter,
            limit=limit,
            model=model,
            max_turns=max_turns,
            output_directory=output_directory,
        )
    )

    LOGGER.info("Pipeline summary")
    LOGGER.info("Discovered: %s", result.discovered_count)
    LOGGER.info("Selected:   %s", result.selected_count)
    LOGGER.info("Fetched:    %s", result.fetched_count)
    LOGGER.info("Skipped:    %s", result.skipped_count)

    for item in result.results:
        LOGGER.info("[%s] %s -> %s", item["status"], item["title"], item["url"])


def main() -> None:
    """Run the Typer command-line interface."""

    typer.run(run_cli)


if __name__ == "__main__":
    main()
