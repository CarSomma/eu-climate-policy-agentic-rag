"""End-to-end pipeline for discovering, fetching, and saving source documents."""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated

import typer

from eu_climate_policy_rag.collection.document_discovery import (
    DOCUMENTATION_URL,
    DocumentLinkScraper,
)
from eu_climate_policy_rag.collection.document_metadata import MetadataEnricher
from eu_climate_policy_rag.collection.fetch_agent import DocumentFetchAgent
from eu_climate_policy_rag.core.logging_utils import get_logger
from eu_climate_policy_rag.core.models import PipelineConfigModel, PipelineResultModel
from eu_climate_policy_rag.core.types import DocumentMetadata


DocumentFilter = Callable[[DocumentMetadata], bool]
LOGGER = get_logger(__name__)


PipelineResult = PipelineResultModel


async def run_fetch_pipeline(
    source_url: str = DOCUMENTATION_URL,
    *,
    scraper: DocumentLinkScraper | None = None,
    enricher: MetadataEnricher | None = None,
    fetch_agent: DocumentFetchAgent | None = None,
    document_filter: DocumentFilter | None = None,
    limit: int | None = None,
    max_turns: int = 12,
    output_directory: str | Path = "climate_policy_docs",
) -> PipelineResult:
    """Discover document links, enrich them, fetch selected URLs, and save Markdown.

    The default filter keeps only documents that look relevant to the EU climate
    policy project. Pass ``document_filter=lambda _: True`` to fetch everything.
    """
    config = PipelineConfigModel(
        source_url=source_url,
        limit=limit,
        max_turns=max_turns,
        output_directory=Path(output_directory),
        fetch_all=document_filter is None,
    )

    scraper = scraper or DocumentLinkScraper()
    enricher = enricher or MetadataEnricher()
    fetch_agent = fetch_agent or DocumentFetchAgent(output_directory=config.output_directory)
    document_filter = document_filter or is_relevant_document

    LOGGER.info("Discovering document links from %s", config.source_url)
    sections = await scraper.get_doc_links_by_section(str(config.source_url))
    LOGGER.info("Discovered %s sections", len(sections))
    enriched_sections = enricher.enrich(sections)
    documents = flatten_sections(enriched_sections)
    LOGGER.info("Prepared %s unique document candidates", len(documents))
    selected_documents = [document for document in documents if document_filter(document)]
    if config.limit is not None:
        selected_documents = selected_documents[: config.limit]
        LOGGER.info("Applied fetch limit: %s", config.limit)
    LOGGER.info(
        "Selected %s documents; skipped %s by metadata preselection",
        len(selected_documents),
        len(documents) - len(selected_documents),
    )

    results = []
    for index, document in enumerate(selected_documents, start=1):
        LOGGER.info(
            "Fetching document %s/%s: %s",
            index,
            len(selected_documents),
            document["title"],
        )
        try:
            message = await fetch_agent.fetch_document(
                document["url"],
                max_turns=config.max_turns,
            )
            results.append(
                {
                    "title": document["title"],
                    "url": document["url"],
                    "status": "fetched",
                    "message": message,
                }
            )
        except Exception as exc:
            LOGGER.exception("Fetch failed for %s", document["title"])
            results.append(
                {
                    "title": document["title"],
                    "url": document["url"],
                    "status": "failed",
                    "message": str(exc),
                }
            )

    fetched_count = sum(1 for result in results if result["status"] == "fetched")
    LOGGER.info(
        "Pipeline finished: %s fetched, %s failed",
        fetched_count,
        len(results) - fetched_count,
    )
    return PipelineResult(
        discovered_count=len(documents),
        selected_count=len(selected_documents),
        fetched_count=fetched_count,
        skipped_count=len(documents) - len(selected_documents),
        results=results,
    )


def flatten_sections(
    sections: Mapping[str, Sequence[DocumentMetadata]],
) -> list[DocumentMetadata]:
    """Flatten section-grouped metadata while preserving the section name."""

    flattened = []
    for section, documents in sections.items():
        for document in documents:
            flattened.append({**document, "section": section})
    return deduplicate_documents(flattened)


def deduplicate_documents(
    documents: Sequence[DocumentMetadata],
) -> list[DocumentMetadata]:
    """Deduplicate documents by normalized URL."""

    deduplicated = []
    seen_urls = set()
    for document in documents:
        url = document["url"]
        if url in seen_urls:
            continue
        deduplicated.append(document)
        seen_urls.add(url)
    return deduplicated


def is_relevant_document(document: DocumentMetadata) -> bool:
    """Return True when metadata suggests a document is worth fetching."""

    title = document["title"].lower()
    topic = document["topic"]
    document_type = document["type"]

    if topic in {"climate_law", "climate_target_2040", "clean_industrial_deal"}:
        return True

    if document_type in {"regulation", "proposal", "staff_document", "factsheet", "qa"}:
        return has_climate_signal(title)

    if document_type in {"communication", "press_release"}:
        return has_climate_signal(title)

    return has_climate_signal(title)


def has_climate_signal(text: str) -> bool:
    """Return True when text appears relevant to EU climate policy."""

    climate_terms = (
        "climate",
        "greenhouse",
        "ghg",
        "emission",
        "carbon",
        "decarbon",
        "net-zero",
        "net zero",
        "energy",
        "adaptation",
        "industrial deal",
    )
    return any(term in text.lower() for term in climate_terms)


async def discover_and_enrich_documents(
    source_url: str = DOCUMENTATION_URL,
    scraper: DocumentLinkScraper | None = None,
    enricher: MetadataEnricher | None = None,
) -> list[DocumentMetadata]:
    """Return enriched, deduplicated document metadata without fetching content."""

    scraper = scraper or DocumentLinkScraper()
    enricher = enricher or MetadataEnricher()
    sections = await scraper.get_doc_links_by_section(source_url)
    return flatten_sections(enricher.enrich(sections))


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
            help="Disable metadata preselection and attempt every discovered document.",
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
