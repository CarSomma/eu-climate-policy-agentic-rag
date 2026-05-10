"""Reusable building blocks for the EU climate policy RAG project."""

from eu_climate_policy_rag.collection.document_metadata import MetadataEnricher
from eu_climate_policy_rag.collection.document_discovery import (
    DOCUMENTATION_URL,
    DocumentLinkScraper,
    get_doc_links_by_section,
)
from eu_climate_policy_rag.collection.fetch_agent import DocumentFetchAgent
from eu_climate_policy_rag.collection.ingestion import (
    CleaningCurationAgent,
    CleaningToolbox,
    FetchedDocumentIngestor,
)
from eu_climate_policy_rag.collection.url_utils import UrlNormalizer
from eu_climate_policy_rag.qa.rag import ClimatePolicyAgent
from eu_climate_policy_rag.core.logging_utils import ColoredLogger
from eu_climate_policy_rag.core.models import (
    CleanedDocumentRecordModel,
    DocumentMetadataModel,
    LinkModel,
)

__all__ = [
    "ClimatePolicyAgent",
    "CleaningCurationAgent",
    "CleaningToolbox",
    "CleanedDocumentRecordModel",
    "ColoredLogger",
    "DOCUMENTATION_URL",
    "DocumentMetadataModel",
    "DocumentFetchAgent",
    "DocumentLinkScraper",
    "FetchedDocumentIngestor",
    "LinkModel",
    "MetadataEnricher",
    "PipelineResult",
    "UrlNormalizer",
    "discover_and_enrich_documents",
    "get_doc_links_by_section",
    "run_fetch_pipeline",
]


def __getattr__(name: str):
    if name in {
        "PipelineResult",
        "discover_and_enrich_documents",
        "run_fetch_pipeline",
    }:
        from eu_climate_policy_rag.collection import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
