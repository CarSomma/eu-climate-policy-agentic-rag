"""Reusable building blocks for the EU climate policy RAG project."""

from eu_climate_policy_rag.collection.discovery.document_link_scraper import (
    DOCUMENTATION_URL,
    DocumentLinkScraper,
    get_doc_links_by_section,
)
from eu_climate_policy_rag.collection.fetching.fetch_agent import DocumentFetchAgent
from eu_climate_policy_rag.collection.fetching.fetch_toolbox import DocumentFetchToolbox
from eu_climate_policy_rag.collection.cleaning.rag_dataset_ingestion import (
    CleaningCurationAgent,
    CleaningToolbox,
)
from eu_climate_policy_rag.collection.document_quality import DocumentQualityCheck
from eu_climate_policy_rag.collection.document_urls import UrlNormalizer
from eu_climate_policy_rag.qa.rag import ClimatePolicyAgent
from eu_climate_policy_rag.core.logging_utils import ColoredLogger
from eu_climate_policy_rag.core.models import (
    CleanedDocumentRecordModel,
    LinkModel,
    PipelineResultModel,
)



__all__ = [
    "ClimatePolicyAgent",
    "CleaningCurationAgent",
    "CleaningToolbox",
    "CleanedDocumentRecordModel",
    "ColoredLogger",
    "DOCUMENTATION_URL",
    "DocumentFetchAgent",
    "DocumentFetchToolbox",
    "DocumentLinkScraper",
    "DocumentQualityCheck",
    "LinkModel",
    "PipelineResultModel",
    "UrlNormalizer",
    "discover_documents",
    "get_doc_links_by_section",
    "run_fetch_pipeline",
]


def __getattr__(name: str):
    if name in {
        "discover_documents",
        "run_fetch_pipeline",
    }:
        from eu_climate_policy_rag.collection import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
