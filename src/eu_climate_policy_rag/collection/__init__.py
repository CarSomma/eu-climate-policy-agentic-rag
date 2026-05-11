"""Document discovery, fetching, and cleaning utilities."""

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
from eu_climate_policy_rag.core.models import PipelineResultModel as PipelineResult
from eu_climate_policy_rag.collection.pipeline import (
    discover_documents,
    run_fetch_pipeline,
)
from eu_climate_policy_rag.collection.document_urls import UrlNormalizer

__all__ = [
    "CleaningCurationAgent",
    "CleaningToolbox",
    "DOCUMENTATION_URL",
    "DocumentFetchAgent",
    "DocumentFetchToolbox",
    "DocumentLinkScraper",
    "DocumentQualityCheck",
    "PipelineResult",
    "UrlNormalizer",
    "discover_documents",
    "get_doc_links_by_section",
    "run_fetch_pipeline",
]
