"""Document discovery, fetching, and cleaning utilities."""

from eu_climate_policy_rag.collection.document_discovery import (
    DOCUMENTATION_URL,
    DocumentLinkScraper,
    get_doc_links_by_section,
)
from eu_climate_policy_rag.collection.document_metadata import MetadataEnricher
from eu_climate_policy_rag.collection.fetch_agent import DocumentFetchAgent
from eu_climate_policy_rag.collection.ingestion import (
    CleaningCurationAgent,
    CleaningToolbox,
    FetchedDocumentIngestor,
)
from eu_climate_policy_rag.collection.pipeline import (
    PipelineResult,
    discover_and_enrich_documents,
    run_fetch_pipeline,
)
from eu_climate_policy_rag.collection.url_utils import UrlNormalizer

__all__ = [
    "CleaningCurationAgent",
    "CleaningToolbox",
    "DOCUMENTATION_URL",
    "DocumentFetchAgent",
    "DocumentLinkScraper",
    "FetchedDocumentIngestor",
    "MetadataEnricher",
    "PipelineResult",
    "UrlNormalizer",
    "discover_and_enrich_documents",
    "get_doc_links_by_section",
    "run_fetch_pipeline",
]
