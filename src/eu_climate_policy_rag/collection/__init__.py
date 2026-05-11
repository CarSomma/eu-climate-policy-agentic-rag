"""Document discovery, fetching, and cleaning utilities."""

from eu_climate_policy_rag.collection.document_discovery import (
    DOCUMENTATION_URL,
    DocumentLinkScraper,
    get_doc_links_by_section,
)
from eu_climate_policy_rag.collection.fetch_agent import DocumentFetchAgent
from eu_climate_policy_rag.collection.fetch_toolbox import (
    DocumentFetchToolbox,
    DocumentQualityCheck,
)
from eu_climate_policy_rag.collection.ingestion import (
    CleaningCurationAgent,
    CleaningToolbox,
    FetchedDocumentIngestor,
)
from eu_climate_policy_rag.core.models import PipelineResultModel as PipelineResult
from eu_climate_policy_rag.collection.pipeline import (
    discover_documents,
    run_fetch_pipeline,
)
from eu_climate_policy_rag.collection.url_utils import UrlNormalizer

__all__ = [
    "CleaningCurationAgent",
    "CleaningToolbox",
    "DOCUMENTATION_URL",
    "DocumentFetchAgent",
    "DocumentFetchToolbox",
    "DocumentLinkScraper",
    "DocumentQualityCheck",
    "FetchedDocumentIngestor",
    "PipelineResult",
    "UrlNormalizer",
    "discover_documents",
    "get_doc_links_by_section",
    "run_fetch_pipeline",
]
