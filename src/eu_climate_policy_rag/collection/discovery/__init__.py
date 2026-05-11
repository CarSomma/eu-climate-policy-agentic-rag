"""Discovery helpers for source document candidates."""

from eu_climate_policy_rag.collection.discovery.document_link_scraper import (
    DOCUMENTATION_URL,
    DocumentLinkScraper,
    get_doc_links_by_section,
)
from eu_climate_policy_rag.collection.discovery.candidate_discovery import (
    discover_document_candidates,
)
from eu_climate_policy_rag.collection.discovery.candidate_utils import (
    DocumentFilter,
    deduplicate_documents,
    flatten_sections,
    has_climate_signal,
    is_relevant_document,
    select_documents,
)

__all__ = [
    "DOCUMENTATION_URL",
    "DocumentFilter",
    "DocumentLinkScraper",
    "deduplicate_documents",
    "discover_document_candidates",
    "flatten_sections",
    "get_doc_links_by_section",
    "has_climate_signal",
    "is_relevant_document",
    "select_documents",
]
