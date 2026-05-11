"""Discovery helpers for the collection pipeline."""

from eu_climate_policy_rag.collection.discovery.document_link_scraper import (
    DOCUMENTATION_URL,
    DocumentLinkScraper,
)
from eu_climate_policy_rag.collection.discovery.candidate_utils import flatten_sections
from eu_climate_policy_rag.core.logging_utils import get_logger
from eu_climate_policy_rag.core.types import DocumentCandidate

LOGGER = get_logger(__name__)


async def discover_document_candidates(
    source_url: str = DOCUMENTATION_URL,
    scraper: DocumentLinkScraper | None = None,
) -> list[DocumentCandidate]:
    """Return deduplicated document candidates discovered from a source page."""

    scraper = scraper or DocumentLinkScraper()
    LOGGER.info("Discovering document links from %s", source_url)
    sections = await scraper.get_doc_links_by_section(source_url)
    LOGGER.info("Discovered %s sections", len(sections))
    documents = flatten_sections(sections)
    LOGGER.info("Prepared %s unique document candidates", len(documents))
    return documents
