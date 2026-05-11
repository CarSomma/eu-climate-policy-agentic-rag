from types import SimpleNamespace
from unittest.mock import AsyncMock

from eu_climate_policy_rag.collection.discovery.candidate_discovery import (
    discover_document_candidates,
)


async def test_discover_document_candidates_uses_scraper_and_flattens() -> None:
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

    documents = await discover_document_candidates(
        source_url="https://example.test",
        scraper=scraper,
    )

    scraper.get_doc_links_by_section.assert_called_once_with("https://example.test")
    assert len(documents) == 2
    assert documents[0]["section"] == "Section A"
