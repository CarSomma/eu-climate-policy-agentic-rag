from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from eu_climate_policy_rag.collection.discovery import document_link_scraper
from eu_climate_policy_rag.collection.discovery.document_link_scraper import (
    DOCUMENTATION_URL,
    DocumentLinkScraper,
    _normalize_sections,
    get_doc_links_by_section,
)


@pytest.mark.parametrize(
    ("sections", "expected"),
    [
        (
            {
                "Climate law": [
                    {
                        "text": "European Climate Law",
                        "href": "https://example.test/law",
                    },
                    {"text": "", "href": "https://example.test/empty"},
                    {"text": "Missing href"},
                ]
            },
            {
                "Climate law": [
                    {
                        "text": "European Climate Law",
                        "href": "https://example.test/law",
                    }
                ]
            },
        ),
        (
            {"Section": [{"text": "", "href": "https://example.test/no-text"}]},
            {"Section": []},
        ),
        ({}, {}),
    ],
    ids=["filters-incomplete", "empty-section", "empty-input"],
)
def test_normalize_sections_keeps_only_complete_links(
    sections: dict[str, list[dict[str, str]]],
    expected: dict[str, list[dict[str, str]]],
) -> None:
    assert _normalize_sections(sections) == expected


def test_documentation_url_is_eu_climate_page() -> None:
    assert "climate.ec.europa.eu" in DOCUMENTATION_URL


def test_document_link_scraper_default_wait_values() -> None:
    scraper = DocumentLinkScraper()

    assert scraper.wait_after_load_ms == 3000
    assert scraper.wait_after_click_ms == 500


def test_document_link_scraper_accepts_custom_wait_values() -> None:
    scraper = DocumentLinkScraper(wait_after_load_ms=1000, wait_after_click_ms=200)

    assert scraper.wait_after_load_ms == 1000
    assert scraper.wait_after_click_ms == 200


async def test_get_doc_links_by_section_calls_playwright_and_normalizes() -> None:
    raw_sections = {
        "Climate Law": [
            {"href": "https://example.test/law.pdf", "text": "Climate Law PDF"}
        ],
        "Empty": [{"href": "", "text": ""}],
    }
    page = SimpleNamespace(evaluate=AsyncMock(return_value=raw_sections))
    navigator = SimpleNamespace(
        open_page=AsyncMock(return_value=page),
        click_all=AsyncMock(),
    )
    nav_instance = MagicMock()
    nav_instance.__aenter__ = AsyncMock(return_value=navigator)
    nav_instance.__aexit__ = AsyncMock(return_value=None)
    navigator_factory = MagicMock(return_value=nav_instance)

    result = await DocumentLinkScraper(
        navigator_factory=navigator_factory,
    ).get_doc_links_by_section("https://example.test")

    navigator_factory.assert_called_once_with(default_wait_ms=3000)
    assert result["Climate Law"] == [
        {"href": "https://example.test/law.pdf", "text": "Climate Law PDF"}
    ]
    assert result["Empty"] == []


async def test_module_level_get_doc_links_delegates_to_scraper_class(
    monkeypatch,
) -> None:
    expected = {"Section": [{"href": "https://example.test/a.pdf", "text": "A"}]}
    scraper_class = MagicMock()
    scraper_class.return_value.get_doc_links_by_section = AsyncMock(
        return_value=expected
    )
    monkeypatch.setattr(document_link_scraper, "DocumentLinkScraper", scraper_class)

    result = await get_doc_links_by_section("https://example.test")

    assert result == expected
