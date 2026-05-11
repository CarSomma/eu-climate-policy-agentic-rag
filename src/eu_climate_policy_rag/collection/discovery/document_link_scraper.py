"""Discover official document links from EU climate policy pages."""

from collections.abc import Mapping
from typing import Any, Callable

from eu_climate_policy_rag.collection.fetching.playwright_navigator import PlaywrightNavigator
from eu_climate_policy_rag.core.models import LinkModel
from eu_climate_policy_rag.core.types import Link


DOCUMENTATION_URL = (
    "https://climate.ec.europa.eu/eu-action/european-climate-law_en#documentation"
)


class DocumentLinkScraper:
    """Scrape document links grouped by documentation section from EU policy pages."""

    def __init__(
        self,
        wait_after_load_ms: int = 3000,
        wait_after_click_ms: int = 500,
        navigator_factory: Callable[..., Any] = PlaywrightNavigator,
    ) -> None:
        self.wait_after_load_ms = wait_after_load_ms
        self.wait_after_click_ms = wait_after_click_ms
        self.navigator_factory = navigator_factory

    async def get_doc_links_by_section(
        self,
        url: str = DOCUMENTATION_URL,
    ) -> dict[str, list[Link]]:
        """Return document links grouped by documentation accordion title."""

        async with self.navigator_factory(
            default_wait_ms=self.wait_after_load_ms,
        ) as navigator:
            page = await navigator.open_page(url)
            await navigator.click_all(
                page,
                "#documentation ~ * .ecl-accordion__toggle",
                wait_ms=self.wait_after_click_ms,
            )

            sections = await page.evaluate(_DOCUMENT_LINKS_BY_SECTION_SCRIPT)

        return _normalize_sections(sections)


async def get_doc_links_by_section(
    url: str = DOCUMENTATION_URL,
) -> dict[str, list[Link]]:
    """Return document links using the notebook-compatible function API."""

    return await DocumentLinkScraper().get_doc_links_by_section(url)


def _normalize_sections(
    sections: Mapping[str, list[dict[str, str]]],
) -> dict[str, list[Link]]:
    return {
        section: [
            LinkModel.model_validate(link).model_dump(exclude_none=True)
            for link in links
            if link.get("text") and link.get("href")
        ]
        for section, links in sections.items()
    }


_DOCUMENT_LINKS_BY_SECTION_SCRIPT = """() => {
    const docHeading = document.querySelector('#documentation');
    if (!docHeading) return {};

    let container = docHeading.parentElement;
    while (container && !container.querySelector('.ecl-accordion__item')) {
        container = container.nextElementSibling || container.parentElement;
    }

    if (!container) return {};

    const result = {};
    container.querySelectorAll('.ecl-accordion__item').forEach(item => {
        const titleEl = item.querySelector('.ecl-accordion__toggle');
        const title = titleEl ? titleEl.textContent.trim() : 'Unknown';

        const links = [...item.querySelectorAll('a[href]')]
            .map(a => ({
                href: a.href,
                text: a.textContent.trim()
            }))
            .filter(link =>
                link.text &&
                !['download', 'preview'].includes(link.text.toLowerCase())
            );

        if (links.length) result[title] = links;
    });

    return result;
}"""
