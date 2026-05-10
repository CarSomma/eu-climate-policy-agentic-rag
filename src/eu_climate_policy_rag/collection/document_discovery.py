"""Discover official document links from EU climate policy pages."""

from collections.abc import Mapping

from playwright.async_api import async_playwright

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
    ) -> None:
        self.wait_after_load_ms = wait_after_load_ms
        self.wait_after_click_ms = wait_after_click_ms

    async def get_doc_links_by_section(self, url: str = DOCUMENTATION_URL) -> dict[str, list[Link]]:
        """Return document links grouped by documentation accordion title."""

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_timeout(self.wait_after_load_ms)

            toggles = await page.query_selector_all(
                "#documentation ~ * .ecl-accordion__toggle"
            )
            for toggle in toggles:
                try:
                    await toggle.click()
                    await page.wait_for_timeout(self.wait_after_click_ms)
                except Exception:
                    continue

            sections = await page.evaluate(_DOCUMENT_LINKS_BY_SECTION_SCRIPT)
            await browser.close()

        return _normalize_sections(sections)


async def get_doc_links_by_section(url: str = DOCUMENTATION_URL) -> dict[str, list[Link]]:
    """Return document links using the notebook-compatible function API."""

    return await DocumentLinkScraper().get_doc_links_by_section(url)


def _normalize_sections(sections: Mapping[str, list[dict[str, str]]]) -> dict[str, list[Link]]:
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
