"""Browser-backed page snapshot inspection."""

from collections.abc import Callable
from typing import Any

from eu_climate_policy_rag.collection.fetching.playwright_navigator import PlaywrightNavigator
from eu_climate_policy_rag.collection.document_urls import (
    filename_from_url,
    is_document_url,
    is_download_button,
)
from eu_climate_policy_rag.core.logging_utils import get_logger
from eu_climate_policy_rag.core.models import PageSnapshotModel
from eu_climate_policy_rag.core.types import Link, PageSnapshot

LOGGER = get_logger(__name__)


class PageSnapshotInspector:
    """Inspect web pages for visible links and document candidates."""

    def __init__(
        self,
        default_wait_ms: int = 2000,
        navigator_factory: Callable[..., Any] = PlaywrightNavigator,
    ) -> None:
        self.default_wait_ms = default_wait_ms
        self.navigator_factory = navigator_factory

    async def get_page_snapshot(self, url: str) -> PageSnapshot:
        """Open a page and return visible links, buttons, and document candidates."""

        LOGGER.info("Inspecting page: %s", url)
        if is_document_url(url):
            filename = filename_from_url(url)
            LOGGER.info("URL is a direct document download: %s", filename)
            return PageSnapshotModel(
                title=filename,
                url=url,
                links=[{"text": filename, "href": url}],
                buttons=[],
                download_buttons=[],
                has_downloadable_documents=True,
                has_download_buttons=False,
                document_links=[{"text": filename, "href": url}],
                note="Direct download URL; fetch it with click_and_capture.",
            ).model_dump(exclude_none=True)

        async with self.navigator_factory(
            default_wait_ms=self.default_wait_ms,
        ) as navigator:
            page = await navigator.open_page(url)
            snapshot = await page.evaluate(PAGE_SNAPSHOT_SCRIPT)

        links = snapshot.get("links", [])
        document_links = collect_document_links(
            links,
            snapshot.get("anchor_button_links", []),
        )
        download_buttons = [
            button
            for button in snapshot.get("buttons", [])
            if is_download_button(str(button["text"]))
        ]
        LOGGER.info(
            "Snapshot ready: %s links, %s document links, %s download buttons",
            len(links),
            len(document_links),
            len(download_buttons),
        )

        return PageSnapshotModel(
            title=snapshot["title"],
            url=snapshot["url"],
            links=links,
            buttons=snapshot.get("buttons", []),
            download_buttons=download_buttons,
            has_downloadable_documents=bool(document_links),
            has_download_buttons=bool(download_buttons),
            document_links=document_links,
        ).model_dump(exclude_none=True)


def collect_document_links(
    links: list[Link],
    anchor_button_links: list[Link],
) -> list[Link]:
    """Return document links deduplicated by href."""

    document_links = [link for link in links if is_document_url(link["href"])]
    seen_hrefs = {link["href"] for link in document_links}
    for link in anchor_button_links:
        if link["href"] not in seen_hrefs:
            document_links.append(link)
            seen_hrefs.add(link["href"])
    return document_links


PAGE_SNAPSHOT_SCRIPT = """() => {
    const links = [...document.querySelectorAll('a[href]')]
        .map((a, i) => ({
            index: i,
            text: a.textContent.trim().slice(0, 120),
            href: a.href
        }))
        .filter(l => l.text)
        .slice(0, 60);

    const anchorButtonLinks = [...document.querySelectorAll('a[href][role="button"]')]
        .map(a => ({
            text: (a.textContent.trim().slice(0, 120)
                   || a.getAttribute('aria-label')
                   || 'Download'),
            href: a.href
        }))
        .filter(l => {
            const href = l.href.toLowerCase();
            return href.includes('document') && href.includes('download');
        })
        .slice(0, 20);

    const buttons = [...document.querySelectorAll(
        'button, input[type="button"], input[type="submit"]'
    )]
        .map((b, i) => ({
            index: i,
            text: (b.textContent || b.value || '').trim().slice(0, 120)
        }))
        .filter(b => b.text)
        .slice(0, 20);

    return {
        title: document.title,
        url: window.location.href,
        links: links,
        anchor_button_links: anchorButtonLinks,
        buttons: buttons
    };
}"""
