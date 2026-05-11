"""Browser-backed document and HTML capture."""

from collections.abc import Callable
from typing import Any

from eu_climate_policy_rag.collection.fetching.content_cache import CachedContentStore
from eu_climate_policy_rag.collection.fetching.playwright_navigator import PlaywrightNavigator
from eu_climate_policy_rag.collection.document_urls import (
    filename_from_url,
    is_document_url,
)
from eu_climate_policy_rag.core.logging_utils import get_logger

LOGGER = get_logger(__name__)


class BrowserContentCapture:
    """Capture downloadable files or HTML pages through a browser session."""

    def __init__(
        self,
        content_store: CachedContentStore,
        navigator_factory: Callable[..., Any] = PlaywrightNavigator,
        default_wait_ms: int = 1500,
    ) -> None:
        self.content_store = content_store
        self.navigator_factory = navigator_factory
        self.default_wait_ms = default_wait_ms

    async def click_download_button(self, url: str, button_text: str) -> dict[str, Any]:
        """Click a button and cache the downloaded file or HTML fallback."""

        LOGGER.info("Clicking download button %r on %s", button_text, url)
        async with self.navigator_factory(
            default_wait_ms=self.default_wait_ms,
        ) as navigator:
            context = await navigator.new_context()
            page = await navigator.open_page(url, context=context)

            selector = (
                f'button:has-text("{button_text}"), '
                f'[role="button"]:has-text("{button_text}"), '
                f'input[value="{button_text}"]'
            )
            button = await page.query_selector(selector)
            if button is None:
                LOGGER.warning("No download button matched %r", button_text)
                return {"error": f"No button found matching text '{button_text}'"}

            try:
                async with page.expect_download(timeout=15000) as download_info:
                    await button.click()
                download = await download_info.value
                result = await self.content_store.cache_download(
                    download.suggested_filename,
                    download.save_as,
                )
                LOGGER.info("Downloaded file: %s", download.suggested_filename)
                return {"url": url, **result}
            except Exception:
                LOGGER.warning("No file download detected; caching landed HTML instead")
                await page.wait_for_load_state("networkidle")
                await navigator.wait(page, wait_ms=2000)
                return self.content_store.cache_html(
                    page.url,
                    await page.title(),
                    await page.content(),
                )

    async def click_and_capture(self, url: str, link_href: str) -> dict[str, Any]:
        """Fetch a document link directly or navigate to an HTML page and cache it."""

        LOGGER.info("Capturing link: %s", link_href)
        async with self.navigator_factory(
            default_wait_ms=self.default_wait_ms,
        ) as navigator:
            context = await navigator.new_context()

            if is_document_url(link_href):
                response = await context.request.get(link_href)
                content_disposition = response.headers.get("content-disposition", "")
                filename = filename_from_content_disposition(content_disposition)
                filename = filename or filename_from_url(link_href)
                result = self.content_store.cache_file_bytes(
                    filename,
                    await response.body(),
                )
                LOGGER.info("Fetched direct document: %s", filename)
                return {"url": link_href, **result}

            page = await navigator.open_page(url, context=context)

            try:
                await page.click(f'a[href="{link_href}"]')
                await page.wait_for_load_state("networkidle")
                await navigator.wait(page, wait_ms=2000)
            except Exception:
                await navigator.goto(page, link_href, wait_ms=2000)

            return self.content_store.cache_html(
                page.url,
                await page.title(),
                await page.content(),
            )


def filename_from_content_disposition(content_disposition: str) -> str | None:
    """Extract a filename parameter from a Content-Disposition header."""

    if "filename=" not in content_disposition:
        return None
    return content_disposition.split("filename=", maxsplit=1)[-1].strip().strip('"')
