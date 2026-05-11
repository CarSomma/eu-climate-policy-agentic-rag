"""Shared Playwright navigation helpers for collection workflows."""

from types import TracebackType
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, async_playwright


class PlaywrightNavigator:
    """Small async wrapper around Playwright browser setup and page navigation."""

    def __init__(
        self,
        headless: bool = True,
        wait_until: str = "networkidle",
        default_wait_ms: int = 0,
    ) -> None:
        self.headless = headless
        self.wait_until = wait_until
        self.default_wait_ms = default_wait_ms
        self._playwright: Any = None
        self._browser: Browser | None = None

    async def __aenter__(self) -> "PlaywrightNavigator":
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()

    @property
    def browser(self) -> Browser:
        if self._browser is None:
            raise RuntimeError("PlaywrightNavigator must be used as an async context manager.")
        return self._browser

    async def new_context(self, **kwargs: object) -> BrowserContext:
        """Create a browser context for workflows that need requests or downloads."""

        return await self.browser.new_context(**kwargs)

    async def new_page(self, context: BrowserContext | None = None) -> Page:
        """Create a page, optionally within an existing browser context."""

        if context is not None:
            return await context.new_page()
        return await self.browser.new_page()

    async def goto(self, page: Page, url: str, wait_ms: int | None = None) -> None:
        """Navigate a page using the navigator's default load and wait settings."""

        await page.goto(url, wait_until=self.wait_until)
        await self.wait(page, wait_ms)

    async def open_page(
        self,
        url: str,
        *,
        context: BrowserContext | None = None,
        wait_ms: int | None = None,
    ) -> Page:
        """Create a page, navigate to a URL, and return the ready page."""

        page = await self.new_page(context)
        await self.goto(page, url, wait_ms=wait_ms)
        return page

    async def wait(self, page: Page, wait_ms: int | None = None) -> None:
        """Wait for dynamic content after navigation or clicks."""

        timeout = self.default_wait_ms if wait_ms is None else wait_ms
        if timeout > 0:
            await page.wait_for_timeout(timeout)

    async def evaluate_page(
        self,
        url: str,
        script: str,
        *,
        wait_ms: int | None = None,
    ) -> object:
        """Open a URL and evaluate a JavaScript snippet on the loaded page."""

        page = await self.open_page(url, wait_ms=wait_ms)
        return await page.evaluate(script)

    async def click_all(self, page: Page, selector: str, wait_ms: int | None = None) -> None:
        """Click every element matching a selector, ignoring failed clicks."""

        elements = await page.query_selector_all(selector)
        for element in elements:
            try:
                await element.click()
                await self.wait(page, wait_ms)
            except Exception:
                continue
