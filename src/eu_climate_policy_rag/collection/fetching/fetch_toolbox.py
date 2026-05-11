"""Browser, conversion, and file operations for fetching."""

from pathlib import Path
from typing import Any

from eu_climate_policy_rag.collection.fetching.content_cache import (
    CachedContentStore,
    ContentCache,
)
from eu_climate_policy_rag.collection.fetching.browser_content_capture import (
    BrowserContentCapture,
    filename_from_content_disposition,
)
from eu_climate_policy_rag.collection.fetching.markdown_conversion import MarkdownConverter
from eu_climate_policy_rag.collection.fetching.fetched_document_store import (
    FetchedDocumentStore,
)
from eu_climate_policy_rag.collection.fetching.markdown_workflow import MarkdownFetchWorkflow
from eu_climate_policy_rag.collection.fetching.page_snapshot import (
    PAGE_SNAPSHOT_SCRIPT,
    PageSnapshotInspector,
    collect_document_links,
)
from eu_climate_policy_rag.collection.document_quality import DocumentQualityCheck
from eu_climate_policy_rag.core.types import Link, PageSnapshot

_PAGE_SNAPSHOT_SCRIPT = PAGE_SNAPSHOT_SCRIPT


class DocumentFetchToolbox:
    """Shared state and actions behind document-fetching tools."""

    def __init__(
        self,
        cache: ContentCache | None = None,
        content_store: CachedContentStore | None = None,
        converter: MarkdownConverter | None = None,
        store: FetchedDocumentStore | None = None,
        quality_check: DocumentQualityCheck | None = None,
        page_inspector: PageSnapshotInspector | None = None,
        content_capture: BrowserContentCapture | None = None,
        markdown_workflow: MarkdownFetchWorkflow | None = None,
        output_directory: str | Path = "climate_policy_docs",
    ) -> None:
        self.cache = cache or ContentCache()
        self.content_store = content_store or CachedContentStore(self.cache)
        self.converter = converter or MarkdownConverter()
        self.store = store or FetchedDocumentStore(output_directory)
        self.quality_check = quality_check or DocumentQualityCheck()
        self.page_inspector = page_inspector or PageSnapshotInspector()
        self.content_capture = content_capture or BrowserContentCapture(
            self.content_store
        )
        self.markdown_workflow = markdown_workflow or MarkdownFetchWorkflow(
            self.cache,
            self.converter,
            self.store,
            self.quality_check,
        )

    @property
    def output_directory(self) -> Path:
        """Directory where fetched Markdown is saved."""

        return self.store.output_directory

    @output_directory.setter
    def output_directory(self, value: str | Path) -> None:
        self.store.output_directory = Path(value)

    async def get_page_snapshot(self, url: str) -> PageSnapshot:
        """Open a page and return visible links, buttons, and document candidates."""

        return await self.page_inspector.get_page_snapshot(url)

    async def click_download_button(self, url: str, button_text: str) -> dict[str, Any]:
        """Click a button and cache the downloaded file or HTML fallback."""

        return await self.content_capture.click_download_button(url, button_text)

    async def click_and_capture(self, url: str, link_href: str) -> dict[str, Any]:
        """Fetch a document link directly or navigate to an HTML page and cache it."""

        return await self.content_capture.click_and_capture(url, link_href)

    def convert_to_markdown(self, content_id: str) -> dict[str, Any]:
        """Convert cached HTML or binary content to Markdown and cache the result."""

        return self.markdown_workflow.convert_to_markdown(content_id)

    def save_content_to_file(
        self,
        markdown_id: str,
        filename: str,
        directory: str = "climate_policy_docs",
    ) -> dict[str, Any]:
        """Write cached Markdown content to disk."""

        return self.markdown_workflow.save_content_to_file(
            markdown_id,
            filename,
            directory,
        )

    @staticmethod
    def _collect_document_links(
        links: list[Link],
        anchor_button_links: list[Link],
    ) -> list[Link]:
        """Return document links deduplicated by href."""

        return collect_document_links(links, anchor_button_links)


def _filename_from_content_disposition(content_disposition: str) -> str | None:
    """Extract a filename parameter from a Content-Disposition header."""

    return filename_from_content_disposition(content_disposition)
