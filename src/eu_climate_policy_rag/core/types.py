"""Core typed dictionaries used across the EU climate policy RAG project."""

from typing import NotRequired, TypedDict


class Link(TypedDict):
    """Raw link captured from a browser page or discovery scraper."""

    text: str
    href: str
    index: NotRequired[int]


class DocumentMetadata(TypedDict):
    """Metadata inferred for one candidate EU climate policy document."""

    title: str
    url: str
    type: str
    year: int | None
    identifier: str | None
    source: str
    format: str
    topic: str


class PageSnapshot(TypedDict):
    """Browser inspection result used by the fetch agent tool loop."""

    title: str
    url: str
    links: list[Link]
    buttons: list[dict[str, str | int]]
    download_buttons: list[dict[str, str | int]]
    has_downloadable_documents: bool
    has_download_buttons: bool
    document_links: list[Link]
    note: NotRequired[str]


class CachedContent(TypedDict, total=False):
    """Cached HTML, file, or Markdown content passed between fetch tools."""

    format: str
    html: str
    tmp_path: str
    title: str
    markdown: str
