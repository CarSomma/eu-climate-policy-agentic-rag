"""Core typed dictionaries used across the EU climate policy RAG project."""

from typing import NotRequired, TypedDict


class Link(TypedDict):
    """Raw link captured from a browser page or discovery scraper."""

    text: str
    href: str
    index: NotRequired[int]


class DocumentCandidate(TypedDict):
    """Document candidate selected from discovered links."""

    title: str
    url: str
    section: NotRequired[str]


class PipelineFetchResult(TypedDict):
    """Fetch status for one pipeline document candidate."""

    title: str
    url: str
    status: str
    message: str


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
    content_hash: str
