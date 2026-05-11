"""Content cache used by document collection workflows."""

import tempfile
import uuid
from pathlib import Path
from typing import Any

from eu_climate_policy_rag.collection.document_urls import detect_format_from_url
from eu_climate_policy_rag.core.types import CachedContent


class ContentCache:
    """Small explicit cache for large HTML, binary files, and converted Markdown."""

    def __init__(self) -> None:
        self._items: dict[str, CachedContent] = {}

    def add(self, item: CachedContent) -> str:
        """Store a cache item and return its short content identifier."""

        content_id = uuid.uuid4().hex[:8]
        self._items[content_id] = item
        return content_id

    def pop(self, content_id: str) -> CachedContent | None:
        """Remove and return a cached item by content identifier."""

        return self._items.pop(content_id, None)


class CachedContentStore:
    """Stage fetched HTML, file bytes, and downloads in a content cache."""

    def __init__(self, cache: ContentCache) -> None:
        self.cache = cache

    def cache_html(self, url: str, title: str, html: str) -> dict[str, Any]:
        """Cache HTML content and return tool-facing metadata."""

        content_id = self.cache.add({"format": "html", "html": html, "title": title})
        return {"url": url, "title": title, "format": "html", "content_id": content_id}

    def cache_file_bytes(self, filename: str, file_bytes: bytes) -> dict[str, Any]:
        """Cache fetched file bytes in a temporary file."""

        suffix = Path(filename).suffix.lower() or ".bin"
        tmp_path = _write_temp_file(file_bytes, suffix)

        fmt = "pdf" if suffix == ".pdf" else detect_format_from_url(filename)
        content_id = self.cache.add(
            {"format": fmt, "tmp_path": tmp_path, "title": filename}
        )
        return {"title": filename, "format": fmt, "content_id": content_id}

    async def cache_download(self, filename: str, save_as: Any) -> dict[str, Any]:
        """Save a Playwright download to a temporary file and cache it."""

        suffix = Path(filename).suffix or ".bin"
        tmp_path = _empty_temp_path(suffix)
        await save_as(tmp_path)

        fmt = "pdf" if suffix.lower() == ".pdf" else "binary"
        content_id = self.cache.add(
            {"format": fmt, "tmp_path": tmp_path, "title": filename}
        )
        return {
            "title": filename,
            "format": fmt,
            "filename": filename,
            "content_id": content_id,
        }


def _write_temp_file(file_bytes: bytes, suffix: str) -> str:
    """Write bytes to a named temporary file and return its path."""

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(file_bytes)
        return tmp.name
    finally:
        tmp.close()


def _empty_temp_path(suffix: str) -> str:
    """Create an empty named temporary file and return its path."""

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        return tmp.name
    finally:
        tmp.close()
