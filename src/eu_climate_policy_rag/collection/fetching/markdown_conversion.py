"""Markdown conversion helpers for fetched content."""

import io
import os

from markitdown import MarkItDown

from eu_climate_policy_rag.core.types import CachedContent


class MarkdownConverter:
    """Convert cached HTML or file content to Markdown."""

    def __init__(self, converter: MarkItDown | None = None) -> None:
        self.converter = converter or MarkItDown()

    def convert_cached(self, cached: CachedContent) -> str:
        """Convert one cached HTML or file payload to Markdown."""

        if cached["format"] == "html":
            converted = self.converter.convert_stream(
                io.BytesIO(cached["html"].encode("utf-8")),
                file_extension=".html",
            )
            return converted.markdown

        converted = self.converter.convert(cached["tmp_path"])
        os.unlink(cached["tmp_path"])
        return converted.markdown
