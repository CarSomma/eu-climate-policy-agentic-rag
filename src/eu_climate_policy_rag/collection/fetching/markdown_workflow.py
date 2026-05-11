"""Markdown conversion and persistence workflow."""

from pathlib import Path
from typing import Any

from eu_climate_policy_rag.collection.fetching.content_cache import ContentCache
from eu_climate_policy_rag.collection.fetching.markdown_conversion import MarkdownConverter
from eu_climate_policy_rag.collection.fetching.fetched_document_store import (
    FetchedDocumentStore,
)
from eu_climate_policy_rag.collection.content_hashing import hash_markdown_content
from eu_climate_policy_rag.collection.document_quality import DocumentQualityCheck
from eu_climate_policy_rag.core.logging_utils import get_logger

LOGGER = get_logger(__name__)


class MarkdownFetchWorkflow:
    """Convert cached fetched content and save accepted Markdown."""

    def __init__(
        self,
        cache: ContentCache,
        converter: MarkdownConverter,
        store: FetchedDocumentStore,
        quality_check: DocumentQualityCheck,
    ) -> None:
        self.cache = cache
        self.converter = converter
        self.store = store
        self.quality_check = quality_check

    def convert_to_markdown(self, content_id: str) -> dict[str, Any]:
        """Convert cached HTML or binary content to Markdown and cache it."""

        cached = self.cache.pop(content_id)
        if cached is None:
            LOGGER.error("Unknown content_id requested: %s", content_id)
            return {"error": f"content_id '{content_id}' not found in cache"}

        LOGGER.info("Converting cached %s content to Markdown", cached["format"])
        markdown = self.converter.convert_cached(cached)

        markdown_id = self.cache.add(
            {
                "markdown": markdown,
                "title": cached["title"],
                "content_hash": hash_markdown_content(markdown),
            }
        )
        preselection = self.quality_check.assess(cached["title"], markdown)
        LOGGER.info(
            "Converted %s characters; preselection=%s",
            len(markdown),
            preselection.reason,
        )
        return {
            "markdown_id": markdown_id,
            "title": cached["title"],
            "length": len(markdown),
            "preview": markdown[:800],
            "preselection": {
                "accepted": preselection.accepted,
                "reason": preselection.reason,
                "content_hash": preselection.content_hash,
            },
        }

    def save_content_to_file(
        self,
        markdown_id: str,
        filename: str,
        directory: str = "climate_policy_docs",
    ) -> dict[str, Any]:
        """Write cached Markdown content to disk when it passes quality checks."""

        cached = self.cache.pop(markdown_id)
        if cached is None:
            LOGGER.error("Unknown markdown_id requested: %s", markdown_id)
            return {"error": f"markdown_id '{markdown_id}' not found in cache"}

        markdown = cached["markdown"]
        output_dir = Path(directory)
        LOGGER.info("Saving Markdown candidate to %s", output_dir)
        preselection = self.quality_check.assess(
            cached["title"],
            markdown,
            existing_hashes=self.store.existing_hashes(output_dir),
        )
        if not preselection.accepted:
            LOGGER.warning("Rejected Markdown before save: %s", preselection.reason)
            return {
                "saved": False,
                "rejected": True,
                "reason": preselection.reason,
                "content_hash": preselection.content_hash,
            }

        filepath = self.store.save_markdown(markdown, filename, output_dir)
        LOGGER.info("Saved Markdown: %s", filepath)
        return {
            "saved": True,
            "path": str(filepath),
            "bytes": len(markdown.encode("utf-8")),
            "content_hash": preselection.content_hash,
        }
