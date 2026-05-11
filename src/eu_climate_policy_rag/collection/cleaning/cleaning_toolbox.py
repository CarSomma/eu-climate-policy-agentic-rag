"""Clean fetched Markdown files and prepare records for RAG ingestion."""

import json
from pathlib import Path
from typing import Any

from eu_climate_policy_rag.collection.cleaning.markdown_cleaning import (
    clean_markdown,
    hash_markdown,
    has_enough_climate_signal,
    metadata_from_path,
)
from eu_climate_policy_rag.core.logging_utils import get_logger
from eu_climate_policy_rag.core.models import (
    CleanedDocumentRecordModel,
    IngestionConfigModel,
)

LOGGER = get_logger(__name__)


class CleaningToolbox:
    """Stateful workspace used by the optional LLM cleaning curator."""

    def __init__(
        self,
        input_directory: str | Path = "climate_policy_docs",
        output_path: str | Path = "data/eu_climate_policy.json",
        minimum_characters: int = 500,
    ) -> None:
        config = IngestionConfigModel(
            input_directory=Path(input_directory),
            output_path=Path(output_path),
        )
        self.input_directory = config.input_directory
        self.output_path = config.output_path
        self.minimum_characters = minimum_characters
        self.records: list[dict[str, Any]] = []
        self.skipped: list[dict[str, str]] = []
        self.seen_hashes: set[str] = set()

    def skip_reason(
        self,
        path: Path,
        cleaned_text: str,
        content_hash: str,
        seen_hashes: set[str] | None = None,
    ) -> str | None:
        """Return the reason a cleaned document should be skipped."""

        hashes = self.seen_hashes if seen_hashes is None else seen_hashes
        if content_hash in hashes:
            return "duplicate content hash"
        if len(cleaned_text) < self.minimum_characters:
            return "cleaned content is too short"
        if not has_enough_climate_signal(f"{path.stem} {cleaned_text}"):
            return "not clearly about EU climate policy"
        return None

    def _prepare_document(self, path: str | Path) -> dict[str, Any]:
        """Read, clean, hash, and evaluate one Markdown document."""

        markdown_path = Path(path)
        raw_markdown = markdown_path.read_text(encoding="utf-8", errors="ignore")
        cleaned_text = clean_markdown(raw_markdown)
        content_hash = hash_markdown(cleaned_text)
        return {
            "path": markdown_path,
            "raw_markdown": raw_markdown,
            "cleaned_text": cleaned_text,
            "content_hash": content_hash,
            "metadata": metadata_from_path(markdown_path),
            "skip_reason": self.skip_reason(
                markdown_path,
                cleaned_text,
                content_hash,
            ),
        }

    def list_documents(self) -> dict[str, Any]:
        """List fetched Markdown files available for curation."""

        paths = [str(path) for path in sorted(self.input_directory.glob("*.md"))]
        LOGGER.info("Cleaning agent listed %s documents", len(paths))
        return {"documents": paths, "count": len(paths)}

    def inspect_document(self, path: str) -> dict[str, Any]:
        """Return cleaning metadata, a preview, and any skip reason."""

        prepared = self._prepare_document(path)
        markdown_path = prepared["path"]
        raw_markdown = prepared["raw_markdown"]
        cleaned_text = prepared["cleaned_text"]
        reason = prepared["skip_reason"]
        LOGGER.info(
            "Inspected %s: raw=%s cleaned=%s skip=%s",
            markdown_path,
            len(raw_markdown),
            len(cleaned_text),
            reason,
        )
        return {
            "path": str(markdown_path),
            "raw_characters": len(raw_markdown),
            "cleaned_characters": len(cleaned_text),
            "topic": prepared["metadata"]["topic"],
            "skip_reason": reason,
            "preview": cleaned_text[:1200],
        }

    def save_cleaned_document(self, path: str) -> dict[str, Any]:
        """Save one cleaned document record unless cleaning checks reject it."""

        prepared = self._prepare_document(path)
        markdown_path = prepared["path"]
        reason = prepared["skip_reason"]
        if reason:
            return self.skip_document(str(markdown_path), reason)

        return self._save_prepared_document(prepared)

    def skip_document(self, path: str, reason: str) -> dict[str, Any]:
        """Record that a document was skipped with a reason."""

        markdown_path = Path(path)
        if markdown_path.exists():
            prepared = self._prepare_document(markdown_path)
            if prepared["skip_reason"] is None:
                LOGGER.warning(
                    "Ignoring requested skip for accepted document %s: %s",
                    markdown_path,
                    reason,
                )
                return self._save_prepared_document(prepared)

        skipped = {"path": path, "reason": reason}
        if skipped not in self.skipped:
            self.skipped.append(skipped)
        LOGGER.warning("Agent skipped %s: %s", path, reason)
        return {"skipped": True, **skipped}

    def _save_prepared_document(self, prepared: dict[str, Any]) -> dict[str, Any]:
        """Save one already prepared document record."""

        markdown_path = prepared["path"]
        self.seen_hashes.add(prepared["content_hash"])
        self.records.append(
            CleanedDocumentRecordModel(
                **prepared["metadata"],
                source=self.input_directory.name,
                article="document",
                text=prepared["cleaned_text"],
                content_hash=prepared["content_hash"],
            ).model_dump()
        )
        LOGGER.info("Agent saved cleaned document: %s", markdown_path)
        return {"saved": True, "path": str(markdown_path), "records": len(self.records)}

    def finalize(self) -> dict[str, Any]:
        """Write curated records to the configured JSON output path."""

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")
        LOGGER.info(
            "Agent finalized %s records to %s",
            len(self.records),
            self.output_path,
        )
        return {
            "output_path": str(self.output_path),
            "record_count": len(self.records),
            "skipped_count": len(self.skipped),
            "skipped": self.skipped,
        }
