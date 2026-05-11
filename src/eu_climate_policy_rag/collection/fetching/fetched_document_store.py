"""Filesystem storage for fetched Markdown documents."""

from pathlib import Path

from eu_climate_policy_rag.collection.content_hashing import existing_markdown_hashes


class FetchedDocumentStore:
    """Read and write fetched Markdown files."""

    def __init__(self, output_directory: str | Path = "climate_policy_docs") -> None:
        self.output_directory = Path(output_directory)

    def existing_hashes(self, directory: str | Path | None = None) -> set[str]:
        """Return content hashes for Markdown files already in a directory."""

        output_dir = Path(directory) if directory is not None else self.output_directory
        return existing_markdown_hashes(output_dir)

    def save_markdown(
        self,
        markdown: str,
        filename: str,
        directory: str | Path | None = None,
    ) -> Path:
        """Write Markdown to disk and return the file path."""

        output_dir = Path(directory) if directory is not None else self.output_directory
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename
        filepath.write_text(markdown, encoding="utf-8")
        return filepath
