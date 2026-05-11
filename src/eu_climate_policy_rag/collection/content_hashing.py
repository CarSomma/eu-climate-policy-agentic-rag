"""Markdown normalization and hashing helpers."""

import hashlib
import re
from pathlib import Path


def normalize_markdown_for_hash(markdown: str) -> str:
    """Normalize Markdown enough for stable duplicate detection."""

    normalized_lines = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped:
            normalized_lines.append(re.sub(r"\s+", " ", stripped))
    return "\n".join(normalized_lines)


def hash_normalized_markdown(normalized_markdown: str) -> str:
    """Return a stable hash for normalized Markdown text."""

    return hashlib.sha1(normalized_markdown.encode("utf-8")).hexdigest()


def hash_markdown_content(markdown: str) -> str:
    """Normalize and hash Markdown content."""

    return hash_normalized_markdown(normalize_markdown_for_hash(markdown))


def existing_markdown_hashes(directory: Path) -> set[str]:
    """Return content hashes for Markdown files already in a directory."""

    hashes = set()
    for path in directory.glob("*.md"):
        markdown = path.read_text(encoding="utf-8", errors="ignore")
        hashes.add(hash_markdown_content(markdown))
    return hashes
