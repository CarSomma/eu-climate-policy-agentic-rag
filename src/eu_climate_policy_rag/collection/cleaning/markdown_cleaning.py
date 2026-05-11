"""Pure Markdown cleaning and metadata helpers."""

import re
from pathlib import Path
from typing import Any

from eu_climate_policy_rag.collection.content_hashing import hash_markdown_content
from eu_climate_policy_rag.collection.document_quality import (
    CLIMATE_KEYWORDS,
    NAVIGATION_MARKERS,
)


def clean_markdown(markdown: str) -> str:
    """Remove common boilerplate and PDF conversion artifacts from Markdown."""

    text = markdown.replace("\f", "\n")
    lines = []
    previous_blank = False
    for raw_line in text.splitlines():
        line = normalize_line(raw_line)
        if should_drop_line(line):
            continue

        is_blank = not line
        if is_blank and previous_blank:
            continue
        lines.append(line)
        previous_blank = is_blank

    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def normalize_line(line: str) -> str:
    """Normalize whitespace inside one converted Markdown line."""

    return re.sub(r"\s+", " ", line).strip()


def should_drop_line(line: str) -> bool:
    """Return True for boilerplate, navigation, and PDF page artifact lines."""

    lower_line = line.lower()
    if not line:
        return False
    if line in {"EN", "en"}:
        return True
    if re.fullmatch(r"\d{1,4}", line):
        return True
    if re.fullmatch(r"[.\s]{8,}", line):
        return True
    if any(marker in lower_line for marker in NAVIGATION_MARKERS):
        return True
    if lower_line.startswith(("press contacts:", "general public inquiries:")):
        return True
    if lower_line.startswith(("print isbn", "pdf isbn", "isbn ", "doi:")):
        return True
    if lower_line.startswith(("reuse is authorised", "reuse of this document")):
        return True
    return False


def metadata_from_path(path: Path) -> dict[str, Any]:
    """Infer simple RAG metadata from a Markdown filename."""

    title = path.stem.replace("_", " ").replace("-", " ").strip()
    return {
        "title": title,
        "topic": infer_topic(title),
        "file_path": str(path),
    }


def infer_topic(text: str) -> str:
    """Infer a coarse topic label from a filename or title."""

    text_lower = text.lower()
    if "2040" in text_lower:
        return "climate_target_2040"
    if "climate law" in text_lower:
        return "climate_law"
    if "adaptation" in text_lower or "amendment" in text_lower:
        return "adaptation"
    if "energy" in text_lower:
        return "energy_and_climate"
    if "industrial deal" in text_lower:
        return "clean_industrial_deal"
    return "general"


def has_enough_climate_signal(text: str, threshold: int = 2) -> bool:
    """Return True when text contains enough climate-policy keywords."""

    text_lower = text.lower()
    return sum(1 for keyword in CLIMATE_KEYWORDS if keyword in text_lower) >= threshold


def hash_markdown(markdown: str) -> str:
    """Return a stable content hash for cleaned Markdown."""

    return hash_markdown_content(markdown)
