"""URL and filename helpers for EU climate policy document fetching."""

from pathlib import Path
from urllib.parse import parse_qs, urlparse

DOCUMENT_EXTENSIONS = (
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".odt",
    ".rtf",
)
DOWNLOAD_BUTTON_KEYWORDS = (
    "download",
    "telecharger",
    "télécharger",
    "descargar",
    "scarica",
    "herunterladen",
    "pobierz",
)


class UrlNormalizer:
    """Normalize source-specific URL variants used by fetched EU documents."""

    def normalize(self, url: str) -> str:
        """Return a normalized URL while preserving already-valid links."""

        if "eur-lex" in url and "/TXT/" in url and "/TXT/HTML/" not in url:
            return url.replace("/TXT/", "/TXT/HTML/")
        return url


def is_pdf_url(url: str) -> bool:
    """Return True when a URL likely points to a PDF."""

    lower_url = url.lower()
    path = lower_url.split("?")[0]
    return path.endswith(".pdf") or "/pdf" in path or (
        "download" in lower_url and ".pdf" in lower_url
    )


def is_document_url(href: str) -> bool:
    """Return True when a URL appears to be a downloadable document."""

    lower_href = href.lower()
    path = lower_href.split("?")[0]
    return (
        any(path.endswith(extension) for extension in DOCUMENT_EXTENSIONS)
        or "/download" in lower_href
        or "download=" in lower_href
        or ("document" in path and "download" in path)
        or is_pdf_url(href)
    )


def is_download_button(text: str) -> bool:
    """Return True when visible button text suggests a file download."""

    normalized = text.lower()
    return any(keyword in normalized for keyword in DOWNLOAD_BUTTON_KEYWORDS)


def detect_format_from_url(url: str) -> str:
    """Return a coarse content format for fetched document URLs."""

    path = url.lower().split("?")[0]
    if path.endswith(".pdf") or "pdf" in path:
        return "pdf"
    return "binary"


def filename_from_url(url: str, fallback: str = "document") -> str:
    """Extract a readable filename from a URL or its ``filename`` query parameter."""

    parsed_url = urlparse(url)
    query_filename = parse_qs(parsed_url.query).get("filename", [None])[0]
    if query_filename:
        return query_filename

    path_name = Path(parsed_url.path.rstrip("/")).name
    return path_name or fallback
