"""URL and filename helpers for EU climate policy document fetching."""

from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

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


def canonical_document_key(url: str) -> str:
    """Return a stable identity key for URLs that may have multiple public forms."""

    parsed_url = urlparse(url)
    if "eur-lex.europa.eu" not in parsed_url.netloc.lower():
        return url

    celex_id = _celex_from_query(parsed_url.query) or _celex_from_eli_path(
        parsed_url.path
    )
    if celex_id:
        return f"eur-lex:celex:{celex_id}"

    return UrlNormalizer().normalize(url)


def _celex_from_query(query: str) -> str | None:
    for values in parse_qs(query).values():
        for value in values:
            normalized = unquote(value).upper()
            if "CELEX:" in normalized:
                return normalized.split("CELEX:", 1)[1].split("&", 1)[0]
    return None


def _celex_from_eli_path(path: str) -> str | None:
    parts = [part for part in path.strip("/").split("/") if part]
    if len(parts) < 4 or parts[0].lower() != "eli":
        return None

    document_type = parts[1].lower()
    year = parts[2]
    number = parts[3]
    celex_type = {
        "reg": "R",
        "dir": "L",
        "dec": "D",
    }.get(document_type)
    if celex_type is None or not year.isdigit() or not number.isdigit():
        return None

    return f"3{year}{celex_type}{int(number):04d}"


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
