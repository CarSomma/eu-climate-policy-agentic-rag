import pytest

from eu_climate_policy_rag.collection.document_urls import (
    UrlNormalizer,
    canonical_document_key,
    detect_format_from_url,
    filename_from_url,
    is_document_url,
    is_download_button,
    is_pdf_url,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32021R1119",
            "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32021R1119",
        ),
        (
            "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32021R1119",
            "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32021R1119",
        ),
        (
            "https://commission.europa.eu/some/page",
            "https://commission.europa.eu/some/page",
        ),
    ],
    ids=["already-html", "txt-to-html", "non-eur-lex"],
)
def test_url_normalizer(url: str, expected: str) -> None:
    assert UrlNormalizer().normalize(url) == expected


def test_canonical_document_key_matches_eur_lex_celex_and_eli_urls() -> None:
    celex_url = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32026R0667"
    eli_url = "https://eur-lex.europa.eu/eli/reg/2026/667/oj/eng"

    assert canonical_document_key(celex_url) == "eur-lex:celex:32026R0667"
    assert canonical_document_key(eli_url) == canonical_document_key(celex_url)


@pytest.mark.parametrize(
    "url",
    [
        "https://commission.europa.eu/document/download/abc-123?filename=climate-law.pdf",
        "https://example.test/report.pdf",
        "https://example.test/report.PDF",
        "https://example.test/report.docx",
        "https://example.test/data.xlsx",
    ],
    ids=["commission-download", "pdf", "uppercase-pdf", "docx", "xlsx"],
)
def test_is_document_url_detects_document_links(url: str) -> None:
    assert is_document_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.test/report.pdf",
        "https://example.test/report.PDF",
        "https://example.test/pdf/climate-law",
    ],
    ids=["pdf-extension", "uppercase-pdf", "pdf-path-segment"],
)
def test_is_pdf_url_detects_pdf_links(url: str) -> None:
    assert is_pdf_url(url)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.test/report.pdf", "pdf"),
        ("https://example.test/report.PDF", "pdf"),
        ("https://example.test/page.html", "binary"),
        ("https://example.test/report.docx", "binary"),
    ],
    ids=["pdf", "uppercase-pdf", "html", "unsupported-binary"],
)
def test_detect_format_from_url(url: str, expected: str) -> None:
    assert detect_format_from_url(url) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://commission.europa.eu/document/download/abc?filename=climate-law.pdf",
            "climate-law.pdf",
        ),
        (
            "https://example.test/document/download?filename=Climate%20Law.pdf",
            "Climate Law.pdf",
        ),
        ("https://example.test/docs/climate-law.pdf", "climate-law.pdf"),
        ("https://example.test/", "document"),
    ],
    ids=["query-filename", "encoded-query-filename", "path-name", "fallback"],
)
def test_filename_from_url(url: str, expected: str) -> None:
    assert filename_from_url(url) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Download PDF", True),
        ("Télécharger", True),
        ("Read more", False),
        ("Next page", False),
    ],
    ids=["english", "french", "read-more", "next-page"],
)
def test_is_download_button(text: str, expected: bool) -> None:
    assert is_download_button(text) is expected
