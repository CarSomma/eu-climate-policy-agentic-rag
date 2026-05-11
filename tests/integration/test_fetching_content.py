import os

import pytest

from eu_climate_policy_rag.collection.fetching.content_cache import (
    CachedContentStore,
    ContentCache,
)
from eu_climate_policy_rag.collection.fetching.fetched_document_store import (
    FetchedDocumentStore,
)
from eu_climate_policy_rag.collection.fetching.fetch_agent import DocumentFetchAgent
from eu_climate_policy_rag.collection.fetching.fetch_toolbox import (
    DocumentFetchToolbox,
    _filename_from_content_disposition,
)


def test_fetched_document_store_saves_markdown_and_reads_hashes(
    tmp_path,
    climate_markdown: str,
) -> None:
    store = FetchedDocumentStore(tmp_path)

    path = store.save_markdown(climate_markdown, "climate.md")

    assert path == tmp_path / "climate.md"
    assert path.exists()
    assert store.existing_hashes()


def test_save_content_to_file_rejects_duplicate_markdown(
    tmp_path,
    climate_markdown: str,
    write_markdown_file,
) -> None:
    write_markdown_file(tmp_path, "existing.md", climate_markdown)
    cache = ContentCache()
    markdown_id = cache.add({"markdown": climate_markdown, "title": "Duplicate"})
    agent = DocumentFetchAgent(cache=cache)

    result = agent.save_content_to_file(markdown_id, "duplicate.md", directory=tmp_path)

    assert result["rejected"]
    assert result["reason"] == "duplicate content already exists"
    assert not (tmp_path / "duplicate.md").exists()


def test_content_cache_add_and_pop_returns_original_item() -> None:
    cache = ContentCache()
    item = {"format": "html", "html": "<p>Hello</p>", "title": "Test"}

    content_id = cache.add(item)

    assert cache.pop(content_id) == item
    assert cache.pop(content_id) is None


def test_cached_content_store_stores_html_with_correct_format() -> None:
    cache = ContentCache()
    store = CachedContentStore(cache)

    result = store.cache_html(
        "https://example.test/page",
        "My Page",
        "<p>Hello</p>",
    )

    assert result["format"] == "html"
    assert result["title"] == "My Page"
    assert result["url"] == "https://example.test/page"
    cached = cache.pop(result["content_id"])
    assert cached["html"] == "<p>Hello</p>"


def test_cached_content_store_stores_pdf_format() -> None:
    cache = ContentCache()
    store = CachedContentStore(cache)

    result = store.cache_file_bytes("report.pdf", b"%PDF-1.4 fake content")

    assert result["format"] == "pdf"
    assert result["title"] == "report.pdf"
    cached = cache.pop(result["content_id"])
    os.unlink(cached["tmp_path"])


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ('attachment; filename="report.pdf"', "report.pdf"),
        ("attachment; filename=report.pdf", "report.pdf"),
        ("attachment", None),
    ],
    ids=["quoted", "unquoted", "missing"],
)
def test_filename_from_content_disposition(header: str, expected: str | None) -> None:
    assert _filename_from_content_disposition(header) == expected


def test_convert_to_markdown_returns_error_for_unknown_id() -> None:
    toolbox = DocumentFetchToolbox()

    result = toolbox.convert_to_markdown("nonexistent")

    assert "error" in result
    assert "nonexistent" in result["error"]


def test_save_content_to_file_returns_error_for_unknown_id(tmp_path) -> None:
    toolbox = DocumentFetchToolbox(output_directory=tmp_path)

    result = toolbox.save_content_to_file("nonexistent", "file.md", str(tmp_path))

    assert "error" in result
    assert "nonexistent" in result["error"]


def test_convert_to_markdown_converts_cached_html() -> None:
    html = "<html><body><p>Hello world.</p></body></html>"
    toolbox = DocumentFetchToolbox()
    content_id = toolbox.cache.add(
        {"format": "html", "html": html, "title": "Test Page"}
    )

    result = toolbox.convert_to_markdown(content_id)

    assert "markdown_id" in result
    assert result["title"] == "Test Page"
    assert "preselection" in result
    assert toolbox.cache.pop(content_id) is None


def test_convert_to_markdown_uses_injected_converter() -> None:
    class Converter:
        def __init__(self) -> None:
            self.called = False

        def convert_cached(self, cached) -> str:
            self.called = True
            return "Climate policy markdown text. " * 40

    converter = Converter()
    toolbox = DocumentFetchToolbox(converter=converter)
    content_id = toolbox.cache.add(
        {"format": "html", "html": "<p>ignored</p>", "title": "Test"}
    )

    result = toolbox.convert_to_markdown(content_id)

    assert "markdown_id" in result
    assert converter.called
