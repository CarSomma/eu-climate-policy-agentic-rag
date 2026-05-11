from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from eu_climate_policy_rag.collection.fetching.fetch_toolbox import DocumentFetchToolbox
from eu_climate_policy_rag.collection.fetching.page_snapshot import (
    PageSnapshotInspector,
    collect_document_links,
)


def test_collect_document_links_deduplicates_by_href() -> None:
    links = [{"href": "https://example.test/doc.pdf", "text": "PDF"}]
    anchor_buttons = [
        {"href": "https://example.test/doc.pdf", "text": "Duplicate"},
        {"href": "https://example.test/other.pdf", "text": "Other"},
    ]

    result = DocumentFetchToolbox._collect_document_links(links, anchor_buttons)

    hrefs = [item["href"] for item in result]
    assert hrefs.count("https://example.test/doc.pdf") == 1
    assert "https://example.test/other.pdf" in hrefs


def test_collect_document_links_helper_matches_toolbox_compatibility() -> None:
    links = [{"href": "https://example.test/doc.pdf", "text": "PDF"}]
    anchor_buttons = [{"href": "https://example.test/other.pdf", "text": "Other"}]

    assert DocumentFetchToolbox._collect_document_links(
        links,
        anchor_buttons,
    ) == collect_document_links(links, anchor_buttons)


async def test_get_page_snapshot_handles_pdf_url_without_playwright() -> None:
    toolbox = DocumentFetchToolbox(page_inspector=PageSnapshotInspector())

    result = await toolbox.get_page_snapshot("https://example.test/report.pdf")

    assert result["has_downloadable_documents"] is True
    assert result["has_download_buttons"] is False
    assert result["url"] == "https://example.test/report.pdf"


async def test_get_page_snapshot_delegates_to_injected_inspector() -> None:
    expected = {"title": "Test", "url": "https://example.test", "links": []}
    inspector = SimpleNamespace(get_page_snapshot=AsyncMock(return_value=expected))
    toolbox = DocumentFetchToolbox(page_inspector=inspector)

    result = await toolbox.get_page_snapshot("https://example.test")

    inspector.get_page_snapshot.assert_called_once_with("https://example.test")
    assert result == expected


async def test_click_and_capture_delegates_to_injected_capture() -> None:
    expected = {"url": "https://example.test/report.pdf", "format": "pdf"}
    content_capture = SimpleNamespace(
        click_and_capture=AsyncMock(return_value=expected)
    )
    toolbox = DocumentFetchToolbox(content_capture=content_capture)

    result = await toolbox.click_and_capture(
        "https://example.test",
        "https://example.test/report.pdf",
    )

    content_capture.click_and_capture.assert_called_once_with(
        "https://example.test",
        "https://example.test/report.pdf",
    )
    assert result == expected


async def test_click_download_button_delegates_to_injected_capture() -> None:
    expected = {"url": "https://example.test", "format": "html"}
    content_capture = SimpleNamespace(
        click_download_button=AsyncMock(return_value=expected)
    )
    toolbox = DocumentFetchToolbox(content_capture=content_capture)

    result = await toolbox.click_download_button("https://example.test", "Download")

    content_capture.click_download_button.assert_called_once_with(
        "https://example.test",
        "Download",
    )
    assert result == expected


def test_convert_to_markdown_delegates_to_injected_workflow() -> None:
    expected = {"markdown_id": "abc123"}
    markdown_workflow = SimpleNamespace(
        convert_to_markdown=MagicMock(return_value=expected)
    )
    toolbox = DocumentFetchToolbox(markdown_workflow=markdown_workflow)

    result = toolbox.convert_to_markdown("content123")

    markdown_workflow.convert_to_markdown.assert_called_once_with("content123")
    assert result == expected


def test_save_content_to_file_delegates_to_injected_workflow() -> None:
    expected = {"saved": True, "path": "out.md"}
    markdown_workflow = SimpleNamespace(
        save_content_to_file=MagicMock(return_value=expected)
    )
    toolbox = DocumentFetchToolbox(markdown_workflow=markdown_workflow)

    result = toolbox.save_content_to_file("markdown123", "out.md", "docs")

    markdown_workflow.save_content_to_file.assert_called_once_with(
        "markdown123",
        "out.md",
        "docs",
    )
    assert result == expected


async def test_page_snapshot_inspector_uses_injected_navigator() -> None:
    raw_snapshot = {
        "title": "Climate page",
        "url": "https://example.test",
        "links": [{"href": "https://example.test/report.pdf", "text": "Report"}],
        "buttons": [{"text": "Download"}],
        "anchor_button_links": [],
    }
    mock_page = SimpleNamespace(evaluate=AsyncMock(return_value=raw_snapshot))
    mock_navigator = SimpleNamespace(open_page=AsyncMock(return_value=mock_page))
    mock_nav_instance = MagicMock()
    mock_nav_instance.__aenter__ = AsyncMock(return_value=mock_navigator)
    mock_nav_instance.__aexit__ = AsyncMock(return_value=None)
    navigator_factory = MagicMock(return_value=mock_nav_instance)
    inspector = PageSnapshotInspector(
        default_wait_ms=1234,
        navigator_factory=navigator_factory,
    )

    result = await inspector.get_page_snapshot("https://example.test")

    navigator_factory.assert_called_once_with(default_wait_ms=1234)
    mock_navigator.open_page.assert_called_once_with("https://example.test")
    assert result["document_links"] == raw_snapshot["links"]
    assert result["has_download_buttons"] is True
