"""Notebook-friendly document fetching agent for EU policy source documents."""

import io
import json
import os
import re
import hashlib
import tempfile
import uuid
from pathlib import Path
from typing import Any

from markitdown import MarkItDown
from openai import OpenAI
from playwright.async_api import async_playwright

from eu_climate_policy_rag.collection.url_utils import (
    detect_format_from_url,
    filename_from_url,
    is_document_url,
    is_download_button,
)
from eu_climate_policy_rag.core.logging_utils import get_logger
from eu_climate_policy_rag.core.models import (
    PageSnapshotModel,
    PreselectionResultModel,
)
from eu_climate_policy_rag.core.types import CachedContent, Link, PageSnapshot

LOGGER = get_logger(__name__)

PROJECT_TOPIC = """
This is an EU Climate Policy Q&A RAG system. It helps students and policy learners
navigate official EU climate policy documents. Relevant content includes:
  - EU climate legislation and regulations (e.g. European Climate Law, EU ETS)
  - EU climate targets (2030, 2040, net-zero 2050)
  - EU Commission communications and proposals on climate policy
  - Staff working documents, factsheets, Q&As, and press releases about EU climate policy
  - The Clean Industrial Deal and related EU green transition documents

IRRELEVANT content includes: cookie policies, login pages, navigation-only pages,
general EU portal homepages, unrelated legislation, press releases about non-climate topics,
and any document that does not substantively discuss EU climate policy.
""".strip()


FETCH_AGENT_INSTRUCTIONS = f"""
You are a document-fetching agent. Given a URL, your goal is to retrieve, convert, and
save the document as Markdown, but only if it is relevant to the project topic below.

PROJECT TOPIC:
{PROJECT_TOPIC}

Follow these steps in order:
1. Call get_page_snapshot on the starting URL to inspect the page.
2. If has_downloadable_documents is true, call click_and_capture with the best document href.
3. If has_download_buttons is true, call click_download_button using the button text.
4. Otherwise, choose the most likely document link and call click_and_capture.
5. If an HTML result lands on another download page, inspect that page and repeat.
6. Call convert_to_markdown with the content_id.
7. Read the returned title and preview. Save only documents relevant to EU climate policy.
8. Call save_content_to_file with a descriptive markdown filename.
""".strip()


class ContentCache:
    """Small explicit cache for large HTML, binary files, and converted Markdown."""

    def __init__(self) -> None:
        self._items: dict[str, CachedContent] = {}

    def add(self, item: CachedContent) -> str:
        """Store a cache item and return its short content identifier."""

        content_id = uuid.uuid4().hex[:8]
        self._items[content_id] = item
        return content_id

    def pop(self, content_id: str) -> CachedContent | None:
        """Remove and return a cached item by content identifier."""

        return self._items.pop(content_id, None)


PreselectionResult = PreselectionResultModel


class DocumentPreselector:
    """Conservative deterministic checks for fetched EU climate policy documents."""

    def __init__(
        self,
        minimum_characters: int = 800,
        climate_keyword_threshold: int = 2,
        navigation_marker_threshold: int = 5,
    ) -> None:
        self.minimum_characters = minimum_characters
        self.climate_keyword_threshold = climate_keyword_threshold
        self.navigation_marker_threshold = navigation_marker_threshold

    def assess(
        self,
        title: str,
        markdown: str,
        existing_hashes: set[str] | None = None,
    ) -> PreselectionResult:
        """Return whether converted Markdown is worth saving into fetched data."""

        normalized_text = normalize_markdown_for_hash(markdown)
        content_hash = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()
        existing_hashes = existing_hashes or set()

        if content_hash in existing_hashes:
            return PreselectionResult(
                accepted=False,
                reason="duplicate content already exists",
                content_hash=content_hash,
            )

        if len(normalized_text) < self.minimum_characters:
            return PreselectionResult(
                accepted=False,
                reason="content is too short to be useful",
                content_hash=content_hash,
            )

        searchable_text = f"{title}\n{normalized_text}".lower()
        navigation_hits = _count_keyword_hits(searchable_text, NAVIGATION_MARKERS)
        climate_hits = _count_keyword_hits(searchable_text, CLIMATE_KEYWORDS)
        if (
            navigation_hits >= self.navigation_marker_threshold
            and climate_hits < self.climate_keyword_threshold * 2
        ):
            return PreselectionResult(
                accepted=False,
                reason="content appears to be mostly page navigation",
                content_hash=content_hash,
            )

        if climate_hits < self.climate_keyword_threshold:
            return PreselectionResult(
                accepted=False,
                reason="content is not clearly about EU climate policy",
                content_hash=content_hash,
            )

        return PreselectionResult(
            accepted=True,
            reason="accepted",
            content_hash=content_hash,
        )


class DocumentFetchAgent:
    """Coordinate Playwright, MarkItDown, and OpenAI tool calls for document capture."""

    def __init__(
        self,
        openai_client: OpenAI | None = None,
        model: str = "gpt-5.4-mini",
        cache: ContentCache | None = None,
        preselector: DocumentPreselector | None = None,
        output_directory: str | Path = "climate_policy_docs",
    ) -> None:
        self.openai_client = openai_client or OpenAI()
        self.model = model
        self.cache = cache or ContentCache()
        self.preselector = preselector or DocumentPreselector()
        self.output_directory = Path(output_directory)
        LOGGER.debug("Fetch agent output directory: %s", self.output_directory)

    async def get_page_snapshot(self, url: str) -> PageSnapshot:
        """Open a page and return visible links, buttons, and document candidates."""

        LOGGER.info("Inspecting page: %s", url)
        if is_document_url(url):
            filename = filename_from_url(url)
            LOGGER.info("URL is a direct document download: %s", filename)
            return PageSnapshotModel(
                title=filename,
                url=url,
                links=[{"text": filename, "href": url}],
                buttons=[],
                download_buttons=[],
                has_downloadable_documents=True,
                has_download_buttons=False,
                document_links=[{"text": filename, "href": url}],
                note="Direct download URL; fetch it with click_and_capture.",
            ).model_dump(exclude_none=True)

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_timeout(2000)
            snapshot = await page.evaluate(_PAGE_SNAPSHOT_SCRIPT)
            await browser.close()

        links = snapshot.get("links", [])
        document_links = self._collect_document_links(
            links,
            snapshot.get("anchor_button_links", []),
        )
        download_buttons = [
            button
            for button in snapshot.get("buttons", [])
            if is_download_button(str(button["text"]))
        ]
        LOGGER.info(
            "Snapshot ready: %s links, %s document links, %s download buttons",
            len(links),
            len(document_links),
            len(download_buttons),
        )

        return PageSnapshotModel(
            title=snapshot["title"],
            url=snapshot["url"],
            links=links,
            buttons=snapshot.get("buttons", []),
            download_buttons=download_buttons,
            has_downloadable_documents=bool(document_links),
            has_download_buttons=bool(download_buttons),
            document_links=document_links,
        ).model_dump(exclude_none=True)

    async def click_download_button(self, url: str, button_text: str) -> dict[str, Any]:
        """Click a visible download button and cache the downloaded file or HTML fallback."""

        LOGGER.info("Clicking download button %r on %s", button_text, url)
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_timeout(1500)

            selector = (
                f'button:has-text("{button_text}"), '
                f'[role="button"]:has-text("{button_text}"), '
                f'input[value="{button_text}"]'
            )
            button = await page.query_selector(selector)
            if button is None:
                await browser.close()
                LOGGER.warning("No download button matched %r", button_text)
                return {"error": f"No button found matching text '{button_text}'"}

            try:
                async with page.expect_download(timeout=15000) as download_info:
                    await button.click()
                download = await download_info.value
                result = await self._cache_download(download.suggested_filename, download.save_as)
                await browser.close()
                LOGGER.info("Downloaded file: %s", download.suggested_filename)
                return {"url": url, **result}
            except Exception:
                LOGGER.warning("No file download detected; caching landed HTML instead")
                await page.wait_for_load_state("networkidle")
                await page.wait_for_timeout(2000)
                result = self._cache_html(page.url, await page.title(), await page.content())
                await browser.close()
                return result

    async def click_and_capture(self, url: str, link_href: str) -> dict[str, Any]:
        """Fetch a document link directly or navigate to an HTML page and cache it."""

        LOGGER.info("Capturing link: %s", link_href)
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context()

            if is_document_url(link_href):
                response = await context.request.get(link_href)
                content_disposition = response.headers.get("content-disposition", "")
                filename = _filename_from_content_disposition(content_disposition)
                filename = filename or filename_from_url(link_href)
                result = self._cache_file_bytes(filename, await response.body())
                await browser.close()
                LOGGER.info("Fetched direct document: %s", filename)
                return {"url": link_href, **result}

            page = await context.new_page()
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_timeout(1500)

            try:
                await page.click(f'a[href="{link_href}"]')
                await page.wait_for_load_state("networkidle")
                await page.wait_for_timeout(2000)
            except Exception:
                await page.goto(link_href, wait_until="networkidle")
                await page.wait_for_timeout(2000)

            result = self._cache_html(page.url, await page.title(), await page.content())
            await browser.close()
            return result

    def convert_to_markdown(self, content_id: str) -> dict[str, Any]:
        """Convert cached HTML or binary content to Markdown and cache the result."""

        cached = self.cache.pop(content_id)
        if cached is None:
            LOGGER.error("Unknown content_id requested: %s", content_id)
            return {"error": f"content_id '{content_id}' not found in cache"}

        LOGGER.info("Converting cached %s content to Markdown", cached["format"])
        markdown_converter = MarkItDown()
        if cached["format"] == "html":
            converted = markdown_converter.convert_stream(
                io.BytesIO(cached["html"].encode("utf-8")),
                file_extension=".html",
            )
        else:
            converted = markdown_converter.convert(cached["tmp_path"])
            os.unlink(cached["tmp_path"])

        markdown_id = self.cache.add(
            {
                "markdown": converted.markdown,
                "title": cached["title"],
                "content_hash": hashlib.sha1(
                    normalize_markdown_for_hash(converted.markdown).encode("utf-8")
                ).hexdigest(),
            }
        )
        preselection = self.preselector.assess(cached["title"], converted.markdown)
        LOGGER.info(
            "Converted %s characters; preselection=%s",
            len(converted.markdown),
            preselection.reason,
        )
        return {
            "markdown_id": markdown_id,
            "title": cached["title"],
            "length": len(converted.markdown),
            "preview": converted.markdown[:800],
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
        """Write cached Markdown content to disk."""

        cached = self.cache.pop(markdown_id)
        if cached is None:
            LOGGER.error("Unknown markdown_id requested: %s", markdown_id)
            return {"error": f"markdown_id '{markdown_id}' not found in cache"}

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        markdown = cached["markdown"]
        LOGGER.info("Saving Markdown candidate to %s", output_dir)
        preselection = self.preselector.assess(
            cached["title"],
            markdown,
            existing_hashes=_existing_markdown_hashes(output_dir),
        )
        if not preselection.accepted:
            LOGGER.warning("Rejected Markdown before save: %s", preselection.reason)
            return {
                "saved": False,
                "rejected": True,
                "reason": preselection.reason,
                "content_hash": preselection.content_hash,
            }

        filepath = output_dir / filename
        filepath.write_text(markdown, encoding="utf-8")
        LOGGER.info("Saved Markdown: %s", filepath)
        return {
            "saved": True,
            "path": str(filepath),
            "bytes": len(markdown.encode("utf-8")),
            "content_hash": preselection.content_hash,
        }

    async def run_tool(self, name: str, args: dict[str, Any]) -> str:
        """Dispatch OpenAI tool calls to instance methods."""

        tool_map = {
            "get_page_snapshot": self.get_page_snapshot,
            "click_and_capture": self.click_and_capture,
            "click_download_button": self.click_download_button,
            "convert_to_markdown": self.convert_to_markdown,
            "save_content_to_file": self.save_content_to_file,
        }
        tool = tool_map.get(name)
        if tool is None:
            LOGGER.error("Unknown tool requested: %s", name)
            return json.dumps({"error": f"Unknown tool: {name}"})

        if name == "save_content_to_file":
            args["directory"] = str(self.output_directory)

        LOGGER.debug("Running tool %s with args %s", name, _preview_args(args))
        result = await tool(**args) if name in _ASYNC_TOOLS else tool(**args)
        return json.dumps(result)

    async def fetch_document(self, url: str, max_turns: int = 12) -> str:
        """Use an LLM tool loop to fetch, convert, and save a relevant document."""

        messages: list[Any] = [
            {"role": "system", "content": FETCH_AGENT_INSTRUCTIONS},
            {"role": "user", "content": f"Fetch and save the document at: {url}"},
        ]
        tools = [
            GET_PAGE_SNAPSHOT_TOOL,
            CLICK_AND_CAPTURE_TOOL,
            CLICK_DOWNLOAD_BUTTON_TOOL,
            CONVERT_TO_MARKDOWN_TOOL,
            SAVE_CONTENT_TO_FILE_TOOL,
        ]

        LOGGER.info("Starting fetch-agent loop for URL: %s", url)
        for turn in range(max_turns):
            LOGGER.info("Fetch turn %s/%s: asking LLM what to do next", turn + 1, max_turns)
            response = self.openai_client.responses.create(
                model=self.model,
                input=messages,
                tools=tools,
            )

            self._print_message_output(response.output)
            tool_calls = [item for item in response.output if item.type == "function_call"]
            if not tool_calls:
                LOGGER.info("Fetch agent finished with no more tool calls")
                return response.output_text

            messages.extend(response.output)
            for tool_call in tool_calls:
                args = json.loads(tool_call.arguments)
                LOGGER.info("Calling tool: %s args=%s", tool_call.name, _preview_args(args))
                result = await self.run_tool(tool_call.name, args)
                self._print_tool_result(tool_call.name, result)
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": result,
                    }
                )

        return "Agent reached max turns without a final answer."

    @staticmethod
    def _collect_document_links(
        links: list[Link],
        anchor_button_links: list[Link],
    ) -> list[Link]:
        document_links = [link for link in links if is_document_url(link["href"])]
        seen_hrefs = {link["href"] for link in document_links}
        for link in anchor_button_links:
            if link["href"] not in seen_hrefs:
                document_links.append(link)
                seen_hrefs.add(link["href"])
        return document_links

    async def _cache_download(self, filename: str, save_as: Any) -> dict[str, Any]:
        suffix = Path(filename).suffix or ".bin"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        await save_as(tmp.name)

        fmt = "pdf" if suffix.lower() == ".pdf" else "binary"
        content_id = self.cache.add({"format": fmt, "tmp_path": tmp.name, "title": filename})
        return {"title": filename, "format": fmt, "filename": filename, "content_id": content_id}

    def _cache_file_bytes(self, filename: str, file_bytes: bytes) -> dict[str, Any]:
        suffix = Path(filename).suffix.lower() or ".bin"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(file_bytes)
        tmp.close()

        fmt = "pdf" if suffix == ".pdf" else detect_format_from_url(filename)
        content_id = self.cache.add({"format": fmt, "tmp_path": tmp.name, "title": filename})
        return {"title": filename, "format": fmt, "content_id": content_id}

    def _cache_html(self, url: str, title: str, html: str) -> dict[str, Any]:
        content_id = self.cache.add({"format": "html", "html": html, "title": title})
        return {"url": url, "title": title, "format": "html", "content_id": content_id}

    @staticmethod
    def _print_message_output(output: list[Any]) -> None:
        for item in output:
            if getattr(item, "type", None) != "message":
                continue
            for block in getattr(item, "content", []):
                text = getattr(block, "text", None)
                if text:
                    LOGGER.info("LLM: %s", text.strip())

    @staticmethod
    def _print_tool_result(tool_name: str, result: str) -> None:
        result_preview = json.loads(result)
        if tool_name == "get_page_snapshot":
            LOGGER.info(
                'Page "%s": %s links, %s doc links, %s download buttons',
                result_preview.get("title", ""),
                len(result_preview.get("links", [])),
                len(result_preview.get("document_links", [])),
                len(result_preview.get("download_buttons", [])),
            )
        elif tool_name in {"click_and_capture", "click_download_button"}:
            LOGGER.info(
                'Fetched "%s" (%s), content_id=%s',
                result_preview.get("title", ""),
                result_preview.get("format", "?"),
                result_preview.get("content_id", "?"),
            )
        elif tool_name == "convert_to_markdown":
            preselection = result_preview.get("preselection", {})
            LOGGER.info(
                "Converted %s chars, markdown_id=%s, preselection=%s",
                f"{result_preview.get('length', 0):,}",
                result_preview.get("markdown_id", "?"),
                preselection.get("reason", "?"),
            )
        elif tool_name == "save_content_to_file":
            if result_preview.get("rejected"):
                LOGGER.warning("Rejected save: %s", result_preview.get("reason", "?"))
                return
            LOGGER.info(
                "Saved to %s (%s bytes)",
                result_preview.get("path", "?"),
                f"{result_preview.get('bytes', 0):,}",
            )
        else:
            LOGGER.info("Tool result: %s", json.dumps(result_preview)[:140])


def _filename_from_content_disposition(content_disposition: str) -> str | None:
    if "filename=" not in content_disposition:
        return None
    return content_disposition.split("filename=", maxsplit=1)[-1].strip().strip('"')


def _preview_args(args: dict[str, Any]) -> dict[str, str]:
    return {key: str(value)[:80] for key, value in args.items()}


def normalize_markdown_for_hash(markdown: str) -> str:
    """Normalize Markdown enough for stable duplicate detection."""

    normalized_lines = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped:
            normalized_lines.append(re.sub(r"\s+", " ", stripped))
    return "\n".join(normalized_lines)


def _existing_markdown_hashes(directory: Path) -> set[str]:
    hashes = set()
    for path in directory.glob("*.md"):
        markdown = path.read_text(encoding="utf-8", errors="ignore")
        normalized = normalize_markdown_for_hash(markdown)
        hashes.add(hashlib.sha1(normalized.encode("utf-8")).hexdigest())
    return hashes


def _count_keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


CLIMATE_KEYWORDS = (
    "climate",
    "greenhouse gas",
    "ghg",
    "emission",
    "decarbon",
    "net-zero",
    "net zero",
    "carbon",
    "eu ets",
    "climate neutrality",
    "european climate law",
    "adaptation",
    "paris agreement",
    "energy transition",
)

NAVIGATION_MARKERS = (
    "accept all cookies",
    "accept only essential cookies",
    "skip to main content",
    "select your language",
    "official website of the european union",
    "how do you know?",
    "see all eu institutions",
    "type of search results",
    "search on:",
    "switch to mobile",
    "switch to desktop",
    "log in",
    "my eur-lex",
)


_ASYNC_TOOLS = {
    "get_page_snapshot",
    "click_and_capture",
    "click_download_button",
}

_PAGE_SNAPSHOT_SCRIPT = """() => {
    const links = [...document.querySelectorAll('a[href]')]
        .map((a, i) => ({
            index: i,
            text: a.textContent.trim().slice(0, 120),
            href: a.href
        }))
        .filter(l => l.text)
        .slice(0, 60);

    const anchorButtonLinks = [...document.querySelectorAll('a[href][role="button"]')]
        .map(a => ({
            text: (a.textContent.trim().slice(0, 120)
                   || a.getAttribute('aria-label')
                   || 'Download'),
            href: a.href
        }))
        .filter(l => {
            const href = l.href.toLowerCase();
            return href.includes('document') && href.includes('download');
        })
        .slice(0, 20);

    const buttons = [...document.querySelectorAll(
        'button, input[type="button"], input[type="submit"]'
    )]
        .map((b, i) => ({
            index: i,
            text: (b.textContent || b.value || '').trim().slice(0, 120)
        }))
        .filter(b => b.text)
        .slice(0, 20);

    return {
        title: document.title,
        url: window.location.href,
        links: links,
        anchor_button_links: anchorButtonLinks,
        buttons: buttons
    };
}"""

GET_PAGE_SNAPSHOT_TOOL = {
    "type": "function",
    "name": "get_page_snapshot",
    "description": "Open a page and return title, visible links, buttons, and document candidates.",
    "parameters": {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    },
}

CLICK_AND_CAPTURE_TOOL = {
    "type": "function",
    "name": "click_and_capture",
    "description": "Click or fetch a document link and cache the resulting HTML or file.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "link_href": {"type": "string"},
        },
        "required": ["url", "link_href"],
    },
}

CLICK_DOWNLOAD_BUTTON_TOOL = {
    "type": "function",
    "name": "click_download_button",
    "description": "Click a download button by visible text and cache the downloaded file.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "button_text": {"type": "string"},
        },
        "required": ["url", "button_text"],
    },
}

CONVERT_TO_MARKDOWN_TOOL = {
    "type": "function",
    "name": "convert_to_markdown",
    "description": "Convert cached HTML or files to Markdown.",
    "parameters": {
        "type": "object",
        "properties": {"content_id": {"type": "string"}},
        "required": ["content_id"],
    },
}

SAVE_CONTENT_TO_FILE_TOOL = {
    "type": "function",
    "name": "save_content_to_file",
    "description": "Save cached Markdown to a local .md file.",
    "parameters": {
        "type": "object",
        "properties": {
            "markdown_id": {"type": "string"},
            "filename": {"type": "string"},
            "directory": {"type": "string"},
        },
        "required": ["markdown_id", "filename"],
    },
}
