"""Document fetching agent for EU policy source documents."""

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from eu_climate_policy_rag.collection.fetching.content_cache import ContentCache
from eu_climate_policy_rag.collection.fetching.fetch_logging import (
    log_message_output,
    log_tool_result,
    preview_args,
)
from eu_climate_policy_rag.collection.fetching.fetch_toolbox import DocumentFetchToolbox
from eu_climate_policy_rag.collection.document_quality import DocumentQualityCheck
from eu_climate_policy_rag.collection.fetching.fetch_tools import build_fetch_tools
from eu_climate_policy_rag.core.agent import AbstractAgent
from eu_climate_policy_rag.core.logging_utils import get_logger
from eu_climate_policy_rag.core.types import PageSnapshot

LOGGER = get_logger(__name__)

PROJECT_TOPIC = """
This is an EU Climate Policy Q&A RAG system. It helps students and policy learners
navigate official EU climate policy documents. Relevant content includes:
  - EU climate legislation and regulations (e.g. European Climate Law, EU ETS)
  - EU climate targets (2030, 2040, net-zero 2050)
  - EU Commission communications and proposals on climate policy
  - Staff working documents, Q&As, and press releases about EU climate policy
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


class DocumentFetchAgent(AbstractAgent):
    """Coordinate OpenAI tool calls for document capture."""

    def __init__(
        self,
        toolbox: DocumentFetchToolbox | None = None,
        openai_client: OpenAI | None = None,
        model: str = "gpt-5.4-mini",
        cache: ContentCache | None = None,
        quality_check: DocumentQualityCheck | None = None,
        output_directory: str | Path = "climate_policy_docs",
        instructions: str = FETCH_AGENT_INSTRUCTIONS,
        max_turns: int = 12,
    ) -> None:
        self.toolbox = toolbox or DocumentFetchToolbox(
            cache=cache,
            quality_check=quality_check,
            output_directory=output_directory,
        )
        super().__init__(
            openai_client=openai_client,
            model=model,
            instructions=instructions,
            tools=build_fetch_tools(self.toolbox),
            max_turns=max_turns,
        )
        LOGGER.debug("Fetch agent output directory: %s", self.output_directory)

    @property
    def cache(self) -> ContentCache:
        """Cache used by the fetch toolbox."""

        return self.toolbox.cache

    @property
    def quality_check(self) -> DocumentQualityCheck:
        """Quality check used by the fetch toolbox."""

        return self.toolbox.quality_check

    @property
    def output_directory(self) -> Path:
        """Directory where fetched Markdown is saved."""

        return self.toolbox.output_directory

    @output_directory.setter
    def output_directory(self, value: str | Path) -> None:
        self.toolbox.output_directory = Path(value)

    async def get_page_snapshot(self, url: str) -> PageSnapshot:
        """Compatibility wrapper for the toolbox page snapshot operation."""

        return await self.toolbox.get_page_snapshot(url)

    async def click_download_button(self, url: str, button_text: str) -> dict[str, Any]:
        """Compatibility wrapper for the toolbox download-button operation."""

        return await self.toolbox.click_download_button(url, button_text)

    async def click_and_capture(self, url: str, link_href: str) -> dict[str, Any]:
        """Compatibility wrapper for the toolbox link-capture operation."""

        return await self.toolbox.click_and_capture(url, link_href)

    def convert_to_markdown(self, content_id: str) -> dict[str, Any]:
        """Compatibility wrapper for the toolbox Markdown conversion operation."""

        return self.toolbox.convert_to_markdown(content_id)

    def save_content_to_file(
        self,
        markdown_id: str,
        filename: str,
        directory: str = "climate_policy_docs",
    ) -> dict[str, Any]:
        """Compatibility wrapper for the toolbox file-save operation."""

        return self.toolbox.save_content_to_file(markdown_id, filename, directory)

    async def run_tool(self, name: str, args: dict[str, Any]) -> str:
        """Dispatch OpenAI tool calls to registered fetch tools."""

        result = await self._run_tool_by_name(name, args)
        return json.dumps(result)

    async def _run_tool_by_name(self, name: str, args: dict[str, Any]) -> Any:
        """Validate and run a registered fetch tool."""

        tool_args = dict(args)
        if name == "save_content_to_file":
            tool_args["directory"] = str(self.output_directory)

        if self.tools.get(name) is None:
            LOGGER.error("Unknown tool requested: %s", name)
            return {"error": f"Unknown tool: {name}"}

        LOGGER.debug("Running tool %s with args %s", name, preview_args(tool_args))
        return await self.tools.run(name, tool_args)

    async def _execute_tool_call_async(self, tool_call: Any) -> dict[str, Any]:
        """Dispatch a function_call message to the matching fetch tool."""

        arguments = json.loads(tool_call.arguments)
        LOGGER.info("Calling tool: %s args=%s", tool_call.name, preview_args(arguments))
        result = await self._run_tool_by_name(tool_call.name, arguments)
        output = json.dumps(result)
        self._print_tool_result(tool_call.name, output)
        return {
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": output,
        }

    def _on_message(self, message: Any) -> None:
        """Log LLM text messages during the fetch loop."""

        log_message_output([message])

    def run(self, query: str) -> Any:
        """Implement the abstract ``run`` method; use ``fetch_document`` for async entry."""

        raise NotImplementedError("DocumentFetchAgent is async — use fetch_document() instead.")

    async def fetch_document(
        self,
        url: str,
        max_turns: int | None = None,
    ) -> str:
        """Use an LLM tool loop to fetch, convert, and save a relevant document."""

        if max_turns is not None:
            self.max_turns = max_turns
        LOGGER.info("Starting fetch-agent loop for URL: %s", url)
        final_answer, _ = await self._run_loop_async(
            f"Fetch and save the document at: {url}"
        )
        return final_answer

    @staticmethod
    def _print_message_output(output: list[Any]) -> None:
        log_message_output(output)

    @staticmethod
    def _print_tool_result(tool_name: str, result: str) -> None:
        log_tool_result(tool_name, result)
