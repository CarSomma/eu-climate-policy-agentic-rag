"""Tool builders for document fetching workflows."""

from typing import Any

from eu_climate_policy_rag.core.models import (
    ClickAndCaptureInputModel,
    ClickDownloadButtonInputModel,
    ConvertToMarkdownInputModel,
    GetPageSnapshotInputModel,
    SaveContentToFileInputModel,
)
from eu_climate_policy_rag.core.tooling import OpenAIFunctionTool, ToolRegistry


def build_fetch_tools(toolbox: Any) -> ToolRegistry:
    """Build OpenAI tools for a document fetch agent."""

    return ToolRegistry(
        [
            OpenAIFunctionTool(
                name="get_page_snapshot",
                description=(
                    "Open a page and return title, visible links, buttons, "
                    "and document candidates."
                ),
                input_model=GetPageSnapshotInputModel,
                handler=toolbox.get_page_snapshot,
            ),
            OpenAIFunctionTool(
                name="click_and_capture",
                description="Click or fetch a document link and cache the resulting HTML or file.",
                input_model=ClickAndCaptureInputModel,
                handler=toolbox.click_and_capture,
            ),
            OpenAIFunctionTool(
                name="click_download_button",
                description="Click a download button by visible text and cache the downloaded file.",
                input_model=ClickDownloadButtonInputModel,
                handler=toolbox.click_download_button,
            ),
            OpenAIFunctionTool(
                name="convert_to_markdown",
                description="Convert cached HTML or files to Markdown.",
                input_model=ConvertToMarkdownInputModel,
                handler=toolbox.convert_to_markdown,
            ),
            OpenAIFunctionTool(
                name="save_content_to_file",
                description="Save cached Markdown to a local .md file.",
                input_model=SaveContentToFileInputModel,
                handler=toolbox.save_content_to_file,
            ),
        ]
    )
