"""Tool builders for document fetching workflows."""

from collections.abc import Callable, Mapping
from typing import Any

from eu_climate_policy_rag.core.models import (
    ClickAndCaptureInputModel,
    ClickDownloadButtonInputModel,
    ConvertToMarkdownInputModel,
    GetPageSnapshotInputModel,
    SaveContentToFileInputModel,
)
from eu_climate_policy_rag.core.tools import (
    FunctionTool,
    PydanticSchemaProvider,
    ToolContext,
    ToolMiddleware,
)
from eu_climate_policy_rag.core.tooling import ToolRegistry


class SaveContentDirectoryMiddleware(ToolMiddleware):
    """Force save operations to use the agent/toolbox output directory."""

    def __init__(self, output_directory: Callable[[], str]) -> None:
        self.output_directory = output_directory

    def before_validate(
        self,
        context: ToolContext,
        args: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Override model-provided save directories before validation."""

        if context.tool_name != "save_content_to_file":
            return args
        updated = dict(args)
        updated["directory"] = self.output_directory()
        return updated


def build_fetch_tools(toolbox: Any) -> ToolRegistry:
    """Build OpenAI tools for a document fetch agent."""

    return ToolRegistry(
        [
            FunctionTool(
                name="get_page_snapshot",
                description=(
                    "Open a page and return title, visible links, buttons, "
                    "and document candidates."
                ),
                schema_provider=PydanticSchemaProvider(GetPageSnapshotInputModel),
                handler=toolbox.get_page_snapshot,
            ),
            FunctionTool(
                name="click_and_capture",
                description="Click or fetch a document link and cache the resulting HTML or file.",
                schema_provider=PydanticSchemaProvider(ClickAndCaptureInputModel),
                handler=toolbox.click_and_capture,
            ),
            FunctionTool(
                name="click_download_button",
                description="Click a download button by visible text and cache the downloaded file.",
                schema_provider=PydanticSchemaProvider(ClickDownloadButtonInputModel),
                handler=toolbox.click_download_button,
            ),
            FunctionTool(
                name="convert_to_markdown",
                description="Convert cached HTML or files to Markdown.",
                schema_provider=PydanticSchemaProvider(ConvertToMarkdownInputModel),
                handler=toolbox.convert_to_markdown,
            ),
            FunctionTool(
                name="save_content_to_file",
                description="Save cached Markdown to a local .md file.",
                schema_provider=PydanticSchemaProvider(SaveContentToFileInputModel),
                handler=toolbox.save_content_to_file,
            ),
        ],
        middleware=[
            SaveContentDirectoryMiddleware(
                output_directory=lambda: str(toolbox.output_directory),
            )
        ],
    )
