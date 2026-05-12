"""Tool builders for document cleaning workflows."""

from collections.abc import Sequence
from typing import Any

from eu_climate_policy_rag.core.models import (
    DocumentPathInputModel,
    EmptyToolInputModel,
    SkipDocumentInputModel,
)
from eu_climate_policy_rag.core.tools import (
    FunctionTool,
    PydanticSchemaProvider,
    ToolContext,
    ToolMiddleware,
)
from eu_climate_policy_rag.core.tooling import ToolRegistry

OBSERVED_CLEANING_TOOLS = {
    "save_cleaned_document",
    "skip_document",
    "finalize",
}


class CleaningToolMetricsMiddleware(ToolMiddleware):
    """Collect lightweight metrics for cleaning mutation/finalization tools."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def after_call(self, context: ToolContext, result: object) -> object:
        """Observe cleaning tool results without changing model-visible values."""

        if context.tool_name not in OBSERVED_CLEANING_TOOLS:
            return result
        if not isinstance(result, dict):
            return result

        event: dict[str, object] = {"tool_name": context.tool_name}
        if "path" in result:
            event["path"] = result["path"]
        if "saved" in result:
            event["saved"] = result["saved"]
        if "skipped" in result and isinstance(result["skipped"], bool):
            event["skipped"] = result["skipped"]
        if "record_count" in result:
            event["record_count"] = result["record_count"]
        if "skipped_count" in result:
            event["skipped_count"] = result["skipped_count"]

        self.events.append(event)
        return result


def build_cleaning_tools(
    toolbox: Any,
    *,
    middleware: Sequence[ToolMiddleware] | None = None,
) -> ToolRegistry:
    """Build OpenAI tools for the cleaning curation workflow."""

    return ToolRegistry(
        [
            FunctionTool(
                name="list_documents",
                description="List fetched Markdown documents available for cleaning.",
                schema_provider=PydanticSchemaProvider(EmptyToolInputModel),
                handler=toolbox.list_documents,
            ),
            FunctionTool(
                name="inspect_document",
                description="Return a cleaning preview and skip recommendation.",
                schema_provider=PydanticSchemaProvider(DocumentPathInputModel),
                handler=toolbox.inspect_document,
            ),
            FunctionTool(
                name="save_cleaned_document",
                description="Save one cleaned JSON record for the document.",
                schema_provider=PydanticSchemaProvider(DocumentPathInputModel),
                handler=toolbox.save_cleaned_document,
            ),
            FunctionTool(
                name="skip_document",
                description="Skip one document with a reason.",
                schema_provider=PydanticSchemaProvider(SkipDocumentInputModel),
                handler=toolbox.skip_document,
            ),
            FunctionTool(
                name="finalize",
                description="Write cleaned records to the output JSON file.",
                schema_provider=PydanticSchemaProvider(EmptyToolInputModel),
                handler=toolbox.finalize,
            ),
        ],
        middleware=middleware,
    )
