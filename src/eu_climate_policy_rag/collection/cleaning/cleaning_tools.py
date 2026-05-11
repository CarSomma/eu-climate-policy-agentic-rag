"""Tool builders for document cleaning workflows."""

from typing import Any

from eu_climate_policy_rag.core.models import (
    DocumentPathInputModel,
    EmptyToolInputModel,
    SkipDocumentInputModel,
)
from eu_climate_policy_rag.core.tooling import OpenAIFunctionTool, ToolRegistry


def build_cleaning_tools(toolbox: Any) -> ToolRegistry:
    """Build OpenAI tools for the cleaning curation workflow."""

    return ToolRegistry(
        [
            OpenAIFunctionTool(
                name="list_documents",
                description="List fetched Markdown documents available for cleaning.",
                input_model=EmptyToolInputModel,
                handler=toolbox.list_documents,
            ),
            OpenAIFunctionTool(
                name="inspect_document",
                description="Return a cleaning preview and skip recommendation.",
                input_model=DocumentPathInputModel,
                handler=toolbox.inspect_document,
            ),
            OpenAIFunctionTool(
                name="save_cleaned_document",
                description="Save one cleaned JSON record for the document.",
                input_model=DocumentPathInputModel,
                handler=toolbox.save_cleaned_document,
            ),
            OpenAIFunctionTool(
                name="skip_document",
                description="Skip one document with a reason.",
                input_model=SkipDocumentInputModel,
                handler=toolbox.skip_document,
            ),
            OpenAIFunctionTool(
                name="finalize",
                description="Write cleaned records to the output JSON file.",
                input_model=EmptyToolInputModel,
                handler=toolbox.finalize,
            ),
        ]
    )
