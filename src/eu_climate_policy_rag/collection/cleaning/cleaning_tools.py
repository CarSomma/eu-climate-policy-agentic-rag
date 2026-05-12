"""Tool builders for document cleaning workflows."""

from typing import Any

from eu_climate_policy_rag.core.models import (
    DocumentPathInputModel,
    EmptyToolInputModel,
    SkipDocumentInputModel,
)
from eu_climate_policy_rag.core.tools import FunctionTool, PydanticSchemaProvider
from eu_climate_policy_rag.core.tooling import ToolRegistry


def build_cleaning_tools(toolbox: Any) -> ToolRegistry:
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
        ]
    )
