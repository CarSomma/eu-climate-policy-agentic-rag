"""Tool classes used by the EU climate policy RAG agent."""

from collections.abc import Callable, Sequence
from typing import Any

from minsearch import Index

from eu_climate_policy_rag.core.models import SearchDocumentsInputModel, SearchDocumentsResultModel
from eu_climate_policy_rag.core.tools import (
    FunctionTool,
    PydanticSchemaProvider,
    ToolContext,
    ToolMiddleware,
)


class SearchDocumentsTool:
    """Search a local Minsearch index and format results for LLM tool output."""

    name = "search_documents"
    description = (
        "Search the EU climate policy document index for passages relevant to a query. "
        "Call this tool whenever you need to look up facts, targets, articles, or definitions. "
        "You may call it multiple times with different queries to gather enough context."
    )

    def __init__(
        self,
        documents: Sequence[dict[str, Any]],
        num_results: int = 5,
        max_chars_per_doc: int = 2000,
    ) -> None:
        self.documents = list(documents)
        self.num_results = num_results
        self.max_chars_per_doc = max_chars_per_doc
        self.index = self._build_index(self.documents)
        self.function_tool = FunctionTool(
            name=self.name,
            description=self.description,
            schema_provider=PydanticSchemaProvider(SearchDocumentsInputModel),
            handler=self.run,
        )

    @property
    def schema(self) -> dict[str, Any]:
        """OpenAI Responses API tool schema."""

        return self.function_tool.schema

    def search(self, query: str) -> list[dict[str, Any]]:
        """Return ranked local documents for a query."""

        validated = SearchDocumentsInputModel(query=query)
        return self.index.search(validated.query, num_results=self.num_results)

    def run(self, query: str) -> SearchDocumentsResultModel:
        """Run the search tool and return structured, validated output."""

        validated = SearchDocumentsInputModel(query=query)
        documents = self.search(validated.query)
        context = "\n\n".join(
            format_context_item(document, self.max_chars_per_doc) for document in documents
        )
        sources = list(
            dict.fromkeys(
                source
                for document in documents
                if (source := str(document.get("source", "")).strip())
            )
        )
        return SearchDocumentsResultModel(
            query=validated.query,
            context=context or "No results found.",
            sources=sources,
            documents=documents,
        )

    @staticmethod
    def _build_index(documents: Sequence[dict[str, Any]]) -> Index:
        """Build the Minsearch index used by the search tool."""

        index = Index(
            text_fields=["text", "source", "topic"],
            keyword_fields=["article"],
        )
        index.fit(documents)
        return index


def format_context_item(document: dict[str, Any], max_chars: int = 2000) -> str:
    """Format one retrieved document chunk for inclusion in the LLM prompt."""

    source = document.get("source", "unknown source")
    article = document.get("article", "unknown article")
    topic = document.get("topic", "general")
    text = document.get("text", "")
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return f"[{source} | {article} | topic: {topic}]\n{text}"


class SearchDocumentsResultMiddleware(ToolMiddleware):
    """Collect RAG sources and return model-facing context for search results."""

    def __init__(self, collect_sources: Callable[[list[str]], None]) -> None:
        self.collect_sources = collect_sources

    def after_call(self, context: ToolContext, result: object) -> object:
        """Collect sources from search results and return plain context text."""

        if (
            context.tool_name != SearchDocumentsTool.name
            or not isinstance(result, SearchDocumentsResultModel)
        ):
            return result
        self.collect_sources(result.sources)
        return result.context
