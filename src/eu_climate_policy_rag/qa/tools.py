"""Tool classes used by the EU climate policy RAG agent."""

from collections.abc import Callable, Sequence
from typing import Any

from chromadb.api.models.Collection import Collection

from eu_climate_policy_rag.core.models import SearchDocumentsInputModel, SearchDocumentsResultModel
from eu_climate_policy_rag.core.tools import (
    FunctionTool,
    PydanticSchemaProvider,
    ToolContext,
    ToolMiddleware,
)
from eu_climate_policy_rag.qa.vector_store import search_collection, write_records


class SearchDocumentsTool:
    """Search a ChromaDB collection and format results for LLM tool output.

    The collection stores each chunk's embedding together with its text and
    metadata (see ``vector_store.py``), so results come straight from Chroma.
    Pass a stored ``collection`` (the CLI opens ``data/chroma``), or pass
    ``documents`` to embed them into a temporary in-memory collection.

    Results are ranked by cosine distance. Vector search always returns the
    nearest records, so a query unrelated to the corpus still gets results; the
    LLM decides whether they are relevant.
    """

    name = "search_documents"
    description = (
        "Search the EU climate policy document index for passages relevant to a query. "
        "Call this tool whenever you need to look up facts, targets, articles, or definitions. "
        "You may call it multiple times with different queries to gather enough context."
    )

    def __init__(
        self,
        documents: Sequence[dict[str, Any]] = (),
        num_results: int = 5,
        max_chars_per_doc: int = 2000,
        collection: Collection | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        self.num_results = num_results
        self.max_chars_per_doc = max_chars_per_doc
        if collection is None:
            collection = write_records(list(documents), embedding_function=embedding_function)
        self.collection = collection
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
        return search_collection(self.collection, validated.query, self.num_results)

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
                if (source := document_label(document))
            )
        )
        return SearchDocumentsResultModel(
            query=validated.query,
            context=context or "No results found.",
            sources=sources,
            documents=documents,
        )


def format_context_item(document: dict[str, Any], max_chars: int = 2000) -> str:
    """Format one retrieved document chunk for inclusion in the LLM prompt."""

    source = document_label(document) or "unknown source"
    article = document.get("article", "unknown article")
    topic = document.get("topic", "general")
    text = document.get("text", "")
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    parts = [source, str(article), f"topic: {topic}"]
    for field in ("doc_type", "status", "date"):
        if document.get(field):
            parts.append(f"{field.removeprefix('doc_')}: {document[field]}")
    return f"[{' | '.join(parts)}]\n{text}"


def document_label(document: dict[str, Any]) -> str:
    """Name a document for citations: "reference: title" when known, else title or source."""

    title = str(document.get("title") or document.get("source") or "").strip()
    reference = str(document.get("reference") or "").strip()
    if reference and title:
        return f"{reference}: {title}"
    return reference or title


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
