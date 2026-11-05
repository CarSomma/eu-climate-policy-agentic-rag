"""Tool classes used by the EU climate policy RAG agent."""

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from eu_climate_policy_rag.core.models import SearchDocumentsInputModel, SearchDocumentsResultModel
from eu_climate_policy_rag.core.tools import (
    FunctionTool,
    PydanticSchemaProvider,
    ToolContext,
    ToolMiddleware,
)


class SearchDocumentsTool:
    """Search a ChromaDB vector index and format results for LLM tool output.

    Each record is embedded as its title (or source) followed by its text.
    By default Chroma uses the local all-MiniLM-L6-v2 model, which reads about
    the first 1,000 characters of each record (256 word pieces). With 2,000
    character chunks and a 1,000 character step, every part of a document still
    falls inside the embedded half of some chunk.

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
        documents: Sequence[dict[str, Any]],
        num_results: int = 5,
        max_chars_per_doc: int = 2000,
        persist_directory: str | Path | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        self.documents = list(documents)
        self.num_results = num_results
        self.max_chars_per_doc = max_chars_per_doc
        self.collection = self._build_collection(
            self.documents,
            persist_directory=persist_directory,
            embedding_function=embedding_function,
        )
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
        if self.collection is None:
            return []
        result = self.collection.query(
            query_texts=[validated.query],
            n_results=min(self.num_results, len(self.documents)),
            include=[],
        )
        return [self.documents[int(record_id)] for record_id in result["ids"][0]]

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
    def _build_collection(
        documents: Sequence[dict[str, Any]],
        persist_directory: str | Path | None = None,
        embedding_function: Any | None = None,
    ) -> Collection | None:
        """Embed the records into a Chroma collection, reusing a persisted one if present.

        Record ids are list positions, so search results map straight back to the
        original dicts. The collection name is a hash of the records, so a changed
        dataset gets a new collection instead of stale embeddings.
        """

        if not documents:
            return None

        settings = Settings(anonymized_telemetry=False)
        if persist_directory is None:
            client = chromadb.EphemeralClient(settings=settings)
        else:
            client = chromadb.PersistentClient(path=str(persist_directory), settings=settings)

        texts = [
            f"{document.get('title') or document.get('source', '')}\n{document.get('text', '')}"
            for document in documents
        ]
        fingerprint = hashlib.sha1("\x00".join(texts).encode("utf-8")).hexdigest()[:16]
        options: dict[str, Any] = {"metadata": {"hnsw:space": "cosine"}}
        if embedding_function is not None:
            options["embedding_function"] = embedding_function
        collection = client.get_or_create_collection(f"climate-docs-{fingerprint}", **options)

        if collection.count() != len(documents):
            if collection.count():
                client.delete_collection(collection.name)
                collection = client.create_collection(collection.name, **options)
            ids = [str(position) for position in range(len(documents))]
            batch_size = 500
            for start in range(0, len(documents), batch_size):
                collection.add(
                    ids=ids[start : start + batch_size],
                    documents=texts[start : start + batch_size],
                )
        return collection


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
