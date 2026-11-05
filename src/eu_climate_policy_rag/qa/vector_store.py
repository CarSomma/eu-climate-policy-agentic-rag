"""ChromaDB vector store: each chunk's text, metadata and embedding in one collection.

Ingestion writes the store once (``eu-climate-index`` or ``eu-climate-ingest
--chroma-dir``). The assistant then opens it and searches it directly, with no
JSON file and no embedding at start-up.

Embeddings come from Chroma's default model, all-MiniLM-L6-v2, which runs
locally and reads about the first 1,000 characters (256 word pieces) of each
input. Each chunk is embedded as its title followed by its text.
"""

import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import chromadb
import typer
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from eu_climate_policy_rag.core.logging_utils import get_logger
from eu_climate_policy_rag.core.models import CleanedDocumentRecordModel

LOGGER = get_logger(__name__)

COLLECTION_NAME = "eu_climate_policy"
DEFAULT_CHROMA_DIR = Path("data/chroma")
BATCH_SIZE = 256
METADATA_TYPES = (str, int, float, bool)


def _client(persist_directory: str | Path | None) -> Any:
    settings = Settings(anonymized_telemetry=False)
    if persist_directory is None:
        return chromadb.EphemeralClient(settings=settings)
    return chromadb.PersistentClient(path=str(persist_directory), settings=settings)


def embedding_input(record: dict[str, Any]) -> str:
    """Text the embedding model sees for one record: title, then the chunk text."""

    title = record.get("title") or record.get("source", "")
    return f"{title}\n{record.get('text', '')}"


def record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Every field except the text, dropping values Chroma cannot store (None, lists)."""

    return {
        key: value
        for key, value in record.items()
        if key != "text" and isinstance(value, METADATA_TYPES)
    }


def write_records(
    records: Sequence[dict[str, Any]],
    persist_directory: str | Path | None = None,
    collection_name: str | None = None,
    embedding_function: Any | None = None,
) -> Collection:
    """Embed records and store them with their text and metadata, replacing any old collection.

    Without ``persist_directory`` the collection lives in memory and gets a unique
    name, which keeps separate in-memory stores (for example in tests) apart.
    """

    client = _client(persist_directory)
    if collection_name is None:
        collection_name = COLLECTION_NAME if persist_directory else f"memory-{uuid.uuid4().hex}"
    if collection_name in [collection.name for collection in client.list_collections()]:
        client.delete_collection(collection_name)

    embed = embedding_function or DefaultEmbeddingFunction()
    collection = client.create_collection(
        collection_name,
        embedding_function=embed,
        metadata={"hnsw:space": "cosine"},
    )
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        collection.add(
            ids=[f"{start + offset:06d}" for offset in range(len(batch))],
            embeddings=embed([embedding_input(record) for record in batch]),
            documents=[str(record.get("text", "")) for record in batch],
            metadatas=[record_metadata(record) for record in batch],
        )
    return collection


def open_collection(
    persist_directory: str | Path = DEFAULT_CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_function: Any | None = None,
) -> Collection:
    """Open a stored collection, or explain how to build it if it is missing."""

    if Path(persist_directory).exists():
        client = _client(persist_directory)
        if collection_name in [collection.name for collection in client.list_collections()]:
            return client.get_collection(
                collection_name,
                embedding_function=embedding_function or DefaultEmbeddingFunction(),
            )
    raise FileNotFoundError(
        f"No vector store '{collection_name}' in {persist_directory}. "
        "Build it with: uv run eu-climate-index --data <records.json>"
    )


def search_collection(collection: Collection, query: str, num_results: int) -> list[dict[str, Any]]:
    """Return the nearest records, rebuilt from the stored text and metadata."""

    count = collection.count()
    if count == 0:
        return []
    result = collection.query(
        query_texts=[query],
        n_results=min(num_results, count),
        include=["documents", "metadatas"],
    )
    return [
        {**(metadata or {}), "text": text}
        for text, metadata in zip(result["documents"][0], result["metadatas"][0], strict=True)
    ]


def run_index_cli(
    data: Annotated[
        Path,
        typer.Option("--data", help="JSON file of cleaned (ideally chunked) records."),
    ] = Path("data/eu_climate_policy.json"),
    chroma_dir: Annotated[
        Path,
        typer.Option("--chroma-dir", help="Folder for the ChromaDB vector store."),
    ] = DEFAULT_CHROMA_DIR,
) -> None:
    """Embed a JSON dataset into the ChromaDB vector store used by eu-climate-ask."""

    import json

    records = [
        CleanedDocumentRecordModel.model_validate(record).model_dump()
        for record in json.loads(data.read_text(encoding="utf-8"))
    ]
    started = time.perf_counter()
    collection = write_records(records, persist_directory=chroma_dir)
    LOGGER.info(
        "Stored %s records with embeddings in %s (collection %s) in %.0f s",
        collection.count(),
        chroma_dir,
        collection.name,
        time.perf_counter() - started,
    )


def main() -> None:
    """Run the eu-climate-index command."""

    typer.run(run_index_cli)


if __name__ == "__main__":
    main()
