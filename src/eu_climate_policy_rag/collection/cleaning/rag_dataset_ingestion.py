"""CLI entrypoint for cleaning fetched Markdown into RAG records."""

import json
from pathlib import Path
from typing import Annotated

from gitsource import chunk_documents
import typer

from eu_climate_policy_rag.collection.cleaning.cleaning_agent import (
    CLEANING_AGENT_INSTRUCTIONS as CLEANING_AGENT_INSTRUCTIONS,
    CleaningCurationAgent,
)
from eu_climate_policy_rag.collection.cleaning.cleaning_toolbox import CleaningToolbox
from eu_climate_policy_rag.core.logging_utils import get_logger
from eu_climate_policy_rag.core.models import IngestionConfigModel

LOGGER = get_logger(__name__)


def run_cli(
    input_directory: Annotated[
        Path,
        typer.Option(
            "--input-directory",
            help="Directory containing fetched Markdown files.",
            show_default=True,
        ),
    ] = Path("climate_policy_docs"),
    output_path: Annotated[
        Path,
        typer.Option(
            "--output-path",
            help="JSON path to write cleaned RAG records.",
            show_default=True,
        ),
    ] = Path("data/eu_climate_policy.json"),
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="OpenAI model to use for cleaning curation.",
            show_default=True,
        ),
    ] = "gpt-5.4-mini",
    max_turns: Annotated[
        int,
        typer.Option(
            "--max-turns",
            help="Maximum LLM tool-loop turns for agentic cleaning.",
            show_default=True,
        ),
    ] = 50,
    chunk: Annotated[
        bool,
        typer.Option(
            "--chunk",
            help="Split cleaned records into sliding-window chunks for RAG.",
            show_default=True,
        ),
    ] = False,
    chunk_size: Annotated[
        int,
        typer.Option(
            "--chunk-size",
            help="Chunk size in characters when --chunk is enabled.",
            show_default=True,
        ),
    ] = 2000,
    chunk_step: Annotated[
        int,
        typer.Option(
            "--chunk-step",
            help="Sliding-window step in characters when --chunk is enabled.",
            show_default=True,
        ),
    ] = 1000,
    chroma_dir: Annotated[
        Path | None,
        typer.Option(
            "--chroma-dir",
            help="Also embed the records into a ChromaDB vector store in this folder.",
            show_default=True,
        ),
    ] = None,
) -> None:
    """Clean fetched Markdown and write one JSON record per kept file."""

    config = IngestionConfigModel(
        input_directory=input_directory,
        output_path=output_path,
        model=model,
        max_turns=max_turns,
        chunk=chunk,
        chunk_size=chunk_size,
        chunk_step=chunk_step,
    )
    message = CleaningCurationAgent(
        toolbox=CleaningToolbox(config.input_directory, config.output_path),
        model=config.model,
    ).run(max_turns=config.max_turns)
    if config.chunk:
        _chunk_output_file(
            config.output_path,
            chunk_size=config.chunk_size,
            chunk_step=config.chunk_step,
        )
    LOGGER.info(message)
    if chroma_dir is not None:
        from eu_climate_policy_rag.qa.vector_store import run_index_cli

        run_index_cli(data=config.output_path, chroma_dir=chroma_dir)


def chunk_records(
    records: list[dict[str, object]],
    *,
    chunk_size: int = 2000,
    chunk_step: int = 1000,
) -> list[dict[str, object]]:
    """Return sliding-window chunks for cleaned RAG records."""

    documents = [
        {
            **record,
            "content": str(record["text"]),
            "filename": str(record.get("file_path", record.get("source", "document"))),
        }
        for record in records
    ]
    chunks = chunk_documents(documents, size=chunk_size, step=chunk_step)
    chunk_indices: dict[str, int] = {}
    chunked_records: list[dict[str, object]] = []
    for chunk in chunks:
        content_hash = str(chunk["content_hash"])
        start = int(chunk["start"])
        chunk_index = chunk_indices.get(content_hash, 0)
        chunk_indices[content_hash] = chunk_index + 1
        record = {
            key: value
            for key, value in chunk.items()
            if key not in {"content", "filename", "start"}
        }
        record["text"] = chunk["content"]
        record["chunk_id"] = f"{content_hash}:{start}"
        record["chunk_index"] = chunk_index
        record["chunk_start"] = start
        record["parent_content_hash"] = content_hash
        chunked_records.append(record)
    return chunked_records


def _chunk_output_file(
    output_path: Path,
    *,
    chunk_size: int,
    chunk_step: int,
) -> None:
    records = json.loads(output_path.read_text(encoding="utf-8"))
    chunked_records = chunk_records(
        records,
        chunk_size=chunk_size,
        chunk_step=chunk_step,
    )
    output_path.write_text(json.dumps(chunked_records, indent=2), encoding="utf-8")
    LOGGER.info(
        "Chunked %s cleaned records into %s RAG chunks",
        len(records),
        len(chunked_records),
    )


def main() -> None:
    """Run the Typer command-line interface."""

    typer.run(run_cli)


if __name__ == "__main__":
    main()
