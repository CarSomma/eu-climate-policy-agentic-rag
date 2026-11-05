import json
from unittest.mock import MagicMock

import pytest

from eu_climate_policy_rag.qa.rag import ClimatePolicyAgent
from eu_climate_policy_rag.qa.vector_store import (
    open_collection,
    record_metadata,
    run_index_cli,
    search_collection,
    write_records,
)

CHUNK = {
    "source": "climate_policy_docs",
    "title": "European Climate Law",
    "topic": "general",
    "article": "document",
    "file_path": "climate_policy_docs/law.md",
    "text": "Net greenhouse gas emissions shall be reduced by at least 55% by 2030.",
    "content_hash": "law1",
    "chunk_id": "law1:0",
    "chunk_index": 0,
    "chunk_start": 0,
    "parent_content_hash": "law1",
}


def test_record_metadata_keeps_fields_but_not_text_or_none() -> None:
    metadata = record_metadata({**CHUNK, "title": None})

    assert "text" not in metadata
    assert "title" not in metadata
    assert metadata["chunk_id"] == "law1:0"
    assert metadata["chunk_index"] == 0


def test_stored_collection_returns_text_and_metadata(tmp_path) -> None:
    write_records([CHUNK], persist_directory=tmp_path)

    collection = open_collection(tmp_path)
    results = search_collection(collection, "2030 emissions target", num_results=5)

    assert collection.count() == 1
    assert results == [CHUNK]


def test_write_records_replaces_previous_collection(tmp_path) -> None:
    write_records([CHUNK, {**CHUNK, "chunk_id": "law1:1"}], persist_directory=tmp_path)
    write_records([CHUNK], persist_directory=tmp_path)

    assert open_collection(tmp_path).count() == 1


def test_open_collection_explains_missing_store(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="eu-climate-index"):
        open_collection(tmp_path / "missing")


def test_index_cli_builds_store_from_json(tmp_path) -> None:
    data = tmp_path / "records.json"
    data.write_text(json.dumps([CHUNK]), encoding="utf-8")

    run_index_cli(data=data, chroma_dir=tmp_path / "chroma")

    assert open_collection(tmp_path / "chroma").count() == 1


def test_agent_from_chroma_searches_stored_chunks(tmp_path) -> None:
    write_records([CHUNK], persist_directory=tmp_path)

    agent = ClimatePolicyAgent.from_chroma(tmp_path, openai_client=MagicMock())
    result = agent.search_tool.run("What is the 2030 target?")

    assert result.documents[0]["chunk_id"] == "law1:0"
    assert "at least 55%" in result.context


LAW_METADATA = {
    "climate_policy_docs/law.md": {
        "reference": "Regulation (EU) 2021/1119",
        "title": "European Climate Law",
        "doc_type": "law",
        "status": "in force",
        "date": "2021-06-30",
    }
}


def test_index_cli_attaches_document_metadata_to_every_chunk(tmp_path) -> None:
    data = tmp_path / "records.json"
    data.write_text(json.dumps([CHUNK, {**CHUNK, "chunk_id": "law1:1"}]), encoding="utf-8")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(LAW_METADATA), encoding="utf-8")

    run_index_cli(data=data, chroma_dir=tmp_path / "chroma", metadata=metadata)
    results = search_collection(open_collection(tmp_path / "chroma"), "2030 target", num_results=5)

    assert len(results) == 2
    assert all(result["reference"] == "Regulation (EU) 2021/1119" for result in results)
    assert all(result["status"] == "in force" for result in results)


def test_agent_context_header_shows_reference_type_and_status(tmp_path) -> None:
    write_records([{**CHUNK, **LAW_METADATA["climate_policy_docs/law.md"]}], persist_directory=tmp_path)

    agent = ClimatePolicyAgent.from_chroma(tmp_path, openai_client=MagicMock())
    result = agent.search_tool.run("What is the 2030 target?")

    assert result.context.startswith(
        "[Regulation (EU) 2021/1119: European Climate Law | document | topic: general"
        " | type: law | status: in force | date: 2021-06-30]"
    )
    assert result.sources == ["Regulation (EU) 2021/1119: European Climate Law"]
