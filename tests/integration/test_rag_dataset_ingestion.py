import inspect
import json

from eu_climate_policy_rag.collection.cleaning import rag_dataset_ingestion
from eu_climate_policy_rag.collection.cleaning.rag_dataset_ingestion import (
    chunk_records,
    run_cli,
)


def test_run_cli_calls_cleaning_agent(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    output_path = tmp_path / "out.json"

    # Use list to capture instances without class-level mutable state
    captured_agents = []

    class MockAgent:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.run_calls = []
            captured_agents.append(self)

        def run(self, max_turns: int) -> str:
            self.run_calls.append(max_turns)
            return "Cleaned 0 docs."

    monkeypatch.setattr(rag_dataset_ingestion, "CleaningCurationAgent", MockAgent)

    run_cli(
        input_directory=input_dir,
        output_path=output_path,
        model="gpt-4o-mini",
        max_turns=5,
    )

    assert len(captured_agents) == 1
    assert captured_agents[0].kwargs["model"] == "gpt-4o-mini"
    assert captured_agents[0].run_calls == [5]


def test_run_cli_writes_output_file(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    (input_dir / "doc.md").write_text("# Climate Law\nContent here.", encoding="utf-8")
    output_path = tmp_path / "out.json"

    captured_agents = []

    class MockAgent:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            captured_agents.append(self)

        def run(self, max_turns: int) -> str:
            # Simulate creating output
            sample_record = {
                "source": "Climate Law",
                "article": "document",
                "topic": "climate_law",
                "text": "Content here.",
                "file_path": str(input_dir / "doc.md"),
                "content_hash": "abc123",
            }
            output_path.write_text(json.dumps([sample_record]), encoding="utf-8")
            return "Cleaned 1 docs."

    monkeypatch.setattr(rag_dataset_ingestion, "CleaningCurationAgent", MockAgent)

    run_cli(
        input_directory=input_dir,
        output_path=output_path,
        model="gpt-4o-mini",
        max_turns=3,
    )

    assert output_path.exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["source"] == "Climate Law"


def test_run_cli_handles_empty_input_directory(tmp_path, monkeypatch) -> None:
    """Test that run_cli handles directories with no markdown files gracefully."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    output_path = tmp_path / "out.json"

    captured_agents = []

    class MockAgent:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            captured_agents.append(self)

        def run(self, max_turns: int) -> str:
            # Simulate processing 0 documents
            output_path.write_text("[]", encoding="utf-8")
            return "Cleaned 0 docs."

    monkeypatch.setattr(rag_dataset_ingestion, "CleaningCurationAgent", MockAgent)

    run_cli(
        input_directory=empty_dir,
        output_path=output_path,
        model="gpt-4o-mini",
    )

    assert len(captured_agents) == 1
    assert output_path.exists()


def test_run_cli_signature_accepts_model() -> None:
    signature = inspect.signature(run_cli)

    assert "model" in signature.parameters
    assert signature.parameters["model"].default == "gpt-5.4-mini"


def test_chunk_records_preserves_metadata_and_adds_stable_chunk_ids() -> None:
    record = {
        "source": "European Climate Law",
        "title": "European Climate Law",
        "article": "document",
        "topic": "climate_law",
        "file_path": "law.md",
        "text": "abcdefghij",
        "content_hash": "hash123",
    }

    chunks = chunk_records([record], chunk_size=4, chunk_step=3)

    assert chunks == [
        {
            **record,
            "text": "abcd",
            "chunk_id": "hash123:0",
            "chunk_index": 0,
            "chunk_start": 0,
            "parent_content_hash": "hash123",
        },
        {
            **record,
            "text": "defg",
            "chunk_id": "hash123:3",
            "chunk_index": 1,
            "chunk_start": 3,
            "parent_content_hash": "hash123",
        },
        {
            **record,
            "text": "ghij",
            "chunk_id": "hash123:6",
            "chunk_index": 2,
            "chunk_start": 6,
            "parent_content_hash": "hash123",
        },
        {
            **record,
            "text": "j",
            "chunk_id": "hash123:9",
            "chunk_index": 3,
            "chunk_start": 9,
            "parent_content_hash": "hash123",
        },
    ]


def test_run_cli_can_chunk_output_records(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    output_path = tmp_path / "out.json"

    class MockAgent:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def run(self, max_turns: int) -> str:
            sample_record = {
                "source": "European Climate Law",
                "article": "document",
                "topic": "climate_law",
                "text": "abcdefghij",
                "file_path": str(input_dir / "doc.md"),
                "content_hash": "hash123",
            }
            output_path.write_text(json.dumps([sample_record]), encoding="utf-8")
            return "Cleaned 1 docs."

    monkeypatch.setattr(rag_dataset_ingestion, "CleaningCurationAgent", MockAgent)

    run_cli(
        input_directory=input_dir,
        output_path=output_path,
        model="gpt-4o-mini",
        chunk=True,
        chunk_size=4,
        chunk_step=3,
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(data) == 4
    assert data[0]["chunk_id"] == "hash123:0"
    assert data[1]["chunk_start"] == 3
