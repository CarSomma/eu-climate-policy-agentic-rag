import json

import pytest

from eu_climate_policy_rag.collection.cleaning.cleaning_toolbox import CleaningToolbox


@pytest.fixture
def input_directory(tmp_path):
    path = tmp_path / "docs"
    path.mkdir()
    return path


@pytest.fixture
def output_path(tmp_path):
    return tmp_path / "records.json"


def test_cleaning_toolbox_inspects_saves_skips_and_finalizes(
    input_directory,
    output_path,
    climate_markdown: str,
    write_markdown_file,
) -> None:
    climate_path = write_markdown_file(
        input_directory,
        "climate-law.md",
        climate_markdown,
    )
    toolbox = CleaningToolbox(input_directory, output_path)

    assert toolbox.list_documents()["count"] == 1
    inspection = toolbox.inspect_document(str(climate_path))
    assert inspection["skip_reason"] is None

    save_result = toolbox.save_cleaned_document(str(climate_path))
    assert save_result["saved"]

    skip_result = toolbox.skip_document("off-topic.md", "not climate policy")
    assert skip_result["skipped"]

    final_result = toolbox.finalize()
    records = json.loads(output_path.read_text(encoding="utf-8"))
    assert final_result["record_count"] == 1
    assert final_result["skipped_count"] == 1
    assert records[0]["article"] == "document"


def test_skip_document_saves_deterministically_accepted_file(
    input_directory,
    output_path,
    write_markdown_file,
) -> None:
    climate_text = (
        "Regulation (EU) 2026/667 amends the European Climate Law. "
        "It sets a 2040 climate target for greenhouse gas emissions. "
    ) * 10
    climate_path = write_markdown_file(
        input_directory,
        "eur-lex-regulation-eu-2026-667-en.md",
        climate_text,
    )
    toolbox = CleaningToolbox(input_directory, output_path)

    result = toolbox.skip_document(str(climate_path), "Not EU climate policy.")

    assert result["saved"]
    assert len(toolbox.records) == 1
    assert toolbox.skipped == []


def test_skip_document_allows_deterministic_reject(input_directory, output_path) -> None:
    off_topic_path = input_directory / "single-market.md"
    off_topic_path.write_text("single market rules", encoding="utf-8")
    toolbox = CleaningToolbox(input_directory, output_path)

    result = toolbox.skip_document(
        str(off_topic_path),
        "not clearly about EU climate policy",
    )

    assert result["skipped"]
    assert len(toolbox.records) == 0


def test_finalize_writes_empty_records_when_no_documents(output_path) -> None:
    toolbox = CleaningToolbox(output_path.parent / "missing-docs", output_path)

    result = toolbox.finalize()

    assert result["record_count"] == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == []


def test_save_cleaned_document_skips_duplicate_content(
    input_directory,
    output_path,
    climate_markdown: str,
    write_markdown_file,
) -> None:
    first = write_markdown_file(input_directory, "first.md", climate_markdown)
    second = write_markdown_file(input_directory, "second.md", climate_markdown)
    toolbox = CleaningToolbox(input_directory, output_path)

    first_result = toolbox.save_cleaned_document(str(first))
    second_result = toolbox.save_cleaned_document(str(second))

    assert first_result["saved"]
    assert second_result["skipped"]
    assert second_result["reason"] == "duplicate content hash"


def test_inspect_document_raises_for_missing_path(input_directory, output_path) -> None:
    toolbox = CleaningToolbox(input_directory, output_path)

    with pytest.raises(FileNotFoundError):
        toolbox.inspect_document(str(input_directory / "missing.md"))
