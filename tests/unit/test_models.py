from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from eu_climate_policy_rag.core.models import (
    CleanedDocumentRecordModel,
    IngestionConfigModel,
    LinkModel,
    PipelineConfigModel,
)


def test_link_model_strips_text_and_href() -> None:
    link = LinkModel(text=" Climate Law ", href=" https://example.test/doc ")

    assert link.text == "Climate Law"
    assert link.href == "https://example.test/doc"


def test_link_model_rejects_blank_href() -> None:
    with pytest.raises(ValidationError):
        LinkModel(text="Climate Law", href=" ")


def test_cleaned_document_record_model_defaults_article() -> None:
    record = CleanedDocumentRecordModel(
        source="European Climate Law",
        topic="climate_law",
        file_path="climate_policy_docs/law.md",
        text="Climate neutrality by 2050.",
        content_hash="abc123",
    )

    assert record.article == "document"


def test_cleaned_document_record_model_requires_text() -> None:
    with pytest.raises(ValidationError):
        CleanedDocumentRecordModel(
            source="European Climate Law",
            topic="climate_law",
            file_path="climate_policy_docs/law.md",
            text="",
            content_hash="abc123",
        )


def test_pipeline_config_accepts_valid_values() -> None:
    config = PipelineConfigModel(
        source_url="https://example.test/docs",
        limit=1,
        model="gpt-4.1-mini",
        max_turns=3,
        output_directory=Path("docs"),
    )

    assert config.output_directory == Path("docs")
    assert config.model == "gpt-4.1-mini"


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: PipelineConfigModel(
                source_url="https://example.test/docs",
                limit=0,
            ),
            id="pipeline-limit-zero",
        ),
        pytest.param(
            lambda: PipelineConfigModel(
                source_url="https://example.test/docs",
                model="",
            ),
            id="pipeline-empty-model",
        ),
        pytest.param(
            lambda: IngestionConfigModel(model=""),
            id="ingestion-empty-model",
        ),
        pytest.param(
            lambda: IngestionConfigModel(max_turns=0),
            id="ingestion-zero-turns",
        ),
    ],
)
def test_config_models_reject_invalid_values(factory: Callable[[], object]) -> None:
    with pytest.raises(ValidationError):
        factory()


def test_ingestion_config_accepts_valid_values() -> None:
    config = IngestionConfigModel(
        input_directory=Path("docs"),
        output_path=Path("records.json"),
        model="gpt-4.1-mini",
        max_turns=5,
    )

    assert config.model == "gpt-4.1-mini"
