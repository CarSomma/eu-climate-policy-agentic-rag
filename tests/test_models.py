from pathlib import Path

import pytest
from pydantic import ValidationError

from eu_climate_policy_rag.core.models import (
    CleanedDocumentRecordModel,
    DocumentMetadataModel,
    LinkModel,
    PipelineConfigModel,
)


class CoreModelTests:
    def test_link_model_strips_text_and_rejects_blank_href(self) -> None:
        link = LinkModel(text=" Climate Law ", href=" https://example.test/doc ")

        assert link.text == "Climate Law"
        assert link.href == "https://example.test/doc"

        with pytest.raises(ValidationError):
            LinkModel(text="Climate Law", href=" ")

    def test_document_metadata_model_rejects_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            DocumentMetadataModel(
                title="European Climate Law",
                url="https://example.test",
                type="regulation",
                source="eur-lex",
                format="html",
            )

    def test_cleaned_document_record_model_requires_text(self) -> None:
        record = CleanedDocumentRecordModel(
            source="European Climate Law",
            topic="climate_law",
            file_path="climate_policy_docs/law.md",
            text="Climate neutrality by 2050.",
            content_hash="abc123",
        )

        assert record.article == "document"

        with pytest.raises(ValidationError):
            CleanedDocumentRecordModel(
                source="European Climate Law",
                topic="climate_law",
                file_path="climate_policy_docs/law.md",
                text="",
                content_hash="abc123",
            )

    def test_pipeline_config_validates_positive_turns_and_limit(self) -> None:
        config = PipelineConfigModel(
            source_url="https://example.test/docs",
            limit=1,
            max_turns=3,
            output_directory=Path("docs"),
        )

        assert config.output_directory == Path("docs")

        with pytest.raises(ValidationError):
            PipelineConfigModel(
                source_url="https://example.test/docs",
                limit=0,
            )
