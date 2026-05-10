import unittest
from pathlib import Path

from pydantic import ValidationError

from eu_climate_policy_rag.core.models import (
    CleanedDocumentRecordModel,
    DocumentMetadataModel,
    LinkModel,
    PipelineConfigModel,
)


class CoreModelTests(unittest.TestCase):
    def test_link_model_strips_text_and_rejects_blank_href(self) -> None:
        link = LinkModel(text=" Climate Law ", href=" https://example.test/doc ")

        self.assertEqual(link.text, "Climate Law")
        self.assertEqual(link.href, "https://example.test/doc")

        with self.assertRaises(ValidationError):
            LinkModel(text="Climate Law", href=" ")

    def test_document_metadata_model_rejects_missing_required_fields(self) -> None:
        with self.assertRaises(ValidationError):
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

        self.assertEqual(record.article, "document")

        with self.assertRaises(ValidationError):
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

        self.assertEqual(config.output_directory, Path("docs"))

        with self.assertRaises(ValidationError):
            PipelineConfigModel(
                source_url="https://example.test/docs",
                limit=0,
            )


if __name__ == "__main__":
    unittest.main()
