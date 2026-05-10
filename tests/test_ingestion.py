import json
import tempfile
import unittest
from pathlib import Path

from eu_climate_policy_rag.collection.ingestion import (
    CleaningToolbox,
    FetchedDocumentIngestor,
    clean_markdown,
    infer_topic,
    should_drop_line,
)


class IngestionTests(unittest.TestCase):
    def test_clean_markdown_removes_common_pdf_and_web_artifacts(self) -> None:
        markdown = """
        EN
        12
        Accept all cookies
        The European Climate Law supports climate neutrality.


        Greenhouse gas emissions must fall.
        Press contacts:
        """

        cleaned = clean_markdown(markdown)

        self.assertNotIn("Accept all cookies", cleaned)
        self.assertNotIn("Press contacts", cleaned)
        self.assertNotIn("\n12\n", cleaned)
        self.assertIn("European Climate Law", cleaned)

    def test_should_drop_line_identifies_artifacts(self) -> None:
        self.assertTrue(should_drop_line("EN"))
        self.assertTrue(should_drop_line("42"))
        self.assertTrue(should_drop_line("Select your language"))
        self.assertFalse(should_drop_line("The 2040 climate target matters."))

    def test_ingest_directory_filters_duplicates_and_off_topic_files(self) -> None:
        climate_text = (
            "The European Climate Law sets a climate neutrality objective. "
            "Greenhouse gas emissions and the 2040 climate target are central. "
        ) * 10
        off_topic_text = (
            "Single market enforcement, product labels, and capital markets. "
        ) * 20

        with tempfile.TemporaryDirectory() as directory:
            input_directory = Path(directory) / "docs"
            output_path = Path(directory) / "eu_climate_policy.json"
            input_directory.mkdir()
            (input_directory / "climate-law.md").write_text(climate_text, encoding="utf-8")
            (input_directory / "duplicate.md").write_text(climate_text, encoding="utf-8")
            (input_directory / "single-market.md").write_text(off_topic_text, encoding="utf-8")

            result = FetchedDocumentIngestor().ingest_directory(
                input_directory,
                output_path,
            )
            records = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(result.input_count, 3)
            self.assertEqual(result.kept_count, 1)
            self.assertEqual(result.skipped_count, 2)
            self.assertEqual(result.record_count, 1)
            self.assertEqual(len(records), result.record_count)
            self.assertEqual(records[0]["topic"], "climate_law")
            self.assertEqual(records[0]["article"], "document")

    def test_cleaning_toolbox_inspects_saves_skips_and_finalizes(self) -> None:
        climate_text = (
            "The European Climate Law sets a climate neutrality objective. "
            "Greenhouse gas emissions and the 2040 climate target are central. "
        ) * 10

        with tempfile.TemporaryDirectory() as directory:
            input_directory = Path(directory) / "docs"
            output_path = Path(directory) / "records.json"
            input_directory.mkdir()
            climate_path = input_directory / "climate-law.md"
            climate_path.write_text(climate_text, encoding="utf-8")

            toolbox = CleaningToolbox(input_directory, output_path)

            self.assertEqual(toolbox.list_documents()["count"], 1)
            inspection = toolbox.inspect_document(str(climate_path))
            self.assertIsNone(inspection["deterministic_skip_reason"])

            save_result = toolbox.save_cleaned_document(str(climate_path))
            self.assertTrue(save_result["saved"])

            skip_result = toolbox.skip_document("off-topic.md", "not climate policy")
            self.assertTrue(skip_result["skipped"])

            final_result = toolbox.finalize()
            records = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(final_result["record_count"], 1)
            self.assertEqual(final_result["skipped_count"], 1)
            self.assertEqual(records[0]["article"], "document")

    def test_infer_topic(self) -> None:
        self.assertEqual(infer_topic("2040 climate target"), "climate_target_2040")
        self.assertEqual(infer_topic("climate law regulation"), "climate_law")


if __name__ == "__main__":
    unittest.main()
