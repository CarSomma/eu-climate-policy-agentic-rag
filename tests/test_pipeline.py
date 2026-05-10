import unittest
from pathlib import Path

from eu_climate_policy_rag.collection.pipeline import (
    deduplicate_documents,
    flatten_sections,
    has_climate_signal,
    is_relevant_document,
    run_cli,
)


class PipelineTests(unittest.TestCase):
    def test_flatten_sections_adds_section_and_deduplicates_by_url(self) -> None:
        document = {
            "title": "European Climate Law",
            "url": "https://example.test/law",
            "type": "regulation",
            "year": 2021,
            "identifier": "abc",
            "source": "eur-lex",
            "format": "html",
            "topic": "climate_law",
        }

        flattened = flatten_sections({"Main": [document], "Duplicate": [document]})

        self.assertEqual(len(flattened), 1)
        self.assertEqual(flattened[0]["section"], "Main")

    def test_deduplicate_documents_keeps_first_url(self) -> None:
        first = {
            "title": "First",
            "url": "https://example.test/doc",
            "type": "other",
            "year": None,
            "identifier": None,
            "source": "other",
            "format": "html",
            "topic": "general",
        }
        second = {**first, "title": "Second"}

        self.assertEqual(deduplicate_documents([first, second]), [first])

    def test_is_relevant_document_keeps_climate_documents(self) -> None:
        document = {
            "title": "Questions and answers on the 2040 EU climate target",
            "url": "https://example.test/q-and-a",
            "type": "qa",
            "year": 2025,
            "identifier": None,
            "source": "commission",
            "format": "html",
            "topic": "climate_target_2040",
        }

        self.assertTrue(is_relevant_document(document))

    def test_is_relevant_document_rejects_off_topic_annex(self) -> None:
        document = {
            "title": "Single market focus areas for enforcement",
            "url": "https://example.test/single-market",
            "type": "communication",
            "year": 2026,
            "identifier": None,
            "source": "commission",
            "format": "pdf",
            "topic": "general",
        }

        self.assertFalse(is_relevant_document(document))
        self.assertFalse(has_climate_signal(document["title"]))

    def test_run_cli_signature_accepts_output_directory(self) -> None:
        self.assertIn("output_directory", run_cli.__annotations__)
        self.assertEqual(run_cli.__defaults__[3], Path("climate_policy_docs"))


if __name__ == "__main__":
    unittest.main()
