import tempfile
import unittest
from pathlib import Path

from eu_climate_policy_rag.collection.fetch_agent import (
    ContentCache,
    DocumentFetchAgent,
    DocumentPreselector,
)


class DocumentPreselectorTests(unittest.TestCase):
    def test_accepts_substantive_climate_policy_content(self) -> None:
        markdown = (
            "The European Climate Law sets a binding climate neutrality objective. "
            "The 2040 climate target reduces greenhouse gas emissions and supports "
            "the energy transition. "
        ) * 8

        result = DocumentPreselector().assess("2040 climate target", markdown)

        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "accepted")

    def test_rejects_off_topic_content(self) -> None:
        markdown = (
            "This annex discusses single market enforcement, product labelling, "
            "capital markets, savings accounts, and cross-border services. "
        ) * 10

        result = DocumentPreselector().assess("Single market focus areas", markdown)

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "content is not clearly about EU climate policy")

    def test_rejects_navigation_heavy_content_with_weak_climate_signal(self) -> None:
        markdown = """
        Accept all cookies
        Accept only essential cookies
        Skip to main content
        Select your language
        Official website of the European Union
        How do you know?
        See all EU institutions
        Type of search results
        Search on:
        Climate
        """ * 12

        result = DocumentPreselector().assess("Commission page", markdown)

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "content appears to be mostly page navigation")


class DocumentFetchAgentPreselectionTests(unittest.TestCase):
    def test_save_content_to_file_rejects_duplicate_markdown(self) -> None:
        markdown = (
            "The European Climate Law and 2040 climate target concern greenhouse "
            "gas emission reductions and climate neutrality. "
        ) * 10

        with tempfile.TemporaryDirectory() as directory:
            existing_path = Path(directory) / "existing.md"
            existing_path.write_text(markdown, encoding="utf-8")

            cache = ContentCache()
            markdown_id = cache.add({"markdown": markdown, "title": "Duplicate"})
            agent = DocumentFetchAgent(cache=cache)

            result = agent.save_content_to_file(
                markdown_id,
                "duplicate.md",
                directory=directory,
            )

            self.assertTrue(result["rejected"])
            self.assertEqual(result["reason"], "duplicate content already exists")
            self.assertFalse((Path(directory) / "duplicate.md").exists())

    def test_run_tool_forces_agent_output_directory(self) -> None:
        markdown = (
            "The European Climate Law and 2040 climate target concern greenhouse "
            "gas emission reductions and climate neutrality. "
        ) * 10

        with tempfile.TemporaryDirectory() as output_directory:
            cache = ContentCache()
            markdown_id = cache.add({"markdown": markdown, "title": "Climate law"})
            agent = DocumentFetchAgent(
                cache=cache,
                output_directory=output_directory,
            )

            import asyncio
            import json

            result = asyncio.run(
                agent.run_tool(
                    "save_content_to_file",
                    {
                        "markdown_id": markdown_id,
                        "filename": "saved.md",
                        "directory": ".",
                    },
                )
            )
            payload = json.loads(result)

            self.assertTrue(payload["saved"])
            self.assertTrue((Path(output_directory) / "saved.md").exists())
            self.assertEqual(payload["path"], str(Path(output_directory) / "saved.md"))


if __name__ == "__main__":
    unittest.main()
