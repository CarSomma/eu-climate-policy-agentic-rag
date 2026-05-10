import unittest

from eu_climate_policy_rag.qa.rag import format_context_item


class RagTests(unittest.TestCase):
    def test_format_context_item_uses_expected_notebook_prompt_shape(self) -> None:
        document = {
            "source": "European Climate Law",
            "article": "Article 4",
            "topic": "climate_law",
            "text": "The Union-wide 2030 climate target is binding.",
        }

        self.assertEqual(
            format_context_item(document),
            "[European Climate Law | Article 4 | topic: climate_law]\n"
            "The Union-wide 2030 climate target is binding.",
        )


if __name__ == "__main__":
    unittest.main()
