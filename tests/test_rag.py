from eu_climate_policy_rag.qa.rag import format_context_item
from eu_climate_policy_rag.qa.tools import SearchDocumentsTool


class RagTests:
    def test_format_context_item(self) -> None:
        document = {
            "source": "European Climate Law",
            "article": "Article 4",
            "topic": "climate_law",
            "text": "The Union-wide 2030 climate target is binding.",
        }

        assert format_context_item(document) == (
            "[European Climate Law | Article 4 | topic: climate_law]\n"
            "The Union-wide 2030 climate target is binding."
        )

    def test_search_documents_tool(self) -> None:
        documents = [
            {
                "source": "European Climate Law",
                "article": "Article 4",
                "topic": "climate_law",
                "text": "The Union-wide 2030 climate target is binding.",
                "file_path": "law.md",
                "content_hash": "abc123",
            }
        ]

        tool = SearchDocumentsTool(documents, num_results=1)
        result = tool.run(" 2030 target ")

        assert result.query == "2030 target"
        assert result.sources == ["European Climate Law"]
        assert "Article 4" in result.context
        assert tool.schema["name"] == "search_documents"
