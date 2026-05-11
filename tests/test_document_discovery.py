from eu_climate_policy_rag.collection.document_discovery import _normalize_sections


class DocumentDiscoveryTests:
    def test_normalize_sections_removes_incomplete_links(self) -> None:
        sections = {
            "Climate law": [
                {"text": "European Climate Law", "href": "https://example.test/law"},
                {"text": "", "href": "https://example.test/empty"},
                {"text": "Missing href"},
            ]
        }

        assert _normalize_sections(sections) == {
            "Climate law": [
                {
                    "text": "European Climate Law",
                    "href": "https://example.test/law",
                }
            ]
        }
