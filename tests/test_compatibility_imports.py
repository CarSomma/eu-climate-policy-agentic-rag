import unittest


class PublicImportTests(unittest.TestCase):
    def test_public_root_imports_still_work(self) -> None:
        from eu_climate_policy_rag import (
            ClimatePolicyAgent,
            DocumentFetchAgent,
            FetchedDocumentIngestor,
        )

        self.assertEqual(ClimatePolicyAgent.__name__, "ClimatePolicyAgent")
        self.assertEqual(DocumentFetchAgent.__name__, "DocumentFetchAgent")
        self.assertEqual(FetchedDocumentIngestor.__name__, "FetchedDocumentIngestor")

    def test_supported_subpackage_imports_work(self) -> None:
        from eu_climate_policy_rag.collection.fetch_agent import DocumentFetchAgent
        from eu_climate_policy_rag.collection.ingestion import FetchedDocumentIngestor
        from eu_climate_policy_rag.qa.rag import ClimatePolicyAgent

        self.assertEqual(
            DocumentFetchAgent.__module__,
            "eu_climate_policy_rag.collection.fetch_agent",
        )
        self.assertEqual(
            FetchedDocumentIngestor.__module__,
            "eu_climate_policy_rag.collection.ingestion",
        )
        self.assertEqual(ClimatePolicyAgent.__module__, "eu_climate_policy_rag.qa.rag")


if __name__ == "__main__":
    unittest.main()
