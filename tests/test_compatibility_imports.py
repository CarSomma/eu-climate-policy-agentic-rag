class PublicImportTests:
    def test_public_root_imports_still_work(self) -> None:
        from eu_climate_policy_rag import (
            ClimatePolicyAgent,
            DocumentFetchAgent,
            FetchedDocumentIngestor,
        )

        assert ClimatePolicyAgent.__name__ == "ClimatePolicyAgent"
        assert DocumentFetchAgent.__name__ == "DocumentFetchAgent"
        assert FetchedDocumentIngestor.__name__ == "FetchedDocumentIngestor"

    def test_supported_subpackage_imports_work(self) -> None:
        from eu_climate_policy_rag.collection.fetch_agent import DocumentFetchAgent
        from eu_climate_policy_rag.collection.ingestion import FetchedDocumentIngestor
        from eu_climate_policy_rag.qa.rag import ClimatePolicyAgent

        assert DocumentFetchAgent.__module__ == "eu_climate_policy_rag.collection.fetch_agent"
        assert FetchedDocumentIngestor.__module__ == "eu_climate_policy_rag.collection.ingestion"
        assert ClimatePolicyAgent.__module__ == "eu_climate_policy_rag.qa.rag"
