# Tests

Run the full test suite with:

    uv run pytest

Individual test files map to source modules:

| Test file | Module under test |
|---|---|
| `tests/test_candidate_discovery.py` | `collection/discovery/candidate_discovery.py` |
| `tests/test_candidate_utils.py` | `collection/discovery/candidate_utils.py` |
| `tests/test_cleaning_agent.py` | `collection/cleaning/cleaning_agent.py` |
| `tests/test_cleaning_toolbox.py` | `collection/cleaning/cleaning_toolbox.py` |
| `tests/test_compatibility_imports.py` | public package imports |
| `tests/test_document_discovery.py` | `collection/discovery/document_link_scraper.py` |
| `tests/test_document_quality.py` | `collection/document_quality.py` |
| `tests/test_fetch_agent.py` | `collection/fetching/fetch_agent.py` |
| `tests/test_fetching_content.py` | fetching cache, conversion, and storage |
| `tests/test_fetching_toolbox.py` | `collection/fetching/fetch_toolbox.py` |
| `tests/test_logging_utils.py` | `core/logging_utils.py` |
| `tests/test_markdown_cleaning.py` | `collection/cleaning/markdown_cleaning.py` |
| `tests/test_models.py` | `core/models.py` |
| `tests/test_pipeline.py` | `collection/pipeline.py` |
| `tests/test_rag_dataset_ingestion.py` | `collection/cleaning/rag_dataset_ingestion.py` |
| `tests/test_rag.py` | `qa/rag.py` |
| `tests/test_urls.py` | `collection/document_urls.py` |
