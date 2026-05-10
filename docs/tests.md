# Tests

Run the full test suite with:

    uv run pytest

Individual test files map to source modules:

| Test file | Module under test |
|---|---|
| `tests/test_document_discovery.py` | `collection/document_discovery.py` |
| `tests/test_document_metadata.py` | `collection/document_metadata.py` |
| `tests/test_fetch_agent_preselection.py` | `collection/fetch_agent.py` |
| `tests/test_ingestion.py` | `collection/ingestion.py` |
| `tests/test_pipeline.py` | `collection/pipeline.py` |
| `tests/test_rag.py` | `qa/rag.py` |
| `tests/test_url_utils.py` | `collection/url_utils.py` |
