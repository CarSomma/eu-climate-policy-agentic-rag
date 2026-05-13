# Tests

## Running Tests

Run the full test suite:
```bash
uv run pytest
```

Run only unit tests (fast):
```bash
uv run pytest -m unit
```

Run only integration tests:
```bash
uv run pytest -m integration
```

## Test Organization

Tests are organized into **unit tests** (pure logic, no I/O) and **integration tests** (I/O, async, multi-component):

```
tests/
├── conftest.py              # Shared fixtures + auto-markers
├── unit/                    # Fast, isolated unit tests
│   ├── test_candidate_utils.py
│   ├── test_compatibility_imports.py
│   ├── test_document_quality.py
│   ├── test_logging_utils.py
│   ├── test_markdown_cleaning.py
│   ├── test_models.py
│   ├── test_openai_responses_adapter.py # OpenAI Responses tool schema adapter
│   ├── test_rag_validation.py     # Model validation (Pydantic)
│   ├── test_tool_executor.py       # ToolExecutor sync/async behavior
│   ├── test_tool_framework.py      # FunctionTool, ToolRegistry, built-ins
│   ├── test_tool_middleware.py     # Tool middleware lifecycle hooks
│   └── test_urls.py
└── integration/             # Slower tests with I/O, async, APIs
    ├── test_candidate_discovery.py
    ├── test_cleaning_agent.py
    ├── test_cleaning_toolbox.py
    ├── test_document_discovery.py
    ├── test_fetch_agent.py
    ├── test_fetching_content.py
    ├── test_fetching_toolbox.py
    ├── test_pipeline.py
    ├── test_rag.py
    ├── test_rag_dataset_ingestion.py
    └── test_rag_web_search.py      # Web search integration
```

### Auto-Markers

Tests are automatically tagged based on their location:
- `tests/unit/` → `@pytest.mark.unit`
- `tests/integration/` → `@pytest.mark.integration`

No manual decorators needed! The `pytest_collection_modifyitems` hook in `conftest.py` handles this automatically.
