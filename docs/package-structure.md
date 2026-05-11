# Package Structure

The implementation is split by concern:

- `collection/`: document discovery, fetching, cleaning, and pipeline CLI
- `qa/`: RAG question-answering code
- `core/`: shared agent loop, tool registry, validation models, typed dictionaries, and logging utilities

Important modules:

- `collection/discovery/`: scrapes and selects source document candidates
- `collection/fetching/`: uses Playwright, MarkItDown, and OpenAI tool calls to fetch and save documents
- `collection/cleaning/`: cleans Markdown, curates records, and provides the ingestion CLI
- `collection/pipeline.py`: command-line pipeline that wires discovery, title preselection, fetch, and save
- `collection/fetch_pipeline_steps.py`: helper functions for running pipeline fetch steps and status checks
- `collection/content_hashing.py`: Markdown normalization and duplicate-detection hashes
- `collection/document_quality.py`: quality checks for fetched and cleaned document content
- `collection/document_urls.py`: URL normalization, document detection, and filename helpers
- `collection/fetching/fetch_tools.py` and `collection/cleaning/cleaning_tools.py`: collection-specific OpenAI function-tool builders
- `qa/rag.py`: agentic RAG orchestration with an LLM-driven tool loop, `RagAnswerModel` responses, and the `eu-climate-ask` CLI (`ClimatePolicyAgent`)
- `qa/tools.py`: class-based RAG tools, including the `search_documents` Minsearch tool and prompt context formatting
- `core/agent.py`: reusable OpenAI Responses API tool-call loop used by collection and QA agents
- `core/logging_utils.py`: provides colored project loggers for CLI progress output
- `core/models.py`: data models that validate links, configs, results, and cleaned records
- `core/tooling.py`: shared OpenAI function-tool schema and dispatch helpers
- `core/types.py`: core typed dictionaries
