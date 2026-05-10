# Package Structure

The implementation is split by concern:

- `collection/`: document discovery, metadata enrichment, fetching, cleaning, and pipeline CLI
- `qa/`: RAG question-answering code
- `core/`: core types and colored logging utilities

Important modules:

- `collection/document_discovery.py`: scrapes document links grouped by documentation section
- `collection/document_metadata.py`: normalizes links and enriches them with type, year, source, format, and topic
- `collection/fetch_agent.py`: uses Playwright, MarkItDown, and OpenAI tool calls to fetch and save documents
- `collection/ingestion.py`: cleans fetched Markdown, deduplicates, filters off-topic files, and writes JSON records
- `collection/pipeline.py`: command-line pipeline that wires discovery, enrichment, preselection, fetch, and save
- `qa/rag.py`: agentic RAG with an LLM-driven search loop via `search_documents` tool calls, Minsearch index, `RagAnswerModel` responses, and the `eu-climate-ask` CLI (`ClimatePolicyAgent`)
- `core/logging_utils.py`: provides colored project loggers for CLI progress output
- `core/models.py`: data models that validate links, metadata, configs, results, and cleaned records
- `core/types.py`: core typed dictionaries

The old top-level module paths still re-export these objects for backward compatibility.
