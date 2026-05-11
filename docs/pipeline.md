# Document Pipeline

## Run the document pipeline

The pipeline discovers document links from the European Climate Law documentation page, preselects likely climate-policy documents by title, fetches them, converts them to Markdown, and saves them.

Run a small smoke test first:

       uv run eu-climate-pipeline --limit 1

Run the full preselected pipeline:

       uv run eu-climate-pipeline

Show all options:

       uv run eu-climate-pipeline --help

Useful options:

- `--limit 3`: fetch only the first three selected documents
- `--max-turns 20`: allow more LLM tool-loop turns per document
- `--output-directory climate_policy_docs`: choose where fetched Markdown is saved
- `--fetch-all`: disable title preselection and attempt every discovered document

By default, fetched Markdown is saved to `climate_policy_docs/`. The fetch agent also performs save-time preselection to reject duplicate content, very short content, off-topic content, and navigation-heavy pages.

## Clean fetched documents

After fetching, run the ingestion/curation step to clean Markdown and write one JSON record per kept file:

       uv run eu-climate-ingest

Show all options:

       uv run eu-climate-ingest --help

Useful options:

- `--input-directory climate_policy_docs`: read fetched Markdown from this folder
- `--output-path data/eu_climate_policy.json`: write cleaned JSON records here
- `--agentic`: use an LLM curator with deterministic cleaning tools
- `--max-turns 50`: maximum LLM tool-loop turns for agentic cleaning

This step removes common boilerplate, deduplicates by content hash, strips PDF artifacts like repeated `EN`, page breaks and page numbers, and excludes files that are not clearly about EU climate policy. It intentionally does not chunk yet.

The default ingestion mode is deterministic. The optional agentic mode can inspect previews and decide whether to save or skip files, but it can only use the cleaning tools; it does not rewrite document content.
