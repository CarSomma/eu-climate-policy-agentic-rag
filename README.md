# EU Climate Policy Q&A RAG Assistant

An agentic retrieval-augmented generation (RAG) system that helps students and early-stage policy learners navigate EU climate policy documents quickly and accurately.

## Table of Contents

- [The Problem](#the-problem)
- [What It Does](#what-it-does)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)

## Further Documentation

- [docs/pipeline.md](docs/pipeline.md): running the document pipeline and ingestion
- [docs/notebooks.md](docs/notebooks.md): available notebooks
- [docs/package-structure.md](docs/package-structure.md): codebase layout and key modules
- [docs/data.md](docs/data.md): data stages and usage examples
- [docs/tests.md](docs/tests.md): running the test suite

## The Problem

Students and early-stage policy learners struggle to understand EU climate policy because key information is spread across multiple complex and technical documents, such as the European Climate Law and related EU policy pages. This makes it difficult to extract accurate, connected, and trustworthy answers without misinterpretation.

## What It Does

A user asks a question like: *"How does the EU's 2030 climate target relate to the 2040 goal?"* The system uses an **Agentic RAG approach**: the LLM drives its own search loop, calling a `search_documents` tool one or more times with different queries until it has enough context, then returns a concise, cited answer. If the documents do not contain enough information, the system explicitly states that rather than guessing.

The codebase has two main workflows:

1. **Document pipeline**: discover and fetch official EU climate policy documents, convert them to Markdown, and save them locally.
2. **RAG assistant**: load a structured JSON dataset and answer questions with cited responses.

## Prerequisites

- Python 3.13 or later (you can install it with `uv`)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- An [OpenAI API key](https://platform.openai.com/api-keys) (used by both the fetch agent and the RAG assistant)
- Playwright browsers (installed automatically via `uv sync`; run `uv run playwright install` if needed)

## Setup

1. Install `uv` if you do not have it yet:

   https://docs.astral.sh/uv/getting-started/installation/

2. Clone this repository, then move into the project folder:

   ```bash
   git clone <repository-url>
   cd ai-buildcamp-project-starter
   ```

   If you downloaded the project as a zip file, extract it and open a terminal in the extracted folder.

3. Install Python 3.13 with `uv` if it is not already available:

   ```bash
   uv python install 3.13
   ```

   The project requires Python 3.13 or later, and `uv sync` will use a compatible interpreter.

4. Create a local environment file and add your OpenAI API key:

   ```bash
   cp .env.example .env
   ```

   Then open `.env` and replace the placeholder value:

   ```bash
   OPENAI_API_KEY=your-openai-key-here
   ```

5. Install the project dependencies:

   ```bash
   uv sync
   ```

6. Install Playwright browsers if they were not installed automatically:

   ```bash
   uv run playwright install
   ```

7. Check that the CLI is available:

   ```bash
   uv run eu-climate-ask --help
   ```

8. Start Jupyter if you want to run the notebooks:

   ```bash
   uv run jupyter notebook
   ```

## Usage

### Sample questions

Users can ask broad explanatory questions, targeted policy questions, or comparison questions, for example:

- How does the EU's 2030 climate target relate to the 2040 goal?
- What is the European Climate Law, and what does it require?
- What is the Carbon Border Adjustment Mechanism?
- How does the European Union Emissions Trading System support climate neutrality?
- What is the Fit for 55 package?
- How does the Paris Agreement shape the EU's climate targets?
- What role do carbon removals play in the EU's 2040 climate target?
- What does the Clean Industrial Deal add to EU climate policy?

### Ask a question from the terminal

```bash
# Basic question
uv run eu-climate-ask "How does the EU's 2030 target relate to the 2040 goal?"

# Use a different model or data file
uv run eu-climate-ask "What is CBAM?" --model gpt-4o --data data/eu_climate_policy.json

# Limit agent turns (default 10)
uv run eu-climate-ask "What is CBAM?" --max-turns 5

# See all options
uv run eu-climate-ask --help
```

### Ask a question in Python

```python
from eu_climate_policy_rag import ClimatePolicyAgent

rag = ClimatePolicyAgent.from_json("data/eu_climate_policy.json")
result = rag.answer("How does the EU's 2030 climate target relate to the 2040 goal?")
print(result.answer)   # the cited answer text
print(result.sources)  # list of source document names used
```

### Run the document pipeline (fetch + ingest)

```bash
# Fetch documents (smoke test with one document)
uv run eu-climate-pipeline --limit 1

# Fetch all preselected documents
uv run eu-climate-pipeline

# Clean and convert fetched Markdown to JSON
uv run eu-climate-ingest
```

### Read some documents

```python
from eu_climate_policy_rag import discover_and_enrich_documents

# Run inside an async context or Jupyter cell
documents = await discover_and_enrich_documents()
print(documents[:3])
```

See [docs/pipeline.md](docs/pipeline.md) for the full list of CLI options and [docs/data.md](docs/data.md) for details on data stages.
