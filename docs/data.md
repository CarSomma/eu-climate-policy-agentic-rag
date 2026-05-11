# Data

There are two data stages:

- `climate_policy_docs/*.md`: fetched Markdown source documents from the pipeline
- `data/eu_climate_policy.json`: structured records used by `ClimatePolicyAgent`

The RAG assistant currently reads from `data/eu_climate_policy.json`, not directly from the Markdown folder. The ingestion step should first clean, deduplicate, filter, and convert fetched Markdown into JSON records. Chunking can be added later as a separate step.

## CLI usage

```bash
# Ask a question directly from the terminal
uv run eu-climate-ask "How does the EU's 2030 climate target relate to the 2050 goal?"

# Options
uv run eu-climate-ask --help
#   QUESTION                    The question to ask the RAG assistant.
#   --data PATH                 Path to the JSON data file. [default: data/eu_climate_policy.json]
#   --model TEXT                OpenAI model to use.        [default: gpt-4o-mini]
#   --num-results INTEGER       Number of documents to retrieve per search. [default: 5]
#   --max-chars-per-doc INTEGER Max characters per retrieved document.      [default: 2000]
#   --max-turns INTEGER         Max agent turns before stopping.            [default: 10]
```

## Python usage

```python
from eu_climate_policy_rag import ClimatePolicyAgent

rag = ClimatePolicyAgent.from_json("data/eu_climate_policy.json")
result = rag.answer("How does the EU's 2030 climate target relate to the 2050 goal?")
print(result.answer)   # cited answer text
print(result.sources)  # list of source document names used
```

Example discovery-only usage (run inside an async context or Jupyter cell):

```python
from eu_climate_policy_rag import discover_documents

documents = await discover_documents()
print(documents[:3])
```
