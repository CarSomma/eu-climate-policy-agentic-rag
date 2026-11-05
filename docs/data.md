# Data

There are three data stages:

- `climate_policy_docs/*.md`: fetched Markdown source documents from the pipeline
- `data/eu_climate_policy.json`: cleaned JSON records (optionally chunked), the readable export of ingestion
- `data/chroma/`: the ChromaDB vector store the assistant searches (git-ignored)

The ingestion step cleans, deduplicates, filters, and converts fetched Markdown into JSON records. It can also write chunked records with `eu-climate-ingest --chunk`. `eu-climate-index` then embeds those records into ChromaDB: each chunk is stored with its text, its metadata (everything except `text`, dropping empty values) and its embedding. The assistant opens `data/chroma/` and searches it directly; it does not read the JSON at question time.

Embeddings come from Chroma's default model, all-MiniLM-L6-v2, which runs locally and reads about the first 1,000 characters of each input. Each chunk is embedded as its `title` followed by its text. Results are ranked by cosine distance.

Chunked records preserve the original document metadata and add:

- `chunk_id`
- `chunk_index`
- `chunk_start`
- `parent_content_hash`

## CLI usage

```bash
# Ask a question directly from the terminal
uv run eu-climate-ask "How does the EU's 2030 climate target relate to the 2050 goal?"

# Options
uv run eu-climate-ask --help
#   QUESTION                    The question to ask the RAG assistant.
#   --chroma-dir PATH           Vector store built by eu-climate-index. [default: data/chroma]
#   --data PATH                 Search a JSON file instead (embedded in memory on every run).
#   --model TEXT                OpenAI model to use.        [default: gpt-4o-mini]
#   --num-results INTEGER       Number of documents to retrieve per search. [default: 5]
#   --max-chars-per-doc INTEGER Max characters per retrieved document.      [default: 2000]
#   --max-turns INTEGER         Max agent turns before stopping.            [default: 10]
```

Create a chunked RAG dataset:

```bash
uv run eu-climate-ingest --chunk --chunk-size 2000 --chunk-step 1000
```

Embed it into the vector store (add `--chroma-dir data/chroma` to the ingest command to do both at once):

```bash
uv run eu-climate-index --data data/eu_climate_policy.json --chroma-dir data/chroma
```

## Python usage

```python
from eu_climate_policy_rag import ClimatePolicyAgent

rag = ClimatePolicyAgent.from_chroma("data/chroma")
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
