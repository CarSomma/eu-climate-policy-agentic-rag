"""Clean fetched Markdown files and convert them into JSON records for RAG."""

import json
import re
import hashlib
from pathlib import Path
from typing import Annotated, Any

from openai import OpenAI
import typer

from eu_climate_policy_rag.collection.fetch_agent import (
    CLIMATE_KEYWORDS,
    NAVIGATION_MARKERS,
    normalize_markdown_for_hash,
)
from eu_climate_policy_rag.core.logging_utils import get_logger
from eu_climate_policy_rag.core.models import (
    CleanedDocumentRecordModel,
    IngestionConfigModel,
    IngestionResultModel,
)


LOGGER = get_logger(__name__)


IngestionResult = IngestionResultModel


class FetchedDocumentIngestor:
    """Clean fetched Markdown source files and write records for the RAG dataset."""

    def __init__(
        self,
        minimum_characters: int = 500,
    ) -> None:
        self.minimum_characters = minimum_characters

    def ingest_directory(
        self,
        input_directory: str | Path = "climate_policy_docs",
        output_path: str | Path = "data/eu_climate_policy.json",
    ) -> IngestionResult:
        """Clean all Markdown files in a directory and write the kept records to JSON."""

        input_path = Path(input_directory)
        output_path = Path(output_path)
        markdown_paths = sorted(input_path.glob("*.md"))
        LOGGER.info("Starting deterministic ingestion from %s", input_path)
        LOGGER.info("Found %s Markdown files", len(markdown_paths))

        records: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        seen_hashes: set[str] = set()

        for markdown_path in markdown_paths:
            LOGGER.info("Cleaning %s", markdown_path)
            raw_markdown = markdown_path.read_text(encoding="utf-8", errors="ignore")
            cleaned_text = clean_markdown(raw_markdown)
            content_hash = hash_markdown(cleaned_text)
            reason = self._skip_reason(markdown_path, cleaned_text, content_hash, seen_hashes)
            if reason:
                LOGGER.warning("Skipping %s: %s", markdown_path, reason)
                skipped.append({"path": str(markdown_path), "reason": reason})
                continue

            seen_hashes.add(content_hash)
            metadata = metadata_from_path(markdown_path)
            LOGGER.info(
                "Keeping %s as topic=%s (%s chars)",
                markdown_path,
                metadata["topic"],
                len(cleaned_text),
            )
            records.append(
                CleanedDocumentRecordModel(
                    **metadata,
                    article="document",
                    text=cleaned_text,
                    content_hash=content_hash,
                ).model_dump()
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        LOGGER.info("Wrote %s cleaned records to %s", len(records), output_path)
        return IngestionResult(
            input_count=len(markdown_paths),
            kept_count=len(markdown_paths) - len(skipped),
            skipped_count=len(skipped),
            record_count=len(records),
            output_path=str(output_path),
            skipped=skipped,
        )

    def _skip_reason(
        self,
        path: Path,
        cleaned_text: str,
        content_hash: str,
        seen_hashes: set[str],
    ) -> str | None:
        """Return the deterministic reason a cleaned document should be skipped."""

        if content_hash in seen_hashes:
            return "duplicate content hash"
        if len(cleaned_text) < self.minimum_characters:
            return "cleaned content is too short"
        if not has_enough_climate_signal(f"{path.stem} {cleaned_text}"):
            return "not clearly about EU climate policy"
        return None


class CleaningToolbox:
    """Stateful deterministic tools used by the optional LLM cleaning curator."""

    def __init__(
        self,
        input_directory: str | Path = "climate_policy_docs",
        output_path: str | Path = "data/eu_climate_policy.json",
        ingestor: FetchedDocumentIngestor | None = None,
    ) -> None:
        config = IngestionConfigModel(
            input_directory=Path(input_directory),
            output_path=Path(output_path),
        )
        self.input_directory = config.input_directory
        self.output_path = config.output_path
        self.ingestor = ingestor or FetchedDocumentIngestor()
        self.records: list[dict[str, Any]] = []
        self.skipped: list[dict[str, str]] = []
        self.seen_hashes: set[str] = set()

    def list_documents(self) -> dict[str, Any]:
        """List fetched Markdown files available for curation."""

        paths = [str(path) for path in sorted(self.input_directory.glob("*.md"))]
        LOGGER.info("Cleaning agent listed %s documents", len(paths))
        return {"documents": paths, "count": len(paths)}

    def inspect_document(self, path: str) -> dict[str, Any]:
        """Return cleaning metadata, a preview, and any deterministic skip reason."""

        markdown_path = Path(path)
        raw_markdown = markdown_path.read_text(encoding="utf-8", errors="ignore")
        cleaned_text = clean_markdown(raw_markdown)
        content_hash = hash_markdown(cleaned_text)
        reason = self.ingestor._skip_reason(
            markdown_path,
            cleaned_text,
            content_hash,
            self.seen_hashes,
        )
        LOGGER.info(
            "Inspected %s: raw=%s cleaned=%s skip=%s",
            markdown_path,
            len(raw_markdown),
            len(cleaned_text),
            reason,
        )
        return {
            "path": str(markdown_path),
            "raw_characters": len(raw_markdown),
            "cleaned_characters": len(cleaned_text),
            "topic": infer_topic(markdown_path.stem),
            "deterministic_skip_reason": reason,
            "preview": cleaned_text[:1200],
        }

    def save_cleaned_document(self, path: str) -> dict[str, Any]:
        """Save one cleaned document record unless deterministic checks reject it."""

        markdown_path = Path(path)
        raw_markdown = markdown_path.read_text(encoding="utf-8", errors="ignore")
        cleaned_text = clean_markdown(raw_markdown)
        content_hash = hash_markdown(cleaned_text)
        reason = self.ingestor._skip_reason(
            markdown_path,
            cleaned_text,
            content_hash,
            self.seen_hashes,
        )
        if reason:
            return self.skip_document(str(markdown_path), reason)

        self.seen_hashes.add(content_hash)
        self.records.append(
            CleanedDocumentRecordModel(
                **metadata_from_path(markdown_path),
                article="document",
                text=cleaned_text,
                content_hash=content_hash,
            ).model_dump()
        )
        LOGGER.info("Agent saved cleaned document: %s", markdown_path)
        return {"saved": True, "path": str(markdown_path), "records": len(self.records)}

    def skip_document(self, path: str, reason: str) -> dict[str, Any]:
        """Record that a document was skipped with a reason."""

        skipped = {"path": path, "reason": reason}
        if skipped not in self.skipped:
            self.skipped.append(skipped)
        LOGGER.warning("Agent skipped %s: %s", path, reason)
        return {"skipped": True, **skipped}

    def finalize(self) -> dict[str, Any]:
        """Write curated records to the configured JSON output path."""

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")
        LOGGER.info(
            "Agent finalized %s records to %s",
            len(self.records),
            self.output_path,
        )
        return {
            "output_path": str(self.output_path),
            "record_count": len(self.records),
            "skipped_count": len(self.skipped),
            "skipped": self.skipped,
        }


class CleaningCurationAgent:
    """LLM-assisted cleaner restricted to deterministic curation tools."""

    def __init__(
        self,
        openai_client: OpenAI | None = None,
        model: str = "gpt-5.4-mini",
        toolbox: CleaningToolbox | None = None,
    ) -> None:
        self.openai_client = openai_client or OpenAI()
        self.model = model
        self.toolbox = toolbox or CleaningToolbox()

    def run(self, max_turns: int = 50) -> str:
        """Run the cleaning tool loop until the model finalizes or reaches the turn limit."""

        messages: list[Any] = [
            {"role": "system", "content": CLEANING_AGENT_INSTRUCTIONS},
            {
                "role": "user",
                "content": "Clean and curate the fetched Markdown documents.",
            },
        ]

        LOGGER.info("Starting cleaning curation agent")
        for _ in range(max_turns):
            response = self.openai_client.responses.create(
                model=self.model,
                input=messages,
                tools=CLEANING_TOOLS,
            )
            tool_calls = [item for item in response.output if item.type == "function_call"]
            if not tool_calls:
                LOGGER.info("Cleaning curation agent finished")
                return response.output_text

            messages.extend(response.output)
            for tool_call in tool_calls:
                args = json.loads(tool_call.arguments)
                LOGGER.info("Cleaning agent calling tool: %s", tool_call.name)
                result = self.run_tool(tool_call.name, args)
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": json.dumps(result),
                    }
                )

        return "Cleaning agent reached max turns without a final answer."

    def run_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one cleaning tool call to the toolbox."""

        tool_map = {
            "list_documents": self.toolbox.list_documents,
            "inspect_document": self.toolbox.inspect_document,
            "save_cleaned_document": self.toolbox.save_cleaned_document,
            "skip_document": self.toolbox.skip_document,
            "finalize": self.toolbox.finalize,
        }
        tool = tool_map.get(name)
        if tool is None:
            LOGGER.error("Unknown cleaning tool requested: %s", name)
            return {"error": f"Unknown tool: {name}"}
        return tool(**args)


def clean_markdown(markdown: str) -> str:
    """Remove common boilerplate and PDF conversion artifacts from Markdown."""

    text = markdown.replace("\f", "\n")
    lines = []
    previous_blank = False
    for raw_line in text.splitlines():
        line = normalize_line(raw_line)
        if should_drop_line(line):
            continue

        is_blank = not line
        if is_blank and previous_blank:
            continue
        lines.append(line)
        previous_blank = is_blank

    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def normalize_line(line: str) -> str:
    """Normalize whitespace inside one converted Markdown line."""

    return re.sub(r"\s+", " ", line).strip()


def should_drop_line(line: str) -> bool:
    """Return True for boilerplate, navigation, and PDF page artifact lines."""

    lower_line = line.lower()
    if not line:
        return False
    if line in {"EN", "en"}:
        return True
    if re.fullmatch(r"\d{1,4}", line):
        return True
    if re.fullmatch(r"[.\s]{8,}", line):
        return True
    if any(marker in lower_line for marker in NAVIGATION_MARKERS):
        return True
    if lower_line.startswith(("press contacts:", "general public inquiries:")):
        return True
    if lower_line.startswith(("print isbn", "pdf isbn", "isbn ", "doi:")):
        return True
    if lower_line.startswith(("reuse is authorised", "reuse of this document")):
        return True
    return False


def metadata_from_path(path: Path) -> dict[str, str]:
    """Infer simple RAG metadata from a Markdown filename."""

    title = path.stem.replace("_", " ").replace("-", " ").strip()
    return {
        "source": title,
        "topic": infer_topic(title),
        "file_path": str(path),
    }


def infer_topic(text: str) -> str:
    """Infer a coarse topic label from a filename or title."""

    text_lower = text.lower()
    if "2040" in text_lower:
        return "climate_target_2040"
    if "climate law" in text_lower:
        return "climate_law"
    if "adaptation" in text_lower:
        return "adaptation"
    if "energy" in text_lower:
        return "energy_and_climate"
    if "industrial deal" in text_lower:
        return "clean_industrial_deal"
    return "general"


def has_enough_climate_signal(text: str, threshold: int = 2) -> bool:
    """Return True when text contains enough climate-policy keywords."""

    text_lower = text.lower()
    return sum(1 for keyword in CLIMATE_KEYWORDS if keyword in text_lower) >= threshold


def hash_markdown(markdown: str) -> str:
    """Return a stable content hash for cleaned Markdown."""

    normalized = normalize_markdown_for_hash(markdown)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def run_cli(
    input_directory: Annotated[
        Path,
        typer.Option(
            "--input-directory",
            help="Directory containing fetched Markdown files.",
            show_default=True,
        ),
    ] = Path("climate_policy_docs"),
    output_path: Annotated[
        Path,
        typer.Option(
            "--output-path",
            help="JSON path to write cleaned RAG records.",
            show_default=True,
        ),
    ] = Path("data/eu_climate_policy.json"),
    agentic: Annotated[
        bool,
        typer.Option(
            "--agentic/--deterministic",
            help="Use an LLM curator with deterministic cleaning tools.",
            show_default=True,
        ),
    ] = False,
    max_turns: Annotated[
        int,
        typer.Option(
            "--max-turns",
            help="Maximum LLM tool-loop turns for agentic cleaning.",
            show_default=True,
        ),
    ] = 50,
) -> None:
    """Clean fetched Markdown and write one JSON record per kept file."""

    config = IngestionConfigModel(
        input_directory=input_directory,
        output_path=output_path,
        agentic=agentic,
        max_turns=max_turns,
    )
    if agentic:
        message = CleaningCurationAgent(
            toolbox=CleaningToolbox(config.input_directory, config.output_path)
        ).run(max_turns=config.max_turns)
        LOGGER.info(message)
        return

    result = FetchedDocumentIngestor().ingest_directory(
        config.input_directory,
        config.output_path,
    )
    LOGGER.info("Input files: %s", result.input_count)
    LOGGER.info("Kept files:  %s", result.kept_count)
    LOGGER.info("Skipped:     %s", result.skipped_count)
    LOGGER.info("Records:     %s", result.record_count)
    LOGGER.info("Output:      %s", result.output_path)
    for skipped in result.skipped:
        LOGGER.warning("Skipped %s: %s", skipped["path"], skipped["reason"])


def main() -> None:
    """Run the Typer command-line interface."""

    typer.run(run_cli)


CLEANING_AGENT_INSTRUCTIONS = """
You are a document cleaning curator for an EU climate policy RAG dataset.

Use the provided tools only. Do not rewrite document content yourself.

Workflow:
1. Call list_documents.
2. For every document path, call inspect_document.
3. If deterministic_skip_reason is present, call skip_document with that reason.
4. If the cleaned preview is substantively about EU climate policy, call save_cleaned_document.
5. Otherwise call skip_document with a concise reason.
6. After every document has been saved or skipped, call finalize.
7. Return a short summary with record and skipped counts.
""".strip()


CLEANING_TOOLS = [
    {
        "type": "function",
        "name": "list_documents",
        "description": "List fetched Markdown documents available for cleaning.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "inspect_document",
        "description": "Return deterministic cleaning preview and skip recommendation.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "save_cleaned_document",
        "description": "Save one deterministic cleaned JSON record for the document.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "skip_document",
        "description": "Skip one document with a reason.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["path", "reason"],
        },
    },
    {
        "type": "function",
        "name": "finalize",
        "description": "Write cleaned records to the output JSON file.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]


if __name__ == "__main__":
    main()
