"""Agentic RAG wrapper for EU climate policy Q&A."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Annotated

import typer
from openai import OpenAI

from eu_climate_policy_rag.core.agent import AbstractAgent
from eu_climate_policy_rag.core.logging_utils import get_logger
from eu_climate_policy_rag.core.models import (
    CleanedDocumentRecordModel,
    RagAnswerModel,
    RagConfigModel,
    SearchDocumentsResultModel,
)
from eu_climate_policy_rag.core.tooling import ToolRegistry
from eu_climate_policy_rag.qa.tools import (
    SearchDocumentsTool,
    format_context_item as format_context_item,
)

LOGGER = get_logger(__name__)

DEFAULT_INSTRUCTIONS = """
You are an expert assistant on EU climate policy.

You have access to a search tool that queries a local index of official EU policy documents
(European Climate Law, European Green Deal, Fit for 55, EU ETS, CBAM, and the Paris Agreement).

Your process:
1. Use the search_documents tool one or more times to retrieve relevant passages.
2. Once you have enough information, write a final answer.

Rules:
- Cite the source and article for every claim you make (e.g. "European Climate Law, Article 4").
- If the retrieved documents do not contain enough information to answer, say so explicitly. Do not guess.
- Be concise and accurate. Connect related targets or policies when the context supports it.
""".strip()


class ClimatePolicyAgent(AbstractAgent):
    """Answer EU climate policy questions with an LLM-driven search loop."""

    def __init__(
        self,
        documents: Sequence[dict[str, Any]],
        openai_client: OpenAI | None = None,
        model: str = "gpt-4o-mini",
        instructions: str = DEFAULT_INSTRUCTIONS,
        num_results: int = 5,
        max_chars_per_doc: int = 2000,
        max_turns: int = 10,
        search_tool: SearchDocumentsTool | None = None,
    ) -> None:
        self.documents = [
            CleanedDocumentRecordModel.model_validate(document).model_dump()
            for document in documents
        ]
        self.config = RagConfigModel(
            model=model,
            num_results=num_results,
            instructions=instructions,
            max_chars_per_doc=max_chars_per_doc,
        )
        self.search_tool = search_tool or SearchDocumentsTool(
            documents=self.documents,
            num_results=self.config.num_results,
            max_chars_per_doc=self.config.max_chars_per_doc,
        )
        super().__init__(
            openai_client=openai_client,
            model=self.config.model,
            instructions=self.config.instructions,
            tools=ToolRegistry([self.search_tool.function_tool]),
            max_turns=max_turns,
        )
        self._current_run_sources: list[str] = []

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        openai_client: OpenAI | None = None,
        model: str = "gpt-4o-mini",
        instructions: str = DEFAULT_INSTRUCTIONS,
        num_results: int = 5,
        max_chars_per_doc: int = 2000,
        max_turns: int = 10,
    ) -> "ClimatePolicyAgent":
        """Load cleaned JSON records and build a configured RAG agent."""

        with Path(path).open(encoding="utf-8") as file:
            documents = json.load(file)
        return cls(
            documents=documents,
            openai_client=openai_client,
            model=model,
            instructions=instructions,
            num_results=num_results,
            max_chars_per_doc=max_chars_per_doc,
            max_turns=max_turns,
        )

    def search(self, query: str) -> list[dict[str, Any]]:
        """Search the local document index for relevant records."""

        return self.search_tool.search(query)

    def _execute_tool_call(self, tool_call: Any) -> dict[str, Any]:
        """Dispatch a function_call to the search tool and collect sources as a side-effect."""

        arguments = json.loads(tool_call.arguments)
        if self.tools.get(tool_call.name) is None:
            LOGGER.error("Unknown RAG tool requested: %s", tool_call.name)
        elif tool_call.name == self.search_tool.name:
            LOGGER.info("Searching: %s", arguments["query"])

        result = self.tools.run_sync(tool_call.name, arguments)
        if isinstance(result, SearchDocumentsResultModel):
            output = result.context
            self._current_run_sources.extend(result.sources)
        else:
            output = str(result.get("error", result)) if isinstance(result, dict) else str(result)

        return {
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": output,
        }

    def run(self, query: str) -> RagAnswerModel:
        """Implement the abstract ``run`` method; delegates to ``answer``."""

        return self.answer(query)

    def answer(self, query: str) -> RagAnswerModel:
        """Answer a user question using iterative document search."""

        self._current_run_sources = []
        final_answer, _ = self._run_loop(query)
        sources = list(dict.fromkeys(s for s in self._current_run_sources if s))
        return RagAnswerModel(query=query, answer=final_answer, sources=sources)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    question: Annotated[
        str,
        typer.Argument(help="The question to ask the RAG assistant."),
    ],
    data: Annotated[
        str,
        typer.Option(help="Path to the JSON data file."),
    ] = "data/eu_climate_policy.json",
    model: Annotated[str, typer.Option(help="OpenAI model to use.")] = "gpt-4o-mini",
    num_results: Annotated[
        int,
        typer.Option(help="Number of documents to retrieve."),
    ] = 5,
    max_chars_per_doc: Annotated[
        int,
        typer.Option(help="Max characters per retrieved document."),
    ] = 2000,
    max_turns: Annotated[
        int,
        typer.Option(help="Max agent turns before stopping."),
    ] = 10,
) -> None:
    """Ask a question about EU climate policy and get a cited answer."""

    rag = ClimatePolicyAgent.from_json(
        data,
        model=model,
        num_results=num_results,
        max_chars_per_doc=max_chars_per_doc,
        max_turns=max_turns,
    )
    result = rag.answer(question)
    typer.echo(result.answer)


if __name__ == "__main__":
    app()
