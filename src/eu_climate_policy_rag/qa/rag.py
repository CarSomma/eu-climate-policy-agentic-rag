"""Agentic RAG wrapper for EU climate policy Q&A."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Annotated

import typer
from openai import OpenAI

from eu_climate_policy_rag.core.logging_utils import get_logger
from eu_climate_policy_rag.core.models import CleanedDocumentRecordModel, RagAnswerModel, RagConfigModel
from eu_climate_policy_rag.qa.tools import SearchDocumentsTool, format_context_item

logger = get_logger(__name__)

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


class ClimatePolicyAgent:
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
    ) -> None:
        self.documents = [CleanedDocumentRecordModel.model_validate(d).model_dump() for d in documents]
        self.openai_client = openai_client or OpenAI()
        self.config = RagConfigModel(
            model=model,
            num_results=num_results,
            instructions=instructions,
            max_chars_per_doc=max_chars_per_doc,
        )
        self.max_turns = max_turns
        self.search_tool = SearchDocumentsTool(
            documents=self.documents,
            num_results=self.config.num_results,
            max_chars_per_doc=self.config.max_chars_per_doc,
        )
        self.tools = {self.search_tool.name: self.search_tool}

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

    def _execute_tool_call(self, tool_call: Any) -> tuple[dict[str, Any], list[str]]:
        """Dispatch a function_call message to the matching tool and return its output."""
        arguments = json.loads(tool_call.arguments)
        tool = self.tools.get(tool_call.name)
        sources: list[str] = []
        if tool:
            query = arguments["query"]
            logger.info("Searching: %s", query)
            result = tool.run(query)
            output = result.context
            sources = result.sources
        else:
            output = f"Unknown tool: {tool_call.name}"

        return {
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": output,
        }, sources

    def answer(self, query: str) -> RagAnswerModel:
        """Answer a user question using iterative document search."""

        message_history: list[Any] = [
            {"role": "system", "content": self.config.instructions},
            {"role": "user", "content": query},
        ]

        all_sources: list[str] = []
        final_answer = ""

        for turn in range(1, self.max_turns + 1):
            logger.info("Agent turn %d", turn)
            response = self.openai_client.responses.create(
                model=self.config.model,
                input=message_history,
                tools=[tool.schema for tool in self.tools.values()],
            )
            message_history.extend(response.output)

            has_tool_call = False
            for message in response.output:
                if message.type == "function_call":
                    has_tool_call = True
                    tool_output, sources = self._execute_tool_call(message)
                    all_sources.extend(sources)
                    message_history.append(tool_output)
                elif message.type == "message":
                    final_answer = message.content[0].text

            if not has_tool_call:
                break

        sources = list(dict.fromkeys(s for s in all_sources if s))
        return RagAnswerModel(query=query, answer=final_answer, sources=sources)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    question: Annotated[str, typer.Argument(help="The question to ask the RAG assistant.")],
    data: Annotated[str, typer.Option(help="Path to the JSON data file.")] = "data/eu_climate_policy.json",
    model: Annotated[str, typer.Option(help="OpenAI model to use.")] = "gpt-4o-mini",
    num_results: Annotated[int, typer.Option(help="Number of documents to retrieve.")] = 5,
    max_chars_per_doc: Annotated[int, typer.Option(help="Max characters per retrieved document.")] = 2000,
    max_turns: Annotated[int, typer.Option(help="Max agent turns before stopping.")] = 10,
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
