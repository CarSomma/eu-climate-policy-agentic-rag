"""LLM-assisted cleaning curation agent."""

from typing import Any

from openai import OpenAI

from eu_climate_policy_rag.collection.cleaning.cleaning_toolbox import CleaningToolbox
from eu_climate_policy_rag.collection.cleaning.cleaning_tools import build_cleaning_tools
from eu_climate_policy_rag.core.agent import AbstractAgent
from eu_climate_policy_rag.core.logging_utils import get_logger

LOGGER = get_logger(__name__)


CLEANING_AGENT_INSTRUCTIONS = """
You are a document cleaning curator for an EU climate policy RAG dataset.

Use the provided tools only. Do not rewrite document content yourself.

Workflow:
1. Call list_documents.
2. For every document path, call inspect_document.
3. If skip_reason is present, call skip_document with that reason.
4. If skip_reason is absent, call save_cleaned_document. Treat EU climate laws,
   amendments, proposals, communications, staff working documents, factsheets,
   press releases, and Q&As about climate targets, adaptation, emissions,
   decarbonisation, or the Clean Industrial Deal as relevant policy content.
5. Only call skip_document for documents rejected by inspect_document.
6. After every document has been saved or skipped, call finalize.
7. Return a short summary with record and skipped counts.
""".strip()


class CleaningCurationAgent(AbstractAgent):
    """LLM-assisted cleaner restricted to curation tools."""

    def __init__(
        self,
        openai_client: OpenAI | None = None,
        model: str = "gpt-5.4-mini",
        toolbox: CleaningToolbox | None = None,
    ) -> None:
        super().__init__(
            openai_client=openai_client,
            model=model,
            instructions=CLEANING_AGENT_INSTRUCTIONS,
            tools=build_cleaning_tools(toolbox or CleaningToolbox()),
            max_turns=50,
        )

    def run_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Dispatch one cleaning tool call by name."""

        result = self._tool_executor.run_sync(name, args)
        if result.ok:
            return result.value
        if result.error is not None:
            return {"error": result.error.message}
        return {"error": result.output}

    def run(self, query: str = "Clean and curate the fetched Markdown documents.", max_turns: int | None = None) -> str:
        """Run the cleaning tool loop until the model finalizes or stops."""

        if max_turns is not None:
            self.max_turns = max_turns
        LOGGER.info("Starting cleaning curation agent")
        final_answer, _ = self._run_loop(query)
        LOGGER.info("Cleaning curation agent finished")
        return final_answer
