"""CLI entrypoint for cleaning fetched Markdown into RAG records."""

from pathlib import Path
from typing import Annotated

import typer

from eu_climate_policy_rag.collection.cleaning.cleaning_agent import (
    CLEANING_AGENT_INSTRUCTIONS as CLEANING_AGENT_INSTRUCTIONS,
    CleaningCurationAgent,
)
from eu_climate_policy_rag.collection.cleaning.cleaning_toolbox import CleaningToolbox
from eu_climate_policy_rag.core.logging_utils import get_logger
from eu_climate_policy_rag.core.models import IngestionConfigModel

LOGGER = get_logger(__name__)


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
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="OpenAI model to use for cleaning curation.",
            show_default=True,
        ),
    ] = "gpt-5.4-mini",
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
        model=model,
        max_turns=max_turns,
    )
    message = CleaningCurationAgent(
        toolbox=CleaningToolbox(config.input_directory, config.output_path),
        model=config.model,
    ).run(max_turns=config.max_turns)
    LOGGER.info(message)


def main() -> None:
    """Run the Typer command-line interface."""

    typer.run(run_cli)


if __name__ == "__main__":
    main()
