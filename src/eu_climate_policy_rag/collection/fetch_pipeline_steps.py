"""Fetch execution helpers for the collection pipeline."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from eu_climate_policy_rag.collection.fetching.fetch_agent import DocumentFetchAgent
from eu_climate_policy_rag.core.logging_utils import get_logger
from eu_climate_policy_rag.core.types import DocumentCandidate, PipelineFetchResult

LOGGER = get_logger(__name__)


async def fetch_selected_documents(
    documents: list[DocumentCandidate],
    fetch_agent: DocumentFetchAgent,
    max_turns: int,
) -> list[PipelineFetchResult]:
    """Fetch selected document candidates and return status records."""

    results = []
    for index, document in enumerate(documents, start=1):
        LOGGER.info(
            "Fetching document %s/%s: %s",
            index,
            len(documents),
            document["title"],
        )
        results.append(await fetch_document_candidate(document, fetch_agent, max_turns))
    return results


async def fetch_document_candidate(
    document: DocumentCandidate,
    fetch_agent: DocumentFetchAgent,
    max_turns: int,
) -> PipelineFetchResult:
    """Fetch one document candidate and return a pipeline result record."""

    try:
        before_snapshot = _markdown_snapshot(fetch_agent)
        message = await fetch_agent.fetch_document(
            document["url"],
            max_turns=max_turns,
        )
        if before_snapshot is not None and not _markdown_changed(
            before_snapshot,
            _markdown_snapshot(fetch_agent) or {},
        ):
            return {
                "title": document["title"],
                "url": document["url"],
                "status": "failed",
                "message": f"No Markdown file was written. Agent message: {message}",
            }

        return {
            "title": document["title"],
            "url": document["url"],
            "status": "fetched",
            "message": message,
        }
    except Exception as exc:
        LOGGER.exception("Fetch failed for %s", document["title"])
        return {
            "title": document["title"],
            "url": document["url"],
            "status": "failed",
            "message": str(exc),
        }


def count_fetched(results: Sequence[Mapping[str, str]]) -> int:
    """Count successful fetch result records."""

    return sum(1 for result in results if result["status"] == "fetched")


def _markdown_snapshot(fetch_agent: DocumentFetchAgent) -> dict[Path, tuple[int, int]] | None:
    output_directory = getattr(fetch_agent, "output_directory", None)
    if not isinstance(output_directory, (str, Path)):
        return None

    output_path = Path(output_directory)
    if not output_path.exists():
        return {}

    return {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in output_path.glob("*.md")
    }


def _markdown_changed(
    before: Mapping[Path, tuple[int, int]],
    after: Mapping[Path, tuple[int, int]],
) -> bool:
    return any(after.get(path) != metadata for path, metadata in before.items()) or any(
        path not in before for path in after
    )
