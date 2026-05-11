"""Helper functions for document fetch pipeline selection."""

from collections.abc import Callable, Mapping, Sequence

from eu_climate_policy_rag.collection.document_urls import (
    UrlNormalizer,
    canonical_document_key,
)
from eu_climate_policy_rag.core.types import DocumentCandidate, Link


DocumentFilter = Callable[[DocumentCandidate], bool]


def flatten_sections(
    sections: Mapping[str, Sequence[Link]],
    url_normalizer: UrlNormalizer | None = None,
) -> list[DocumentCandidate]:
    """Flatten section-grouped discovered links while preserving the section name."""

    normalizer = url_normalizer or UrlNormalizer()
    flattened = []
    for section, links in sections.items():
        for link in links:
            flattened.append(
                {
                    "title": link["text"],
                    "url": normalizer.normalize(link["href"]),
                    "section": section,
                }
            )
    return deduplicate_documents(flattened)


def deduplicate_documents(
    documents: Sequence[DocumentCandidate],
) -> list[DocumentCandidate]:
    """Deduplicate documents by canonical document identity."""

    deduplicated = []
    seen_keys = set()
    for document in documents:
        key = canonical_document_key(document["url"])
        if key in seen_keys:
            continue
        deduplicated.append(document)
        seen_keys.add(key)
    return deduplicated


def select_documents(
    documents: Sequence[DocumentCandidate],
    document_filter: DocumentFilter,
    limit: int | None = None,
) -> list[DocumentCandidate]:
    """Return documents accepted by a filter, optionally capped by a limit."""

    selected = [document for document in documents if document_filter(document)]
    if limit is not None:
        return selected[:limit]
    return selected


def is_relevant_document(document: DocumentCandidate) -> bool:
    """Return True when a discovered document title looks relevant."""

    return has_climate_signal(document["title"])


def has_climate_signal(text: str) -> bool:
    """Return True when text appears relevant to EU climate policy."""

    climate_terms = (
        "climate",
        "greenhouse",
        "ghg",
        "emission",
        "carbon",
        "decarbon",
        "net-zero",
        "net zero",
        "energy",
        "adaptation",
        "industrial deal",
    )
    return any(term in text.lower() for term in climate_terms)
