"""Quality and duplicate-detection checks for collection workflows."""

from eu_climate_policy_rag.collection.content_hashing import (
    existing_markdown_hashes as existing_markdown_hashes,
    hash_markdown_content as hash_markdown_content,
    hash_normalized_markdown,
    normalize_markdown_for_hash,
)
from eu_climate_policy_rag.core.models import PreselectionResultModel


class DocumentQualityCheck:
    """Conservative checks for fetched EU climate policy documents."""

    def __init__(
        self,
        minimum_characters: int = 800,
        climate_keyword_threshold: int = 2,
        navigation_marker_threshold: int = 5,
    ) -> None:
        self.minimum_characters = minimum_characters
        self.climate_keyword_threshold = climate_keyword_threshold
        self.navigation_marker_threshold = navigation_marker_threshold

    def assess(
        self,
        title: str,
        markdown: str,
        existing_hashes: set[str] | None = None,
    ) -> PreselectionResultModel:
        """Return whether converted Markdown is worth saving into fetched data."""

        normalized_text = normalize_markdown_for_hash(markdown)
        content_hash = hash_normalized_markdown(normalized_text)
        existing_hashes = existing_hashes or set()

        if content_hash in existing_hashes:
            return PreselectionResultModel(
                accepted=False,
                reason="duplicate content already exists",
                content_hash=content_hash,
            )

        if len(normalized_text) < self.minimum_characters:
            return PreselectionResultModel(
                accepted=False,
                reason="content is too short to be useful",
                content_hash=content_hash,
            )

        searchable_text = f"{title}\n{normalized_text}".lower()
        navigation_hits = count_keyword_hits(searchable_text, NAVIGATION_MARKERS)
        climate_hits = count_keyword_hits(searchable_text, CLIMATE_KEYWORDS)
        if (
            navigation_hits >= self.navigation_marker_threshold
            and climate_hits < self.climate_keyword_threshold * 2
        ):
            return PreselectionResultModel(
                accepted=False,
                reason="content appears to be mostly page navigation",
                content_hash=content_hash,
            )

        if climate_hits < self.climate_keyword_threshold:
            return PreselectionResultModel(
                accepted=False,
                reason="content is not clearly about EU climate policy",
                content_hash=content_hash,
            )

        return PreselectionResultModel(
            accepted=True,
            reason="accepted",
            content_hash=content_hash,
        )


def count_keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    """Count how many configured keywords appear in text."""

    return sum(1 for keyword in keywords if keyword in text)


CLIMATE_KEYWORDS = (
    "climate",
    "greenhouse gas",
    "ghg",
    "emission",
    "decarbon",
    "net-zero",
    "net zero",
    "carbon",
    "eu ets",
    "climate neutrality",
    "european climate law",
    "adaptation",
    "paris agreement",
    "energy transition",
)

NAVIGATION_MARKERS = (
    "accept all cookies",
    "accept only essential cookies",
    "skip to main content",
    "select your language",
    "official website of the european union",
    "how do you know?",
    "see all eu institutions",
    "type of search results",
    "search on:",
    "switch to mobile",
    "switch to desktop",
    "log in",
    "my eur-lex",
)
