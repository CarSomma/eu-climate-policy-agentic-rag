import pytest

from eu_climate_policy_rag.collection.content_hashing import hash_markdown_content
from eu_climate_policy_rag.collection.document_quality import (
    DocumentQualityCheck,
    normalize_markdown_for_hash,
)


def test_accepts_substantive_climate_policy_content(climate_markdown: str) -> None:
    result = DocumentQualityCheck().assess("2040 climate target", climate_markdown)

    assert result.accepted
    assert result.reason == "accepted"


@pytest.mark.parametrize(
    ("title", "markdown", "expected_reason"),
    [
        (
            "Climate Law",
            "Too short.",
            "content is too short to be useful",
        ),
        (
            "Single market focus areas",
            (
                "This annex discusses single market enforcement, product labelling, "
                "capital markets, savings accounts, and cross-border services. "
            )
            * 10,
            "content is not clearly about EU climate policy",
        ),
        (
            "Commission page",
            """
            Accept all cookies
            Accept only essential cookies
            Skip to main content
            Select your language
            Official website of the European Union
            How do you know?
            See all EU institutions
            Type of search results
            Search on:
            Climate
            """
            * 12,
            "content appears to be mostly page navigation",
        ),
    ],
    ids=["too-short", "off-topic", "navigation-heavy"],
)
def test_rejects_low_quality_content(
    title: str,
    markdown: str,
    expected_reason: str,
) -> None:
    result = DocumentQualityCheck().assess(title, markdown)

    assert not result.accepted
    assert result.reason == expected_reason


def test_rejects_duplicate_content_hash(climate_markdown: str) -> None:
    existing_hash = hash_markdown_content(climate_markdown)

    result = DocumentQualityCheck().assess(
        "Climate Law",
        climate_markdown,
        existing_hashes={existing_hash},
    )

    assert not result.accepted
    assert result.reason == "duplicate content already exists"


def test_rejects_duplicate_after_whitespace_normalization() -> None:
    original = "\n".join(["Climate law greenhouse gas emissions."] * 40)
    equivalent = "\n  Climate   law greenhouse gas emissions.  \n" * 40
    existing_hash = hash_markdown_content(original)

    result = DocumentQualityCheck().assess(
        "Climate Law",
        equivalent,
        existing_hashes={existing_hash},
    )

    assert not result.accepted
    assert result.reason == "duplicate content already exists"


def test_accepts_content_at_minimum_character_boundary() -> None:
    result = DocumentQualityCheck(minimum_characters=10).assess(
        "Climate Law",
        "Climate emissions target",
    )

    assert result.accepted


def test_assess_returns_consistent_content_hash(climate_markdown: str) -> None:
    result1 = DocumentQualityCheck().assess("Climate Law", climate_markdown)
    result2 = DocumentQualityCheck().assess("Climate Law", climate_markdown)

    assert result1.content_hash == result2.content_hash


def test_normalize_markdown_for_hash_strips_whitespace() -> None:
    markdown = "  Hello   world  \n\n  foo  bar  "

    result = normalize_markdown_for_hash(markdown)

    assert result == "Hello world\nfoo bar"
