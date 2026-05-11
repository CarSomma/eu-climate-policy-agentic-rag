import pytest

from eu_climate_policy_rag.collection.discovery.candidate_utils import (
    deduplicate_documents,
    flatten_sections,
    has_climate_signal,
    is_relevant_document,
    select_documents,
)


def test_flatten_sections_adds_section_and_deduplicates_by_url() -> None:
    link = {
        "text": "European Climate Law",
        "href": "https://example.test/law",
    }

    flattened = flatten_sections({"Main": [link], "Duplicate": [link]})

    assert flattened == [
        {
            "section": "Main",
            "title": "European Climate Law",
            "url": "https://example.test/law",
        }
    ]


def test_deduplicate_documents_keeps_first_url() -> None:
    first = {
        "title": "First",
        "url": "https://example.test/doc",
    }
    second = {**first, "title": "Second"}

    assert deduplicate_documents([first, second]) == [first]


def test_deduplicate_documents_matches_eur_lex_eli_and_celex_urls() -> None:
    eli = {
        "title": "Regulation (EU) 2026/667",
        "url": "https://eur-lex.europa.eu/eli/reg/2026/667/oj/eng",
    }
    celex = {
        "title": "Regulation - EU - 2026/667",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32026R0667",
    }

    assert deduplicate_documents([eli, celex]) == [eli]


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Questions and answers on the 2040 EU climate target", True),
        ("Single market focus areas for enforcement", False),
    ],
    ids=["climate-title", "off-topic-title"],
)
def test_is_relevant_document_uses_title_climate_signal(
    title: str,
    expected: bool,
) -> None:
    document = {"title": title, "url": "https://example.test/doc"}

    assert is_relevant_document(document) is expected
    assert has_climate_signal(title) is expected


def test_select_documents_filters_and_applies_limit() -> None:
    documents = [
        {"title": "Climate target", "url": "https://example.test/1"},
        {"title": "Single market", "url": "https://example.test/2"},
        {"title": "Climate law", "url": "https://example.test/3"},
    ]

    selected = select_documents(documents, is_relevant_document, limit=1)

    assert selected == [documents[0]]
