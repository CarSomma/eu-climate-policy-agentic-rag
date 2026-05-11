import pytest

from eu_climate_policy_rag.collection.cleaning.markdown_cleaning import (
    clean_markdown,
    infer_topic,
    should_drop_line,
)


def test_clean_markdown_removes_common_pdf_and_web_artifacts() -> None:
    markdown = """
    EN
    12
    Accept all cookies
    The European Climate Law supports climate neutrality.


    Greenhouse gas emissions must fall.
    Press contacts:
    """

    cleaned = clean_markdown(markdown)

    assert "Accept all cookies" not in cleaned
    assert "Press contacts" not in cleaned
    assert "\n12\n" not in cleaned
    assert "European Climate Law" in cleaned


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("EN", True),
        ("42", True),
        ("Select your language", True),
        ("The 2040 climate target matters.", False),
    ],
    ids=["language-code", "page-number", "navigation", "content"],
)
def test_should_drop_line_identifies_artifacts(line: str, expected: bool) -> None:
    assert should_drop_line(line) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2040 climate target", "climate_target_2040"),
        ("climate law regulation", "climate_law"),
    ],
    ids=["target-2040", "climate-law"],
)
def test_infer_topic(text: str, expected: str) -> None:
    assert infer_topic(text) == expected
