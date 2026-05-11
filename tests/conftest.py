import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = str(item.fspath)
        if "/unit/" in path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in path:
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def sample_document() -> dict[str, str]:
    return {
        "source": "European Climate Law",
        "article": "Article 4",
        "topic": "climate_law",
        "text": "The Union-wide 2030 climate target is binding.",
        "file_path": "law.md",
        "content_hash": "abc123",
    }


@pytest.fixture
def climate_markdown() -> str:
    return (
        "The European Climate Law and 2040 climate target concern greenhouse "
        "gas emission reductions, climate neutrality, and the energy transition. "
    ) * 10


@pytest.fixture
def off_topic_markdown() -> str:
    return (
        "This annex discusses single market enforcement, product labelling, "
        "capital markets, savings accounts, and cross-border services. "
    ) * 10


@pytest.fixture
def make_message_response() -> Callable[[str], SimpleNamespace]:
    def factory(text: str) -> SimpleNamespace:
        message = SimpleNamespace(
            type="message",
            content=[SimpleNamespace(text=text)],
        )
        return SimpleNamespace(output=[message])

    return factory


@pytest.fixture
def make_tool_call() -> Callable[..., SimpleNamespace]:
    def factory(
        name: str,
        arguments: dict[str, Any] | None = None,
        call_id: str = "call_001",
    ) -> SimpleNamespace:
        return SimpleNamespace(
            type="function_call",
            name=name,
            arguments=json.dumps(arguments or {}),
            call_id=call_id,
        )

    return factory


@pytest.fixture
def make_tool_call_response(
    make_tool_call: Callable[..., SimpleNamespace],
) -> Callable[..., SimpleNamespace]:
    def factory(
        name: str,
        arguments: dict[str, Any] | None = None,
        call_id: str = "call_001",
    ) -> SimpleNamespace:
        return SimpleNamespace(output=[make_tool_call(name, arguments, call_id)])

    return factory


@pytest.fixture
def write_markdown_file() -> Callable[[Path, str, str], Path]:
    def factory(directory: Path, name: str, markdown: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        path.write_text(markdown, encoding="utf-8")
        return path

    return factory
