import json
from unittest.mock import MagicMock

from eu_climate_policy_rag.core.models import ClimateAnswerOutputModel
from eu_climate_policy_rag.qa.rag import ClimatePolicyAgent

STRUCTURED = {
    "answer": "2030: at least 55%; 2040: 90%.",
    "targets": [
        {
            "year": 2030,
            "description": "at least 55% net GHG reduction",
            "reduction_percent": 55.0,
            "baseline_year": 1990,
            "binding": True,
            "legal_basis": "Regulation (EU) 2021/1119",
            "article": "Article 4(1)",
        }
    ],
    "sources": ["Regulation (EU) 2021/1119"],
}


def test_answer_structured_sends_strict_json_schema_and_parses_it(
    sample_document, make_message_response
) -> None:
    client = MagicMock()
    client.responses.create.return_value = make_message_response(json.dumps(STRUCTURED))
    agent = ClimatePolicyAgent([sample_document], openai_client=client)

    result = agent.answer_structured("What is the 2030 target?")

    text_format = client.responses.create.call_args.kwargs["text"]["format"]
    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True
    assert text_format["schema"] == ClimateAnswerOutputModel.model_json_schema()
    assert result.query == "What is the 2030 target?"
    assert result.targets[0].reduction_percent == 55.0
    assert result.targets[0].legal_basis == "Regulation (EU) 2021/1119"
    assert agent.text_format is None


def test_plain_answer_sends_no_text_format(sample_document, make_message_response) -> None:
    client = MagicMock()
    client.responses.create.return_value = make_message_response("Plain text.")
    agent = ClimatePolicyAgent([sample_document], openai_client=client)

    agent.answer("What is the 2030 target?")

    assert "text" not in client.responses.create.call_args.kwargs


def test_answer_structured_falls_back_to_text_when_json_is_invalid(
    sample_document, make_message_response
) -> None:
    client = MagicMock()
    client.responses.create.return_value = make_message_response("Agent stopped: max turns reached.")
    agent = ClimatePolicyAgent([sample_document], openai_client=client)

    result = agent.answer_structured("What is the 2030 target?")

    assert result.answer == "Agent stopped: max turns reached."
    assert result.targets == []


def test_structured_schema_meets_openai_strict_rules() -> None:
    """Strict mode needs every property required and no extra properties, at every level."""

    schema = ClimateAnswerOutputModel.model_json_schema()
    objects = [schema, *schema["$defs"].values()]

    for obj in objects:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
