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


def test_answer_structured_reports_usage_summed_over_model_calls(
    sample_document, make_tool_call, make_message_response
) -> None:
    from types import SimpleNamespace

    def with_usage(response, input_tokens, output_tokens, cached=0, extra_output=()):
        response.model = "gpt-5.4-mini-2026-03-17"
        response.output = [*response.output, *extra_output]
        response.usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=cached),
        )
        return response

    tool_turn = SimpleNamespace(output=[make_tool_call("search_documents", {"query": "2030 target"})])
    client = MagicMock()
    client.responses.create.side_effect = [
        with_usage(tool_turn, 1000, 20, extra_output=[SimpleNamespace(type="web_search_call")]),
        with_usage(make_message_response(json.dumps(STRUCTURED)), 3000, 200, cached=1000),
    ]
    agent = ClimatePolicyAgent([sample_document], openai_client=client)

    usage = agent.answer_structured("What is the 2030 target?").usage

    assert usage.model_calls == 2
    assert usage.input_tokens == 4000
    assert usage.cached_input_tokens == 1000
    assert usage.output_tokens == 220
    assert usage.web_search_calls == 1
    assert usage.estimated_cost_usd == round((3000 * 0.75 + 1000 * 0.075 + 220 * 4.50) / 1_000_000, 10)
    assert agent.usage_tracker is None
