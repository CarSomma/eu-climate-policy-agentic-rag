"""Unit tests for model response usage and cost metrics."""

from types import SimpleNamespace

from eu_climate_policy_rag.core.metrics import (
    ModelPricing,
    ResponseUsageTracker,
)


def test_response_usage_tracker_records_token_usage_and_estimated_cost() -> None:
    """Response metrics should estimate cost from usage and configured pricing."""

    tracker = ResponseUsageTracker(
        pricing_by_model={
            "gpt-test": ModelPricing(
                input_per_1m=2.0,
                output_per_1m=8.0,
                cached_input_per_1m=0.5,
            ),
        },
    )
    response = SimpleNamespace(
        model="gpt-test",
        usage=SimpleNamespace(
            input_tokens=1000,
            input_tokens_details=SimpleNamespace(cached_tokens=200),
            output_tokens=500,
            total_tokens=1500,
        ),
    )

    record = tracker.record_response(response)

    assert record.model == "gpt-test"
    assert record.input_tokens == 1000
    assert record.cached_input_tokens == 200
    assert record.output_tokens == 500
    assert record.total_tokens == 1500
    assert record.estimated_cost_usd == 0.0057
    assert tracker.total_input_tokens == 1000
    assert tracker.total_output_tokens == 500
    assert tracker.total_tokens == 1500
    assert tracker.total_estimated_cost_usd == 0.0057


def test_response_usage_tracker_handles_unpriced_models() -> None:
    """Token usage should still be recorded when pricing is not configured."""

    tracker = ResponseUsageTracker()
    response = {
        "model": "unknown-model",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
    }

    record = tracker.record_response(response)

    assert record.model == "unknown-model"
    assert record.estimated_cost_usd is None
    assert tracker.total_tokens == 15
    assert tracker.total_estimated_cost_usd is None
