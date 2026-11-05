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


def test_pricing_matches_dated_model_snapshots() -> None:
    from eu_climate_policy_rag.core.metrics import DEFAULT_PRICING

    tracker = ResponseUsageTracker(DEFAULT_PRICING)
    record = tracker.record_response(
        SimpleNamespace(
            model="gpt-5.4-mini-2026-03-17",
            usage=SimpleNamespace(
                input_tokens=1_000_000,
                output_tokens=1_000_000,
                total_tokens=2_000_000,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        )
    )

    assert record.estimated_cost_usd == 0.75 + 4.50


def test_unknown_model_has_no_cost_estimate() -> None:
    tracker = ResponseUsageTracker({"gpt-5.4-mini": ModelPricing(0.75, 4.50)})

    record = tracker.record_response(
        SimpleNamespace(model="gpt-5.4", usage=SimpleNamespace(input_tokens=10, output_tokens=1, total_tokens=11))
    )

    assert record.estimated_cost_usd is None
