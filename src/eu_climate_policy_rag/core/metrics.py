"""Provider-neutral observability helpers."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Text-token pricing in USD per 1M tokens."""

    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float | None = None


# Standard text-token prices in USD per 1M tokens. Checked 2026-11-05 against
# public price trackers (OpenAI's pricing page is the authority); update when
# OpenAI changes prices. Web search calls are billed per call, not per token,
# and are not included in token cost estimates.
DEFAULT_PRICING: dict[str, ModelPricing] = {
    "gpt-5.4-mini": ModelPricing(input_per_1m=0.75, output_per_1m=4.50, cached_input_per_1m=0.075),
    "gpt-5-mini": ModelPricing(input_per_1m=0.25, output_per_1m=2.00, cached_input_per_1m=0.025),
    "gpt-4o-mini": ModelPricing(input_per_1m=0.15, output_per_1m=0.60, cached_input_per_1m=0.075),
}


@dataclass(frozen=True)
class ResponseUsageRecord:
    """Token usage and optional estimated cost for one model response."""

    model: str | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int
    estimated_cost_usd: float | None


class ResponseUsageTracker:
    """Accumulate Responses API token usage and optional cost estimates."""

    def __init__(
        self,
        pricing_by_model: Mapping[str, ModelPricing] | None = None,
    ) -> None:
        self.pricing_by_model = dict(pricing_by_model or {})
        self.records: list[ResponseUsageRecord] = []

    @property
    def total_input_tokens(self) -> int:
        """Return accumulated input tokens."""

        return sum(record.input_tokens for record in self.records)

    @property
    def total_output_tokens(self) -> int:
        """Return accumulated output tokens."""

        return sum(record.output_tokens for record in self.records)

    @property
    def total_tokens(self) -> int:
        """Return accumulated total tokens."""

        return sum(record.total_tokens for record in self.records)

    @property
    def total_estimated_cost_usd(self) -> float | None:
        """Return accumulated estimated cost when all records are priced."""

        costs = [record.estimated_cost_usd for record in self.records]
        if not costs:
            return 0.0
        if any(cost is None for cost in costs):
            return None
        return round(sum(cost for cost in costs if cost is not None), 10)

    def record_response(self, response: object) -> ResponseUsageRecord:
        """Extract usage from a model response and append one usage record."""

        model = _get_value(response, "model")
        usage = _get_value(response, "usage")
        input_tokens = _get_int(usage, "input_tokens")
        output_tokens = _get_int(usage, "output_tokens")
        total_tokens = _get_int(usage, "total_tokens")
        cached_input_tokens = _get_cached_input_tokens(usage)
        estimated_cost_usd = self._estimate_cost(
            model if isinstance(model, str) else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        )
        record = ResponseUsageRecord(
            model=model if isinstance(model, str) else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_input_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )
        self.records.append(record)
        return record

    def _find_pricing(self, model: str) -> ModelPricing | None:
        """Exact model name, else the longest priced name it starts with.

        The API reports dated snapshots such as "gpt-5.4-mini-2026-03-17".
        """

        if model in self.pricing_by_model:
            return self.pricing_by_model[model]
        prefixes = [name for name in self.pricing_by_model if model.startswith(f"{name}-")]
        return self.pricing_by_model[max(prefixes, key=len)] if prefixes else None

    def _estimate_cost(
        self,
        model: str | None,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int,
    ) -> float | None:
        if model is None:
            return None
        pricing = self._find_pricing(model)
        if pricing is None:
            return None

        cached_rate = (
            pricing.cached_input_per_1m
            if pricing.cached_input_per_1m is not None
            else pricing.input_per_1m
        )
        uncached_input_tokens = max(input_tokens - cached_input_tokens, 0)
        cost = (
            (uncached_input_tokens * pricing.input_per_1m)
            + (cached_input_tokens * cached_rate)
            + (output_tokens * pricing.output_per_1m)
        ) / 1_000_000
        return round(cost, 10)


def _get_value(value: object, key: str) -> object:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _get_int(value: object, key: str) -> int:
    item = _get_value(value, key)
    return item if isinstance(item, int) else 0


def _get_cached_input_tokens(usage: object) -> int:
    details = _get_value(usage, "input_tokens_details")
    return _get_int(details, "cached_tokens")
