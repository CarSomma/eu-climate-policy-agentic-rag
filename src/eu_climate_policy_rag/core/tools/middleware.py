"""Middleware hooks for tool execution."""

from collections.abc import Callable, Mapping
from time import perf_counter

from eu_climate_policy_rag.core.tools.context import ToolContext


class ToolMiddleware:
    """Base class for synchronous tool execution middleware."""

    def before_validate(
        self,
        context: ToolContext,
        args: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Run before argument validation."""

        return args

    def after_validate(self, context: ToolContext, validated: object) -> object:
        """Run after argument validation."""

        return validated

    def before_call(
        self,
        context: ToolContext,
        call_args: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Run before the handler call."""

        return call_args

    def after_call(self, context: ToolContext, result: object) -> object:
        """Run after the handler call."""

        return result

    def on_error(self, context: ToolContext, error: Exception) -> None:
        """Run when validation or handler execution fails."""


class ToolMetricsMiddleware(ToolMiddleware):
    """Record compact provider-neutral metrics for tool execution."""

    def __init__(
        self,
        emit: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.emit = emit
        self.events: list[dict[str, object]] = []

    def before_validate(
        self,
        context: ToolContext,
        args: Mapping[str, object],
    ) -> Mapping[str, object]:
        context.metadata["_metrics_started_at"] = perf_counter()
        return args

    def after_call(self, context: ToolContext, result: object) -> object:
        event = self._build_event(context, ok=True)
        context.metadata.pop("_metrics_started_at", None)
        context.metadata["metrics"] = event
        self._record(event)
        return result

    def on_error(self, context: ToolContext, error: Exception) -> None:
        event = self._build_event(
            context,
            ok=False,
            error_type=type(error).__name__,
        )
        context.metadata.pop("_metrics_started_at", None)
        context.metadata["metrics"] = event
        self._record(event)

    def _build_event(
        self,
        context: ToolContext,
        *,
        ok: bool,
        error_type: str | None = None,
    ) -> dict[str, object]:
        started_at = context.metadata.get("_metrics_started_at")
        duration_ms = 0.0
        if isinstance(started_at, float):
            duration_ms = max((perf_counter() - started_at) * 1000, 0.0)

        event: dict[str, object] = {
            "tool_name": context.tool_name,
            "call_id": context.call_id,
            "attempt": context.attempt,
            "ok": ok,
            "duration_ms": duration_ms,
        }
        if error_type is not None:
            event["error_type"] = error_type
        return event

    def _record(self, event: dict[str, object]) -> None:
        self.events.append(event)
        if self.emit is not None:
            self.emit(event)
