"""Middleware hooks for tool execution."""

from collections.abc import Mapping

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
