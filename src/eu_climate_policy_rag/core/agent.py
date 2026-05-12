"""Abstract base class for OpenAI agentic tool-call loops."""

import json
from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI

from eu_climate_policy_rag.core.agent_loop import OpenAIResponsesToolLoop
from eu_climate_policy_rag.core.logging_utils import get_logger
from eu_climate_policy_rag.core.tools import (
    ToolExecutor,
    ToolRegistry as BaseToolRegistry,
)
from eu_climate_policy_rag.core.tools.adapters import OpenAIResponsesToolAdapter
from eu_climate_policy_rag.core.tooling import ToolRegistry as LegacyToolRegistry

LOGGER = get_logger(__name__)


class AbstractAgent(ABC):
    """Base class implementing the OpenAI agentic tool-call loop.

    Follows the pattern from homework-2: iteratively call the model,
    dispatch tool calls, and loop until the model stops requesting tools
    or ``max_turns`` is exceeded.

    Subclasses must implement ``run()`` and may override
    ``_execute_tool_call()`` for domain-specific dispatch logic.
    """

    def __init__(
        self,
        openai_client: OpenAI | None = None,
        model: str = "gpt-4o-mini",
        instructions: str = "",
        tools: LegacyToolRegistry | BaseToolRegistry | None = None,
        max_turns: int = 10,
    ) -> None:
        self.openai_client = openai_client or OpenAI()
        self.model = model
        self.instructions = instructions
        self.tools = tools or LegacyToolRegistry([])
        self.tool_registry = self._normalize_tool_registry(self.tools)
        self._tool_executor = ToolExecutor(self.tool_registry)
        self.tool_adapter = OpenAIResponsesToolAdapter(self.tool_registry)
        self.max_turns = max_turns

        # Log available tools for visibility
        schemas = self.tool_adapter.tools
        function_tools = [s["name"] for s in schemas if s.get("type") == "function"]
        builtin_tools = [s["type"] for s in schemas if s.get("type") != "function"]

        if function_tools:
            LOGGER.info("Custom function tools: %s", ", ".join(function_tools))
        if builtin_tools:
            LOGGER.info("Built-in tools: %s", ", ".join(builtin_tools))

    @staticmethod
    def _normalize_tool_registry(
        tools: LegacyToolRegistry | BaseToolRegistry,
    ) -> BaseToolRegistry:
        """Return the provider-neutral registry behind the agent boundary."""

        if isinstance(tools, LegacyToolRegistry):
            return tools.base_registry
        if isinstance(tools, BaseToolRegistry):
            return tools
        msg = "tools must be a core.tools.ToolRegistry or core.tooling.ToolRegistry."
        raise TypeError(msg)

    def _execute_tool_call(self, tool_call: Any) -> dict[str, Any]:
        """Dispatch a ``function_call`` message to the tool registry.

        Returns a ``function_call_output`` dict ready to be appended to
        the message history.  Override in subclasses for domain-specific
        dispatch logic (e.g. collecting sources, custom serialisation).
        """
        arguments = json.loads(tool_call.arguments)
        if self.tool_registry.get(tool_call.name) is None:
            LOGGER.error("Unknown tool requested: %s", tool_call.name)
        if isinstance(self.tools, LegacyToolRegistry):
            result = self.tools.run_sync(tool_call.name, arguments)
            output = json.dumps(result) if not isinstance(result, str) else result
            return {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": output,
            }
        result = self._tool_executor.run_sync(
            tool_call.name,
            arguments,
            call_id=tool_call.call_id,
        )
        return self.tool_adapter.to_function_call_output(result)

    def _create_response(self, message_history: list[Any]) -> Any:
        """Call the OpenAI Responses API with current tool schemas."""

        return self.openai_client.responses.create(
            model=self.model,
            input=message_history,
            tools=self.tool_adapter.tools,
        )

    def _build_loop(self, *, async_tools: bool = False) -> OpenAIResponsesToolLoop:
        """Build the reusable Responses tool-call loop for this agent."""

        execute_tool_call = (
            self._execute_tool_call_async if async_tools else self._execute_tool_call
        )
        return OpenAIResponsesToolLoop(
            create_response=self._create_response,
            execute_tool_call=execute_tool_call,
            max_turns=self.max_turns,
            on_message=self._on_message,
            on_builtin_tool_result=self._log_builtin_tool_result,
        )

    @staticmethod
    def _log_builtin_tool_result() -> None:
        LOGGER.info("Built-in tool(s) appear to have been used (detected web search citations)")

    def _run_loop(self, query: str) -> tuple[str, list[Any]]:
        """Run the agentic tool-call loop and return ``(final_answer, message_history)``.

        Each iteration sends the current message history to the model.
        Tool calls are dispatched via ``_execute_tool_call``, their results
        appended to the history, and the loop continues until the model
        returns a plain message or ``max_turns`` is exceeded.
        """
        return self._build_loop().run(query=query, instructions=self.instructions)

    async def _execute_tool_call_async(self, tool_call: Any) -> dict[str, Any]:
        """Async variant of ``_execute_tool_call``.

        Dispatches via ``tools.run()`` (awaitable) rather than
        ``tools.run_sync()``.  Override in subclasses as needed.
        """
        arguments = json.loads(tool_call.arguments)
        if self.tool_registry.get(tool_call.name) is None:
            LOGGER.error("Unknown tool requested: %s", tool_call.name)
        if isinstance(self.tools, LegacyToolRegistry):
            result = await self.tools.run(tool_call.name, arguments)
            output = json.dumps(result) if not isinstance(result, str) else result
            return {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": output,
            }
        result = await self._tool_executor.run(
            tool_call.name,
            arguments,
            call_id=tool_call.call_id,
        )
        return self.tool_adapter.to_function_call_output(result)

    async def _run_loop_async(self, query: str) -> tuple[str, list[Any]]:
        """Async version of ``_run_loop`` for agents with async tools.

        Uses ``_execute_tool_call_async`` to await async tool handlers.
        """
        return await self._build_loop(async_tools=True).run_async(
            query=query,
            instructions=self.instructions,
        )

    def _on_message(self, message: Any) -> None:
        """Hook called when the model returns a plain message during the loop.

        Override in subclasses to add logging or other side-effects.
        """

    @abstractmethod
    def run(self, query: str) -> Any:
        """Answer a user query.

        Subclasses define the return type and any post-processing of the
        result returned by ``_run_loop``.
        """
