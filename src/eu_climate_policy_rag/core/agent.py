"""Abstract base class for OpenAI agentic tool-call loops."""

import json
from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI

from eu_climate_policy_rag.core.logging_utils import get_logger
from eu_climate_policy_rag.core.tooling import ToolRegistry

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
        tools: ToolRegistry | None = None,
        max_turns: int = 10,
    ) -> None:
        self.openai_client = openai_client or OpenAI()
        self.model = model
        self.instructions = instructions
        self.tools = tools or ToolRegistry([])
        self.max_turns = max_turns

        # Log available tools for visibility
        function_tools = [s["name"] for s in self.tools.schemas if s.get("type") == "function"]
        builtin_tools = [s["type"] for s in self.tools.schemas if s.get("type") != "function"]

        if function_tools:
            LOGGER.info("Custom function tools: %s", ", ".join(function_tools))
        if builtin_tools:
            LOGGER.info("Built-in tools: %s", ", ".join(builtin_tools))

    def _execute_tool_call(self, tool_call: Any) -> dict[str, Any]:
        """Dispatch a ``function_call`` message to the tool registry.

        Returns a ``function_call_output`` dict ready to be appended to
        the message history.  Override in subclasses for domain-specific
        dispatch logic (e.g. collecting sources, custom serialisation).
        """
        arguments = json.loads(tool_call.arguments)
        if self.tools.get(tool_call.name) is None:
            LOGGER.error("Unknown tool requested: %s", tool_call.name)
        result = self.tools.run_sync(tool_call.name, arguments)
        output = json.dumps(result) if not isinstance(result, str) else result
        return {
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": output,
        }

    def _create_response(self, message_history: list[Any]) -> Any:
        """Call the OpenAI Responses API with current tool schemas."""

        return self.openai_client.responses.create(
            model=self.model,
            input=message_history,
            tools=self.tools.schemas,
        )

    def _append_response_output(
        self,
        response: Any,
        message_history: list[Any],
    ) -> tuple[bool, str, bool]:
        """Append one Responses output batch and execute requested function calls."""

        message_history.extend(response.output)
        has_tool_call = False
        final_answer = ""
        has_builtin_tool_result = False

        for message in response.output:
            if message.type == "function_call":
                has_tool_call = True
                LOGGER.info("Executing tool: %s", message.name)
                tool_output = self._execute_tool_call(message)
                message_history.append(tool_output)
            elif message.type == "message":
                self._on_message(message)
                final_answer = message.content[0].text

                # Check if web search was likely used (heuristic: citations with utm_source=openai)
                if hasattr(message.content[0], "text") and "?utm_source=openai" in message.content[0].text:
                    has_builtin_tool_result = True

        return has_tool_call, final_answer, has_builtin_tool_result

    async def _append_response_output_async(
        self,
        response: Any,
        message_history: list[Any],
    ) -> tuple[bool, str]:
        """Async variant of ``_append_response_output``."""

        message_history.extend(response.output)
        has_tool_call = False
        final_answer = ""

        for message in response.output:
            if message.type == "function_call":
                has_tool_call = True
                LOGGER.info("Executing tool: %s", message.name)
                tool_output = await self._execute_tool_call_async(message)
                message_history.append(tool_output)
            elif message.type == "message":
                self._on_message(message)
                final_answer = message.content[0].text

        return has_tool_call, final_answer

    def _run_loop(self, query: str) -> tuple[str, list[Any]]:
        """Run the agentic tool-call loop and return ``(final_answer, message_history)``.

        Each iteration sends the current message history to the model.
        Tool calls are dispatched via ``_execute_tool_call``, their results
        appended to the history, and the loop continues until the model
        returns a plain message or ``max_turns`` is exceeded.
        """
        message_history: list[Any] = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": query},
        ]
        final_answer = ""

        for turn in range(1, self.max_turns + 1):
            LOGGER.info("Agent turn %d", turn)
            response = self._create_response(message_history)
            has_tool_call, turn_answer, has_builtin_tool_result = (
                self._append_response_output(response, message_history)
            )
            final_answer = turn_answer or final_answer

            if has_builtin_tool_result and not has_tool_call:
                LOGGER.info("Built-in tool(s) appear to have been used (detected web search citations)")

            if not has_tool_call:
                break
        else:
            final_answer = "Agent reached max turns without a final answer."

        return final_answer, message_history

    async def _execute_tool_call_async(self, tool_call: Any) -> dict[str, Any]:
        """Async variant of ``_execute_tool_call``.

        Dispatches via ``tools.run()`` (awaitable) rather than
        ``tools.run_sync()``.  Override in subclasses as needed.
        """
        arguments = json.loads(tool_call.arguments)
        if self.tools.get(tool_call.name) is None:
            LOGGER.error("Unknown tool requested: %s", tool_call.name)
        result = await self.tools.run(tool_call.name, arguments)
        output = json.dumps(result) if not isinstance(result, str) else result
        return {
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": output,
        }

    async def _run_loop_async(self, query: str) -> tuple[str, list[Any]]:
        """Async version of ``_run_loop`` for agents with async tools.

        Uses ``_execute_tool_call_async`` to await async tool handlers.
        """
        message_history: list[Any] = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": query},
        ]
        final_answer = ""

        for turn in range(1, self.max_turns + 1):
            LOGGER.info("Agent turn %d", turn)
            response = self._create_response(message_history)
            has_tool_call, turn_answer = await self._append_response_output_async(
                response,
                message_history,
            )
            final_answer = turn_answer or final_answer

            if not has_tool_call:
                break
        else:
            final_answer = "Agent reached max turns without a final answer."

        return final_answer, message_history

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
