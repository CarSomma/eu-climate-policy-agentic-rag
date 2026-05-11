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
            response = self.openai_client.responses.create(
                model=self.model,
                input=message_history,
                tools=self.tools.schemas,
            )
            message_history.extend(response.output)

            has_tool_call = False
            for message in response.output:
                if message.type == "function_call":
                    has_tool_call = True
                    LOGGER.info("Executing tool: %s", message.name)
                    tool_output = self._execute_tool_call(message)
                    message_history.append(tool_output)
                elif message.type == "message":
                    self._on_message(message)
                    final_answer = message.content[0].text

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
            response = self.openai_client.responses.create(
                model=self.model,
                input=message_history,
                tools=self.tools.schemas,
            )
            message_history.extend(response.output)

            has_tool_call = False
            for message in response.output:
                if message.type == "function_call":
                    has_tool_call = True
                    LOGGER.info("Executing tool: %s", message.name)
                    tool_output = await self._execute_tool_call_async(message)
                    message_history.append(tool_output)
                elif message.type == "message":
                    self._on_message(message)
                    final_answer = message.content[0].text

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
