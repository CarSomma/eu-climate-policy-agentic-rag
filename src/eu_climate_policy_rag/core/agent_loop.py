"""Reusable OpenAI Responses tool-call loop."""

from collections.abc import Awaitable, Callable
from inspect import isawaitable
from typing import Any

from eu_climate_policy_rag.core.logging_utils import get_logger

LOGGER = get_logger(__name__)

ResponseFactory = Callable[[list[Any]], Any]
ToolCallHandler = Callable[[Any], dict[str, Any] | Awaitable[dict[str, Any]]]
MessageHandler = Callable[[Any], None]
BuiltinToolHandler = Callable[[], None]


class OpenAIResponsesToolLoop:
    """Run the core Responses API function-calling loop."""

    def __init__(
        self,
        *,
        create_response: ResponseFactory,
        execute_tool_call: ToolCallHandler,
        max_turns: int,
        on_message: MessageHandler | None = None,
        on_builtin_tool_result: BuiltinToolHandler | None = None,
    ) -> None:
        self.create_response = create_response
        self.execute_tool_call = execute_tool_call
        self.max_turns = max_turns
        self.on_message = on_message or (lambda message: None)
        self.on_builtin_tool_result = on_builtin_tool_result

    def run(self, *, query: str, instructions: str) -> tuple[str, list[Any]]:
        """Run sync Responses tool-call turns until a final message or turn limit."""

        message_history = self._initial_history(query, instructions)
        final_answer = ""

        for turn in range(1, self.max_turns + 1):
            LOGGER.info("Agent turn %d", turn)
            response = self.create_response(message_history)
            has_tool_call, turn_answer, has_builtin_tool_result = (
                self._append_response_output(response, message_history)
            )
            final_answer = turn_answer or final_answer

            if (
                has_builtin_tool_result
                and not has_tool_call
                and self.on_builtin_tool_result is not None
            ):
                self.on_builtin_tool_result()

            if not has_tool_call:
                break
        else:
            final_answer = "Agent reached max turns without a final answer."

        return final_answer, message_history

    async def run_async(self, *, query: str, instructions: str) -> tuple[str, list[Any]]:
        """Run async Responses tool-call turns until a final message or turn limit."""

        message_history = self._initial_history(query, instructions)
        final_answer = ""

        for turn in range(1, self.max_turns + 1):
            LOGGER.info("Agent turn %d", turn)
            response = self.create_response(message_history)
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
                tool_output = self.execute_tool_call(message)
                if isawaitable(tool_output):
                    msg = "Async tool call handler used in sync Responses loop."
                    raise TypeError(msg)
                message_history.append(tool_output)
            elif message.type == "message":
                self.on_message(message)
                final_answer = message.content[0].text
                has_builtin_tool_result = (
                    has_builtin_tool_result or self._looks_like_builtin_tool_result(message)
                )

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
                tool_output = self.execute_tool_call(message)
                if isawaitable(tool_output):
                    tool_output = await tool_output
                message_history.append(tool_output)
            elif message.type == "message":
                self.on_message(message)
                final_answer = message.content[0].text

        return has_tool_call, final_answer

    @staticmethod
    def _initial_history(query: str, instructions: str) -> list[Any]:
        return [
            {"role": "system", "content": instructions},
            {"role": "user", "content": query},
        ]

    @staticmethod
    def _looks_like_builtin_tool_result(message: Any) -> bool:
        return (
            message.type == "message"
            and hasattr(message.content[0], "text")
            and "?utm_source=openai" in message.content[0].text
        )
