"""Structured exceptions for the tool framework."""

from dataclasses import dataclass, field


class ToolFrameworkError(Exception):
    """Base class for tool framework errors."""

    retryable = False

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, object] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        if retryable is not None:
            self.retryable = retryable


class ToolValidationError(ToolFrameworkError):
    """Raised when tool arguments fail validation."""


class ToolExecutionError(ToolFrameworkError):
    """Raised when a tool handler fails during execution."""


class UnknownToolError(ToolFrameworkError):
    """Raised when a requested local function tool is not registered."""


class SchemaGenerationError(ToolFrameworkError):
    """Raised when a schema cannot be compiled for a provider."""


@dataclass(frozen=True)
class ToolErrorPayload:
    """Serializable error payload for model-visible tool failures."""

    type: str
    message: str
    details: dict[str, object] = field(default_factory=dict)
    retryable: bool = False

    @classmethod
    def from_exception(cls, error: BaseException) -> "ToolErrorPayload":
        """Build a payload from an exception."""

        if isinstance(error, ToolFrameworkError):
            return cls(
                type=error.__class__.__name__,
                message=error.message,
                details=error.details,
                retryable=error.retryable,
            )
        return cls(
            type=error.__class__.__name__,
            message=str(error),
            retryable=False,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary."""

        return {
            "type": self.type,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }
