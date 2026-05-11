import logging

from eu_climate_policy_rag.core.logging_utils import ColoredFormatter, ColoredLogger


class LoggingUtilsTests:
    def test_colored_formatter_preserves_record_levelname(self) -> None:
        formatter = ColoredFormatter(use_color=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        assert "hello" in formatted
        assert record.levelname == "INFO"

    def test_colored_logger_returns_configured_logger(self) -> None:
        logger = ColoredLogger("test.colored.logger", use_color=False).get()

        assert logger.name == "test.colored.logger"
        assert not logger.propagate
        assert logger.handlers

