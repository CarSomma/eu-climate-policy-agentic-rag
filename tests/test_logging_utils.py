import logging
import unittest

from eu_climate_policy_rag.core.logging_utils import ColoredFormatter, ColoredLogger


class LoggingUtilsTests(unittest.TestCase):
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

        self.assertIn("hello", formatted)
        self.assertEqual(record.levelname, "INFO")

    def test_colored_logger_returns_configured_logger(self) -> None:
        logger = ColoredLogger("test.colored.logger", use_color=False).get()

        self.assertEqual(logger.name, "test.colored.logger")
        self.assertFalse(logger.propagate)
        self.assertTrue(logger.handlers)


if __name__ == "__main__":
    unittest.main()
