"""Tests for core.logger_utils module."""

import logging
import sys
from io import StringIO
from unittest.mock import Mock

import pytest

from core.logger_utils import (
    ConsoleLogger,
    LoggerProtocol,
    create_module_logger,
    ensure_logger,
    get_logger,
)


class TestLoggerProtocol:
    """Test LoggerProtocol."""

    def test_standard_logger_implements_protocol(self):
        """Test that standard Python logger implements protocol."""
        logger = logging.getLogger("test")

        # Should have all protocol methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")

    def test_console_logger_implements_protocol(self):
        """Test that ConsoleLogger implements protocol."""
        logger = ConsoleLogger()

        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")


class TestConsoleLogger:
    """Test ConsoleLogger class."""

    def test_initialization(self):
        """Test ConsoleLogger initialization."""
        logger = ConsoleLogger("my_logger")
        assert logger.name == "my_logger"

    def test_default_name(self):
        """Test default logger name."""
        logger = ConsoleLogger()
        assert logger.name == "console"

    def test_info_prints_to_stdout(self, capsys):
        """Test that info messages go to stdout."""
        logger = ConsoleLogger("test")
        logger.info("Test message")

        captured = capsys.readouterr()
        assert "INFO [test]: Test message" in captured.out

    def test_warning_prints_to_stderr(self, capsys):
        """Test that warning messages go to stderr."""
        logger = ConsoleLogger("test")
        logger.warning("Warning message")

        captured = capsys.readouterr()
        assert "WARNING [test]: Warning message" in captured.err

    def test_error_prints_to_stderr(self, capsys):
        """Test that error messages go to stderr."""
        logger = ConsoleLogger("test")
        logger.error("Error message")

        captured = capsys.readouterr()
        assert "ERROR [test]: Error message" in captured.err

    def test_debug_prints_to_stdout(self, capsys):
        """Test that debug messages go to stdout."""
        logger = ConsoleLogger("test")
        logger.debug("Debug message")

        captured = capsys.readouterr()
        assert "DEBUG [test]: Debug message" in captured.out

    def test_message_with_args(self, capsys):
        """Test logging with additional arguments."""
        logger = ConsoleLogger("test")
        logger.info("Message", "arg1", "arg2")

        captured = capsys.readouterr()
        # Should include the message and args
        assert "INFO [test]: Message" in captured.out


class TestGetLogger:
    """Test get_logger function."""

    def test_returns_object_logger_if_present(self):
        """Test that object's logger is returned if present."""
        mock_logger = Mock()

        class MyClass:
            def __init__(self):
                self.logger = mock_logger

        obj = MyClass()
        result = get_logger(obj)

        assert result is mock_logger

    def test_returns_console_logger_if_no_logger(self):
        """Test that ConsoleLogger is returned if object has no logger."""
        class MyClass:
            pass

        obj = MyClass()
        result = get_logger(obj)

        assert isinstance(result, ConsoleLogger)
        assert result.name == "MyClass"

    def test_returns_console_logger_if_logger_is_none(self):
        """Test that ConsoleLogger is returned if logger is None."""
        class MyClass:
            def __init__(self):
                self.logger = None

        obj = MyClass()
        result = get_logger(obj)

        assert isinstance(result, ConsoleLogger)

    def test_uses_custom_name(self):
        """Test using custom name for fallback logger."""
        class MyClass:
            def __init__(self):
                self.logger = None

        obj = MyClass()
        result = get_logger(obj, name="CustomName")

        assert isinstance(result, ConsoleLogger)
        assert result.name == "CustomName"

    def test_uses_class_name_as_fallback(self):
        """Test that class name is used for fallback logger."""
        class MySpecificClass:
            pass

        obj = MySpecificClass()
        result = get_logger(obj)

        assert isinstance(result, ConsoleLogger)
        assert result.name == "MySpecificClass"

    def test_with_object_without_class(self):
        """Test with object that has no __class__ attribute."""
        obj = object()
        result = get_logger(obj, name="provided_name")

        assert isinstance(result, ConsoleLogger)
        assert result.name == "provided_name"


class TestEnsureLogger:
    """Test ensure_logger function."""

    def test_returns_provided_logger(self):
        """Test that provided logger is returned."""
        mock_logger = Mock()
        result = ensure_logger(mock_logger)

        assert result is mock_logger

    def test_returns_console_logger_when_none(self):
        """Test that ConsoleLogger is returned when logger is None."""
        result = ensure_logger(None)

        assert isinstance(result, ConsoleLogger)
        assert result.name == "default"

    def test_uses_custom_name(self):
        """Test using custom name for fallback logger."""
        result = ensure_logger(None, name="custom")

        assert isinstance(result, ConsoleLogger)
        assert result.name == "custom"

    def test_preserves_logger_type(self):
        """Test that logger type is preserved."""
        python_logger = logging.getLogger("test")
        result = ensure_logger(python_logger)

        assert result is python_logger
        assert isinstance(result, logging.Logger)


class TestCreateModuleLogger:
    """Test create_module_logger function."""

    def test_creates_logger(self):
        """Test that logger is created."""
        logger = create_module_logger("test.module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_returns_same_logger_for_same_name(self):
        """Test that same logger is returned for same name."""
        logger1 = create_module_logger("my.module")
        logger2 = create_module_logger("my.module")

        assert logger1 is logger2

    def test_different_names_return_different_loggers(self):
        """Test that different names return different loggers."""
        logger1 = create_module_logger("module1")
        logger2 = create_module_logger("module2")

        assert logger1 is not logger2
        assert logger1.name == "module1"
        assert logger2.name == "module2"

    def test_works_with_name_attribute(self):
        """Test using __name__ attribute pattern."""
        module_name = __name__
        logger = create_module_logger(module_name)

        assert logger.name == module_name


class TestLoggerIntegration:
    """Integration tests for logger utilities."""

    def test_get_logger_with_real_logger(self):
        """Test get_logger with real Python logger."""
        class MyClass:
            def __init__(self):
                self.logger = logging.getLogger("test")

        obj = MyClass()
        result = get_logger(obj)

        assert isinstance(result, logging.Logger)
        assert result.name == "test"

    def test_console_logger_as_fallback_pattern(self):
        """Test typical usage pattern with ConsoleLogger fallback."""
        class ServiceClass:
            def __init__(self, logger=None):
                self.logger = logger

            def do_work(self):
                logger = get_logger(self, name="ServiceClass")
                return logger

        # Without logger
        service1 = ServiceClass()
        logger1 = service1.do_work()
        assert isinstance(logger1, ConsoleLogger)

        # With logger
        mock_logger = Mock()
        service2 = ServiceClass(logger=mock_logger)
        logger2 = service2.do_work()
        assert logger2 is mock_logger

    def test_ensure_logger_pattern(self):
        """Test typical usage pattern with ensure_logger."""
        def process_data(logger=None):
            logger = ensure_logger(logger, name="data_processor")
            logger.info("Processing data")
            return logger

        # Without logger
        logger1 = process_data()
        assert isinstance(logger1, ConsoleLogger)

        # With logger
        mock_logger = Mock()
        logger2 = process_data(mock_logger)
        assert logger2 is mock_logger


class TestConsoleLoggerFormatting:
    """Test ConsoleLogger message formatting."""

    def test_info_format(self, capsys):
        """Test info message format."""
        logger = ConsoleLogger("formatter_test")
        logger.info("Test info message")

        captured = capsys.readouterr()
        assert "INFO [formatter_test]: Test info message" in captured.out

    def test_warning_format(self, capsys):
        """Test warning message format."""
        logger = ConsoleLogger("formatter_test")
        logger.warning("Test warning message")

        captured = capsys.readouterr()
        assert "WARNING [formatter_test]: Test warning message" in captured.err

    def test_error_format(self, capsys):
        """Test error message format."""
        logger = ConsoleLogger("formatter_test")
        logger.error("Test error message")

        captured = capsys.readouterr()
        assert "ERROR [formatter_test]: Test error message" in captured.err

    def test_debug_format(self, capsys):
        """Test debug message format."""
        logger = ConsoleLogger("formatter_test")
        logger.debug("Test debug message")

        captured = capsys.readouterr()
        assert "DEBUG [formatter_test]: Test debug message" in captured.out


class TestLoggerUsagePatterns:
    """Test common logger usage patterns."""

    def test_class_with_optional_logger(self):
        """Test class accepting optional logger parameter."""
        class DataProcessor:
            def __init__(self, logger=None):
                self.logger = ensure_logger(logger, name="DataProcessor")

            def process(self):
                self.logger.info("Processing")
                return "done"

        # Without logger
        processor1 = DataProcessor()
        assert isinstance(processor1.logger, ConsoleLogger)
        result1 = processor1.process()
        assert result1 == "done"

        # With logger
        mock_logger = Mock()
        processor2 = DataProcessor(logger=mock_logger)
        assert processor2.logger is mock_logger
        processor2.process()
        mock_logger.info.assert_called_with("Processing")

    def test_function_with_logger_parameter(self):
        """Test function with logger parameter."""
        def analyze_data(data, logger=None):
            logger = ensure_logger(logger, name="analyzer")
            logger.info(f"Analyzing {len(data)} items")
            return len(data)

        # Without logger
        result1 = analyze_data([1, 2, 3])
        assert result1 == 3

        # With logger
        mock_logger = Mock()
        result2 = analyze_data([1, 2, 3, 4], logger=mock_logger)
        assert result2 == 4
        mock_logger.info.assert_called_once()

    def test_nested_function_logger_access(self):
        """Test accessing logger in nested functions."""
        class OuterClass:
            def __init__(self):
                self.logger = None

            def outer_method(self):
                logger = get_logger(self, name="OuterClass")

                def inner_function():
                    # Can use the same logger
                    logger.info("Inner function")
                    return "inner"

                logger.info("Outer method")
                return inner_function()

        obj = OuterClass()
        result = obj.outer_method()
        assert result == "inner"
