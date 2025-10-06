"""Tests for core.error_handling module."""

import logging
from unittest.mock import Mock, patch

import pytest

from core.error_handling import (
    SGFAError,
    ConfigurationError,
    DataValidationError,
    ConvergenceError,
    ModelExecutionError,
    create_error_result,
    log_and_return_error,
    safe_execute,
    handle_errors,
    validate_required_keys,
    validate_convergence,
    create_success_result,
)


class TestExceptionHierarchy:
    """Test custom exception classes."""

    def test_sgfa_error_base(self):
        """Test base SGFAError exception."""
        error = SGFAError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_configuration_error(self):
        """Test ConfigurationError inherits from SGFAError."""
        error = ConfigurationError("Invalid config")
        assert isinstance(error, SGFAError)
        assert isinstance(error, Exception)

    def test_data_validation_error(self):
        """Test DataValidationError inherits from SGFAError."""
        error = DataValidationError("Invalid data")
        assert isinstance(error, SGFAError)

    def test_convergence_error(self):
        """Test ConvergenceError inherits from SGFAError."""
        error = ConvergenceError("Failed to converge")
        assert isinstance(error, SGFAError)

    def test_model_execution_error(self):
        """Test ModelExecutionError inherits from SGFAError."""
        error = ModelExecutionError("Model failed")
        assert isinstance(error, SGFAError)


class TestCreateErrorResult:
    """Test create_error_result function."""

    def test_basic_error_result(self):
        """Test creating basic error result."""
        error = ValueError("Test error")
        result = create_error_result(error)

        assert result["status"] == "failed"
        assert result["error"] == "Test error"
        assert result["error_type"] == "ValueError"

    def test_error_result_with_context(self):
        """Test error result with context."""
        error = ValueError("Test error")
        result = create_error_result(error, context="Loading data")

        assert result["status"] == "failed"
        assert result["error"] == "Test error"
        assert result["error_context"] == "Loading data"

    def test_error_result_with_additional_fields(self):
        """Test error result with additional fields."""
        error = ValueError("Test error")
        result = create_error_result(
            error,
            additional_fields={"data_loaded": False, "filepath": "/path/to/file"}
        )

        assert result["status"] == "failed"
        assert result["data_loaded"] is False
        assert result["filepath"] == "/path/to/file"


class TestLogAndReturnError:
    """Test log_and_return_error function."""

    def test_logs_error_message(self):
        """Test that error is logged."""
        logger = Mock(spec=logging.Logger)
        error = ValueError("Test error")

        result = log_and_return_error(error, logger, context="Test operation")

        logger.error.assert_called_once()
        assert "Test operation" in logger.error.call_args[0][0]
        assert "Test error" in logger.error.call_args[0][0]

    def test_returns_error_dict(self):
        """Test that error dict is returned."""
        logger = Mock(spec=logging.Logger)
        error = ValueError("Test error")

        result = log_and_return_error(error, logger)

        assert result["status"] == "failed"
        assert result["error"] == "Test error"

    def test_different_log_levels(self):
        """Test logging at different levels."""
        logger = Mock(spec=logging.Logger)
        error = ValueError("Test error")

        # Test warning level
        log_and_return_error(error, logger, log_level="warning")
        logger.warning.assert_called_once()

        # Test info level
        log_and_return_error(error, logger, log_level="info")
        logger.info.assert_called_once()

    def test_include_traceback(self):
        """Test including traceback in log."""
        logger = Mock(spec=logging.Logger)
        error = ValueError("Test error")

        result = log_and_return_error(
            error, logger, include_traceback=True
        )

        # Should have called error with traceback in message
        log_message = logger.error.call_args[0][0]
        assert "Traceback" in log_message or "Test error" in log_message


class TestSafeExecute:
    """Test safe_execute function."""

    def test_successful_execution(self):
        """Test successful function execution."""
        def add(a, b):
            return a + b

        result = safe_execute(add, 2, 3)
        assert result == 5

    def test_failed_execution_returns_default(self):
        """Test failed execution returns default."""
        def failing_func():
            raise ValueError("Error")

        result = safe_execute(failing_func, default_return="default")
        assert result == "default"

    def test_logs_error_when_logger_provided(self):
        """Test error is logged when logger provided."""
        logger = Mock(spec=logging.Logger)

        def failing_func():
            raise ValueError("Error")

        result = safe_execute(
            failing_func,
            logger_instance=logger,
            context="Test operation"
        )

        logger.error.assert_called_once()


class TestHandleErrorsDecorator:
    """Test handle_errors decorator."""

    def test_successful_function_execution(self):
        """Test decorator doesn't interfere with success."""
        @handle_errors()
        def successful_func(x):
            return x * 2

        result = successful_func(5)
        assert result == 10

    def test_returns_error_dict_on_failure(self):
        """Test decorator returns error dict on failure."""
        @handle_errors(return_dict=True)
        def failing_func():
            raise ValueError("Test error")

        result = failing_func()

        assert result["status"] == "failed"
        assert "ValueError" in result["error_type"]

    def test_returns_none_when_return_dict_false(self):
        """Test decorator returns None when configured."""
        @handle_errors(return_dict=False)
        def failing_func():
            raise ValueError("Test error")

        result = failing_func()
        assert result is None

    def test_reraise_option(self):
        """Test reraise option."""
        @handle_errors(reraise=True)
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_func()

    def test_finds_logger_in_kwargs(self):
        """Test decorator finds logger in kwargs."""
        logger = Mock(spec=logging.Logger)

        @handle_errors()
        def func_with_logger(x, logger=None):
            raise ValueError("Test error")

        func_with_logger(5, logger=logger)
        logger.error.assert_called()

    def test_finds_logger_in_self(self):
        """Test decorator finds logger in self."""
        logger = Mock(spec=logging.Logger)

        class TestClass:
            def __init__(self):
                self.logger = logger

            @handle_errors()
            def method(self):
                raise ValueError("Test error")

        obj = TestClass()
        obj.method()
        logger.error.assert_called()


class TestValidateRequiredKeys:
    """Test validate_required_keys function."""

    def test_valid_keys(self):
        """Test validation passes with all required keys."""
        data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        required = ["key1", "key2"]

        # Should not raise
        validate_required_keys(data, required)

    def test_missing_keys_raises_error(self):
        """Test validation raises error for missing keys."""
        data = {"key1": "value1"}
        required = ["key1", "key2", "key3"]

        with pytest.raises(DataValidationError) as exc_info:
            validate_required_keys(data, required)

        assert "key2" in str(exc_info.value)
        assert "key3" in str(exc_info.value)

    def test_custom_context_in_error(self):
        """Test custom context appears in error message."""
        data = {}
        required = ["key1"]

        with pytest.raises(DataValidationError) as exc_info:
            validate_required_keys(data, required, context="MCMC results")

        assert "MCMC results" in str(exc_info.value)


class TestValidateConvergence:
    """Test validate_convergence function."""

    def test_good_convergence(self):
        """Test validation passes for good convergence."""
        convergence_info = {"ess": 200.0, "rhat": 1.01}

        # Should not raise
        validate_convergence(convergence_info)

    def test_low_ess_raises_error(self):
        """Test low ESS raises ConvergenceError."""
        convergence_info = {"ess": 50.0, "rhat": 1.01}

        with pytest.raises(ConvergenceError) as exc_info:
            validate_convergence(convergence_info, min_ess=100.0)

        assert "effective sample size" in str(exc_info.value).lower()

    def test_high_rhat_raises_error(self):
        """Test high R-hat raises ConvergenceError."""
        convergence_info = {"ess": 200.0, "rhat": 1.5}

        with pytest.raises(ConvergenceError) as exc_info:
            validate_convergence(convergence_info, max_rhat=1.1)

        assert "rhat" in str(exc_info.value).lower() or "r-hat" in str(exc_info.value).lower()

    def test_custom_thresholds(self):
        """Test custom convergence thresholds."""
        convergence_info = {"ess": 75.0, "rhat": 1.15}

        # Should pass with relaxed thresholds
        validate_convergence(
            convergence_info,
            min_ess=50.0,
            max_rhat=1.2
        )


class TestCreateSuccessResult:
    """Test create_success_result function."""

    def test_basic_success_result(self):
        """Test creating basic success result."""
        result = create_success_result()

        assert result["status"] == "completed"

    def test_success_result_with_fields(self):
        """Test success result with additional fields."""
        result = create_success_result(
            execution_time=1.5,
            num_samples=1000,
            convergence=True
        )

        assert result["status"] == "completed"
        assert result["execution_time"] == 1.5
        assert result["num_samples"] == 1000
        assert result["convergence"] is True


class TestIntegration:
    """Integration tests for error handling patterns."""

    def test_complete_workflow(self):
        """Test complete error handling workflow."""
        logger = Mock(spec=logging.Logger)

        @handle_errors(return_dict=True)
        def analysis_function(data, logger=None):
            # Validate inputs
            validate_required_keys(data, ["X", "y"])

            # Simulate processing
            if data.get("should_fail"):
                raise ModelExecutionError("Processing failed")

            return create_success_result(result="success")

        # Test success case
        result = analysis_function({"X": [1, 2], "y": [3, 4]}, logger=logger)
        assert result["status"] == "completed"

        # Test failure case
        result = analysis_function(
            {"X": [1, 2], "y": [3, 4], "should_fail": True},
            logger=logger
        )
        assert result["status"] == "failed"
        logger.error.assert_called()

    def test_nested_error_handling(self):
        """Test nested error handling scenarios."""
        logger = Mock(spec=logging.Logger)

        def inner_function():
            raise ValueError("Inner error")

        @handle_errors(return_dict=True)
        def outer_function(logger=None):
            try:
                inner_function()
            except ValueError as e:
                return log_and_return_error(e, logger, "Inner function")

        result = outer_function(logger=logger)
        assert result["status"] == "failed"
        assert "Inner function" in result.get("error_context", "")
