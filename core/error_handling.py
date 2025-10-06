"""
Standard error handling patterns for SGFA qMAP-PD codebase.

Provides utilities and decorators for consistent error handling across modules.
"""

from __future__ import annotations

# Standard library imports
import functools
import logging
import traceback
from typing import Any, Callable, Dict, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar('T')


class SGFAError(Exception):
    """Base exception for SGFA-specific errors."""
    pass


class ConfigurationError(SGFAError):
    """Raised when configuration is invalid."""
    pass


class DataValidationError(SGFAError):
    """Raised when data validation fails."""
    pass


class ConvergenceError(SGFAError):
    """Raised when MCMC fails to converge."""
    pass


class ModelExecutionError(SGFAError):
    """Raised when model execution fails."""
    pass


def create_error_result(
    error: Exception,
    context: str = "",
    additional_fields: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized error result dictionary.

    Args:
        error: The exception that occurred
        context: Additional context about where/why error occurred
        additional_fields: Optional additional fields to include

    Returns:
        Standardized error dictionary with status, error message, and context
    """
    result = {
        "status": "failed",
        "error": str(error),
        "error_type": type(error).__name__,
    }

    if context:
        result["error_context"] = context

    if additional_fields:
        result.update(additional_fields)

    return result


def log_and_return_error(
    error: Exception,
    logger_instance: logging.Logger,
    context: str = "",
    log_level: str = "error",
    include_traceback: bool = False,
    additional_fields: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Log an error and return standardized error dictionary.

    Args:
        error: The exception that occurred
        logger_instance: Logger to use for logging
        context: Additional context about the error
        log_level: Logging level ('error', 'warning', 'info')
        include_traceback: Whether to include full traceback in log
        additional_fields: Optional additional fields for result dict

    Returns:
        Standardized error dictionary
    """
    # Format error message
    if context:
        message = f"{context}: {error}"
    else:
        message = str(error)

    # Log at appropriate level
    log_func = getattr(logger_instance, log_level.lower())

    if include_traceback:
        log_func(f"❌ {message}\n{traceback.format_exc()}")
    else:
        log_func(f"❌ {message}")

    # Return standardized error dict
    return create_error_result(error, context, additional_fields)


def safe_execute(
    func: Callable[..., T],
    *args,
    logger_instance: Optional[logging.Logger] = None,
    context: str = "",
    default_return: Any = None,
    **kwargs
) -> Union[T, Any]:
    """
    Safely execute a function with error handling.

    Args:
        func: Function to execute
        *args: Positional arguments for func
        logger_instance: Optional logger for error logging
        context: Context description for error messages
        default_return: Default value to return on error
        **kwargs: Keyword arguments for func

    Returns:
        Function result on success, default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger_instance:
            logger_instance.error(f"❌ {context or 'Operation'} failed: {e}")
        return default_return


def handle_errors(
    return_dict: bool = True,
    log_level: str = "error",
    include_traceback: bool = False,
    reraise: bool = False,
):
    """
    Decorator for standardized error handling in functions.

    Args:
        return_dict: If True, return error dict; if False, return None
        log_level: Logging level for errors
        include_traceback: Include full traceback in logs
        reraise: Re-raise exception after logging

    Returns:
        Decorated function with error handling

    Example:
        @handle_errors(return_dict=True)
        def my_function(x, logger=None):
            # function implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Step 1: Try to locate a logger instance
            # First check kwargs for explicit logger parameter
            logger_instance = kwargs.get('logger')

            # If not in kwargs, check if first arg is a class instance with logger
            if not logger_instance and len(args) > 0:
                if hasattr(args[0], 'logger'):
                    logger_instance = args[0].logger

            # Fall back to module logger if no logger found
            if not logger_instance:
                logger_instance = logger

            try:
                # Step 2: Execute the wrapped function
                return func(*args, **kwargs)

            except Exception as e:
                # Step 3: Handle the exception
                # Create context string identifying where error occurred
                context = f"{func.__module__}.{func.__name__}"

                # Log the error with appropriate level and detail
                if include_traceback:
                    # Full traceback for debugging
                    logger_instance.log(
                        getattr(logging, log_level.upper()),
                        f"❌ {context} failed: {e}\n{traceback.format_exc()}"
                    )
                else:
                    # Brief error message
                    logger_instance.log(
                        getattr(logging, log_level.upper()),
                        f"❌ {context} failed: {e}"
                    )

                # Step 4: Optionally re-raise the exception
                if reraise:
                    raise

                # Step 5: Return appropriate error result
                if return_dict:
                    return create_error_result(e, context)
                else:
                    return None

        return wrapper
    return decorator


def validate_required_keys(
    data: Dict[str, Any],
    required_keys: list[str],
    context: str = "Data validation"
) -> None:
    """
    Validate that required keys exist in dictionary.

    Args:
        data: Dictionary to validate
        required_keys: List of required keys
        context: Context for error message

    Raises:
        DataValidationError: If required keys are missing
    """
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise DataValidationError(
            f"{context}: Missing required keys: {missing}"
        )


def validate_convergence(
    convergence_info: Dict[str, Any],
    min_ess: float = 100.0,
    max_rhat: float = 1.1,
) -> None:
    """
    Validate MCMC convergence diagnostics.

    Args:
        convergence_info: Dictionary with 'ess' and 'rhat' values
        min_ess: Minimum effective sample size
        max_rhat: Maximum R-hat value

    Raises:
        ConvergenceError: If convergence criteria not met
    """
    ess = convergence_info.get("ess", float('inf'))
    rhat = convergence_info.get("rhat", 0.0)

    if ess < min_ess:
        raise ConvergenceError(
            f"Low effective sample size: {ess:.1f} < {min_ess}"
        )

    if rhat > max_rhat:
        raise ConvergenceError(
            f"High R-hat indicating poor convergence: {rhat:.3f} > {max_rhat}"
        )


# Standard error result patterns
ERROR_RESULT_TEMPLATE = {
    "status": "failed",
    "error": "",
    "error_type": "",
}

SUCCESS_RESULT_TEMPLATE = {
    "status": "completed",
}


def create_success_result(**kwargs) -> Dict[str, Any]:
    """
    Create a standardized success result dictionary.

    Args:
        **kwargs: Additional fields to include in result

    Returns:
        Dictionary with status='completed' and additional fields
    """
    result = SUCCESS_RESULT_TEMPLATE.copy()
    result.update(kwargs)
    return result
