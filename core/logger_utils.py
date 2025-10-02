"""Logger utilities and protocols for consistent logging across the codebase."""

import logging
import sys
from typing import Protocol, Optional


class LoggerProtocol(Protocol):
    """Protocol for logger objects.

    Any object implementing these methods can be used as a logger.
    """

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        ...

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        ...

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        ...

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        ...


class ConsoleLogger:
    """Simple console logger fallback when no logger is configured."""

    def __init__(self, name: str = "console"):
        """Initialize console logger."""
        self.name = name

    def info(self, msg: str, *args, **kwargs) -> None:
        """Print info message to stdout."""
        print(f"INFO [{self.name}]: {msg}", *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Print warning message to stderr."""
        print(f"WARNING [{self.name}]: {msg}", *args, **kwargs, file=sys.stderr)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Print error message to stderr."""
        print(f"ERROR [{self.name}]: {msg}", *args, **kwargs, file=sys.stderr)

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Print debug message to stdout."""
        print(f"DEBUG [{self.name}]: {msg}", *args, **kwargs)


def get_logger(obj, name: Optional[str] = None) -> LoggerProtocol:
    """Get logger from object or create fallback.

    Replaces the common pattern:
        if hasattr(self, 'logger') and self.logger:
            self.logger.info("message")
        else:
            print("message")

    With:
        logger = get_logger(self)
        logger.info("message")

    Parameters
    ----------
    obj : Any
        Object that may have a logger attribute
    name : str, optional
        Name for fallback logger if object has no logger

    Returns
    -------
    LoggerProtocol
        Logger from object or ConsoleLogger fallback

    Examples
    --------
    >>> class MyClass:
    ...     def __init__(self):
    ...         self.logger = None
    ...     def do_something(self):
    ...         logger = get_logger(self, name='MyClass')
    ...         logger.info("Doing something")
    """
    # Check for logger attribute
    if hasattr(obj, 'logger') and obj.logger is not None:
        return obj.logger

    # Try to get class name for fallback logger
    if name is None:
        name = obj.__class__.__name__ if hasattr(obj, '__class__') else 'unknown'

    # Return console logger as fallback
    return ConsoleLogger(name)


def ensure_logger(logger: Optional[LoggerProtocol], name: str = "default") -> LoggerProtocol:
    """Ensure a logger exists, creating fallback if needed.

    Parameters
    ----------
    logger : LoggerProtocol or None
        Existing logger or None
    name : str
        Name for fallback logger

    Returns
    -------
    LoggerProtocol
        The provided logger or a ConsoleLogger fallback
    """
    if logger is not None:
        return logger
    return ConsoleLogger(name)


def create_module_logger(module_name: str) -> logging.Logger:
    """Create a standard Python logger for a module.

    Parameters
    ----------
    module_name : str
        Module name (usually __name__)

    Returns
    -------
    logging.Logger
        Configured logger
    """
    return logging.getLogger(module_name)
