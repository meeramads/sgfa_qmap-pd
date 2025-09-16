"""Validation utilities for consistent parameter and result validation."""

import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_parameters(**validation_rules):
    """
    Decorator for parameter validation with detailed rules.

    Parameters
    ----------
    **validation_rules
        Validation rules as keyword arguments where keys are parameter names
        and values are validation functions or tuples of (validator, error_message)

    Examples
    --------
    >>> @validate_parameters(
    ...     n_samples=lambda x: x > 0,
    ...     K=(lambda x: 1 <= x <= 50, "K must be between 1 and 50")
    ... )
    ... def my_function(n_samples, K):
    ...     pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature to map args to parameter names
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate each parameter
            for param_name, rule in validation_rules.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]

                    # Handle different rule formats
                    if callable(rule):
                        validator = rule
                        error_msg = f"Validation failed for parameter '{param_name}'"
                    elif isinstance(rule, tuple) and len(rule) == 2:
                        validator, error_msg = rule
                    else:
                        raise ValueError(f"Invalid validation rule for {param_name}")

                    # Apply validation
                    try:
                        if not validator(value):
                            raise ValueError(error_msg)
                    except Exception as e:
                        raise ValueError(
                            f"Parameter validation error for '{param_name}': {e}"
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_data_types(**type_rules):
    """
    Decorator for data type validation.

    Parameters
    ----------
    **type_rules
        Type rules as keyword arguments where keys are parameter names
        and values are expected types or tuples of types

    Examples
    --------
    >>> @validate_data_types(
    ...     data=np.ndarray,
    ...     config=(dict, type(None)),
    ...     n_samples=int
    ... )
    ... def my_function(data, config, n_samples):
    ...     pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for param_name, expected_type in type_rules.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{param_name}' must be of type {expected_type}, "
                            f"got {
                                type(value)}"
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator


class ParameterValidator:
    """Utility class for parameter validation with common patterns."""

    @staticmethod
    def validate_positive(value: Union[int, float], name: str = "parameter") -> None:
        """Validate that a value is positive."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    @staticmethod
    def validate_range(
        value: Union[int, float],
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        name: str = "parameter",
    ) -> None:
        """Validate that a value is within a specified range."""
        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {value}")

    @staticmethod
    def validate_array_shape(
        array: np.ndarray,
        expected_shape: Optional[tuple] = None,
        min_dims: Optional[int] = None,
        max_dims: Optional[int] = None,
        name: str = "array",
    ) -> None:
        """Validate numpy array shape and dimensions."""
        if expected_shape is not None and array.shape != expected_shape:
            raise ValueError(
                f"{name} shape must be {expected_shape}, got {array.shape}"
            )

        if min_dims is not None and array.ndim < min_dims:
            raise ValueError(
                f"{name} must have at least {min_dims} dimensions, got {array.ndim}"
            )

        if max_dims is not None and array.ndim > max_dims:
            raise ValueError(
                f"{name} must have at most {max_dims} dimensions, got {array.ndim}"
            )

    @staticmethod
    def validate_list_type(lst: List, expected_type: Type, name: str = "list") -> None:
        """Validate that all elements in a list are of expected type."""
        if not isinstance(lst, list):
            raise TypeError(f"{name} must be a list, got {type(lst)}")

        for i, item in enumerate(lst):
            if not isinstance(item, expected_type):
                raise TypeError(
                    f"{name}[{i}] must be of type {expected_type.__name__}, "
                    f"got {type(item).__name__}"
                )

    @staticmethod
    def validate_dict_keys(
        dct: Dict,
        required_keys: List[str],
        optional_keys: Optional[List[str]] = None,
        name: str = "dictionary",
    ) -> None:
        """Validate dictionary has required keys and no unexpected keys."""
        if not isinstance(dct, dict):
            raise TypeError(f"{name} must be a dictionary, got {type(dct)}")

        # Check required keys
        missing_keys = [key for key in required_keys if key not in dct]
        if missing_keys:
            raise ValueError(f"{name} missing required keys: {missing_keys}")

        # Check for unexpected keys
        if optional_keys is not None:
            allowed_keys = set(required_keys + optional_keys)
            unexpected_keys = [key for key in dct.keys() if key not in allowed_keys]
            if unexpected_keys:
                raise ValueError(f"{name} has unexpected keys: {unexpected_keys}")


class ResultValidator:
    """Utility class for validating experiment results."""

    @staticmethod
    def validate_experiment_result(
        result: Dict, required_fields: List[str] = None
    ) -> None:
        """
        Validate that an experiment result has required structure.

        Parameters
        ----------
        result : Dict
            Experiment result dictionary
        required_fields : List[str], optional
            List of required fields
        """
        if not isinstance(result, dict):
            raise TypeError(
                f"Experiment result must be a dictionary, got {type(result)}"
            )

        # Default required fields
        if required_fields is None:
            required_fields = ["success", "experiment_name"]

        # Check required fields
        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            raise ValueError(
                f"Experiment result missing required fields: {missing_fields}"
            )

        # Validate success field
        if "success" in result:
            if not isinstance(result["success"], bool):
                raise TypeError("'success' field must be boolean")

            # If not successful, should have error information
            if not result["success"] and "error" not in result:
                logger.warning("Failed experiment result should include 'error' field")

    @staticmethod
    def validate_mcmc_samples(
        samples: Dict, expected_keys: Optional[List[str]] = None
    ) -> None:
        """
        Validate MCMC samples dictionary.

        Parameters
        ----------
        samples : Dict
            MCMC samples dictionary
        expected_keys : List[str], optional
            Expected parameter names in samples
        """
        if not isinstance(samples, dict):
            raise TypeError(f"MCMC samples must be a dictionary, got {type(samples)}")

        if not samples:
            raise ValueError("MCMC samples dictionary is empty")

        # Validate all values are arrays
        for key, value in samples.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"Sample '{key}' must be numpy array, got {type(value)}"
                )

        # Check for expected keys if provided
        if expected_keys is not None:
            missing_keys = [key for key in expected_keys if key not in samples]
            if missing_keys:
                logger.warning(f"MCMC samples missing expected keys: {missing_keys}")

    @staticmethod
    def validate_data_matrices(
        X_list: List[np.ndarray], min_samples: int = 10
    ) -> None:
        """
        Validate list of data matrices for experiments.

        Parameters
        ----------
        X_list : List[np.ndarray]
            List of data matrices
        min_samples : int, optional
            Minimum number of samples required
        """
        if not isinstance(X_list, list):
            raise TypeError(f"Data must be a list of arrays, got {type(X_list)}")

        if len(X_list) == 0:
            raise ValueError("Data list cannot be empty")

        for i, X in enumerate(X_list):
            if not isinstance(X, np.ndarray):
                raise TypeError(f"Data[{i}] must be numpy array, got {type(X)}")

            if X.ndim != 2:
                raise ValueError(f"Data[{i}] must be 2D array, got {X.ndim}D")

            if X.shape[0] < min_samples:
                raise ValueError(
                    f"Data[{i}] must have at least {min_samples} samples, got {
                        X.shape[0]}"
                )

            if np.any(np.isnan(X)):
                raise ValueError(f"Data[{i}] contains NaN values")

            if np.any(np.isinf(X)):
                raise ValueError(f"Data[{i}] contains infinite values")


def validate_config_structure(config: Dict, schema: Dict) -> List[str]:
    """
    Validate configuration structure against a schema.

    Parameters
    ----------
    config : Dict
        Configuration dictionary to validate
    schema : Dict
        Schema dictionary with expected structure

    Returns
    -------
    List[str]
        List of validation warnings (empty if all valid)
    """
    warnings = []

    def _validate_nested(cfg: Dict, sch: Dict, path: str = "") -> None:
        for key, expected_type in sch.items():
            current_path = f"{path}.{key}" if path else key

            if key not in cfg:
                warnings.append(f"Missing config key: {current_path}")
                continue

            value = cfg[key]

            if isinstance(expected_type, dict):
                # Nested dictionary
                if not isinstance(value, dict):
                    warnings.append(
                        f"Config key {current_path} should be dict, got {type(value)}"
                    )
                else:
                    _validate_nested(value, expected_type, current_path)
            elif isinstance(expected_type, type):
                # Type validation
                if not isinstance(value, expected_type):
                    warnings.append(
                        f"Config key {current_path} should be {
                            expected_type.__name__}, "
                        f"got {
                            type(value).__name__}"
                    )
            elif callable(expected_type):
                # Custom validator function
                try:
                    if not expected_type(value):
                        warnings.append(f"Config key {current_path} failed validation")
                except Exception as e:
                    warnings.append(f"Config key {current_path} validation error: {e}")

    _validate_nested(config, schema)
    return warnings


# Common validation schemas for experiments
EXPERIMENT_CONFIG_SCHEMA = {
    "model": {
        "K": lambda x: isinstance(x, int) and 1 <= x <= 50,
        "num_samples": lambda x: isinstance(x, int) and x > 0,
        "num_warmup": lambda x: isinstance(x, int) and x > 0,
    },
    "data": {
        "data_dir": str,
    },
    "experiments": {
        "base_output_dir": str,
    },
}

SGFA_RESULT_SCHEMA = {
    "success": bool,
    "experiment_name": str,
    "samples": dict,
    "model": object,
}
