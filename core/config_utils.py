"""Configuration utilities for safe and consistent config access."""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

from .config_schema import ConfigurationValidator, ConfigValidationError

logger = logging.getLogger(__name__)


def safe_get(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely get nested configuration values.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    *keys : str
        Nested keys to access (e.g., 'data', 'data_dir')
    default : Any, optional
        Default value if key not found

    Returns
    -------
    Any
        Configuration value or default

    Examples
    --------
    >>> config = {'data': {'data_dir': '/path/to/data'}}
    >>> safe_get(config, 'data', 'data_dir', default='./data')
    '/path/to/data'
    >>> safe_get(config, 'missing', 'key', default='fallback')
    'fallback'
    """
    current = config
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def safe_get_path(config: Dict[str, Any], *keys: str, default: str = ".") -> Path:
    """
    Safely get a path from configuration, returning Path object.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    *keys : str
        Nested keys to access
    default : str, optional
        Default path if key not found

    Returns
    -------
    Path
        Path object
    """
    path_str = safe_get(config, *keys, default=default)
    return Path(path_str)


def get_data_dir(config: Dict[str, Any]) -> Path:
    """Get data directory from config with safe fallback."""
    return safe_get_path(config, "data", "data_dir", default="./data")


def get_output_dir(config: Dict[str, Any]) -> Path:
    """Get output directory from config with safe fallback."""
    return safe_get_path(config, "experiments", "base_output_dir", default="./results")


def get_checkpoint_dir(config: Dict[str, Any]) -> Path:
    """Get checkpoint directory from config with safe fallback."""
    return safe_get_path(
        config, "monitoring", "checkpoint_dir", default="./results/checkpoints"
    )


def validate_required_config(
    config: Dict[str, Any], required_keys: List[List[str]]
) -> None:
    """
    Validate that required configuration keys exist.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    required_keys : List[List[str]]
        List of key paths that must exist

    Raises
    ------
    KeyError
        If any required key is missing
    """
    missing_keys = []

    for key_path in required_keys:
        if safe_get(config, *key_path) is None:
            missing_keys.append(".".join(key_path))

    if missing_keys:
        raise KeyError(
            f"Missing required configuration keys: {', '.join(missing_keys)}"
        )


def update_config_safely(
    config: Dict[str, Any], updates: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Safely update configuration with nested structure.

    Parameters
    ----------
    config : Dict[str, Any]
        Original configuration
    updates : Dict[str, Any]
        Updates to apply

    Returns
    -------
    Dict[str, Any]
        Updated configuration
    """
    import copy

    updated_config = copy.deepcopy(config)

    def _deep_update(base_dict: Dict, update_dict: Dict) -> None:
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                _deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    _deep_update(updated_config, updates)
    return updated_config


def get_experiment_config(
    config: Dict[str, Any], experiment_name: str
) -> Dict[str, Any]:
    """
    Get experiment-specific configuration with fallbacks.

    Parameters
    ----------
    config : Dict[str, Any]
        Main configuration
    experiment_name : str
        Name of experiment

    Returns
    -------
    Dict[str, Any]
        Experiment configuration with safe defaults
    """
    # Get experiment-specific config
    exp_config = safe_get(config, experiment_name, default={})

    # Apply common defaults
    defaults = {"n_repetitions": 1, "save_intermediate": True, "generate_plots": True}

    return {**defaults, **exp_config}


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Ensure required directories exist.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    """
    # Create output directory
    output_dir = get_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured output directory exists: {output_dir}")

    # Create checkpoint directory if specified
    checkpoint_dir = get_checkpoint_dir(config)
    if checkpoint_dir != output_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured checkpoint directory exists: {checkpoint_dir}")


# Common configuration access patterns
class ConfigAccessor:
    """Helper class for safe configuration access."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get(self, key: str, default=None):
        """Get configuration value with default fallback (dict-like interface)."""
        return self.config.get(key, default)

    @property
    def dataset(self) -> str:
        """Get dataset name."""
        return self.config.get("dataset", "qmap_pd")

    @property
    def K(self) -> int:
        """Get number of factors."""
        return self.config.get("K", 5)

    @property
    def percW(self) -> float:
        """Get percentage of nonzero loadings."""
        return self.config.get("percW", 25.0)

    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return get_data_dir(self.config)

    @property
    def output_dir(self) -> Path:
        """Get output directory."""
        return get_output_dir(self.config)

    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory."""
        return get_checkpoint_dir(self.config)

    def get_experiment_setting(
        self, experiment: str, setting: str, default: Any = None
    ) -> Any:
        """Get experiment-specific setting."""
        return safe_get(self.config, experiment, setting, default=default)

    def has_shared_data(self) -> bool:
        """Check if shared data is available."""
        return safe_get(self.config, "_shared_data") is not None

    def get_shared_data(self) -> Dict[str, Any]:
        """Get shared data configuration."""
        return safe_get(self.config, "_shared_data", default={})


def validate_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fix configuration using schema validation.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to validate

    Returns
    -------
    Dict[str, Any]
        Validated and potentially fixed configuration

    Raises
    ------
    ConfigValidationError
        If configuration is invalid and cannot be fixed
    """
    try:
        return ConfigurationValidator.validate_and_fix_configuration(config)
    except ConfigValidationError:
        # Try merging with defaults first
        logger.warning(
            "Configuration validation failed, attempting to merge with defaults"
        )
        merged_config = ConfigurationValidator.merge_with_defaults(config)
        return ConfigurationValidator.validate_and_fix_configuration(merged_config)


def get_default_configuration() -> Dict[str, Any]:
    """Get default configuration."""
    return ConfigurationValidator.get_default_configuration()


def check_configuration_warnings(config: Dict[str, Any]) -> List[str]:
    """
    Check configuration for potential issues and return warnings.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    List[str]
        List of warning messages
    """
    warnings = []

    # Check for performance warnings
    model_config = safe_get(config, "model", default={})

    # Large number of samples warning
    num_samples = safe_get(model_config, "num_samples", default=1000)
    if num_samples > 10000:
        warnings.append(
            f"Large num_samples ({num_samples}) may take a long time to complete"
        )

    # Many factors warning
    K = safe_get(model_config, "K", default=5)
    if K > 20:
        warnings.append(f"Large number of factors (K={K}) may lead to overfitting")

    # GPU availability warning
    system_config = safe_get(config, "system", default={})
    if safe_get(system_config, "use_gpu", default=True):
        try:
            import jax

            devices = jax.devices()
            gpu_devices = [d for d in devices if d.platform in ["gpu", "cuda"]]
            if len(gpu_devices) == 0:
                warnings.append(
                    "GPU requested but no GPU devices available - will use CPU (slower)"
                )
        except ImportError:
            warnings.append("GPU requested but JAX not available")

    return warnings


class ConfigHelper:
    """Utility class for standardized configuration handling across the codebase."""

    @staticmethod
    def to_dict(config_or_dict) -> Dict[str, Any]:
        """
        Convert any config object to a dictionary in a standardized way.

        Parameters
        ----------
        config_or_dict : Any
            Config object (with .to_dict() method), dictionary, or object with __dict__

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the config
        """
        if isinstance(config_or_dict, dict):
            return config_or_dict
        elif hasattr(config_or_dict, "to_dict") and callable(
            getattr(config_or_dict, "to_dict")
        ):
            return config_or_dict.to_dict()
        elif hasattr(config_or_dict, "__dict__"):
            return config_or_dict.__dict__
        else:
            # If it's a simple value, wrap it
            return {"value": config_or_dict}

    @staticmethod
    def get_output_dir_safe(config_or_dict) -> Path:
        """
        Get output directory from any config format safely.

        Parameters
        ----------
        config_or_dict : Any
            Config object or dictionary

        Returns
        -------
        Path
            Output directory path
        """
        config_dict = ConfigHelper.to_dict(config_or_dict)
        return get_output_dir(config_dict)

    @staticmethod
    def get_data_dir_safe(config_or_dict) -> Path:
        """
        Get data directory from any config format safely.

        Parameters
        ----------
        config_or_dict : Any
            Config object or dictionary

        Returns
        -------
        Path
            Data directory path
        """
        config_dict = ConfigHelper.to_dict(config_or_dict)
        return get_data_dir(config_dict)

    @staticmethod
    def safe_get_from_config(config_or_dict, *keys: str, default: Any = None) -> Any:
        """
        Safely get nested values from any config format.

        Parameters
        ----------
        config_or_dict : Any
            Config object or dictionary
        *keys : str
            Nested keys to access
        default : Any
            Default value if keys not found

        Returns
        -------
        Any
            Retrieved value or default
        """
        config_dict = ConfigHelper.to_dict(config_or_dict)
        return safe_get(config_dict, *keys, default=default)


@contextmanager
def JAXMemoryManager():
    """
    Context manager for automatic JAX memory cleanup.

    Automatically clears JAX device memory and caches on exit,
    preventing memory buildup in long-running experiments.

    Examples
    --------
    >>> with JAXMemoryManager():
    ...     # JAX computations here
    ...     result = jax.numpy.array([1, 2, 3])
    # Memory automatically cleaned up
    """
    try:
        yield
    finally:
        try:
            import jax

            # Clear JAX device memory and compilation cache
            jax.clear_caches()
            logger.debug("JAX memory and caches cleared")
        except ImportError:
            # JAX not available, nothing to clean up
            pass
        except Exception as e:
            logger.warning(f"Failed to clear JAX memory: {e}")


@contextmanager
def PlotManager():
    """
    Context manager for automatic matplotlib cleanup.

    Automatically closes all matplotlib figures on exit,
    preventing memory buildup from unclosed plots.

    Examples
    --------
    >>> with PlotManager():
    ...     import matplotlib.pyplot as plt
    ...     plt.figure()
    ...     plt.plot([1, 2, 3])
    # All figures automatically closed
    """
    try:
        yield
    finally:
        try:
            import matplotlib.pyplot as plt

            # Close all matplotlib figures to free memory
            plt.close("all")
            logger.debug("All matplotlib figures closed")
        except ImportError:
            # Matplotlib not available, nothing to clean up
            pass
        except Exception as e:
            logger.warning(f"Failed to close matplotlib figures: {e}")
