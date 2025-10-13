# models/factory.py
"""
Factory for creating model instances.

This module provides a unified interface for creating different GFA model types
with proper configuration, validation, and extensibility.

Note: This project primarily uses Sparse GFA (sparseGFA/sparse_gfa).
Standard GFA models are registered for completeness but are NOT actively
used due to memory constraints with high-dimensional neuroimaging data.

LCA (Latent Class Analysis) has been removed - it's a clustering method, not
factor analysis, and should not be compared to GFA methods.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from .base import BaseGFAModel
from .sparse_gfa import SparseGFAModel
from .standard_gfa import StandardGFAModel
from .variants.neuroimaging_gfa import NeuroimagingGFAModel

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory for creating GFA model instances.

    This factory supports:
    - Model registration and discovery
    - Configuration validation
    - Model metadata and requirements
    - Extensibility for new model types

    Examples:
        >>> # Basic model creation
        >>> model = ModelFactory.create_model("sparseGFA", config, hypers)

        >>> # Register custom model
        >>> ModelFactory.register_model("myGFA", MyGFAModel)

        >>> # Get model info
        >>> info = ModelFactory.get_model_info("sparseGFA")
    """

    # Model registry with metadata
    _models: Dict[str, Dict[str, Any]] = {
        "sparseGFA": {
            "class": SparseGFAModel,
            "description": "Sparse Group Factor Analysis with automatic relevance determination",
            "required_params": ["K"],
            "optional_params": ["percW", "tau_param"],
        },
        "sparse_gfa": {  # Alias for compatibility
            "class": SparseGFAModel,
            "description": "Sparse Group Factor Analysis (snake_case alias)",
            "required_params": ["K"],
            "optional_params": ["percW", "tau_param"],
        },
        "GFA": {
            "class": StandardGFAModel,
            "description": "Standard Group Factor Analysis (NOT USED - memory intensive)",
            "required_params": ["K"],
            "optional_params": [],
            "warnings": ["Not used in this project due to memory constraints"],
        },
        "standard_gfa": {  # Alias for compatibility
            "class": StandardGFAModel,
            "description": "Standard Group Factor Analysis (NOT USED - memory intensive)",
            "required_params": ["K"],
            "optional_params": [],
            "warnings": ["Not used in this project due to memory constraints"],
        },
        "neuroGFA": {
            "class": NeuroimagingGFAModel,
            "description": "Neuroimaging-specific GFA with spatial priors",
            "required_params": ["K", "spatial_info"],
            "optional_params": ["spatial_weight"],
        },
        # NOTE: LCA (Latent Class Analysis) removed - it's a clustering/mixture model,
        # not a factor analysis method. LCA finds discrete latent classes (categorical)
        # whereas GFA finds continuous latent factors. They solve different problems.
    }

    @classmethod
    def create_model(
        cls,
        model_type: str,
        config: Any = None,
        hypers: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseGFAModel:
        """
        Create a model instance with configuration.

        Args:
            model_type: Type of model (e.g., 'sparseGFA', 'GFA', 'neuroGFA')
            config: Configuration object or dict
            hypers: Hyperparameters dictionary
            **kwargs: Additional model-specific arguments

        Returns:
            BaseGFAModel instance

        Raises:
            ValueError: If model type is unknown
            TypeError: If required parameters are missing

        Examples:
            >>> config = {"K": 10, "percW": 25.0}
            >>> hypers = {"Dm": [100, 200]}
            >>> model = ModelFactory.create_model("sparseGFA", config, hypers)
        """
        # Validate model type
        if model_type not in cls._models:
            available = cls.list_models()
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                f"Available models: {available}"
            )

        model_info = cls._models[model_type]
        model_class = model_info["class"]

        # Handle config-only call (for test compatibility)
        if hypers is None and config is not None and isinstance(config, dict):
            # Check if config contains hyperparameters
            if "Dm" in config or all(k in config for k in ["K", "num_samples"]):
                hypers = config
                logger.debug(f"Using config as hypers for {model_type}")

        # Log warnings if specified
        if "warnings" in model_info:
            for warning in model_info["warnings"]:
                logger.warning(f"{model_type}: {warning}")

        # Model-specific instantiation
        if model_type in ["neuroGFA"]:
            # Neuroimaging model requires spatial info
            spatial_info = kwargs.get("spatial_info")
            if spatial_info is None and config:
                spatial_info = getattr(config, "spatial_info", None)

            if spatial_info is None:
                logger.warning(
                    "No spatial_info provided for neuroimaging model. "
                    "Model may not leverage spatial structure."
                )

            return model_class(config, hypers, spatial_info=spatial_info)

        else:
            # Standard model instantiation
            return model_class(config, hypers)

    @classmethod
    def register_model(
        cls,
        name: str,
        model_class: Type[BaseGFAModel],
        description: str = "",
        required_params: Optional[List[str]] = None,
        optional_params: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ):
        """
        Register a new model type.

        Args:
            name: Model type identifier
            model_class: Model class (must inherit from BaseGFAModel)
            description: Human-readable description
            required_params: List of required parameter names
            optional_params: List of optional parameter names
            warnings: List of warnings to log when creating this model

        Raises:
            TypeError: If model_class doesn't inherit from BaseGFAModel

        Examples:
            >>> ModelFactory.register_model(
            ...     "customGFA",
            ...     CustomGFAModel,
            ...     description="Custom GFA implementation",
            ...     required_params=["K", "alpha"],
            ... )
        """
        if not issubclass(model_class, BaseGFAModel):
            raise TypeError(
                f"Model class must inherit from BaseGFAModel, "
                f"got {model_class.__name__}"
            )

        cls._models[name] = {
            "class": model_class,
            "description": description,
            "required_params": required_params or [],
            "optional_params": optional_params or [],
        }

        if warnings:
            cls._models[name]["warnings"] = warnings

        logger.info(f"Registered new model type: '{name}'")

    @classmethod
    def list_models(cls) -> List[str]:
        """
        List available model types.

        Returns:
            List of model type names

        Examples:
            >>> models = ModelFactory.list_models()
            >>> print(models)
            ['sparseGFA', 'GFA', 'neuroGFA']
        """
        return sorted(cls._models.keys())

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Alias for list_models() for test compatibility."""
        return cls.list_models()

    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """
        Get metadata for a model type.

        Args:
            model_type: Model type identifier

        Returns:
            Dictionary with model metadata (description, params, etc.)

        Raises:
            ValueError: If model type is unknown

        Examples:
            >>> info = ModelFactory.get_model_info("sparseGFA")
            >>> print(info["description"])
            'Sparse Group Factor Analysis with automatic relevance determination'
        """
        if model_type not in cls._models:
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                f"Available: {cls.list_models()}"
            )

        model_info = cls._models[model_type].copy()
        # Remove class reference from public info
        model_info.pop("class", None)
        return model_info

    @classmethod
    def get_required_parameters(cls, model_type: str) -> List[str]:
        """
        Get required parameters for a model type.

        Args:
            model_type: Model type identifier

        Returns:
            List of required parameter names

        Examples:
            >>> params = ModelFactory.get_required_parameters("sparseGFA")
            >>> print(params)
            ['K']
        """
        info = cls.get_model_info(model_type)
        return info.get("required_params", [])

    @classmethod
    def get_default_parameters(cls, model_type: str) -> Dict[str, Any]:
        """
        Get default parameters for a model type.

        Args:
            model_type: Model type identifier

        Returns:
            Dictionary of default parameters

        Note:
            This returns sensible defaults for MCMC and model configuration.
            Actual model classes may have additional defaults.

        Examples:
            >>> defaults = ModelFactory.get_default_parameters("sparseGFA")
            >>> print(defaults["num_samples"])
            2000
        """
        # Base defaults for all models
        defaults = {
            "num_samples": 2000,
            "num_warmup": 1000,
            "num_chains": 4,
            "target_accept_prob": 0.8,
        }

        # Model-specific defaults
        if model_type in ["sparseGFA", "sparse_gfa"]:
            defaults.update({
                "percW": 25.0,
                "tau_param": 0.1,
            })

        return defaults

    @classmethod
    def validate_config(
        cls,
        model_type: str,
        config: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """
        Validate configuration for a model type.

        Args:
            model_type: Model type identifier
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_errors)

        Examples:
            >>> config = {"K": 10, "percW": 25.0}
            >>> valid, errors = ModelFactory.validate_config("sparseGFA", config)
            >>> if not valid:
            ...     print(f"Errors: {errors}")
        """
        errors = []

        # Check model type exists
        if model_type not in cls._models:
            errors.append(f"Unknown model type: '{model_type}'")
            return False, errors

        # Check required parameters
        required = cls.get_required_parameters(model_type)
        for param in required:
            if param not in config:
                errors.append(f"Missing required parameter: '{param}'")

        # Validate K if present
        if "K" in config:
            K = config["K"]
            if not isinstance(K, int) or K <= 0:
                errors.append(f"K must be a positive integer, got {K}")

        return len(errors) == 0, errors
