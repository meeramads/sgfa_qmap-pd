"""
Model execution interface to prevent circular dependencies.

This module provides a single entry point for model execution, breaking the circular
dependency between core/run_analysis.py and analysis modules.

Usage:
    from core.model_interface import execute_sgfa_model

    samples = execute_sgfa_model(X_list, hypers, model_args, mcmc_config)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def execute_sgfa_model(
    X_list: List[np.ndarray],
    hypers: Dict[str, Any],
    model_args: Any,
    mcmc_config: Optional[Dict[str, Any]] = None,
    extra_fields: Optional[tuple] = None
) -> Dict[str, Any]:
    """
    Execute SGFA model with given configuration.

    This function encapsulates the model execution logic, preventing direct
    imports from core.run_analysis.models which creates circular dependencies.

    Args:
        X_list: List of data matrices (multi-view data)
        hypers: Hyperparameters dictionary
        model_args: Model arguments (Namespace or dict)
        mcmc_config: MCMC configuration (num_warmup, num_samples, num_chains, etc.)
        extra_fields: Extra fields to collect during MCMC (e.g., "potential_energy")

    Returns:
        Dictionary containing:
            - samples: MCMC samples
            - extra_fields: Additional collected fields
            - mcmc: MCMC object for further analysis

    Example:
        >>> hypers = {"Dm": [100, 200], "percW": 25.0}
        >>> args = argparse.Namespace(K=5, model="sparseGFA")
        >>> mcmc_config = {"num_warmup": 1000, "num_samples": 2000}
        >>> result = execute_sgfa_model(X_list, hypers, args, mcmc_config)
        >>> samples = result["samples"]
    """
    # Import here to avoid circular dependency
    from core.run_analysis import models

    # Import MCMC components
    import jax
    from numpyro.infer import MCMC, NUTS

    # Default MCMC config
    if mcmc_config is None:
        mcmc_config = {
            "num_warmup": 1000,
            "num_samples": 2000,
            "num_chains": 4,
        }

    # Setup MCMC
    kernel = NUTS(
        models,
        target_accept_prob=mcmc_config.get("target_accept_prob", 0.8),
        max_tree_depth=mcmc_config.get("max_tree_depth", 10),
    )

    rng_key = jax.random.PRNGKey(mcmc_config.get("random_seed", 42))

    mcmc = MCMC(
        kernel,
        num_warmup=mcmc_config["num_warmup"],
        num_samples=mcmc_config["num_samples"],
        num_chains=mcmc_config["num_chains"],
        progress_bar=mcmc_config.get("progress_bar", True),
        chain_method=mcmc_config.get("chain_method", "parallel"),
    )

    # Run inference
    logger.info(f"Running MCMC with {mcmc_config['num_samples']} samples, {mcmc_config['num_chains']} chains")

    extra_fields_tuple = extra_fields if extra_fields else ()
    mcmc.run(rng_key, X_list, hypers, model_args, extra_fields=extra_fields_tuple)

    # Extract results
    samples = mcmc.get_samples()
    collected_extra_fields = {}

    if extra_fields:
        extra_fields_dict = mcmc.get_extra_fields()
        for field_name in extra_fields:
            if field_name in extra_fields_dict:
                collected_extra_fields[field_name] = extra_fields_dict[field_name]

    return {
        "samples": samples,
        "extra_fields": collected_extra_fields,
        "mcmc": mcmc,
    }


def get_model_function():
    """
    Get the underlying model function.

    This is a convenience function for cases where direct access to the model
    is needed (e.g., for custom MCMC setups).

    Returns:
        The models function from core.run_analysis

    Note:
        Prefer using execute_sgfa_model() when possible to maintain abstraction.
    """
    from core.run_analysis import models
    return models
