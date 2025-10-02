"""Utility functions to bridge experiments with analysis pipeline components."""

import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data_manager import DataManager
from .model_runner import ModelRunner


logger = logging.getLogger(__name__)


def create_analysis_components(
    config_dict: Dict,
    results_dir: Optional[str] = None
) -> Tuple[DataManager, ModelRunner]:
    """
    Create DataManager and ModelRunner instances from experiment configuration.

    This bridges the gap between experiment configurations and analysis components.

    Args:
        config_dict: Experiment configuration dictionary
        results_dir: Directory for results (optional)

    Returns:
        Tuple of (DataManager, ModelRunner) instances
    """
    # Convert config dict to args-like object
    args = argparse.Namespace(**config_dict)

    # Ensure required attributes exist with defaults
    if not hasattr(args, 'dataset'):
        args.dataset = 'synthetic'
    if not hasattr(args, 'num_sources'):
        args.num_sources = config_dict.get('num_sources', 3)
    if not hasattr(args, 'K'):
        args.K = config_dict.get('K', 5)
    if not hasattr(args, 'model'):
        args.model = config_dict.get('model', 'sparseGFA')

    # Create components with proper configs
    from analysis.data_manager import DataManagerConfig
    from analysis.model_runner import ModelRunnerConfig

    dm_config = DataManagerConfig.from_object(args)
    data_manager = DataManager(dm_config)

    mr_config = ModelRunnerConfig.from_object(args)
    model_runner = ModelRunner(mr_config, results_dir)

    return data_manager, model_runner


def run_sgfa_with_components(
    X_list: List[np.ndarray],
    hypers: Dict,
    args_dict: Dict,
    results_dir: Optional[str] = None
) -> Dict:
    """
    Run SGFA analysis using the analysis pipeline components.

    This provides a unified interface for experiments to use the analysis pipeline.

    Args:
        X_list: List of data matrices
        hypers: Hyperparameters dictionary
        args_dict: Arguments dictionary
        results_dir: Results directory (optional)

    Returns:
        Analysis results dictionary
    """
    logger.info("Running SGFA analysis with pipeline components")

    # Create components
    data_manager, model_runner = create_analysis_components(args_dict, results_dir)

    # Prepare data if needed
    if not isinstance(hypers.get('Dm'), list):
        hypers = data_manager.prepare_for_analysis({'X_list': X_list})[1]

    # Run analysis
    try:
        results = model_runner.run_standard_analysis(X_list, hypers, {'X_list': X_list})
        logger.info("✅ SGFA analysis completed successfully")
        return results
    except Exception as e:
        logger.error(f"❌ SGFA analysis failed: {e}")
        # Fallback to direct model import (backward compatibility)
        logger.warning("Falling back to direct model execution")
        return _run_sgfa_fallback(X_list, hypers, args_dict)


def _run_sgfa_fallback(X_list: List[np.ndarray], hypers: Dict, args_dict: Dict) -> Dict:
    """Fallback to direct model execution for backward compatibility."""
    try:
        from core.run_analysis import models
        import argparse

        # Convert args dict to namespace
        args = argparse.Namespace(**args_dict)

        # Set defaults for required fields
        if not hasattr(args, 'model'):
            args.model = 'sparseGFA'
        if not hasattr(args, 'K'):
            args.K = hypers.get('K', 5)
        if not hasattr(args, 'num_sources'):
            args.num_sources = len(X_list)
        if not hasattr(args, 'reghsZ'):
            args.reghsZ = True

        # Run model directly
        results = models(X_list, hypers, args)
        logger.info("✅ Fallback SGFA analysis completed")
        return results

    except Exception as e:
        logger.error(f"❌ Fallback SGFA analysis also failed: {e}")
        return {
            "error": str(e),
            "convergence": False,
            "execution_time": float('inf')
        }


def prepare_experiment_data(
    config_dict: Dict,
    X_list: Optional[List[np.ndarray]] = None
) -> Tuple[List[np.ndarray], Dict]:
    """
    Prepare data for experiments using DataManager.

    Args:
        config_dict: Experiment configuration
        X_list: Optional existing data (if None, will load/generate)

    Returns:
        Tuple of (X_list, hypers)
    """
    data_manager, _ = create_analysis_components(config_dict)

    if X_list is None:
        # Load data using DataManager
        data = data_manager.load_data()
        X_list, hypers = data_manager.prepare_for_analysis(data)
    else:
        # Prepare existing data
        X_list, hypers = data_manager.prepare_for_analysis({'X_list': X_list})

    return X_list, hypers


# Convenience functions for common experiment patterns
def quick_sgfa_run(
    X_list: List[np.ndarray],
    K: int = 5,
    percW: float = 25.0,
    model: str = "sparseGFA",
    **kwargs
) -> Dict:
    """
    Quick SGFA run with minimal configuration.

    Args:
        X_list: Data matrices
        K: Number of factors
        percW: Percentage of within-group variance
        model: Model type
        **kwargs: Additional arguments

    Returns:
        Analysis results
    """
    hypers = {
        "percW": percW,
        "Dm": [X.shape[1] for X in X_list],
        "a_sigma": 1,
        "b_sigma": 1,
        "nu_local": 1,
        "nu_global": 1,
        "slab_scale": 2,
        "slab_df": 4,
    }

    args_dict = {
        "K": K,
        "model": model,
        "num_sources": len(X_list),
        "reghsZ": True,
        **kwargs
    }

    return run_sgfa_with_components(X_list, hypers, args_dict)