"""Utility functions to bridge experiments with analysis pipeline components."""

from __future__ import annotations

# Standard library imports
import argparse
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np

# Local imports
from .data_manager import DataManager
from .model_runner import ModelRunner


logger = logging.getLogger(__name__)


class AnalysisMode(Enum):
    """Analysis execution modes."""
    STRUCTURED = "structured"  # Full framework with DataManager/ModelRunner
    FALLBACK = "fallback"      # Fallback to direct core.run_analysis
    BASIC = "basic"            # Basic analysis without framework


@dataclass
class AnalysisComponents:
    """Container for analysis pipeline components.

    Attributes:
        mode: Execution mode (STRUCTURED, FALLBACK, or BASIC)
        data_manager: DataManager instance (None if mode != STRUCTURED)
        model_runner: ModelRunner instance (None if mode != STRUCTURED)
        config_manager: ConfigManager instance (None if not available)
        error: Error message if initialization failed
        metadata: Additional metadata about the framework setup
    """
    mode: AnalysisMode
    data_manager: Optional[DataManager] = None
    model_runner: Optional[ModelRunner] = None
    config_manager: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_structured(self) -> bool:
        """Check if using structured framework."""
        return self.mode == AnalysisMode.STRUCTURED

    @property
    def success(self) -> bool:
        """Check if initialization was successful."""
        return self.error is None

    def load_and_prepare_data(self) -> Tuple[List[np.ndarray], Dict]:
        """Load and prepare data using DataManager."""
        from core.error_handling import log_and_return_error

        # Check if DataManager is available
        if not self.data_manager:
            return [], {"data_loaded": False, "error": "DataManager not available"}

        try:
            # Load raw data using DataManager
            data = self.data_manager.load_data()

            # Prepare data for analysis (compute hyperparameters, apply preprocessing)
            X_list, hypers = self.data_manager.prepare_for_analysis(data)

            # Return data with comprehensive metadata for debugging/logging
            return X_list, {
                "data_loaded": True,
                "loader": "DataManager",
                "hyperparameters": hypers,
                "preprocessing_applied": "preprocessing" in data,
                "data_characteristics": {
                    "num_views": len(X_list),
                    "view_shapes": [X.shape for X in X_list],
                    "total_features": sum(X.shape[1] for X in X_list),
                    "num_subjects": X_list[0].shape[0],
                },
            }
        except Exception as e:
            # Use standardized error handling
            error_result = log_and_return_error(
                e, logger, "Structured data loading",
                additional_fields={"data_loaded": False}
            )
            return [], error_result

    def run_analysis(self, X_list: List[np.ndarray]) -> Dict:
        """Run analysis using ModelRunner."""
        from core.error_handling import create_error_result, log_and_return_error

        # Validate that required components are available
        if not self.model_runner or not self.data_manager:
            return create_error_result(
                ValueError("Components not available"),
                "Analysis execution"
            )

        try:
            # Prepare data and compute hyperparameters
            X_prepared, hypers = self.data_manager.prepare_for_analysis({"X_list": X_list})

            # Execute MCMC analysis using ModelRunner
            results = self.model_runner.run_standard_analysis(X_prepared, hypers, {"X_list": X_prepared})

            # Return standardized success result
            return {
                "status": "completed",
                "analysis_type": "structured_mcmc",
                "runs": results,
                "num_runs": len(results),
                "hyperparameters": hypers,
                "structured_framework": True,
            }
        except Exception as e:
            # Use standardized error handling
            return log_and_return_error(
                e, logger, "Structured MCMC analysis",
                additional_fields={"analysis_type": "structured_mcmc"}
            )


def _create_analysis_args_from_config(config: Dict) -> argparse.Namespace:
    """Create analysis args from configuration dictionary."""
    hyperparam_config = config.get("hyperparameter_optimization", {})
    training_config = config.get("training", {})

    return argparse.Namespace(
        # Basic parameters
        model="sparseGFA",
        dataset="qmap_pd",
        data_dir=config.get("data", {}).get("data_dir", "./qMAP-PD_data"),
        # Model parameters
        K=hyperparam_config.get("fallback_K", 10),
        percW=hyperparam_config.get("fallback_percW", 33),
        reghsZ=True,
        # MCMC parameters
        num_samples=training_config.get("mcmc_config", {}).get("num_samples", 2000),
        num_warmup=training_config.get("mcmc_config", {}).get("num_warmup", 1000),
        num_chains=training_config.get("mcmc_config", {}).get("num_chains", 4),
        num_runs=training_config.get("mcmc_config", {}).get("num_runs", 1),
        # Data parameters
        clinical_rel="data_clinical/pd_motor_gfa_data.tsv",
        volumes_rel="volume_matrices",
        id_col="sid",
        roi_views=True,
        noise=0,
        seed=42,
        num_sources=4,
        # Analysis control
        run_cv=config.get("cross_validation", {}).get("enabled", False),
        cv_only=False,
        neuroimaging_cv=config.get("cross_validation", {}).get("neuroimaging_cv", False),
        nested_cv=config.get("cross_validation", {}).get("nested_cv", False),
        # Preprocessing
        enable_preprocessing=config.get("preprocessing", {}).get("enabled", True),
        enable_spatial_processing=config.get("preprocessing", {}).get("spatial_processing", False),
        preprocessing_params=config.get("preprocessing", {}).get("params", {}),
    )


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
    from core.config_utils import dict_to_namespace

    # Convert config dict to args with defaults
    defaults = {
        'dataset': 'synthetic',
        'num_sources': 3,
        'K': 5,
        'model': 'sparseGFA',
    }
    args = dict_to_namespace(config_dict, defaults)

    # Create components with proper configs
    from analysis.data_manager import DataManagerConfig
    from analysis.model_runner import ModelRunnerConfig

    dm_config = DataManagerConfig.from_object(args)
    data_manager = DataManager(dm_config)

    mr_config = ModelRunnerConfig.from_object(args)
    model_runner = ModelRunner(mr_config, results_dir)

    return data_manager, model_runner


def integrate_analysis_with_pipeline(
    config: Dict,
    data_dir: str,
    X_list: Optional[List[np.ndarray]] = None,
    output_dir: Optional[str] = None,
) -> Tuple[Optional[DataManager], Optional[ModelRunner], Dict]:
    """
    Main integration function for analysis framework in the pipeline.

    Args:
        config: Configuration dictionary
        data_dir: Data directory path
        X_list: Optional preprocessed data
        output_dir: Output directory for analysis results

    Returns:
        Tuple of (data_manager, model_runner, comprehensive_analysis_info)
    """
    try:
        logger.info("🚀 === ANALYSIS FRAMEWORK PIPELINE INTEGRATION ===")

        # Import analysis components
        from analysis.config_manager import ConfigManager

        # Create args object from config
        analysis_args = _create_analysis_args_from_config(config)

        # Initialize config manager
        config_manager = ConfigManager(
            analysis_args,
            results_base=config.get("experiments", {}).get("base_output_dir", "./results"),
        )

        # Setup analysis configuration
        analysis_config = config_manager.setup_analysis_config()

        # Initialize DataManager and ModelRunner
        from analysis.data_manager import DataManager, DataManagerConfig
        from analysis.model_runner import ModelRunner, ModelRunnerConfig

        dm_config = DataManagerConfig.from_object(config_manager.args)
        data_manager = DataManager(dm_config)

        mr_config = ModelRunnerConfig.from_object(config_manager.args)
        model_runner = ModelRunner(mr_config)

        logger.info("✅ Analysis framework components initialized")

        # Create comprehensive analysis info
        metadata = {
            "analysis_integration_enabled": True,
            "framework_available": True,
            "structured_analysis": True,
            "components_available": ["DataManager", "ModelRunner", "ConfigManager"],
            "run_standard_analysis": analysis_config.run_standard,
            "run_cross_validation": analysis_config.run_cv,
            "integration_summary": {
                "structured_analysis": True,
                "data_management": True,
                "model_execution": True,
            },
        }

        return data_manager, model_runner, metadata

    except Exception as e:
        from core.error_handling import log_and_return_error

        error_info = log_and_return_error(
            e, logger, "Analysis framework pipeline integration",
            additional_fields={
                "analysis_integration_enabled": False,
                "framework_available": False,
                "fallback_mode": True,
            }
        )
        return None, None, error_info


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
        from core.model_interface import get_model_function
        from core.config_utils import dict_to_namespace

        models = get_model_function()

        # Convert args dict to namespace with defaults
        defaults = {
            'model': 'sparseGFA',
            'K': hypers.get('K', 5),
            'num_sources': len(X_list),
            'reghsZ': True,
        }
        args = dict_to_namespace(args_dict, defaults)

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
    percW: float = 33.0,
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