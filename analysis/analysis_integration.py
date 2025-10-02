"""
Analysis Framework Integration for Remote Workstation Pipeline
Integrates structured DataManager, ModelRunner, and ConfigManager into the pipeline.
"""

import argparse
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AnalysisMode(Enum):
    """Analysis execution modes."""
    STRUCTURED = "structured"  # Full framework with DataManager/ModelRunner
    FALLBACK = "fallback"      # Fallback to direct core.run_analysis
    BASIC = "basic"            # Basic analysis without framework


@dataclass
class AnalysisFrameworkResult:
    """Result from analysis framework initialization.

    Attributes:
        mode: Execution mode (STRUCTURED, FALLBACK, or BASIC)
        data_manager: DataManager instance (None if mode != STRUCTURED)
        model_runner: ModelRunner instance (None if mode != STRUCTURED)
        config_manager: ConfigManager instance (None if not available)
        error: Error message if initialization failed
        metadata: Additional metadata about the framework setup
    """
    mode: AnalysisMode
    data_manager: Optional[Any] = None
    model_runner: Optional[Any] = None
    config_manager: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_structured(self) -> bool:
        """Check if using structured framework."""
        return self.mode == AnalysisMode.STRUCTURED

    @property
    def is_fallback(self) -> bool:
        """Check if using fallback mode."""
        return self.mode == AnalysisMode.FALLBACK

    @property
    def success(self) -> bool:
        """Check if initialization was successful."""
        return self.error is None


def get_optimal_analysis_configuration(
    config: Dict, data_characteristics: Dict = None
) -> Tuple[Any, Dict]:
    """
    Determine optimal analysis configuration based on system and data characteristics.

    Args:
        config: Configuration dictionary
        data_characteristics: Optional data characteristics for optimization

    Returns:
        Tuple of (analysis_config, configuration_summary)
    """
    try:
        logger.info("🚀 === OPTIMAL ANALYSIS CONFIGURATION ===")

        # Import analysis components
        from analysis.config_manager import (
            ConfigManager,
        )

        # Create args object from config
        analysis_args = _create_analysis_args_from_config(config)

        # Initialize config manager
        config_manager = ConfigManager(
            analysis_args,
            results_base=config.get("experiments", {}).get(
                "base_output_dir", "./results"
            ),
        )

        # Setup analysis configuration
        analysis_config = config_manager.setup_analysis_config()

        # Log configuration details
        logger.info("Analysis configuration setup completed:")
        logger.info(f"   Standard analysis: {analysis_config.run_standard}")
        logger.info(f"   Cross-validation: {analysis_config.run_cv}")
        logger.info(
            f"   Dependencies checked: {config_manager.dependencies.cv_available}"
        )

        config_summary = {
            "analysis_framework": True,
            "config_manager": config_manager,
            "analysis_config": analysis_config,
            "dependencies": config_manager.dependencies,
            "run_standard": analysis_config.run_standard,
            "run_cv": analysis_config.run_cv,
            "directories": {
                "standard": (
                    str(analysis_config.standard_res_dir)
                    if analysis_config.standard_res_dir
                    else None
                ),
                "cv": (
                    str(analysis_config.cv_res_dir)
                    if analysis_config.cv_res_dir
                    else None
                ),
            },
        }

        logger.info("✅ Analysis configuration optimization completed")

        return config_manager, config_summary

    except Exception as e:
        logger.error(f"Analysis configuration failed: {e}")
        return None, _fallback_analysis_configuration(config)


def _create_analysis_args_from_config(config: Dict) -> argparse.Namespace:
    """Create analysis args from configuration dictionary."""

    # Extract hyperparameter configuration
    hyperparam_config = config.get("hyperparameter_optimization", {})
    training_config = config.get("training", {})

    args = argparse.Namespace(
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
        neuroimaging_cv=config.get("cross_validation", {}).get(
            "neuroimaging_cv", False
        ),
        nested_cv=config.get("cross_validation", {}).get("nested_cv", False),
        # Preprocessing
        enable_preprocessing=config.get("preprocessing", {}).get("enabled", True),
        enable_spatial_processing=config.get("preprocessing", {}).get(
            "spatial_processing", False
        ),
        preprocessing_params=config.get("preprocessing", {}).get("params", {}),
    )

    return args


def _create_fallback_result(error_msg: str) -> AnalysisFrameworkResult:
    """Create fallback analysis framework result."""
    logger.warning(f"Using fallback analysis: {error_msg}")

    return AnalysisFrameworkResult(
        mode=AnalysisMode.FALLBACK,
        data_manager=None,
        model_runner=None,
        config_manager=None,
        error=error_msg,
        metadata={
            "run_standard": True,
            "run_cv": False,
            "fallback_approach": "direct_core_analysis",
        }
    )


def apply_analysis_framework_to_pipeline(
    config: Dict,
    X_list: List[np.ndarray] = None,
    data_dir: str = None,
    output_dir: str = None,
) -> AnalysisFrameworkResult:
    """
    Apply comprehensive analysis framework to the remote workstation pipeline.

    Args:
        config: Configuration dictionary
        X_list: Optional data list
        data_dir: Data directory path
        output_dir: Output directory for analysis results

    Returns:
        AnalysisFrameworkResult with mode, components, and metadata
    """
    try:
        logger.info("🚀 === ANALYSIS FRAMEWORK INTEGRATION ===")

        # Get analysis configuration
        config_manager, config_summary = get_optimal_analysis_configuration(config)

        if not config_manager:
            return _create_fallback_result("config_manager_unavailable")

        # Initialize DataManager
        logger.info("📊 Initializing DataManager...")
        from analysis.data_manager import DataManager, DataManagerConfig

        # Create data manager with proper config
        dm_config = DataManagerConfig.from_object(config_manager.args)
        data_manager = DataManager(dm_config)

        # Initialize ModelRunner
        logger.info("🔄 Initializing ModelRunner...")
        from analysis.model_runner import ModelRunner, ModelRunnerConfig

        mr_config = ModelRunnerConfig.from_object(config_manager.args)
        model_runner = ModelRunner(mr_config)

        logger.info("✅ Analysis framework components initialized")
        logger.info(f"   DataManager: {type(data_manager).__name__}")
        logger.info(f"   ModelRunner: {type(model_runner).__name__}")
        logger.info(f"   ConfigManager: {type(config_manager).__name__}")

        metadata = {
            "config_summary": config_summary,
            "dependencies": config_summary.get("dependencies"),
            "result_directories": config_summary.get("directories", {}),
            "components_initialized": ["DataManager", "ModelRunner", "ConfigManager"],
            "run_standard": config_summary.get('run_standard', False),
            "run_cv": config_summary.get('run_cv', False),
        }

        logger.info("✅ Analysis framework integration completed")
        logger.info(f"   Framework type: Structured analysis components")
        logger.info(f"   Standard analysis: {metadata['run_standard']}")
        logger.info(f"   Cross-validation: {metadata['run_cv']}")

        return AnalysisFrameworkResult(
            mode=AnalysisMode.STRUCTURED,
            data_manager=data_manager,
            model_runner=model_runner,
            config_manager=config_manager,
            error=None,
            metadata=metadata
        )

    except Exception as e:
        logger.error(f"❌ Analysis framework integration failed: {e}")
        return _create_fallback_result(f"initialization_failed: {str(e)}")


def run_structured_mcmc_analysis(
    model_runner, data_manager, X_list: List[np.ndarray], config: Dict
) -> Dict:
    """
    Run MCMC analysis using structured analysis framework.

    Args:
        model_runner: ModelRunner instance
        data_manager: DataManager instance
        X_list: Data arrays
        config: Configuration dictionary

    Returns:
        Analysis results dictionary
    """
    try:
        logger.info("🔄 Running structured MCMC analysis...")

        # Prepare data for analysis
        X_prepared, hypers = data_manager.prepare_for_analysis({"X_list": X_list})

        # Create data dictionary for model runner
        data_dict = {"X_list": X_prepared, "hypers": hypers}

        logger.info(f"Data prepared for analysis:")
        logger.info(f"   Views: {len(X_prepared)}")
        logger.info(f"   Shapes: {[X.shape for X in X_prepared]}")
        logger.info(f"   Hyperparameters: {list(hypers.keys())}")

        # Run standard analysis
        results = model_runner.run_standard_analysis(X_prepared, hypers, data_dict)

        logger.info("✅ Structured MCMC analysis completed")
        logger.info(f"   Runs completed: {len(results)}")

        # Process results
        processed_results = {
            "analysis_type": "structured_mcmc",
            "runs": results,
            "num_runs": len(results),
            "hyperparameters": hypers,
            "data_info": {
                "num_views": len(X_prepared),
                "view_shapes": [X.shape for X in X_prepared],
                "total_features": sum(X.shape[1] for X in X_prepared),
                "num_subjects": X_prepared[0].shape[0],
            },
            "structured_framework": True,
        }

        return processed_results

    except Exception as e:
        logger.error(f"Structured MCMC analysis failed: {e}")
        return {
            "analysis_type": "structured_mcmc",
            "status": "failed",
            "error": str(e),
            "structured_framework": False,
        }


def integrate_analysis_with_pipeline(
    config: Dict,
    data_dir: str,
    X_list: List[np.ndarray] = None,
    output_dir: str = None,
) -> Tuple[Any, Any, Dict]:
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

        # Apply comprehensive analysis framework
        data_manager, model_runner, framework_info = (
            apply_analysis_framework_to_pipeline(config, X_list, data_dir, output_dir)
        )

        # Create integration summary
        integration_summary = {
            "analysis_integration_enabled": True,
            "framework_available": framework_info.get("framework_available", False),
            "structured_analysis": framework_info.get("structured_analysis", False),
            "components_available": framework_info.get("components_initialized", []),
            "dependencies_checked": framework_info.get("dependencies") is not None,
            "result_directories_setup": bool(
                framework_info.get("result_directories", {})
            ),
            "data_management": data_manager is not None,
            "model_execution": model_runner is not None,
        }

        # Add configuration details
        if framework_info.get("config_summary"):
            config_summary = framework_info["config_summary"]
            integration_summary.update(
                {
                    "run_standard_analysis": config_summary.get("run_standard", False),
                    "run_cross_validation": config_summary.get("run_cv", False),
                    "cv_dependencies_available": (
                        config_summary.get("dependencies", {}).cv_available
                        if config_summary.get("dependencies")
                        else False
                    ),
                    "preprocessing_dependencies_available": (
                        config_summary.get("dependencies", {}).preprocessing_available
                        if config_summary.get("dependencies")
                        else False
                    ),
                    "factor_mapping_available": (
                        config_summary.get("dependencies", {}).factor_mapping_available
                        if config_summary.get("dependencies")
                        else False
                    ),
                }
            )

        if data_manager and model_runner:
            logger.info("✅ Analysis framework pipeline integration completed")
            logger.info(f"   DataManager available: {data_manager is not None}")
            logger.info(f"   ModelRunner available: {model_runner is not None}")
            logger.info(
                f" Structured analysis: { integration_summary.get( 'structured_analysis', False)}"
            )
            logger.info(
                f" Components: { ', '.join( integration_summary.get( 'components_available', []))}"
            )
        else:
            logger.info("⚠️  Analysis framework unavailable - using direct approach")
            integration_summary.update(
                {
                    "fallback_mode": True,
                    "fallback_reason": framework_info.get("error", "unknown"),
                }
            )

        comprehensive_analysis_info = {
            **framework_info,
            "integration_summary": integration_summary,
        }

        return data_manager, model_runner, comprehensive_analysis_info

    except Exception as e:
        logger.error(f"❌ Analysis framework pipeline integration failed: {e}")
        return (
            None,
            None,
            {
                "analysis_integration_enabled": False,
                "framework_available": False,
                "error": str(e),
                "fallback_mode": True,
            },
        )


class AnalysisFrameworkWrapper:
    """Wrapper for structured analysis framework components."""

    def __init__(self, data_manager, model_runner, framework_info):
        self.data_manager = data_manager
        self.model_runner = model_runner
        self.framework_info = framework_info

    def load_and_prepare_data(self) -> Tuple[List[np.ndarray], Dict]:
        """
        Load and prepare data using DataManager.

        Returns:
            Tuple of (X_list, prepared_data_info)
        """
        try:
            if not self.data_manager:
                raise ValueError("DataManager not available")

            logger.info("📊 Loading data with structured DataManager...")

            # Load data
            data = self.data_manager.load_data()

            # Prepare for analysis
            X_list, hypers = self.data_manager.prepare_for_analysis(data)

            prepared_data_info = {
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

            if "preprocessing" in data:
                prepared_data_info["preprocessing_results"] = data["preprocessing"]

            logger.info("✅ Data loaded and prepared with structured framework")
            logger.info(f"   Views: {len(X_list)}")
            logger.info(f"   Total features: {sum(X.shape[1] for X in X_list):,}")
            logger.info(f"   Subjects: {X_list[0].shape[0]}")

            return X_list, prepared_data_info

        except Exception as e:
            logger.error(f"Structured data loading failed: {e}")
            return [], {"data_loaded": False, "error": str(e)}

    def run_analysis(self, X_list: List[np.ndarray]) -> Dict:
        """
        Run analysis using ModelRunner.

        Args:
            X_list: Data arrays

        Returns:
            Analysis results
        """
        if not self.model_runner:
            return {"status": "failed", "error": "ModelRunner not available"}

        return run_structured_mcmc_analysis(
            self.model_runner, self.data_manager, X_list, self.framework_info
        )

    def get_framework_status(self) -> Dict:
        """Get comprehensive framework status."""
        return {
            "data_manager_available": self.data_manager is not None,
            "model_runner_available": self.model_runner is not None,
            "framework_info": self.framework_info,
            "structured_analysis_ready": all(
                [
                    self.data_manager is not None,
                    self.model_runner is not None,
                    self.framework_info.get("framework_available", False),
                ]
            ),
        }


def _wrap_analysis_framework(data_manager, model_runner, framework_info):
    """Wrap analysis framework components."""
    return AnalysisFrameworkWrapper(data_manager, model_runner, framework_info)
