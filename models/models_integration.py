"""
Models Framework Integration for Remote Workstation Pipeline
Integrates structured ModelFactory, model variants, and model management into the pipeline.
"""

import argparse
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ModelCreationMode(Enum):
    """Model creation modes."""
    FACTORY = "factory"        # Full ModelFactory with structured management
    DIRECT = "direct"          # Direct model instantiation
    FALLBACK = "fallback"      # Fallback to basic model creation


@dataclass
class ModelCreationResult:
    """Result from model creation/framework initialization.

    Attributes:
        mode: Creation mode (FACTORY, DIRECT, or FALLBACK)
        model_type: Type of model created
        model_instance: The created model instance (None if creation failed)
        factory_available: Whether ModelFactory was available
        error: Error message if creation failed
        metadata: Additional metadata about model creation
    """
    mode: ModelCreationMode
    model_type: str
    model_instance: Optional[Any] = None
    factory_available: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_factory(self) -> bool:
        """Check if using factory pattern."""
        return self.mode == ModelCreationMode.FACTORY

    @property
    def is_direct(self) -> bool:
        """Check if using direct instantiation."""
        return self.mode == ModelCreationMode.DIRECT

    @property
    def is_fallback(self) -> bool:
        """Check if using fallback mode."""
        return self.mode == ModelCreationMode.FALLBACK

    @property
    def success(self) -> bool:
        """Check if model creation was successful."""
        return self.error is None and self.model_instance is not None


def get_optimal_model_configuration(
    config: Dict, data_characteristics: Dict = None, verbose: bool = True
) -> Tuple[Any, Dict]:
    """
    Determine optimal model configuration based on data characteristics and user preferences.

    Args:
        config: Configuration dictionary
        data_characteristics: Optional data characteristics for model selection
        verbose: If True, log detailed information. If False, suppress repetitive logs.

    Returns:
        Tuple of (selected_model_type, model_configuration_summary)
    """
    try:
        if verbose:
            logger.info("🚀 === OPTIMAL MODEL CONFIGURATION ===")

        # Extract model preferences from config
        model_config = config.get("model", {})
        model_type = model_config.get("model_type", "sparseGFA")  # Default to sparse GFA

        # Analyze data characteristics for model selection
        if data_characteristics and verbose:
            total_features = data_characteristics.get("total_features", 0)
            n_views = data_characteristics.get("n_views", 0)
            n_subjects = data_characteristics.get("n_subjects", 0)
            has_imaging_data = data_characteristics.get("has_imaging_data", False)

            logger.info("Data characteristics for model selection:")
            logger.info(f"   Total features: {total_features:,}")
            logger.info(f"   Views: {n_views}")
            logger.info(f"   Subjects: {n_subjects}")
            logger.info(f"   Has imaging data: {has_imaging_data}")

            # Automatic model selection based on data characteristics
            if model_config.get("auto_select", False):
                if has_imaging_data and total_features > 10000:
                    recommended_model = "neuroGFA"
                    reason = "Large neuroimaging dataset - spatial priors recommended"
                elif total_features > 5000:
                    recommended_model = "sparseGFA"
                    reason = (
                        "High-dimensional data - sparsity regularization recommended"
                    )
                else:
                    recommended_model = "GFA"
                    reason = "Standard dataset - basic factor analysis sufficient"

                logger.info(f"🎯 Model recommendation: {recommended_model}")
                logger.info(f"   Reason: {reason}")

                # Use recommendation if not explicitly overridden
                if model_config.get("use_recommendation", True):
                    model_type = recommended_model
                    logger.info(f"✅ Using recommended model: {model_type}")
                else:
                    logger.info(
                        f"⚠️  Recommendation ignored - using configured: {model_type}"
                    )

        # Validate model availability
        from models import ModelFactory

        # Include both Bayesian models and traditional baselines
        bayesian_models = ModelFactory.list_models()
        baseline_methods = ["PCA", "ICA", "FactorAnalysis", "NMF", "KMeans", "CCA"]
        available_models = bayesian_models + baseline_methods

        if model_type not in bayesian_models:
            if verbose:
                logger.warning(f"Requested model '{model_type}' not available in Bayesian models")
                logger.info(f"Available Bayesian models: {bayesian_models}")
                logger.info(f"Available baseline methods: {baseline_methods}")
            model_type = "sparseGFA"  # Fallback to default
            if verbose:
                logger.info(f"Falling back to: {model_type}")

        if verbose:
            logger.info(f"✅ Selected model type: {model_type}")

        model_configuration = {
            "model_type": model_type,
            "available_models": available_models,
            "model_factory_available": True,
            "configuration_strategy": (
                "auto_selected"
                if model_config.get("auto_select", False)
                else "configured"
            ),
            "neuroimaging_optimized": model_type == "neuroGFA",
            "sparsity_regularization": model_type in ["sparseGFA", "neuroGFA"],
            "spatial_priors": model_type == "neuroGFA",
        }

        return model_type, model_configuration

    except Exception as e:
        logger.error(f"Model configuration failed: {e}")
        return None, _fallback_model_configuration(config)


def _create_fallback_model_result(model_type: str, error_msg: str) -> ModelCreationResult:
    """Create fallback model creation result."""
    logger.warning(f"Using fallback model creation: {error_msg}")

    return ModelCreationResult(
        mode=ModelCreationMode.FALLBACK,
        model_type=model_type or "sparseGFA",
        model_instance=None,
        factory_available=False,
        error=error_msg,
        metadata={
            "available_models": [],
            "configuration_strategy": "fallback",
        }
    )


def create_model_instance(
    model_type: str, config: Dict, hypers: Dict, data_characteristics: Dict = None, verbose: bool = True
) -> ModelCreationResult:
    """
    Create a model instance using the ModelFactory.

    Args:
        model_type: Model type to create
        config: Configuration dictionary
        hypers: Hyperparameters dictionary
        data_characteristics: Optional data characteristics for model customization
        verbose: If True, log detailed information. If False, suppress repetitive logs.

    Returns:
        ModelCreationResult with model instance and metadata
    """
    try:
        if verbose:
            logger.info(f"🏗️ Creating model instance: {model_type}")

        from models import create_model

        # Convert config dict to args-like object for model compatibility
        model_args = _create_model_args_from_config(config, hypers)

        # Prepare model-specific arguments
        model_kwargs = {}

        # Special handling for neuroimaging model
        if model_type == "neuroGFA":
            spatial_info = _extract_spatial_info(data_characteristics, config)
            if spatial_info:
                model_kwargs["spatial_info"] = spatial_info
                if verbose:
                    logger.info("   Added spatial information for neuroimaging model")

        # Create model instance
        model_instance = create_model(model_type, model_args, hypers, **model_kwargs)

        metadata = {
            "model_name": model_instance.get_model_name(),
            "model_class": type(model_instance).__name__,
            "spatial_info_available": "spatial_info" in model_kwargs,
            "factory_method_used": True,
            "hyperparameters": hypers,
        }

        if verbose:
            logger.info(f"✅ Model instance created successfully")
            logger.info(f"   Model type: {model_type}")
            logger.info(f"   Model class: {metadata['model_class']}")

        return ModelCreationResult(
            mode=ModelCreationMode.FACTORY,
            model_type=model_type,
            model_instance=model_instance,
            factory_available=True,
            error=None,
            metadata=metadata
        )

    except Exception as e:
        if verbose:
            logger.error(f"Model instance creation failed: {e}")
        return _create_fallback_model_result(model_type, f"creation_failed: {str(e)}")


def _create_model_args_from_config(config: Dict, hypers: Dict) -> argparse.Namespace:
    """Create model args from configuration dictionary."""

    # Extract hyperparameter and model configuration
    hyperparam_config = config.get("hyperparameter_optimization", {})
    model_config = config.get("model", {})

    args = argparse.Namespace(
        # Model parameters
        model=model_config.get("type", "sparseGFA"),
        K=hyperparam_config.get("fallback_K", 10),
        percW=hyperparam_config.get("fallback_percW", 33),
        # Model-specific settings
        reghsZ=model_config.get("reghsZ", True),
        use_sparse=model_config.get("use_sparse", True),
        use_group=model_config.get("use_group", True),
        # Source information
        num_sources=model_config.get("num_sources", 4),
        # Additional model parameters
        slab_scale=hypers.get("slab_scale", 2.0),
        slab_df=hypers.get("slab_df", 4.0),
    )

    return args


def _extract_spatial_info(
    data_characteristics: Dict = None, config: Dict = None
) -> Optional[Dict]:
    """Extract spatial information for neuroimaging models."""
    if not data_characteristics:
        return None

    # Check if spatial information is available
    spatial_info = data_characteristics.get("spatial_info")
    if spatial_info:
        logger.info("   Using provided spatial information")
        return spatial_info

    # Check for imaging views that might need spatial processing
    imaging_views = data_characteristics.get("imaging_views", [])
    if imaging_views:
        # Create basic spatial info structure
        spatial_info = {
            "imaging_views": imaging_views,
            "spatial_smoothing": config.get("model", {}).get("spatial_smoothing", True),
            "neighborhood_structure": "voxel_grid",  # Default structure
        }
        logger.info(
            f"   Created basic spatial info for {len(imaging_views)} imaging views"
        )
        return spatial_info

    return None


def apply_models_framework_to_pipeline(
    config: Dict,
    X_list: List[np.ndarray] = None,
    data_characteristics: Dict = None,
    hypers: Dict = None,
    verbose: bool = True,
) -> Tuple[Any, Any, Dict]:
    """
    Apply comprehensive models framework to the remote workstation pipeline.

    Args:
        config: Configuration dictionary
        X_list: Optional data list for analysis
        data_characteristics: Optional data characteristics for model selection
        hypers: Optional hyperparameters dictionary
        verbose: If True, log detailed information. If False, suppress repetitive logs.

    Returns:
        Tuple of (model_type, model_instance, framework_info)
    """
    try:
        if verbose:
            logger.info("🚀 === MODELS FRAMEWORK INTEGRATION ===")

        # Get optimal model configuration
        model_type, model_config = get_optimal_model_configuration(
            config, data_characteristics, verbose=verbose
        )

        if not model_type:
            return None, None, {
                "framework_available": False,
                "error": "No model type selected",
                "fallback": True
            }

        # Create default hyperparameters if not provided
        if hypers is None:
            hypers = _create_default_hyperparameters(config, X_list)

        # Create model instance
        creation_result = create_model_instance(
            model_type, config, hypers, data_characteristics, verbose=verbose
        )

        if not creation_result.success:
            if verbose:
                logger.warning("Model instance creation failed - using fallback")
            return model_type, None, {
                "framework_available": False,
                "error": creation_result.error,
                "fallback": True
            }

        if verbose:
            logger.info("✅ Models framework integration completed")
            logger.info(f"   Model type: {model_type}")
            logger.info(f"   Model class: {creation_result.metadata.get('model_class', 'unknown')}")
            logger.info(
                f"   Factory method: {creation_result.metadata.get('factory_method_used', False)}"
            )

        framework_info = {
            "framework_available": True,
            "model_type": model_type,
            "model_instance": creation_result.model_instance,
            "model_configuration": model_config,
            "creation_info": creation_result.metadata,
            "hyperparameters": hypers,
            "models_factory_used": True,
            "structured_model_management": True,
        }

        return model_type, creation_result.model_instance, framework_info

    except Exception as e:
        logger.error(f"❌ Models framework integration failed: {e}")
        return None, None, {
            "framework_available": False,
            "error": str(e),
            "fallback": True
        }


def _create_default_hyperparameters(
    config: Dict, X_list: List[np.ndarray] = None
) -> Dict:
    """Create default hyperparameters for model creation."""
    hypers = {
        "a_sigma": 1.0,
        "b_sigma": 1.0,
        "nu_local": 1.0,
        "nu_global": 1.0,
        "slab_scale": 2.0,
        "slab_df": 4.0,
        "percW": config.get("hyperparameter_optimization", {}).get(
            "fallback_percW", 33
        ),
    }

    # Add data dimensions if available
    if X_list:
        hypers["Dm"] = [X.shape[1] for X in X_list]

    return hypers


def _create_direct_model_result(model_instance: Any, model_type: str) -> ModelCreationResult:
    """Create result for direct model instantiation (without factory)."""
    return ModelCreationResult(
        mode=ModelCreationMode.DIRECT,
        model_type=model_type,
        model_instance=model_instance,
        factory_available=False,
        error=None,
        metadata={
            "structured_model_management": False,
            "creation_approach": "direct_instantiation",
        }
    )


def run_model_comparison_analysis(
    available_models: List[str], config: Dict, X_list: List[np.ndarray], hypers: Dict
) -> Dict:
    """
    Run comparative analysis across different model types.

    Args:
        available_models: List of available model types to compare
        config: Configuration dictionary
        X_list: Data arrays
        hypers: Hyperparameters

    Returns:
        Model comparison results
    """
    try:
        logger.info("🔄 Running comprehensive model comparison...")

        comparison_results = {}

        # Analyze data characteristics for model selection
        data_characteristics = {
            "n_subjects": len(X_list[0]) if X_list else 0,
            "n_views": len(X_list) if X_list else 0,
            "total_features": sum(X.shape[1] for X in X_list) if X_list else 0,
            "view_dimensions": [X.shape[1] for X in X_list] if X_list else [],
            "has_imaging_data": (
                any(X.shape[1] > 1000 for X in X_list) if X_list else False
            ),
        }

        logger.info(f"Comparing {len(available_models)} model types:")
        logger.info(f"   Models: {available_models}")
        logger.info(
            f" Data: { data_characteristics['n_subjects']} subjects, { data_characteristics['total_features']:, } features"
        )

        for model_type in available_models:
            logger.info(f"\n🧠 Testing {model_type} model...")

            try:
                # Create model instance
                model_instance, creation_info = create_model_instance(
                    model_type, config, hypers, data_characteristics
                )

                if model_instance:
                    model_result = {
                        "model_type": model_type,
                        "creation_successful": True,
                        "model_class": creation_info.get("model_class", "unknown"),
                        "spatial_info_used": creation_info.get(
                            "spatial_info_available", False
                        ),
                        "suitable_for_data": _assess_model_suitability(
                            model_type, data_characteristics
                        ),
                        "recommended_use_cases": _get_model_use_cases(model_type),
                    }

                    logger.info(f"   ✅ {model_type}: {model_result['model_class']}")
                    logger.info(
                        f" Suitability: { model_result['suitable_for_data']['score']:.2f}"
                    )
                else:
                    model_result = {
                        "model_type": model_type,
                        "creation_successful": False,
                        "error": creation_info.get("error", "unknown"),
                        "suitable_for_data": {
                            "score": 0.0,
                            "reason": "creation_failed",
                        },
                    }
                    logger.info(f"   ❌ {model_type}: Creation failed")

                comparison_results[model_type] = model_result

            except Exception as e:
                logger.warning(f"   ⚠️  {model_type}: {str(e)}")
                comparison_results[model_type] = {
                    "model_type": model_type,
                    "creation_successful": False,
                    "error": str(e),
                    "suitable_for_data": {"score": 0.0, "reason": "exception"},
                }

        # Determine best model based on suitability scores
        best_model = None
        best_score = 0.0

        for model_type, result in comparison_results.items():
            if result.get("creation_successful", False):
                score = result.get("suitable_for_data", {}).get("score", 0.0)
                if score > best_score:
                    best_score = score
                    best_model = model_type

        logger.info(f"\n🏆 Model comparison completed")
        logger.info(f"   Best model: {best_model} (score: {best_score:.2f})")

        return {
            "comparison_type": "comprehensive_model_comparison",
            "models_tested": available_models,
            "results": comparison_results,
            "best_model": best_model,
            "best_score": best_score,
            "data_characteristics": data_characteristics,
            "recommendation": {
                "model_type": best_model,
                "confidence": best_score,
                "reason": (
                    comparison_results.get(best_model, {})
                    .get("suitable_for_data", {})
                    .get("reason", "highest_score")
                    if best_model
                    else "no_suitable_model"
                ),
            },
        }

    except Exception as e:
        logger.error(f"Model comparison analysis failed: {e}")
        return {
            "comparison_type": "comprehensive_model_comparison",
            "status": "failed",
            "error": str(e),
            "models_tested": available_models,
        }


def _assess_model_suitability(model_type: str, data_characteristics: Dict) -> Dict:
    """Assess how suitable a model is for the given data characteristics."""

    n_subjects = data_characteristics.get("n_subjects", 0)
    total_features = data_characteristics.get("total_features", 0)
    data_characteristics.get("n_views", 0)
    has_imaging_data = data_characteristics.get("has_imaging_data", False)

    if model_type == "neuroGFA":
        # NeuroGFA is best for large neuroimaging datasets
        if has_imaging_data and total_features > 5000:
            score = 0.9
            reason = "Excellent for large neuroimaging datasets with spatial structure"
        elif has_imaging_data:
            score = 0.7
            reason = "Good for neuroimaging data, spatial priors beneficial"
        else:
            score = 0.3
            reason = "Spatial priors not beneficial for non-imaging data"

    elif model_type == "sparseGFA":
        # SparseGFA is good for high-dimensional data
        if total_features > 1000:
            score = 0.8
            reason = "Excellent for high-dimensional data requiring sparsity"
        elif total_features > 500:
            score = 0.7
            reason = "Good for moderate-dimensional data"
        else:
            score = 0.5
            reason = "Adequate for lower-dimensional data"

    elif model_type == "GFA":
        # Standard GFA is good for basic datasets
        if total_features < 1000:
            score = 0.8
            reason = "Excellent for standard factor analysis tasks"
        elif total_features < 5000:
            score = 0.6
            reason = "Adequate but may benefit from sparsity regularization"
        else:
            score = 0.4
            reason = "May struggle with very high-dimensional data"
    else:
        score = 0.5
        reason = "Unknown model type"

    # Adjust based on sample size
    if n_subjects < 50:
        score *= 0.8  # Penalize for small sample size
    elif n_subjects > 200:
        score = min(1.0, score * 1.1)  # Bonus for large sample size

    return {
        "score": score,
        "reason": reason,
        "considerations": {
            "sample_size": n_subjects,
            "dimensionality": total_features,
            "imaging_data": has_imaging_data,
        },
    }


def _get_model_use_cases(model_type: str) -> List[str]:
    """Get recommended use cases for each model type."""
    use_cases = {
        "neuroGFA": [
            "Large neuroimaging datasets (fMRI, structural MRI)",
            "Voxel-wise analysis requiring spatial smoothness",
            "Multi-modal neuroimaging fusion",
            "Brain connectivity analysis",
        ],
        "sparseGFA": [
            "High-dimensional genomics data",
            "Multi-omics integration",
            "Sparse factor discovery",
            "Feature selection in factor analysis",
        ],
        "GFA": [
            "Standard factor analysis tasks",
            "Educational/research demonstrations",
            "Low to moderate dimensional data",
            "Baseline model comparison",
        ],
    }

    return use_cases.get(model_type, ["General factor analysis"])


def integrate_models_with_pipeline(
    config: Dict,
    X_list: List[np.ndarray] = None,
    data_characteristics: Dict = None,
    hypers: Dict = None,
    verbose: bool = True,
) -> Tuple[Any, Any, Dict]:
    """
    Main integration function for models framework in the pipeline.

    Args:
        config: Configuration dictionary
        X_list: Optional data arrays
        data_characteristics: Optional data characteristics
        hypers: Optional hyperparameters
        verbose: If True, log detailed information. If False, suppress repetitive logs.

    Returns:
        Tuple of (model_type, model_instance, comprehensive_models_info)
    """
    try:
        if verbose:
            logger.info("🚀 === MODELS FRAMEWORK PIPELINE INTEGRATION ===")

        # Apply comprehensive models framework
        model_type, model_instance, framework_info = apply_models_framework_to_pipeline(
            config, X_list, data_characteristics, hypers, verbose=verbose
        )

        # Create integration summary
        integration_summary = {
            "models_integration_enabled": True,
            "framework_available": framework_info.get("framework_available", False),
            "structured_model_management": framework_info.get(
                "structured_model_management", False
            ),
            "model_factory_used": framework_info.get("models_factory_used", False),
            "model_type_selected": framework_info.get("model_type", "unknown"),
            "model_instance_created": framework_info.get("model_instance") is not None,
            "neuroimaging_optimized": framework_info.get("model_type") == "neuroGFA",
            "sparsity_regularization": framework_info.get("model_type")
            in ["sparseGFA", "neuroGFA"],
        }

        # Add model comparison if multiple models available
        if framework_info.get("framework_available", False):
            try:
                from models import ModelFactory

                # Include both Bayesian models and traditional baselines
                bayesian_models = ModelFactory.list_models()
                baseline_methods = ["PCA", "ICA", "FactorAnalysis", "NMF", "KMeans", "CCA"]
                available_models = bayesian_models + baseline_methods

                if len(bayesian_models) > 1 and config.get("model", {}).get(
                    "enable_comparison", False
                ):
                    logger.info("Running model comparison analysis...")
                    comparison_results = run_model_comparison_analysis(
                        bayesian_models, config, X_list or [], hypers or {}
                    )
                    integration_summary["model_comparison"] = comparison_results
                    integration_summary["comparison_completed"] = True
                else:
                    integration_summary["comparison_completed"] = False

                integration_summary["available_models"] = available_models

            except Exception as comp_e:
                logger.warning(f"Model comparison failed: {comp_e}")
                integration_summary["comparison_completed"] = False

        if model_instance:
            if verbose:
                logger.info("✅ Models framework pipeline integration completed")
                logger.info(
                    f" Model type: { integration_summary.get( 'model_type_selected', 'unknown')}"
                )
                logger.info(
                    f" Model factory: { integration_summary.get( 'model_factory_used', False)}"
                )
                logger.info(
                    f" Structured management: { integration_summary.get( 'structured_model_management', False)}"
                )
        else:
            if verbose:
                logger.info("⚠️  Models framework unavailable - using direct approach")
            integration_summary.update(
                {
                    "fallback_mode": True,
                    "fallback_reason": framework_info.get("error", "unknown"),
                }
            )

        comprehensive_models_info = {
            **framework_info,
            "integration_summary": integration_summary,
        }

        return model_type, model_instance, comprehensive_models_info

    except Exception as e:
        logger.error(f"❌ Models framework pipeline integration failed: {e}")
        return (
            None,
            None,
            {
                "models_integration_enabled": False,
                "framework_available": False,
                "error": str(e),
                "fallback_mode": True,
            },
        )


class ModelsFrameworkWrapper:
    """Wrapper for structured models framework components."""

    def __init__(self, model_type, model_instance, framework_info):
        self.model_type = model_type
        self.model_instance = model_instance
        self.framework_info = framework_info

    def get_model_for_execution(self):
        """
        Get model ready for MCMC execution.

        Returns:
            Model instance ready for inference
        """
        if self.model_instance:
            logger.info(f"📊 Using structured model: {self.model_type}")
            logger.info(f"   Model family: {self.model_instance.__class__.__name__}")
            return self.model_instance
        else:
            logger.warning(
                "No structured model available - falling back to core models"
            )
            return None

    def get_model_info(self) -> Dict:
        """Get comprehensive model information."""
        return {
            "model_type": self.model_type,
            "model_instance_available": self.model_instance is not None,
            "model_name": (
                self.model_instance.get_model_name() if self.model_instance else "none"
            ),
            "framework_info": self.framework_info,
            "structured_model_ready": all(
                [
                    self.model_type is not None,
                    self.model_instance is not None,
                    self.framework_info.get("framework_available", False),
                ]
            ),
        }

    def compare_with_alternatives(
        self, config: Dict, X_list: List[np.ndarray], hypers: Dict
    ) -> Dict:
        """Compare current model with available alternatives."""
        try:
            from models import ModelFactory

            # Only compare Bayesian models (baselines handled separately in model_comparison experiment)
            bayesian_models = ModelFactory.list_models()

            return run_model_comparison_analysis(
                bayesian_models, config, X_list, hypers
            )
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {"error": str(e), "comparison_available": False}


def _wrap_models_framework(model_type, model_instance, framework_info):
    """Wrap models framework components."""
    return ModelsFrameworkWrapper(model_type, model_instance, framework_info)
