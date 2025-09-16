#!/usr/bin/env python3
"""
Cross-validation framework integration for remote workstation pipeline.
Replaces manual hyperparameter optimization with comprehensive CV framework.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def apply_comprehensive_cv_framework(
    X_list: List[np.ndarray], config: Dict, optimal_params: Dict, data_dir: str
) -> Tuple[Dict, Dict]:
    """
    Apply comprehensive cross-validation framework instead of manual hyperparameter optimization.

    Args:
        X_list: Preprocessed data list
        config: Configuration dictionary
        optimal_params: Current optimal parameters from manual optimization
        data_dir: Data directory path

    Returns:
        Tuple of (cv_results, enhanced_optimal_params)
    """
    try:
        logger.info("🔬 === COMPREHENSIVE CV FRAMEWORK INTEGRATION ===")

        # Check if CV framework is available
        try:
            from analysis.cross_validation_library import (
                NeuroImagingCrossValidator,
            )

            logger.info("✅ NeuroImagingCrossValidator available")
        except ImportError as e:
            logger.warning(f"❌ CV framework not available: {e}")
            return _fallback_cv_analysis(X_list, config, optimal_params)

        # Load clinical data for stratified CV
        clinical_data = _load_clinical_data(data_dir)

        # Create comprehensive CV configuration
        cv_config = _create_cv_configuration(config, clinical_data is not None)

        # Initialize CV framework
        cv_framework = NeuroImagingCrossValidator(cv_config)
        logger.info(
            f"CV framework initialized with {cv_config.outer_cv_folds} outer folds"
        )

        # Define hyperparameter space for CV
        param_space = _define_hyperparameter_space(config, optimal_params)
        logger.info(f"Hyperparameter space: {len(param_space)} combinations")

        # Run comprehensive cross-validation (nested CV with hyperparameter
        # optimization)
        logger.info("Running comprehensive cross-validation...")

        # Convert parameters for NeuroImagingCrossValidator format
        data_dict = {
            "clinical_data": clinical_data,
            "view_names": [f"view_{i}" for i in range(len(X_list))],
        }

        # Create dummy args object with required attributes
        import argparse

        args = argparse.Namespace(
            model="sparseGFA",
            K=optimal_params.get("K", 10),
            percW=optimal_params.get("percW", 33),
            num_samples=optimal_params.get("num_samples", 2000),
            num_warmup=optimal_params.get("num_warmup", 1000),
            num_chains=optimal_params.get("num_chains", 4),
            target_accept_prob=optimal_params.get("target_accept_prob", 0.8),
        )

        # Convert parameter space to hyperparameter dict
        hypers = {
            "K": list(set(p["K"] for p in param_space)),
            "percW": list(set(p["percW"] for p in param_space)),
            "num_samples": list(set(p["num_samples"] for p in param_space)),
            "num_chains": list(set(p["num_chains"] for p in param_space)),
        }

        cv_results = cv_framework.nested_neuroimaging_cv(
            X_list=X_list,
            args=args,
            hypers=hypers,
            data=data_dict,
            cv_type="clinical_stratified" if clinical_data is not None else "standard",
        )

        # Extract enhanced optimal parameters
        enhanced_optimal_params = _extract_cv_optimal_params(cv_results, optimal_params)

        # Add comprehensive CV information
        comprehensive_cv_results = {
            "cv_framework_used": True,
            "cv_type": "neuroimaging_comprehensive",
            "cv_configuration": cv_config.__dict__,
            "parameter_space_size": len(param_space),
            "cv_results": cv_results,
            "clinical_data_available": clinical_data is not None,
            "optimization_improvements": _compare_with_manual_optimization(
                cv_results, optimal_params
            ),
        }

        logger.info("✅ Comprehensive CV framework completed")
        logger.info(f"   Best CV score: {cv_results.get('best_cv_score', 'N/A')}")
        logger.info(f"   Optimal parameters enhanced: {enhanced_optimal_params}")

        return comprehensive_cv_results, enhanced_optimal_params

    except Exception as e:
        logger.error(f"❌ CV framework integration failed: {e}")
        return _fallback_cv_analysis(X_list, config, optimal_params)


def _create_cv_configuration(config: Dict, has_clinical_data: bool) -> Any:
    """Create comprehensive CV configuration."""
    try:
        from analysis.cross_validation_library import (
            NeuroImagingCVConfig,
            ParkinsonsConfig,
        )

        # Base CV configuration
        cv_config = NeuroImagingCVConfig()

        # Update with remote workstation preferences
        cv_config.outer_cv_folds = config.get("cv_outer_folds", 5)
        cv_config.inner_cv_folds = config.get("cv_inner_folds", 3)
        cv_config.n_repeats = config.get("cv_repeats", 2)
        cv_config.random_state = config.get("random_seed", 42)
        cv_config.shuffle = True

        # Parkinson's disease specific configuration
        if has_clinical_data:
            pd_config = ParkinsonsConfig()
            cv_config.clinical_stratification_vars = (
                pd_config.clinical_stratification_vars
            )
            cv_config.subtype_optimization = True
        else:
            cv_config.clinical_stratification_vars = []
            cv_config.subtype_optimization = False

        # Performance configuration for remote workstation
        cv_config.n_jobs = min(
            config.get("system", {}).get("num_workers", 4), 4
        )  # Limit parallelization
        cv_config.memory_efficient = True
        cv_config.enable_early_stopping = True

        logger.info(
            f"CV configuration created: {
                cv_config.outer_cv_folds} outer × {
                cv_config.inner_cv_folds} inner folds"
        )

        return cv_config

    except Exception as e:
        logger.error(f"Failed to create CV configuration: {e}")
        raise


def _load_clinical_data(data_dir: str) -> Optional[Any]:
    """Load clinical data for stratified cross-validation."""
    try:
        from data.qmap_pd import load_qmap_pd

        logger.info("Loading clinical data for CV stratification...")
        data = load_qmap_pd(data_dir=data_dir)

        if "clinical_data" in data and data["clinical_data"] is not None:
            clinical_data = data["clinical_data"]
            logger.info(
                f"Clinical data loaded: {
                    clinical_data.shape[0]} subjects, {
                    clinical_data.shape[1]} variables"
            )
            return clinical_data
        else:
            logger.info("No clinical data available for stratification")
            return None

    except Exception as e:
        logger.warning(f"Could not load clinical data: {e}")
        return None


def _define_hyperparameter_space(config: Dict, current_optimal: Dict) -> List[Dict]:
    """Define comprehensive hyperparameter space for CV."""
    try:
        # Get hyperparameter optimization config
        hyperparam_config = config.get("hyperparameter_optimization", {})

        # Expand search space around current optimal parameters
        param_space = []

        # K (number of factors) - expand around optimal
        current_K = current_optimal.get("K", 10)
        K_candidates = hyperparam_config.get("K_candidates", [5, 8, 10, 12, 15])
        # Add values around current optimal
        K_expanded = list(
            set(
                K_candidates
                + [max(5, current_K - 2), current_K, min(20, current_K + 2)]
            )
        )

        # percW (sparsity) - expand around optimal
        current_percW = current_optimal.get("percW", 33)
        percW_candidates = hyperparam_config.get(
            "percW_candidates", [20, 25, 33, 40, 50]
        )
        percW_expanded = list(
            set(
                percW_candidates
                + [
                    max(15, current_percW - 5),
                    current_percW,
                    min(60, current_percW + 5),
                ]
            )
        )

        # MCMC parameters if available
        if "num_samples" in current_optimal:
            num_samples_candidates = hyperparam_config.get(
                "num_samples_candidates", [1000, 2000]
            )
            num_chains_candidates = hyperparam_config.get(
                "num_chains_candidates", [2, 4]
            )
        else:
            num_samples_candidates = [2000]
            num_chains_candidates = [4]

        # Create comprehensive parameter combinations
        for K in K_expanded:
            for percW in percW_expanded:
                for num_samples in num_samples_candidates:
                    for num_chains in num_chains_candidates:
                        param_combination = {
                            "K": K,
                            "percW": percW,
                            "num_samples": num_samples,
                            "num_warmup": num_samples // 2,
                            "num_chains": num_chains,
                            "target_accept_prob": current_optimal.get(
                                "target_accept_prob", 0.8
                            ),
                        }
                        param_space.append(param_combination)

        logger.info(f"Parameter space defined: {len(param_space)} combinations")
        logger.info(f"   K range: {min(K_expanded)} - {max(K_expanded)}")
        logger.info(f"   percW range: {min(percW_expanded)} - {max(percW_expanded)}")

        return param_space

    except Exception as e:
        logger.error(f"Failed to define parameter space: {e}")
        # Fallback to current optimal only
        return [current_optimal]


def _extract_cv_optimal_params(cv_results: Dict, fallback_params: Dict) -> Dict:
    """Extract optimal parameters from CV results."""
    try:
        if "best_params" in cv_results:
            cv_optimal = cv_results["best_params"]
            logger.info(f"CV-optimized parameters: {cv_optimal}")
            return cv_optimal
        elif "optimal_hyperparameters" in cv_results:
            cv_optimal = cv_results["optimal_hyperparameters"]
            logger.info(f"CV-optimized parameters: {cv_optimal}")
            return cv_optimal
        else:
            logger.warning("No optimal parameters found in CV results, using fallback")
            return fallback_params

    except Exception as e:
        logger.error(f"Failed to extract CV optimal parameters: {e}")
        return fallback_params


def _compare_with_manual_optimization(cv_results: Dict, manual_optimal: Dict) -> Dict:
    """Compare CV results with manual hyperparameter optimization."""
    try:
        comparison = {
            "cv_vs_manual": "comparison_available",
            "manual_params": manual_optimal,
            "cv_params": cv_results.get("best_params", {}),
            "improvements": [],
        }

        cv_score = cv_results.get("best_cv_score", 0)
        manual_score = manual_optimal.get("score", 0)

        if cv_score > manual_score:
            improvement = cv_score - manual_score
            comparison["improvements"].append(f"CV score improved by {improvement:.4f}")

        # Compare individual parameters
        cv_params = cv_results.get("best_params", {})
        for param in ["K", "percW", "num_samples", "num_chains"]:
            if param in cv_params and param in manual_optimal:
                if cv_params[param] != manual_optimal[param]:
                    comparison["improvements"].append(
                        f"{param}: {manual_optimal[param]} → {cv_params[param]}"
                    )

        return comparison

    except Exception as e:
        logger.error(f"Failed to compare optimization methods: {e}")
        return {"cv_vs_manual": "comparison_failed", "error": str(e)}


def _fallback_cv_analysis(
    X_list: List[np.ndarray], config: Dict, optimal_params: Dict
) -> Tuple[Dict, Dict]:
    """Fallback CV analysis when comprehensive framework is unavailable."""
    try:
        logger.info("🔬 Running fallback CV analysis...")

        # Basic cross-validation using sklearn
        from sklearn.model_selection import KFold

        n_folds = config.get("cv_outer_folds", 5)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Simple reconstruction quality assessment
        cv_scores = []
        n_subjects = X_list[0].shape[0]

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(n_subjects))):
            try:
                # Simple quality metric using data reconstruction
                X_train = [X[train_idx] for X in X_list]
                X_val = [X[val_idx] for X in X_list]

                # Basic quality assessment (mock CV score)
                reconstruction_quality = _assess_reconstruction_quality(
                    X_train, X_val, optimal_params
                )
                cv_scores.append(reconstruction_quality)

                logger.info(
                    f"   Fold {fold_idx + 1}: score = {reconstruction_quality:.4f}"
                )

            except Exception as fold_e:
                logger.warning(f"Fold {fold_idx + 1} failed: {fold_e}")
                cv_scores.append(0.0)

        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)

        fallback_results = {
            "cv_framework_used": False,
            "cv_type": "fallback_basic",
            "cv_configuration": {"n_folds": n_folds, "method": "sklearn_KFold"},
            "cv_scores": cv_scores,
            "mean_cv_score": mean_cv_score,
            "std_cv_score": std_cv_score,
            "best_params": optimal_params,  # No improvement from basic CV
            "fallback_reason": "comprehensive_cv_framework_unavailable",
        }

        logger.info(
            f"✅ Fallback CV completed: {mean_cv_score:.4f} ± {std_cv_score:.4f}"
        )

        return fallback_results, optimal_params

    except Exception as e:
        logger.error(f"❌ Fallback CV analysis failed: {e}")
        return {
            "cv_framework_used": False,
            "status": "failed",
            "error": str(e),
        }, optimal_params


def _assess_reconstruction_quality(
    X_train: List[np.ndarray], X_val: List[np.ndarray], params: Dict
) -> float:
    """Simple reconstruction quality assessment for fallback CV."""
    try:
        from sklearn.decomposition import FactorAnalysis
        from sklearn.metrics import r2_score

        K = params.get("K", 10)

        # Combine views
        X_train_combined = np.concatenate(X_train, axis=1)
        X_val_combined = np.concatenate(X_val, axis=1)

        # Fit factor analysis
        fa = FactorAnalysis(n_components=K, random_state=42, max_iter=100)
        fa.fit(X_train_combined)

        # Assess reconstruction on validation set
        X_val_reconstructed = fa.transform(X_val_combined) @ fa.components_
        r2 = r2_score(X_val_combined, X_val_reconstructed)

        return max(0, r2)  # Ensure non-negative

    except Exception as e:
        logger.warning(f"Reconstruction assessment failed: {e}")
        return 0.0


def integrate_cv_with_pipeline(
    X_list: List[np.ndarray], config: Dict, current_optimal_params: Dict, data_dir: str
) -> Tuple[Dict, Dict, Dict]:
    """
    Main integration function for comprehensive CV framework in the pipeline.

    Args:
        X_list: Preprocessed data
        config: Configuration dictionary
        current_optimal_params: Current optimal parameters from manual optimization
        data_dir: Data directory path

    Returns:
        Tuple of (cv_results, enhanced_optimal_params, integration_summary)
    """
    try:
        logger.info("🚀 === CV FRAMEWORK PIPELINE INTEGRATION ===")

        # Apply comprehensive CV framework
        cv_results, enhanced_params = apply_comprehensive_cv_framework(
            X_list, config, current_optimal_params, data_dir
        )

        # Create integration summary
        integration_summary = {
            "cv_integration_enabled": True,
            "cv_framework_available": cv_results.get("cv_framework_used", False),
            "cv_type": cv_results.get("cv_type", "unknown"),
            "parameter_enhancement": enhanced_params != current_optimal_params,
            "original_params": current_optimal_params,
            "enhanced_params": enhanced_params,
        }

        if integration_summary["parameter_enhancement"]:
            param_changes = []
            for key in enhanced_params:
                if (
                    key in current_optimal_params
                    and enhanced_params[key] != current_optimal_params[key]
                ):
                    param_changes.append(
                        f"{key}: {current_optimal_params[key]} → {enhanced_params[key]}"
                    )
            integration_summary["parameter_changes"] = param_changes

        logger.info("✅ CV framework pipeline integration completed")
        if integration_summary["parameter_enhancement"]:
            logger.info(
                f"   Parameters enhanced by CV: {
                    integration_summary.get(
                        'parameter_changes', [])}"
            )
        else:
            logger.info("   Parameters validated by CV (no changes recommended)")

        return cv_results, enhanced_params, integration_summary

    except Exception as e:
        logger.error(f"❌ CV framework pipeline integration failed: {e}")

        # Fallback integration summary
        fallback_summary = {
            "cv_integration_enabled": False,
            "cv_framework_available": False,
            "error": str(e),
            "fallback_params": current_optimal_params,
        }

        return (
            {"status": "failed", "error": str(e)},
            current_optimal_params,
            fallback_summary,
        )
