#!/usr/bin/env python3
"""
Comprehensive preprocessing integration for remote workstation pipeline.
Replaces basic load_qmap_pd() with full NeuroImagingPreprocessor capabilities.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def get_advanced_preprocessing_data(
    config: Dict, data_dir: str, preprocessing_strategy: str = "standard"
) -> Tuple[List[np.ndarray], Dict]:
    """
    Load data using comprehensive NeuroImagingPreprocessor instead of basic load_qmap_pd.

    Args:
        config: Configuration dictionary
        data_dir: Data directory path
        preprocessing_strategy: Strategy from config (minimal, standard, aggressive, clinical_focused)

    Returns:
        Tuple of (X_list, preprocessing_info)
    """
    try:
        logger.info(f"🔧 === ADVANCED PREPROCESSING INTEGRATION ===")
        logger.info(f"Strategy: {preprocessing_strategy}")
        logger.info(f"Data directory: {data_dir}")

        # Get preprocessing configuration
        preprocessing_strategies = config.get("data_validation", {}).get(
            "preprocessing_strategies", {}
        )
        strategy_config = preprocessing_strategies.get(preprocessing_strategy, {})

        if not strategy_config:
            logger.warning(
                f"Preprocessing strategy '{preprocessing_strategy}' not found in config, using standard"
            )
            strategy_config = preprocessing_strategies.get("standard", {})

        logger.info(f"Preprocessing configuration: {strategy_config}")

        # Load basic data first
        from data.qmap_pd import load_qmap_pd

        logger.info("Loading basic qMAP-PD data...")
        basic_data = load_qmap_pd(data_dir=data_dir)
        X_list_raw = basic_data["X_list"]
        view_names = basic_data.get(
            "view_names", [f"view_{i}" for i in range(len(X_list_raw))]
        )

        logger.info(
            f"Loaded data: {len(X_list_raw)} views with shapes {[X.shape for X in X_list_raw]}"
        )

        # Determine preprocessing approach
        if not strategy_config.get("enable_advanced_preprocessing", False):
            logger.info("Using minimal preprocessing (basic scaling only)")
            return _apply_minimal_preprocessing(X_list_raw, view_names, strategy_config)
        else:
            logger.info("Using advanced NeuroImagingPreprocessor")
            return _apply_advanced_preprocessing(
                X_list_raw, view_names, strategy_config, data_dir
            )

    except Exception as e:
        logger.error(f"❌ Advanced preprocessing failed: {e}")
        logger.info("Falling back to basic load_qmap_pd...")
        # Fallback to basic loading
        from data.qmap_pd import load_qmap_pd

        basic_data = load_qmap_pd(data_dir=data_dir)
        preprocessing_info = {
            "status": "fallback_basic",
            "strategy": "basic_load_qmap_pd",
            "error": str(e),
        }
        return basic_data["X_list"], preprocessing_info


def _apply_minimal_preprocessing(
    X_list: List[np.ndarray], view_names: List[str], strategy_config: Dict
) -> Tuple[List[np.ndarray], Dict]:
    """Apply minimal preprocessing using BasicPreprocessor."""
    try:
        from data.preprocessing import BasicPreprocessor

        logger.info("Initializing BasicPreprocessor...")
        preprocessor = BasicPreprocessor()

        # Apply basic preprocessing
        X_processed = preprocessor.fit_transform(X_list, view_names)

        preprocessing_info = {
            "status": "completed",
            "strategy": "minimal",
            "preprocessor_type": "BasicPreprocessor",
            "steps_applied": ["scaling"],
            "imputation_strategy": strategy_config.get("imputation_strategy", "mean"),
            "n_views": len(X_list),
            "original_shapes": [X.shape for X in X_list],
            "processed_shapes": [X.shape for X in X_processed],
        }

        logger.info(f"✅ Minimal preprocessing completed")
        logger.info(f"   Applied: {preprocessing_info['steps_applied']}")

        return X_processed, preprocessing_info

    except Exception as e:
        logger.error(f"Minimal preprocessing failed: {e}")
        raise


def _apply_advanced_preprocessing(
    X_list: List[np.ndarray],
    view_names: List[str],
    strategy_config: Dict,
    data_dir: str,
) -> Tuple[List[np.ndarray], Dict]:
    """Apply advanced preprocessing using NeuroImagingPreprocessor."""
    try:
        from data.preprocessing import NeuroImagingConfig, NeuroImagingPreprocessor

        logger.info("Initializing NeuroImagingPreprocessor...")

        # Create comprehensive preprocessing configuration
        preprocessing_config = NeuroImagingConfig(
            # Basic parameters
            imputation_strategy=strategy_config.get("imputation_strategy", "median"),
            feature_selection_method=strategy_config.get(
                "feature_selection_method", "variance"
            ),
            variance_threshold=strategy_config.get("variance_threshold", 0.01),
            n_top_features=strategy_config.get("n_top_features", None),
            missing_threshold=strategy_config.get("missing_threshold", 0.1),
            # Neuroimaging-specific parameters
            enable_spatial_processing=strategy_config.get(
                "enable_spatial_processing", False
            ),
            spatial_imputation=strategy_config.get("spatial_imputation", False),
            roi_based_selection=strategy_config.get("roi_based_selection", False),
            harmonize_scanners=strategy_config.get("harmonize_scanners", False),
            qc_outlier_threshold=strategy_config.get("qc_outlier_threshold", 3.0),
            spatial_neighbor_radius=strategy_config.get("spatial_neighbor_radius", 5.0),
            min_voxel_distance=strategy_config.get("min_voxel_distance", 3.0),
        )

        # Validate configuration
        preprocessing_config.validate()
        logger.info(f"Configuration validated: {preprocessing_config.to_dict()}")

        # Initialize preprocessor
        preprocessor = NeuroImagingPreprocessor(
            data_dir=data_dir, **preprocessing_config.to_dict()
        )

        # Apply advanced preprocessing
        logger.info("Applying comprehensive neuroimaging preprocessing...")
        X_processed = preprocessor.fit_transform(X_list, view_names)

        # Collect preprocessing information
        steps_applied = ["scaling", "imputation"]
        if preprocessing_config.feature_selection_method != "none":
            steps_applied.append("feature_selection")
        if preprocessing_config.enable_spatial_processing:
            steps_applied.extend(["spatial_processing", "quality_control"])
        if preprocessing_config.harmonize_scanners:
            steps_applied.append("scanner_harmonization")

        preprocessing_info = {
            "status": "completed",
            "strategy": strategy_config,
            "preprocessor_type": "NeuroImagingPreprocessor",
            "steps_applied": steps_applied,
            "config": preprocessing_config.to_dict(),
            "n_views": len(X_list),
            "original_shapes": [X.shape for X in X_list],
            "processed_shapes": [X.shape for X in X_processed],
            "features_before_selection": [X.shape[1] for X in X_list],
            "features_after_selection": [X.shape[1] for X in X_processed],
            "preprocessing_details": {
                "imputation_strategy": preprocessing_config.imputation_strategy,
                "feature_selection": preprocessing_config.feature_selection_method,
                "spatial_processing_enabled": preprocessing_config.enable_spatial_processing,
                "scanner_harmonization_enabled": preprocessing_config.harmonize_scanners,
            },
        }

        # Add feature reduction summary
        total_features_before = sum(X.shape[1] for X in X_list)
        total_features_after = sum(X.shape[1] for X in X_processed)
        reduction_ratio = (
            total_features_after / total_features_before
            if total_features_before > 0
            else 1.0
        )

        preprocessing_info["feature_reduction"] = {
            "total_before": total_features_before,
            "total_after": total_features_after,
            "reduction_ratio": reduction_ratio,
            "features_removed": total_features_before - total_features_after,
        }

        logger.info(f"✅ Advanced preprocessing completed")
        logger.info(f"   Applied steps: {steps_applied}")
        logger.info(
            f"   Feature reduction: {total_features_before} → {total_features_after} ({
                reduction_ratio:.3f} ratio)"
        )

        return X_processed, preprocessing_info

    except Exception as e:
        logger.error(f"Advanced preprocessing failed: {e}")
        raise


def get_optimal_preprocessing_strategy(
    config: Dict, X_list: List[np.ndarray], data_dir: str
) -> Tuple[str, Dict]:
    """
    Automatically determine optimal preprocessing strategy based on data characteristics.

    Args:
        config: Configuration dictionary
        X_list: Raw data list
        data_dir: Data directory path

    Returns:
        Tuple of (optimal_strategy_name, strategy_evaluation)
    """
    try:
        logger.info("🔍 === OPTIMAL PREPROCESSING STRATEGY SELECTION ===")

        # Analyze data characteristics
        n_subjects = X_list[0].shape[0]
        total_features = sum(X.shape[1] for X in X_list)
        n_views = len(X_list)

        # Calculate missing data percentage
        missing_percentages = []
        for X in X_list:
            missing_pct = np.isnan(X).mean() * 100
            missing_percentages.append(missing_pct)
        avg_missing_pct = np.mean(missing_percentages)

        # Check for neuroimaging views
        view_names = [f"view_{i}" for i in range(len(X_list))]
        has_imaging_views = any(
            "volume" in name.lower() or "voxel" in name.lower() for name in view_names
        )

        logger.info(f"Data characteristics:")
        logger.info(f"   Subjects: {n_subjects}")
        logger.info(f"   Total features: {total_features}")
        logger.info(f"   Views: {n_views}")
        logger.info(f"   Average missing data: {avg_missing_pct:.2f}%")
        logger.info(f"   Has imaging views: {has_imaging_views}")

        # Get available strategies
        strategies = config.get("data_validation", {}).get(
            "preprocessing_strategies", {}
        )
        strategy_names = list(strategies.keys())

        logger.info(f"Available strategies: {strategy_names}")

        # Strategy selection logic
        if avg_missing_pct > 20:
            optimal_strategy = "aggressive"  # KNN imputation for high missing data
            reason = f"High missing data ({
                avg_missing_pct:.1f}%) requires robust imputation"
        elif total_features > 5000 and has_imaging_views:
            optimal_strategy = (
                "aggressive"  # Feature selection needed for high-dimensional imaging
            )
            reason = f"High-dimensional imaging data ({total_features} features) requires feature selection"
        elif has_imaging_views:
            optimal_strategy = "clinical_focused"  # Scanner harmonization for imaging
            reason = "Neuroimaging data benefits from scanner harmonization"
        elif n_subjects < 100:
            optimal_strategy = "standard"  # Conservative for small samples
            reason = (
                f"Small sample size ({n_subjects}) requires conservative preprocessing"
            )
        else:
            optimal_strategy = "standard"  # Default
            reason = "Standard preprocessing appropriate for dataset characteristics"

        # Ensure selected strategy exists
        if optimal_strategy not in strategies:
            optimal_strategy = "standard"
            reason = f"Fallback to standard (original strategy not available)"

        strategy_evaluation = {
            "selected_strategy": optimal_strategy,
            "reason": reason,
            "data_characteristics": {
                "n_subjects": n_subjects,
                "total_features": total_features,
                "n_views": n_views,
                "avg_missing_pct": avg_missing_pct,
                "has_imaging_views": has_imaging_views,
            },
            "available_strategies": strategy_names,
        }

        logger.info(f"✅ Optimal strategy selected: {optimal_strategy}")
        logger.info(f"   Reason: {reason}")

        return optimal_strategy, strategy_evaluation

    except Exception as e:
        logger.error(f"Strategy selection failed: {e}")
        return "standard", {"selected_strategy": "standard", "error": str(e)}


def apply_preprocessing_to_pipeline(
    config: Dict,
    data_dir: str,
    auto_select_strategy: bool = True,
    preferred_strategy: str = "standard",
) -> Tuple[List[np.ndarray], Dict]:
    """
    Main integration function for the remote workstation pipeline.

    Args:
        config: Configuration dictionary
        data_dir: Data directory path
        auto_select_strategy: Whether to automatically select optimal strategy
        preferred_strategy: Preferred strategy if not auto-selecting

    Returns:
        Tuple of (processed_X_list, comprehensive_preprocessing_info)
    """
    try:
        logger.info("🚀 === PIPELINE PREPROCESSING INTEGRATION ===")

        # Load basic data for analysis
        from data.qmap_pd import load_qmap_pd

        basic_data = load_qmap_pd(data_dir=data_dir)
        X_list_raw = basic_data["X_list"]

        # Determine strategy
        if auto_select_strategy:
            strategy_name, strategy_evaluation = get_optimal_preprocessing_strategy(
                config, X_list_raw, data_dir
            )
        else:
            strategy_name = preferred_strategy
            strategy_evaluation = {
                "selected_strategy": strategy_name,
                "method": "user_specified",
            }

        # Apply preprocessing
        X_processed, preprocessing_info = get_advanced_preprocessing_data(
            config, data_dir, strategy_name
        )

        # Combine information
        comprehensive_info = {
            "preprocessing_integration": True,
            "strategy_selection": strategy_evaluation,
            "preprocessing_results": preprocessing_info,
            "data_summary": {
                "original_data": basic_data,
                "processed_shapes": [X.shape for X in X_processed],
                "view_names": basic_data.get(
                    "view_names", [f"view_{i}" for i in range(len(X_processed))]
                ),
            },
        }

        logger.info("✅ Pipeline preprocessing integration completed")
        logger.info(f"   Strategy used: {strategy_name}")
        logger.info(f"   Status: {preprocessing_info['status']}")

        return X_processed, comprehensive_info

    except Exception as e:
        logger.error(f"❌ Pipeline preprocessing integration failed: {e}")
        # Fallback to basic loading
        from data.qmap_pd import load_qmap_pd

        basic_data = load_qmap_pd(data_dir=data_dir)
        fallback_info = {
            "preprocessing_integration": False,
            "status": "fallback_basic",
            "error": str(e),
        }
        return basic_data["X_list"], fallback_info
