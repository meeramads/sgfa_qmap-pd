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

        # Get preprocessing configuration with global config as fallback
        preprocessing_strategies = config.get("data_validation", {}).get(
            "preprocessing_strategies", {}
        )
        strategy_config = preprocessing_strategies.get(preprocessing_strategy, {})

        # Use global preprocessing config as defaults
        global_preprocessing = config.get("preprocessing", {})

        # Merge global config with strategy-specific config (strategy takes precedence)
        merged_config = {**global_preprocessing, **strategy_config}

        if not strategy_config and not global_preprocessing:
            logger.warning(
                f"Preprocessing strategy '{preprocessing_strategy}' not found in config, using defaults"
            )
            merged_config = {
                "imputation_strategy": "median",
                "feature_selection_method": "variance",
                "variance_threshold": 0.01,
                "missing_threshold": 0.1,
                "enable_advanced_preprocessing": True
            }

        strategy_config = merged_config

        logger.info(f"🔧 Preprocessing configuration: {strategy_config}")

        # Log advanced preprocessing features being used
        advanced_features = []
        if strategy_config.get("feature_selection_method") != "none":
            advanced_features.append(f"feature_selection: {strategy_config.get('feature_selection_method')}")
        if strategy_config.get("variance_threshold", 0) > 0:
            advanced_features.append(f"variance_threshold: {strategy_config.get('variance_threshold')}")
        if strategy_config.get("missing_threshold", 1.0) < 1.0:
            advanced_features.append(f"missing_threshold: {strategy_config.get('missing_threshold')}")
        if strategy_config.get("enable_spatial_processing"):
            advanced_features.append("spatial_processing: enabled")

        if advanced_features:
            logger.info(f"🚀 Advanced preprocessing features: {', '.join(advanced_features)}")
        else:
            logger.info("📝 Using basic preprocessing only")

        # Load basic data first
        from data.qmap_pd import load_qmap_pd

        logger.info("Loading basic qMAP-PD data...")
        # Extract ROI selection and clinical exclusion from config if present
        select_rois = strategy_config.get("select_rois")
        exclude_clinical_features = strategy_config.get("exclude_clinical_features")

        basic_data = load_qmap_pd(
            data_dir=data_dir,
            select_rois=select_rois,
            exclude_clinical_features=exclude_clinical_features
        )
        X_list_raw = basic_data["X_list"]
        view_names = basic_data.get(
            "view_names", [f"view_{i}" for i in range(len(X_list_raw))]
        )

        logger.info(
            f"Loaded data: {len(X_list_raw)} views with shapes {[X.shape for X in X_list_raw]}"
        )

        # Determine preprocessing approach
        if preprocessing_strategy == "differentiated_imaging_clinical":
            logger.info("Using differentiated preprocessing (imaging vs clinical)")
            return _apply_differentiated_preprocessing(
                X_list_raw, view_names, strategy_config, data_dir
            )
        elif not strategy_config.get("enable_advanced_preprocessing", False):
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

        # Extract ROI selection and clinical exclusion from config if present
        select_rois = config.get("preprocessing", {}).get("select_rois")
        exclude_clinical_features = config.get("preprocessing", {}).get("exclude_clinical_features")

        basic_data = load_qmap_pd(
            data_dir=data_dir,
            select_rois=select_rois,
            exclude_clinical_features=exclude_clinical_features
        )
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
            f" Feature reduction: {total_features_before} → {total_features_after} ({ reduction_ratio:.3f} ratio)"
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

        # Strategy selection logic (uses only strategies defined in config)
        if avg_missing_pct > 20:
            optimal_strategy = "combined"  # Combined feature selection for high missing data
            reason = f"High missing data ({avg_missing_pct:.1f}%) requires robust feature selection"
        elif total_features > 5000 and has_imaging_views:
            optimal_strategy = "differentiated_imaging_clinical"  # Preserve imaging, select clinical
            reason = f"High-dimensional imaging data ({total_features} features) benefits from differentiated preprocessing"
        elif has_imaging_views:
            optimal_strategy = "differentiated_imaging_clinical"  # Best for neuroimaging + clinical
            reason = "Neuroimaging data with clinical data benefits from differentiated preprocessing"
        elif n_subjects < 100:
            optimal_strategy = "minimal"  # Conservative for small samples
            reason = (
                f"Small sample size ({n_subjects}) requires conservative preprocessing"
            )
        else:
            optimal_strategy = "standard"  # Default
            reason = "Standard preprocessing appropriate for dataset characteristics"

        # Ensure selected strategy exists, fallback if needed
        if optimal_strategy not in strategies:
            logger.warning(f"Selected strategy '{optimal_strategy}' not found in config")
            # Try fallback strategies in order of preference
            fallback_order = ["differentiated_imaging_clinical", "standard", "minimal"]
            for fallback in fallback_order:
                if fallback in strategies:
                    optimal_strategy = fallback
                    reason = f"Fallback to {fallback} (original strategy not available)"
                    break
            else:
                # If no predefined strategies exist, use first available
                if strategies:
                    optimal_strategy = list(strategies.keys())[0]
                    reason = f"Using first available strategy: {optimal_strategy}"
                else:
                    optimal_strategy = "standard"
                    reason = "Using hardcoded standard (no strategies in config)"

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

        # Extract ROI selection and clinical exclusion from config if present
        select_rois = config.get("preprocessing", {}).get("select_rois")
        exclude_clinical_features = config.get("preprocessing", {}).get("exclude_clinical_features")

        basic_data = load_qmap_pd(
            data_dir=data_dir,
            select_rois=select_rois,
            exclude_clinical_features=exclude_clinical_features
        )
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

        # Extract ROI selection and clinical exclusion from config if present
        select_rois = config.get("preprocessing", {}).get("select_rois")
        exclude_clinical_features = config.get("preprocessing", {}).get("exclude_clinical_features")

        basic_data = load_qmap_pd(
            data_dir=data_dir,
            select_rois=select_rois,
            exclude_clinical_features=exclude_clinical_features
        )
        fallback_info = {
            "preprocessing_integration": False,
            "status": "fallback_basic",
            "error": str(e),
        }
        return basic_data["X_list"], fallback_info


def _apply_differentiated_preprocessing(
    X_list: List[np.ndarray],
    view_names: List[str],
    strategy_config: Dict,
    data_dir: str,
) -> Tuple[List[np.ndarray], Dict]:
    """Apply differentiated preprocessing for imaging vs clinical data."""
    try:
        from data.preprocessing import NeuroImagingConfig, NeuroImagingPreprocessor
        from sklearn.preprocessing import RobustScaler
        from sklearn.impute import SimpleImputer

        logger.info("Applying differentiated preprocessing (imaging vs clinical)...")

        X_processed = []
        steps_applied = []

        for i, (X, view_name) in enumerate(zip(X_list, view_names)):
            logger.info(f"Processing view: {view_name} (shape: {X.shape})")

            # Determine if this is imaging or clinical data
            is_imaging = _is_imaging_view(view_name)

            if is_imaging:
                logger.info(f"  → IMAGING data: preserving spatial structure")
                # For imaging: minimal processing to preserve spatial mapping

                # Only basic scaling and imputation, NO outlier removal or feature selection
                # Imputation only
                imputer = SimpleImputer(strategy=strategy_config.get("imputation_strategy", "median"))
                X_imputed = imputer.fit_transform(X)

                # Scaling only
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X_imputed)

                X_processed.append(X_scaled)
                steps_applied.extend(["imaging_imputation", "imaging_scaling"])

                logger.info(f"  → Preserved all {X.shape[1]} imaging voxels for spatial mapping")

            else:
                logger.info(f"  → CLINICAL data: applying comprehensive preprocessing")
                # For clinical: full preprocessing including outlier removal and feature selection

                clinical_config = NeuroImagingConfig(
                    imputation_strategy=strategy_config.get("imputation_strategy", "median"),
                    feature_selection_method=strategy_config.get("feature_selection_method", "variance"),
                    variance_threshold=strategy_config.get("variance_threshold", 0.01),
                    missing_threshold=strategy_config.get("missing_threshold", 0.5),
                    enable_spatial_processing=False,  # Clinical data isn't spatial
                )

                # Apply full preprocessing to clinical data
                clinical_preprocessor = NeuroImagingPreprocessor(data_dir=data_dir, **clinical_config.to_dict())
                X_clinical_processed = clinical_preprocessor.fit_transform([X], [view_name])
                X_processed.append(X_clinical_processed[0])

                steps_applied.extend(["clinical_imputation", "clinical_scaling", "clinical_feature_selection"])

                logger.info(f"  → Clinical features: {X.shape[1]} → {X_clinical_processed[0].shape[1]}")

        # Collect preprocessing information
        preprocessing_info = {
            "status": "completed",
            "strategy": "differentiated_imaging_clinical",
            "steps_applied": list(set(steps_applied)),
            "preprocessing_type": "DifferentiatedPreprocessor",
        }

        logger.info("✅ Differentiated preprocessing completed")
        logger.info(f"   Applied steps: {preprocessing_info['steps_applied']}")

        return X_processed, preprocessing_info

    except Exception as e:
        logger.error(f"Differentiated preprocessing failed: {e}")
        logger.info("Falling back to basic preprocessing...")
        return _apply_basic_preprocessing(X_list, view_names, strategy_config)


def _is_imaging_view(view_name: str) -> bool:
    """Check if view contains imaging data."""
    imaging_keywords = [
        "imaging", "volume_", "sn", "putamen", "lentiform", "bg-all", "voxels"
    ]
    return any(keyword in view_name.lower() for keyword in imaging_keywords)
