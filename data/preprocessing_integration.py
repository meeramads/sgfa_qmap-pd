#!/usr/bin/env python3
"""
Comprehensive preprocessing integration for remote workstation pipeline.
Replaces basic load_qmap_pd() with full NeuroImagingPreprocessor capabilities.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _save_filtered_position_lookups(
    preprocessor, view_names: List[str], data_dir: str
) -> Dict[str, str]:
    """
    Save filtered position lookup vectors for imaging views.

    This creates CSV files mapping the retained voxels to their 3D spatial coordinates,
    enabling brain remapping after voxel dropping (QC outlier removal, ROI selection).

    Parameters
    ----------
    preprocessor : NeuroImagingPreprocessor
        Fitted preprocessor with outlier_masks_ and selected_features_
    view_names : List[str]
        Names of views
    data_dir : str
        Data directory for loading original position lookups

    Returns
    -------
    Dict[str, str]
        Mapping of view names to saved position lookup file paths
    """
    from data.preprocessing import SpatialProcessingUtils

    saved_files = {}

    for view_name in view_names:
        # Only process imaging views
        if not view_name.startswith("volume_"):
            continue

        logger.info(f"Creating filtered position lookup for {view_name}...")

        # Load original position lookup
        roi_name = view_name.replace("volume_", "").replace("_voxels", "")
        original_positions = SpatialProcessingUtils.load_position_lookup(data_dir, roi_name)

        if original_positions is None:
            logger.warning(f"No original position lookup found for {view_name}, skipping")
            continue

        # Start with all voxels
        keep_mask = np.ones(len(original_positions), dtype=bool)

        # Apply QC outlier mask if available
        if hasattr(preprocessor, 'outlier_masks_') and view_name in preprocessor.outlier_masks_:
            outlier_mask = preprocessor.outlier_masks_[view_name]
            logger.info(f"  Applying QC outlier mask: {np.sum(outlier_mask)}/{len(outlier_mask)} voxels kept")
            keep_mask = keep_mask & outlier_mask[:len(keep_mask)]  # Handle size mismatch

        # Apply ROI-based selection if available
        if hasattr(preprocessor, 'selected_features_'):
            roi_indices_key = f"{view_name}_roi_indices"
            roi_mask_key = f"{view_name}_roi_mask"

            if roi_indices_key in preprocessor.selected_features_:
                # Indices-based selection
                selected_indices = preprocessor.selected_features_[roi_indices_key]
                logger.info(f"  Applying ROI selection (indices): {len(selected_indices)} voxels selected")

                # Create new mask based on selected indices
                new_mask = np.zeros(len(keep_mask), dtype=bool)
                # First apply QC mask, then select indices from kept voxels
                kept_positions = np.where(keep_mask)[0]
                for idx in selected_indices:
                    if idx < len(kept_positions):
                        new_mask[kept_positions[idx]] = True
                keep_mask = new_mask

            elif roi_mask_key in preprocessor.selected_features_:
                # Mask-based selection
                roi_mask = preprocessor.selected_features_[roi_mask_key]
                logger.info(f"  Applying ROI selection (mask): {np.sum(roi_mask)}/{len(roi_mask)} voxels kept")
                # ROI mask applies to already-QC-filtered data
                kept_positions = np.where(keep_mask)[0]
                new_mask = np.zeros(len(keep_mask), dtype=bool)
                for i, kept_idx in enumerate(kept_positions):
                    if i < len(roi_mask) and roi_mask[i]:
                        new_mask[kept_idx] = True
                keep_mask = new_mask

        # Filter position lookup to retained voxels
        filtered_positions = original_positions[keep_mask].copy()
        filtered_positions.reset_index(drop=True, inplace=True)

        # Save to data directory
        output_dir = Path(data_dir) / "preprocessed_position_lookups"
        output_dir.mkdir(exist_ok=True, parents=True)

        output_file = output_dir / f"{roi_name}_filtered_position_lookup.csv"
        filtered_positions.to_csv(output_file, index=False)

        saved_files[view_name] = str(output_file)
        logger.info(f"  Saved filtered position lookup: {output_file}")
        logger.info(f"  {len(original_positions)} → {len(filtered_positions)} voxels ({100*len(filtered_positions)/len(original_positions):.1f}% retained)")

    return saved_files


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

        # Merge global config with strategy-specific config (global takes precedence for command-line overrides)
        merged_config = {**strategy_config, **global_preprocessing}

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
        if strategy_config.get("qc_outlier_threshold"):
            advanced_features.append(f"MAD_threshold: {strategy_config.get('qc_outlier_threshold')}")
        if strategy_config.get("enable_pca"):
            pca_var = strategy_config.get("pca_variance_threshold", 0.95)
            pca_comp = strategy_config.get("pca_n_components")
            if pca_comp:
                advanced_features.append(f"PCA: {pca_comp} components")
            else:
                advanced_features.append(f"PCA: {pca_var*100:.0f}% variance")

        if advanced_features:
            logger.info(f"🚀 Advanced preprocessing features: {', '.join(advanced_features)}")
        else:
            logger.info("📝 Using basic preprocessing only")

        # Check dataset type
        dataset = config.get("data", {}).get("dataset", "qmap_pd")

        # Load data based on dataset type
        if dataset in {"synthetic", "toy"}:
            # Generate synthetic data
            from data.synthetic import generate_synthetic_data

            logger.info("Generating synthetic data...")
            basic_data = generate_synthetic_data(
                num_sources=config.get("data", {}).get("num_sources", 3),
                K=config.get("data", {}).get("K_true", 3),
                percW=config.get("data", {}).get("percW_true", 33.0),
            )
            logger.info("Generated synthetic data for testing")
        else:
            # Load qMAP-PD data
            from data.qmap_pd import load_qmap_pd

            logger.info("Loading basic qMAP-PD data...")
            # Extract ROI selection and confound regression from config if present
            select_rois = strategy_config.get("select_rois")
            regress_confounds = strategy_config.get("regress_confounds")
            drop_confounds_from_clinical = strategy_config.get("drop_confounds_from_clinical", True)

            basic_data = load_qmap_pd(
                data_dir=data_dir,
                select_rois=select_rois,
                regress_confounds=regress_confounds,
                drop_confounds_from_clinical=drop_confounds_from_clinical
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

        # Extract ROI selection and confound regression from config if present
        select_rois = config.get("preprocessing", {}).get("select_rois")
        regress_confounds = config.get("preprocessing", {}).get("regress_confounds")
        drop_confounds_from_clinical = config.get("preprocessing", {}).get("drop_confounds_from_clinical", True)

        basic_data = load_qmap_pd(
            data_dir=data_dir,
            select_rois=select_rois,
            regress_confounds=regress_confounds,
            drop_confounds_from_clinical=drop_confounds_from_clinical
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
            # PCA parameters
            enable_pca=strategy_config.get("enable_pca", False),
            pca_n_components=strategy_config.get("pca_n_components", None),
            pca_variance_threshold=strategy_config.get("pca_variance_threshold", 0.95),
            pca_whiten=strategy_config.get("pca_whiten", False),
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
            "preprocessor": preprocessor,  # Store the preprocessor for PCA inverse transform
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
                "qc_outlier_threshold": preprocessing_config.qc_outlier_threshold,
                "pca_enabled": preprocessing_config.enable_pca,
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

        # Save filtered position lookups for imaging views (for brain remapping)
        if preprocessing_config.enable_spatial_processing:
            saved_positions = _save_filtered_position_lookups(
                preprocessor, view_names, data_dir
            )
            if saved_positions:
                preprocessing_info["filtered_position_lookups"] = saved_positions
                logger.info(f"   Saved {len(saved_positions)} filtered position lookup files")

        # Add PCA information if PCA was used
        if preprocessing_config.enable_pca:
            pca_info = {}
            for view_name in view_names:
                view_pca_info = preprocessor.get_pca_info(view_name)
                if view_pca_info:
                    pca_info[view_name] = view_pca_info
                    logger.info(
                        f"   PCA for {view_name}: {view_pca_info['n_features_original']} → "
                        f"{view_pca_info['n_components']} components "
                        f"({view_pca_info['total_variance']:.2%} variance)"
                    )
            preprocessing_info["pca_info"] = pca_info

        logger.info(f"✅ Advanced preprocessing completed")
        logger.info(f"   Applied steps: {steps_applied}")
        logger.info(
            f"   Feature reduction: {total_features_before} → {total_features_after} ({reduction_ratio:.3f} ratio)"
        )

        # Log key preprocessing parameters
        if preprocessing_config.enable_spatial_processing:
            logger.info(
                f"   MAD outlier threshold: {preprocessing_config.qc_outlier_threshold:.1f}"
            )
        if preprocessing_config.enable_pca:
            if preprocessing_config.pca_n_components:
                logger.info(
                    f"   PCA: fixed {preprocessing_config.pca_n_components} components"
                )
            else:
                logger.info(
                    f"   PCA: {preprocessing_config.pca_variance_threshold*100:.0f}% variance threshold"
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

        # Check dataset type
        dataset = config.get("data", {}).get("dataset", "qmap_pd")
        logger.info(f"Dataset type: {dataset}")

        # Load data based on dataset type
        if dataset in {"synthetic", "toy"}:
            # Generate synthetic data
            from data.synthetic import generate_synthetic_data

            basic_data = generate_synthetic_data(
                num_sources=config.get("data", {}).get("num_sources", 3),
                K=config.get("data", {}).get("K_true", 3),
                percW=config.get("data", {}).get("percW_true", 33.0),
            )
            logger.info("Generated synthetic data for testing")
        else:
            # Load qMAP-PD data
            from data.qmap_pd import load_qmap_pd

            # Extract ROI selection and confound regression from config if present
            select_rois = config.get("preprocessing", {}).get("select_rois")
            regress_confounds = config.get("preprocessing", {}).get("regress_confounds")
            drop_confounds_from_clinical = config.get("preprocessing", {}).get("drop_confounds_from_clinical", True)

            basic_data = load_qmap_pd(
                data_dir=data_dir,
                select_rois=select_rois,
                regress_confounds=regress_confounds,
                drop_confounds_from_clinical=drop_confounds_from_clinical
            )
            logger.info("Loaded qMAP-PD data")

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

        # Extract ROI selection and confound regression from config if present
        select_rois = config.get("preprocessing", {}).get("select_rois")
        regress_confounds = config.get("preprocessing", {}).get("regress_confounds")
        drop_confounds_from_clinical = config.get("preprocessing", {}).get("drop_confounds_from_clinical", True)

        basic_data = load_qmap_pd(
            data_dir=data_dir,
            select_rois=select_rois,
            regress_confounds=regress_confounds,
            drop_confounds_from_clinical=drop_confounds_from_clinical
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
                logger.info(f"  → IMAGING data: applying neuroimaging-aware preprocessing")
                # For imaging: apply full preprocessing WITH feature selection
                # Spatial structure can be reconstructed via position lookups if needed

                imaging_config = NeuroImagingConfig(
                    imputation_strategy=strategy_config.get("imputation_strategy", "median"),
                    feature_selection_method=strategy_config.get("feature_selection_method", "variance"),
                    variance_threshold=strategy_config.get("variance_threshold", 0.01),
                    missing_threshold=strategy_config.get("missing_threshold", 0.5),
                    enable_spatial_processing=strategy_config.get("enable_spatial_processing", False),
                    roi_based_selection=strategy_config.get("roi_based_selection", False),
                    spatial_imputation=strategy_config.get("spatial_imputation", False),
                    enable_pca=strategy_config.get("enable_pca", False),
                    pca_n_components=strategy_config.get("pca_n_components", None),
                    pca_variance_threshold=strategy_config.get("pca_variance_threshold", 0.95),
                    pca_whiten=strategy_config.get("pca_whiten", False),
                )

                # Apply neuroimaging preprocessing including feature selection
                imaging_preprocessor = NeuroImagingPreprocessor(data_dir=data_dir, **imaging_config.to_dict())
                X_imaging_processed = imaging_preprocessor.fit_transform([X], [view_name])
                X_processed.append(X_imaging_processed[0])

                steps_applied.extend(["imaging_imputation", "imaging_scaling", "imaging_feature_selection"])

                logger.info(f"  → Imaging features: {X.shape[1]} → {X_imaging_processed[0].shape[1]} ({100*(1-X_imaging_processed[0].shape[1]/X.shape[1]):.1f}% reduction)")

            else:
                logger.info(f"  → CLINICAL data: applying comprehensive preprocessing")
                # For clinical: full preprocessing including outlier removal and feature selection

                clinical_config = NeuroImagingConfig(
                    imputation_strategy=strategy_config.get("imputation_strategy", "median"),
                    feature_selection_method=strategy_config.get("feature_selection_method", "variance"),
                    variance_threshold=strategy_config.get("variance_threshold", 0.01),
                    missing_threshold=strategy_config.get("missing_threshold", 0.5),
                    enable_spatial_processing=False,  # Clinical data isn't spatial
                    enable_pca=strategy_config.get("enable_pca", False),
                    pca_n_components=strategy_config.get("pca_n_components", None),
                    pca_variance_threshold=strategy_config.get("pca_variance_threshold", 0.95),
                    pca_whiten=strategy_config.get("pca_whiten", False),
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
            "strategy": "full_preprocessing_all_views",
            "steps_applied": list(set(steps_applied)),
            "preprocessing_type": "NeuroImagingPreprocessor",
            "imaging_feature_selection": "enabled",
            "clinical_feature_selection": "enabled",
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
