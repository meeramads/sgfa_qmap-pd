"""
Data loading module for qMAP-PD dataset.
Handles file I/O and basic data organization with integrated preprocessing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)


def load_qmap_pd(
    data_dir: str,
    clinical_rel: str = "data_clinical/pd_motor_gfa_data.tsv",
    volumes_rel: str = "volume_matrices",
    imaging_as_single_view: bool = False,
    drop_constant_clinical: bool = True,
    id_col: str = "sid",
    select_rois: Optional[List[str]] = None,
    regress_confounds: Optional[List[str]] = None,
    drop_confounds_from_clinical: bool = True,  # NEW: drop confounds from clinical view instead of residualizing
    # Preprocessing parameters
    enable_advanced_preprocessing: bool = False,
    enable_spatial_processing: bool = False,
    imputation_strategy: str = "median",
    feature_selection_method: str = "variance",
    n_top_features: Optional[int] = None,
    missing_threshold: float = 0.1,
    variance_threshold: float = 0.0,
    target_variable: Optional[str] = None,
    cross_validate_sources: bool = False,
    optimize_preprocessing: bool = False,
    # Neuroimaging-specific parameters
    spatial_imputation: bool = True,
    roi_based_selection: bool = True,
    harmonize_scanners: bool = False,
    scanner_info_col: Optional[str] = None,
    qc_outlier_threshold: float = 3.0,
    spatial_neighbor_radius: float = 5.0,
    min_voxel_distance: float = 3.0,
    # Output parameters
    output_dir: Optional[str] = None,  # Directory for saving filtered position lookups
) -> Dict[str, Any]:
    """
    Load qMAP-PD dataset with integrated preprocessing options.

    Parameters
    ----------
    data_dir : str
        Path to data directory
    clinical_rel : str
        Relative path to clinical data file
    volumes_rel : str
        Relative path to volume matrices directory
    imaging_as_single_view : bool
        If True, concatenate all imaging data into single view
    drop_constant_clinical : bool
        Whether to drop constant clinical features
    id_col : str
        Name of ID column in clinical data
    select_rois : List[str], optional
        List of specific ROI files to load (e.g., ['volume_sn_voxels.tsv'])
        If None, loads default ROIs (sn, putamen, lentiform) or fallback (bg-all)
        Use this to analyze one ROI at a time
    regress_confounds : List[str], optional
        List of clinical features to regress out from imaging views (e.g., ['age', 'sex', 'tiv'])
        Uses residualization: removes variance explained by confounds while preserving signal
        Behavior for clinical view controlled by drop_confounds_from_clinical parameter
    drop_confounds_from_clinical : bool, default=True
        If True, completely remove confound features from the clinical view
        If False, residualize them (keep as near-zero features)
        Only applies when regress_confounds is specified
    enable_advanced_preprocessing : bool
        Enable advanced preprocessing pipeline
    enable_spatial_processing : bool
        Enable neuroimaging-specific spatial processing
    imputation_strategy : str
        Strategy for missing data imputation
    feature_selection_method : str
        Method for feature selection
    n_top_features : int, optional
        Number of top features to select
    missing_threshold : float
        Threshold for dropping features with too much missing data
    variance_threshold : float
        Threshold for variance-based feature selection
    target_variable : str, optional
        Target variable for supervised preprocessing
    cross_validate_sources : bool
        Whether to cross-validate source combinations
    optimize_preprocessing : bool
        Whether to optimize preprocessing parameters
    spatial_imputation : bool
        Use spatial neighbors for imputation (neuroimaging)
    roi_based_selection : bool
        Use ROI-based feature selection (neuroimaging)
    harmonize_scanners : bool
        Apply scanner harmonization
    scanner_info_col : str, optional
        Column name for scanner information
    qc_outlier_threshold : float
        Threshold for outlier detection in QC
    spatial_neighbor_radius : float
        Radius in mm for spatial neighbors
    min_voxel_distance : float
        Minimum distance in mm between selected voxels

    Returns
    -------
    dict with:
      - X_list:          list of (N x D_v) arrays (views in order)
      - view_names:      list of names for each view
      - feature_names:   {view_name: [feature names]}
      - subject_ids:     list[str], clinical order preserved
      - clinical:        original clinical DataFrame (with 'sid')
      - scalers:         {view_name: scaler info} for reproducibility
      - meta:            paths and config
      - preprocessing:   preprocessing results and fitted transformers (if enabled)
    """

    # === LOAD RAW DATA ===
    root = Path(data_dir)
    clinical_path = root / clinical_rel
    volumes_dir = root / volumes_rel

    # Load clinical data
    clin = pd.read_csv(clinical_path, sep="\t")
    if id_col not in clin.columns:
        raise ValueError(f"Clinical file must contain an ID column '{id_col}'.")
    clin = clin.drop_duplicates(keep="first").reset_index(drop=True)

    subject_ids = clin[id_col].astype(str).tolist()

    # Extract target variable if specified
    y = None
    scanner_info = None
    if target_variable and target_variable in clin.columns:
        y = clin[target_variable].values
        logging.info(f"Using {target_variable} as target variable")

    if scanner_info_col and scanner_info_col in clin.columns:
        scanner_info = clin[scanner_info_col].values
        logging.info(f"Using {scanner_info_col} for scanner harmonization")

    clin_X = clin.drop(columns=[id_col]).copy()

    # Handle constant clinical features
    if drop_constant_clinical and clin_X.shape[1] > 0:
        nunique = clin_X.nunique(axis=0)
        constant_features = clin_X.columns[nunique <= 1].tolist()
        if constant_features:
            logging.info(f"Dropping constant clinical features: {constant_features}")
        clin_X = clin_X.loc[:, nunique > 1]

    # Load imaging data
    roi_files_all = sorted(volumes_dir.glob("*.tsv"))
    if not roi_files_all:
        raise FileNotFoundError(f"No TSVs found in {volumes_rel}")

    name_map = {p.name: p for p in roi_files_all}

    # Determine which ROIs to load
    if select_rois is not None:
        # User specified which ROIs to load
        roi_files = []
        for roi_name in select_rois:
            if roi_name in name_map:
                roi_files.append(name_map[roi_name])
            else:
                raise FileNotFoundError(
                    f"Requested ROI '{roi_name}' not found. Available: {list(name_map.keys())}"
                )
        logging.info(f"Loading selected ROIs: {select_rois}")
    else:
        # Default behavior: load standard ROIs or fallback
        want = [
            "volume_sn_voxels.tsv",
            "volume_putamen_voxels.tsv",
            "volume_lentiform_voxels.tsv",
        ]

        if all(n in name_map for n in want):
            roi_files = [name_map[n] for n in want]  # fixed order
        else:
            bg = name_map.get("volume_bg-all_voxels.tsv")
            if not bg:
                missing = [n for n in want if n not in name_map]
                raise FileNotFoundError(
                    f"Missing ROI TSVs: {missing} and no 'volume_bg-all_voxels.tsv' fallback found."
                )
            roi_files = [bg]

    imaging_blocks: List[pd.DataFrame] = []
    block_names: List[str] = []

    for f in roi_files:
        df = pd.read_csv(f, sep="\t", header=None)
        df = df.drop_duplicates(keep="first").reset_index(drop=True)

        if df.shape[0] < len(subject_ids):
            raise ValueError(
                f"{f.name}: {df.shape[0]} rows < clinical {len(subject_ids)} after de-dup."
            )
        if df.shape[0] > len(subject_ids):
            df = df.iloc[: len(subject_ids), :].copy()

        stem = f.stem
        df.columns = [f"{stem}::v{i}" for i in range(df.shape[1])]
        imaging_blocks.append(df)
        block_names.append(stem)

    # === ORGANIZE AS VIEWS ===
    X_list_raw: List[np.ndarray] = []
    view_names: List[str] = []
    feature_names: Dict[str, List[str]] = {}

    # Process imaging data
    if imaging_as_single_view:
        imaging_df = pd.concat(imaging_blocks, axis=1)
        X_img = imaging_df.to_numpy(dtype=float)
        X_list_raw.append(X_img)
        # If only one ROI, use its name to enable position lookup; otherwise use "imaging"
        view_name = block_names[0] if len(block_names) == 1 else "imaging"
        view_names.append(view_name)
        feature_names[view_name] = imaging_df.columns.tolist()
    else:
        for name, block in zip(block_names, imaging_blocks):
            X = block.to_numpy(dtype=float)
            X_list_raw.append(X)
            view_names.append(name)
            feature_names[name] = block.columns.tolist()

    # Add clinical data
    if clin_X.shape[1] > 0:
        X_clin = clin_X.to_numpy(dtype=float)
        X_list_raw.append(X_clin)
        view_names.append("clinical")
        feature_names["clinical"] = clin_X.columns.tolist()
    else:
        # Empty clinical data
        X_list_raw.append(np.zeros((len(subject_ids), 0), dtype=float))
        view_names.append("clinical")
        feature_names["clinical"] = []

    # === APPLY PREPROCESSING ===
    scalers = {}
    preprocessing_results = {}

    if enable_advanced_preprocessing:
        logging.info("=== Applying Advanced Preprocessing ===")

        # Import preprocessing module
        try:
            from .preprocessing import (
                NeuroImagingConfig,
                PreprocessingConfig,
                cross_validate_source_combinations,
                preprocess_neuroimaging_data,
            )
        except ImportError:
            logging.error("Could not import preprocessing module")
            raise ImportError(
                "Preprocessing module not available. Make sure preprocessing.py is in the path."
            )

        # Create appropriate config
        if enable_spatial_processing:
            config = NeuroImagingConfig(
                enable_preprocessing=True,
                enable_spatial_processing=True,
                imputation_strategy=imputation_strategy,
                feature_selection_method=feature_selection_method,
                n_top_features=n_top_features,
                missing_threshold=missing_threshold,
                variance_threshold=variance_threshold,
                target_variable=target_variable,
                cross_validate_sources=cross_validate_sources,
                optimize_preprocessing=optimize_preprocessing,
                spatial_imputation=spatial_imputation,
                roi_based_selection=roi_based_selection,
                harmonize_scanners=harmonize_scanners,
                scanner_info_col=scanner_info_col,
                qc_outlier_threshold=qc_outlier_threshold,
                spatial_neighbor_radius=spatial_neighbor_radius,
                min_voxel_distance=min_voxel_distance,
            )
        else:
            config = PreprocessingConfig(
                enable_preprocessing=True,
                imputation_strategy=imputation_strategy,
                feature_selection_method=feature_selection_method,
                n_top_features=n_top_features,
                missing_threshold=missing_threshold,
                variance_threshold=variance_threshold,
                target_variable=target_variable,
                cross_validate_sources=cross_validate_sources,
                optimize_preprocessing=optimize_preprocessing,
            )

        # Apply preprocessing
        X_processed, preprocessor, metadata = preprocess_neuroimaging_data(
            X_list_raw,
            view_names,
            config,
            data_dir=data_dir,
            scanner_info=scanner_info,
            y=y,
            output_dir=output_dir,
        )

        X_list = X_processed

        # Update feature names for processed features
        feature_names_processed = {}
        for i, view_name in enumerate(view_names):
            original_features = feature_names[view_name]

            # Start with original features
            current_features = original_features.copy()

            # Apply outlier mask if available (from QC outlier detection)
            if hasattr(preprocessor, "outlier_masks_") and view_name in preprocessor.outlier_masks_:
                outlier_mask = preprocessor.outlier_masks_[view_name]
                if len(outlier_mask) == len(current_features):
                    # Filter features based on which voxels were kept
                    current_features = [
                        feat for feat, keep in zip(current_features, outlier_mask) if keep
                    ]
                    logging.info(
                        f"  Filtered {view_name} feature names by QC outlier mask: "
                        f"{len(original_features)} → {len(current_features)}"
                    )

            # Then apply feature selection indices if available
            if hasattr(preprocessor, "selected_features_"):
                # Try to get the actual selected feature names
                if f"{view_name}_variance_indices" in preprocessor.selected_features_:
                    indices = preprocessor.selected_features_[
                        f"{view_name}_variance_indices"
                    ]
                    current_features = [
                        current_features[idx]
                        for idx in indices
                        if idx < len(current_features)
                    ]
                elif f"{view_name}_roi_indices" in preprocessor.selected_features_:
                    indices = preprocessor.selected_features_[
                        f"{view_name}_roi_indices"
                    ]
                    current_features = [
                        current_features[idx]
                        for idx in indices
                        if idx < len(current_features)
                    ]

            # Verify final feature count matches data
            if len(current_features) != X_list[i].shape[1]:
                logging.warning(
                    f"  Feature name count mismatch for {view_name}: "
                    f"{len(current_features)} names vs {X_list[i].shape[1]} features. "
                    f"Using generic labels."
                )
                current_features = [
                    f"{view_name}_feature_{j}" for j in range(X_list[i].shape[1])
                ]

            feature_names_processed[view_name] = current_features

        feature_names = feature_names_processed

        # Store preprocessing results
        preprocessing_results = {
            "preprocessor": preprocessor,
            "metadata": metadata,
            "config": config,
            "original_shapes": [X.shape for X in X_list_raw],
            "processed_shapes": [X.shape for X in X_list],
        }

        # Cross-validate source combinations if requested
        if cross_validate_sources and y is not None:
            logging.info("Cross-validating source combinations...")
            cv_results = cross_validate_source_combinations(X_list, view_names, y)
            logging.info("Source combination results:")
            for combo, results in sorted(
                cv_results.items(), key=lambda x: x[1]["rmse_mean"]
            ):
                logging.info(
                    f"  {combo}: RMSE = {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}"
                )
            preprocessing_results["source_validation"] = cv_results

        # Create scalers dict for compatibility
        for i, view_name in enumerate(view_names):
            if hasattr(preprocessor, "scalers_") and view_name in preprocessor.scalers_:
                scaler = preprocessor.scalers_[view_name]
                if hasattr(scaler, "center_"):  # RobustScaler
                    scalers[view_name] = {"mu": scaler.center_, "sd": scaler.scale_}
                elif hasattr(scaler, "mean_"):  # StandardScaler
                    scalers[view_name] = {"mu": scaler.mean_, "sd": scaler.scale_}
                else:  # Basic scaler dict
                    scalers[view_name] = scaler
            else:
                # Fallback
                from .preprocessing import BasePreprocessor

                scaler_dict = BasePreprocessor.fit_basic_scaler(X_list[i])
                scalers[view_name] = scaler_dict

    else:
        # Apply basic preprocessing only
        logging.info("=== Applying Basic Preprocessing ===")

        # Import basic functions
        try:
            from .preprocessing import BasePreprocessor
        except ImportError:
            # Fallback to local basic functions if preprocessing module not available
            logging.warning(
                "Preprocessing module not available, using fallback basic scaling"
            )

            def fit_basic_scaler(
                X: np.ndarray, eps: float = 1e-8
            ) -> Dict[str, np.ndarray]:
                mu = np.nanmean(X, axis=0, keepdims=True)
                sd = np.nanstd(X, axis=0, keepdims=True)
                sd = np.where(sd < eps, 1.0, sd)
                return {"mu": mu, "sd": sd}

            def apply_basic_scaler(
                X: np.ndarray, scaler_dict: Dict[str, np.ndarray]
            ) -> np.ndarray:
                Xz = (X - scaler_dict["mu"]) / scaler_dict["sd"]
                return np.where(np.isnan(Xz), 0.0, Xz)

            BasePreprocessor = type(
                "BasePreprocessor",
                (),
                {
                    "fit_basic_scaler": staticmethod(fit_basic_scaler),
                    "apply_basic_scaler": staticmethod(apply_basic_scaler),
                },
            )

        X_list = []

        for i, (X, view_name) in enumerate(zip(X_list_raw, view_names)):
            if X.shape[1] > 0:
                scaler_dict = BasePreprocessor.fit_basic_scaler(X)
                X_processed = BasePreprocessor.apply_basic_scaler(X, scaler_dict)
                scalers[view_name] = scaler_dict
            else:
                X_processed = X
                scalers[view_name] = {"mu": np.zeros((1, 0)), "sd": np.ones((1, 0))}

            X_list.append(X_processed)

    # === METADATA ===
    meta = {
        "root": str(root),
        "clinical_path": str(clinical_path),
        "volumes_dir": str(volumes_dir),
        "id_col": id_col,
        "imaging_as_single_view": imaging_as_single_view,
        "drop_constant_clinical": drop_constant_clinical,
        "roi_files_used": [str(p) for p in roi_files],
        "N": len(subject_ids),
        "preprocessing_enabled": enable_advanced_preprocessing,
        "spatial_processing_enabled": enable_spatial_processing,
        "target_variable": target_variable,
        "scanner_info_col": scanner_info_col,
    }

    if enable_advanced_preprocessing:
        meta.update(
            {
                "imputation_strategy": imputation_strategy,
                "feature_selection_method": feature_selection_method,
                "n_top_features": n_top_features,
                "missing_threshold": missing_threshold,
                "variance_threshold": variance_threshold,
            }
        )

        if enable_spatial_processing:
            meta.update(
                {
                    "spatial_imputation": spatial_imputation,
                    "roi_based_selection": roi_based_selection,
                    "harmonize_scanners": harmonize_scanners,
                    "qc_outlier_threshold": qc_outlier_threshold,
                    "spatial_neighbor_radius": spatial_neighbor_radius,
                    "min_voxel_distance": min_voxel_distance,
                }
            )

    # === CONFOUND REGRESSION ===
    if regress_confounds:
        logging.info(f"Regressing out confounds: {regress_confounds}")

        # Extract confound variables from clinical data
        confounds_to_regress = [c for c in regress_confounds if c in clin_X.columns]
        if not confounds_to_regress:
            logging.warning(f"No valid confounds found in clinical data. Requested: {regress_confounds}")
        else:
            import numpy as np
            from sklearn.linear_model import LinearRegression

            # Get confound matrix
            confound_matrix = clin_X[confounds_to_regress].values

            # Check for missing values in confounds
            if np.isnan(confound_matrix).any():
                logging.warning("Confounds contain missing values. Imputing with mean.")
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='mean')
                confound_matrix = imputer.fit_transform(confound_matrix)

            # Add intercept
            confound_matrix_with_intercept = np.column_stack([np.ones(confound_matrix.shape[0]), confound_matrix])

            # Regress confounds from each view
            X_list_deconfounded = []
            # Start with existing feature names to preserve preprocessing updates
            updated_feature_names = feature_names.copy()

            for i, (X, view_name) in enumerate(zip(X_list, view_names)):
                # Special handling for clinical view
                if view_name == "clinical" and drop_confounds_from_clinical:
                    # DROP confound features from clinical view instead of residualizing
                    clinical_feature_list = feature_names.get("clinical", [])
                    keep_indices = [j for j, fname in enumerate(clinical_feature_list)
                                   if fname not in confounds_to_regress]

                    if len(keep_indices) < len(clinical_feature_list):
                        # Drop confound columns
                        X_deconfounded = X[:, keep_indices]
                        kept_features = [clinical_feature_list[j] for j in keep_indices]
                        updated_feature_names["clinical"] = kept_features
                        dropped_features = [f for f in clinical_feature_list if f in confounds_to_regress]
                        logging.info(f"  Dropped {len(dropped_features)} confound features from {view_name}: {dropped_features}")
                        logging.info(f"  Remaining {view_name} features: {len(kept_features)}")
                    else:
                        # No confounds to drop
                        X_deconfounded = X
                        updated_feature_names["clinical"] = clinical_feature_list
                        logging.info(f"  No confounds found in {view_name} to drop")
                else:
                    # For non-clinical views OR if drop_confounds_from_clinical=False: regress out confounds
                    lr = LinearRegression(fit_intercept=False)  # We already added intercept
                    lr.fit(confound_matrix_with_intercept, X)

                    # Get residuals (data with confounds regressed out)
                    X_deconfounded = X - lr.predict(confound_matrix_with_intercept)

                    # Keep original feature names for non-clinical views
                    if view_name in feature_names:
                        updated_feature_names[view_name] = feature_names[view_name]

                    logging.info(f"  Regressed confounds from {view_name}: {confounds_to_regress}")

                X_list_deconfounded.append(X_deconfounded)

            X_list = X_list_deconfounded
            feature_names = updated_feature_names

            # Store confound regression info in meta
            meta["confound_regression"] = {
                "confounds_regressed": confounds_to_regress,
                "confound_matrix_shape": confound_matrix.shape,
            }

    # === VERIFY FEATURE NAME TRACKING ===
    # Log final feature counts to verify feature names match data dimensions
    for i, view_name in enumerate(view_names):
        n_features_data = X_list[i].shape[1]
        n_features_names = len(feature_names.get(view_name, []))
        if n_features_data != n_features_names:
            logging.warning(
                f"Feature count mismatch for {view_name}: "
                f"{n_features_data} features in data vs {n_features_names} feature names"
            )
        else:
            logging.info(
                f"✓ Feature names verified for {view_name}: {n_features_names} features"
            )

    # === CONSTRUCT RESULT ===
    result = {
        "X_list": X_list,
        "view_names": view_names,
        "feature_names": feature_names,
        "subject_ids": subject_ids,
        "clinical": clin,
        "scalers": scalers,
        "meta": meta,
    }

    if enable_advanced_preprocessing:
        result["preprocessing"] = preprocessing_results

    if y is not None:
        result["target"] = y

    if scanner_info is not None:
        result["scanner_info"] = scanner_info

    return result


# == BACKWARD COMPATIBILITY ==


def load_qmap_pd_basic(*args, **kwargs):
    """Backward compatible version with basic preprocessing only."""
    kwargs["enable_advanced_preprocessing"] = False
    kwargs["enable_spatial_processing"] = False
    return load_qmap_pd(*args, **kwargs)


def load_qmap_pd_raw(*args, **kwargs):
    """Load raw data without any preprocessing."""
    kwargs["enable_advanced_preprocessing"] = False
    kwargs["enable_spatial_processing"] = False
    return load_qmap_pd(*args, **kwargs)
