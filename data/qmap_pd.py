"""
Data loading module for qMAP-PD dataset.
Handles file I/O and basic data organization with integrated preprocessing.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def load_qmap_pd(
    data_dir: str,
    clinical_rel: str = "data_clinical/pd_motor_gfa_data.tsv",
    volumes_rel: str = "volume_matrices",
    imaging_as_single_view: bool = False,
    drop_constant_clinical: bool = True,
    id_col: str = "sid",
    # Preprocessing parameters
    enable_advanced_preprocessing: bool = False,
    enable_spatial_processing: bool = False,
    imputation_strategy: str = 'median',
    feature_selection_method: str = 'variance',
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
    want = ["volume_sn_voxels.tsv", "volume_putamen_voxels.tsv", "volume_lentiform_voxels.tsv"]

    if all(n in name_map for n in want):
        roi_files = [name_map[n] for n in want]  # fixed order
    else:
        bg = name_map.get("volume_bg-all_voxels.tsv")
        if not bg:
            missing = [n for n in want if n not in name_map]
            raise FileNotFoundError(f"Missing ROI TSVs: {missing} and no 'volume_bg-all_voxels.tsv' fallback found.")
        roi_files = [bg]

    imaging_blocks: List[pd.DataFrame] = []
    block_names: List[str] = []

    for f in roi_files:
        df = pd.read_csv(f, sep="\t", header=None)
        df = df.drop_duplicates(keep="first").reset_index(drop=True)

        if df.shape[0] < len(subject_ids):
            raise ValueError(f"{f.name}: {df.shape[0]} rows < clinical {len(subject_ids)} after de-dup.")
        if df.shape[0] > len(subject_ids):
            df = df.iloc[:len(subject_ids), :].copy()

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
        view_names.append("imaging")
        feature_names["imaging"] = imaging_df.columns.tolist()
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
                NeuroImagingConfig, PreprocessingConfig, 
                preprocess_neuroimaging_data, cross_validate_source_combinations
            )
        except ImportError:
            logging.error("Could not import preprocessing module")
            raise ImportError("Preprocessing module not available. Make sure preprocessing.py is in the path.")
        
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
            X_list_raw, view_names, config,
            data_dir=data_dir,
            scanner_info=scanner_info,
            y=y
        )
        
        X_list = X_processed
        
        # Update feature names for processed features
        feature_names_processed = {}
        for i, view_name in enumerate(view_names):
            original_features = feature_names[view_name]
            if hasattr(preprocessor, 'selected_features_'):
                # Try to get the actual selected feature names
                if f"{view_name}_variance_indices" in preprocessor.selected_features_:
                    indices = preprocessor.selected_features_[f"{view_name}_variance_indices"]
                    feature_names_processed[view_name] = [original_features[idx] for idx in indices if idx < len(original_features)]
                elif f"{view_name}_roi_indices" in preprocessor.selected_features_:
                    indices = preprocessor.selected_features_[f"{view_name}_roi_indices"]
                    feature_names_processed[view_name] = [original_features[idx] for idx in indices if idx < len(original_features)]
                elif X_list[i].shape[1] < len(original_features):
                    # Features were filtered but we don't have exact mapping
                    feature_names_processed[view_name] = [f"{view_name}_feature_{j}" for j in range(X_list[i].shape[1])]
                else:
                    feature_names_processed[view_name] = original_features[:X_list[i].shape[1]]
            else:
                feature_names_processed[view_name] = original_features[:X_list[i].shape[1]]
        
        feature_names = feature_names_processed
        
        # Store preprocessing results
        preprocessing_results = {
            'preprocessor': preprocessor,
            'metadata': metadata,
            'config': config,
            'original_shapes': [X.shape for X in X_list_raw],
            'processed_shapes': [X.shape for X in X_list],
        }
        
        # Cross-validate source combinations if requested
        if cross_validate_sources and y is not None:
            logging.info("Cross-validating source combinations...")
            cv_results = cross_validate_source_combinations(X_list, view_names, y)
            logging.info("Source combination results:")
            for combo, results in sorted(cv_results.items(), key=lambda x: x[1]['rmse_mean']):
                logging.info(f"  {combo}: RMSE = {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
            preprocessing_results['source_validation'] = cv_results
        
        # Create scalers dict for compatibility
        for i, view_name in enumerate(view_names):
            if hasattr(preprocessor, 'scalers_') and view_name in preprocessor.scalers_:
                scaler = preprocessor.scalers_[view_name]
                if hasattr(scaler, 'center_'):  # RobustScaler
                    scalers[view_name] = {
                        'mu': scaler.center_,
                        'sd': scaler.scale_
                    }
                elif hasattr(scaler, 'mean_'):  # StandardScaler
                    scalers[view_name] = {
                        'mu': scaler.mean_,
                        'sd': scaler.scale_
                    }
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
            logging.warning("Preprocessing module not available, using fallback basic scaling")
            
            def fit_basic_scaler(X: np.ndarray, eps: float = 1e-8) -> Dict[str, np.ndarray]:
                mu = np.nanmean(X, axis=0, keepdims=True)
                sd = np.nanstd(X, axis=0, keepdims=True)
                sd = np.where(sd < eps, 1.0, sd)
                return {"mu": mu, "sd": sd}
            
            def apply_basic_scaler(X: np.ndarray, scaler_dict: Dict[str, np.ndarray]) -> np.ndarray:
                Xz = (X - scaler_dict["mu"]) / scaler_dict["sd"]
                return np.where(np.isnan(Xz), 0.0, Xz)
            
            BasePreprocessor = type('BasePreprocessor', (), {
                'fit_basic_scaler': staticmethod(fit_basic_scaler),
                'apply_basic_scaler': staticmethod(apply_basic_scaler)
            })
        
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
        meta.update({
            "imputation_strategy": imputation_strategy,
            "feature_selection_method": feature_selection_method,
            "n_top_features": n_top_features,
            "missing_threshold": missing_threshold,
            "variance_threshold": variance_threshold,
        })
        
        if enable_spatial_processing:
            meta.update({
                "spatial_imputation": spatial_imputation,
                "roi_based_selection": roi_based_selection,
                "harmonize_scanners": harmonize_scanners,
                "qc_outlier_threshold": qc_outlier_threshold,
                "spatial_neighbor_radius": spatial_neighbor_radius,
                "min_voxel_distance": min_voxel_distance,
            })

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
    kwargs['enable_advanced_preprocessing'] = False
    kwargs['enable_spatial_processing'] = False
    return load_qmap_pd(*args, **kwargs)

def load_qmap_pd_raw(*args, **kwargs):
    """Load raw data without any preprocessing."""
    kwargs['enable_advanced_preprocessing'] = False
    kwargs['enable_spatial_processing'] = False
    return load_qmap_pd(*args, **kwargs)