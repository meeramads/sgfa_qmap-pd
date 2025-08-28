from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import logging

# Import preprocessing module
try:
    from preprocessing import AdvancedPreprocessor, cross_validate_source_combinations
except ImportError:
    logging.warning("preprocessing module not found - using basic preprocessing only")
    AdvancedPreprocessor = None

# ---------- scaling helpers ----------
def _fit_scaler(X: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return mu, sd

def _transform_with(mu: np.ndarray, sd: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xz = (X - mu) / sd
    return np.where(np.isnan(Xz), 0.0, Xz)

# ---------- main loader ----------
def load_qmap_pd(
    data_dir: str,
    clinical_rel: str = "data_clinical/pd_motor_gfa_data.tsv",
    volumes_rel: str = "volume_matrices",
    imaging_as_single_view: bool = False,
    drop_constant_clinical: bool = True,
    id_col: str = "sid",
    # Additional preprocessing parameters
    enable_advanced_preprocessing: bool = False,
    imputation_strategy: str = 'median',
    feature_selection_method: str = 'variance',
    n_top_features: Optional[int] = None,
    missing_threshold: float = 0.1,
    variance_threshold: float = 0.0,
    target_variable: Optional[str] = None,
    cross_validate_sources: bool = False,
    optimize_preprocessing: bool = False,
) -> Dict[str, Any]:
    """
    Returns
    -------
    dict with:
      - X_list:          list of (N x D_v) arrays (views in order)
      - view_names:      list of names for each view
      - feature_names:   {view_name: [feature names]}
      - subject_ids:     list[str], clinical order preserved
      - clinical:        original clinical DataFrame (with 'sid')
      - scalers:         {view_name: {'mu': mu, 'sd': sd}} for reproducibility
      - meta:            paths and config
      - preprocessing:   preprocessing results and fitted transformers (if enabled)
    """
    root = Path(data_dir)
    clinical_path = root / clinical_rel
    volumes_dir = root / volumes_rel

    # --- clinical: has header, contains 'sid' ---
    clin = pd.read_csv(clinical_path, sep="\t")
    if id_col not in clin.columns:
        raise ValueError(f"Clinical file must contain an ID column '{id_col}'.")
    clin = clin.drop_duplicates(keep="first").reset_index(drop=True)

    subject_ids = clin[id_col].astype(str).tolist()
    
    # Extract target variable if specified for supervised preprocessing
    y = None
    if target_variable and target_variable in clin.columns:
        y = clin[target_variable].values
        logging.info(f"Using {target_variable} as target variable for supervised preprocessing")
    
    clin_X = clin.drop(columns=[id_col]).copy()

    # Handle constant clinical features
    if drop_constant_clinical and clin_X.shape[1] > 0:
        nunique = clin_X.nunique(axis=0)
        constant_features = clin_X.columns[nunique <= 1].tolist()
        if constant_features:
            logging.info(f"Dropping constant clinical features: {constant_features}")
        clin_X = clin_X.loc[:, nunique > 1]

    # --- imaging: headerless numeric; prefer per-ROI over bg-all ---
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

    # --- package as views ---
    X_list_raw: List[np.ndarray] = []
    view_names: List[str] = []
    feature_names: Dict[str, List[str]] = {}
    scalers: Dict[str, Dict[str, np.ndarray]] = {}

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

    # --- Apply advanced preprocessing if enabled ---
    preprocessing_results = {}
    
    if enable_advanced_preprocessing and AdvancedPreprocessor is not None:
        logging.info("=== Applying Advanced Preprocessing ===")
        
        # Initialize preprocessor
        preprocessor = AdvancedPreprocessor(
            missing_threshold=missing_threshold,
            variance_threshold=variance_threshold,
            n_top_features=n_top_features,
            imputation_strategy=imputation_strategy,
            feature_selection_method=feature_selection_method
        )
        
        # Optimize preprocessing parameters if requested
        if optimize_preprocessing:
            logging.info("Optimizing preprocessing parameters...")
            from preprocessing import optimize_preprocessing_parameters
            optimization_results = optimize_preprocessing_parameters(
                X_list_raw, view_names, y
            )
            logging.info(f"Best preprocessing params: {optimization_results['best_params']}")
            
            # Update preprocessor with optimal parameters
            preprocessor = AdvancedPreprocessor(**optimization_results['best_params'])
            preprocessing_results['optimization'] = optimization_results
        
        # Apply preprocessing
        X_list = preprocessor.fit_transform(X_list_raw, view_names, y)
        
        # Cross-validate source combinations if requested
        if cross_validate_sources and y is not None:
            logging.info("Cross-validating source combinations...")
            cv_results = cross_validate_source_combinations(X_list, view_names, y)
            logging.info("Source combination results:")
            for combo, results in sorted(cv_results.items(), key=lambda x: x[1]['rmse_mean']):
                logging.info(f"  {combo}: RMSE = {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
            preprocessing_results['source_validation'] = cv_results
        
        # Store preprocessing information
        preprocessing_results.update({
            'preprocessor': preprocessor,
            'original_shapes': [X.shape for X in X_list_raw],
            'processed_shapes': [X.shape for X in X_list],
            'feature_reduction': {
                view_names[i]: {
                    'original': X_list_raw[i].shape[1],
                    'processed': X_list[i].shape[1],
                    'reduction_ratio': X_list[i].shape[1] / max(1, X_list_raw[i].shape[1])
                }
                for i in range(len(view_names))
            }
        })
        
        # Update feature names for processed features
        feature_names_processed = {}
        for i, view_name in enumerate(view_names):
            original_features = feature_names[view_name]
            if f"{view_name}_variance_indices" in preprocessor.selected_features_:
                # Features selected by indices
                indices = preprocessor.selected_features_[f"{view_name}_variance_indices"]
                feature_names_processed[view_name] = [original_features[idx] for idx in indices if idx < len(original_features)]
            elif X_list[i].shape[1] < len(original_features):
                # Features were filtered but we don't have exact mapping
                feature_names_processed[view_name] = [f"{view_name}_feature_{j}" for j in range(X_list[i].shape[1])]
            else:
                feature_names_processed[view_name] = original_features[:X_list[i].shape[1]]
        
        feature_names = feature_names_processed
        
        # Create scalers dict for compatibility
        for i, view_name in enumerate(view_names):
            if view_name in preprocessor.scalers_:
                scaler = preprocessor.scalers_[view_name]
                scalers[view_name] = {
                    'mu': scaler.center_ if hasattr(scaler, 'center_') else scaler.mean_,
                    'sd': scaler.scale_
                }
            else:
                # Fallback
                mu, sd = _fit_scaler(X_list[i])
                scalers[view_name] = {'mu': mu, 'sd': sd}
        
    else:
        # Apply basic preprocessing (original method)
        logging.info("=== Applying Basic Preprocessing ===")
        X_list = []
        
        for i, (X, view_name) in enumerate(zip(X_list_raw, view_names)):
            if X.shape[1] > 0:
                mu, sd = _fit_scaler(X)
                X_processed = _transform_with(mu, sd, X)
            else:
                X_processed = X
                mu = np.zeros((1, 0), dtype=float)
                sd = np.ones((1, 0), dtype=float)
            
            X_list.append(X_processed)
            scalers[view_name] = {"mu": mu, "sd": sd}

    # --- Metadata ---
    meta = {
        "root": str(root),
        "clinical_path": str(clinical_path),
        "volumes_dir": str(volumes_dir),
        "id_col": id_col,
        "imaging_as_single_view": imaging_as_single_view,
        "drop_constant_clinical": drop_constant_clinical,
        "roi_files_used": [str(p) for p in roi_files],
        "N": len(subject_ids),
        "advanced_preprocessing": enable_advanced_preprocessing,
        "target_variable": target_variable,
    }
    
    if enable_advanced_preprocessing:
        meta.update({
            "imputation_strategy": imputation_strategy,
            "feature_selection_method": feature_selection_method,
            "n_top_features": n_top_features,
            "missing_threshold": missing_threshold,
            "variance_threshold": variance_threshold,
        })

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
    
    return result

# Backward compatibility
def load_qmap_pd_basic(*args, **kwargs):
    """Backward compatible version with basic preprocessing only."""
    kwargs['enable_advanced_preprocessing'] = False
    return load_qmap_pd(*args, **kwargs)