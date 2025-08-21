# sgfa/data/loader_qmap_pd.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

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
    clinical_rel: str = "data_clinical/pd_motor_gfa_data_cleaned.tsv",
    volumes_rel: str = "volume_matrices",
    imaging_as_single_view: bool = False,   # keep ROIs separate
    drop_constant_clinical: bool = True,
    id_col: str = "sid",
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
    clin_X = clin.drop(columns=[id_col]).copy()

    if drop_constant_clinical and clin_X.shape[1] > 0:
        nunique = clin_X.nunique(axis=0)
        clin_X = clin_X.loc[:, nunique > 1]

    if clin_X.shape[1] > 0:
        mu_c, sd_c = _fit_scaler(clin_X.to_numpy(dtype=float))
        X_clin = _transform_with(mu_c, sd_c, clin_X.to_numpy(dtype=float))
    else:
        X_clin = np.zeros((len(subject_ids), 0), dtype=float)
        mu_c = np.zeros((1, 0), dtype=float)
        sd_c = np.ones((1, 0), dtype=float)

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
    X_list: List[np.ndarray] = []
    view_names: List[str] = []
    feature_names: Dict[str, List[str]] = {}
    scalers: Dict[str, Dict[str, np.ndarray]] = {}

    if imaging_as_single_view:
        imaging_df = pd.concat(imaging_blocks, axis=1)
        X = imaging_df.to_numpy(dtype=float)
        mu, sd = _fit_scaler(X)
        X = _transform_with(mu, sd, X)
        X_list.append(X)
        view_names.append("imaging")
        feature_names["imaging"] = imaging_df.columns.tolist()
        scalers["imaging"] = {"mu": mu, "sd": sd}
    else:
        for name, block in zip(block_names, imaging_blocks):
            X = block.to_numpy(dtype=float)
            mu, sd = _fit_scaler(X)
            X = _transform_with(mu, sd, X)
            X_list.append(X)
            view_names.append(name)
            feature_names[name] = block.columns.tolist()
            scalers[name] = {"mu": mu, "sd": sd}

    # clinical last
    X_list.append(X_clin)
    view_names.append("clinical")
    feature_names["clinical"] = clin_X.columns.tolist()
    scalers["clinical"] = {"mu": mu_c, "sd": sd_c}

    meta = {
        "root": str(root),
        "clinical_path": str(clinical_path),
        "volumes_dir": str(volumes_dir),
        "id_col": id_col,
        "imaging_as_single_view": imaging_as_single_view,
        "drop_constant_clinical": drop_constant_clinical,
        "roi_files_used": [str(p) for p in roi_files],
        "N": len(subject_ids),
    }

    return {
        "X_list": X_list,
        "view_names": view_names,
        "feature_names": feature_names,
        "subject_ids": subject_ids,
        "clinical": clin,
        "scalers": scalers,
        "meta": meta,
    }
