# sgfa/data/loader_qmap_pd.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import re

_IDS = ["sid", "participant_id", "sub_id", "subject_id", "ID"]

_NUMERIC_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")

def _is_headerless_tsv(path: Path, sep: str = "\t") -> bool:
    """Heuristic: if every token in the first line is numeric → no header."""
    with open(path, "r") as f:
        first = f.readline().strip().split(sep)
    return len(first) > 0 and all(_NUMERIC_RE.match(tok) for tok in first if tok != "")

def _find_id_col(df, id_col_hint=None):
    """Existing function — keep it, but allow an explicit hint to win."""
    if id_col_hint is not None and id_col_hint in df.columns:
        return id_col_hint
    for c in _IDS:
        if c in df.columns:
            return c
    raise ValueError(f"No subject ID column found. Looked for: {', '.join(_IDS)}")

def _inject_ids_for_headerless(df: pd.DataFrame, id_col: str, sid_series: pd.Series) -> pd.DataFrame:
    """Give numeric TSVs fake voxel headers and prepend the clinical IDs."""
    df = df.copy()
    df.columns = [f"v{i}" for i in range(df.shape[1])]
    if len(sid_series) < len(df):
        raise ValueError(
            f"Imaging rows ({len(df)}) exceed clinical IDs ({len(sid_series)})."
        )
    df.insert(0, id_col, sid_series.iloc[:len(df)].astype(str).tolist())
    return df

def _read_imaging_with_id(path: Path, id_col: str, sid_series: pd.Series, sep: str = "\t") -> pd.DataFrame:
    """
    Read an imaging TSV and ensure it has an ID column named `id_col`.
    Handles headerless numeric files by injecting IDs and voxel names.
    """
    if _is_headerless_tsv(path, sep=sep):
        df = pd.read_csv(path, sep=sep, header=None)
        df = _inject_ids_for_headerless(df, id_col=id_col, sid_series=sid_series)
    else:
        df = pd.read_csv(path, sep=sep)
        # If it had a header but no ID, inject IDs
        try:
            _ = _find_id_col(df, id_col)
        except ValueError:
            df.insert(0, id_col, sid_series.iloc[:len(df)].astype(str).tolist())
    return df


def _zscore(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    Xz = (X - mu) / sd
    # Replace NaNs (e.g., all-NaN cols) with 0
    Xz = np.where(np.isnan(Xz), 0.0, Xz)
    return Xz

def load_qmap_pd(
    data_dir: str,
    clinical_rel: str = "data_clinical/pd_motor_gfa_data_cleaned.tsv",
    volumes_rel: str = "volume_matrices",
    id_col_hint: Optional[str] = None,
    imaging_as_single_view: bool = True,
) -> Dict[str, Any]:
    """
    Returns:
      {
        'X_list': [imaging_view, clinical_view] (or multiple imaging views + clinical),
        'view_names': [...],
        'feature_names': {view: [colnames...]},
        'subject_ids': [...],
        'clinical': clinical_df_aligned,
        'meta': {...},
      }
    """
    root = Path(data_dir)
    clinical_path = root / clinical_rel
    volumes_dir = root / volumes_rel

    # --- Clinical ---
    clin = pd.read_csv(clinical_path, sep="\t")
    id_col = _find_id_col(clin, id_col_hint)
    clin = clin.drop_duplicates(subset=[id_col]).set_index(id_col)
    clin.index = clin.index.astype(str)

    # --- ROI TSVs ---
    roi_files = sorted(volumes_dir.glob("*.tsv"))
    if not roi_files:
        raise FileNotFoundError(f"No TSVs found in {volumes_dir}")

    # Subject ID order we will inject into headerless imaging TSVs
    sid_order = clin.index.to_series().reset_index(drop=True)

    roi_tables: List[pd.DataFrame] = []
    for f in roi_files:
        df = _read_imaging_with_id(f, id_col=id_col, sid_series=sid_order, sep="\t")
        rid = _find_id_col(df, id_col)  # tolerate per-file ID header
        df = df.drop_duplicates(subset=[rid]).set_index(rid)
        df.index = df.index.astype(str)
        roi_tables.append(df)

    # --- Align subjects (intersection across all sources) ---
    common_ids = set(clin.index)
    for df in roi_tables:
        common_ids &= set(df.index)
    if not common_ids:
        raise ValueError("No overlapping subjects across clinical and ROI TSVs.")
    subject_ids = sorted(common_ids)

    # --- Imaging: concat all ROI files (keep provenance via prefix) ---
    imaging_blocks: List[pd.DataFrame] = []
    for f, df in zip(roi_files, roi_tables):
        block = df.loc[subject_ids]
        # drop any stray ID-like columns that slipped through
        block = block.loc[:, [c for c in block.columns if c not in _IDS]]
        block = block.copy()
        block.columns = [f"{f.stem}::{c}" for c in block.columns]
        imaging_blocks.append(block)

    X_list: List[np.ndarray] = []
    view_names: List[str] = []
    feature_names: Dict[str, List[str]] = {}

    if imaging_as_single_view:
        imaging_df = pd.concat(imaging_blocks, axis=1)
        X_img = _zscore(imaging_df.to_numpy(dtype=float))
        X_list.append(X_img)
        view_names.append("imaging")
        feature_names["imaging"] = imaging_df.columns.tolist()
    else:
        for f, block in zip(roi_files, imaging_blocks):
            Xv = _zscore(block.to_numpy(dtype=float))
            X_list.append(Xv)
            vname = f.stem
            view_names.append(vname)
            feature_names[vname] = block.columns.tolist()

    # --- Clinical (encode categoricals, drop constants) ---
    clin_aligned = clin.loc[subject_ids]
    clin_encoded = pd.get_dummies(clin_aligned, drop_first=True)
    nunique = clin_encoded.nunique(axis=0)
    clin_encoded = clin_encoded.loc[:, nunique > 1]
    if clin_encoded.shape[1]:
        X_clin = _zscore(clin_encoded.to_numpy(dtype=float))
    else:
        X_clin = np.zeros((len(subject_ids), 0), dtype=float)

    X_list.append(X_clin)
    view_names.append("clinical")
    feature_names["clinical"] = clin_encoded.columns.tolist()

    meta = {
        "root": str(root),
        "id_col": id_col,
        "roi_files": [str(f) for f in roi_files],
        "mask_nii": str(root / "mri_roi" / "average_space-qmap-384_roi-basal-ganglia.nii"),
        "reference_nii": str(root / "mri_reference" / "average_space-qmap-shoot-384_pdw-brain.nii"),
        "imaging_as_single_view": imaging_as_single_view,
    }

    return {
        "X_list": X_list,
        "view_names": view_names,
        "feature_names": feature_names,
        "subject_ids": subject_ids,
        "clinical": clin_aligned,
        "meta": meta,
    }