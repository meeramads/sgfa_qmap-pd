#!/usr/bin/env python
"""
Lightweight test to verify brain mapping alignment.
Tests if factor loadings map correctly to brain positions.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_alignment():
    """Test if data dimensions and position files align correctly."""

    logger.info("🔍 Testing brain mapping data alignment...")

    # Load your latest results
    results_dir = Path("../results/qmap_pd/sparseGFA_K5_1chs_pW25_s1000_reghsZ")

    # Check if results exist
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return False

    # Load factor loadings CSV (if it exists from our recent addition)
    factor_csv = results_dir / "[1]Factor_loadings_W.csv"
    if factor_csv.exists():
        logger.info(f"Loading factor loadings from: {factor_csv}")
        W_df = pd.read_csv(factor_csv, index_col=0)
        W = W_df.values
        logger.info(f"Factor loadings shape: {W.shape}")
    else:
        logger.warning("No CSV file found, trying numpy file...")
        factor_npy = results_dir / "[1]Factor_loadings_W.npy"
        if factor_npy.exists():
            W = np.load(factor_npy)
            logger.info(f"Factor loadings shape: {W.shape}")
        else:
            logger.error("No factor loadings found!")
            return False

    # Load position files
    data_dir = Path("qMAP-PD_data")
    position_files = {
        "sn": data_dir / "position_sn_voxels.tsv",
        "putamen": data_dir / "position_putamen_voxels.tsv",
        "lentiform": data_dir / "position_lentiform_voxels.tsv"
    }

    position_dims = {}
    total_imaging_features = 0

    for roi_name, pos_file in position_files.items():
        if pos_file.exists():
            positions = pd.read_csv(pos_file, sep="\t", header=None).values.flatten()
            position_dims[roi_name] = len(positions)
            total_imaging_features += len(positions)
            logger.info(f"{roi_name}: {len(positions)} voxels")
        else:
            logger.warning(f"Position file not found: {pos_file}")

    # Check clinical data
    clinical_file = data_dir / "data_clinical" / "pd_motor_gfa_data.tsv"
    clinical_features = 0
    if clinical_file.exists():
        clinical_df = pd.read_csv(clinical_file, sep="\t")
        clinical_features = clinical_df.shape[1] - 1  # Minus ID column
        logger.info(f"Clinical features: {clinical_features}")

    total_expected_features = total_imaging_features + clinical_features

    logger.info("📊 DIMENSION COMPARISON:")
    logger.info(f"  Factor loadings W: {W.shape[0]} features × {W.shape[1]} factors")
    logger.info(f"  Expected features: {total_expected_features}")
    logger.info(f"    - Imaging: {total_imaging_features}")
    logger.info(f"    - Clinical: {clinical_features}")

    # Test alignment
    if W.shape[0] == total_expected_features:
        logger.info("✅ DIMENSIONS MATCH! Testing feature extraction...")

        # Test each ROI extraction
        feature_start = 0
        for roi_name, expected_dim in position_dims.items():
            roi_loadings = W[feature_start:feature_start + expected_dim, 0]  # Factor 1
            logger.info(f"  {roi_name}: features {feature_start} to {feature_start + expected_dim - 1}")
            logger.info(f"    Loading range: [{np.min(roi_loadings):.4f}, {np.max(roi_loadings):.4f}]")
            feature_start += expected_dim

        # Clinical features
        if clinical_features > 0:
            clinical_loadings = W[feature_start:feature_start + clinical_features, 0]
            logger.info(f"  clinical: features {feature_start} to {feature_start + clinical_features - 1}")
            logger.info(f"    Loading range: [{np.min(clinical_loadings):.4f}, {np.max(clinical_loadings):.4f}]")

        return True
    else:
        logger.error("❌ DIMENSION MISMATCH!")
        logger.error(f"  Factor matrix has {W.shape[0]} features")
        logger.error(f"  Expected {total_expected_features} features")
        logger.error("  This explains why brain maps appear as noise!")
        return False

def test_position_file_format():
    """Test position file indexing format (0-based vs 1-based)."""

    logger.info("🔍 Testing position file format...")

    data_dir = Path("qMAP-PD_data")
    pos_file = data_dir / "position_sn_voxels.tsv"

    if pos_file.exists():
        positions = pd.read_csv(pos_file, sep="\t", header=None).values.flatten()
        logger.info(f"SN positions - Min: {np.min(positions)}, Max: {np.max(positions)}")

        if np.min(positions) == 0:
            logger.info("📝 Positions appear to be 0-based (Python style)")
            return "0-based"
        elif np.min(positions) == 1:
            logger.info("📝 Positions appear to be 1-based (MATLAB style)")
            return "1-based"
        else:
            logger.warning(f"📝 Unexpected position format - min value: {np.min(positions)}")
            return "unknown"
    else:
        logger.error("Position file not found!")
        return None

if __name__ == "__main__":
    logger.info("🚀 LIGHTWEIGHT BRAIN MAPPING TEST")
    logger.info("=" * 50)

    # Test 1: Data alignment
    alignment_ok = test_data_alignment()

    logger.info("\n" + "=" * 50)

    # Test 2: Position format
    pos_format = test_position_file_format()

    logger.info("\n" + "=" * 50)
    logger.info("📋 SUMMARY:")

    if alignment_ok:
        logger.info("✅ Data dimensions align correctly")
    else:
        logger.info("❌ Data dimension mismatch - this is likely the cause of noise in .nii files")

    if pos_format:
        logger.info(f"📝 Position files are {pos_format}")

    logger.info("=" * 50)