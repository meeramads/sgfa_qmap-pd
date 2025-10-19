#!/usr/bin/env python3
"""
Create filtered SN position lookup based on standard preprocessing.

This applies the typical MAD-based QC outlier removal to create a filtered
position lookup that matches the preprocessed data dimensions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("./qMAP-PD_data")
QC_THRESHOLD = 3.0  # MAD threshold (matches your config)

def load_sn_data():
    """Load SN imaging data."""
    sn_file = DATA_DIR / "volume_matrices" / "volume_sn_voxels.tsv"
    logger.info(f"Loading SN data from {sn_file}")
    data = pd.read_csv(sn_file, sep='\t', index_col=0)
    logger.info(f"  Shape: {data.shape} (subjects × voxels)")
    return data

def load_position_lookup():
    """Load original SN position lookup."""
    pos_file = DATA_DIR / "position_lookup" / "position_sn_voxels.tsv"
    logger.info(f"Loading position lookup from {pos_file}")
    positions = pd.read_csv(pos_file, sep='\t', header=None)
    logger.info(f"  Shape: {positions.shape}")

    # Check if it has coordinates or just indices
    if positions.shape[1] == 1:
        logger.warning("Position file contains only indices, not x/y/z coordinates!")
        logger.warning("You may need to convert voxel indices to MNI coordinates")
        return positions
    else:
        # Assume columns are x, y, z
        positions.columns = ['x', 'y', 'z'] if positions.shape[1] == 3 else list(range(positions.shape[1]))
        return positions

def apply_mad_filtering(data, threshold=3.0):
    """Apply MAD-based outlier detection (voxel-level QC)."""
    logger.info(f"Applying MAD filtering with threshold={threshold}")

    n_subjects, n_voxels = data.shape

    # For each voxel, calculate MAD across subjects
    voxel_medians = data.median(axis=0)
    voxel_mads = (data - voxel_medians).abs().median(axis=0)

    # Identify outlier voxels (those with MAD > threshold * median MAD)
    median_mad = voxel_mads.median()
    outlier_threshold = threshold * median_mad

    # Keep voxels with MAD below threshold
    keep_mask = voxel_mads <= outlier_threshold

    logger.info(f"  Voxels before filtering: {n_voxels}")
    logger.info(f"  Voxels after filtering: {keep_mask.sum()}")
    logger.info(f"  Percentage retained: {100 * keep_mask.sum() / n_voxels:.1f}%")

    return keep_mask

def create_filtered_positions():
    """Create filtered position lookup."""

    # Load data
    sn_data = load_sn_data()
    positions = load_position_lookup()

    # Verify dimensions match
    if len(positions) != sn_data.shape[1]:
        logger.error(f"Dimension mismatch: {len(positions)} positions vs {sn_data.shape[1]} voxels")
        logger.error("Cannot create filtered position lookup")
        return None

    # Apply MAD filtering
    keep_mask = apply_mad_filtering(sn_data, threshold=QC_THRESHOLD)

    # Filter positions
    filtered_positions = positions[keep_mask].copy()
    filtered_positions.reset_index(drop=True, inplace=True)

    logger.info(f"Filtered positions shape: {filtered_positions.shape}")

    # Save filtered positions
    output_dir = DATA_DIR / "preprocessed_position_lookups"
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / "sn_filtered_position_lookup.csv"
    filtered_positions.to_csv(output_file, index=False)

    logger.info(f"✅ Saved filtered position lookup to {output_file}")
    logger.info(f"   Original: {len(positions)} voxels")
    logger.info(f"   Filtered: {len(filtered_positions)} voxels")
    logger.info(f"   Retention: {100 * len(filtered_positions) / len(positions):.1f}%")

    return output_file

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Creating filtered SN position lookup")
    logger.info("=" * 80)

    result = create_filtered_positions()

    if result:
        logger.info("=" * 80)
        logger.info(f"✅ SUCCESS: {result}")
        logger.info("=" * 80)
    else:
        logger.error("❌ FAILED to create filtered position lookup")
