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
RESULTS_DIR = Path("./results")
QC_THRESHOLD = 3.0  # MAD threshold (matches your config)

def load_sn_data():
    """Load SN imaging data and return both data and mask of kept columns."""
    sn_file = DATA_DIR / "volume_matrices" / "volume_sn_voxels.tsv"
    logger.info(f"Loading SN data from {sn_file}")
    # Load without index column - it's just raw data values
    data = pd.read_csv(sn_file, sep='\t', header=None)
    logger.info(f"  Shape before dedup: {data.shape}")

    # Drop duplicate subjects (rows) - matches qmap_pd.py line 202
    data = data.drop_duplicates(keep="first").reset_index(drop=True)
    logger.info(f"  Shape after row dedup: {data.shape}")

    # Drop duplicate voxels (columns) and track which were kept
    original_cols = data.shape[1]
    data_T = data.T
    # Get indices of non-duplicate columns
    _, unique_indices = np.unique(data_T.values, axis=0, return_index=True)
    unique_indices = sorted(unique_indices)  # Keep original order

    # Create mask of kept columns
    column_keep_mask = np.zeros(original_cols, dtype=bool)
    column_keep_mask[unique_indices] = True

    # Filter data to keep only unique columns
    data = data.iloc[:, column_keep_mask]

    if data.shape[1] < original_cols:
        logger.info(f"  Removed {original_cols - data.shape[1]} duplicate voxel columns")
    logger.info(f"  Final shape: {data.shape} (subjects × voxels)")

    return data, column_keep_mask

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
    """
    Apply MAD-based outlier detection (voxel-level QC).

    For each voxel, check if ANY subject has an extreme value (>threshold MAD scores).
    If so, mark that voxel as an outlier and remove it.

    This matches the logic in data/preprocessing.py:detect_outlier_voxels()
    """
    logger.info(f"Applying MAD filtering with threshold={threshold}")

    n_subjects, n_voxels = data.shape
    outlier_mask = np.zeros(n_voxels, dtype=bool)

    # For each voxel
    for i in range(n_voxels):
        voxel_data = data.iloc[:, i].values

        # Remove NaN values
        clean_data = voxel_data[~np.isnan(voxel_data)]

        if len(clean_data) > 0:
            # Calculate median and MAD for this voxel across subjects
            median = np.median(clean_data)
            mad = np.median(np.abs(clean_data - median))

            # Check for extreme outliers using MAD
            if mad > 0:  # Avoid division by zero
                # Calculate MAD scores for all subjects at this voxel
                mad_scores = np.abs(clean_data - median) / (mad * 1.4826)
                max_mad_score = np.max(mad_scores)

                # If ANY subject has extreme value, mark voxel as outlier
                if max_mad_score > threshold:
                    outlier_mask[i] = True

    # Keep mask is inverse of outlier mask
    keep_mask = ~outlier_mask

    logger.info(f"  Voxels before filtering: {n_voxels}")
    logger.info(f"  Outlier voxels detected: {np.sum(outlier_mask)}")
    logger.info(f"  Voxels after filtering: {keep_mask.sum()}")
    logger.info(f"  Percentage retained: {100 * keep_mask.sum() / n_voxels:.1f}%")

    return keep_mask

def create_filtered_positions():
    """Create filtered position lookup by tracking all dropped columns."""

    # Load data and get initial deduplication mask
    sn_data, dedup_column_mask = load_sn_data()
    positions = load_position_lookup()

    # Start with full position lookup (1794 rows)
    logger.info(f"Original position lookup: {len(positions)} rows")

    # Apply deduplication mask to positions (removes rows for duplicate columns)
    positions_after_dedup = positions[dedup_column_mask].copy()
    positions_after_dedup.reset_index(drop=True, inplace=True)
    logger.info(f"After removing duplicate column positions: {len(positions_after_dedup)} rows")

    # Verify dimensions match after deduplication
    if len(positions_after_dedup) != sn_data.shape[1]:
        logger.error(f"Dimension mismatch after dedup: {len(positions_after_dedup)} positions vs {sn_data.shape[1]} voxels")
        logger.error("This should not happen - dedup mask should align data and positions")
        return None

    # Apply MAD filtering to get mask of columns to KEEP
    mad_keep_mask = apply_mad_filtering(sn_data, threshold=QC_THRESHOLD)

    # Apply MAD mask to positions (removes rows for MAD outlier columns)
    filtered_positions = positions_after_dedup[mad_keep_mask].copy()
    filtered_positions.reset_index(drop=True, inplace=True)

    logger.info(f"Filtered positions shape: {filtered_positions.shape}")

    # Save filtered positions to results folder
    output_dir = RESULTS_DIR / "filtered_position_lookups"
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
