#!/usr/bin/env python3
"""Verify the structure of saved position lookup files."""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def verify_structure():
    """Show the expected directory structure after running data_validation."""

    logger.info("=" * 80)
    logger.info("EXPECTED DIRECTORY STRUCTURE AFTER DATA_VALIDATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("After running:")
    logger.info("  python run_experiments.py --config config_convergence.yaml \\")
    logger.info("    --experiments data_validation --select-rois volume_sn_voxels.tsv")
    logger.info("")
    logger.info("You will get:")
    logger.info("")
    logger.info("results/")
    logger.info("└── data_validation_<timestamp>/")
    logger.info("    ├── config.yaml")
    logger.info("    ├── result.json")
    logger.info("    ├── summary.csv")
    logger.info("    ├── plots/")
    logger.info("    │   ├── data_distribution_comparison.png")
    logger.info("    │   ├── data_validation_feature_reduction.png")
    logger.info("    │   └── ...")
    logger.info("    └── position_lookup_filtered/        ← FILTERED POSITION LOOKUPS")
    logger.info("        └── position_sn_voxels_filtered.tsv")
    logger.info("")
    logger.info("=" * 80)
    logger.info("POSITION LOOKUP FILE FORMAT")
    logger.info("=" * 80)
    logger.info("")
    logger.info("The filtered position lookup file (TSV format, no header):")
    logger.info("")
    logger.info("  Column 1: X coordinate (mm)")
    logger.info("  Column 2: Y coordinate (mm)")
    logger.info("  Column 3: Z coordinate (mm)")
    logger.info("")
    logger.info("Example content:")
    logger.info("  -10.5  -15.2   -8.1")
    logger.info("  -11.3  -14.8   -7.9")
    logger.info("   -9.7  -15.6   -8.3")
    logger.info("  ...")
    logger.info("")
    logger.info("This file contains ONLY the voxel coordinates that survived preprocessing.")
    logger.info("The rows correspond 1-to-1 with the columns in the preprocessed data matrix.")
    logger.info("")
    logger.info("=" * 80)
    logger.info("KEY FEATURES")
    logger.info("=" * 80)
    logger.info("")
    logger.info("✓ Automatic creation during preprocessing")
    logger.info("✓ Saved in experiment-specific directory")
    logger.info("✓ Matches exact preprocessing settings used")
    logger.info("✓ Tracks all filtering steps:")
    logger.info("  - Duplicate voxel removal")
    logger.info("  - MAD-based outlier filtering")
    logger.info("  - Variance-based feature selection")
    logger.info("  - Any other preprocessing steps")
    logger.info("")
    logger.info("✓ Can be used for:")
    logger.info("  - Brain mapping of factor loadings")
    logger.info("  - Spatial visualization")
    logger.info("  - Quality control")
    logger.info("  - Reproducibility verification")
    logger.info("")
    logger.info("=" * 80)

    # Check if there are existing results
    results_dir = Path("./results")
    if results_dir.exists():
        data_val_dirs = sorted(results_dir.glob("data_validation_*"))
        if data_val_dirs:
            logger.info("EXISTING DATA_VALIDATION RESULTS")
            logger.info("=" * 80)
            logger.info("")
            for dir_path in data_val_dirs[-3:]:  # Show last 3
                logger.info(f"📁 {dir_path.name}")
                pos_lookup_dir = dir_path / "position_lookup_filtered"
                if pos_lookup_dir.exists():
                    tsv_files = list(pos_lookup_dir.glob("*.tsv"))
                    if tsv_files:
                        logger.info(f"   ✅ Contains {len(tsv_files)} filtered position lookup file(s)")
                        for tsv_file in tsv_files:
                            logger.info(f"      - {tsv_file.name}")
                    else:
                        logger.info(f"   ⚠️  position_lookup_filtered/ exists but is empty")
                else:
                    logger.info(f"   ❌ No position_lookup_filtered/ directory")
                logger.info("")

    logger.info("=" * 80)

if __name__ == "__main__":
    verify_structure()
