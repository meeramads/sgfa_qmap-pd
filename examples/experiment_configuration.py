#!/usr/bin/env python3
"""
Example: Configuring and Running Experiments with Different Options

Demonstrates how to run experiments with:
- ROI selection
- Clinical feature exclusion  
- Custom K values
- Different experiment types
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_roi_selection():
    """Example: Selecting specific ROIs for imaging data."""
    print("=" * 70)
    print("EXAMPLE 1: ROI Selection")
    print("=" * 70)
    
    print("\n📖 Use --select-rois to load specific imaging ROIs:")
    print("\nCommand examples:")
    
    # Single ROI
    print("\n1. Single ROI (clinical + 1 imaging view = 2 views total):")
    print("   python run_experiments.py --experiments sgfa_hyperparameter_tuning \\")
    print("       --select-rois volume_sn_voxels.tsv")
    print("   → Result directory: sgfa_hyperparameter_tuning_rois-sn_run_TIMESTAMP/")
    
    # Multiple ROIs
    print("\n2. Multiple ROIs (clinical + 3 imaging views = 4 views total):")
    print("   python run_experiments.py --experiments model_comparison \\")
    print("       --select-rois volume_sn_voxels.tsv volume_putamen_voxels.tsv volume_lentiform_voxels.tsv")
    print("   → Result directory: model_comparison_rois-sn+putamen+lentiform_run_TIMESTAMP/")
    
    print("\n💡 Tip: ROI names are automatically extracted from filenames")
    print("   volume_sn_voxels.tsv → 'sn'")
    print("   volume_putamen_voxels.tsv → 'putamen'")


def example_clinical_exclusion():
    """Example: Excluding demographic/clinical features."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Clinical Feature Exclusion")
    print("=" * 70)
    
    print("\n📖 Use --exclude-clinical to drop specific clinical features:")
    print("\nCommand examples:")
    
    # Exclude demographics
    print("\n1. Exclude demographics (age, sex, TIV):")
    print("   python run_experiments.py --experiments model_comparison \\")
    print("       --exclude-clinical age sex tiv")
    print("   → Focuses on motor symptoms only")
    
    # Exclude multiple features
    print("\n2. Exclude demographics + education:")
    print("   python run_experiments.py --experiments clinical_validation \\")
    print("       --exclude-clinical age sex tiv education")
    print("   → Result directory includes: _excl-age+sex+tiv+education_")
    
    print("\n💡 Tip: Common features to exclude:")
    print("   - Demographics: age, sex")
    print("   - Anatomical: tiv (total intracranial volume)")
    print("   - Socioeconomic: education")


def example_k_selection():
    """Example: Specifying K values for parameter comparison."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: K Value Selection")
    print("=" * 70)
    
    print("\n📖 Use --test-k to specify which K values to test:")
    print("\nTheoretical guidance:")
    print("   - 2-view experiments → test K=2,3")
    print("   - 4-view experiments → test K=3,4,5")
    print("   - K should be ≤ or slightly > number of views")
    
    print("\nCommand examples:")
    
    # 2-view scenario
    print("\n1. Two views (clinical + 1 ROI):")
    print("   python run_experiments.py --experiments sgfa_hyperparameter_tuning \\")
    print("       --select-rois volume_sn_voxels.tsv \\")
    print("       --test-k 2 3")
    print("   → Tests K=2 and K=3 (appropriate for 2 views)")
    
    # 4-view scenario
    print("\n2. Four views (clinical + 3 ROIs):")
    print("   python run_experiments.py --experiments sgfa_hyperparameter_tuning \\")
    print("       --select-rois volume_sn_voxels.tsv volume_putamen_voxels.tsv volume_lentiform_voxels.tsv \\")
    print("       --test-k 3 4 5")
    print("   → Tests K=3, K=4, K=5 (appropriate for 4 views)")
    
    print("\n💡 Tip: Default is K=[2,3,4,5] if --test-k not specified")


def example_combined_options():
    """Example: Combining multiple configuration options."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Combined Configuration")
    print("=" * 70)
    
    print("\n📖 Combine options for precise experimental control:")
    print("\nFull example:")
    
    print("\npython run_experiments.py \\")
    print("    --experiments sgfa_hyperparameter_tuning \\")
    print("    --select-rois volume_sn_voxels.tsv volume_putamen_voxels.tsv \\")
    print("    --exclude-clinical age sex tiv \\")
    print("    --test-k 2 3 \\")
    print("    --data-dir ./qMAP-PD_data")
    
    print("\n📊 This configuration:")
    print("   ✓ Runs SGFA parameter comparison experiment")
    print("   ✓ Uses 3 views: clinical (minus demographics) + SN + putamen")
    print("   ✓ Tests K=2 and K=3 factors")
    print("   ✓ Tests 3 sparsity levels (percW: 10%, 25%, 33%)")
    print("   ✓ Total: 2 K × 3 percW = 6 configurations")
    
    print("\n📁 Result directory:")
    print("   sgfa_hyperparameter_tuning_rois-sn+putamen_excl-age+sex+tiv_K-2+3_run_TIMESTAMP/")
    
    print("\n💡 Directory name clearly shows the configuration!")


def example_typical_workflow():
    """Example: Typical experimental workflow."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Typical Experimental Workflow")
    print("=" * 70)
    
    print("\n📖 Recommended workflow for PD subtype analysis:")
    
    print("\n1️⃣ Start with data validation:")
    print("   python run_experiments.py --experiments data_validation \\")
    print("       --select-rois volume_sn_voxels.tsv")
    
    print("\n2️⃣ Run parameter optimization:")
    print("   python run_experiments.py --experiments sgfa_hyperparameter_tuning \\")
    print("       --select-rois volume_sn_voxels.tsv \\")
    print("       --exclude-clinical age sex tiv \\")
    print("       --test-k 2 3")
    
    print("\n3️⃣ Compare models with optimal parameters:")
    print("   python run_experiments.py --experiments model_comparison \\")
    print("       --select-rois volume_sn_voxels.tsv \\")
    print("       --exclude-clinical age sex tiv")
    
    print("\n4️⃣ Run clinical validation:")
    print("   python run_experiments.py --experiments clinical_validation \\")
    print("       --select-rois volume_sn_voxels.tsv \\")
    print("       --exclude-clinical age sex tiv")
    
    print("\n💡 Each step builds on the previous, refining your analysis")


def main():
    """Run all examples."""
    print("\n" + "🚀 " * 35)
    print("SGFA qMAP-PD: Experiment Configuration Examples")
    print("🚀 " * 35)
    
    example_roi_selection()
    example_clinical_exclusion()
    example_k_selection()
    example_combined_options()
    example_typical_workflow()
    
    print("\n" + "=" * 70)
    print("📚 Additional Resources")
    print("=" * 70)
    print("\nFor more information, see:")
    print("  - README.md: Full documentation")
    print("  - config.yaml: Configuration file with all options")
    print("  - docs/configuration.md: Detailed configuration guide")
    
    print("\n✅ Examples complete! Ready to configure your experiments.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
