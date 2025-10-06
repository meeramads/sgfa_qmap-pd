#!/usr/bin/env python3
"""
Quick Start Example: Running Your First SGFA Analysis

This example shows the simplest way to get started with SGFA qMAP-PD.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_start_synthetic():
    """Quickest way to test SGFA with synthetic data."""
    print("=" * 70)
    print("QUICK START 1: Synthetic Data")
    print("=" * 70)
    
    from data.synthetic import generate_synthetic_data
    from core.run_analysis import run_sgfa_analysis
    
    print("\n1️⃣ Generate synthetic data...")
    data = generate_synthetic_data(
        num_sources=2,  # 2 views
        K=3,            # 3 latent factors
        num_subjects=100
    )
    X_list = data['X_list']
    
    print(f"✅ Generated {len(X_list)} views:")
    for i, X in enumerate(X_list):
        print(f"   View {i}: {X.shape} (subjects × features)")
    
    print("\n2️⃣ Run SGFA analysis...")
    results = run_sgfa_analysis(
        X_list,
        K=3,
        sparsity_level=0.25,  # 25% sparsity
        num_samples=500,      # Fast for demo
        num_chains=2
    )
    
    print("\n✅ Analysis complete!")
    print(f"   Factor loadings shape: {results['W_means'][0].shape}")
    print(f"   Factor scores shape: {results['Z_mean'].shape}")
    
    return results


def quick_start_command_line():
    """Show command-line quick start."""
    print("\n" + "=" * 70)
    print("QUICK START 2: Command Line")
    print("=" * 70)
    
    print("\n📝 Simplest command to run experiments:")
    print("\n# Run all experiments with default settings:")
    print("python run_experiments.py --experiments all")
    
    print("\n# Run single experiment:")
    print("python run_experiments.py --experiments data_validation")
    
    print("\n# Debug mode (faster, reduced MCMC samples):")
    print("python debug_experiments.py data_validation")
    
    print("\n💡 Results will be saved to ./results/ with timestamp")


def quick_start_programmatic():
    """Show programmatic quick start."""
    print("\n" + "=" * 70)
    print("QUICK START 3: Python API")
    print("=" * 70)
    
    print("\n📝 Simplest Python code:")
    print("""
from data.synthetic import generate_synthetic_data
from analysis import quick_sgfa_run

# Generate data
data = generate_synthetic_data(num_sources=2, K=3)
X_list = data['X_list']

# Run SGFA (one line!)
results = quick_sgfa_run(X_list, K=3, percW=25.0)

# Access results
factor_scores = results['Z_mean']  # Subject × factors
factor_loadings = results['W_means']  # List of feature × factors per view
    """)
    
    print("✅ That's it! Three lines of code for a complete analysis.")


def quick_start_with_real_data():
    """Show how to use real qMAP-PD data."""
    print("\n" + "=" * 70)
    print("QUICK START 4: Real qMAP-PD Data")
    print("=" * 70)
    
    print("\n📝 Using your qMAP-PD data:")
    print("""
from data.qmap_pd import load_qmap_pd
from core.run_analysis import run_sgfa_analysis

# Load data
data = load_qmap_pd(data_dir='./qMAP-PD_data')
X_list = data['X_list']

# Run analysis
results = run_sgfa_analysis(X_list, K=5, sparsity_level=0.25)
    """)
    
    print("\n⚙️  Or use command line:")
    print("python run_experiments.py \\")
    print("    --experiments sgfa_parameter_comparison \\")
    print("    --data-dir ./qMAP-PD_data")
    
    print("\n💡 Tip: Start with data_validation to check data quality first!")


def quick_start_configuration():
    """Show configuration quick start."""
    print("\n" + "=" * 70)
    print("QUICK START 5: Custom Configuration")
    print("=" * 70)
    
    print("\n📝 Customize your analysis:")
    
    print("\n1. For 2-view experiment (clinical + 1 ROI):")
    print("   python run_experiments.py \\")
    print("       --experiments sgfa_parameter_comparison \\")
    print("       --select-rois volume_sn_voxels.tsv \\")
    print("       --test-k 2 3")
    
    print("\n2. For 4-view experiment (clinical + 3 ROIs):")
    print("   python run_experiments.py \\")
    print("       --experiments sgfa_parameter_comparison \\")
    print("       --select-rois volume_sn_voxels.tsv volume_putamen_voxels.tsv volume_lentiform_voxels.tsv \\")
    print("       --test-k 3 4 5")
    
    print("\n3. Exclude demographics for cleaner factor maps:")
    print("   python run_experiments.py \\")
    print("       --experiments model_comparison \\")
    print("       --exclude-clinical age sex tiv")
    
    print("\n💡 See experiment_configuration.py for more examples!")


def next_steps():
    """Show what to do after quick start."""
    print("\n" + "=" * 70)
    print("📚 NEXT STEPS")
    print("=" * 70)
    
    print("\n1️⃣ Learn more:")
    print("   - examples/data_example.py: Data loading and preprocessing")
    print("   - examples/models_example.py: Model configuration")
    print("   - examples/experiment_configuration.py: Advanced configuration")
    print("   - examples/visualization_example.py: Visualizing results")
    
    print("\n2️⃣ Read documentation:")
    print("   - README.md: Complete project documentation")
    print("   - docs/configuration.md: Configuration guide")
    print("   - docs/TESTING.md: Testing framework")
    
    print("\n3️⃣ Run experiments:")
    print("   - debug_experiments.py: Fast testing with reduced parameters")
    print("   - run_experiments.py: Full production experiments")
    
    print("\n4️⃣ Customize:")
    print("   - config.yaml: Main configuration file")
    print("   - Adjust K, percW, MCMC parameters, preprocessing strategies")
    
    print("\n5️⃣ Get help:")
    print("   - pytest: Run test suite")
    print("   - python -m pydoc [module]: View module documentation")
    print("   - GitHub issues: Report bugs or ask questions")


def main():
    """Run quick start examples."""
    print("\n" + "🚀 " * 35)
    print("SGFA qMAP-PD: Quick Start Guide")
    print("🚀 " * 35)
    
    # Run the simplest example
    try:
        results = quick_start_synthetic()
    except Exception as e:
        logger.warning(f"Could not run synthetic example: {e}")
        logger.info("This is normal if dependencies are not installed")
    
    # Show other quick starts
    quick_start_command_line()
    quick_start_programmatic()
    quick_start_with_real_data()
    quick_start_configuration()
    next_steps()
    
    print("\n" + "=" * 70)
    print("✅ Quick start complete! You're ready to analyze PD subtypes.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
