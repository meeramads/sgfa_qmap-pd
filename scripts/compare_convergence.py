#!/usr/bin/env python3
"""Compare convergence between baseline and fixed model runs."""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def load_convergence_diagnostics(run_dir):
    """Load convergence diagnostics from a run directory."""
    # Try to find the rhat diagnostics file
    possible_paths = [
        run_dir / "stability_analysis" / "rhat_convergence_diagnostics.json",
        run_dir / "intermediate" / "convergence_diagnostics.json",
        run_dir / "convergence_diagnostics.json",
    ]

    for path in possible_paths:
        if path.exists():
            with open(path) as f:
                return json.load(f)

    return None

def compare_runs(baseline_dir, fixed_dir):
    """Compare baseline and fixed model runs."""

    baseline_dir = Path(baseline_dir)
    fixed_dir = Path(fixed_dir)

    print("="*80)
    print("CONVERGENCE COMPARISON: Baseline vs Fixed Model")
    print("="*80)
    print(f"\nBaseline: {baseline_dir.name}")
    print(f"Fixed:    {fixed_dir.name}")
    print()

    # Load diagnostics
    baseline_diag = load_convergence_diagnostics(baseline_dir)
    fixed_diag = load_convergence_diagnostics(fixed_dir)

    if baseline_diag is None:
        print(f"❌ Could not find baseline diagnostics in {baseline_dir}")
        return

    if fixed_diag is None:
        print(f"❌ Could not find fixed model diagnostics in {fixed_dir}")
        return

    # Extract key metrics
    print("CONVERGENCE METRICS COMPARISON:")
    print("-"*80)

    # R-hat for W
    if "W" in baseline_diag and "W" in fixed_diag:
        baseline_W_rhat = baseline_diag["W"]["max_rhat_overall"]
        fixed_W_rhat = fixed_diag["W"]["max_rhat_overall"]
        improvement_W = ((baseline_W_rhat - fixed_W_rhat) / baseline_W_rhat) * 100

        print(f"\nLoadings (W) - Max R-hat:")
        print(f"  Baseline: {baseline_W_rhat:8.3f}")
        print(f"  Fixed:    {fixed_W_rhat:8.3f}")
        print(f"  Improvement: {improvement_W:6.1f}% reduction")

        if fixed_W_rhat < 1.01:
            print(f"  ✅ EXCELLENT: Fixed model R-hat < 1.01 (target achieved!)")
        elif fixed_W_rhat < 1.1:
            print(f"  ✅ GOOD: Fixed model R-hat < 1.1 (acceptable)")
        elif fixed_W_rhat < baseline_W_rhat / 2:
            print(f"  ⚠️  IMPROVED: Fixed model R-hat reduced by >50%, but still needs work")
        else:
            print(f"  ❌ MINIMAL IMPROVEMENT: Further investigation needed")

    # R-hat for Z
    if "Z" in baseline_diag and "Z" in fixed_diag:
        baseline_Z_rhat = baseline_diag["Z"]["max_rhat_overall"]
        fixed_Z_rhat = fixed_diag["Z"]["max_rhat_overall"]
        improvement_Z = ((baseline_Z_rhat - fixed_Z_rhat) / baseline_Z_rhat) * 100

        print(f"\nLatent Factors (Z) - Max R-hat:")
        print(f"  Baseline: {baseline_Z_rhat:8.3f}")
        print(f"  Fixed:    {fixed_Z_rhat:8.3f}")
        print(f"  Improvement: {improvement_Z:6.1f}% reduction")

        if fixed_Z_rhat < 1.01:
            print(f"  ✅ EXCELLENT: Fixed model R-hat < 1.01 (target achieved!)")
        elif fixed_Z_rhat < 1.1:
            print(f"  ✅ GOOD: Fixed model R-hat < 1.1 (acceptable)")

    print("\n" + "="*80)
    print("CONVERGENCE STATUS:")
    print("="*80)

    if "W" in fixed_diag and "Z" in fixed_diag:
        W_converged = fixed_diag["W"]["max_rhat_overall"] < 1.01
        Z_converged = fixed_diag["Z"]["max_rhat_overall"] < 1.01

        if W_converged and Z_converged:
            print("🎉 SUCCESS: Both W and Z show excellent convergence (R-hat < 1.01)")
            print("   Ready for production use!")
        elif fixed_diag["W"]["max_rhat_overall"] < 1.1 and fixed_diag["Z"]["max_rhat_overall"] < 1.1:
            print("✅ ACCEPTABLE: R-hat < 1.1 for all parameters")
            print("   May need longer warmup for perfect convergence")
        else:
            print("⚠️  NEEDS MORE WORK: Some parameters still show poor convergence")
            print("   Consider:")
            print("   - Increasing num_warmup to 2000-3000")
            print("   - Increasing target_accept_prob to 0.99")
            print("   - Adding PCA initialization (Fix #5)")

    # Create comparison plots if available
    baseline_plots = baseline_dir / "plots"
    fixed_plots = fixed_dir / "plots"

    if baseline_plots.exists() and fixed_plots.exists():
        print(f"\n{'='*80}")
        print("GENERATING COMPARISON PLOTS...")
        print("="*80)

        # Find hyperparameter plots
        baseline_hyper_post = baseline_plots / "hyperparameter_posteriors.png"
        fixed_hyper_post = fixed_plots / "hyperparameter_posteriors.png"

        if baseline_hyper_post.exists() and fixed_hyper_post.exists():
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # Baseline
            img_baseline = mpimg.imread(str(baseline_hyper_post))
            axes[0].imshow(img_baseline)
            axes[0].set_title(f"BASELINE\nMax R-hat: {baseline_diag['W']['max_rhat_overall']:.2f}",
                            fontsize=14, fontweight='bold', color='red')
            axes[0].axis('off')

            # Fixed
            img_fixed = mpimg.imread(str(fixed_hyper_post))
            axes[1].imshow(img_fixed)
            axes[1].set_title(f"FIXED MODEL\nMax R-hat: {fixed_diag['W']['max_rhat_overall']:.2f}",
                            fontsize=14, fontweight='bold',
                            color='green' if fixed_diag['W']['max_rhat_overall'] < 1.1 else 'orange')
            axes[1].axis('off')

            plt.suptitle("Hyperparameter Posterior Comparison", fontsize=16, fontweight='bold')
            plt.tight_layout()

            output_path = Path("diagnostics/baseline_vs_fixed_posteriors.png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n📊 Comparison plot saved: {output_path}")
            plt.close()

    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print("="*80)

    if "W" in fixed_diag and fixed_diag["W"]["max_rhat_overall"] > 1.1:
        print("1. Review hyperparameter trace plots for mixing issues")
        print("2. Consider increasing num_warmup and target_accept_prob")
        print("3. Implement PCA initialization (Fix #5)")
        print("4. Check for data preprocessing issues")
    else:
        print("1. Review plots to confirm convergence visually")
        print("2. Validate model on held-out data")
        print("3. Run full production config with all fixes")
        print("4. Document successful convergence parameters")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        baseline_dir = sys.argv[1]
        fixed_dir = sys.argv[2]
    else:
        # Use defaults
        baseline_dir = "results/all_rois-sn_conf-age+sex+tiv_K2_percW33_MAD3.0_run_20251018_142445/03_factor_stability"
        fixed_dir = "results/convergence_test"  # Will be created after test run

        print(f"Usage: {sys.argv[0]} <baseline_dir> <fixed_dir>")
        print(f"\nUsing defaults:")
        print(f"  Baseline: {baseline_dir}")
        print(f"  Fixed:    {fixed_dir}")
        print()

        if not Path(fixed_dir).exists():
            print(f"⚠️  Fixed model results not found at: {fixed_dir}")
            print(f"    Run the test first: python run_experiments.py --config config_convergence_test.yaml")
            sys.exit(1)

    compare_runs(baseline_dir, fixed_dir)
