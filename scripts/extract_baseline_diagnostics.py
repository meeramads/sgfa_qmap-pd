#!/usr/bin/env python3
"""Extract baseline diagnostics from existing results."""

import json
from pathlib import Path
import sys

# Point to your most recent baseline run
baseline_run = Path("results/all_rois-sn_conf-age+sex+tiv_K2_percW33_MAD3.0_run_20251018_142445/03_factor_stability")

if not baseline_run.exists():
    print(f"ERROR: Baseline run folder not found: {baseline_run}")
    sys.exit(1)

# Extract diagnostics from saved chains
baseline_diagnostics = {
    "run_folder": str(baseline_run.parent.name),
    "experiment": "factor_stability",
    "config": {
        "K": 2,
        "percW": 33.0,
        "MAD_threshold": 3.0,
        "model_type": "sparse_gfa",
    }
}

print("="*70)
print("BASELINE DIAGNOSTICS EXTRACTION")
print("="*70)
print(f"Run folder: {baseline_run.parent.name}")
print(f"Experiment: factor_stability")
print()

# Check if plots already exist
plots_dir = baseline_run / "plots"
if plots_dir.exists():
    plot_files = list(plots_dir.glob("*.png"))
    print(f"✓ Found {len(plot_files)} plots")
    baseline_diagnostics["plots"] = {
        "hyperparameter_traces": str(plots_dir / "hyperparameter_trace_plots" / "trace_tauW_SN_factor0.png") if (plots_dir / "hyperparameter_trace_plots").exists() else None,
        "hyperparameter_posteriors": str(plots_dir / "hyperparameter_posteriors.png") if (plots_dir / "hyperparameter_posteriors.png").exists() else None,
    }

    # List key plot files
    for plot in ["hyperparameter_traces.png", "hyperparameter_posteriors.png", "mcmc_trace_diagnostics.png"]:
        if (plots_dir / plot).exists():
            print(f"  - {plot}")

# Check for stability analysis
stability_dir = baseline_run / "stability_analysis"
if stability_dir.exists():
    print(f"\n✓ Found stability analysis")
    # Look for JSON files with metrics
    json_files = list(stability_dir.glob("*.json"))
    if json_files:
        print(f"  Found {len(json_files)} JSON files:")
        for jf in json_files:
            print(f"    - {jf.name}")
            try:
                with open(jf) as f:
                    data = json.load(f)
                    if "stable_factors" in data:
                        baseline_diagnostics["stable_factors"] = data["stable_factors"]
                        print(f"      Stable factors: {data['stable_factors']}")
                    if "convergence" in data:
                        baseline_diagnostics["convergence"] = data["convergence"]
            except:
                pass

# Check chains directory for convergence info
chains_dir = baseline_run / "chains"
if chains_dir.exists():
    print(f"\n✓ Found chains directory")
    chain_files = list(chains_dir.glob("chain_*.npz"))
    print(f"  Chains: {len(chain_files)}")
    baseline_diagnostics["num_chains"] = len(chain_files)

# Save baseline summary
output_dir = Path("diagnostics/baseline")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "baseline_summary.json", "w") as f:
    json.dump(baseline_diagnostics, f, indent=2)

print(f"\n{'='*70}")
print(f"✅ Baseline diagnostics saved to: {output_dir / 'baseline_summary.json'}")
print(f"{'='*70}")

# Print key findings
print("\nKEY DIAGNOSTICS:")
print("-"*70)
if "stable_factors" in baseline_diagnostics:
    print(f"Stable factors: {baseline_diagnostics['stable_factors']}")
print(f"Number of chains: {baseline_diagnostics.get('num_chains', 'Unknown')}")
print(f"\nTo view plots, run:")
if baseline_diagnostics["plots"].get("hyperparameter_traces"):
    print(f"  open {baseline_diagnostics['plots']['hyperparameter_traces']}")
if baseline_diagnostics["plots"].get("hyperparameter_posteriors"):
    print(f"  open {baseline_diagnostics['plots']['hyperparameter_posteriors']}")
