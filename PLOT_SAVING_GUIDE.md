# Individual Plot Saving - Complete Guide

## Overview

All plots generated throughout the pipeline are now saved as **individual PNG and PDF files** in addition to being stored in the results dictionaries.

## Where Plots Are Saved

### 1. Data Validation
**Location:** `results/<run_name>/data_validation/individual_plots/`

**Plots saved:**
- `data_distribution_comparison.png/.pdf`
- `quality_metrics_summary.png/.pdf`
- `strategy_comparison.png/.pdf`
- Plus any additional validation plots

**Example:**
```
results/data_validation_rois-sn_run_20251015_092429/
└── data_validation/
    ├── individual_plots/
    │   ├── data_distribution_comparison.png
    │   ├── data_distribution_comparison.pdf
    │   └── ... (all data validation plots)
    └── ... (other files)
```

### 2. Robustness Testing
**Location:** `results/<run_name>/robustness_testing/individual_plots/`

**Plots saved:**
- `seed_robustness.png/.pdf`
- `data_perturbation_robustness.png/.pdf`
- `initialization_robustness.png/.pdf`
- `computational_robustness.png/.pdf`
- `robustness_summary.png/.pdf`
- Plus any advanced diagnostic plots

**Example:**
```
results/robustness_testing_rois-sn_run_20251016_133002/
└── robustness_testing/
    ├── individual_plots/
    │   ├── seed_robustness.png
    │   ├── seed_robustness.pdf
    │   ├── perturbation_robustness.png
    │   └── ... (all robustness plots)
    └── ... (other files)
```

### 3. Factor Stability
**Location:** `results/<run_name>/factor_stability/individual_plots/`

**Plots saved:**
- `factor_stability_summary.png/.pdf`
- `factor_stability_heatmap.png/.pdf`
- `enhanced_loading_distributions.png/.pdf`
- `brain_visualization_summary.png/.pdf` (if enabled)
- `mcmc_trace_diagnostics.png/.pdf`
- `mcmc_parameter_distributions.png/.pdf`
- `hyperparameter_posteriors.png/.pdf`
- `hyperparameter_traces.png/.pdf`
- `factor_variance_profile.png/.pdf`

**PLUS Hyperparameter Individual Plots:**
**Location:** `results/<run_name>/factor_stability/individual_posteriors/`
- `posterior_tauW_view1_factor0.png/.pdf`
- `posterior_tauW_view1_factor1.png/.pdf`
- `posterior_tauW_view2_factor0.png/.pdf`
- `posterior_tauZ_factor0.png/.pdf`
- `posterior_tauZ_factor1.png/.pdf`
- `posterior_sigma_view1.png/.pdf`
- `posterior_sigma_view2.png/.pdf`
- `posterior_cW_view1_factor0.png/.pdf`
- `posterior_cW_view1_factor1.png/.pdf`
- `posterior_cW_view2_factor0.png/.pdf`
- `posterior_cZ_factor0.png/.pdf`
- `posterior_cZ_factor1.png/.pdf`

**Example:**
```
results/factor_stability_rois-sn_run_20251015_092429/
└── 03_factor_stability/
    ├── individual_plots/
    │   ├── factor_stability_summary.png
    │   ├── factor_stability_summary.pdf
    │   ├── mcmc_trace_diagnostics.png
    │   ├── hyperparameter_posteriors.png
    │   └── ... (all factor stability plots)
    ├── individual_posteriors/
    │   ├── posterior_tauW_view1_factor0.png
    │   ├── posterior_tauW_view1_factor0.pdf
    │   ├── posterior_sigma_view1.png
    │   └── ... (all hyperparameter posteriors)
    ├── chains/
    ├── plots/ (legacy combined plots)
    └── stability_analysis/
```

## Implementation Details

### Automatic Saving
All three experiment types automatically save individual plots:

```python
# This happens automatically in:
# - experiments/data_validation.py (line ~374-383)
# - experiments/robustness_testing.py (line ~2703-2712 & ~3068-3075)
# No manual intervention needed!
```

### What Gets Saved
- **Format:** Both PNG (for viewing) and PDF (for publications)
- **Resolution:** 300 DPI by default
- **Quality:** High-quality, publication-ready figures
- **Naming:** Descriptive names matching the plot type

### Why Two Locations?
Some plots (like hyperparameter posteriors) are saved in BOTH places:
1. **Combined figure** in `individual_plots/` - All parameters in one multi-panel figure
2. **Individual posteriors** in `individual_posteriors/` - Each parameter as its own figure

This gives you flexibility:
- Use combined figures for overview/reports
- Use individual figures for detailed analysis/presentations

## Checking Your Results

After running experiments, verify plots were saved:

```bash
# Check data validation plots
ls results/<your_run>/data_validation/individual_plots/

# Check robustness testing plots
ls results/<your_run>/robustness_testing/individual_plots/

# Check factor stability plots
ls results/<your_run>/03_factor_stability/individual_plots/
ls results/<your_run>/03_factor_stability/individual_posteriors/
```

## Benefits

✅ **No more cramped subplots** - Every plot is clean and full-size
✅ **Publication ready** - High-resolution PDFs for papers
✅ **Easy sharing** - Individual files easy to email/present
✅ **Better analysis** - View each diagnostic in detail
✅ **Automatic** - No configuration needed

## Troubleshooting

If plots aren't being saved:

1. **Check logs** - Look for messages like:
   ```
   ✅ Saved 12 individual plots to results/.../individual_plots
   ```

2. **Check permissions** - Ensure results directory is writable

3. **Check disk space** - Plots can be several MB each

4. **Check errors** - Look for warnings like:
   ```
   Failed to save individual plots: [error message]
   ```

## Legacy Behavior

Old results folders may not have `individual_plots/` directories because this is a new feature. Only new runs (after this implementation) will have individual plots automatically saved.

## Disabling (Not Recommended)

If you need to disable individual plot saving (to save disk space), you can comment out the `save_all_plots_individually` calls in:
- `experiments/data_validation.py` (line ~374-383)
- `experiments/robustness_testing.py` (lines ~2703-2712 and ~3068-3075)

However, this is **not recommended** as individual plots are essential for detailed analysis.
