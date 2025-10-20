# Unified Semantic Subdirectory Naming Update

## Summary

Updated the `--unified-results` mode to use **semantic naming for all subdirectories**, not just the top-level directory. This ensures consistent parameter visibility across the entire directory structure.

## Changes Made

### File: `run_experiments.py` (lines 450-499)

**Before:**
```python
experiment_dir_mapping = {
    "data_validation": "01_data_validation",
    "robustness_testing": "02_robustness_testing",
    "factor_stability": "03_factor_stability",
    "clinical_validation": "04_clinical_validation",
}
```

**After:**
```python
def get_semantic_subdir_name(base_name: str, prefix: str) -> str:
    """Create semantic subdirectory name with experiment parameters."""
    parts = [prefix]

    # Add model type abbreviation if not default
    model_type = config.get("model", {}).get("model_type", "sparseGFA")
    if model_type == "sparse_gfa_fixed":
        parts.append("fixed")
    # ... add all parameters ...

    return "_".join(parts)

experiment_dir_mapping = {
    "data_validation": get_semantic_subdir_name("data_validation", "01_data_validation"),
    "robustness_testing": get_semantic_subdir_name("robustness_testing", "02_robustness_testing"),
    "factor_stability": get_semantic_subdir_name("factor_stability", "03_factor_stability"),
    "clinical_validation": get_semantic_subdir_name("clinical_validation", "04_clinical_validation"),
}
```

## Directory Structure

### Old Structure (Non-Semantic Subdirectories)
```
results/all_rois-sn_conf-age+sex+tiv_K20_percW33_MAD3.0_run_20251020_150000/
├── 01_data_validation/              ← Generic name
├── 02_robustness_testing/           ← Generic name
├── 03_factor_stability/             ← Generic name
└── summaries/
```

### New Structure (Semantic Subdirectories)
```
results/all_rois-sn_conf-age+sex+tiv_K20_percW33_MAD3.0_run_20251020_150000/
├── 01_data_validation_fixed_K20_percW33_slab4_2_MAD3.0/       ← Semantic!
├── 02_robustness_testing_fixed_K20_percW33_slab4_2_MAD3.0_tree10/  ← Semantic!
├── 03_factor_stability_fixed_K20_percW33_slab4_2_MAD3.0_tree10/    ← Semantic!
└── summaries/
```

## Parameters Included in Names

1. **Prefix** - Numbered experiment order (01_, 02_, 03_)
2. **Model Type** - fixed/neuro/std (omitted for default sparseGFA)
3. **K** - Number of factors (K20)
4. **percW** - Sparsity percentage (percW33)
5. **Slab** - Prior parameters (slab4_2)
6. **MAD** - QC threshold (MAD3.0)
7. **Tree** - Max tree depth if non-default (tree10)

## Examples

### Using config_convergence.yaml

**Config:**
```yaml
model:
  model_type: "sparse_gfa_fixed"
  K: 20
  percW: 33
  slab_df: 4
  slab_scale: 2

preprocessing:
  qc_outlier_threshold: 3.0

factor_stability:
  max_tree_depth: 10
```

**Command:**
```bash
python run_experiments.py --config config_convergence.yaml \
  --experiments all --unified-results \
  --select-rois volume_sn_voxels.tsv \
  --regress-confounds age sex tiv
```

**Resulting Structure:**
```
results/
└── all_rois-sn_conf-age+sex+tiv_K20_percW33_MAD3.0_run_20251020_150000/
    ├── 01_data_validation_fixed_K20_percW33_slab4_2_MAD3.0/
    │   ├── config.yaml
    │   ├── plots/
    │   │   ├── data_distribution_comparison.png
    │   │   └── data_validation_feature_reduction.png
    │   ├── position_lookup_filtered/
    │   │   └── position_sn_voxels_filtered.tsv
    │   ├── result.json
    │   └── summary.csv
    │
    ├── 02_robustness_testing_fixed_K20_percW33_slab4_2_MAD3.0_tree10/
    │   ├── config.yaml
    │   ├── plots/
    │   │   └── robustness_summary.png
    │   ├── result.json
    │   └── summary.csv
    │
    ├── 03_factor_stability_fixed_K20_percW33_slab4_2_MAD3.0_tree10/
    │   ├── chains/
    │   │   ├── chain_0/
    │   │   ├── chain_1/
    │   │   ├── chain_2/
    │   │   └── chain_3/
    │   ├── plots/
    │   │   ├── factor_stability_heatmap.png
    │   │   ├── hyperparameter_posteriors.png
    │   │   └── mcmc_trace_diagnostics.png
    │   ├── stability_analysis/
    │   │   ├── factor_stability_summary.json
    │   │   ├── consensus_factor_loadings.csv
    │   │   └── similarity_matrix.npz
    │   └── config.yaml
    │
    ├── plots/              # Shared cross-experiment plots
    ├── brain_maps/         # Shared brain mapping outputs
    ├── summaries/          # Cross-experiment summaries
    │   └── complete_experiment_summary.yaml
    ├── experiments.log     # Unified experiment log
    └── README.md           # Auto-generated summary
```

## Benefits

### 1. Complete Transparency
Every subdirectory clearly shows which parameters were used:
```bash
ls results/all_rois*/
# 01_data_validation_fixed_K20_percW33_slab4_2_MAD3.0/
# 02_robustness_testing_fixed_K20_percW33_slab4_2_MAD3.0_tree10/
# 03_factor_stability_fixed_K20_percW33_slab4_2_MAD3.0_tree10/
```

### 2. Easy Comparison
Compare same experiment across different runs:
```bash
# All robustness testing results
find results/ -name "02_robustness_testing_*" -type d

# All fixed model factor stability results
find results/ -name "03_factor_stability_fixed_*" -type d

# All K=20 experiments
find results/ -name "*_K20_*" -type d
```

### 3. Self-Documenting
Directory names are self-documenting - no need to open config files to know what parameters were used.

### 4. Consistent with Top-Level
Both top-level and subdirectories use semantic naming:
```
all_rois-sn_conf-age+sex+tiv_K20_percW33_MAD3.0_run_20251020_150000/
└── 03_factor_stability_fixed_K20_percW33_slab4_2_MAD3.0_tree10/
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Consistent semantic naming throughout!
```

## Comparison Table

| Aspect | Old (Generic) | New (Semantic) |
|--------|---------------|----------------|
| **Visibility** | Need to open config.yaml | See params in directory name |
| **Comparison** | Hard to compare runs | Easy pattern matching |
| **Documentation** | Requires external notes | Self-documenting |
| **Model Type** | Not visible | Clearly shown (fixed/neuro/std) |
| **Tree Depth** | Hidden | Visible if non-default |
| **Organization** | Generic 01_, 02_, 03_ | Semantic with numbered prefixes |

## Testing

Verify the new naming:

```bash
python test_unified_semantic_naming.py
```

This shows:
- Configuration parameters
- Generated subdirectory names
- Example directory structure
- Benefits summary

## Migration

### For Existing Results
Old results with generic names (e.g., `01_data_validation`) will continue to work. New runs will use semantic names.

### Backward Compatibility
The numbered prefixes (01_, 02_, 03_) are preserved, so existing scripts that rely on these prefixes will continue to work.

## Usage

### Standard Run (Semantic Top-Level)
```bash
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability --select-rois volume_sn_voxels.tsv

# Creates:
# results/factor_stability_fixed_K20_percW33_slab4_2_MAD3.0_tree10_20251020_150000/
```

### Unified Run (Semantic Top-Level AND Subdirectories)
```bash
python run_experiments.py --config config_convergence.yaml \
  --experiments all --unified-results --select-rois volume_sn_voxels.tsv

# Creates:
# results/all_rois-sn_conf-age+sex+tiv_K20_percW33_MAD3.0_run_20251020_150000/
#   ├── 01_data_validation_fixed_K20_percW33_slab4_2_MAD3.0/
#   ├── 02_robustness_testing_fixed_K20_percW33_slab4_2_MAD3.0_tree10/
#   └── 03_factor_stability_fixed_K20_percW33_slab4_2_MAD3.0_tree10/
```

---

**Last Updated:** October 20, 2025
**Version:** 2.0 - Added semantic naming to unified results subdirectories
