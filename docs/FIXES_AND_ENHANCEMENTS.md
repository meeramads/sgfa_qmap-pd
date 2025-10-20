# Fixes and Enhancements Summary

## Recent Code Fixes (October 2025)

### 1. Model Type Configuration Fix
**Problem:** System was ignoring `model_type: "sparse_gfa_fixed"` in config files.

**Files Modified:**
- `experiments/framework.py` - Added `model_type` field to ExperimentConfig
- `run_experiments.py` - Read and pass model_type from config
- `experiments/robustness_testing.py` - Read and pass model_type from config

**Result:** Model type is now correctly propagated through the system.

---

### 2. MCMC Parameter Override Fix
**Problem:** Command-line arguments `--target-accept-prob` and (partially) `--max-tree-depth` were being ignored.

**Files Modified:**
- `run_experiments.py` line 762 - Read target_accept_prob from mcmc config section
- `experiments/robustness_testing.py` line 2940 - Read from config instead of hardcoding

**Result:** Command-line MCMC parameters now work correctly.

---

### 3. Position Lookup Auto-Save
**Problem:** Filtered position lookups weren't saved to experiment-specific directories.

**Files Modified:**
- `experiments/data_validation.py` - Pass experiment-specific output_dir to preprocessing

**Result:** Filtered position lookups now saved alongside experiment results.

---

### 4. Semantic Directory Naming

**Enhancements:**
- Added model type abbreviation to directory names
- Enhanced subdirectory naming in unified results mode
- Added max_tree_depth to semantic names

**Format:**
```
{roi}-{confounds}_K{K}_percW{percW}_slab{df}_{scale}_MAD{threshold}_tree{depth}_{timestamp}
```

**Example:**
```
all_rois-sn_conf-age+sex+tiv_K2_percW33_slab4_2_MAD3.0_tree10_20251018_142445/
├── 01_data_validation_tree10_20251018_142445/
├── 02_robustness_testing_K2_percW33_slab4_2_MAD3.0_tree10_20251018_142501/
└── 03_factor_stability/
```

**Result:** Easy identification and comparison of experimental runs.

---

## Feature Enhancements

### Subject Outlier Analysis Plots
**Location:** `experiments/data_validation.py`

Automatically generates:
- Outlier detection heatmaps (imaging data)
- Clinical outlier detection plots
- Distribution comparison plots

**Access:** Check `results/*/data_validation_*/plots/` directory.

---

## Documentation Organization

All documentation now in `docs/`:
- `docs/` - User guides and references
- `docs/troubleshooting/` - Session-specific debugging
- `docs/api/` - API documentation (Sphinx)

**Key Documents:**
- `CONVERGENCE_TESTING_GUIDE.md` - How to test convergence
- `config_quick_reference.md` - Config parameter reference
- `PCA_CONFIGURATION_GUIDE.md` - PCA usage guide
- `troubleshooting/RECENT_SESSION_SUMMARY.md` - Latest troubleshooting session

---

## Model Comparison: sparse_gfa vs sparse_gfa_fixed

### sparse_gfa (Original)
- Centered parameterization
- Simpler posterior geometry
- Good for: Standard analyses
- MCMC settings: `target_accept_prob=0.8`, `max_tree_depth=13`

### sparse_gfa_fixed (Convergence-Enhanced)
- Non-centered parameterization
- 5 convergence fixes implemented
- Good for: High-dimensional data, difficult convergence cases
- MCMC settings: `target_accept_prob=0.85`, `max_tree_depth=10`, consider `dense_mass=true`

**When to use fixed:**
- Experiencing R-hat > 1.1 with original model
- Need better convergence for high-dimensional neuroimaging
- Want to use all 5 Ferreira et al. 2024 fixes

**When to use original:**
- Current convergence is acceptable
- Want faster sampling
- Lower memory requirements

---

## Command-Line Override Examples

```bash
# Override model type
python run_experiments.py --config config.yaml --model-type sparse_gfa_fixed

# Override MCMC parameters
python run_experiments.py --config config.yaml \
  --target-accept-prob 0.9 --max-tree-depth 10

# Override data parameters
python run_experiments.py --config config.yaml \
  --K 5 --percW 50 --qc-outlier-threshold 2.5

# Full example
python run_experiments.py --config config_convergence.yaml \
  --experiments all \
  --select-rois volume_sn_voxels.tsv \
  --regress-confounds age sex tiv \
  --qc-outlier-threshold 3.0 \
  --percW 33 --K 2 \
  --max-tree-depth 10 --target-accept-prob 0.85
```

---

**Last Updated:** October 20, 2025
