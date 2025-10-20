# Convergence Issue Analysis - October 20, 2025

## Problem Statement

Current run with `sparse_gfa_fixed` model shows poor MCMC convergence (acceptance probability 0.45) compared to previous working run (0.87-0.93).

## Comparison: Working Run vs Current Run

### Working Run (Previous)
```
Model: sparse_gfa (regular model)
Config: config_convergence_K2_tree10.yaml (command line overrides)
Tree depth: 10 (command line override)
Data shape: (133, 850) - 850 voxels after preprocessing
Acceptance probability: 0.87, 0.93 (excellent)
Step size: 1.93e-04
Convergence: GOOD ✅
```

### Current Run
```
Model: sparse_gfa_fixed (non-centered parameterization)
Config: config_convergence.yaml
Tree depth: 13 (from config line 137)
Data shape: (133, 531) - 531 voxels after preprocessing
Acceptance probability: 0.45 (poor)
Step size: 4.19e-04
Convergence: POOR ❌
```

## Root Causes Identified

### 1. Different Model Architecture

**Critical difference:** The working run used `sparse_gfa` (regular centered parameterization), while the current run uses `sparse_gfa_fixed` (non-centered parameterization).

**Impact:**
- Non-centered parameterization changes the geometry of the posterior distribution
- This requires different MCMC step sizes and adaptation strategies
- Default NUTS parameters that worked for `sparse_gfa` may not work for `sparse_gfa_fixed`

**Evidence:**
- Working run: `model_type: "sparseGFA"` (or not specified, defaulting to sparseGFA)
- Current config: `model_type: "sparse_gfa_fixed"` (line 42)

### 2. Different Preprocessing Results

**Data dimension mismatch:**
- Working run: 850 voxels survived preprocessing
- Current run: 531 voxels survived preprocessing
- Difference: 319 fewer voxels (37.5% reduction)

**Possible causes:**
- Different `qc_outlier_threshold` values
- Different `variance_threshold` values
- Different `min_voxel_distance` settings
- Different data files or ROI selections

**Config values (current):**
```yaml
qc_outlier_threshold: 3.0        # Line 68
variance_threshold: 0.02         # Line 62
min_voxel_distance: 3.0          # Line 66
```

### 3. Different Tree Depth

**Current config:**
```yaml
factor_stability:
  max_tree_depth: 13             # Line 137
```

**Working run:**
- Used `--max-tree-depth 10` via command line

**Impact:**
- Higher tree depth (13) allows deeper trajectory exploration
- Can lead to larger step sizes and lower acceptance rates if geometry is challenging
- With poor geometry (from non-centered parameterization), this exacerbates the problem

## Key Issue: Model-Specific MCMC Tuning

The `sparse_gfa_fixed` model uses **non-centered parameterization** which fundamentally changes the MCMC sampling behavior:

### Centered Parameterization (sparse_gfa)
```python
Z ~ Normal(0, 1)  # Latent factors
# Direct correlation between Z and observations
```
- Simpler posterior geometry
- Works well with default NUTS parameters
- Acceptance probability naturally stays near 0.8

### Non-Centered Parameterization (sparse_gfa_fixed)
```python
Z_raw ~ Normal(0, 1)  # Raw latent factors
Z = mu + sigma * Z_raw  # Transformed factors
# Indirect relationship - can create difficult geometry
```
- Can have more complex posterior geometry
- May require smaller step sizes (adaptive tuning)
- May require different target_accept_prob
- May benefit from dense mass matrix

## Solutions

### Option 1: Tune MCMC Parameters for sparse_gfa_fixed (Recommended)

The `sparse_gfa_fixed` model likely needs different MCMC tuning:

```yaml
factor_stability:
  K: 20
  num_chains: 4
  num_samples: 10000
  num_warmup: 3000
  target_accept_prob: 0.85        # Increase from 0.8 → 0.85
  max_tree_depth: 10              # Decrease from 13 → 10
  dense_mass: true                # Enable dense mass matrix
```

**Rationale:**
- **Higher target_accept_prob (0.85)**: Forces smaller step sizes, better for complex geometry
- **Lower max_tree_depth (10)**: Prevents overly ambitious trajectory exploration
- **dense_mass: true**: Better adaptation to posterior geometry correlations

### Option 2: Use Regular sparse_gfa Model

If convergence is critical and you don't specifically need the non-centered parameterization:

```yaml
model:
  model_type: "sparseGFA"         # Use regular model
  max_tree_depth: 10              # Keep at 10
```

This will give you the same good convergence as the working run.

### Option 3: Investigate Preprocessing Differences

The 319 fewer voxels (850 → 531) needs investigation:

1. **Check if different ROI file was used:**
   ```bash
   # Compare voxel counts in different position lookup files
   wc -l qMAP-PD_data/position_lookup/position_sn_voxels.tsv
   wc -l qMAP-PD_data/position_lookup_filtered/*.tsv
   ```

2. **Check preprocessing settings:**
   - Was `qc_outlier_threshold` different in working run?
   - Was `variance_threshold` different?

3. **Use same preprocessing as working run:**
   - Extract config from working run results directory
   - Compare preprocessing sections

## Recommended Action Plan

### Step 1: Immediate Fix (Use Proper MCMC Tuning)

Edit `config_convergence.yaml` lines 136-138:

```yaml
factor_stability:
  # ... existing settings ...
  target_accept_prob: 0.85        # Was: 0.8
  max_tree_depth: 10              # Was: 13
  dense_mass: true                # Was: false
```

### Step 2: Test the Fix

```bash
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability --select-rois volume_sn_voxels.tsv \
  --K 2 --percW 33 --max-tree-depth 10
```

**Expected outcome:**
- Acceptance probability should increase to 0.75-0.85 range
- Smaller step sizes (more conservative sampling)
- Better convergence diagnostics

### Step 3: If Still Poor, Fall Back to Regular Model

Edit `config_convergence.yaml` line 42:

```yaml
model:
  model_type: "sparseGFA"         # Change back to regular model
```

This will give you the same excellent convergence as the working run.

## Technical Details: Why Non-Centered Parameterization is Harder

### Posterior Geometry

**Centered:**
```
P(Z, W | X) - Direct relationship
- Relatively simple posterior surface
- HMC can take larger steps
```

**Non-Centered:**
```
P(Z_raw, mu, sigma, W | X) - Indirect through transformation
- More complex posterior surface
- Funnel-shaped geometry common
- Requires smaller, more careful steps
```

### Step Size Adaptation

- NUTS adapts step size during warmup to achieve target acceptance rate
- With complex geometry, default adaptation may not find optimal step size
- Higher `target_accept_prob` forces more conservative (smaller) steps
- Dense mass matrix helps HMC navigate correlations

## Verification

After applying the fix, check the logs for:

```
✅ GOOD SIGNS:
- Acceptance probability: 0.75 - 0.85
- Step size: < 1e-04 (smaller is better for this model)
- No divergences
- R-hat values near 1.0

❌ BAD SIGNS:
- Acceptance probability: < 0.7 or > 0.95
- Many divergences
- R-hat values > 1.1
```

## References

- Non-centered parameterization: https://mc-stan.org/docs/stan-users-guide/reparameterization.html
- NUTS tuning: https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.hmc.HMC
- Funnel geometries: Betancourt & Girolami (2015) - "Hamiltonian Monte Carlo for Hierarchical Models"

---

**Date:** October 20, 2025
**Analysis by:** Claude Code
**Priority:** HIGH - Blocking convergence testing
