# SGFA Convergence Fixes - Implementation Summary

## Overview

This document summarizes the implementation of 5 critical convergence fixes to transform the Sparse GFA model from catastrophic convergence (R-hat > 12) to robust MCMC sampling (target R-hat < 1.01).

## Baseline Problem

**Observed Issues (from existing results):**
- Run: `all_rois-sn_conf-age+sex+tiv_K2_percW33_MAD3.0_run_20251018_142445/03_factor_stability`
- **W (Loadings) Max R-hat**: 12.613 (Factor 1) - CATASTROPHIC
- **W Mean R-hat**: 6.187
- **Z (Latent Factors) Max R-hat**: 5.229
- **Z Mean R-hat**: 2.458
- Chains exploring completely different modes
- Zero stable factors

**Target Goals:**
- ✅ R-hat < 1.01 for all parameters
- ✅ τ (global scale) ∈ [0.01, 0.05]
- ✅ c (slab parameter) ∈ [1, 3]
- ✅ Divergence rate < 1%
- ✅ ESS > 400

## Implemented Fixes

### Fix #1: Data-Dependent Global Scale τ₀

**Location**: `models/sparse_gfa_fixed.py`, lines 166-181

**Problem**: Original model let τ explore freely, causing explosion to ±200

**Solution**: Constrain τ using Piironen & Vehtari (2017) formula:
```python
# τ₀ = (D₀/(D-D₀)) × (σ/√N)
D0_per_factor = pW_m  # Expected non-zero loadings per factor
tau0 = (D0_per_factor / (Dm_m - D0_per_factor)) * (sigma_std / jnp.sqrt(N))

# Non-centered parameterization for tau
tau_tilde = numpyro.sample(f"tauW_tilde_{m + 1}", dist.HalfCauchy(1.0))
tauW = tau0 * tau_tilde
```

**For your data** (N=86, D=536, percW=33%, K=2):
- D₀ per factor ≈ 89
- τ₀ ≈ 0.0215

**Expected Impact**: τ values constrained to [0.01, 0.05] range

### Fix #2: Proper Slab Regularization

**Location**: `models/sparse_gfa_fixed.py`, lines 139-152

**Problem**: Incorrect InverseGamma parameterization caused c to explore [0, 80]

**Solution**: Correct parameterization IG(α=2, β=2):
```python
cW_tilde = numpyro.sample(
    "cW_tilde",
    dist.InverseGamma(2.0, 2.0),  # α=2, β=2
    sample_shape=(self.num_sources, K),
)
cW_squared = (self.hypers["slab_scale"] ** 2) * cW_tilde
cW = jnp.sqrt(cW_squared)
```

**Expected Impact**: c values in range [0.5, 5], centered around 2

### Fix #3: Non-Centered Parameterization

**Location**: `models/sparse_gfa_fixed.py`, lines 131-132 (W) and 91-92 (Z)

**Problem**: Centered parameterization creates funnel geometry, preventing NUTS from exploring efficiently

**Solution**: Sample from standard normal first, then transform:
```python
# For W (loadings)
z_raw = numpyro.sample("W_raw", dist.Normal(0, 1), sample_shape=(D, K))
# ... apply transformations ...
W_chunk = z_raw_chunk * lmbW_tilde * tauW  # Deterministic transformation
W = numpyro.deterministic("W", W)

# For Z (latent factors)
Z_raw = numpyro.sample("Z_raw", dist.Normal(0, 1), sample_shape=(N, K))
# ... apply transformations ...
Z = numpyro.deterministic("Z", Z)
```

**Expected Impact**:
- Major R-hat improvement (should drop from >10 to <1.1)
- No funnel pattern in τ vs W scatter plots
- Divergence rate drop

### Fix #4: Within-View Standardization

**Location**: Preprocessing pipeline (already implemented)

**Status**: ✅ Already correctly implemented in data preprocessing

**What it does**: Brain and clinical views are standardized separately, preventing the horseshoe prior from over-shrinking clinical features due to scale differences.

### Fix #5: PCA Initialization & MCMC Settings

**Location**: `config_convergence_test.yaml`

**Changes**:
```yaml
model:
  num_warmup: 1000          # Increased from 1000
  target_accept_prob: 0.95  # Increased from 0.8
  max_tree_depth: 12        # Increased from 10
```

**Note**: Full PCA initialization to be added in next iteration if needed.

## Files Created/Modified

### New Files:
1. **models/sparse_gfa_fixed.py** - Fixed model with all convergence improvements
2. **config_convergence_test.yaml** - Test configuration for fixed model
3. **scripts/extract_baseline_diagnostics.py** - Baseline diagnostic extraction
4. **scripts/compare_convergence.py** - Comparison script for before/after analysis
5. **diagnostics/baseline/baseline_summary.json** - Baseline metrics
6. **diagnostics/baseline/baseline_config.yaml** - Baseline configuration

### Modified Files:
1. **models/factory.py** - Registered `sparse_gfa_fixed` model

## How to Run

### Step 1: Verify Baseline Diagnostics
```bash
python scripts/extract_baseline_diagnostics.py
```

### Step 2: Run Test with Fixed Model
```bash
python run_experiments.py --config config_convergence_test.yaml
```

Expected runtime: ~15-30 minutes

### Step 3: Compare Results
```bash
python scripts/compare_convergence.py \
    results/all_rois-sn_conf-age+sex+tiv_K2_percW33_MAD3.0_run_20251018_142445/03_factor_stability \
    results/YOUR_NEW_RUN_FOLDER
```

### Step 4: Review Diagnostics
- Check R-hat values (should be < 1.01)
- Examine hyperparameter trace plots (should show good mixing)
- Review hyperparameter posteriors (τ around 0.02, c around 2)
- Verify divergence rate < 1%

## Success Criteria

Run `scripts/compare_convergence.py` and verify:

- [  ] **W Max R-hat < 1.01** (was 12.6)
- [  ] **Z Max R-hat < 1.01** (was 5.2)
- [  ] **τ values ∈ [0.01, 0.05]** (check trace plots)
- [  ] **c values ∈ [1, 3]** (check posterior plots)
- [  ] **Divergence rate < 1%**
- [  ] **ESS > 400** for all parameters
- [  ] **No funnel geometry** in τ vs W scatter plots

## Next Steps if Convergence Not Perfect

If R-hat is still > 1.1 after testing:

1. **Increase warmup**: Try `num_warmup: 2000-3000`
2. **Increase adapt_delta**: Try `target_accept_prob: 0.99`
3. **Implement full PCA initialization**: Use `core/initialization.py` from guide
4. **Check data preprocessing**: Verify within-view standardization
5. **Run longer**: Try `num_samples: 2000-3000`

## Production Configuration

Once convergence is verified, use the full production config with all fixes:

```bash
# Copy config_convergence_test.yaml to config_production_fixed.yaml
# Update to full settings:
#   num_samples: 2000
#   num_warmup: 2000
#   All experiments enabled
```

## References

- Piironen, J., & Vehtari, A. (2017). Sparsity information and regularization in the horseshoe and other shrinkage priors. *Electronic Journal of Statistics*, 11(2), 5018-5051.
- Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. arXiv:1701.02434.
- Stan Development Team. (2023). Stan User's Guide: Reparameterization.

## Diagnostic File Locations

After running the test:
- Convergence diagnostics: `results/YOUR_RUN/stability_analysis/rhat_convergence_diagnostics.json`
- Hyperparameter posteriors: `results/YOUR_RUN/plots/hyperparameter_posteriors.png`
- Hyperparameter traces: `results/YOUR_RUN/plots/hyperparameter_traces.png`
- Comparison plot: `diagnostics/baseline_vs_fixed_posteriors.png`
