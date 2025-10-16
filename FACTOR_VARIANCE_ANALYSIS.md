# Factor Variance Analysis for ARD Shrinkage Assessment

## Overview

Added comprehensive factor variance profile analysis to assess the effectiveness of Automatic Relevance Determination (ARD) shrinkage in overparameterized models. This is **critical** for understanding whether high-K runs are identifying real factors vs. being affected by measurement artifacts.

## What Was Added

### 1. New Function: `analyze_factor_variance_profile()`

**Location**: `analysis/mcmc_diagnostics.py`

**Purpose**: Compute and visualize the variance profile of factors to identify which factors capture real signal vs. which are shrunk to near-zero by ARD priors.

**Key Features**:
- Calculates variance of each factor across subjects (averaged over MCMC samples)
- Identifies "active" factors (variance > threshold, default 0.1)
- Computes effective dimensionality (factors explaining 90% of variance)
- Detects healthy vs. poor ARD shrinkage via shrinkage ratio metric
- Generates comprehensive 8-panel diagnostic visualization

### 2. Integration into Robustness Testing

**Location**: `experiments/robustness_testing.py` (lines 2447-2500)

The variance profile analysis is now automatically run during factor stability analysis and generates:
- Factor variance profile plot
- Cross-check between stable factors and active factors
- Warnings if mismatch detected (potential measurement artifact)

### 3. Test Suite

**Location**: `test_variance_profile.py`

Comprehensive tests with synthetic data demonstrating:
1. **Healthy ARD (K=20, 3 real factors)**: Sharp drop-off, 3 active factors detected ✅
2. **Poor Shrinkage (K=20, gradual decline)**: 18 active factors, suggests convergence issues ❌
3. **Healthy ARD (K=50, 4 real factors)**: 4 active factors despite K=50 ✅
4. **Baseline (K=2)**: 2 active factors, no shrinkage needed ✅

## Key Metrics

### Shrinkage Ratio

**Definition**: `mean(top 5 variances) / mean(next 10 variances)`

**Interpretation**:
- **Ratio > 10**: ✅ **HEALTHY ARD** - Sharp drop-off, model confidently identified few real factors
- **Ratio 3-10**: ⚠️ **MODERATE SHRINKAGE** - Some uncertainty, check convergence
- **Ratio < 3**: ❌ **POOR SHRINKAGE** - Variance spread across many factors, suggests convergence issues or overfitting

### Active Factors

**Definition**: Factors with variance > 0.1 (configurable threshold)

**Purpose**: Count how many factors exceed threshold to determine effective dimensionality

### Effective Dimensionality

**Definition**: Minimum number of factors needed to explain 90% of variance

**Purpose**: Understand true dimensionality independent of K parameter

## Visualization Output

The variance profile plot contains 8 panels:

1. **Variance Profile (Bar)**: All K factors sorted by variance, color-coded by threshold
2. **Log-Scale Variance**: Better visualization of drop-off
3. **Cumulative Variance**: Shows 90% threshold crossing point
4. **Normalized Variance**: Ratios relative to max variance
5. **Top 15 Factors**: Detailed view of highest-variance factors
6. **Histogram**: Distribution of variance values
7. **Scatter (Original Order)**: Variance by original factor index
8. **Summary Statistics**: Numeric diagnostics and interpretation

## Usage Example

```python
from analysis.mcmc_diagnostics import analyze_factor_variance_profile

# Z_samples shape: (n_chains, n_samples, n_subjects, K)
results = analyze_factor_variance_profile(
    Z_samples=Z_samples,
    variance_threshold=0.1,
    save_path="variance_profile.png"
)

print(f"Active factors: {results['n_active_factors']}")
print(f"Effective dimensionality: {results['effective_dimensionality']}")
print(f"Top 5 variances: {results['sorted_variances'][:5]}")
```

## Answering the Overparameterization Question

### Your Hypothesis

> "I want to verify that this is a real phenomenon and not an artifact of how stability is measured"

### How Variance Analysis Helps

**Real Phenomenon Signature**:
- K=2 run: 2 active factors, effective dim = 2
- K=20 run: 2 active factors, effective dim = 2  ← Same!
- K=50 run: 2 active factors, effective dim = 2  ← Same!
- **Interpretation**: ARD correctly identifies true dimensionality regardless of K

**Measurement Artifact Signature**:
- K=2 run: 2 stable factors, 2 active factors
- K=20 run: 8 stable factors, but only 2-3 active factors  ← Mismatch!
- K=50 run: 15 stable factors, but only 2-3 active factors  ← Worse mismatch!
- **Interpretation**: Stability metric inflated by K, variance reveals truth

**Cross-Check Logic**:
```python
if abs(n_stable_factors - n_active_factors) > 2:
    logger.warning("⚠️ Mismatch between stability and variance")
    logger.warning("   → May indicate measurement artifact")
```

## Expected Results for Your Runs

### config_K2_no_sparsity.yaml
- **Expected**: 2 active factors, effective dim = 2
- **Shrinkage**: N/A (K too small for ARD to matter)
- **Baseline** for comparison

### config_K20_no_sparsity.yaml
- **Healthy ARD**: 2-4 active factors, effective dim = 2-4
- **Poor ARD**: 10+ active factors, gradual decline
- **Test**: Does K=20 find same dimensionality as K=2?

### config_K50_no_sparsity.yaml
- **Healthy ARD**: 2-5 active factors (similar to K=20)
- **Artifact**: 15+ active factors (scales with K)
- **Critical Test**: If effective_dim(K=50) ≈ effective_dim(K=20), phenomenon is real!

## Interpretation Guide

### Scenario 1: Healthy ARD Across All K
```
K=2:  active=2, effective=2
K=20: active=3, effective=3, shrinkage_ratio=15
K=50: active=3, effective=3, shrinkage_ratio=20
```
**Conclusion**: ✅ Real phenomenon. ARD working correctly. True dimensionality ≈ 3.

### Scenario 2: Poor Convergence
```
K=2:  active=2, effective=2
K=20: active=12, effective=8, shrinkage_ratio=2.5
K=50: active=28, effective=15, shrinkage_ratio=1.8
```
**Conclusion**: ❌ Convergence issues. Increase warmup, check R-hat, adjust priors.

### Scenario 3: Stability Artifact
```
K=20: stable_factors=8, active=3
K=50: stable_factors=15, active=3
```
**Conclusion**: ⚠️ Stability metric may be artifact. Variance reveals true dim = 3.

## Technical Details

### Variance Calculation

The function computes variance correctly to preserve scale:

```python
# For each factor k:
for k in range(K):
    # Extract factor k across all chains, samples, subjects
    # Shape: (n_chains, n_samples, n_subjects)
    factor_k = Z_samples[:, :, :, k]

    # Compute variance across subjects for each (chain, sample)
    # Shape: (n_chains, n_samples)
    variances_per_sample = factor_k.var(axis=2)

    # Average across chains and samples
    factor_variances[k] = variances_per_sample.mean()
```

This preserves the scale of variance (typically 0.5-2.0 for active factors) while accounting for MCMC uncertainty.

### Why Not Just Average Then Compute Variance?

**Incorrect** (would give tiny variances):
```python
Z_mean = Z_samples.mean(axis=(0, 1))  # Average first
variance = Z_mean.var(axis=0)  # Then compute variance
# Result: variance ~ 0.001 (way too small!)
```

**Correct** (preserves variance scale):
```python
variances_per_sample = Z_samples.var(axis=2)  # Variance first
factor_variance = variances_per_sample.mean()  # Then average
# Result: variance ~ 1.2 (correct scale!)
```

## Files Modified

1. **`analysis/mcmc_diagnostics.py`**
   - Added `analyze_factor_variance_profile()` function (270 lines)
   - Comprehensive documentation and docstrings

2. **`experiments/robustness_testing.py`**
   - Integrated variance analysis into factor stability plotting
   - Added cross-check between stability and variance metrics
   - Logs warnings if mismatch detected

3. **`test_variance_profile.py`** (NEW)
   - Test suite with 4 scenarios
   - Validates correct ARD detection
   - Demonstrates healthy vs. poor shrinkage

## Next Steps

1. **Run Your K=2, K=20, K=50 Experiments**
   ```bash
   python run_experiments.py --config config_K2_no_sparsity.yaml --experiments factor_stability
   python run_experiments.py --config config_K20_no_sparsity.yaml --experiments factor_stability
   python run_experiments.py --config config_K50_no_sparsity.yaml --experiments factor_stability
   ```

2. **Compare Variance Profiles**
   - Check if effective dimensionality stays constant (real phenomenon)
   - Or if it scales with K (measurement artifact)

3. **Cross-Check Stability Results**
   - Does `n_stable_factors` match `n_active_factors`?
   - Large mismatch suggests stability metric issue

4. **Adjust Accordingly**
   - If ARD working: Use high K for exploration, trust variance profile
   - If artifact: Stick with lower K, investigate stability metric

## References

- Archambeau & Bach 2008: "Sparse Bayesian Factor Analysis"
- Tipping 2001: "Sparse Bayesian Learning and the Relevance Vector Machine"
- Gelman & Rubin 1992: "Inference from Iterative Simulation Using Multiple Sequences"

## Test Results

All tests passed successfully:

```
Test                           K       Active  Effective Status
----------------------------------------------------------------------
Healthy ARD (K=20)             20           3          3 ✅ HEALTHY
Poor Shrinkage (K=20)          20          18         13 ❌ POOR
Healthy ARD (K=50)             50           4          4 ✅ HEALTHY
Baseline (K=2)                 2            2          2 ✅ BASELINE
```

The variance analysis correctly distinguishes healthy ARD (sharp drop-off) from poor shrinkage (gradual decline) and accurately identifies effective dimensionality across different K values.
