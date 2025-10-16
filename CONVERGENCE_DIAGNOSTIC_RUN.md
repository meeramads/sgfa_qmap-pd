# Convergence Diagnostic Run: No Sparsity Configuration

## Purpose

Test whether convergence issues (R-hat > 12) are caused by complex horseshoe priors.

## Configuration

**File**: `config_K2_no_sparsity.yaml`

### Key Changes from Production

| Setting | Production (sparse_gfa) | Diagnostic (standard_gfa) |
|---------|------------------------|---------------------------|
| **Model Type** | `sparse_gfa` (horseshoe) | `standard_gfa` (ARD) |
| **W Prior** | Regularized Horseshoe | Normal(0,1) + ARD |
| **Z Prior** | Regularized Horseshoe | Normal(0,1) |
| **Sparsity** | percW=33% constraint | No constraint (ARD learns) |
| **K (factors)** | Variable | **2** (minimal) |
| **MCMC Samples** | Variable | **2000** |
| **MCMC Warmup** | Variable | **1000** |
| **Chains** | Variable | **4** |
| **PCA** | Optional | **Disabled** |
| **MAD Threshold** | 4.0 | **4.0** (same) |
| **ROI** | Configurable | **sn** (substantia nigra) |
| **Confounds** | Configurable | **age, sex, tiv** |

## What Each Model Uses

### Sparse GFA (Current - Convergence Issues)

```python
# Horseshoe priors
tauW ~ TruncatedCauchy(scale=f(percW, N))  # Complex shrinkage
lmbW ~ TruncatedCauchy(scale=1)            # Per-element shrinkage
cW ~ InverseGamma(...)                     # Slab regularization

tauZ ~ TruncatedCauchy(scale=1)            # Complex shrinkage
lmbZ ~ TruncatedCauchy(scale=1)            # Per-element shrinkage
cZ ~ InverseGamma(...)                     # Slab regularization

W ~ Normal(0, 1) * shrinkage_factor        # Heavily regularized
Z ~ Normal(0, 1) * shrinkage_factor        # Heavily regularized
```

**Pros**: Adaptive sparsity, interpretable loadings
**Cons**: Complex posterior geometry, hard to sample

### Standard GFA (Diagnostic - Simple Priors)

```python
# Simple ARD priors
alpha ~ Gamma(1e-3, 1e-3)                  # Simple precision

W ~ Normal(0, 1) / sqrt(alpha)             # Scaled by ARD
Z ~ Normal(0, 1)                           # Simple Gaussian
```

**Pros**: Simple posterior, easier to sample
**Cons**: Less sparse, less interpretable

## How to Run

```bash
# Run factor stability analysis with diagnostic config
python run_experiments.py --config config_K2_no_sparsity.yaml --experiments robustness_testing

# Or run via experiment runner
cd /Users/meera/Desktop/sgfa_qmap-pd
python -m experiments.robustness_testing --config config_K2_no_sparsity.yaml
```

## Expected Runtime

- **4 chains × (1000 warmup + 2000 samples)** = ~6-8 hours on CPU
- Faster than horseshoe because simpler priors → faster sampling

## Output Location

```
results/robustness_testing_K2_no_sparsity_run_{timestamp}/
├── 03_factor_stability/
│   ├── plots/
│   │   ├── mcmc_trace_diagnostics.png      ⭐ CHECK THIS
│   │   ├── mcmc_parameter_distributions.png
│   │   ├── factor_stability_heatmap.png
│   │   ├── hyperparameter_posteriors.png   ⚠️  Will show 'alpha' not 'tauW'
│   │   └── hyperparameter_traces.png
│   ├── chains/
│   │   ├── chain_0/
│   │   ├── chain_1/
│   │   ├── chain_2/
│   │   └── chain_3/
│   └── stability_analysis/
│       ├── rhat_convergence_diagnostics.json  ⭐ CHECK THIS
│       └── stability_results.json
└── summaries/
    └── complete_experiment_summary.yaml
```

## Interpreting Results

### Scenario 1: Convergence Improves (R-hat < 1.1) ✅

**Diagnosis**: Horseshoe priors are too complex for your data

**Actions**:
1. **Relax horseshoe priors**:
   - Increase Cauchy scale parameter
   - Decrease `percW` (allow less sparsity)
   - Try `percW: 50` or `percW: 70`

2. **Try simpler sparsity priors**:
   - Laplace priors: `W ~ Laplace(0, tau)`
   - Spike-and-slab: Binary inclusion indicators

3. **Adjust MCMC settings**:
   - Increase warmup: `num_warmup: 5000`
   - Use smaller step sizes (if using NUTS)

### Scenario 2: Convergence Still Poor (R-hat > 1.1) ❌

**Diagnosis**: Problem is more fundamental than priors

**Possible causes**:

1. **Multimodality** (different factor orderings)
   - Look for: Separated traces in `mcmc_trace_diagnostics.png`
   - Solution: Use label-switching algorithms, identifiability constraints

2. **K too small/large**
   - K=2 may be insufficient to capture variation
   - Try: K=3, K=5, K=10
   - Check: Effective factors after shrinkage

3. **Data quality issues**
   - MAD filtering too aggressive (losing signal)
   - Try: `mad_threshold: 3.0` (less aggressive)
   - Check: How many voxels remain after filtering

4. **Model misspecification**
   - Gaussian noise assumption violated
   - Data not mean-centered properly
   - Outliers driving behavior

### Scenario 3: Partial Improvement (R-hat: 1.5-3.0) ⚠️

**Diagnosis**: Priors contribute but aren't sole cause

**Actions**:
1. Combine simpler priors + more MCMC samples
2. Try intermediate sparsity (between horseshoe and ARD)
3. Check for specific problematic parameters (W vs Z vs hyperparameters)

## Key Diagnostics to Check

### 1. R-hat Values

```bash
# Check the saved diagnostics
cat results/*/03_factor_stability/stability_analysis/rhat_convergence_diagnostics.json
```

Look for:
- `rhat_W_max` < 1.1 (good convergence)
- `rhat_Z_max` < 1.1 (good convergence)
- `rhat_alpha_max` < 1.1 (hyperparameters converged)

### 2. Trace Plots

Open `mcmc_trace_diagnostics.png` and check:
- **Overlapping traces**: Chains agree ✅
- **Separated traces**: Chains in different modes ❌
- **Drifting traces**: Need more warmup ⚠️
- **Rapid mixing**: Good autocorrelation ✅

### 3. Factor Stability

Check `factor_stability_heatmap.png`:
- **High similarity (>0.8)**: Factors stable across chains ✅
- **Low similarity (<0.5)**: Factor matching failed ❌

### 4. Effective Sample Size (ESS)

From the summary output, check:
- ESS > 400 per parameter (good)
- ESS < 100 per parameter (poor mixing)

## Comparison Table: Prior Geometries

| Aspect | Horseshoe | ARD |
|--------|-----------|-----|
| **Posterior modes** | Multiple (label switching) | Multiple (label switching) |
| **Posterior geometry** | Complex (heavy tails) | Simple (Gaussian-like) |
| **Shrinkage type** | Strong sparsity | Smooth shrinkage |
| **Sampling difficulty** | Hard | Easy |
| **Interpretability** | High (sparse) | Medium (smooth) |
| **Parameters** | Many (tau, lambda, c, ...) | Few (alpha) |

## Follow-up Experiments

Based on results, run:

### If standard_gfa converges:

```bash
# Try intermediate sparsity
# Edit config to use sparse_gfa with relaxed priors:
percW: 50  # Instead of 33
slab_scale: 5  # Instead of 2 (wider slab)
```

### If standard_gfa doesn't converge:

```bash
# Try more factors
cp config_K2_no_sparsity.yaml config_K5_no_sparsity.yaml
# Edit: K: 5

# Try less aggressive MAD filtering
# Edit: mad_threshold: 3.0  # Instead of 4.0
```

## References

### Horseshoe Prior
- Carvalho et al. (2010): "The horseshoe estimator for sparse signals"
- Piironen & Vehtari (2017): "Sparsity information and regularization in the horseshoe and other shrinkage priors"

### ARD Prior
- MacKay (1995): "Probable networks and plausible predictions"
- Bishop (2006): "Pattern Recognition and Machine Learning" (Chapter 7)

### Convergence Diagnostics
- Vehtari et al. (2021): "Rank-normalization, folding, and localization: An improved R-hat"
- Gelman & Rubin (1992): "Inference from iterative simulation using multiple sequences"

## Notes

- Standard GFA has **no tauW/tauZ** hyperparameters
  - Instead uses **alpha** (ARD precision)
  - Hyperparameter plots will show alpha instead
- Standard GFA is **less memory efficient**
  - May need to reduce voxel count if memory issues
  - Try: `mad_threshold: 5.0` for fewer voxels
- Standard GFA provides **baseline comparison**
  - Shows if ANY MCMC sampling is problematic
  - Or if specifically horseshoe priors
