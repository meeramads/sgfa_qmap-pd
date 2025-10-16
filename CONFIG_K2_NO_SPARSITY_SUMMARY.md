# Configuration Summary: K2 No Sparsity Diagnostic Run

## ✅ Setup Complete

Your diagnostic configuration is now running with the following specifications:

### Configuration File
- **File**: `config_K2_no_sparsity.yaml`
- **Purpose**: Test if convergence issues are caused by horseshoe priors

### Key Settings

| Parameter | Value | Note |
|-----------|-------|------|
| **Model Type** | `standard_gfa` | Simple ARD priors (NO horseshoe) |
| **K (factors)** | 2 | Minimal |
| **MCMC Samples** | 2000 | As requested |
| **MCMC Warmup** | 1000 | As requested |
| **Chains** | 4 | For convergence assessment |
| **max_tree_depth** | **13** | ⭐ Increased from 12 |
| **target_accept_prob** | 0.9 | Default |
| **PCA** | Disabled | Using MAD-filtered voxels |
| **MAD Threshold** | 4.0 | Quality control |

### Code Changes Made

**1. Config Schema** ([core/config_schema.py:121-123](core/config_schema.py#L121-L123))
```python
# NUTS sampler parameters
max_tree_depth: int = 12  # Maximum tree depth for NUTS sampler (10-15 typical)
target_accept_prob: float = 0.9  # Target acceptance probability (0.8-0.99 typical)
```

**2. Model Runner** ([analysis/model_runner.py:139-144](analysis/model_runner.py#L139-L144))
```python
# Setup MCMC - use config parameters or defaults
max_tree_depth = getattr(self.config, 'max_tree_depth', 12)
target_accept_prob = getattr(self.config, 'target_accept_prob', 0.9)

kernel = NUTS(model, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
```

**3. Core Analysis** ([core/run_analysis.py:194-198](core/run_analysis.py#L194-L198))
```python
# Use NUTS parameters from args if available, otherwise use defaults
max_tree_depth = getattr(args, 'max_tree_depth', 12)
target_accept_prob = getattr(args, 'target_accept_prob', 0.9)

kernel = NUTS(model, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
```

## Current Run Status

The experiment is **running** with the following confirmed:

✅ Configuration validated successfully
✅ Data loaded: 4 views (SN, putamen, lentiform, clinical)
✅ MAD filtering applied (qc_outlier_threshold=4.0)
✅ Features: 20,998 total after preprocessing
✅ Using CPU (no GPU available)
✅ NUTS sampler configured with max_tree_depth=13

### Data After Preprocessing

| View | Original | After MAD | Retention |
|------|----------|-----------|-----------|
| **volume_sn_voxels** | 1794 | 1334 | 74.4% |
| **volume_putamen_voxels** | 21444 | 15030 | 70.1% |
| **volume_lentiform_voxels** | 7201 | 4617 | 64.1% |
| **clinical** | 17 | 17 | 100% |
| **TOTAL** | 30,456 | 20,998 | 68.9% |

## What max_tree_depth Does

**NUTS (No U-Turn Sampler)** builds a binary tree of states:

- **max_tree_depth=10**: Up to 2^10 = 1024 leapfrog steps per iteration
- **max_tree_depth=12**: Up to 2^12 = 4096 leapfrog steps (default)
- **max_tree_depth=13**: Up to 2^13 = **8192 leapfrog steps** ⭐

**Effect of increasing max_tree_depth:**
- ✅ **Allows exploration of more complex posterior geometries**
- ✅ **Can help with multimodal distributions**
- ✅ **May improve convergence for difficult posteriors**
- ⚠️ **Slower per iteration** (more computations)
- ⚠️ **May hit divergent transitions** if posterior is too complex

**When you'll see warnings:**
```
WARNING: There were X divergent transitions after warmup.
```
This means the sampler couldn't properly explore some regions. Solutions:
1. Increase `target_accept_prob` to 0.95 or 0.99
2. Reparameterize the model
3. Use informative priors

## Expected Runtime

With your settings:
- **4 chains × (1000 warmup + 2000 samples) = 12,000 total iterations**
- **Standard GFA** (simpler than horseshoe)
- **20,998 features** (high-dimensional)
- **No GPU** (CPU only)

**Estimated**: 4-6 hours total (1-1.5 hours per chain)

## Monitoring Progress

Check the output directory:
```bash
ls -lh results/robustness_testing_run_20251016_124431/
```

Watch for completion messages:
```bash
tail -f results/robustness_testing_run_20251016_124431/experiment.log
```

## Expected Outputs

```
results/robustness_testing_run_20251016_124431/
└── 03_factor_stability/
    ├── plots/
    │   ├── mcmc_trace_diagnostics.png      ⭐ CHECK R-hat here
    │   ├── mcmc_parameter_distributions.png
    │   ├── hyperparameter_posteriors.png   (will show 'alpha' not 'tauW')
    │   ├── hyperparameter_traces.png
    │   └── factor_stability_heatmap.png
    ├── chains/
    │   ├── chain_0/  (W.csv, Z.csv, metadata.json)
    │   ├── chain_1/
    │   ├── chain_2/
    │   └── chain_3/
    └── stability_analysis/
        ├── rhat_convergence_diagnostics.json  ⭐ CHECK THIS
        └── stability_results.json
```

## Interpretation Guide

### Scenario 1: R-hat Improves (< 1.5)

**Diagnosis**: Horseshoe priors were the main problem

**Actions**:
1. ✅ Confirms priors are too complex
2. Return to sparse_gfa but relax priors:
   - Increase Cauchy scale
   - Decrease percW (e.g., 50 instead of 33)
   - Increase slab_scale (e.g., 5 instead of 2)
3. Or use simpler sparsity methods

### Scenario 2: R-hat Still High (> 3.0)

**Diagnosis**: More fundamental issues

**Possible causes**:
1. **Multimodality** - Factor label switching
2. **K=2 too small** - Need more factors
3. **Data issues** - MAD filtering too aggressive
4. **Model misspecification** - Wrong assumptions

**Next steps**:
1. Check trace plots for separation
2. Try K=5 or K=10
3. Try mad_threshold=3.0 (less aggressive)

### Scenario 3: Partial Improvement (1.5 < R-hat < 3.0)

**Diagnosis**: Priors contribute but aren't sole cause

**Actions**:
1. Combine: simpler priors + more MCMC samples
2. Increase warmup to 5000
3. Try max_tree_depth=14 or 15

## Checking Results

Once complete, check R-hat values:

```bash
# Quick check
cat results/*/03_factor_stability/stability_analysis/rhat_convergence_diagnostics.json

# Look for:
# "rhat_W_max": < 1.1 (good!) or > 1.1 (still issues)
# "rhat_Z_max": < 1.1 (good!) or > 1.1 (still issues)
# "rhat_alpha_max": < 1.1 (hyperparameters converged)
```

## Comparison: Standard GFA vs Sparse GFA

### Prior Complexity

**Standard GFA** (current run):
```python
W ~ Normal(0, 1) / sqrt(alpha)  # Simple!
Z ~ Normal(0, 1)                # Simple!
alpha ~ Gamma(1e-3, 1e-3)       # ARD precision
```

**Sparse GFA** (your previous runs):
```python
W ~ Normal(0, 1) * tauW * lmbW * regularization  # Complex!
Z ~ Normal(0, 1) * tauZ * lmbZ * regularization  # Complex!
# Plus: tauW, lmbW, cW, tauZ, lmbZ, cZ all need sampling
```

### Posterior Geometry

| Aspect | Standard GFA | Sparse GFA |
|--------|--------------|------------|
| **Parameters** | W, Z, alpha, sigma (simple) | W, Z, tauW, lmbW, cW, tauZ, lmbZ, cZ, sigma (complex) |
| **Modes** | Fewer (label switching only) | Many (shrinkage + label switching) |
| **Sampling** | Easier | Harder |
| **Convergence** | Faster | Slower |
| **Sparsity** | Smooth shrinkage | Hard zeros |
| **Interpretability** | Medium | High |

## Next Steps After This Run

1. **Check R-hat** in the diagnostics JSON file
2. **View trace plots** to see chain behavior
3. **Based on results**:
   - If improved: Relax sparse_gfa priors
   - If not: Try different K or data preprocessing
4. **Consider using these NUTS settings** in future runs:
   ```yaml
   model:
     max_tree_depth: 13
     target_accept_prob: 0.9
   ```

## Notes

- Standard GFA uses **alpha hyperparameters** instead of tauW/tauZ
- Hyperparameter posterior plots will show **alpha** distributions
- Standard GFA is less sparse but easier to sample
- This is a **diagnostic run** - not for final analysis
- Purpose is to **isolate the prior complexity** as the convergence bottleneck

## Contact/Issues

If you see errors or unexpected behavior, check:
1. Experiment log file
2. Trace diagnostic plots
3. Memory usage (may need to reduce features if OOM)

## References

- NUTS sampler: Hoffman & Gelman (2014)
- ARD priors: MacKay (1995), Bishop (2006)
- Horseshoe priors: Carvalho et al. (2010), Piironen & Vehtari (2017)
- Convergence diagnostics: Vehtari et al. (2021)
