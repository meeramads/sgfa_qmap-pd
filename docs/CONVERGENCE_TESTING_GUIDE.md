# Convergence Testing Guide

## Overview

Guide for testing MCMC convergence with the `sparse_gfa_fixed` model using `config_convergence.yaml`.

**Model**: The fixed model (`sparse_gfa_fixed`) implements all convergence improvements:

1. Data-dependent global scale τ₀ (Piironen & Vehtari 2017)
2. Proper slab regularization (InverseGamma prior)
3. Non-centered parameterization for better geometry
4. Regularized horseshoe priors

## Quick Start

### Factor Stability Analysis (Production)

```bash
python run_experiments.py \
  --config config_convergence.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --regress-confounds age sex tiv \
  --K 20 --percW 33 --qc-outlier-threshold 3.0
```

### Test Different K Values

```bash
# K=2 (smallest)
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability --select-rois volume_sn_voxels.tsv \
  --K 2 --max-tree-depth 10

# K=10 (medium)
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability --select-rois volume_sn_voxels.tsv \
  --K 10 --max-tree-depth 13

# K=20 (default)
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability --select-rois volume_sn_voxels.tsv \
  --K 20 --max-tree-depth 13
```

### Test Different MAD Thresholds

```bash
# Stringent QC (cleanest signal)
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability --select-rois volume_sn_voxels.tsv \
  --qc-outlier-threshold 2.5

# Moderate QC (default)
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability --select-rois volume_sn_voxels.tsv \
  --qc-outlier-threshold 3.0

# Permissive QC (preserve heterogeneity)
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability --select-rois volume_sn_voxels.tsv \
  --qc-outlier-threshold 5.0
```

## Configuration

### Main Config: `config_convergence.yaml`

**Model Parameters**:

```yaml
model:
  model_type: "sparse_gfa_fixed"
  K: 20
  percW: 33
  slab_df: 4
  slab_scale: 2
  reghsZ: true
```

**MCMC Settings**:

```yaml
factor_stability:
  num_samples: 10000
  num_warmup: 3000
  num_chains: 4
  target_accept_prob: 0.8
  max_tree_depth: 13
```

**Preprocessing**:

```yaml
preprocessing:
  qc_outlier_threshold: 3.0
  variance_threshold: 0.0       # Disabled
  roi_based_selection: false    # Disabled
```

## Command-Line Overrides

```bash
--K <value>                     # Number of factors
--percW <value>                 # Sparsity percentage
--max-tree-depth <value>        # NUTS tree depth limit
--qc-outlier-threshold <value>  # MAD threshold
--target-accept-prob <value>    # NUTS acceptance probability
--select-rois <file>            # ROI selection
--regress-confounds <vars>      # Confound regression
```

## Convergence Diagnostics

### Good Convergence

**Acceptance Probability**: 0.6-0.8 (NUTS optimal range)

```
INFO: Mean acceptance probability: 0.79
```

**R-hat**: < 1.1 for all parameters

```
INFO: R-hat for W: max=1.02, mean=1.01
INFO: R-hat for Z: max=1.03, mean=1.01
```

**Effective Factors**: Match expected K

```
INFO: 2/2 factors are effective (100.0%)
```

**Global Shrinkage**: Reasonable τ_W values

```
INFO: τ_W (global shrinkage, view 1): 0.053 ± 0.002
```

### Warning Signs

**Low Acceptance**: < 0.5

```
WARNING: Mean acceptance probability: 0.42
Action: Decrease max_tree_depth or increase target_accept_prob
```

**High R-hat**: > 1.1

```
WARNING: R-hat for W: max=1.15, mean=1.08
Action: Increase num_warmup or num_samples
```

**Over-shrinkage**: τ_W extremely small

```
WARNING: τ_W (global shrinkage): 0.0005 ± 0.0001
Action: Check data quality (likely too much noise)
```

**No Effective Factors**: 0/K factors

```
WARNING: 0/2 factors are effective (0.0%)
Action: Check preprocessing (likely MAD threshold too permissive)
```

## Real Examples

### Success: K=2, MAD 3.0

**Preprocessing**:

- 1794 → 531 voxels (70.4% removed)

**Convergence**:

- Acceptance: 0.79
- R-hat (W): max=1.02
- R-hat (Z): max=1.03
- Effective factors: 2/2 (100%)
- τ_W: 0.053 ± 0.002

**Interpretation**: Perfect convergence, model found 2 stable factors.

### Failure: K=2, MAD 100.0

**Preprocessing**:

- 1794 → 1776 voxels (1% removed - essentially no QC)

**Convergence**:

- Acceptance: varies
- R-hat (W): max=1339 (catastrophic)
- R-hat (Z): max=109 (catastrophic)
- Effective factors: 0/2 (0%)
- τ_W: 0.0005 ± 0.0001 (over-shrinkage)

**Interpretation**: Complete failure. Too much noise without QC causes regularized horseshoe to shrink everything to zero.

## Troubleshooting

### Issue: Low Acceptance Rate

**Symptom**: Acceptance < 0.5

**Solutions**:

```bash
# Option 1: Increase target acceptance
--target-accept-prob 0.85

# Option 2: Decrease tree depth (more conservative steps)
--max-tree-depth 10
```

### Issue: High R-hat

**Symptom**: R-hat > 1.1

**Solutions**:

```bash
# Increase warmup
# Edit config_convergence.yaml:
factor_stability:
  num_warmup: 5000  # Was 3000

# Or increase samples
  num_samples: 15000  # Was 10000
```

### Issue: Divergences

**Symptom**: Many divergent transitions

**Solutions**:

```bash
# Increase acceptance probability
--target-accept-prob 0.9

# Use dense mass matrix (more memory, better adaptation)
# Edit config_convergence.yaml:
factor_stability:
  dense_mass: true
```

### Issue: All Factors Shrunk to Zero

**Symptom**: 0/K effective factors

**Root Cause**: Usually too much noise in data

**Solutions**:

```bash
# Decrease MAD threshold (stricter QC)
--qc-outlier-threshold 2.5

# Check data quality manually
python -c "import pandas as pd; X = pd.read_csv('...');
print('Mean:', X.mean().mean()); print('Std:', X.std().mean())"
```

## MCMC Parameter Guide

### Tree Depth (`max_tree_depth`)

**Default**: 13

- Higher: Better exploration, slower, more memory
- Lower: Faster, less exploration

**When to adjust**:

- K ≤ 5: Use 10
- K = 10-20: Use 13
- K > 20: May need 15 (memory permitting)

### Acceptance Probability (`target_accept_prob`)

**Default**: 0.8

- Higher (0.85-0.9): More careful steps, better for complex posteriors
- Lower (0.6-0.7): Larger steps, faster but riskier

**When to adjust**:

- Divergences: Increase to 0.85-0.9
- Very slow: Decrease to 0.7 (but watch R-hat)

### Warmup/Samples

**Default**: 3000 warmup + 10000 samples per chain

**Guidelines**:

- Minimum: 1000 warmup + 2000 samples (testing only)
- Production: 3000 warmup + 10000 samples
- Difficult convergence: 5000 warmup + 15000 samples

## Results Interpretation

### Stability Analysis

**Stability Rate**: % of factors appearing in >50% of chains

```
INFO: Stability rate: 100.0%  # All factors stable
```

**Consensus Loadings**: Available when factors are stable

```
results/.../stability_analysis/consensus_factor_loadings.csv
```

**Similarity Matrix**: Cross-chain factor matching

```
results/.../stability_analysis/similarity_matrix.csv
```

### Factor Effectiveness

**Threshold**: Factors with >5% non-zero loadings

**Good**: Most factors effective

```
INFO: 18/20 factors are effective (90.0%)
```

**Warning**: High shrinkage

```
WARNING: 5/20 factors are effective (25.0%)
Action: May indicate K too high or poor data quality
```

## Performance Notes

**Typical Runtime** (4 chains × 13K samples):

- K=2: ~30 min
- K=10: ~2 hours
- K=20: ~4-6 hours

**Memory Usage**:

- K=2: ~4-8 GB
- K=10: ~8-12 GB
- K=20: ~12-16 GB

**Scaling**: Roughly O(K²) in time and memory

## References

- Piironen & Vehtari 2017: Regularized horseshoe priors
- Ferreira et al. 2024: Factor stability methodology
- Hoffman & Gelman 2014: NUTS sampler
