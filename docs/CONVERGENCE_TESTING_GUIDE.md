# Convergence Testing Guide

## Overview

Test the `sparse_gfa_fixed` model with all 5 convergence fixes across different factor counts (K).

**Key Difference**: `config_convergence.yaml` uses `model_type: "sparse_gfa_fixed"` instead of the default `"sparse_gfa"`. The fixed model implements:
1. Data-dependent global scale τ₀
2. Proper slab regularization
3. Non-centered parameterization
4. Within-view standardization
5. PCA initialization support

## Quick Start

### Run All Tests
```bash
bash run_convergence_tests.sh
```

### Run Single Test (K=2)
```bash
python run_experiments.py \
  --config config_convergence.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --K 2 --max-tree-depth 10
```

### Run with Custom Parameters
```bash
python run_experiments.py \
  --config config_convergence.yaml \
  --experiments all \
  --select-rois volume_sn_voxels.tsv \
  --regress-confounds age sex tiv \
  --qc-outlier-threshold 3.0 \
  --percW 33 --K 2 --max-tree-depth 10
```

## Configuration Files

All configurations use **max_tree_depth=10** by default (can be overridden via command line):

| Config File | K Value | Description |
|-------------|---------|-------------|
| `config_convergence.yaml` | 20 (default) | Main convergence test config |
| Command-line override | Any | Use `--K <value>` to test different factor counts |

## Expected Results

### Good Convergence Signs
- Acceptance probability: 0.75-0.90
- R-hat values: < 1.01
- No divergences
- Stable factor loadings across chains

### Warning Signs
- Acceptance probability: < 0.6 or > 0.95
- R-hat values: > 1.1
- Many divergences
- High variance in factor estimates

## MCMC Parameters

### Default Settings (sparse_gfa_fixed)
```yaml
target_accept_prob: 0.8
max_tree_depth: 10
dense_mass: false
num_warmup: 3000
num_samples: 10000
```

### If Convergence Issues Persist
Try increasing adaptation:
```bash
--target-accept-prob 0.9 --max-tree-depth 10
```

Or enable dense mass matrix (better adaptation, more memory):
```yaml
dense_mass: true
```

## Verification

Check the logs for:
```
NUTS kernel parameters: target_accept_prob=0.8, max_tree_depth=10
sample: 100%|...| acc. prob=0.87
```

And results in:
- `results/*/03_factor_stability/stability_analysis/factor_stability_summary.json`
- `results/*/03_factor_stability/plots/mcmc_trace_diagnostics.pdf`

## Troubleshooting

**Low acceptance probability (< 0.6):**
- Increase `target_accept_prob` to 0.9
- Enable `dense_mass: true`
- Increase `num_warmup`

**Hitting max tree depth consistently:**
- This is normal! Not a problem for convergence
- Don't increase `max_tree_depth` - it may worsen acceptance

**R-hat > 1.01:**
- Increase `num_warmup` and `num_samples`
- Check for data issues (outliers, missingness)

See `docs/troubleshooting/RECENT_SESSION_SUMMARY.md` for more details.
