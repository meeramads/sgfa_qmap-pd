# Convergence Tests Quick Start

## Overview

Test the `sparse_gfa_fixed` model with all 5 convergence fixes.

**Key Difference**: `config_convergence.yaml` is identical to `config.yaml` except it uses `model_type: "sparse_gfa_fixed"` instead of `"sparse_gfa"`.

## Quick Commands

### Single Test
```bash
python run_experiments.py \
  --config config_convergence.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --K 2 \
  --max-tree-depth 10 \
  --qc-outlier-threshold 3.0
```

### Run All K Values (2, 5, 8, 20, 50)
```bash
bash run_convergence_tests.sh
```

### Run Specific K Values
```bash
bash run_convergence_tests.sh 2 5 8
```

## Customization

### Edit Script Parameters
Edit `run_convergence_tests.sh` (lines 14-18):
```bash
MAX_TREE_DEPTH=10      # NUTS max tree depth
MAD_THRESHOLD=3.0      # MAD outlier threshold
TARGET_ACCEPT=0.95     # MCMC acceptance probability
PERCW=33               # Sparsity level %
```

### Override Individual Test
```bash
python run_experiments.py \
  --config config_convergence.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --K 20 \
  --percW 50 \
  --max-tree-depth 12 \
  --qc-outlier-threshold 5.0 \
  --target-accept-prob 0.8
```

## Available Flags

| Flag | Description | Example |
|------|-------------|---------|
| `--K` | Number of latent factors | `--K 20` |
| `--percW` | Sparsity level (%) | `--percW 33` |
| `--max-tree-depth` | NUTS max tree depth | `--max-tree-depth 10` |
| `--qc-outlier-threshold` | MAD threshold | `--qc-outlier-threshold 3.0` |
| `--target-accept-prob` | MCMC acceptance probability | `--target-accept-prob 0.95` |
| `--select-rois` | ROI selection | `--select-rois volume_sn_voxels.tsv` |

## Comparing with Baseline

To match the Oct 18 baseline (522 voxels with MAD 3.0):

```bash
python run_experiments.py \
  --config config_convergence.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --K 2 \
  --qc-outlier-threshold 3.0
```

The config has `enable_spatial_processing: true` which is required for MAD filtering to match baseline behavior.

## Expected Results

- **Baseline R-hat**: 12.6 (catastrophic divergence)
- **Target R-hat**: < 1.01 (good convergence)
- **Feature count** (with MAD 3.0 + spatial processing): ~522 voxels
- **Feature count** (without spatial processing): ~850 voxels

## 5 Convergence Fixes Implemented

1. ✅ Data-dependent global scale τ₀
2. ✅ Proper slab regularization (InverseGamma(2,2))
3. ✅ Non-centered parameterization
4. ✅ Within-view standardization
5. ✅ Enhanced MCMC settings (target_accept_prob=0.95, max_tree_depth=10)
