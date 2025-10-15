# PCA Quick Start Guide

## Simple Command-Line Interface

### Basic Usage

```bash
# Baseline (no PCA)
python run_experiments.py --experiments factor_stability --K 5

# With PCA (balanced strategy - 85% variance)
python run_experiments.py --experiments factor_stability --K 5 --pca-strategy balanced

# With PCA (aggressive - 80% variance)
python run_experiments.py --experiments factor_stability --K 5 --pca-strategy aggressive

# With PCA (conservative - 90% variance)
python run_experiments.py --experiments factor_stability --K 5 --pca-strategy conservative
```

---

## Testing Different Configurations

### Phase 1: Compare approaches (K=5)

```bash
# Baseline
python run_experiments.py --experiments factor_stability --K 5

# PCA balanced
python run_experiments.py --experiments factor_stability --K 5 --pca-strategy balanced
```

Compare the results (stability rate, N/D ratio, runtime) to see if PCA helps.

### Phase 2: Optimize K (if PCA wins)

```bash
# Test different K values with PCA
python run_experiments.py --experiments factor_stability --K 3 --pca-strategy balanced
python run_experiments.py --experiments factor_stability --K 8 --pca-strategy balanced
python run_experiments.py --experiments factor_stability --K 10 --pca-strategy balanced
```

---

## Advanced Options

### Custom PCA variance threshold

```bash
python run_experiments.py --experiments factor_stability --K 5 --enable-pca --pca-variance 0.87
```

### Fixed number of PCA components

```bash
python run_experiments.py --experiments factor_stability --K 5 --enable-pca --pca-components 120
```

### Combine with other options

```bash
# PCA + specific ROI
python run_experiments.py \
    --experiments factor_stability \
    --K 8 \
    --pca-strategy balanced \
    --select-rois volume_sn_voxels.tsv

# PCA + confound regression
python run_experiments.py \
    --experiments factor_stability \
    --K 8 \
    --pca-strategy balanced \
    --regress-confounds age sex tiv
```

---

## PCA Strategies

| Strategy | Variance | Expected Components | N/D Ratio @ K=10 | Use Case |
|----------|----------|---------------------|------------------|----------|
| **aggressive** | 80% | ~90 | 0.89:1 ✓ | Maximum speed |
| **balanced** ⭐ | 85% | ~120 | 0.74:1 ✓ | Recommended default |
| **conservative** | 90% | ~150 | 0.59:1 ✓ | Maximum information |

---

## What Gets Saved

With PCA enabled, factor stability results include:

```
03_factor_stability/chains/chain_0/
├── W_pcs_view_0.csv       # PC-space loadings (50 × K)
├── W_voxels_view_0.csv    # Voxel-space loadings (850 × K) ← USE FOR BRAIN MAPPING
├── Z.csv                   # Factor scores
└── metadata.json

PCA_BRAIN_REMAPPING_README.md  # Auto-generated instructions
```

**Your MATLAB brain remapping code works unchanged** - just use `W_voxels` files!

---

## Expected Benefits

| Metric | Without PCA | With PCA (balanced) | Improvement |
|--------|-------------|---------------------|-------------|
| Features | 1,334 | ~120 | **91% reduction** |
| N/D ratio (K=10) | 0.06:1 ⚠️ | 0.74:1 ✓ | **12x better** |
| Runtime | ~60 min | ~12 min | **5x faster** |
| Stability | ~75% | ~75% | Maintained |

---

## Permanent Configuration

Once you find your optimal settings, update `config.yaml`:

```yaml
# config.yaml
preprocessing:
  strategy: "pca_balanced"  # Uses 85% variance

model:
  K: 8  # Or whatever K worked best
  percW: 20
```

Then run without flags:
```bash
python run_experiments.py --experiments factor_stability
```

---

## Parallel Testing on Multiple Machines

```bash
# Machine 1: Baseline
ssh gpu1
python run_experiments.py --experiments factor_stability --K 5

# Machine 2: PCA balanced
ssh gpu2
python run_experiments.py --experiments factor_stability --K 5 --pca-strategy balanced

# Machine 3: PCA + K=8
ssh gpu3
python run_experiments.py --experiments factor_stability --K 8 --pca-strategy balanced

# Machine 4: PCA + K=10
ssh gpu4
python run_experiments.py --experiments factor_stability --K 10 --pca-strategy balanced
```

Compare results and pick the winner!

---

## Full Documentation

See [docs/PCA_CONFIGURATION_GUIDE.md](docs/PCA_CONFIGURATION_GUIDE.md) for:
- Detailed PCA explanation
- Brain remapping details
- Connection to MAD elbow analysis
- Mathematical background

---

**That's it! Simple command-line flags for testing different configurations.** 🚀
