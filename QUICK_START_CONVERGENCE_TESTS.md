# Quick Start: Convergence Tests

## 🚀 Run All Tests
```bash
bash run_convergence_tests.sh
```

## 🎯 Run Single Test (K=2)
```bash
python run_experiments.py \
    --config config_convergence_K2_tree10.yaml \
    --experiments factor_stability \
    --select-rois volume_sn_voxels.tsv
```

## 📊 Available Configurations

| Command | K Value | Config File |
|---------|---------|-------------|
| `bash run_convergence_tests.sh K2` | 2 | `config_convergence_K2_tree10.yaml` |
| `bash run_convergence_tests.sh K5` | 5 | `config_convergence_K5_tree10.yaml` |
| `bash run_convergence_tests.sh K8` | 8 | `config_convergence_K8_tree10.yaml` |
| `bash run_convergence_tests.sh K20` | 20 | `config_convergence_K20_tree10.yaml` |
| `bash run_convergence_tests.sh K50` | 50 | `config_convergence_K50_tree10.yaml` |

## ⚙️ Test Settings

All tests use:
- ✅ Model: `sparse_gfa_fixed` (with all 5 convergence fixes)
- ✅ Data: SN voxels + clinical (1,811 features)
- ✅ Chains: 4 × (2000 samples + 1000 warmup)
- ✅ MCMC: target_accept_prob=0.95, max_tree_depth=10
- ✅ Sparsity: percW=33%, slab_df=4, slab_scale=2

## 📈 Expected Results

**Target**: R-hat < 1.01 for all parameters

**Baseline (K=2)**: R-hat = 12.6 (catastrophic) ❌
**Fixed (K=2)**: R-hat < 1.01 (excellent) ✅

## 🕐 Estimated Runtime (GPU)

- K=2: ~5 hours
- K=5: ~6 hours
- K=8: ~7 hours
- K=20: ~8 hours
- K=50: ~10 hours

**Total (all tests)**: ~36 hours sequential

## 📂 Results Location

```
results/factor_stability_rois-sn_run_[timestamp]/03_factor_stability/
```

## 🔍 Check Results

```bash
# View R-hat diagnostics
cat results/factor_stability_rois-sn_run_*/03_factor_stability/stability_analysis/rhat_convergence_diagnostics.json

# Compare K=2 with baseline
python scripts/compare_convergence.py \
    results/all_rois-sn_conf-age+sex+tiv_K2_percW33_MAD3.0_run_20251018_142445/03_factor_stability \
    results/factor_stability_rois-sn_run_*/03_factor_stability
```

## ⚠️ Important

**Always use `--select-rois volume_sn_voxels.tsv`** to load only SN data (matching baseline)!

Without this flag, it will load all 4 ROIs (30k features) which is different from baseline.
