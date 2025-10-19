# Multi-K Convergence Test Suite

Tests the `sparse_gfa_fixed` model across different factor counts (K) to evaluate convergence performance.

## Configuration Files

All configurations use **max_tree_depth=10** and include all 5 convergence fixes:

| Config File | K Value | Description |
|-------------|---------|-------------|
| `config_convergence_K2_tree10.yaml` | K=2 | Baseline comparison (matches original failing run) |
| `config_convergence_K5_tree10.yaml` | K=5 | Low complexity |
| `config_convergence_K8_tree10.yaml` | K=8 | Medium complexity |
| `config_convergence_K20_tree10.yaml` | K=20 | Standard complexity |
| `config_convergence_K50_tree10.yaml` | K=50 | High complexity |

## Running Tests

### Run All Tests (Sequential)
```bash
bash run_convergence_tests.sh
```
**Estimated time**: ~20-25 hours total on GPU (5 tests × 4-5 hours each)

### Run Individual Tests
```bash
# Single test
bash run_convergence_tests.sh K2

# Multiple specific tests
bash run_convergence_tests.sh K2 K5 K8

# Or run directly
python run_experiments.py \
    --config config_convergence_K2_tree10.yaml \
    --experiments factor_stability \
    --select-rois volume_sn_voxels.tsv
```

## What Each Test Does

Each test runs:
- **4 MCMC chains** (for R-hat calculation)
- **2000 samples + 1000 warmup** per chain
- **SN voxels + clinical data** (1,811 features)
- **target_accept_prob=0.95, max_tree_depth=10**

## Expected Results

### Convergence Metrics (Target: R-hat < 1.01)
- **K=2**: Should achieve R-hat < 1.01 (transforms baseline's 12.6)
- **K=5**: Should achieve R-hat < 1.01
- **K=8**: Should achieve R-hat < 1.01
- **K=20**: Should achieve R-hat < 1.01
- **K=50**: May be challenging, but should improve significantly

### Performance Metrics
- **Acceptance probability**: Target 0.90-0.95
- **Divergences**: < 1% of samples
- **ESS**: > 400 effective samples

## Results Location

Results are saved to:
```
results/factor_stability_rois-sn_run_[timestamp]/03_factor_stability/
```

Key files:
- `stability_analysis/rhat_convergence_diagnostics.json` - R-hat values
- `stability_analysis/factor_stability_summary.json` - Stability metrics
- `chains/chain_*/` - Individual chain results

## Comparing with Baseline

For K=2, compare with the catastrophic baseline:
```bash
python scripts/compare_convergence.py \
    results/all_rois-sn_conf-age+sex+tiv_K2_percW33_MAD3.0_run_20251018_142445/03_factor_stability \
    results/factor_stability_rois-sn_run_[timestamp]/03_factor_stability
```

Expected improvement: **R-hat 12.6 → <1.01**

## Convergence Fixes Tested

All configurations include:

1. ✅ **Data-dependent τ₀** (Piironen & Vehtari 2017)
2. ✅ **Proper slab regularization** (InverseGamma(2,2))
3. ✅ **Non-centered parameterization** (for W and Z)
4. ✅ **Within-view standardization** (preprocessing)
5. ✅ **Enhanced MCMC** (target_accept_prob=0.95, max_tree_depth=10)

## Performance Expectations

Based on Chain 1 of K=2 test (~1.59s/iteration on GPU):

| K | Iterations/Chain | Est. Time/Chain | Est. Total (4 chains) |
|---|------------------|-----------------|----------------------|
| 2 | 3000 | 79 min | ~5.3 hours |
| 5 | 3000 | ~90 min | ~6 hours |
| 8 | 3000 | ~100 min | ~6.7 hours |
| 20 | 3000 | ~120 min | ~8 hours |
| 50 | 3000 | ~150 min | ~10 hours |

**Total for all tests**: ~36 hours sequentially, or ~8-10 hours if run in parallel (5 jobs)

## Troubleshooting

### Memory Issues
If you encounter OOM errors with K=50:
- Reduce `num_samples` to 1000
- Reduce `num_chains` to 2
- Use `dense_mass: false` (already set)

### Slow Sampling
If iterations are very slow (>10s each):
- Check if running on CPU (should be GPU)
- Verify data is SN-only (use --select-rois flag)
- Check if hitting max_tree_depth frequently (increase if needed)

### Configuration Errors
Ensure you use the `--select-rois volume_sn_voxels.tsv` flag to load only SN data, matching the baseline.

## Notes

- All tests use the **same data** (SN + clinical) for fair comparison
- `max_tree_depth=10` is more conservative than the K=2 baseline (which used 12)
- Results should demonstrate that convergence fixes work across different K values
- Higher K values may require more samples for stability analysis
