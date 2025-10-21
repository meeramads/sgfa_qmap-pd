# Quick Reference: Convergence Fixes

## Test Command (GPU Workstation)

```bash
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --regress-confounds age sex tiv \
  --K 2 --max-tree-depth 10
```

## What to Check

### 1. Logs (Critical)

```
✅ tau0_Z: 0.00257
✅ tau0_view_1: 0.0532
✅ tau0_view_2: 0.0539
✅ Acceptance probability: 0.85+
✅ Divergent transitions: 0-3%
```

### 2. Trace Plots

```
results/.../03_factor_stability/plots/mcmc_trace_diagnostics.pdf
```

- tauZ should be in [0, ~0.01]
- NOT in [-400, +200]

### 3. Convergence

```
results/.../03_factor_stability/stability_analysis/rhat_convergence_diagnostics.json
```

- All R-hat < 1.05

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| τ range | [-400, +200] | [0, ~0.1] |
| Acceptance | 0.45 | 0.85+ |
| Divergent % | 10-30% | 0-3% |
| R-hat | >10 | <1.05 |

## Files Changed

**Critical fix**:

- `models/sparse_gfa_fixed.py` (lines 95-110) - Added τ₀ for Z

**Config**:

- `config_convergence.yaml` (line 43) - PCA init disabled
- `core/config_schema.py` (line 133) - Added config field

**PCA init** (has bug, disabled):

- `core/pca_initialization.py` (created)
- Integration in robustness_testing.py

## 5 Fixes Status

1. ✅ Data-dependent τ₀ - **FIXED**
2. ✅ Slab regularization - Verified
3. ✅ Non-centered param - **FIXED**
4. ✅ Within-view scaling - Verified
5. ⚠️ PCA initialization - Implemented but disabled (has integration issue)

## Documentation

- **HANDOFF_SUMMARY.md** - Start here
- **TESTING_GUIDE_FOR_CONVERGENCE_FIXES.md** - Detailed testing
- **CONVERGENCE_FIXES_COMPLETE_SUMMARY.md** - Technical details

## Troubleshooting

**If acceptance still low**:

```bash
--target-accept-prob 0.90
```

**If divergent trans still high**:

```bash
--max-tree-depth 12
```

**If τ still unbounded**:

- Check model_type is "sparse_gfa_fixed"
- Verify tau0_Z appears in logs
- Check model code has the fix

## Report Back

Share these from your run:

1. tau0_Z, tau0_view_1, tau0_view_2 values
2. Acceptance probability
3. Divergent transitions %
4. R-hat values
5. τ range from traces
