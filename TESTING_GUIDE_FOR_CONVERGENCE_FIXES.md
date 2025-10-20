# Testing Guide: Convergence Fixes Implementation

**Date**: October 20, 2025
**Status**: All 5 fixes implemented, ready for testing on GPU workstation

## What Was Implemented

### ✅ Critical Fixes to `sparse_gfa_fixed` Model

All 5 convergence fixes from Piironen & Vehtari (2017) are now implemented:

1. **Data-Dependent Global Scale τ₀** - FIXED
2. **Slab Regularization** - VERIFIED (was already correct)
3. **Non-Centered Parameterization** - FIXED
4. **Within-View Standardization** - VERIFIED (was already correct)
5. **PCA Initialization** - IMPLEMENTED (has integration issue, see below)

## Files Modified

### Core Model Fix
**File**: `models/sparse_gfa_fixed.py`

**Changes**:
- Lines 95-110: Added data-dependent τ₀ for Z global shrinkage
- Line 104-107: Made tauZ non-centered (τ = τ₀ × τ_tilde)
- Line 114: Changed TruncatedCauchy → HalfCauchy for local scales λ
- Line 163: Changed TruncatedCauchy → HalfCauchy for W local scales

**Impact**: Prevents τ from exploring extreme values [-400, +200], constrains to reasonable range based on data.

### PCA Initialization System
**Files Created**:
- `core/pca_initialization.py` - Complete PCA initialization module

**Files Modified**:
- `experiments/robustness_testing.py` - Lines 839-856, 987-1007, 2970
- `run_experiments.py` - Line 775
- `core/config_schema.py` - Line 133
- `config_convergence.yaml` - Line 43

**Status**: ⚠️ **PCA init has integration issue** - works in isolation but fails in full pipeline. Temporarily disabled for testing.

## Testing Procedure

### Test 1: Verify τ₀ Constraint (Most Important)

This tests that the critical fix for unbounded τ exploration is working.

```bash
# Run on GPU workstation
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --regress-confounds age sex tiv \
  --K 2 --max-tree-depth 10
```

**What to check in logs**:

```
✅ EXPECTED (Good):
tau0_Z: 0.00257                    # Data-dependent constraint
tau0_view_1: 0.0532                # Imaging view constraint
tau0_view_2: 0.0539                # Clinical view constraint
Acceptance probability: 0.85+      # Should improve significantly
Divergent transitions: 0-3%        # Should be very low
```

```
❌ IF YOU SEE (Bad):
τ values in trace plots: [-400, +200]  # Unbounded exploration
Acceptance probability: 0.45           # Poor
Divergent transitions: 10-30%          # High
```

**Where to find τ values**:
- Check MCMC trace plots in results directory
- Look for `tau0_Z`, `tau0_view_1`, `tau0_view_2` in logs
- Trace plots should show tauZ exploring ~[0, 0.01], NOT [-400, +200]

### Test 2: Compare Before/After Convergence

**Before fix** (baseline - use old code):
```bash
git stash  # Temporarily save new changes
python run_experiments.py --config config.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --K 2
git stash pop  # Restore new changes
```

**After fix** (with convergence fixes):
```bash
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --K 2 --max-tree-depth 10
```

**Compare**:
| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Acceptance prob | 0.45 | 0.85+ |
| Divergent trans | 10-30% | 0-3% |
| R-hat | >10 | <1.05 |
| τ range | [-400, +200] | [0, ~0.1] |

### Test 3: Verify Non-Centered Parameterization

**Check in model code** (`models/sparse_gfa_fixed.py`):

```python
# Should see (lines 95-110):
tau0_Z = (K / (N - K)) * (1.0 / jnp.sqrt(N))
tauZ_tilde = numpyro.sample("tauZ_tilde", dist.HalfCauchy(1.0), ...)
tauZ = tau0_Z * tauZ_tilde  # Non-centered!
numpyro.deterministic("tau0_Z", tau0_Z)
```

**Should NOT see**:
```python
# OLD (wrong):
tauZ = numpyro.sample("tauZ", dist.HalfCauchy(1.0), ...)  # Centered, unbounded
```

### Test 4: Check Slab Regularization

**Config should have** (`config_convergence.yaml`):
```yaml
slab_df: 4
slab_scale: 2
```

**Model should compute** (check lines 119-138, 168-219):
```python
cW_tilde ~ InverseGamma(2.0, 2.0)
cW_squared = (slab_scale ** 2) * cW_tilde
lmbW_tilde = sqrt((cW² × λ²) / (cW² + τ² × λ²))  # Regularized!
```

### Test 5: Within-View Standardization

**Check preprocessing logs**:
```
Processing view: imaging (shape: (86, 536))
Applying standard scaling to imaging

Processing view: clinical (shape: (86, 9))
Applying standard scaling to clinical
```

**Verify in results**:
- Each view should have mean≈0, std≈1 **independently**
- NOT global standardization across all features

## Expected Results

### τ₀ Values for Your Data

With N=86, K=2, percW=33%:

```
Z global shrinkage:
  D₀ = K = 2
  τ₀_Z = (2/(86-2)) × (1/√86) = 0.00257

W imaging view (D=536):
  D₀ = 0.33 × 536 = 177
  τ₀_imaging = (177/(536-177)) × (1/√86) = 0.0532

W clinical view (D=9):
  D₀ = 0.33 × 9 = 3
  τ₀_clinical = (3/(9-3)) × (1/√86) = 0.0539
```

**These should appear in logs as `tau0_Z`, `tau0_view_1`, `tau0_view_2`.**

### Convergence Improvements

**Piironen & Vehtari (2017)** tested on microarray data with D/N ratios 28-99:
- Standard horseshoe: 1-30% divergent transitions
- Regularized horseshoe: 0-3% divergent transitions

**Your data** has D/N = 6.3, which is **much easier**. You should see:
- Acceptance probability: >0.85
- Divergent transitions: 0-3%
- R-hat: <1.05 for all parameters
- τ exploring constrained range [0, ~0.1]

## Known Issues

### PCA Initialization Integration Issue

**Status**: ⚠️ Implemented but has integration bug

**Symptoms**:
```
INFO: Using PCA initialization for MCMC
ERROR: ❌ Chain 1 FAILED after 10.0s
ERROR: Error:                           # <-- Empty error message
```

**What works**:
- PCA initialization module creates init_params correctly
- Standalone test with MCMC succeeds
- Module is well-tested in isolation

**What doesn't work**:
- Integration with full robustness_testing pipeline
- Empty error message suggests bare exception or assertion failure
- Likely parameter shape/name mismatch when NumPyro applies init_params

**Workaround**:
Set `use_pca_initialization: false` in `config_convergence.yaml` (already done).

**Impact**:
- PCA init helps avoid mode trapping (K=2 has 8 modes: 2^K × K!)
- Not critical for regularized horseshoe to work
- Main fixes (#1-4) are more important and working

**To debug** (on your workstation):
1. Enable PCA init: `use_pca_initialization: true`
2. Add detailed logging to `experiments/robustness_testing.py` line 854
3. Print shapes of all init_params before `mcmc_single.run()`
4. Compare with model's expected parameter shapes

## Debugging Tips

### If Acceptance Probability Still Low

**Check**:
1. Is `tau0_Z` appearing in logs? If not, fix didn't apply
2. Are trace plots showing bounded τ? Should be ~[0, 0.1]
3. Is `model_type: "sparse_gfa_fixed"` in config?
4. Are logs showing "sparse_gfa_fixed" model selected?

**Try**:
```bash
# Increase target acceptance probability
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --K 2 --max-tree-depth 10 \
  --target-accept-prob 0.90
```

### If τ Still Exploring Extreme Values

**This means the fix didn't apply.** Check:

1. Verify you're using `sparse_gfa_fixed`, not `sparse_gfa`:
   ```bash
   grep "model_type" config_convergence.yaml
   # Should show: model_type: "sparse_gfa_fixed"
   ```

2. Check model code has the fix:
   ```bash
   grep -A 5 "tau0_Z = " models/sparse_gfa_fixed.py
   # Should show the formula
   ```

3. Verify logs show model selection:
   ```
   ✅ Selected model type: sparse_gfa_fixed
   ```

### If Divergent Transitions Still High

**Even with all fixes**, you might see some divergent transitions if:
- max_tree_depth too low (try 12-14)
- target_accept_prob too low (try 0.90)
- num_warmup too low (try 2000+)

**Tune MCMC**:
```yaml
# In config_convergence.yaml
factor_stability:
  num_warmup: 2000
  max_tree_depth: 12
  target_accept_prob: 0.90
```

## What to Report Back

After testing, please share:

### 1. Log Excerpts
```
tau0_Z: [value]
tau0_view_1: [value]
tau0_view_2: [value]
Acceptance probability: [value]
Divergent transitions: [X]% ([count] out of [total])
```

### 2. Convergence Diagnostics
```
R-hat values (from stability analysis output):
  tauZ: [value]
  W: [value]
  Z: [value]
```

### 3. τ Range
From trace plots or summary statistics:
```
tauZ explored: [min, max]
tauW explored: [min, max]
```

### 4. Comparison
If you tested before/after:
```
                Before    After
Acceptance:     [X]       [Y]
Divergent %:    [X]       [Y]
R-hat (worst):  [X]       [Y]
```

## Files to Check

**Results directory structure**:
```
results/factor_stability_rois-sn_conf-age+sex+tiv_K2_run_[timestamp]/
├── 03_factor_stability/
│   ├── plots/
│   │   ├── mcmc_trace_diagnostics.pdf    # Check τ trace plots
│   │   ├── hyperparameter_posteriors.pdf # Check τ posteriors
│   │   └── hyperparameter_traces.pdf     # Check τ evolution
│   └── stability_analysis/
│       ├── rhat_convergence_diagnostics.json  # Check R-hat values
│       └── factor_stability_summary.json      # Overall convergence
└── experiments.log  # Full execution log
```

## Reference Documentation

- **[CONVERGENCE_FIXES_COMPLETE_SUMMARY.md](CONVERGENCE_FIXES_COMPLETE_SUMMARY.md)** - Technical details of all 5 fixes
- **[REGULARIZED_HORSESHOE_VERIFICATION.md](REGULARIZED_HORSESHOE_VERIFICATION.md)** - Implementation verification
- **[docs/troubleshooting/NON_CENTERED_PARAMETERIZATION_FIX.md](docs/troubleshooting/NON_CENTERED_PARAMETERIZATION_FIX.md)** - τ₀ fix details
- **[SESSION_SUMMARY_OCT20_2025.md](SESSION_SUMMARY_OCT20_2025.md)** - Complete session log

## Quick Verification Checklist

Before running on GPU workstation:

- [ ] Pulled latest code with all fixes
- [ ] `models/sparse_gfa_fixed.py` has `tau0_Z` calculation (line 101)
- [ ] `models/sparse_gfa_fixed.py` has non-centered `tauZ` (line 107)
- [ ] `config_convergence.yaml` has `model_type: "sparse_gfa_fixed"`
- [ ] `config_convergence.yaml` has `use_pca_initialization: false`
- [ ] GPU workstation has sufficient memory (4+ chains × ~2GB each)

After running:

- [ ] Logs show `tau0_Z: 0.00257` (approximately)
- [ ] Logs show `tau0_view_1: 0.053` (approximately)
- [ ] Trace plots show τ in range [0, ~0.1], NOT [-400, +200]
- [ ] Acceptance probability >0.80
- [ ] Divergent transitions <5%
- [ ] R-hat <1.05 for main parameters

---

**Status**: Ready for GPU workstation testing
**Priority**: Test Fix #1 (data-dependent τ₀) first - this is the most critical fix
**Expected outcome**: Significantly improved convergence with bounded τ exploration
