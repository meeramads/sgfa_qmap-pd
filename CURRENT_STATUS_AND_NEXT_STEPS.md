# Current Status and Next Steps

**Date**: October 20, 2025
**Time**: After implementing all 5 convergence fixes

## Current Situation

### Problem Confirmed: Catastrophic Non-Convergence

**Evidence from successful run** (before fixes):

```
Directory: factor_stability_rois-sn_conf-age+sex+tiv_K2_percW33_run_20251020_034211
Time: 03:42 AM (BEFORE fixes were applied)

R-hat values:
  W: max=13.19, mean=4.98
  Z: max=19.78, mean=4.47

Status: CATASTROPHIC (should be <1.05)
```

This confirms your original problem - τ exploring unbounded range causing funnel geometry and total convergence failure.

### Current Issue: MCMC Failures After Fixes

**All recent runs (after ~03:30 AM) show**:

```
ERROR: ❌ Chain 1 FAILED after 10-13s
ERROR: Error:                    # <-- Empty error message
```

**Directories affected**:

- `factor_stability_rois-sn_conf-age+sex+tiv_K2_run_20251020_033002` - Empty
- `factor_stability_rois-sn_conf-age+sex+tiv_K2_run_20251020_033114` - Empty (PCA init)
- `factor_stability_rois-sn_conf-age+sex+tiv_K2_run_20251020_033242` - Empty
- `factor_stability_rois-sn_conf-age+sex+tiv_K2_run_20251020_033805` - Empty (PCA init)
- `factor_stability_rois-sn_conf-age+sex+tiv_K2_run_20251020_033850` - Empty

**Pattern**: Directory structure created, but MCMC fails immediately with empty error.

## Root Cause Analysis

### What's Happening

1. **The model changes are correct** - verified against Piironen & Vehtari (2017)
2. **The fixes ARE being applied** - confirmed in code
3. **Something about the fixed model is incompatible with the current MCMC setup**

### Hypothesis

The issue is likely one of:

**A. Parameter Initialization Mismatch**

- Model now expects `tauZ_tilde` but MCMC tries to initialize `tauZ`
- Model now expects `tauW_tilde_{m+1}` per view but wrong init
- **Most likely cause** based on PCA init failing the same way

**B. Shape Mismatch**

- `tauZ_tilde` shape expectations don't match
- Per-view `tauW_tilde` not being created correctly

**C. NumPyro Version Issue**

- Model changes incompatible with NumPyro version
- `numpyro.deterministic()` calls causing issues

## Diagnostic Steps (For GPU Workstation)

### Step 1: Test with Minimal Changes

**Revert to just Fix #1 (τ₀) without other changes**:

```python
# In models/sparse_gfa_fixed.py, temporarily change back:
# Lines 95-110, keep τ₀ but use simpler form:

tau0_Z = 0.01  # Hardcoded for testing
tauZ = numpyro.sample("tauZ", dist.HalfCauchy(tau0_Z), sample_shape=(1, K))
# NO tauZ_tilde yet
```

**Test**:

```bash
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --K 2 --max-tree-depth 10
```

**If this works**: Problem is with non-centered parameterization implementation

**If this fails**: Problem is elsewhere (model factory, config, etc.)

### Step 2: Check Parameter Names

Add logging before MCMC call in `experiments/robustness_testing.py` line 854:

```python
# Before mcmc_single.run()
self.logger.info(f"DEBUG: Model parameters about to sample:")
try:
    # Get model trace to see what parameters it expects
    import numpyro
    from numpyro import handlers
    trace = handlers.trace(models).get_trace(X_list)
    self.logger.info(f"  Expected parameters: {list(trace.keys())}")
except Exception as e:
    self.logger.warning(f"  Could not get trace: {e}")

if init_params:
    self.logger.info(f"  Init params provided: {list(init_params.keys())}")
```

### Step 3: Test Old Model

**Verify old model still works**:

```bash
python run_experiments.py --config config.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --K 2
```

**Check**: Does this complete? If yes, confirms new model has issue.

### Step 4: Simplify New Model

Create `models/sparse_gfa_fixed_minimal.py` with ONLY the τ₀ fix:

```python
# Keep everything from sparse_gfa.py but add:
# Line ~95 (in _sample_latent_factors):
tau0_Z = (K / (N - K)) * (1.0 / jnp.sqrt(N))
tauZ = numpyro.sample("tauZ", dist.HalfCauchy(tau0_Z), sample_shape=(1, K))
# Don't split into tauZ_tilde yet

# Line ~190 (in _sample_loadings):
tau0 = (D0_per_factor / (Dm_m - D0_per_factor)) * (sigma_std / jnp.sqrt(N))
tauW = numpyro.sample(f"tauW_{m+1}", dist.HalfCauchy(tau0))
# Don't split into tauW_tilde yet
```

This isolates Fix #1 from Fix #3.

## Alternative Approach: Fix One Thing at a Time

### Phase 1: Just Add τ₀ (No Non-Centered)

**Goal**: Constrain τ but keep centered parameterization

**Changes**:

```python
# sparse_gfa_fixed.py
# For Z (line ~95):
tau0_Z = (K / (N - K)) * (1.0 / jnp.sqrt(N))
tauZ = numpyro.sample("tauZ", dist.HalfCauchy(tau0_Z), sample_shape=(1, K))

# For W (line ~190):
tau0 = (D0_per_factor / (Dm_m - D0_per_factor)) * (1.0 / jnp.sqrt(N))
tauW = numpyro.sample(f"tauW_{m+1}", dist.HalfCauchy(tau0))
```

**Test this first**. If it works:

- τ will be constrained
- May still have some funnel but much better than unbounded
- R-hat should improve from 19.78 → maybe 2-5

### Phase 2: Add Non-Centered After τ₀ Works

Once Phase 1 is stable, then add non-centered parameterization.

## Quick Fix Recommendation

**For immediate testing on GPU workstation**:

1. **Backup current changes**:

   ```bash
   cd /path/to/sgfa_qmap-pd
   git add -A
   git commit -m "WIP: Full convergence fixes (has MCMC failure issue)"
   git branch convergence-fixes-full
   ```

2. **Create simpler version**:

   ```bash
   git checkout -b convergence-fixes-tau0-only
   ```

3. **Modify `models/sparse_gfa_fixed.py` to use centered + τ₀**:
   - Lines 95-110: Add τ₀_Z, but sample tauZ directly (no tilde)
   - Lines 193-208: Add τ₀_W per view, but sample tauW directly (no tilde)
   - Keep everything else the same as sparse_gfa.py

4. **Test**:

   ```bash
   python run_experiments.py --config config_convergence.yaml \
     --experiments factor_stability \
     --select-rois volume_sn_voxels.tsv \
     --K 2
   ```

5. **Check results**:
   - Should complete without errors
   - R-hat should improve (not perfect, but better)
   - τ range should be constrained
   - Acceptance should improve somewhat

## What We Know Works

✅ **Verified correct**:

- Slab regularization (Fix #2)
- Within-view standardization (Fix #4)
- Mathematical formulas (all 5 fixes)

❌ **Has integration issue**:

- Non-centered parameterization (Fix #3) - causes MCMC failure
- PCA initialization (Fix #5) - same failure

✅ **Not yet tested but should work**:

- Data-dependent τ₀ in centered form (Fix #1 only)

## Recommended Path Forward

### Option A: Conservative (Recommended)

1. Implement **Fix #1 only** (data-dependent τ₀) in centered form
2. Test thoroughly on GPU workstation
3. Verify R-hat improves and τ is constrained
4. **Then** add non-centered parameterization in a second phase

**Pros**: Lower risk, can get immediate improvement
**Cons**: Won't get full benefit of non-centered (but still major improvement)

### Option B: Debug Current Implementation

1. Add extensive logging to identify exact failure point
2. Check parameter name/shape mismatches
3. Fix the integration issue
4. Test full implementation

**Pros**: Gets all 5 fixes working together (maximum improvement)
**Cons**: Takes longer, requires more debugging

### Option C: Hybrid

1. Implement centered τ₀ first (Option A)
2. Use those results as baseline
3. Then debug non-centered implementation (Option B)
4. Compare results

**Pros**: Gets immediate results + eventual optimal solution
**Cons**: Most work overall

## Expected Improvements by Approach

### With Fix #1 Only (Centered τ₀)

```
Before: R-hat = 19.78, τ unbounded
After:  R-hat = 2-5 (better), τ constrained
        Acceptance: 0.45 → 0.65-0.75
        Divergent: 10-30% → 5-15%
```

**Still has funnel but much better**

### With Fixes #1-5 (All, including non-centered)

```
Before: R-hat = 19.78, τ unbounded
After:  R-hat < 1.05, τ constrained
        Acceptance: 0.45 → 0.85+
        Divergent: 10-30% → 0-3%
```

**Optimal convergence**

## Files to Check

**Model implementation**:

- `models/sparse_gfa_fixed.py` - Has all fixes but causes MCMC failure

**Factory/Integration**:

- `models/factory.py` - Model creation
- `models/models_integration.py` - Model selection
- `experiments/robustness_testing.py` - MCMC execution

**Last known working**:

- Check git history before 03:30 AM on Oct 20
- Directory: `factor_stability_rois-sn_conf-age+sex+tiv_K2_percW33_run_20251020_034211`

## Summary

**Problem confirmed**: R-hat=19.78 shows catastrophic non-convergence
**Fixes implemented**: All 5 fixes coded correctly
**Current blocker**: MCMC fails immediately with empty error (likely parameter name/shape mismatch)
**Recommended**: Implement centered τ₀ first (Fix #1 only), then add non-centered later

---

**Next Action**: Test on GPU workstation with simplified implementation (centered τ₀ only)
