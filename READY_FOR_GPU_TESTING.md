# Ready for GPU Workstation Testing

**Status**: Enhanced logging added, ready for debugging on GPU workstation
**Date**: 2025-10-20
**Commits**: 1aaa169, 29c76b2

## Summary

All convergence fixes from Piironen & Vehtari (2017) are implemented in the code, but MCMC fails immediately with empty error messages. Enhanced logging has been added to identify the exact failure point and cause.

## What's Been Done

### ✅ Implementation Complete
1. **Fix #1**: Data-dependent τ₀ for both Z and W ([sparse_gfa_fixed.py:95-110](models/sparse_gfa_fixed.py#L95-L110))
2. **Fix #2**: Slab regularization with proper InverseGamma(2,2) ([sparse_gfa_fixed.py:117-128](models/sparse_gfa_fixed.py#L117-L128))
3. **Fix #3**: Non-centered parameterization ([sparse_gfa_fixed.py:93-107](models/sparse_gfa_fixed.py#L93-L107))
4. **Fix #4**: Within-view standardization (verified in preprocessing)
5. **Fix #5**: PCA initialization module ([core/pca_initialization.py](core/pca_initialization.py))

### ✅ Enhanced Logging Added
- **Model**: Tracks all dimensions, τ₀ calculations, sampling steps, shapes
- **MCMC**: Full tracebacks, context on failures, init_params details
- **PCA Init**: Shows all parameters created with shapes and dtypes

### ✅ Documentation
- [LOGGING_ENHANCED_TEST_GUIDE.md](LOGGING_ENHANCED_TEST_GUIDE.md) - Complete testing guide
- [TESTING_GUIDE_FOR_CONVERGENCE_FIXES.md](TESTING_GUIDE_FOR_CONVERGENCE_FIXES.md) - Original test guide
- [CURRENT_STATUS_AND_NEXT_STEPS.md](CURRENT_STATUS_AND_NEXT_STEPS.md) - Detailed status

## The Problem

**Before fixes**: R-hat = 19.78 (catastrophic non-convergence)
- Evidence: [results/factor_stability_rois-sn_conf-age+sex+tiv_K2_percW33_run_20251020_034211](results/factor_stability_rois-sn_conf-age+sex+tiv_K2_percW33_run_20251020_034211/03_factor_stability_fixed_K2_percW33_slab4_2_MAD3.0_tree10/stability_analysis/rhat_convergence_diagnostics.json)

**After fixes**: MCMC fails immediately with empty error
- All runs after ~03:30 AM Oct 20 produce empty result directories
- Error message is blank, suggesting parameter mismatch

**Now**: Enhanced logging should reveal the exact issue

## Quick Start on GPU Workstation

### 1. Pull Latest Code
```bash
git pull origin master
# Should show commits: 1aaa169, 29c76b2, 4a093a6
```

### 2. Run Basic Test (No PCA Init)
```bash
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --regress-confounds age sex tiv \
  --K 2 --max-tree-depth 10
```

### 3. Check Logs
```bash
# Find latest results directory
ls -lt results/factor_stability* | head -1

# View logs
cd results/factor_stability_rois-sn_conf-age+sex+tiv_K2_run_*/
tail -f experiments.log

# Search for key info
grep "Calculated τ₀" experiments.log
grep "FAILED\|Error type" experiments.log
grep "Model dimensions" experiments.log
```

## What You Should See

### If Logging Works (Expected)
```
🔵 SparseGFAFixedModel.__call__ starting
  Model dimensions: N=86, M=2, D=864, K=2, percW=33
  View dimensions: [850, 14]

🟢 _sample_latent_factors starting
  Calculated τ₀_Z = 0.002571 (from D₀=2, N=86)
  Sampling tauZ_tilde ~ HalfCauchy(1.0)...

🚀 Starting MCMC sampling for chain 1...
  Model: SparseGFAFixedModel
  Warmup: 3000, Samples: 10000

❌ Chain 1 FAILED after 10.5s
Error type: ValueError
Error message: [detailed message]
Full traceback: [complete stack trace]
Context: Model=SparseGFAFixedModel, K=2, N=86
```

### Expected τ₀ Values (for Validation)
- **τ₀_Z** ≈ 0.00257 (K=2, N=86)
- **τ₀_W_imaging** ≈ 0.0532 (D≈850, percW=33%)
- **τ₀_W_clinical** ≈ 0.539 (D=14, percW=33%)

If you see different values, the formula might be wrong.

## Debugging Steps

### Step 1: Identify Failure Point
Look for where logging stops:
- Before model call? → Model instantiation issue
- During Z sampling? → Z parameter issue
- During W sampling? → W parameter issue
- After all sampling? → Likelihood computation issue

### Step 2: Check Error Details
With new logging, you'll see:
- **Error type**: ValueError, TypeError, RuntimeError, etc.
- **Error message**: Specific details
- **Context**: Model, dimensions, parameters
- **Traceback**: Exact line in code

### Step 3: Test PCA Init Separately
If Step 1 works without PCA init, enable it:
```yaml
# config_convergence.yaml line 43
use_pca_initialization: true
```

Then re-run and compare logs.

## Likely Scenarios

### Scenario A: Shape Mismatch
**Symptoms**: "operands could not be broadcast", "shape mismatch"
**Cause**: Parameter dimensions don't match model expectations
**Fix**: Adjust initialization shapes

### Scenario B: Invalid Values
**Symptoms**: "invalid value encountered", NaN or Inf
**Cause**: Numerical instability in τ₀ calculation or sampling
**Fix**: Add bounds checks, use jnp.clip()

### Scenario C: Missing Parameters
**Symptoms**: "KeyError: 'param_name'", "unexpected keyword"
**Cause**: Model expects different parameters than provided
**Fix**: Align parameter names between model and init

### Scenario D: NumPyro Internal
**Symptoms**: Error in numpyro internals, not our code
**Cause**: NumPyro version incompatibility or JAX issue
**Fix**: Check NumPyro/JAX versions, try different NUTS settings

## Information Needed from GPU Test

Please collect and share:

1. **Full error output** (with new logging, should be detailed)
2. **Calculated τ₀ values** (grep "Calculated τ₀" experiments.log)
3. **Model dimensions** (grep "Model dimensions" experiments.log)
4. **Failure point** (where logging stops)
5. **Parameter shapes** if PCA init used

## Next Steps After Results

### If Clear Error Appears
→ Debug specific issue based on error type

### If Still Empty Error
→ Very unlikely now, but would suggest deeper NumPyro issue

### If Works but Convergence Bad
→ Check τ ranges, may need to tune τ₀ formula

### If Everything Works
→ Run full analysis with K=20 and multiple sparsity levels

## Files Modified (Recent Commits)

**Commit 1aaa169**: Add comprehensive logging
- `models/sparse_gfa_fixed.py` - Model logging
- `experiments/robustness_testing.py` - MCMC logging
- `core/pca_initialization.py` - Init logging

**Commit 29c76b2**: Add test guide
- `LOGGING_ENHANCED_TEST_GUIDE.md`

**Commit 4a093a6**: Add documentation
- `CURRENT_STATUS_AND_NEXT_STEPS.md`
- `HANDOFF_SUMMARY.md`
- `TESTING_GUIDE_FOR_CONVERGENCE_FIXES.md`

## Reference Links

- **Piironen & Vehtari (2017)**: https://arxiv.org/abs/1707.01694
- **Regularized Horseshoe**: Sections 2.3-2.4 in paper
- **Non-centered Parameterization**: Section 3.2 in paper
- **NumPyro Docs**: https://num.pyro.ai/en/stable/

## Contact & Questions

After running the test, please report:
- ✅ MCMC started successfully? (yes/no)
- ✅ τ₀ values calculated? (what values?)
- ✅ Where did it fail? (specific function)
- ✅ Error type and message? (full details)
- ✅ Parameter shapes? (if available)

With this information, we can quickly identify and fix the issue.

---

**Ready to test**: YES ✅
**GPU required**: YES (CPU test would take >8 hours)
**Expected duration**: ~5-10 minutes on GPU for K=2 test
**Risk**: Low (just testing, not changing model code)
