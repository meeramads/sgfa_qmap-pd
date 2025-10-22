# Enhanced Logging Test Guide for GPU Workstation

## Overview

Comprehensive logging has been added to identify where MCMC is failing and why. The empty error messages should now show full context and tracebacks.

## What Was Added

### 1. Model Logging (`models/sparse_gfa_fixed.py`)

- **Entry point**: Shows model dimensions (N, M, D, K, percW)
- **τ₀ calculations**: Logs data-dependent global scale values

  ```
  Calculated τ₀_Z = 0.002571 (from D₀=2, N=86)
  Calculated τ₀_W_view1 = 0.053200
  ```

- **Sampling steps**: Each numpyro.sample() call is logged with shapes
- **Regularization**: Shows when slab regularization formula is applied
- **Completion**: Marks when each function completes successfully

### 2. MCMC Execution Logging (`experiments/robustness_testing.py`)

- **PCA initialization**: Shows if/when PCA init is used and parameter names
- **Chain start**: Logs model class, warmup/samples, init_params status
- **Error handling**: Full traceback + context on failures:

  ```
  Error type: ValueError
  Error message: Shape mismatch...
  Context: Model=SparseGFAFixedModel, K=2, N=86
  Init params used: True
  Init param keys: ['Z_raw', 'W_raw', ...]
  ```

### 3. PCA Initialization Logging (`core/pca_initialization.py`)

- **Input shapes**: Shows view shapes before PCA
- **PCA computation**: Variance explained, n_components
- **Parameter creation**: Every init param with name, shape, dtype

  ```
  Z_raw: shape=(86, 2), dtype=float32
  tauW_tilde_1: shape=(), dtype=float32
  ```

## Test Commands

### Basic Test (No PCA Init)

```bash
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --regress-confounds age sex tiv \
  --K 2 --max-tree-depth 10
```

### With PCA Initialization

First enable in `config_convergence.yaml` line 43:

```yaml
use_pca_initialization: true  # Change from false
```

Then run same command as above.

## What to Look For in Logs

### 1. **If MCMC Starts Successfully**

Look for:

```
🔵 SparseGFAFixedModel.__call__ starting
  Model dimensions: N=86, M=2, D=864, K=2, percW=33
  🟢 _sample_latent_factors starting
    Calculated τ₀_Z = 0.002571
```

**Expected τ₀ values for K=2, N=86, percW=33:**

- τ₀_Z ≈ 0.00257
- τ₀_W_imaging ≈ 0.0532 (for D=850 after QC)
- τ₀_W_clinical ≈ 0.539 (for D=14)

### 2. **If MCMC Fails During Initialization**

Look for where logging stops:

```
🚀 Starting MCMC sampling for chain 1...
  Model: SparseGFAFixedModel
  Warmup: 3000, Samples: 10000
  Using init_params: False
❌ Chain 1 FAILED after 10.5s
Error type: ValueError
Error message: operands could not be broadcast together with shapes (86,2) (86,20)
Full traceback: [...]
```

### 3. **Parameter Shape Issues**

The logging will now show all shapes:

```
Z_raw: shape=(86, 2), dtype=float32
tauZ_tilde: shape=(1, 2), dtype=float32
lmbZ: shape=(86, 2), dtype=float32
```

If there's a mismatch, you'll see it immediately.

### 4. **PCA Init Issues**

If PCA init is enabled:

```
🔧 Creating PCA initialization for chain 1...
✓ PCA initialization created with params: ['Z_raw', 'W_raw', 'tauZ_tilde', ...]
```

Or if it fails:

```
⚠️ PCA initialization failed, using default: Shape mismatch...
Traceback: [full details]
```

## Debugging Strategy

### Step 1: Test WITHOUT PCA Init First

```yaml
# config_convergence.yaml
use_pca_initialization: false
```

Run and collect logs. Look for:

1. Does model initialization complete?
2. What τ₀ values are calculated?
3. Where does it fail (if it fails)?

### Step 2: If Step 1 Works, Test WITH PCA Init

```yaml
use_pca_initialization: true
```

Compare logs to see if PCA init introduces issues.

### Step 3: Analyze Failure Point

**Scenario A: Fails in Model Initialization**

- Check for τ₀ calculation errors
- Look for shape mismatches in Z_raw or W_raw

**Scenario B: Fails During Warmup**

- Check acceptance probability
- Look for divergent transitions
- Check τ ranges in traces

**Scenario C: Silent Failure (Empty Error)**

- Should NOT happen anymore with enhanced logging
- If it does, likely NumPyro internal issue

## Expected Output Locations

### Log Files

```
results/factor_stability_rois-sn_conf-age+sex+tiv_K2_run_YYYYMMDD_HHMMSS/
  experiments.log          # Main log with all our new markers
  03_factor_stability/
    chains/chain_0/        # Individual chain outputs (if successful)
```

### Grep for Key Information

```bash
# Find τ₀ values
grep "Calculated τ₀" experiments.log

# Find failures
grep "FAILED\|Error type" experiments.log

# Find PCA init info
grep "PCA init" experiments.log

# Find model dimensions
grep "Model dimensions" experiments.log

# Track chain progress
grep "Chain [0-9]" experiments.log
```

## Key Insights Expected

### If Original Problem Still Exists (R-hat > 10)

You'll see:

- MCMC completes but with warnings
- Chains finish successfully
- R-hat diagnostics show high values
- τ ranges are still wide (not constrained by τ₀)

This would mean the fixes didn't work as intended.

### If New Problem (MCMC Fails)

You'll see:

- Detailed error at specific point
- Context about what was being sampled
- Parameter shapes and values
- Stack trace

This is what we're debugging now.

### If Fixes Work

You'll see:

- All chains complete successfully
- τ values constrained: τ_Z ≈ 0.0026, τ_W ≈ 0.05
- Acceptance prob > 0.85
- R-hat < 1.05
- No divergent transitions

## Next Steps After Test

### Scenario 1: Clear Error Message Now Appears

Share the error type, message, and context. We can fix the specific issue.

### Scenario 2: Works But Convergence Still Bad

Check the τ ranges and R-hat values. May need to adjust τ₀ formula.

### Scenario 3: Everything Works

Celebrate and run full analysis with K=20.

## Questions to Answer

1. **Does MCMC start?** (Look for "🔵 SparseGFAFixedModel.**call** starting")
2. **What are the τ₀ values?** (Should be ~0.0026 for Z, ~0.05 for W)
3. **Where does it fail?** (Specific function/line in traceback)
4. **What error type?** (ValueError, TypeError, RuntimeError, etc.)
5. **What are the shapes?** (All parameters logged with shapes now)

## Contact Points

If you see:

- **"Shape mismatch"** → Parameter initialization issue
- **"Invalid value"** → Numerical stability issue (NaN/Inf)
- **"Memory"** → Need more GPU RAM or reduce batch size
- **"Divergent transitions"** → NUTS sampler issue (increase adapt_delta)
- **Empty error still** → NumPyro internal, share full log

---

**Date Created**: 2025-10-20
**Commit**: 1aaa169 (Add comprehensive logging to debug MCMC failures)
**Status**: Ready for GPU workstation testing
