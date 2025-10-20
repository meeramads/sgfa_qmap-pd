# Recent Troubleshooting Session Summary
**Date:** October 20, 2025

## Issues Identified and Fixed

### 1. Command-Line MCMC Parameter Override Bug ✅ FIXED

**Problem:** The `--target-accept-prob` command-line argument was being ignored by factor_stability and robustness_testing experiments.

**Root Cause:**
- `run_experiments.py` was reading `target_accept_prob` only from `factor_stability` config section, not checking `mcmc` section where command-line args are stored
- `robustness_testing.py` had `target_accept_prob` hardcoded to 0.8

**Files Fixed:**
- `run_experiments.py` line 762: Now reads from `mcmc` config first, then falls back to `factor_stability`
- `experiments/robustness_testing.py` line 2940: Now reads from `mcmc` config instead of hardcoding

**Impact:** Command-line overrides like `--target-accept-prob 0.85` now work correctly across all experiments.

---

### 2. Model Selection: sparse_gfa vs sparse_gfa_fixed

**Discovery:** The acceptance probability differences are due to using DIFFERENT MODELS, not configuration issues:

**sparse_gfa (original model):**
- Centered parameterization
- Simpler posterior geometry
- Acceptance probability: 0.87-0.93 with `max_tree_depth=13`
- Used by: robustness_testing (default)

**sparse_gfa_fixed (new model, created Oct 19):**
- Non-centered parameterization
- More complex posterior geometry (designed to fix other convergence issues)
- Acceptance probability: TBD (needs proper tuning)
- Used by: factor_stability when config has `model_type: "sparse_gfa_fixed"`

**Key Insight:** `sparse_gfa_fixed` was created to improve convergence for high-dimensional data, but requires different MCMC tuning than `sparse_gfa`.

---

## Recommended MCMC Settings

### For sparse_gfa (original model):
```yaml
factor_stability:
  target_accept_prob: 0.8
  max_tree_depth: 13
  dense_mass: false
```

### For sparse_gfa_fixed (convergence-improved model):
```yaml
factor_stability:
  target_accept_prob: 0.85      # Higher than sparse_gfa
  max_tree_depth: 10            # Lower than sparse_gfa
  dense_mass: true              # Better for non-centered parameterization
  num_warmup: 3000              # More warmup for adaptation
```

**Rationale:** Non-centered parameterization creates funnel-shaped geometry that benefits from:
- Higher target acceptance (forces smaller, safer steps)
- Lower max tree depth (prevents overly ambitious trajectories)
- Dense mass matrix (adapts to correlations)

---

## Files Reorganized

Moved documentation to `docs/` directory:
- `docs/troubleshooting/` - Debugging and issue analysis
- `docs/` - Feature summaries and guides

All temporary analysis files removed from project root. Only `README.md` remains in root.

---

## Next Steps for User

1. **Decide which model to use:**
   - Use `sparse_gfa` if current convergence is acceptable
   - Use `sparse_gfa_fixed` if you need the convergence improvements it was designed for

2. **If using sparse_gfa_fixed:**
   - Test with recommended MCMC settings above
   - Verify acceptance probability improves to 0.75-0.85 range
   - Check R-hat diagnostics for convergence

3. **Test the bug fixes:**
   ```bash
   python run_experiments.py --config config_convergence.yaml \
     --experiments all --select-rois volume_sn_voxels.tsv \
     --regress-confounds age sex tiv --qc-outlier-threshold 3.0 \
     --percW 33 --K 2 --max-tree-depth 10 --target-accept-prob 0.85
   ```

   Verify that both `--max-tree-depth 10` and `--target-accept-prob 0.85` are actually used.

---

## Questions Remaining

1. What specific convergence issues prompted creation of `sparse_gfa_fixed`?
2. What acceptance probability does `sparse_gfa_fixed` achieve with recommended settings?
3. Should `config_convergence.yaml` default to `sparse_gfa` or `sparse_gfa_fixed`?

---

## References

- Bug fix details: `docs/troubleshooting/MCMC_PARAMETER_OVERRIDE_BUG_FIX.md`
- Config change analysis: `docs/troubleshooting/CONFIG_CHANGE_SMOKING_GUN.md`
- Model implementation: `models/sparse_gfa_fixed.py` (created Oct 19, 2025)
