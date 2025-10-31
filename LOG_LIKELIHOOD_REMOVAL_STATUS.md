# Log Likelihood Removal Status

**Date**: 2025-10-24
**Status**: IN PROGRESS
**Reason**: Log likelihood is not a meaningful metric for factor analysis

---

## Summary

You correctly identified that log likelihood calculations should be removed from the SGFA robustness testing pipeline. The metric is misleading because:

1. **Potential energy ≠ likelihood**: The value extracted is `-mean(potential_energy)` from MCMC, which equals `-log[p(θ|data)]` - the negative log posterior, NOT the pure likelihood
2. **Factor indeterminacies**: Factor models have inherent rotation, scale, and sign indeterminacies that make likelihood comparisons meaningless
3. **Wrong evaluation criterion**: For factor analysis, we care about **convergence, factor stability, and interpretability**, not likelihood values

---

## What is Potential Energy?

In NumPyro's HMC/NUTS sampling:
```
potential_energy = -log[p(θ|data)] = -log[p(data|θ)p(θ)]
                 = -log_likelihood - log_prior
```

It's the negative log posterior used by the sampler to navigate parameter space. The code was computing:
```python
log_likelihood = -np.mean(potential_energy)  # WRONG: This is NOT the true likelihood!
```

---

## Changes Completed ✅

### 1. MCMC Sampling (Core)
- ✅ **Line 905-908**: Removed `extra_fields=("potential_energy",)` from parallel chain MCMC
- ✅ **Line 1135-1137**: Removed `extra_fields=("potential_energy",)` from single MCMC
- ✅ **Line 1168-1173**: Removed potential_energy extraction and log_likelihood calculation
- ✅ **Line 790-793**: Added comment explaining removal

### 2. Result Dictionaries
- ✅ **Line 1178-1195**: Removed "potential_energy" and "log_likelihood" from multi-chain result dict
- ✅ **Line 1209-1226**: Removed "log_likelihood" from single-chain result dict
- ✅ **Line 1236-1240**: Removed "log_likelihood" from error fallback dict

### 3. Performance Metrics
- ✅ **Line 125-129**: Removed "log_likelihood" from seed robustness performance_metrics
- ✅ **Line 237-243**: Removed "log_likelihood" from data perturbation level_results (success case)
- ✅ **Line 247-253**: Removed "log_likelihood" from data perturbation level_results (error case)
- ✅ **Line 326-333**: Removed "log_likelihood" from initialization strategy_results (success case)
- ✅ **Line 349-356**: Removed "log_likelihood" from initialization strategy_results (error case)

### 4. Analysis Functions
- ✅ **Line 1242-1269**: Simplified `_analyze_seed_robustness()` - removed likelihood_variation analysis
- ✅ **Line 1271-1303**: Simplified `_analyze_data_perturbation_robustness()` - now convergence-based only
- ✅ **Line 1305-1343**: Simplified `_analyze_initialization_robustness()` - now convergence-based only
- ✅ **Line 1345-1363**: Simplified `_analyze_computational_robustness()` - removed likelihood-based exact robustness

### 5. Plotting Functions
- ✅ **Line 1399-1451**: Simplified `_plot_seed_robustness()` - removed 2 log likelihood plots, kept convergence & time

---

## Changes Still Needed ⚠️

### 6. Data Perturbation Plotting (Lines 1453-1582)
**Function**: `_plot_data_perturbation_robustness()`
**Issue**: Heavily relies on log_likelihood for all 4 plots

**Current structure**:
- Plot 1: Log likelihood vs perturbation level
- Plot 2: Convergence rate (GOOD - keep this!)
- Plot 3: Likelihood drop boxplot
- Plot 4: Likelihood heatmap

**Recommended fix**: Replace with convergence-focused version:
- Plot 1: Convergence rate vs perturbation level (keep current Plot 2)
- Plot 2: Execution time vs perturbation level (NEW)

**Lines to remove/replace**:
- 1459-1461: `baseline_likelihood` extraction
- 1472-1485: log_likelihood-based perturbation_data collection
- 1497-1517: Plot 1 (log likelihood by type/level)
- 1539-1550: Plot 3 (likelihood drop boxplot)
- 1552-1570: Plot 4 (likelihood heatmap)

### 7. Initialization Plotting (Lines 1584-~1750)
**Function**: `_plot_initialization_robustness()`
**Issue**: Heavily relies on log_likelihood

**Current plots**:
- Log likelihood by strategy (boxplot)
- Convergence rate by strategy
- Iterations by strategy
- Log likelihood vs iterations scatter

**Recommended fix**: Keep convergence & iterations plots only

**Lines to remove/replace**:
- 1596-1602: log_likelihood in initialization_data collection
- All log_likelihood-based plots

### 8. External Log Likelihood References (Lines 3800-4030)
**Location**: Main experiment runner sections
**Issue**: Logging statements that reference `result.get('log_likelihood', 0)`

**Lines to fix**:
- 3850-3857: Seed robustness logging with LL
- 3902-3909: Data perturbation logging with LL
- 3946-3953: Initialization logging with LL
- 4000-4024: Summary plot data collection using LL

---

## Recommendations

Given the extensive nature of the plotting changes, I recommend one of these approaches:

### Option A: Minimal Plotting (Fast)
**Remove plotting functions temporarily** - focus on getting the core analysis working
- Comment out plotting function calls
- Keep analysis functions (already fixed)
- Add back plotting later with convergence-focused plots

### Option B: Simplified Plotting (Moderate)
**Keep only convergence & time plots** - remove all likelihood-based visualizations
- Drastically simplify `_plot_data_perturbation_robustness()` to 1-2 plots
- Drastically simplify `_plot_initialization_robustness()` to 1-2 plots
- Remove likelihood from logging statements

### Option C: Complete Rewrite (Thorough)
**Rewrite all plotting functions** to be convergence-focused
- Create publication-quality convergence diagnostics
- Add execution time analysis
- Add memory usage tracking
- This is most work but gives best end result

---

## Decision Needed

**Current status**: Core MCMC sampling and analysis functions are cleaned up ✅

**Blocking issue**: Plotting functions still reference log_likelihood (29 remaining references)

**Your call**: Which option do you prefer?
1. Skip plotting for now (fastest - get factor stability working)
2. Simplify plotting (moderate - keep basic diagnostics)
3. Rewrite plotting (thorough - publication-quality plots)

I'm currently mid-way through option 2 (simplified plotting). Let me know if you want me to:
- Continue with simplified plotting
- Switch to just disabling plots
- Do full rewrite later when you need the plots

---

## Final Status Update (2025-10-24)

### ✅ COMPLETED - Factor Stability Pipeline is Clean!

After discovering that `run_experiments.py` **skips robustness_testing altogether** and only uses `run_factor_stability_analysis()`, we've successfully removed ALL log_likelihood references from the active code paths.

**Remaining 16 references** are ONLY in unused plotting functions:
- `_plot_data_perturbation_robustness()` (called by `run_data_perturbation_robustness` - NOT used)
- `_plot_initialization_robustness()` (called by `run_initialization_robustness` - NOT used)

**Factor stability analysis uses**:
- ✅ `run_factor_stability_analysis()` - CLEAN
- ✅ `_run_sgfa_analysis()` - CLEAN
- ✅ `_plot_factor_stability()` - CLEAN (no log_likelihood refs)

### Additional Changes in run_factor_stability_analysis():
- ✅ **Line 3790-3799**: Removed log_likelihood from seed robustness performance dict and logging
- ✅ **Line 3840-3848**: Removed log_likelihood from data perturbation performance dict and logging
- ✅ **Line 3878-3889**: Removed log_likelihood from initialization performance dict and logging
- ✅ **Line 3935-3941**: Changed seed robustness plot to use execution time instead of log_likelihood
- ✅ **Line 3943-3953**: Changed data perturbation plot to use execution time instead of log_likelihood
- ✅ **Line 3955-3965**: Changed initialization plot to use execution time instead of log_likelihood
- ✅ **Line 1080**: Removed orphaned `log_likelihood = 0.0` variable initialization

---

## Testing After Removal

**Ready to test!** The factor stability pipeline is now clean:

```bash
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --regress-confounds age sex tiv --K 5
```

Should run without any log_likelihood errors!

---

## Future Refactoring Needed

### Critical: Better Separation of Concerns

**Current problem**: `experiments/robustness_testing.py` (4100+ lines) contains:
1. Core factor stability analysis (`run_factor_stability_analysis`, `_run_sgfa_analysis`)
2. Robustness testing utilities (`run_seed_robustness_test`, `run_data_perturbation_robustness`, etc.)
3. Mixed plotting functions (some clean, some with log_likelihood)

**Issues**:
- ❌ **Misleading file organization**: Factor stability code is buried in "robustness_testing.py"
- ❌ **Code reuse confusion**: `_run_sgfa_analysis()` is core SGFA execution but lives in robustness file
- ❌ **Maintenance burden**: Changes to factor stability require editing robustness testing file
- ❌ **Circular naming**: File called "robustness_testing" but primarily used for factor stability
- ❌ **Dead code accumulation**: Unused plotting functions with stale dependencies (log_likelihood)

**Recommended refactoring**:

```
experiments/
├── factor_stability.py          # NEW - Extract factor stability analysis
│   ├── FactorStabilityExperiments(ExperimentFramework)
│   ├── run_factor_stability_analysis()
│   ├── _plot_factor_stability()
│   └── assess_factor_stability_procrustes()
│
├── sgfa_runner.py                # NEW - Core SGFA execution (model-agnostic)
│   ├── SGFARunner
│   ├── run_sgfa_analysis()       # Renamed from _run_sgfa_analysis
│   └── create_model_instance()   # From models_integration.py
│
├── robustness_testing.py         # REFACTORED - Only robustness utilities
│   ├── RobustnessExperiments(ExperimentFramework)
│   ├── run_seed_robustness_test()
│   ├── run_data_perturbation_robustness()
│   ├── run_initialization_robustness()
│   └── # Remove or fix plotting functions with log_likelihood
│
└── clinical_validation.py        # Keep as-is (already well-separated)
```

**Benefits**:
- ✅ Clear separation: Factor stability has its own module
- ✅ Reusable SGFA runner for all experiments
- ✅ Robustness testing becomes optional utility (as intended)
- ✅ Easier to maintain and test each component independently
- ✅ New developers can find code intuitively

**Migration path**:
1. Create `experiments/sgfa_runner.py` with core SGFA execution logic
2. Create `experiments/factor_stability.py` and move `run_factor_stability_analysis` there
3. Update `robustness_testing.py` to import from sgfa_runner
4. Update `clinical_validation.py` to import from sgfa_runner
5. Update `run_experiments.py` to import from factor_stability instead of robustness_testing
6. Delete or fix unused plotting functions in robustness_testing.py

**Priority**: Medium-High (affects code maintainability but not functionality)
