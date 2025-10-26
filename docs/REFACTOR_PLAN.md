# Refactoring Plan

**Created**: October 6, 2025  
**Last Updated**: October 21, 2025  
**Status**: Deferred - Production pipeline is functional  
**Focus**: Large file splitting for maintainability

## Overview

This document outlines refactoring opportunities for the SGFA codebase. The **current production pipeline is functional and ready for research use**. These refactors would improve long-term maintainability but are not blocking research progress.

## Current Production Pipeline

The active pipeline consists of:
- `run_experiments.py` (1,271 lines) - Main entry point
- `experiments/data_validation.py` (3,291 lines) 🔴 **LARGE**
- `experiments/robustness_testing.py` (3,401 lines) 🔴 **LARGE**
- `experiments/framework.py` (1,537 lines) 🟡 **MODERATE**
- `analysis/factor_stability.py` (709 lines) ✅ **GOOD**

**Total**: ~10,200 lines for core pipeline

## Priority: Production Pipeline Files

### 🔴 HIGH: experiments/robustness_testing.py (4,100+ lines)

**Status**: Actively used in production pipeline (but contains mixed concerns)

**Current Issues**:
- ❌ **Misleading file organization**: Core factor stability code is buried in "robustness_testing.py"
- ❌ **Mixed responsibilities**: Contains both robustness utilities AND core SGFA/factor stability logic
- ❌ **Code reuse confusion**: `_run_sgfa_analysis()` is core SGFA execution but lives in robustness file
- ❌ **Maintenance burden**: Changes to factor stability require editing robustness testing file
- ❌ **Circular naming**: File called "robustness_testing" but primarily used for factor stability
- ❌ **Dead code accumulation**: Unused plotting functions with stale dependencies

**Current Structure**:
- Core SGFA execution (`_run_sgfa_analysis`) - **SHOULD BE SEPARATE**
- Factor stability analysis (`run_factor_stability_analysis`) - **SHOULD BE SEPARATE**
- Seed reproducibility tests
- Perturbation analysis
- Initialization stability
- All visualization and result processing

**Proposed Refactor** (3-module split):

```bash
# NEW: Core SGFA execution (model-agnostic runner)
experiments/sgfa_runner.py (~800 lines)
├── SGFARunner class
├── run_sgfa_analysis()              # Renamed from _run_sgfa_analysis
├── create_model_instance()          # Moved from models_integration.py
└── MCMC execution utilities

# NEW: Factor stability analysis (primary experiment)
experiments/factor_stability.py (~1,200 lines)
├── FactorStabilityExperiments(ExperimentFramework)
├── run_factor_stability_analysis()
├── _plot_factor_stability()
├── assess_factor_stability_procrustes()
└── Stability metrics and visualization

# REFACTORED: Robustness testing (optional utilities)
experiments/robustness/
├── __init__.py                      # Export main class
├── orchestrator.py                  # RobustnessExperiments class (~500 lines)
├── seed_tests.py                    # Seed reproducibility (~600 lines)
├── perturbation_tests.py            # Perturbation analysis (~600 lines)
├── initialization_tests.py          # Initialization stability (~600 lines)
├── metrics.py                       # Metric calculations (~400 lines)
└── visualization.py                 # Plot generation (~400 lines)
                                     # (Remove or fix log_likelihood refs)
```

**Migration Path**:
1. Create `experiments/sgfa_runner.py` with core SGFA execution logic
2. Create `experiments/factor_stability.py` and move `run_factor_stability_analysis` there
3. Update `robustness_testing.py` to import from `sgfa_runner`
4. Update `clinical_validation.py` to import from `sgfa_runner`
5. Update `run_experiments.py` to import from `factor_stability` instead of `robustness_testing`
6. Split remaining robustness code into `experiments/robustness/` package
7. Delete or fix unused plotting functions with log_likelihood references

**Benefits**:
- ✅ **Clear separation of concerns**: Factor stability has its own module
- ✅ **Reusable SGFA runner**: Can be used by all experiments (factor stability, clinical validation, robustness)
- ✅ **Intuitive organization**: New developers can find code based on what it does
- ✅ **Reduced file size**: 4,100-line file becomes 3 focused modules (800 + 1,200 + 3×600 = ~3,800 lines across 8 files)
- ✅ **Better testability**: Each component can be tested independently
- ✅ **Dead code removal**: Easier to identify and remove unused plotting functions

**Effort**: 6-8 hours (3-module split + migration)
**Risk**: Medium-High (actively used, but high reward for maintainability)

**Priority**: **CRITICAL** - This addresses architectural confusion that impacts daily development

---

### 🔴 HIGH: experiments/data_validation.py (3,291 lines)

**Status**: Actively used in production pipeline

**Current Issues**:
- ❌ **Mixed concerns**: EDA (exploratory data analysis) is conflated with preprocessing
- ❌ **Unclear purpose**: File name suggests validation but includes exploratory analysis
- ❌ **Reusability**: EDA code should be separate for use across multiple experiments

**Current Structure**:
- Data quality assessment
- Distribution analysis (EDA)
- PCA dimensionality analysis (EDA)
- Preprocessing strategy comparison
- SNR estimation (EDA)
- All plotting and reporting

**Proposed Refactor**:
```bash
# NEW: Separate EDA from preprocessing validation
experiments/exploratory_data_analysis/
├── __init__.py                 # Export EDA functions
├── distribution_analysis.py    # Distribution plots (~400 lines)
├── pca_analysis.py             # PCA dimensionality analysis (~400 lines)
├── snr_estimation.py           # Signal-to-noise ratio analysis (~300 lines)
├── correlation_analysis.py     # Feature correlations (~300 lines)
└── visualization.py            # EDA-specific plots (~400 lines)

# REFACTORED: Keep data validation focused
experiments/data_validation/
├── __init__.py                 # Export main class
├── orchestrator.py             # DataValidationExperiments (~400 lines)
├── quality_checks.py           # Data quality metrics (~500 lines)
├── preprocessing_comparison.py # Strategy comparison (~600 lines)
├── metrics.py                  # Quality metrics (~300 lines)
└── visualization.py            # Validation plots (~300 lines)
```

**Benefits**:
- ✅ **Clear separation**: EDA is independent from data validation
- ✅ **Reusability**: EDA functions can be used across experiments
- ✅ **Better naming**: Files accurately reflect their purpose
- ✅ **Modular quality checks**: Easier to add new validation metrics
- ✅ **Focused validation**: Data validation focuses on preprocessing quality, not exploration

**Effort**: 6-8 hours (includes EDA extraction)
**Risk**: Medium (actively used experiment)

**Priority**: **HIGH** - Separating EDA from preprocessing improves clarity and reusability

---

### 🟡 MODERATE: experiments/framework.py (1,537 lines)

**Status**: Foundation for all experiments

**Current Structure**:
- ExperimentFramework base class
- Result handling
- Plot saving utilities
- Matrix export

**Proposed Refactor**:
```bash
experiments/framework/
├── __init__.py              # Export main classes
├── base.py                  # ExperimentFramework class (~600 lines)
├── results.py               # Result handling (~400 lines)
├── io_utils.py              # Matrix saving (~300 lines)
└── plotting.py              # Plot utilities (~300 lines)
```

**Benefits**:
- Clearer separation of base functionality
- Easier to extend framework
- Better testability

**Effort**: 2-3 hours  
**Risk**: Low (well-tested foundation)

---

### 🟡 MODERATE: run_experiments.py (1,271 lines)

**Status**: Main entry point

**Current Structure**:
- Command-line argument parsing (~300 lines)
- Configuration loading and merging (~200 lines)
- Experiment orchestration (~400 lines)
- Results directory management (~200 lines)
- Documentation generation (~200 lines)

**Proposed Refactor**:
```bash
experiments/cli/
├── __init__.py              # Export main function
├── args.py                  # Argument parsing (~300 lines)
├── config.py                # Config loading/merging (~250 lines)
├── orchestration.py         # Experiment running (~400 lines)
└── reporting.py             # Results/docs generation (~300 lines)

# Keep simple entry point:
run_experiments.py (50 lines) - Just calls cli.main()
```

**Benefits**:
- Easier to test CLI components
- Clearer config merging logic
- Better separation of concerns

**Effort**: 3-4 hours  
**Risk**: Medium (main entry point)

---

## Priority: Future Work Files

These files are part of the "future work" and not currently used in production:

### 🟢 LOW: experiments/model_comparison.py (3,312 lines)

**Status**: Future work (blocked on comparison metrics)

**Refactor**: Only if/when this experiment becomes production-ready

---

### 🟢 LOW: experiments/sgfa_configuration_comparison.py (2,882 lines)

**Status**: Future work (blocked on comparison metrics)

**Refactor**: Only if/when this experiment becomes production-ready

---

### 🟢 LOW: experiments/clinical_validation.py (2,987 lines)

**Status**: Future work

**Refactor**: Only if/when expanded for production use

---

## Refactoring Strategy

### Phase 1: Foundation (Low Risk)
1. **experiments/framework.py** → `experiments/framework/`
   - Low risk, well-tested
   - Benefits all other experiments
   - **Effort**: 2-3 hours

### Phase 2: Production Experiments (Medium Risk)
2. **experiments/data_validation.py** → `experiments/data_validation/`
   - Active experiment, moderate risk
   - **Effort**: 4-6 hours

3. **experiments/robustness_testing.py** → `experiments/robustness/`
   - Active experiment, moderate risk
   - **Effort**: 4-6 hours

### Phase 3: Entry Point (If Needed)
4. **run_experiments.py** → `experiments/cli/`
   - Only if maintainability becomes an issue
   - **Effort**: 3-4 hours

**Total Effort**: 13-19 hours for full production pipeline refactor

---

## File Size Guidelines

**Target Sizes** (for reference):
- ✅ **< 500 lines**: Ideal - Easy to navigate
- 🟡 **500-1000 lines**: Acceptable - Still manageable  
- 🟠 **1000-2000 lines**: Large - Consider splitting if actively developed
- 🔴 **> 2000 lines**: Very large - Should split for maintainability

**Current Production Files**:
- ✅ `analysis/factor_stability.py` (709 lines)
- 🟡 `run_experiments.py` (1,271 lines)
- 🟡 `experiments/framework.py` (1,537 lines)
- 🔴 `experiments/data_validation.py` (3,291 lines)
- 🔴 `experiments/robustness_testing.py` (3,401 lines)

---

## Benefits of Refactoring

### Maintainability
- Easier to navigate and understand code
- Clearer module responsibilities
- Better separation of concerns

### Testability
- Can test components in isolation
- Easier to mock dependencies
- Better coverage of edge cases

### Extensibility
- Easier to add new features
- Clearer extension points
- Less risk of breaking existing code

### Developer Experience
- Faster to locate relevant code
- Clearer code organization
- Better IDE support (faster indexing)

---

## When to Refactor

**Good Times**:
- After major research milestone completed
- Before adding substantial new features
- When file becomes difficult to navigate
- When multiple people need to work on same file

**Bad Times**:
- In middle of active research/experiments
- When results are pending
- Right before important deadline
- Without comprehensive tests

---

## Testing Strategy

For each refactor:

1. **Before**: Run full test suite
   ```bash
   pytest tests/ -v
   ```

2. **During**: Maintain 100% test pass rate
   - Move tests alongside code
   - Add tests for new module boundaries
   - Keep integration tests

3. **After**: Verify production pipeline
   ```bash
   # Test full pipeline still works
   python run_experiments.py --config config_convergence.yaml \
     --experiments data_validation robustness_testing factor_stability \
     --select-rois volume_sn_voxels.tsv
   ```

4. **Regression**: Compare results
   - Check output files match previous runs
   - Verify plots are identical
   - Confirm metrics unchanged

---

## Current Recommendation

**For Now**: ⏸️ **Defer refactoring**
- Production pipeline is functional
- Focus on research results
- Document any pain points for future refactor

**When Ready**: Start with Phase 1 (framework.py)
- Lowest risk
- Benefits all other experiments
- Good practice for larger refactors

**Future**: Consider Phases 2-3 based on:
- How actively files are being modified
- Whether multiple developers are working on them
- Specific maintainability pain points encountered

---

## Notes

- All refactors preserve external APIs (no breaking changes for users)
- Internal structure changes only
- Tests move with code
- Documentation updated alongside refactoring
- Each refactor is a separate, testable change
