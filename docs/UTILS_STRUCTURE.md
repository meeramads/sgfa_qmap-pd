# core/utils.py Structure

## Overview

`core/utils.py` is a comprehensive utility module (1231 lines) containing 5 functional categories. While large, it's well-organized with clear separation between categories.

**Total Size**: 1231 lines
**Import Sites**: 11 files
**Categories**: 5 distinct functional areas

## Category Breakdown

### 1. Memory Management (7 functions, ~250 lines)

**Purpose**: Handle memory-intensive MCMC operations and JAX memory management

**Functions**:
- `cleanup_memory(aggressive=False)` - Force garbage collection and JAX cache cleanup
- `cleanup_mcmc_samples(samples, keep_keys=None)` - Clean up MCMC sample dictionaries
- `safe_plotting_context()` - Context manager for memory-safe plotting
- `memory_monitoring_context(operation_name)` - Monitor memory during operations
- `check_available_memory()` - Check system memory availability
- `estimate_memory_requirements(args, X_list)` - Estimate MCMC memory needs
- `check_memory_before_analysis(args, X_list)` - Pre-flight memory check

**Key Use Cases**:
```python
# Before large MCMC run
check_memory_before_analysis(args, X_list)

# During MCMC
with memory_monitoring_context("MCMC sampling"):
    mcmc.run(...)

# After MCMC
cleanup_mcmc_samples(samples, keep_keys=["W", "Z"])
cleanup_memory(aggressive=True)
```

**Imports**: `gc`, `psutil`, `jax`

---

### 2. File Operations (10 functions, ~400 lines)

**Purpose**: Safe file I/O with error handling, backups, and retries

**Functions**:
- `ensure_directory(path)` - Create directory if doesn't exist
- `safe_file_path(directory, filename)` - Construct safe file paths
- `validate_file_exists(filepath, description)` - Validate file existence
- `create_results_structure(base_dir, dataset, model, flag, flag_regZ)` - Create result directory structure
- `get_model_files(results_dir, run_id)` - Get paths for model files
- `safe_pickle_load(filepath, max_retries=3, description)` - Load pickle with retries
- `safe_pickle_save_with_backup(data, filepath, description)` - Save with automatic backup
- `safe_pickle_save(data, filepath, description)` - Save pickle safely
- `backup_file(filepath, max_backups=5)` - Create numbered backup
- `clean_filename(filename)` - Sanitize filename
- `get_relative_path(filepath, base_path)` - Get relative path

**Key Features**:
- **Retry logic**: Automatically retries failed I/O operations
- **Backup management**: Creates numbered backups before overwriting
- **Error handling**: Detailed error messages with context
- **Path safety**: Handles spaces, special characters, cross-platform paths

**Use Cases**:
```python
# Safe pickle operations
data = safe_pickle_load("results/model.pkl", max_retries=3)
safe_pickle_save_with_backup(data, "results/model.pkl")

# Directory management
results_dir = create_results_structure("/results", "qmap_pd", "sparseGFA", True, True)
model_files = get_model_files(results_dir, run_id=1)
```

**Imports**: `pathlib`, `pickle`, `shutil`, `time`

---

### 3. Context Managers (2 functions, ~100 lines)

**Purpose**: Reusable context managers for common patterns

**Functions**:
- `safe_plotting_context()` - Matplotlib memory management
- `memory_monitoring_context(operation_name)` - Memory tracking

**Implementation**:
```python
@contextmanager
def safe_plotting_context():
    """Context manager for safe matplotlib usage."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    try:
        yield plt
    finally:
        plt.close('all')
        gc.collect()
```

**Use Cases**:
```python
with safe_plotting_context() as plt:
    plt.figure()
    plt.plot(x, y)
    plt.savefig("plot.png")
# Automatically cleans up

with memory_monitoring_context("Large computation"):
    result = expensive_operation()
# Logs memory usage before/after
```

---

### 4. Model Utilities (2 functions, ~300 lines)

**Purpose**: SGFA model-specific computations

**Functions**:
- `get_robustK(thrs, args, params, d_comps)` - Determine robust number of factors
- `get_infparams(samples, hypers, args)` - Compute inference parameters from MCMC

**Details**:

**`get_robustK()`**:
- Analyzes factor loadings to determine "robust" number of factors
- Uses thresholding and component analysis
- Returns refined K estimate

**`get_infparams()`**:
- Processes MCMC samples into summary statistics
- Computes means, std devs, credible intervals
- Handles W (loadings) and Z (scores) separately
- Splits concatenated W back into views

**Use Cases**:
```python
# After MCMC
robust_k = get_robustK(threshold=0.3, args, params, d_comps)
inf_params = get_infparams(samples, hypers, args)

# inf_params contains:
# - W_mean, W_std (per view)
# - Z_mean, Z_std
# - Credible intervals
# - Factor statistics
```

**Imports**: `numpy`, `scipy.stats`

---

### 5. Validation & Setup (6 functions, ~180 lines)

**Purpose**: Validate configuration and log settings

**Functions**:
- `validate_core_parameters(args)` - Validate basic parameters (K, percW, etc.)
- `validate_model_parameters(args)` - Validate model-specific parameters
- `validate_cv_parameters(args, cv_available, neuroimaging_cv_available)` - Validate CV config
- `validate_gpu_availability(args)` - Check GPU availability if requested
- `validate_file_paths(args)` - Validate required file paths exist
- `log_parameter_summary(args)` - Log parameter summary for debugging
- `validate_and_setup_args(args, logger)` - Comprehensive validation and setup

**Validation Rules**:
```python
# Core parameters
assert 1 <= args.K <= 50, "K must be between 1 and 50"
assert 0 < args.percW <= 100, "percW must be between 0 and 100"
assert args.num_samples > 0, "num_samples must be positive"

# CV parameters
if args.run_cv:
    assert cv_available, "CV requested but module not available"
    if args.neuroimaging_cv:
        assert neuroimaging_cv_available, "Neuroimaging CV not available"
```

**Use Cases**:
```python
# At experiment start
validate_and_setup_args(args, logger)
# Runs all validations and logs summary

# Individual validations
validate_core_parameters(args)
validate_gpu_availability(args)
log_parameter_summary(args)
```

---

## Import Patterns

### Files Using Memory Functions (3):
- `core/run_analysis.py`
- `analysis/model_runner.py`
- `experiments/sgfa_parameter_comparison.py`

### Files Using File Operations (8):
- `core/run_analysis.py`
- `core/visualization.py` → `visualization/core_plots.py`
- `visualization/brain_plots.py`
- `analysis/config_manager.py`
- `analysis/model_runner.py`
- `experiments/framework.py`

### Files Using Model Utilities (2):
- `analysis/model_runner.py`
- `core/run_analysis.py`

### Files Using Validation (4):
- `core/run_analysis.py`
- `debug_experiments.py`
- `run_experiments.py`

---

## Potential Splitting Strategy (Future Work)

If splitting becomes necessary, here's the recommended structure:

### Option A: Category-Based Split
```
core/
├── memory_utils.py      (cleanup, monitoring, estimation)
├── file_utils.py        (I/O, paths, backups)
├── model_utils.py       (robustK, infparams)
└── validation_utils.py  (already exists, merge validation functions)
```

### Option B: Keep as Single Module

**Reasons to keep together**:
1. **Cohesive**: All are "utility" functions
2. **Small functions**: No individual function > 100 lines
3. **Clear organization**: Already separated into categories with comments
4. **Low coupling**: Functions don't depend on each other
5. **Limited imports**: Only 11 import sites total

**Recommendation**: Keep as single module unless it grows significantly larger (>1500 lines).

---

## Usage Guidelines

### When to Use Memory Utils
- Before/after large MCMC runs
- When handling large numpy arrays
- In long-running experiments

### When to Use File Utils
- Saving model checkpoints
- Loading preprocessed data
- Creating result directories
- Any file I/O with potential for failure

### When to Use Model Utils
- After MCMC completion
- For factor analysis post-processing
- Computing summary statistics

### When to Use Validation
- At experiment initialization
- Before running MCMC
- When loading configuration files

---

## Dependencies

**External**:
- `numpy` - Numerical operations
- `psutil` - System memory monitoring
- `jax` - JAX-specific memory management
- `pickle` - Serialization
- `pathlib` - Path operations

**Internal**:
- None (foundation layer)

---

## Best Practices

1. **Memory Management**
   - Always use `memory_monitoring_context()` for large operations
   - Call `cleanup_memory()` after MCMC runs
   - Check memory before analysis with `check_memory_before_analysis()`

2. **File Operations**
   - Use `safe_pickle_*` functions instead of direct pickle
   - Always specify `description` parameter for better error messages
   - Use `create_results_structure()` for consistent directory layout

3. **Validation**
   - Call `validate_and_setup_args()` early in experiments
   - Log parameter summary for reproducibility

---

## Summary

`core/utils.py` is well-organized into 5 clear categories:
1. ✅ **Memory Management** - JAX and system memory handling
2. ✅ **File Operations** - Safe I/O with retries and backups
3. ✅ **Context Managers** - Reusable patterns
4. ✅ **Model Utilities** - SGFA-specific computations
5. ✅ **Validation** - Configuration checking

While large (1231 lines), the module is:
- Well-documented
- Clearly organized
- Widely used (11 import sites)
- Provides essential functionality

**Recommendation**: Keep as single module for now; splitting would add complexity without clear benefit.
