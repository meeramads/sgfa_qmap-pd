# Cross-Validation Architecture

## Overview

The CV module provides neuroimaging-aware cross-validation for SGFA models with clinical data integration. The architecture consists of 4 modules with clear separation of concerns.

## Module Structure

```
analysis/
├── cross_validation.py          (114 lines) - Orchestrator
├── cross_validation_library.py  (1835 lines) - Core implementation
├── cv_integration.py             (491 lines) - Pipeline integration
└── cv_fallbacks.py               (247 lines) - Fallback handlers
```

## Modules

### 1. `cross_validation.py` - **Unified Orchestrator**

**Purpose**: High-level CV orchestration and module selection

**Key Classes**:
- `CVRunner` - Main orchestrator class

**Responsibilities**:
- Detect available CV modules (basic vs neuroimaging-aware)
- Select appropriate CV strategy based on config
- Coordinate between different CV implementations
- Provide simple interface for experiments

**Usage**:
```python
from analysis.cross_validation import CVRunner

cv_runner = CVRunner(config, results_dir)
results, cv = cv_runner.run_cv_analysis(X_list, hypers, data)
```

**Decision Logic**:
```python
if neuroimaging_cv_available and config.neuroimaging_cv:
    → run_neuroimaging_cv_analysis()  # Advanced features
elif cv_available:
    → run_basic_cv_analysis()         # Standard CV
else:
    → Error: No CV module available
```

---

### 2. `cross_validation_library.py` - **Core Implementation**

**Purpose**: Comprehensive neuroimaging CV with clinical awareness

**Key Classes**:
- `NeuroImagingCrossValidator` - Main CV executor
- `NeuroImagingCVConfig` - CV configuration
- `ParkinsonsConfig` - PD-specific settings
- `NeuroImagingMetrics` - Evaluation metrics
- `NeuroImagingHyperOptimizer` - Hyperparameter optimization
- `ClinicalAwareSplitter` - Clinical-stratified splits

**Capabilities**:
1. **Clinical Stratification**
   - Stratify by diagnosis, motor subtypes, disease severity
   - Preserve clinical group distributions across folds

2. **Neuroimaging-Aware Processing**
   - Robust scaling (median/MAD instead of mean/std)
   - View-specific preprocessing
   - Spatial coherence preservation

3. **PD-Specific Features**
   - UPDRS-based stratification
   - Motor subtype awareness (tremor-dominant, akinetic-rigid, mixed)
   - Disease progression modeling

4. **Advanced Validation**
   - Nested CV for hyperparameter optimization
   - Convergence checking with timeouts
   - Comprehensive metrics (reconstruction, clustering, clinical)

**Usage**:
```python
from analysis.cross_validation_library import (
    NeuroImagingCrossValidator,
    NeuroImagingCVConfig,
    ParkinsonsConfig
)

cv_config = NeuroImagingCVConfig(
    outer_cv_folds=5,
    stratified=True,
    robust_scaling=True
)

pd_config = ParkinsonsConfig()

cv = NeuroImagingCrossValidator(cv_config, pd_config)
results = cv.neuroimaging_cross_validate(X_list, args, hypers, data)
```

---

### 3. `cv_integration.py` - **Pipeline Integration**

**Purpose**: Integrate CV framework with preprocessing and hyperparameter optimization

**Key Functions**:
- `apply_comprehensive_cv_framework()` - Main integration point
- `_create_cv_configuration()` - CV config from global config
- `_load_clinical_data()` - Clinical data loading
- `_fallback_cv_analysis()` - Fallback when framework unavailable

**Workflow**:
```
Global Config → CV Configuration → NeuroImagingCrossValidator → Results
      ↓
Clinical Data Loading
      ↓
Stratified CV Execution
      ↓
Enhanced Results with Clinical Metrics
```

**Usage**:
```python
from analysis.cv_integration import apply_comprehensive_cv_framework

cv_results, optimal_params = apply_comprehensive_cv_framework(
    X_list, config, optimal_params, data_dir
)
```

---

### 4. `cv_fallbacks.py` - **Fallback Handlers**

**Purpose**: Graceful degradation when advanced features unavailable

**Key Classes**:
- `CVFallbackHandler` - Handles CV split fallbacks
- `MetricsFallbackHandler` - Handles metric calculation fallbacks
- `HyperoptFallbackHandler` - Handles hyperopt fallbacks

**Fallback Strategy**:
```
Try: Advanced neuroimaging CV
  → ClinicalAwareSplitter with disease-specific stratification

Soft Failure: Missing clinical data
  → Fall back to sklearn StratifiedKFold
  → Use basic stratification

Hard Failure: Module import error
  → Use sklearn KFold (no stratification)
  → Log warning and continue
```

**Usage**:
```python
from analysis.cv_fallbacks import CVFallbackHandler

handler = CVFallbackHandler(logger)

splits = handler.with_cv_split_fallback(
    advanced_split_func,
    X, y, groups, clinical_data,
    cv_folds=5
)
```

---

## Data Flow

```
Experiment
    ↓
CVRunner.run_cv_analysis()
    ↓
┌─────────────────────────────────┐
│ Detect Available Modules        │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Neuroimaging CV Available?      │
└─────────────────────────────────┘
    Yes ↓                     No ↓
┌─────────────────┐    ┌──────────────────┐
│ Advanced:       │    │ Basic:           │
│ - Clinical      │    │ - Standard K-Fold│
│   stratification│    │ - Basic metrics  │
│ - Robust scaling│    │                  │
│ - PD-specific   │    │                  │
└─────────────────┘    └──────────────────┘
    ↓                          ↓
┌─────────────────────────────────┐
│ Execute CV Folds                │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Compute Metrics                 │
│ - Reconstruction error          │
│ - Clustering metrics            │
│ - Clinical interpretability     │
└─────────────────────────────────┘
    ↓
Return Results
```

## Configuration

### Basic CV Config
```python
config = {
    "cv_folds": 5,
    "run_cv": True,
    "cv_only": False,
    "seed": 42
}
```

### Neuroimaging CV Config
```python
config = {
    "neuroimaging_cv": True,
    "nested_cv": True,
    "cv_folds": 5,
    "stratified": True,
    "robust_scaling": True,
    "pd_specific": True
}
```

## Clinical Stratification

The CV system can stratify by:
- **Diagnosis**: PD vs Control
- **Motor Subtype**: Tremor-dominant, Akinetic-rigid, Mixed
- **Disease Severity**: UPDRS score ranges
- **Disease Duration**: Early vs Advanced
- **Custom Groups**: Any clinical variable

## Metrics Computed

### Standard Metrics
- Reconstruction error (per view)
- Mean squared error
- R² score

### Clinical Metrics
- Clinical interpretability scores
- Factor-clinical correlations
- Subtype discrimination ability

### Clustering Metrics
- Silhouette score
- Calinski-Harabasz index
- Davies-Bouldin index
- Adjusted Rand index (if true labels available)

## Integration Points

### From Experiments
```python
# experiments/clinical_validation.py
from analysis.cross_validation import CVRunner

cv_runner = CVRunner(config, results_dir)
results, cv = cv_runner.run_cv_analysis(X_list, hypers, data)
```

### From Pipeline
```python
# core/run_analysis.py
from analysis.cross_validation import should_run_cv_analysis

if should_run_cv_analysis(args):
    cv_runner = CVRunner(args, cv_res_dir)
    cv_results = cv_runner.run_cv_analysis(X_list, hypers, data)
```

## Design Decisions

### Why 4 Modules?

1. **Separation of Concerns**
   - Orchestration (cross_validation.py)
   - Implementation (cross_validation_library.py)
   - Integration (cv_integration.py)
   - Fallbacks (cv_fallbacks.py)

2. **Maintainability**
   - Each module has clear responsibility
   - Easy to extend or replace individual components

3. **Reusability**
   - Library can be used independently
   - Fallbacks provide graceful degradation

### Why Neuroimaging-Aware?

Standard CV doesn't account for:
- **Neuroimaging data characteristics** (high-dimensional, spatial structure)
- **Clinical heterogeneity** (disease subtypes, severity levels)
- **Domain-specific requirements** (robust to outliers, preserve spatial coherence)

## Extension Points

### Adding New CV Strategies

1. Implement in `cross_validation_library.py`
2. Add detection in `CVRunner._check_cv_availability()`
3. Add orchestration method in `CVRunner`
4. Update fallback handlers if needed

### Adding New Metrics

1. Add to `NeuroImagingMetrics` class
2. Integrate into `neuroimaging_cross_validate()`
3. Add fallback in `MetricsFallbackHandler`

### Adding New Clinical Stratification

1. Extend `ClinicalAwareSplitter`
2. Add configuration in `ParkinsonsConfig`
3. Update documentation

## Testing

Key test scenarios:
- Basic CV with minimal config
- Neuroimaging CV with clinical data
- Fallback when modules unavailable
- Stratification preservation
- Metrics calculation correctness

## Summary

The CV architecture provides:
- ✅ **Unified interface** via CVRunner
- ✅ **Advanced neuroimaging features** via cross_validation_library
- ✅ **Pipeline integration** via cv_integration
- ✅ **Graceful fallbacks** via cv_fallbacks
- ✅ **Clinical awareness** throughout
- ✅ **Extensibility** for future enhancements

This design balances sophistication (neuroimaging-aware features) with pragmatism (fallbacks when unavailable).
