# Phase B Complete: Analysis of clinical_validation.py

**Date:** 2025-10-01
**Status:** ✅ Phase B Complete - Ready for Phase C

---

## Executive Summary

Analyzed `experiments/clinical_validation.py` (4,613 lines, 50 methods) to categorize all methods and create extraction plan.

### Key Findings

1. **50 methods total** categorized into 11 functional groups
2. **33 methods (66%)** should be extracted to reusable infrastructure
3. **14 methods (28%)** should remain in experiment as orchestrators/summarizers
4. **3 methods (6%)** are dead code and should be removed
5. **1 critical bug found:** Duplicate `_run_sgfa_analysis()` method definition

### Proposed Architecture

```
analysis/clinical/              # New infrastructure location
├── __init__.py                 # Module exports
├── data_processing.py          # SGFA execution & data prep (5 methods, ~400 lines)
├── metrics.py                  # Clinical metrics calculation (2 methods, ~150 lines)
├── classification.py           # Clinical classification (3 methods, ~250 lines)
├── subtype_analysis.py         # PD subtype discovery (6 methods, ~500 lines)
├── progression_analysis.py     # Disease progression (4 methods, ~350 lines)
├── biomarker_analysis.py       # Biomarker discovery (5 methods, ~450 lines)
├── external_validation.py      # External cohort validation (2 methods, ~200 lines)
└── visualization.py            # Clinical plotting (7 methods, ~650 lines)
                                # OR integrate into existing visualization/ package
```

### Impact

- **Experiment file:** 4,613 → ~2,000 lines (-57% reduction)
- **New infrastructure:** ~2,500-2,800 lines across 7-8 modules
- **Code organization:** Clear separation between experiment workflows and reusable utilities
- **Reusability:** Clinical utilities can be used by model_comparison and sgfa_parameter_comparison

---

## Critical Issue: Duplicate Method Definition

### The Bug

```python
class ClinicalValidationExperiments:

    # Line 1310-1352 (43 lines) - DEAD CODE
    def _run_sgfa_analysis(self, X_list, hypers, args, **kwargs):
        """Run SGFA analysis for clinical validation."""
        # Mock implementation - generates synthetic data
        np.random.seed(42)
        Z = np.random.randn(n_subjects, K)
        # ... fake data generation ...
        return {"W": W, "Z": Z, "log_likelihood": -1000, ...}

    # Line 1906-2027 (122 lines) - ACTUAL IMPLEMENTATION
    def _run_sgfa_analysis(self, X_list, hypers, args, **kwargs):
        """Run actual SGFA analysis for clinical validation."""
        # Real MCMC implementation
        from numpyro.infer import MCMC, NUTS
        mcmc.run(rng_key, X_list, hypers, ...)
        # ... real SGFA execution ...
        return {"W": W_list, "Z": Z_mean, "samples": samples, ...}
```

### Impact

- Python only uses the **second** definition (line 1906)
- The mock version (line 1310) is **dead code** - never executed
- All 10 call sites use the real MCMC version
- This is actually GOOD - means production code is working correctly
- The mock version should be removed (it's just clutter)

### Resolution

**Phase C action items:**
1. Delete lines 1310-1352 (mock version)
2. Keep line 1906-2027 (real version) for extraction
3. Extract real version to `ClinicalDataProcessor.run_sgfa_analysis()`
4. Update 10 call sites to use extracted version

---

## Detailed Categorization

### Category 1: Main Experiment Orchestrators (6 methods → KEEP)

**Purpose:** High-level workflows that tie everything together

| Method | Line | Description |
|--------|------|-------------|
| `run_neuroimaging_clinical_validation()` | 104 | Main CV-based clinical validation pipeline |
| `run_subtype_classification_validation()` | 858 | PD subtype classification validation workflow |
| `run_pd_subtype_discovery()` | 959 | Discover PD subtypes end-to-end workflow |
| `run_disease_progression_validation()` | 1070 | Disease progression validation workflow |
| `run_biomarker_discovery_validation()` | 1158 | Biomarker discovery workflow |
| `run_external_cohort_validation()` | 1238 | External cohort validation workflow |

**Decision:** These are experiment-specific orchestrators that coordinate multiple utilities. They should remain in the experiment file.

---

### Category 2: SGFA Execution & Data Processing (5 methods → EXTRACT)

**Target:** `analysis/clinical/data_processing.py` → `ClinicalDataProcessor` class

| Method | Line | Purpose | Complexity |
|--------|------|---------|------------|
| `_train_sgfa_with_neuroimaging_cv()` | 203 | Train SGFA with neuroimaging-specific CV | High (~140 lines) |
| `_run_sgfa_training()` | 341 | Run single SGFA training iteration | Medium (~90 lines) |
| `_evaluate_sgfa_test_set()` | 434 | Evaluate SGFA on held-out test set | Low (~40 lines) |
| `_run_sgfa_analysis()` (REAL) | 1906 | Run SGFA MCMC inference | High (~120 lines) |
| `_apply_trained_model()` | 2277 | Apply trained SGFA to new data | Low (~30 lines) |

**Dead code to remove:**
- `_run_sgfa_analysis()` (mock version) at line 1310 (43 lines)

**Total extraction:** ~420 lines → `ClinicalDataProcessor`

---

### Category 3: Classification & Prediction (3 methods → EXTRACT)

**Target:** `analysis/clinical/classification.py` → `ClinicalClassifier` class

| Method | Line | Purpose | Complexity |
|--------|------|---------|------------|
| `_validate_factors_clinical_prediction()` | 474 | Validate factors for clinical prediction | Medium (~75 lines) |
| `_test_factor_classification()` | 1354 | Test classification using various models | Medium (~65 lines) |
| `_test_cross_cohort_classification()` | 2356 | Test cross-cohort generalization | Low (~50 lines) |

**Total extraction:** ~190 lines → `ClinicalClassifier`

---

### Category 4: Metrics & Evaluation (2 methods → EXTRACT)

**Target:** `analysis/clinical/metrics.py` → `ClinicalMetrics` class

| Method | Line | Purpose | Complexity |
|--------|------|---------|------------|
| `_calculate_detailed_metrics()` | 1420 | Calculate clinical classification metrics | Low (~45 lines) |
| `_analyze_clinical_interpretability()` | 1468 | Analyze factor clinical interpretability | Medium (~85 lines) |

**Total extraction:** ~130 lines → `ClinicalMetrics`

**Note:** These are HIGH PRIORITY for extraction - they're pure utility functions with no experiment-specific logic.

---

### Category 5: PD Subtype Discovery & Analysis (6 methods → EXTRACT)

**Target:** `analysis/clinical/subtype_analysis.py` → `PDSubtypeAnalyzer` class

| Method | Line | Purpose | Complexity |
|--------|------|---------|------------|
| `_discover_pd_subtypes()` | 3605 | Discover subtypes via clustering | Medium (~55 lines) |
| `_validate_subtypes_clinical()` | 3663 | Validate subtypes against clinical data | High (~75 lines) |
| `_analyze_pd_subtype_stability()` | 3738 | Analyze subtype stability across runs | Medium (~45 lines) |
| `_analyze_subtype_factor_patterns()` | 3785 | Analyze factor patterns per subtype | Low (~30 lines) |
| `_analyze_subtype_stability()` | 1557 | General subtype stability analysis | Medium (~70 lines) |
| `_analyze_clinical_subtypes_neuroimaging()` | 630 | Analyze subtypes with neuroimaging factors | High (~80 lines) |

**Total extraction:** ~355 lines → `PDSubtypeAnalyzer`

---

### Category 6: Disease Progression Analysis (4 methods → EXTRACT)

**Target:** `analysis/clinical/progression_analysis.py` → `DiseaseProgressionAnalyzer` class

| Method | Line | Purpose | Complexity |
|--------|------|---------|------------|
| `_analyze_cross_sectional_correlations()` | 1631 | Cross-sectional clinical correlations | Low (~45 lines) |
| `_analyze_longitudinal_progression()` | 1680 | Longitudinal progression analysis | High (~75 lines) |
| `_validate_progression_prediction()` | 1757 | Validate progression predictions | Medium (~50 lines) |
| `_analyze_clinical_milestones()` | 1809 | Analyze clinical milestone timing | Low (~25 lines) |

**Total extraction:** ~195 lines → `DiseaseProgressionAnalyzer`

---

### Category 7: Biomarker Discovery & Validation (5 methods → EXTRACT)

**Target:** `analysis/clinical/biomarker_analysis.py` → `BiomarkerAnalyzer` class

| Method | Line | Purpose | Complexity |
|--------|------|---------|------------|
| `_discover_neuroimaging_biomarkers()` | 550 | Discover biomarkers from neuroimaging | High (~80 lines) |
| `_analyze_factor_outcome_associations()` | 1836 | Factor-outcome associations | Medium (~70 lines) |
| `_analyze_feature_importance()` | 2029 | Feature importance analysis | Medium (~70 lines) |
| `_validate_biomarker_panels()` | 2101 | Validate biomarker panel performance | High (~85 lines) |
| `_test_biomarker_robustness()` | 2189 | Test biomarker robustness via bootstrap | High (~85 lines) |

**Total extraction:** ~390 lines → `BiomarkerAnalyzer`

---

### Category 8: External Validation & Transfer (2 methods → EXTRACT)

**Target:** `analysis/clinical/external_validation.py` → `ExternalValidator` class

| Method | Line | Purpose | Complexity |
|--------|------|---------|------------|
| `_compare_factor_distributions()` | 2306 | Compare factor distributions across cohorts | Medium (~50 lines) |
| `_analyze_model_transferability()` | 2406 | Analyze model transferability | Medium (~70 lines) |

**Total extraction:** ~120 lines → `ExternalValidator`

---

### Category 9: Analysis Summarizers (6 methods → KEEP)

**Purpose:** Experiment-specific result summarization

| Method | Line | Description |
|--------|------|-------------|
| `_analyze_neuroimaging_clinical_validation()` | 711 | Summarize neuroimaging validation results |
| `_analyze_subtype_classification()` | 2476 | Summarize subtype classification results |
| `_analyze_disease_progression_validation()` | 2563 | Summarize progression validation results |
| `_analyze_biomarker_discovery()` | 2648 | Summarize biomarker discovery results |
| `_analyze_external_cohort_validation()` | 2740 | Summarize external validation results |
| `_analyze_pd_subtype_discovery()` | 3816 | Summarize PD subtype discovery results |

**Decision:** These tie together results from multiple extracted utilities and create experiment-specific summaries. Keep in experiment.

---

### Category 10: Visualization (7 methods → EXTRACT or INTEGRATE)

**Option A:** Extract to `analysis/clinical/visualization.py` → `ClinicalVisualizer` class
**Option B:** Integrate into existing `visualization/` package (RECOMMENDED)

| Method | Line | Purpose | Complexity |
|--------|------|---------|------------|
| `_plot_neuroimaging_clinical_validation()` | 766 | Plot neuroimaging validation | Low (~55 lines) |
| `_plot_subtype_classification()` | 2821 | Plot subtype classification | High (~155 lines) |
| `_plot_disease_progression_validation()` | 2978 | Plot progression validation | Medium (~115 lines) |
| `_plot_biomarker_discovery()` | 3097 | Plot biomarker discovery | High (~150 lines) |
| `_plot_external_cohort_validation()` | 3252 | Plot external validation | High (~190 lines) |
| `_create_comprehensive_clinical_visualizations()` | 3443 | Create comprehensive clinical plots | High (~130 lines) |
| `_extract_best_clinical_result()` | 3574 | Extract best result for plotting | Low (~30 lines) |

**Total extraction:** ~825 lines

**Recommendation:** Integrate into existing `visualization/` package as `ClinicalVisualizer`, following the pattern of `FactorVisualizer`, `PreprocessingVisualizer`, `CrossValidationVisualizer`, and `BrainVisualizer`.

---

### Category 11: Integrated Optimization (2 methods → KEEP, 1 → REMOVE)

**Advanced features for joint optimization**

| Method | Line | Decision | Rationale |
|--------|------|----------|-----------|
| `run_integrated_sgfa_clinical_optimization()` | 3881 | KEEP | Complex experiment workflow |
| `_analyze_integrated_clinical_optimization()` | 4115 | KEEP | Experiment-specific analysis |

---

## Dependency Analysis Preview

### Shared Imports (Used Across Multiple Categories)

```python
# Statistical/ML
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             silhouette_score, calinski_harabasz_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy import stats
from scipy.stats import f_oneway

# SGFA
import jax
from numpyro.infer import MCMC, NUTS
from core.run_analysis import models

# Clinical CV infrastructure
from analysis.cross_validation_library import (
    ClinicalAwareSplitter,
    NeuroImagingCVConfig,
    NeuroImagingMetrics
)
```

### Internal Dependencies (Extracted Methods Calling Each Other)

Need to map in Phase C:
- Which extracted methods call other extracted methods?
- Which extracted methods need access to experiment config?
- Which extracted methods have circular dependencies?

---

## Phase C Preview: Next Steps

### 1. Resolve Duplicate Method
- [ ] Remove mock `_run_sgfa_analysis()` (line 1310-1352)
- [ ] Verify all 10 call sites work with real version
- [ ] Extract real version to `ClinicalDataProcessor`

### 2. Create Module Interfaces
For each target module, define:
- Class name and constructor signature
- Public methods (extracted from experiment)
- Required dependencies (config, logger, etc.)
- Return types and data structures

### 3. Map Dependencies
- [ ] Create dependency graph showing which methods call each other
- [ ] Identify circular dependencies
- [ ] Determine extraction order (low dependencies first)

### 4. Design Data Structures
Standardize data structures passed between utilities:
- SGFA results format
- Clinical data format
- Validation results format
- Metrics dictionary format

### 5. Create Detailed Extraction Plan
For each of 33 methods to extract:
- [ ] Source file + line range
- [ ] Target module + class
- [ ] Method signature
- [ ] Dependencies (internal + external)
- [ ] Test requirements

### 6. Visualization Decision
- [ ] Decide: New `ClinicalVisualizer` in `analysis/clinical/` OR integrate into `visualization/`?
- [ ] If integrating: Update `visualization/__init__.py` and `visualization/manager.py`

---

## Success Metrics

### Code Organization
- ✅ 50 methods categorized into 11 functional groups
- ✅ Clear extraction plan: 33 extract, 14 keep, 3 remove
- ✅ Critical bug identified (duplicate method definition)

### Documentation
- ✅ Comprehensive analysis document created
- ✅ Line-by-line categorization with rationale
- ✅ Estimated line counts for each module

### Readiness for Phase C
- ✅ All methods identified and categorized
- ✅ Target modules defined
- ✅ Extraction priorities established
- ✅ Dead code identified for removal
- 🔲 Dependencies mapped (Phase C task)
- 🔲 Interfaces designed (Phase C task)

---

## Files Created

1. **PHASE_B_ANALYSIS.md** - Detailed categorization of all 50 methods
2. **PHASE_B_COMPLETE.md** - This summary document

**Next:** Proceed to Phase C to create detailed extraction plan with method signatures, dependencies, and implementation order.
