# Phase B: Analysis of clinical_validation.py

**File:** `experiments/clinical_validation.py`
**Total Lines:** 4,613
**Total Methods:** 50

## Method Categorization

### 1. **Main Experiment Orchestrators** (6 methods)
High-level entry points that orchestrate complete validation workflows.

| Line | Method | Purpose | Should Extract? |
|------|--------|---------|----------------|
| 104 | `run_neuroimaging_clinical_validation()` | Main CV-based clinical validation | Keep in experiment |
| 858 | `run_subtype_classification_validation()` | PD subtype classification validation | Keep in experiment |
| 959 | `run_pd_subtype_discovery()` | Discover PD subtypes workflow | Keep in experiment |
| 1070 | `run_disease_progression_validation()` | Disease progression validation workflow | Keep in experiment |
| 1158 | `run_biomarker_discovery_validation()` | Biomarker discovery workflow | Keep in experiment |
| 1238 | `run_external_cohort_validation()` | External cohort validation workflow | Keep in experiment |

**Decision:** These stay in experiments - they are workflow orchestrators that tie everything together.

---

### 2. **SGFA Execution & Data Processing** (5 methods)
Core SGFA model execution and data preparation utilities.

| Line | Method | Purpose | Target Module |
|------|--------|---------|---------------|
| 203 | `_train_sgfa_with_neuroimaging_cv()` | Train SGFA with neuroimaging CV | `analysis/clinical/data_processing.py` |
| 341 | `_run_sgfa_training()` | Run single SGFA training | `analysis/clinical/data_processing.py` |
| 434 | `_evaluate_sgfa_test_set()` | Evaluate SGFA on test set | `analysis/clinical/data_processing.py` |
| 1310 | `_run_sgfa_analysis()` | Run complete SGFA analysis | `analysis/clinical/data_processing.py` |
| 2277 | `_apply_trained_model()` | Apply trained SGFA to new data | `analysis/clinical/data_processing.py` |

**Rationale:** These are reusable utilities for executing SGFA in clinical contexts. Should be extracted to a `ClinicalDataProcessor` class.

---

### 3. **Classification & Prediction** (3 methods)
Clinical classification and prediction utilities.

| Line | Method | Purpose | Target Module |
|------|--------|---------|---------------|
| 474 | `_validate_factors_clinical_prediction()` | Validate factors for clinical prediction | `analysis/clinical/classification.py` |
| 1354 | `_test_factor_classification()` | Test factor-based classification | `analysis/clinical/classification.py` |
| 2356 | `_test_cross_cohort_classification()` | Test cross-cohort classification | `analysis/clinical/classification.py` |

**Rationale:** Reusable classification utilities that could be shared across experiments.

---

### 4. **Metrics & Evaluation** (2 methods)
Core metrics calculation utilities.

| Line | Method | Purpose | Target Module |
|------|--------|---------|---------------|
| 1420 | `_calculate_detailed_metrics()` | Calculate classification metrics | `analysis/clinical/metrics.py` |
| 1468 | `_analyze_clinical_interpretability()` | Analyze clinical interpretability | `analysis/clinical/metrics.py` |

**Rationale:** Fundamental reusable metrics that any experiment could use. High priority for extraction.

---

### 5. **PD Subtype Discovery & Analysis** (6 methods)
Specialized utilities for Parkinson's disease subtype discovery.

| Line | Method | Purpose | Target Module |
|------|--------|---------|---------------|
| 3605 | `_discover_pd_subtypes()` | Discover PD subtypes via clustering | `analysis/clinical/subtype_analysis.py` |
| 3663 | `_validate_subtypes_clinical()` | Validate subtypes against clinical data | `analysis/clinical/subtype_analysis.py` |
| 3738 | `_analyze_pd_subtype_stability()` | Analyze subtype stability | `analysis/clinical/subtype_analysis.py` |
| 3785 | `_analyze_subtype_factor_patterns()` | Analyze factor patterns per subtype | `analysis/clinical/subtype_analysis.py` |
| 1557 | `_analyze_subtype_stability()` | General subtype stability analysis | `analysis/clinical/subtype_analysis.py` |
| 630 | `_analyze_clinical_subtypes_neuroimaging()` | Analyze clinical subtypes with neuroimaging | `analysis/clinical/subtype_analysis.py` |

**Rationale:** Coherent set of subtyping utilities that form a natural module.

---

### 6. **Disease Progression Analysis** (4 methods)
Longitudinal disease progression modeling utilities.

| Line | Method | Purpose | Target Module |
|------|--------|---------|---------------|
| 1631 | `_analyze_cross_sectional_correlations()` | Cross-sectional clinical correlations | `analysis/clinical/progression_analysis.py` |
| 1680 | `_analyze_longitudinal_progression()` | Longitudinal progression analysis | `analysis/clinical/progression_analysis.py` |
| 1757 | `_validate_progression_prediction()` | Validate progression predictions | `analysis/clinical/progression_analysis.py` |
| 1809 | `_analyze_clinical_milestones()` | Analyze clinical milestone timing | `analysis/clinical/progression_analysis.py` |

**Rationale:** Specialized longitudinal analysis utilities forming a cohesive module.

---

### 7. **Biomarker Discovery & Validation** (5 methods)
Biomarker identification and validation utilities.

| Line | Method | Purpose | Target Module |
|------|--------|---------|---------------|
| 550 | `_discover_neuroimaging_biomarkers()` | Discover biomarkers from neuroimaging | `analysis/clinical/biomarker_analysis.py` |
| 1836 | `_analyze_factor_outcome_associations()` | Factor-outcome associations | `analysis/clinical/biomarker_analysis.py` |
| 2029 | `_analyze_feature_importance()` | Feature importance analysis | `analysis/clinical/biomarker_analysis.py` |
| 2101 | `_validate_biomarker_panels()` | Validate biomarker panel performance | `analysis/clinical/biomarker_analysis.py` |
| 2189 | `_test_biomarker_robustness()` | Test biomarker robustness | `analysis/clinical/biomarker_analysis.py` |

**Rationale:** Comprehensive biomarker utilities that form a natural module.

---

### 8. **External Validation & Transfer** (2 methods)
External cohort validation and model transferability.

| Line | Method | Purpose | Target Module |
|------|--------|---------|---------------|
| 2306 | `_compare_factor_distributions()` | Compare distributions across cohorts | `analysis/clinical/external_validation.py` |
| 2406 | `_analyze_model_transferability()` | Analyze model transferability | `analysis/clinical/external_validation.py` |

**Rationale:** Specialized external validation utilities.

---

### 9. **Analysis & Summary Methods** (6 methods)
High-level analysis summarizers for each validation type.

| Line | Method | Purpose | Target Module |
|------|--------|---------|---------------|
| 711 | `_analyze_neuroimaging_clinical_validation()` | Summarize neuroimaging validation | Keep in experiment |
| 2476 | `_analyze_subtype_classification()` | Summarize subtype classification | Keep in experiment |
| 2563 | `_analyze_disease_progression_validation()` | Summarize progression validation | Keep in experiment |
| 2648 | `_analyze_biomarker_discovery()` | Summarize biomarker discovery | Keep in experiment |
| 2740 | `_analyze_external_cohort_validation()` | Summarize external validation | Keep in experiment |
| 3816 | `_analyze_pd_subtype_discovery()` | Summarize PD subtype discovery | Keep in experiment |

**Decision:** These are experiment-specific summarizers that tie results together - keep in experiment.

---

### 10. **Visualization Methods** (7 methods)
Plotting and visualization utilities.

| Line | Method | Purpose | Target Module |
|------|--------|---------|---------------|
| 766 | `_plot_neuroimaging_clinical_validation()` | Plot neuroimaging validation results | `analysis/clinical/visualization.py` |
| 2821 | `_plot_subtype_classification()` | Plot subtype classification results | `analysis/clinical/visualization.py` |
| 2978 | `_plot_disease_progression_validation()` | Plot progression validation results | `analysis/clinical/visualization.py` |
| 3097 | `_plot_biomarker_discovery()` | Plot biomarker discovery results | `analysis/clinical/visualization.py` |
| 3252 | `_plot_external_cohort_validation()` | Plot external validation results | `analysis/clinical/visualization.py` |
| 3443 | `_create_comprehensive_clinical_visualizations()` | Create comprehensive clinical plots | `analysis/clinical/visualization.py` |
| 3574 | `_extract_best_clinical_result()` | Extract best result for plotting | `analysis/clinical/visualization.py` |

**Rationale:** Specialized clinical plotting utilities that could be shared. Alternatively, could integrate with existing `visualization/` package.

---

### 11. **Integrated Optimization** (3 methods)
Advanced integrated clinical optimization (special case).

| Line | Method | Purpose | Decision |
|------|--------|---------|----------|
| 3881 | `run_integrated_sgfa_clinical_optimization()` | Joint model + clinical optimization | Keep in experiment (advanced) |
| 4115 | `_analyze_integrated_clinical_optimization()` | Analyze integrated optimization | Keep in experiment (advanced) |
| 1310 | `_run_sgfa_analysis()` (mock version) | Mock SGFA for testing | **DEAD CODE - Remove** |
| 1906 | `_run_sgfa_analysis()` (real version) | Real SGFA MCMC implementation | **Extract to data_processing.py** |

---

## 🚨 CRITICAL ISSUE FOUND: Duplicate Method Definition

**Problem:** There are TWO `_run_sgfa_analysis()` methods defined:

1. **Line 1310-1352:** Mock/simulation version
   - Generates synthetic clinical data with `np.random`
   - Creates fake factors, loadings, and convergence metrics
   - Used for testing without running actual MCMC

2. **Line 1906-2027:** Real MCMC implementation
   - Imports JAX/NumPyro and runs actual SGFA inference
   - Uses NUTS sampler with real convergence
   - Production implementation

**Impact:** In Python, when a class has two methods with the same name, only the SECOND definition is used. This means:
- The mock version (line 1310) is **dead code** and never executed
- All 10 calls to `_run_sgfa_analysis()` use the real MCMC version (line 1906)
- The mock version should be REMOVED

**Calls to `_run_sgfa_analysis()` (10 total):**
```
Line 873:  run_subtype_classification_validation()
Line 999:  run_pd_subtype_discovery()
Line 1087: run_disease_progression_validation()
Line 1173: run_biomarker_discovery_validation()
Line 1255: run_external_cohort_validation()
Line 1580: _analyze_subtype_stability() [bootstrap loop]
Line 1591: _analyze_subtype_stability() [bootstrap loop]
Line 2215: _test_biomarker_robustness() [bootstrap loop]
Line 3751: _analyze_pd_subtype_stability() [multi-run loop]
Line 4334: run_integrated_sgfa_clinical_optimization()
```

All of these currently use the real MCMC version since it's defined second.

**Resolution for Phase C:**
1. Remove the mock version (lines 1310-1352)
2. Rename the real version if needed for clarity
3. Extract the real version to `analysis/clinical/data_processing.py`
4. Update all 10 call sites to use the extracted version

---

## Summary Statistics

### Extraction Plan Overview

| Category | Methods | Action | Target |
|----------|---------|--------|--------|
| Orchestrators | 6 | **Keep in experiment** | - |
| SGFA Execution | 5 (4 real + 1 dead code) | **Extract 4, remove 1** | `data_processing.py` |
| Classification | 3 | **Extract** | `classification.py` |
| Metrics | 2 | **Extract** | `metrics.py` |
| Subtype Analysis | 6 | **Extract** | `subtype_analysis.py` |
| Progression | 4 | **Extract** | `progression_analysis.py` |
| Biomarkers | 5 | **Extract** | `biomarker_analysis.py` |
| External Validation | 2 | **Extract** | `external_validation.py` |
| Analysis Summarizers | 6 | **Keep in experiment** | - |
| Visualization | 7 | **Extract or integrate** | `visualization.py` or existing viz package |
| Integrated Optimization | 2 + 1 dead code | **Keep 2, remove 1** | - |
| **TOTAL** | **50** | **Extract: 33, Keep: 14, Remove: 3** | **7-8 modules** |

### Line Count Estimates

- **Total methods:** 50
- **Methods to extract:** 33 methods (~2,500-2,800 lines)
- **Methods to keep in experiment:** 14 methods (~1,800-2,000 lines)
- **Dead code to remove:** 3 methods (~50 lines total)
  - Mock `_run_sgfa_analysis()` at line 1310 (43 lines)
  - Plus any other identified dead code
- **Estimated experiment file after refactor:** ~2,000 lines (down from 4,613, -57% reduction)
- **Estimated new infrastructure:** ~2,500-2,800 lines across 7-8 modules in `analysis/clinical/`

---

## Key Dependencies to Map (Phase C)

For Phase C, we need to identify:

1. **Shared imports** - What sklearn, scipy, numpy utilities are used across methods
2. **Internal dependencies** - Which extracted methods call each other
3. **Config dependencies** - What config fields each method needs
4. **Data dependencies** - Standard data structures expected (X_list, clinical_data, etc.)
5. **Duplicate code** - The duplicate `_run_sgfa_analysis()` at lines 1310 and 1906

---

## Visualization Decision Point

**Question for Phase C:** Should clinical visualization methods be:

**Option A:** Extracted to `analysis/clinical/visualization.py` (new clinical-specific visualizer)

**Option B:** Integrated into existing `visualization/` package structure
- Add `ClinicalVisualizer` class to existing package
- Follows pattern of `FactorVisualizer`, `PreprocessingVisualizer`, etc.
- Listed in `visualization/__init__.py` alongside other visualizers

**Recommendation:** Option B - integrate with existing visualization infrastructure for consistency.

---

## Next Steps for Phase C

1. **Investigate duplicate** `_run_sgfa_analysis()` methods (lines 1310 vs 1906)
2. **Map dependencies** between methods to determine extraction order
3. **Create detailed extraction plan** showing:
   - Exact method signatures
   - Import requirements
   - Dependency chains
   - Class structures for each new module
4. **Define interfaces** for each new class/module
5. **Identify shared utilities** that multiple modules need
