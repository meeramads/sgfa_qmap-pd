# Clinical Validation Modules - Usage Examples

Quick reference for using the extracted clinical validation modules.

---

## Basic Imports

```python
from analysis.clinical import (
    ClinicalMetrics,
    ClinicalDataProcessor,
    ClinicalClassifier,
    PDSubtypeAnalyzer,
    DiseaseProgressionAnalyzer,
    BiomarkerAnalyzer,
    ExternalValidator
)
```

---

## 1. ClinicalMetrics

### Calculate Classification Metrics
```python
import numpy as np

# Initialize
metrics = ClinicalMetrics(
    metrics_list=["accuracy", "precision", "recall", "f1_score", "roc_auc"],
    logger=logger  # optional
)

# Calculate metrics
y_true = np.array([0, 1, 0, 1, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 0])
y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8], ...])  # (N, 2)

detailed_metrics = metrics.calculate_detailed_metrics(
    y_true, y_pred, y_pred_proba
)

print(detailed_metrics["accuracy"])  # 0.833
print(detailed_metrics["f1_score"])  # 0.8
print(detailed_metrics["confusion_matrix"])  # [[2, 1], [0, 3]]
```

### Analyze Factor Interpretability
```python
# Analyze how factors relate to clinical labels
Z = np.random.randn(100, 5)  # 100 subjects, 5 factors
clinical_labels = np.random.randint(0, 2, 100)  # binary diagnosis
sgfa_result = {"W": [W1, W2]}  # loading matrices

interpretability = metrics.analyze_clinical_interpretability(
    Z, clinical_labels, sgfa_result
)

print(interpretability["factor_label_associations"])  # statistical tests
print(interpretability["interpretability_analysis"])  # sparsity scores
```

---

## 2. ClinicalDataProcessor

### Run SGFA Analysis
```python
# Initialize
processor = ClinicalDataProcessor(logger=logger)

# Run SGFA with MCMC
X_list = [X1, X2, X3]  # multiple views
hypers = {"K": 10, "a_W": 1, "b_W": 1}
args = {
    "num_warmup": 100,
    "num_samples": 300,
    "num_chains": 1,
    "model": "sparseGFA"
}

sgfa_result = processor.run_sgfa_analysis(X_list, hypers, args)

print(sgfa_result["Z"].shape)  # (N, 10) - latent factors
print(sgfa_result["log_likelihood"])  # -1234.56
print(sgfa_result["convergence"])  # True
```

### Apply Trained Model to New Data
```python
# Apply to test cohort
X_test_list = [X1_test, X2_test, X3_test]

test_result = processor.apply_trained_model(
    X_test_list, sgfa_result, hypers, args
)

print(test_result["Z"].shape)  # (N_test, 10)
```

---

## 3. ClinicalClassifier

### Initialize with Dependencies
```python
metrics = ClinicalMetrics()
config = {
    "cross_validation": {
        "n_folds": 5,
        "stratified": True,
        "random_seed": 42
    }
}

classifier = ClinicalClassifier(
    metrics_calculator=metrics,
    config=config,
    logger=logger
)
```

### Test Factor Classification
```python
# Test classification using latent factors
Z = sgfa_result["Z"]  # (N, K)
labels = np.array([0, 1, 0, 1, ...])  # (N,)

results = classifier.test_factor_classification(
    features=Z,
    labels=labels,
    feature_type="SGFA_factors"
)

# Results for each classifier (logistic, random forest, SVM)
print(results["logistic_regression"]["cross_validation"]["accuracy"])
print(results["random_forest"]["detailed_metrics"]["f1_score"])
```

### Cross-Cohort Classification
```python
# Train on one cohort, test on another
results = classifier.test_cross_cohort_classification(
    Z_train=Z_cohort1,
    Z_test=Z_cohort2,
    labels_train=labels_cohort1,
    labels_test=labels_cohort2
)

print(results["random_forest"]["cross_cohort_performance"]["accuracy"])
print(results["random_forest"]["performance_drop"])  # degradation
```

---

## 4. PDSubtypeAnalyzer

### Discover Subtypes
```python
analyzer = PDSubtypeAnalyzer(
    data_processor=processor,  # optional, for stability analysis
    logger=logger
)

# Discover subtypes via clustering
Z = sgfa_result["Z"]
subtype_results = analyzer.discover_pd_subtypes(Z)

print(subtype_results["optimal_k"])  # 3 (optimal number of subtypes)
print(subtype_results["silhouette_scores"])  # [0.45, 0.52, 0.48, ...]
print(subtype_results["best_solution"]["labels"])  # cluster assignments
```

### Validate Against Clinical Data
```python
clinical_data = {
    "diagnosis": [...],
    "updrs": [...],
    "moca": [...],
    "disease_duration": [...]
}

validation = analyzer.validate_subtypes_clinical(
    subtype_results, clinical_data, Z
)

print(validation["clinical_associations"])  # significant differences
print(validation["summary"]["proportion_significant"])  # 0.75
```

### Analyze Stability
```python
# Multi-run stability
stability = analyzer.analyze_pd_subtype_stability(
    X_list, hypers, args, n_runs=5
)

print(stability["mean_ari"])  # 0.82 (high stability)
print(stability["stability_grade"])  # "High"
```

---

## 5. DiseaseProgressionAnalyzer

### Cross-Sectional Correlations
```python
analyzer = DiseaseProgressionAnalyzer(
    classifier=classifier,  # optional, for milestone prediction
    logger=logger
)

progression_scores = np.array([...])  # UPDRS or similar
correlations = analyzer.analyze_cross_sectional_correlations(
    Z, progression_scores
)

print(correlations["factor_progression_correlations"])
print(correlations["multiple_regression"]["r2_score"])  # 0.65
```

### Longitudinal Analysis
```python
# Requires longitudinal data
time_points = np.array([0, 6, 12, 18, ...])  # months
subject_ids = np.array([1, 1, 1, 2, 2, 2, ...])

longitudinal = analyzer.analyze_longitudinal_progression(
    Z, progression_scores, time_points, subject_ids
)

print(longitudinal["progression_analysis"]["mean_progression_rate"])
print(longitudinal["factor_change_analysis"])
```

---

## 6. BiomarkerAnalyzer

### Discover Biomarkers
```python
analyzer = BiomarkerAnalyzer(
    data_processor=processor,  # optional, for robustness
    config=config,
    logger=logger
)

# From CV results
biomarkers = analyzer.discover_neuroimaging_biomarkers(
    sgfa_cv_results, clinical_data
)

print(biomarkers["factor_importance"]["top_factors"])  # [4, 2, 7]
print(biomarkers["discriminative_factors"])  # factors with p < 0.05
```

### Factor-Outcome Associations
```python
clinical_outcomes = {
    "motor_score": np.array([...]),
    "cognitive_score": np.array([...]),
    "tremor": np.array([0, 1, 0, 1, ...])  # binary
}

associations = analyzer.analyze_factor_outcome_associations(
    Z, clinical_outcomes
)

for outcome, results in associations.items():
    for factor_result in results:
        if factor_result.get("pearson_p_value", 1) < 0.05:
            print(f"Factor {factor_result['factor']} → {outcome}: "
                  f"r={factor_result['pearson_correlation']:.3f}")
```

### Validate Biomarker Panels
```python
panels = analyzer.validate_biomarker_panels(Z, clinical_outcomes)

print(panels["motor_score"]["panel_size_3"]["cv_r2_mean"])  # 0.45
print(panels["tremor"]["panel_size_2"]["cv_accuracy_mean"])  # 0.78
```

---

## 7. ExternalValidator

### Compare Factor Distributions
```python
validator = ExternalValidator(logger=logger)

# Compare two cohorts
Z_train = sgfa_result_cohort1["Z"]
Z_test = sgfa_result_cohort2["Z"]

comparison = validator.compare_factor_distributions(
    Z_train, Z_test,
    cohort_names=("Discovery", "Validation")
)

print(comparison["overall_similarity"]["similarity_score"])  # 0.85
print(comparison["factor_comparisons"])  # per-factor KS tests
```

### Analyze Transferability
```python
transferability = validator.analyze_model_transferability(
    train_result=sgfa_result_cohort1,
    test_result=sgfa_result_cohort2,
    labels_train=labels_cohort1,
    labels_test=labels_cohort2
)

print(transferability["model_fit_comparison"])
print(transferability["factor_space_similarity"]["mean_canonical_correlation"])
print(transferability["demographic_transferability"])
```

---

## Complete Example: Full Clinical Validation Pipeline

```python
import numpy as np
from analysis.clinical import *

# 1. Setup
metrics = ClinicalMetrics()
processor = ClinicalDataProcessor()
config = {"cross_validation": {"n_folds": 5}}

classifier = ClinicalClassifier(metrics, config=config)
subtype_analyzer = PDSubtypeAnalyzer(processor)
progression_analyzer = DiseaseProgressionAnalyzer(classifier)
biomarker_analyzer = BiomarkerAnalyzer(processor, config=config)

# 2. Run SGFA
X_list = [X1, X2, X3]  # your data
hypers = {"K": 10, "a_W": 1, "b_W": 1}
args = {"num_warmup": 100, "num_samples": 300}

sgfa_result = processor.run_sgfa_analysis(X_list, hypers, args)
Z = sgfa_result["Z"]

# 3. Clinical validation
clinical_data = {"diagnosis": [...], "updrs": [...]}

# Classification
classification_results = classifier.test_factor_classification(
    Z, clinical_data["diagnosis"], "SGFA"
)

# Subtype discovery
subtypes = subtype_analyzer.discover_pd_subtypes(Z)
subtype_validation = subtype_analyzer.validate_subtypes_clinical(
    subtypes, clinical_data, Z
)

# Progression
progression = progression_analyzer.analyze_cross_sectional_correlations(
    Z, clinical_data["updrs"]
)

# Biomarkers
outcomes = {"motor": clinical_data["updrs"]}
biomarkers = biomarker_analyzer.analyze_factor_outcome_associations(
    Z, outcomes
)

# 4. Print summary
print(f"Classification accuracy: {classification_results['random_forest']['detailed_metrics']['accuracy']:.3f}")
print(f"Optimal subtypes: {subtypes['optimal_k']}")
print(f"Progression R²: {progression['multiple_regression']['r2_score']:.3f}")
print(f"Significant biomarkers: {len([r for r in biomarkers['motor'] if r['pearson_p_value'] < 0.05])}")
```

---

## Integration with Existing Experiments

### In clinical_validation.py (after refactoring)
```python
from analysis.clinical import *

class ClinicalValidationExperiments(ExperimentFramework):
    def __init__(self, config, logger):
        super().__init__(config, None, logger)

        # Initialize clinical utilities
        self.clinical_metrics = ClinicalMetrics(logger=logger)
        self.data_processor = ClinicalDataProcessor(logger=logger)
        self.classifier = ClinicalClassifier(
            self.clinical_metrics,
            config=config_dict,
            logger=logger
        )
        self.subtype_analyzer = PDSubtypeAnalyzer(self.data_processor, logger)
        # ... etc

    def run_neuroimaging_clinical_validation(self, X_list, clinical_data, hypers, args):
        # Now just orchestrate calls to utilities
        sgfa_result = self.data_processor.run_sgfa_analysis(X_list, hypers, args)
        classification = self.classifier.test_factor_classification(...)
        # ... etc
```

### In model_comparison.py
```python
from analysis.clinical import ClinicalMetrics, ClinicalClassifier

class ModelArchitectureComparison(ExperimentFramework):
    def __init__(self, config, logger):
        super().__init__(config, None, logger)

        # Add clinical validation
        self.clinical_metrics = ClinicalMetrics()
        self.clinical_classifier = ClinicalClassifier(self.clinical_metrics)

    def compare_methods(self, ...):
        # ... existing comparison code ...

        # Add clinical performance for methods with factors
        if 'Z' in sgfa_result:
            clinical_perf = self.clinical_classifier.test_factor_classification(
                sgfa_result['Z'], clinical_labels, "sparseGFA"
            )
            sgfa_result['clinical_performance'] = clinical_perf
```

---

## Tips & Best Practices

### 1. Logger Usage
Always pass a logger for visibility into what's happening:
```python
import logging
logger = logging.getLogger(__name__)

metrics = ClinicalMetrics(logger=logger)  # Good
metrics = ClinicalMetrics()  # OK, but less visibility
```

### 2. Optional Dependencies
Use optional dependencies when you don't need all features:
```python
# Minimal - just metrics
metrics = ClinicalMetrics()
classifier = ClinicalClassifier(metrics)  # No config needed

# Full-featured - with config
classifier = ClinicalClassifier(
    metrics,
    config={"cross_validation": {"n_folds": 10}},
    logger=logger
)
```

### 3. Error Handling
Modules return dicts with 'error' key on failure:
```python
result = processor.run_sgfa_analysis(X_list, hypers, args)

if "error" in result:
    print(f"SGFA failed: {result['error']}")
elif result.get("convergence"):
    print(f"Success! LL = {result['log_likelihood']}")
```

### 4. Memory Management
For large datasets, modules use garbage collection:
```python
# PDSubtypeAnalyzer automatically cleans up KMeans objects
subtypes = analyzer.discover_pd_subtypes(Z)  # GC after each clustering
```

---

## Common Patterns

### Pattern 1: Sequential Pipeline
```python
# Run SGFA → Classify → Discover subtypes → Validate
result = processor.run_sgfa_analysis(X_list, hypers, args)
classification = classifier.test_factor_classification(result['Z'], labels, "SGFA")
subtypes = subtype_analyzer.discover_pd_subtypes(result['Z'])
validation = subtype_analyzer.validate_subtypes_clinical(subtypes, clinical_data, result['Z'])
```

### Pattern 2: Parallel Analysis
```python
# Multiple analyses on same factors
Z = sgfa_result['Z']

classification = classifier.test_factor_classification(Z, labels, "SGFA")
subtypes = subtype_analyzer.discover_pd_subtypes(Z)
progression = progression_analyzer.analyze_cross_sectional_correlations(Z, updrs_scores)
biomarkers = biomarker_analyzer.analyze_factor_outcome_associations(Z, outcomes)
```

### Pattern 3: Robustness Testing
```python
# Test stability/robustness
stability = subtype_analyzer.analyze_pd_subtype_stability(
    X_list, hypers, args, n_runs=10
)
robustness = biomarker_analyzer.test_biomarker_robustness(
    X_list, outcomes, hypers, args, n_bootstrap=100
)
```

---

## Troubleshooting

### Import Errors
```python
# If imports fail, check Python path
import sys
sys.path.insert(0, '/path/to/sgfa_qmap-pd')

from analysis.clinical import ClinicalMetrics
```

### Missing Dependencies
```python
# If a method requires an optional dependency:
biomarker = BiomarkerAnalyzer()  # No processor

# This will fail:
robustness = biomarker.test_biomarker_robustness(...)
# Returns: {"error": "ClinicalDataProcessor required for robustness testing"}

# Fix: Provide the dependency
biomarker = BiomarkerAnalyzer(data_processor=processor)
robustness = biomarker.test_biomarker_robustness(...)  # Now works!
```

### Config Issues
```python
# If config format is wrong, modules use defaults:
classifier = ClinicalClassifier(metrics, config={})
# Uses default: n_folds=5, stratified=True, random_seed=42

# To customize:
classifier = ClinicalClassifier(
    metrics,
    config={"cross_validation": {"n_folds": 10}}
)
```
