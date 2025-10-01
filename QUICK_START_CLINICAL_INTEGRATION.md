# Quick Start: Using Clinical Validation in Experiments

## Summary

Clinical validation has been integrated into both `model_comparison.py` and `sgfa_parameter_comparison.py`. It's **optional** and only runs when you provide clinical labels.

## What Was Added

### Files Modified
1. `experiments/model_comparison.py` (+16 lines)
   - Added clinical validation to sparseGFA and all traditional methods
2. `experiments/sgfa_parameter_comparison.py` (+15 lines)
   - Added clinical validation to SGFA parameter variants

### No Changes Needed
- ❌ Config files
- ❌ debug_experiments.py
- ❌ run_experiments.py

## Usage Examples

### Example 1: Python API with Clinical Labels

```python
from experiments.model_comparison import run_model_comparison
import numpy as np

# Your clinical labels (e.g., PD diagnosis: 0=Control, 1=PD)
clinical_labels = np.array([0, 1, 1, 0, 1, ...])  # Shape: (n_subjects,)

# Run model comparison WITH clinical validation
result = run_model_comparison(
    config="config.yaml",
    clinical_labels=clinical_labels  # Add this to enable clinical validation
)

# Access clinical performance for each method
for method_name, method_result in result["results"].model_results.items():
    if "clinical_performance" in method_result:
        print(f"\n{method_name} Clinical Performance:")
        for classifier, metrics in method_result["clinical_performance"].items():
            print(f"  {classifier}: accuracy={metrics['accuracy']:.3f}, "
                  f"f1={metrics['f1']:.3f}, roc_auc={metrics['roc_auc']:.3f}")
```

### Example 2: Load Clinical Labels from File

```python
import pandas as pd
from experiments.sgfa_parameter_comparison import run_sgfa_parameter_comparison

# Load clinical data
clinical_df = pd.read_csv("qMAP-PD_data/data_clinical/clinical.tsv", sep="\t")
clinical_labels = clinical_df["diagnosis"].values  # or whatever your column is

# Run parameter comparison WITH clinical validation
result = run_sgfa_parameter_comparison(
    config="config.yaml",
    clinical_labels=clinical_labels
)

# Compare clinical performance across parameters
for variant, variant_result in result["results"].model_results.items():
    if "clinical_performance" in variant_result:
        print(f"\n{variant}:")
        avg_accuracy = np.mean([
            m["accuracy"]
            for m in variant_result["clinical_performance"].values()
        ])
        print(f"  Average accuracy: {avg_accuracy:.3f}")
```

### Example 3: Without Clinical Labels (Backward Compatible)

```python
from experiments.model_comparison import run_model_comparison

# Run WITHOUT clinical validation (works exactly as before)
result = run_model_comparison(config="config.yaml")
# No clinical validation will run - computational metrics only
```

## What You Get

When you provide `clinical_labels`, each method is automatically tested with 4 classifiers:

### Classifiers
1. **Random Forest** - Ensemble decision trees
2. **Logistic Regression** - Linear classification
3. **SVM** - Support vector machine
4. **Gradient Boosting** - Advanced ensemble

### Metrics (for each classifier)
- **accuracy** - Overall classification accuracy
- **precision** - Positive predictive value
- **recall** - Sensitivity/true positive rate
- **f1** - Harmonic mean of precision/recall
- **roc_auc** - Area under ROC curve
- **confusion_matrix** - Full confusion matrix

### Results Structure
```python
result["clinical_performance"] = {
    "random_forest": {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1": 0.85,
        "roc_auc": 0.90,
        "confusion_matrix": [[45, 5], [6, 44]]
    },
    "logistic_regression": {...},
    "svm": {...},
    "gradient_boosting": {...}
}
```

## Log Output

With clinical validation enabled, you'll see:

```
✅ sparseGFA: 45.2s
  Clinical validation: 4 classifiers tested
✅ PCA: 2.1s
  Clinical validation: 4 classifiers tested
✅ ICA: 3.5s
  Clinical validation: 4 classifiers tested
```

Without clinical labels (default):

```
✅ sparseGFA: 45.2s
✅ PCA: 2.1s
✅ ICA: 3.5s
```

## Clinical Label Format

Your clinical labels should be:
- **Type:** numpy array or list
- **Shape:** `(n_subjects,)` - one label per subject
- **Values:**
  - Binary classification: `[0, 1, 1, 0, ...]`
  - Multi-class: `[0, 1, 2, 0, 2, 1, ...]`
  - String labels: `["Control", "PD", "PD", "Control", ...]`

## Advanced Clinical Modules

The integration uses 2 of the 7 available clinical modules:

### Currently Used
- ✅ `ClinicalMetrics` - Calculate accuracy, precision, recall, F1, ROC AUC
- ✅ `ClinicalClassifier` - Test multiple classifiers (RF, LR, SVM, GB)

### Available for Future Integration
- `ClinicalDataProcessor` - SGFA training and evaluation
- `PDSubtypeAnalyzer` - PD subtype discovery via clustering
- `DiseaseProgressionAnalyzer` - Longitudinal progression modeling
- `BiomarkerAnalyzer` - Biomarker discovery and validation
- `ExternalValidator` - Cross-cohort validation

All modules are in `analysis/clinical/` and ready to use.

## Error Handling

Clinical validation is **fail-safe**:

```python
# If clinical validation fails, experiments continue
try:
    clinical_perf = self.clinical_classifier.test_factor_classification(...)
    result["clinical_performance"] = clinical_perf
    self.logger.info(f"  Clinical validation: {len(clinical_perf)} classifiers tested")
except Exception as e:
    self.logger.warning(f"  Clinical validation failed: {str(e)}")
    # Experiment continues with computational metrics only
```

You'll see a warning in logs but experiments won't crash.

## Key Benefits

1. **Method Selection** - Choose sparseGFA vs PCA/ICA/FA based on clinical utility, not just computational metrics
2. **Parameter Optimization** - Find K and percW that maximize clinical performance
3. **Fair Comparison** - All methods evaluated on same clinical metrics
4. **Backward Compatible** - No breaking changes, works with or without clinical data

## Files for Reference

- Implementation: `experiments/model_comparison.py` (lines 76-81, 829-838, 988-997)
- Implementation: `experiments/sgfa_parameter_comparison.py` (lines 66-71, 762-771)
- Clinical modules: `analysis/clinical/` (7 modules, 2,593 lines)
- Documentation: `CLINICAL_INTEGRATION_COMPLETE.md`
- Verification: `INTEGRATION_VERIFICATION.md`

## Questions?

The integration is complete and ready to use. Just add `clinical_labels=your_labels` to enable clinical validation in your experiments!
