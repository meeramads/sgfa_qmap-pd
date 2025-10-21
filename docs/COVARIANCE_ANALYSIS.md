# Covariance Structure Analysis

## Overview

The covariance analysis module provides tools to analyze and visualize the covariance structure of multi-view data, specifically examining:

1. **Intra-modality variance** - How features within each view (e.g., imaging, clinical) relate to each other
2. **Inter-modality covariance** - How features across different views relate to each other

This is critical for understanding the relationships between different data modalities and validating that multi-view integration methods like GFA are appropriate.

## Automatic Generation

Covariance analysis is automatically performed during data validation experiments:

```bash
python run_experiments.py \
  --config config_convergence.yaml \
  --experiments data_validation \
  --select-rois volume_sn_voxels.tsv
```

Results are saved to:

```bash
results/{run_name}/data_validation_{timestamp}/plots/
├── block_covariance_matrix.png       # Full block-structured covariance
└── inter_vs_intra_correlation.png    # Comparison of inter vs intra
```

## Generated Plots

### 1. Block-Structured Covariance Matrix

Shows the full correlation/covariance matrix with clear block structure:

- **Diagonal blocks**: Intra-modality correlations (within imaging, within clinical)
- **Off-diagonal blocks**: Inter-modality correlations (between imaging and clinical)

**Features:**

- Block boundaries clearly marked
- View names labeled
- Summary statistics displayed
- Automatic subsampling for large matrices (default: 500 features)

### 2. Inter vs Intra Comparison

Two-panel comparison:

- **Left panel**: Violin plots showing distribution of correlations for each view/pair
- **Right panel**: Bar chart of mean absolute correlations

**Color coding:**

- Blue: Intra-modality (within-view) correlations
- Red: Inter-modality (between-view) correlations

## Interpretation

### Expected Patterns for Multi-View Data

**Good multi-view structure:**

- Intra-modality correlations > Inter-modality correlations
- Clear block structure in covariance matrix
- Distinct modalities contribute different information

**Example:**

```yaml
Intra-modality correlation statistics:
  imaging:
    Mean correlation: 0.450
    Max |correlation|: 0.920
  clinical:
    Mean correlation: 0.320
    Max |correlation|: 0.780

Inter-modality correlation statistics:
  imaging_vs_clinical:
    Mean correlation: 0.120
    Max |correlation|: 0.450
```

**Interpretation**: Imaging features are more correlated with each other (0.45 mean) than with clinical features (0.12 mean), indicating distinct information sources. This supports multi-view integration.

### Warning Signs

**High inter-modality correlations (> 0.5 mean):**

- Views may be redundant
- Consider whether both views are needed
- May indicate data leakage or preprocessing artifacts

**Low intra-modality correlations (< 0.1 mean):**

- Features within view are nearly independent
- May benefit from feature selection
- Check for preprocessing issues (e.g., over-normalization)

**No block structure visible:**

- Views may not be distinct
- Check preprocessing and view definitions
- May need dimensionality reduction

## Programmatic Usage

### Compute Covariance Structure

```python
from visualization.covariance_plots import compute_covariance_structure

cov_structure = compute_covariance_structure(
    X_list=[X_imaging, X_clinical],
    view_names=['imaging', 'clinical']
)

# Access results
full_corr = cov_structure['full_corr']  # Full correlation matrix
intra_cov = cov_structure['intra_cov']  # Dict of within-view covariances
inter_cov = cov_structure['inter_cov']  # Dict of between-view covariances
```

### Analyze Statistics

```python
from visualization.covariance_plots import analyze_covariance_statistics

stats = analyze_covariance_statistics(cov_structure)

# Intra-modality stats
for view_name, view_stats in stats['intra_stats'].items():
    print(f"{view_name}: mean_corr={view_stats['corr_mean']:.3f}")

# Inter-modality stats
for pair_name, pair_stats in stats['inter_stats'].items():
    print(f"{pair_name}: mean_corr={pair_stats['corr_mean']:.3f}")
```

### Create Plots

```python
from visualization.covariance_plots import (
    plot_block_covariance_matrix,
    plot_inter_vs_intra_comparison
)

# Block covariance matrix
fig = plot_block_covariance_matrix(
    cov_structure,
    output_path='covariance_matrix.png',
    use_correlation=True,  # Use correlation instead of covariance
    subsample=500  # Subsample to 500 features for visualization
)

# Inter vs intra comparison
fig = plot_inter_vs_intra_comparison(
    cov_structure,
    output_path='inter_vs_intra.png'
)
```

### Complete Report

```python
from visualization.covariance_plots import create_covariance_report

report = create_covariance_report(
    X_list=[X_imaging, X_clinical],
    view_names=['imaging', 'clinical'],
    output_dir='./plots',
    subsample=500
)

# Access results
cov_structure = report['cov_structure']
stats = report['stats']
```

## Technical Details

### Subsampling Strategy

For large matrices (> 500 features), automatic subsampling is applied:

- Samples proportionally from each view
- Preserves block structure
- Uses linear interpolation to select representative features

### Correlation vs Covariance

**Correlation** (default):

- Scale-invariant
- Easier to interpret (-1 to 1 range)
- Recommended for most analyses

**Covariance**:

- Scale-dependent
- Shows actual variance/covariance magnitudes
- Useful for understanding raw data properties

### Computational Complexity

- **Time**: O(D² × N) where D = total features, N = samples
- **Memory**: O(D²) for covariance matrix
- **Recommendation**: Use subsampling for D > 1000

## Integration with GFA

Covariance analysis helps validate GFA assumptions:

1. **Multi-view appropriateness**: If inter-modality correlations ≈ intra-modality correlations, single-view analysis may suffice

2. **Shared structure**: GFA assumes views share latent factors. High inter-modality correlations support this.

3. **View-specific variance**: GFA models view-specific loadings. Clear block structure validates this design.

## References

- Klami, A., et al. (2015). Group factor analysis. IEEE TNNLS.
- Virtanen, S., et al. (2012). Bayesian group factor analysis. AISTATS.

## Example Output

```bash
================================================================================
COVARIANCE STRUCTURE ANALYSIS
================================================================================

Data structure:
  Number of views: 2
  Number of samples: 86
  volume_sn_voxels: 531 features
  clinical: 14 features

Intra-modality correlation statistics:
  volume_sn_voxels:
    Mean correlation: 0.324
    Std correlation: 0.182
    Max |correlation|: 0.891
  clinical:
    Mean correlation: 0.156
    Std correlation: 0.245
    Max |correlation|: 0.678

Inter-modality correlation statistics:
  volume_sn_voxels_vs_clinical:
    Mean correlation: 0.089
    Std correlation: 0.124
    Max |correlation|: 0.432

Generating covariance plots...
Plots saved to results/data_validation_20251020/plots
================================================================================
```

---

**Module**: `visualization/covariance_plots.py`
**Integration**: `experiments/data_validation.py`
**Added**: October 20, 2025
