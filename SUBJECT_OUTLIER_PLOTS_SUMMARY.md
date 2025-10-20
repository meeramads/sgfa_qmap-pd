# Subject Outlier Analysis Plots

## Overview

The data_validation experiment automatically generates comprehensive subject-level outlier analysis plots for both **imaging** and **clinical** data.

## Existing Plots

The following plots are automatically created during data_validation:

### 1. Subject Outlier Distributions
**File:** `subject_outlier_distributions.png`
- **Purpose:** Histogram showing distribution of outlier percentages across all subjects
- **Modalities:** Imaging views (SN, Putamen, Lentiform, etc.)
- **Shows:** How many subjects have what percentage of outlier voxels

### 2. Subject Outlier Comparison
**File:** `subject_outlier_comparison.png`
- **Purpose:** Box plot comparing flagged vs normal subjects
- **Shows:** Distribution differences between healthy and problematic subjects
- **Threshold:** Subjects with >5% outlier voxels are flagged

### 3. Subject Outlier Scatter
**File:** `subject_outlier_scatter.png`
- **Purpose:** Scatter plot of outlier percentage per subject
- **Shows:** Each subject as a point, flagged subjects annotated
- **Interactive:** Subject IDs labeled for flagged subjects

### 4. Subject Outlier Heatmap
**File:** `subject_outlier_heatmap.png`
- **Purpose:** Heatmap showing outliers across subjects × ROIs
- **Shows:** Which subjects have outliers in which brain regions
- **Sorting:** Subjects sorted by severity (worst first)

### 5. Top Outlier Subjects
**File:** `subject_outlier_top20.png`
- **Purpose:** Bar chart of 20 subjects with highest outlier percentages
- **Shows:** Subject IDs and their max outlier percentage
- **Useful:** Quick identification of problematic subjects

### 6. Flagged Subjects Report
**File:** `subject_outliers_flagged.csv` and `subject_outliers_summary.txt`
- **Purpose:** Detailed list of flagged subjects
- **Contains:** Subject IDs, outlier percentages per ROI, severity ranking

## Location

These plots are saved in the data_validation experiment output directory:

```
results/data_validation_<timestamp>/
├── plots/
│   ├── subject_outlier_distributions.png
│   ├── subject_outlier_comparison.png
│   ├── subject_outlier_scatter.png
│   ├── subject_outlier_heatmap.png
│   └── subject_outlier_top20.png
├── subject_outliers_flagged.csv
└── subject_outliers_summary.txt
```

## How Outliers Are Detected

### Imaging Data (Voxel-Level)
1. For each voxel, compute median and MAD across subjects
2. For each subject, count voxels where value > median + (MAD × threshold)
3. Calculate percentage of outlier voxels per subject
4. Flag subjects with >5% outlier voxels

### Clinical Data
Currently, the clinical data outlier analysis is not yet implemented. We'll add this below.

## Currently Missing: Clinical Data Outlier Analysis

The current plots only show **imaging** data outliers. We need to add **clinical** data outlier analysis.

### Proposed Addition

Add a comprehensive multi-modal outlier analysis that includes:

1. **Clinical Variables:**
   - Age outliers (z-score > 3)
   - Motor symptoms outliers (unusual patterns)
   - TIV outliers

2. **Combined View:**
   - Subject-level summary across all modalities
   - Identify subjects that are outliers in multiple modalities
   - Cross-modal outlier patterns

3. **New Plots:**
   - Clinical outlier scatter (similar to imaging)
   - Multi-modal outlier summary (imaging + clinical)
   - Correlation between imaging and clinical outliers

## Implementation Status

✅ **Implemented:**
- Imaging outlier detection (voxel-level MAD)
- Subject-level outlier aggregation
- 5 visualization plots for imaging data
- CSV/TXT reports for flagged subjects

❌ **To Implement:**
- Clinical data outlier detection
- Multi-modal outlier visualization
- Cross-modal outlier correlation analysis

## Usage

To generate subject outlier plots:

```bash
python run_experiments.py --config config_convergence.yaml \
  --experiments data_validation \
  --select-rois volume_sn_voxels.tsv
```

Plots will be in:
```
results/data_validation_fixed_K20_percW33_slab4_2_MAD3.0_<timestamp>/plots/
```

## Example Output

After running data_validation, you'll see:

```
INFO:experiments.data_validation:👤 Analyzing subject-level data quality...
INFO:experiments.data_validation:      Mean outlier %: 2.3%
INFO:experiments.data_validation:      Max outlier %: 15.7%
INFO:experiments.data_validation:      Saved subject outlier distribution plot: subject_outlier_distributions.png
INFO:experiments.data_validation:      Saved subject outlier comparison plot: subject_outlier_comparison.png
INFO:experiments.data_validation:      Saved subject outlier scatter plot: subject_outlier_scatter.png
INFO:experiments.data_validation:      Saved subject outlier heatmap: subject_outlier_heatmap.png
INFO:experiments.data_validation:      Saved top outlier subjects plot: subject_outlier_top20.png
INFO:experiments.data_validation:   📝 Flagged 8 subjects with >5% outlier voxels
```

## References

Subject-level outlier analysis implementation:
- **File:** `experiments/data_validation.py`
- **Functions:**
  - `_analyze_subject_level_outliers()` - Main analysis function
  - `_plot_subject_outlier_distributions()` - Histogram plots
  - `_plot_subject_outlier_comparison()` - Box plots
  - `_plot_subject_outlier_scatter()` - Scatter plots
  - `_plot_subject_outlier_heatmap()` - Heatmap visualization
  - `_plot_top_outlier_subjects()` - Top 20 bar chart
  - `_identify_flagged_subjects()` - Subject flagging logic

---

**Status:** Imaging outlier analysis is fully implemented.
**Next:** Add clinical data outlier analysis and multi-modal summary visualization.
