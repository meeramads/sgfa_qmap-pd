# Adjusting MAD Threshold for QC Outlier Detection

## Quick Reference

### Command-Line Flag

```bash
python run_experiments.py \
  --config config.yaml \
  --experiments factor_stability \
  --qc-outlier-threshold 5.0
```

**Default**: 3.0 (if not specified)
**Recommended range**: 2.0 - 5.0
**Effect**: Higher values → retain more voxels

### Config File

Alternatively, set in `config.yaml`:

```yaml
preprocessing:
  qc_outlier_threshold: 5.0  # Adjust this value
  roi_based_selection: false  # Optional: disable spatial subsampling
```

## How to Choose the Right Threshold

### Step 1: Run Data Validation with MAD Analysis

```bash
python run_experiments.py \
  --config config.yaml \
  --experiments data_validation
```

This automatically runs MAD threshold EDA and generates:

**Outputs** (in `results/data_validation_run_YYYYMMDD_HHMMSS/`):
- `mad_distribution_{roi}.png` - Shows distribution of MAD scores
- `elbow_analysis_all_views.png` - Data-driven threshold recommendation
- `mad_threshold_recommendations.txt` - Detailed recommendations
- `mad_threshold_summary_table.csv` - Comparison table

### Step 2: Review the Recommendations

Open `mad_threshold_recommendations.txt` and look for:

```
ELBOW POINT: 4.5
  - Retention at elbow: 85.2%
  - Interpretation: Permissive - retains most voxels

CURRENT THRESHOLD (3.0) PERFORMANCE:
  - Voxels retained: 850 (47.4%)
  - Voxels removed: 944
  - Variance preserved: 82.3%

RECOMMENDATION:
  → Consider INCREASING threshold to 4.5
    Current threshold may be too strict, removing biological variability
```

### Step 3: Run with Adjusted Threshold

```bash
python run_experiments.py \
  --config config_K50.yaml \
  --experiments factor_stability \
  --qc-outlier-threshold 4.5
```

## Interpretation Guide

### Threshold Values

| Threshold | Effect | Typical Retention | Use Case |
|-----------|--------|-------------------|----------|
| **2.0-2.5** | Very strict | 30-50% voxels | Clean signal extraction, hypothesis testing |
| **3.0** (default) | Moderate | 40-60% voxels | Balanced artifact removal |
| **3.5-4.5** | Permissive | 70-85% voxels | Exploratory analysis, preserve variability |
| **5.0+** | Very permissive | 85-95% voxels | Preserve biological heterogeneity, subtypes |

### Your Current Situation (from logs)

```
WARNING:root:Removing 944 outlier voxels from volume_sn_voxels
INFO:root:Final shape for volume_sn_voxels: (86, 850)
```

- **Original**: 1794 voxels
- **After QC (threshold=3.0)**: 850 voxels (**47.4% retained**)
- **Removed**: 944 voxels (52.6%)

This is **unusually high** removal rate! Suggests:
1. High biological variability in your PD patients (may be real subtypes)
2. Potential data quality issues in substantia nigra imaging
3. Threshold of 3.0 is too strict for your specific data

## Recommendations for Different Goals

### For Exploratory Factor Analysis (Finding PD Subtypes)
**Use higher threshold** to preserve heterogeneity:

```bash
--qc-outlier-threshold 4.5
# Or
--qc-outlier-threshold 5.0
```

**Rationale**: "Outliers" may represent distinct disease subtypes. Removing them discards exactly the biological variability you're trying to find.

**Expected**: ~1,400-1,600 voxels retained (75-90%)

### For Clean Signal Extraction (Hypothesis Testing)
**Use moderate threshold** for better SNR:

```bash
--qc-outlier-threshold 3.0  # Current default
# Or slightly more permissive:
--qc-outlier-threshold 3.5
```

**Rationale**: Removes artifacts while preserving main effects.

**Expected**: ~850-1,100 voxels retained (47-60%)

### For Complete Brain Mapping
**Disable QC entirely** or use very high threshold:

```bash
--qc-outlier-threshold 10.0  # Effectively disabled
```

**Rationale**: Need all voxel positions for spatial remapping.

**Expected**: ~1,750-1,794 voxels retained (98-100%)

## Trade-offs

```
Lower Threshold (2.0-3.0)        vs        Higher Threshold (4.0-5.0)
├─ Cleaner data                           ├─ More complete coverage
├─ Better SNR                             ├─ Preserves variability
├─ More stable factors                    ├─ May find more subtypes
├─ Faster computation                     ├─ Better spatial remapping
└─ ❌ Incomplete brain maps               └─ ❌ Includes some artifacts
```

## Examples

### Example 1: Conservative Analysis
```bash
python run_experiments.py \
  --config config.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --regress-confounds age sex tiv \
  --qc-outlier-threshold 3.0
```
- Clean data, ~850 voxels
- Good for hypothesis testing

### Example 2: Exploratory Subtyping
```bash
python run_experiments.py \
  --config config.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --regress-confounds age sex tiv \
  --qc-outlier-threshold 5.0
```
- More voxels (~1,500), preserves heterogeneity
- Good for finding PD subtypes

### Example 3: Complete Brain Mapping
```bash
python run_experiments.py \
  --config config.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --regress-confounds age sex tiv \
  --qc-outlier-threshold 10.0
```
- Nearly all voxels (~1,794)
- Enables complete spatial remapping

## Understanding MAD Scores

The MAD (Median Absolute Deviation) score measures how extreme a voxel's values are across subjects:

```
MAD score = max(|value - median| / (MAD × 1.4826))
```

- **MAD score < 2.0**: Normal variation (retained)
- **MAD score 2.0-3.0**: Moderate outlier (depends on threshold)
- **MAD score 3.0-5.0**: Strong outlier (removed at default threshold)
- **MAD score > 5.0**: Extreme outlier (likely artifact)

If **any** subject has MAD score > threshold for a voxel, that entire voxel is removed from ALL subjects.

## Spatial Remapping Considerations

When voxels are removed, you need filtered position lookups for brain remapping.

**Automatic (with latest code)**:
```bash
# Filtered position lookups saved automatically to:
qMAP-PD_data/preprocessed_position_lookups/sn_filtered_position_lookup.csv
```

These files map the retained voxels (e.g., 850) back to their 3D brain coordinates.

**To verify**:
```bash
# Check filtered position lookup exists
ls -lh qMAP-PD_data/preprocessed_position_lookups/

# Should match W matrix size
wc -l qMAP-PD_data/preprocessed_position_lookups/sn_filtered_position_lookup.csv
wc -l results/.../chains/chain_0/W_view_0.csv
# Both should be 851 (850 voxels + header)
```

## Checking Results

After running with a new threshold, check the logs:

```bash
INFO:root:Applying quality control to volume_sn_voxels
WARNING:root:Removing XXX outlier voxels from volume_sn_voxels
INFO:root:Final shape for volume_sn_voxels: (86, YYY)
```

- **XXX** = number of voxels removed
- **YYY** = number of voxels retained

Calculate retention rate:
```
Retention % = YYY / (YYY + XXX) × 100
```

**Target retention rates**:
- Exploratory: 75-90%
- Balanced: 50-75%
- Conservative: 40-60%

## Further Reading

- MAD threshold EDA outputs: `results/data_validation_run_*/`
- Preprocessing code: `data/preprocessing.py` (line 215-234)
- Filtered position lookups: `BRAIN_REMAPPING_WITH_VOXEL_FILTERING.md`
