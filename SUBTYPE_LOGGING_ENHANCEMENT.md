# Subtype Logging Enhancement

This branch adds automatic subtype determination with clear logging to the SGFA qMAP-PD analysis pipeline.

## New Features

### 1. **Automatic Subtype Optimization**
- Tests literature-supported cluster numbers: **2, 3, 4** (based on PD research)
- Uses composite scoring: 60% silhouette + 40% Calinski-Harabasz
- Evaluates cluster balance and minimum cluster sizes
- Falls back to traditional 3-cluster solution if optimization fails

### 2. **Clear Console Logging**
During analysis, you'll now see:

```
Determining optimal number of PD subtypes...
  2 clusters: Silhouette=0.2371, Calinski-Harabasz=18.16, Balance=0.724
  3 clusters: Silhouette=0.2992, Calinski-Harabasz=22.12, Balance=0.750
  4 clusters: Silhouette=0.2560, Calinski-Harabasz=17.04, Balance=0.200

OPTIMAL SUBTYPES FOUND: 3 clusters for fold 1
   Best silhouette score: 0.2992
   Best Calinski-Harabasz score: 22.12
   Subtype distribution: {0: 15, 1: 20, 2: 15}
```

### 3. **Cross-Fold Summary**
At the end of CV, you get a comprehensive summary:

```
============================================================
FINAL SUBTYPE OPTIMIZATION SUMMARY:
   Most frequent optimal clusters: 3 (4/5 folds)
   Range of optimal clusters: 2 - 4
   Distribution across folds: {3: 4, 2: 1}
   Interpretation: Classic TD/PIGD/Mixed motor subtypes
============================================================
```

## Configuration Options

### Enable/Disable Automatic Optimization
```python
config = NeuroImagingCVConfig()
config.auto_optimize_subtypes = True  # Enable automatic determination
config.subtype_candidate_range = [2, 3, 4]  # Test these cluster numbers
```

### Command Line Usage
```bash
# Enable subtype optimization
python -m analysis.cross_validation_library --dataset qmap_pd --optimize_for_subtypes

# Traditional fixed 3 clusters (backward compatibility)
python -m analysis.cross_validation_library --dataset qmap_pd
```

## Methods Added

### `_find_optimal_subtypes()`
- **Input**: Factor scores (Z), candidate cluster numbers, fold ID
- **Output**: Optimal number of clusters, detailed metrics
- **Evaluation**: Tests each candidate with K-means, computes validation metrics
- **Logging**: Detailed progress and results for each candidate

### Enhanced `_analyze_subtype_consistency()`
- **New**: Cross-fold summary with interpretation
- **Added**: Distribution analysis across CV folds
- **Improved**: Clinical interpretation of subtype numbers

## Testing

Run the test script to verify functionality:
```bash
python test_subtype_logging.py
```

Expected output shows automatic detection of 3 optimal subtypes with clear logging.

## Backward Compatibility

- **Default behavior**: Automatic optimization enabled
- **Fallback**: If optimization fails, uses traditional 3 clusters
- **Configuration**: Can disable via `config.auto_optimize_subtypes = False`

## Literature Support

The candidate range [2, 3, 4] is based on recent PD literature:

- **2 clusters**: Fast vs. slow progressors (Hähnel et al. 2024)
- **3 clusters**: TD/PIGD/Mixed or mild/moderate/severe (Chen et al. 2023)
- **4 clusters**: Extended phenotype classifications

## Usage in Remote Workstation

This enhancement is safely branched and won't affect your running experiments on the remote workstation. When ready, you can:

1. **Test locally**: Use the test script
2. **Merge when ready**: After current experiments complete
3. **Enable in configs**: Set `auto_optimize_subtypes: true` in YAML files

---

**Branch**: `feature/subtype-logging-enhancement`
**Files Modified**:
- `analysis/cross_validation_library.py` (main implementation)
- `test_subtype_logging.py` (test script)
- `SUBTYPE_LOGGING_ENHANCEMENT.md` (documentation)