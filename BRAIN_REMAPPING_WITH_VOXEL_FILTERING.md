# Brain Remapping with Voxel Filtering

## Problem

When preprocessing drops voxels (via QC outlier removal or ROI-based spatial selection), the factor loadings W matrix has fewer rows than the original number of voxels. This breaks brain remapping because you can't map 850 loadings back to 1794 original voxel positions.

**Example**:
- Original: 1794 voxels with known (x, y, z) coordinates
- After QC outlier removal: 850 voxels (944 dropped)
- After ROI selection: 850 voxels
- Factor loadings W: 850 × K matrix
- **Problem**: Which 850 of the original 1794 positions do these correspond to?

## Solution

The preprocessing pipeline now **automatically saves filtered position lookup vectors** that track which voxels were retained and their spatial coordinates.

### Implementation

**File**: [data/preprocessing_integration.py](data/preprocessing_integration.py#L17-114)

The `_save_filtered_position_lookups()` function:

1. **Loads original position lookup** (all 1794 voxels with x, y, z coordinates)
2. **Applies QC outlier mask** from `preprocessor.outlier_masks_[view_name]`
3. **Applies ROI selection** from `preprocessor.selected_features_[view_name]`
4. **Saves filtered positions** to `{data_dir}/preprocessed_position_lookups/{roi_name}_filtered_position_lookup.csv`

### When It Runs

Automatically during preprocessing when:
- `enable_spatial_processing: true` (in config)
- Using `NeuroImagingPreprocessor` (advanced preprocessing)
- Processing imaging views (names starting with `volume_`)

### Output Files

**Location**: `{data_dir}/preprocessed_position_lookups/`

**Format** (same as original position lookups):
```csv
x,y,z
-12.5,8.3,4.2
-11.2,9.1,5.5
...
```

**Files created**:
- `sn_filtered_position_lookup.csv` - If processing substantia nigra
- `putamen_filtered_position_lookup.csv` - If processing putamen
- `lentiform_filtered_position_lookup.csv` - If processing lentiform nucleus
- etc.

**Size**: Matches the number of rows in the corresponding W matrix (e.g., 850 voxels)

## Usage for Brain Remapping

### Current Factor Stability Outputs

After running factor stability analysis, you have:
```
results/factor_stability_run_YYYYMMDD_HHMMSS/
├── chains/
│   └── chain_0/
│       ├── W_view_0.csv          # (850, K) - Factor loadings for imaging view
│       └── Z.csv                 # (86, K) - Factor scores
└── stability_analysis/
    └── consensus_W.csv           # Consensus loadings across chains
```

### Filtered Position Lookups

Now available in:
```
qMAP-PD_data/preprocessed_position_lookups/
├── sn_filtered_position_lookup.csv        # 850 voxels (if 850 retained)
├── putamen_filtered_position_lookup.csv
└── ...
```

### Brain Remapping Workflow

To map factor loadings back to brain space:

```python
import pandas as pd
import numpy as np

# 1. Load factor loadings (e.g., for Factor 0)
W = pd.read_csv("results/.../chains/chain_0/W_view_0.csv", index_col=0)
factor_0_loadings = W["Factor_0"].values  # Shape: (850,)

# 2. Load filtered position lookup
positions = pd.read_csv("qMAP-PD_data/preprocessed_position_lookups/sn_filtered_position_lookup.csv")
# Shape: (850, 3) with columns [x, y, z]

# 3. Create 3D brain map
brain_map = pd.DataFrame({
    'x': positions['x'],
    'y': positions['y'],
    'z': positions['z'],
    'loading': factor_0_loadings
})

# 4. Visualize (example with matplotlib)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    brain_map['x'],
    brain_map['y'],
    brain_map['z'],
    c=brain_map['loading'],
    cmap='RdBu_r',
    s=50,
    alpha=0.6
)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Factor 0 Loadings in Brain Space')
plt.colorbar(scatter, label='Factor Loading')
plt.savefig('factor_0_brain_map.png')
```

## Verification

After your current runs complete, check:

```bash
# Check if filtered position lookups were created
ls -lh qMAP-PD_data/preprocessed_position_lookups/

# Verify the size matches your data
# Machine 1 & 2 (850 voxels):
wc -l qMAP-PD_data/preprocessed_position_lookups/sn_filtered_position_lookup.csv
# Should show: 851 (850 voxels + 1 header row)

# Machine 3 (1794 voxels):
wc -l qMAP-PD_data/preprocessed_position_lookups/sn_filtered_position_lookup.csv
# Should show: 1795 (1794 voxels + 1 header row)
```

Compare factor loadings matrix size:
```bash
# Check W matrix dimensions
head -1 results/.../chains/chain_0/W_view_0.csv  # See how many factors (columns)
wc -l results/.../chains/chain_0/W_view_0.csv     # Should match filtered positions + 1
```

## Preprocessing Steps That Drop Voxels

The filtered position lookup accounts for:

### 1. QC Outlier Removal
- **Config parameter**: `qc_outlier_threshold: 3.0`
- **What it does**: Removes voxels with values >3 standard deviations from mean
- **Tracking**: Saved in `preprocessor.outlier_masks_[view_name]`

### 2. ROI-Based Spatial Selection
- **Config parameter**: `roi_based_selection: true`
- **Config parameter**: `min_voxel_distance: 3.0` (mm)
- **What it does**: Selects spatially distributed voxels (avoids clustering)
- **Tracking**: Saved in `preprocessor.selected_features_[f"{view_name}_roi_indices"]`

### Comparison of Your Current Runs

| Machine | Config | Voxels | QC Outlier | ROI Selection |
|---------|--------|--------|------------|---------------|
| 1 | config.yaml | 850 | Yes (3.0σ) | Yes (3mm) |
| 2 | config_K50.yaml | 850 | Yes (3.0σ) | Yes (3mm) |
| 3 | config_K50_full_voxels.yaml | ? | Yes (3.0σ) | No |

**Machine 3 Note**: Even with `roi_based_selection: false`, QC outlier removal still applies, so you'll likely have ~850-900 voxels (not full 1794). To get truly all voxels, you'd need to also set `qc_outlier_threshold` very high (e.g., 10.0) or disable QC.

## Log Messages

During preprocessing, you'll see:

```
INFO: Creating filtered position lookup for volume_sn_voxels...
INFO:   Applying QC outlier mask: 850/1794 voxels kept
INFO:   Applying ROI selection (indices): 850 voxels selected
INFO:   Saved filtered position lookup: /path/to/qMAP-PD_data/preprocessed_position_lookups/sn_filtered_position_lookup.csv
INFO:   1794 → 850 voxels (47.4% retained)
INFO:   Saved 1 filtered position lookup files
```

## Future Runs

For future experiments, if you want to preserve all voxels for complete brain coverage:

```yaml
preprocessing:
  qc_outlier_threshold: 10.0      # Very permissive (keeps almost all)
  roi_based_selection: false      # No spatial subsampling
  min_voxel_distance: 0.0         # No distance constraint
```

**Trade-off**: More voxels = more complete brain maps, but:
- Higher memory usage
- More noise (outlier voxels included)
- Spatially redundant information
- Longer computation time

## Technical Details

### Position Lookup Application Order

1. **Start**: Original position lookup (1794 voxels)
2. **QC filter**: Apply `outlier_masks_[view_name]` → reduces to ~850
3. **ROI filter**: Apply `selected_features_[view_name]` to QC-filtered set → final 850
4. **Save**: Filtered positions matching the final feature matrix

### Coordinate System

The position lookup uses **MNI space** (Montreal Neurological Institute):
- **Origin**: Anterior commissure
- **Units**: Millimeters (mm)
- **Orientation**: RAS (Right-Anterior-Superior)
- **Range**: Typically -90 to +90 mm in each dimension

### Matching Factor Loadings to Positions

```python
# Row i in filtered_position_lookup.csv corresponds to:
# - Row i in W_view_j.csv (factor loadings)
# - Feature i in the preprocessed imaging view

# Example:
W_df = pd.read_csv("W_view_0.csv", index_col=0)  # Index = feature names
positions = pd.read_csv("sn_filtered_position_lookup.csv")

assert len(W_df) == len(positions), "Mismatch in voxel count!"
# Should pass if everything is working correctly
```

## References

- Original position lookups: `qMAP-PD_data/volume_matrices/*_position_lookup.tsv`
- Preprocessing code: [data/preprocessing.py](data/preprocessing.py)
- Integration code: [data/preprocessing_integration.py](data/preprocessing_integration.py)
- Spatial utilities: `SpatialProcessingUtils` class in preprocessing.py
