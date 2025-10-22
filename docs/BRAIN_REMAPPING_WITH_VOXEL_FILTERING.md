# Brain Remapping with Voxel Filtering

## Problem

When preprocessing drops voxels via MAD-based quality control, the factor loadings W matrix has fewer rows than the original number of voxels. This breaks brain remapping because you can't map reduced loadings back to the original voxel positions.

**Example**:

- Original: 1794 voxels with known (x, y, z) coordinates
- After MAD QC (threshold=3.0): ~531 voxels (70% dropped)
- Factor loadings W: 531 × K matrix
- **Problem**: Which 531 of the original 1794 positions do these correspond to?

## Solution

The preprocessing pipeline **automatically saves filtered position lookup vectors** that track which voxels were retained and their spatial coordinates.

### Implementation

**File**: [data/preprocessing.py](../data/preprocessing.py) (lines 1229-1297)

The `_filter_and_save_position_lookups()` function:

1. **Loads original position lookup** (all 1794 voxels with x, y, z coordinates)
2. **Applies cumulative mask** tracking all preprocessing steps (MAD filtering, imputation)
3. **Saves filtered positions** to experiment output directory

### When It Runs

Automatically during preprocessing when:

- `enable_spatial_processing: true` (in config)
- Using `NeuroImagingPreprocessor`
- Processing imaging views (names starting with `volume_`)

### Output Files

**Location**: `results/<run>/filtered_positions/`

**Format**:

```tsv
-12.5 8.3 4.2
-11.2 9.1 5.5
...
```

(No header, tab-separated, matching original position lookup format)

**Files created**:

- `position_sn_voxels_filtered.tsv` - If processing substantia nigra
- `position_putamen_voxels_filtered.tsv` - If processing putamen
- etc.

**Size**: Matches the number of rows in the corresponding W matrix (e.g., 531 voxels for MAD 3.0)

## Current Preprocessing Pipeline (config_convergence.yaml)

```yaml
preprocessing:
  qc_outlier_threshold: 3.0       # MAD threshold
  variance_threshold: 0.0         # Disabled
  roi_based_selection: false      # Disabled
```

### Preprocessing Flow

```
Original: 1794 voxels
    ↓
MAD filtering (threshold=3.0)
    ↓
~531 voxels retained (~70% removed)
    ↓
Position tracking saves filtered positions
```

**No additional filtering**: Variance threshold and ROI-based selection are disabled to preserve biological signal.

## Usage for Brain Remapping

### Factor Stability Outputs

After running factor stability analysis:

```bash
results/all_rois-sn_K20_percW33_MAD3.0_YYYYMMDD_HHMMSS/
├── 03_factor_stability_K20_percW33/
│   ├── chains/
│   │   ├── chain_0/
│   │   │   ├── W_view_0.csv     # (531, K) - Imaging loadings
│   │   │   └── Z.csv            # (86, K) - Factor scores
│   │   └── ...
│   └── stability_analysis/
│       ├── consensus_factor_loadings_volume_sn_voxels.csv  # Consensus W
│       └── consensus_factor_scores.csv                      # Consensus Z
└── filtered_positions/
    └── position_sn_voxels_filtered.tsv  # (531, 3) - Voxel coordinates
```

### Brain Remapping Workflow

```python
import pandas as pd
import numpy as np

# 1. Load consensus factor loadings
W = pd.read_csv("results/.../consensus_factor_loadings_volume_sn_voxels.csv", index_col=0)
factor_0_loadings = W["Factor_0"].values  # Shape: (531,)

# 2. Load filtered position lookup
positions = pd.read_csv(
    "results/.../filtered_positions/position_sn_voxels_filtered.tsv",
    sep='\t',
    header=None,
    names=['x', 'y', 'z']
)
# Shape: (531, 3)

# 3. Verify dimensions match
assert len(factor_0_loadings) == len(positions), "Dimension mismatch!"

# 4. Create brain map
brain_map = pd.DataFrame({
    'x': positions['x'],
    'y': positions['y'],
    'z': positions['z'],
    'loading': factor_0_loadings
})

# 5. Visualize
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
ax.set_title('Factor 0 Loadings in Substantia Nigra')
plt.colorbar(scatter, label='Factor Loading')
plt.savefig('factor_0_brain_map.png', dpi=300)
```

## Verification

Check that filtered positions were created:

```bash
# List filtered position files
ls -lh results/all_rois-sn_*/filtered_positions/

# Verify size matches loadings (MAD 3.0 example)
wc -l results/.../filtered_positions/position_sn_voxels_filtered.tsv
# Should show: 531 (no header)

# Compare to W matrix
wc -l results/.../consensus_factor_loadings_volume_sn_voxels.csv
# Should show: 532 (531 voxels + 1 header)
```

## Log Messages During Preprocessing

You'll see output like:

```
INFO:root:Applying MAD-based quality control to volume_sn_voxels (1794 voxels, threshold=3.0)
INFO:root:  MAD outlier detection: threshold=3.0, outliers=1263/1794 voxels
WARNING:root:  Outlier voxels detected: 1263/1794 (70.4%)
INFO:root:  Voxels retained: 531/1794 (29.6%)
INFO:root:Final shape for volume_sn_voxels: (86, 531)
INFO:root:Feature retention: 531/1794 (29.1%)
INFO:root:  Saved filtered position file: .../position_sn_voxels_filtered.tsv
```

## MAD Threshold Effects

| Threshold | Voxels Retained | Use Case |
|-----------|----------------|----------|
| 2.5 | ~450 (25%) | Stringent QC, cleanest signal |
| 3.0 | ~531 (30%) | **Default** - Moderate QC |
| 4.5 | ~650 (36%) | Permissive, preserve heterogeneity |
| 5.0 | ~700 (39%) | Very permissive, subtype discovery |
| 100.0 | 1794 (100%) | No QC (not recommended) |

**Command-line override**:

```bash
python run_experiments.py --config config_convergence.yaml \
  --qc-outlier-threshold 5.0
```

## Technical Details

### Position Tracking Implementation

The pipeline uses **cumulative masking** to track all transformations:

```python
# Start with all voxels
cumulative_mask = np.ones(1794, dtype=bool)

# Apply MAD filtering
mad_mask = mad_scores <= threshold
cumulative_mask &= mad_mask  # Now: 531 True values

# Apply to positions
filtered_positions = original_positions[cumulative_mask]
# Result: 531 rows matching W matrix
```

### Coordinate System

Position lookups use **MNI space** (Montreal Neurological Institute):

- **Origin**: Anterior commissure
- **Units**: Millimeters (mm)
- **Orientation**: RAS (Right-Anterior-Superior)
- **Typical range**: -90 to +90 mm per dimension

### Row Correspondence

```python
# Row i in position_sn_voxels_filtered.tsv corresponds to:
# - Row i in consensus_factor_loadings_volume_sn_voxels.csv
# - Voxel i in the preprocessed imaging matrix

# Verification
W_df = pd.read_csv("consensus_factor_loadings_volume_sn_voxels.csv", index_col=0)
positions = pd.read_csv("position_sn_voxels_filtered.tsv", sep='\t', header=None)

assert len(W_df) == len(positions)  # Must be equal!
```

## Preprocessing Steps That Drop Voxels

### 1. MAD-based Quality Control (Active)

- **Config**: `qc_outlier_threshold: 3.0`
- **Method**: Median Absolute Deviation outlier detection
- **Effect**: Removes ~70% of voxels (1794 → 531)
- **Purpose**: Remove unreliable measurements (artifacts, motion)

### 2. Variance Threshold (Disabled)

- **Config**: `variance_threshold: 0.0`
- **Effect**: None (disabled to preserve low-variance biomarkers)

### 3. ROI-based Selection (Disabled)

- **Config**: `roi_based_selection: false`
- **Effect**: None (disabled to preserve spatial clustering)

## References

- Original position lookups: `qMAP-PD_data/position_lookup/position_*_voxels.tsv`
- Preprocessing implementation: [data/preprocessing.py](../data/preprocessing.py)
- Config integration: [data/preprocessing_integration.py](../data/preprocessing_integration.py)
- Production config: [config_convergence.yaml](../config_convergence.yaml)
