# Spatial Remapping Guide: From Factor Loadings to Brain Maps

## Overview

After feature selection reduces imaging voxels from ~1,794 to ~600-1,000 informative voxels, you can remap factor loadings back to 3D brain space using position lookups. This guide explains how spatial mapping is preserved and how to use the remapping utilities.

## How Spatial Mapping is Preserved

### 1. Position Lookup Files

Each ROI has a position lookup file:
```
qMAP-PD_data/position_lookup/
├── position_sn_voxels.tsv        # 1794 rows (one per voxel)
├── position_putamen_voxels.tsv   # ~4600 rows
├── position_lentiform_voxels.tsv # ~8200 rows
└── position_bg-all_voxels.tsv    # ~15000 rows
```

Each file contains spatial coordinates or voxel indices for all voxels in the ROI.

### 2. Feature Selection Preserves Indices

When feature selection reduces voxels, the **indices of selected voxels** are stored:

```python
# Example: variance-based selection on SN (1794 voxels)
variances = np.var(X_sn, axis=0)  # Shape: (1794,)
top_indices = np.argsort(variances)[::-1][:687]  # Select top 687

# Stored in preprocessor
preprocessor.selected_features_["volume_sn_voxels_variance_indices"] = top_indices
# e.g., array([0, 5, 12, 23, 45, 67, ..., 1789]) - 687 indices
```

### 3. Spatial Remapping

Factor loadings align with selected voxels:

```python
# Factor loadings after SGFA
W_sn = results['W'][sn_view_idx]  # Shape: (687, K) - K factors

# Position lookup for selected voxels
positions_all = load_positions("sn")  # Shape: (1794, 3) - [x, y, z]
positions_selected = positions_all[top_indices]  # Shape: (687, 3)

# Now: W_sn[i, k] corresponds to positions_selected[i, :]
# Each loading value has a spatial coordinate!
```

## Quick Start: Using Helper Functions

### Import

```python
from data.preprocessing import (
    get_selected_voxel_positions,
    create_brain_map_from_factors,
    remap_all_factors_to_brain
)
```

### Method 1: Single Factor, Single ROI

```python
# After running SGFA
results = run_sgfa(X_list, hypers, args)

# Get positions for selected voxels
positions = get_selected_voxel_positions(
    preprocessor=results['preprocessor'],
    view_name='volume_sn_voxels',
    data_dir='qMAP-PD_data'
)

# Create brain map for factor 0
W_sn = results['W'][sn_view_idx]  # Shape: (687, K)
brain_map_factor0 = create_brain_map_from_factors(
    factor_loadings=W_sn,
    positions=positions,
    factor_index=0
)

# brain_map_factor0 is a DataFrame:
#      x     y     z    loading
# 0   45   -12    8    0.234
# 1   46   -12    8    0.198
# 2   44   -13    7    0.187
# ...

# Save or visualize
brain_map_factor0.to_csv('sn_factor0_map.csv', index=False)
```

### Method 2: All Factors, All ROIs (Recommended)

```python
# Remap all factors for all imaging views at once
all_brain_maps = remap_all_factors_to_brain(
    W_list=results['W'],
    view_names=results['view_names'],
    preprocessor=results['preprocessor'],
    data_dir='qMAP-PD_data'
)

# Access brain maps
# Structure: all_brain_maps[view_name][factor_k] = brain_map_df

# Get SN factor 0
sn_factor_0 = all_brain_maps['volume_sn_voxels'][0]

# Get Putamen factor 1
putamen_factor_1 = all_brain_maps['volume_putamen_voxels'][1]

# Save all brain maps
for view_name, factor_maps in all_brain_maps.items():
    roi = view_name.replace('volume_', '').replace('_voxels', '')
    for k, brain_map in factor_maps.items():
        brain_map.to_csv(f'brain_maps/{roi}_factor{k}.csv', index=False)
```

## Detailed Example: Manual Remapping

If you want full control:

```python
import numpy as np
import pandas as pd
from pathlib import Path

# 1. Load SGFA results
results = load_sgfa_results('results/experiment_xyz/')

# 2. Load position lookup for SN
positions_all = pd.read_csv(
    'qMAP-PD_data/position_lookup/position_sn_voxels.tsv',
    sep='\t',
    header=None,
    names=['voxel_index']  # or ['x', 'y', 'z'] if coordinates
)
# Shape: (1794, 1) or (1794, 3)

# 3. Get selected voxel indices from preprocessor
preprocessor = results['preprocessor']
selected_indices = preprocessor.selected_features_['volume_sn_voxels_variance_indices']
# Shape: (687,) - indices into the 1794 voxels

# 4. Get positions for selected voxels only
positions_selected = positions_all.iloc[selected_indices].reset_index(drop=True)
# Shape: (687, 1) or (687, 3)

# 5. Get factor loadings for SN
sn_view_idx = results['view_names'].index('volume_sn_voxels')
W_sn = results['W'][sn_view_idx]
# Shape: (687, K) where K is number of factors

# 6. Create brain map for each factor
K = W_sn.shape[1]
brain_maps = []

for k in range(K):
    factor_k_loadings = W_sn[:, k]  # Shape: (687,)

    # Combine positions with loadings
    brain_map_k = positions_selected.copy()
    brain_map_k['factor'] = k
    brain_map_k['loading'] = factor_k_loadings
    brain_map_k['abs_loading'] = np.abs(factor_k_loadings)

    # Sort by loading magnitude
    brain_map_k = brain_map_k.sort_values('abs_loading', ascending=False)

    brain_maps.append(brain_map_k)

    # Save
    brain_map_k.to_csv(f'sn_factor{k}_map.csv', index=False)

    # Print top 5 voxels for this factor
    print(f"\nFactor {k} - Top 5 voxels:")
    print(brain_map_k.head())
```

## Visualization Example

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_brain_map_3d(brain_map, factor_name='Factor 0'):
    """Plot 3D brain map colored by loading magnitude."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates and loadings
    x = brain_map['x'].values
    y = brain_map['y'].values
    z = brain_map['z'].values
    loadings = brain_map['loading'].values

    # Color by loading magnitude
    colors = np.abs(loadings)

    # Plot
    scatter = ax.scatter(x, y, z, c=colors, cmap='hot', s=50, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{factor_name} - Spatial Distribution')

    plt.colorbar(scatter, label='|Loading|')
    plt.tight_layout()
    plt.savefig(f'{factor_name}_3d.png', dpi=300)
    plt.close()

# Use it
brain_map = all_brain_maps['volume_sn_voxels'][0]
plot_brain_map_3d(brain_map, 'SN Factor 0')
```

## Common Issues and Solutions

### Issue 1: No Position Lookup Found

```
WARNING: No position lookup found for sn
```

**Solution:** Check that position lookup files exist:
```bash
ls qMAP-PD_data/position_lookup/position_*_voxels.tsv
```

### Issue 2: Dimension Mismatch

```
ValueError: Dimension mismatch: 687 loadings vs 1794 positions
```

**Cause:** Using full position lookup instead of selected positions.

**Solution:** Use `get_selected_voxel_positions()` to automatically index:
```python
# Wrong
positions = pd.read_csv('position_sn_voxels.tsv')  # All 1794

# Right
positions = get_selected_voxel_positions(
    preprocessor, 'volume_sn_voxels', 'qMAP-PD_data'
)  # Only selected 687
```

### Issue 3: No Selected Indices Found

```
WARNING: No selected feature indices found for volume_sn_voxels
```

**Cause:** Feature selection wasn't enabled or preprocessor wasn't saved.

**Solution:** Enable feature selection:
```yaml
preprocessing:
  enable_advanced_preprocessing: true
  feature_selection_method: "variance"
  variance_threshold: 0.01
```

## Advanced: Computing with Brain Maps

### Example 1: Find Peak Voxel for Each Factor

```python
brain_map = all_brain_maps['volume_sn_voxels'][0]

# Get voxel with highest absolute loading
peak_idx = brain_map['loading'].abs().idxmax()
peak_voxel = brain_map.iloc[peak_idx]

print(f"Peak voxel for Factor 0:")
print(f"  Position: ({peak_voxel['x']}, {peak_voxel['y']}, {peak_voxel['z']})")
print(f"  Loading: {peak_voxel['loading']:.3f}")
```

### Example 2: Compute Center of Mass

```python
def compute_center_of_mass(brain_map):
    """Compute loading-weighted center of mass."""
    weights = np.abs(brain_map['loading'].values)
    weights = weights / weights.sum()  # Normalize

    com_x = np.average(brain_map['x'].values, weights=weights)
    com_y = np.average(brain_map['y'].values, weights=weights)
    com_z = np.average(brain_map['z'].values, weights=weights)

    return (com_x, com_y, com_z)

brain_map = all_brain_maps['volume_sn_voxels'][0]
com = compute_center_of_mass(brain_map)
print(f"Center of mass: ({com[0]:.1f}, {com[1]:.1f}, {com[2]:.1f})")
```

### Example 3: Spatial Correlation Between Factors

```python
def spatial_correlation(brain_map1, brain_map2):
    """Compute spatial correlation between two factors."""
    # Both must have same voxels (same view)
    assert len(brain_map1) == len(brain_map2)

    return np.corrcoef(
        brain_map1['loading'].values,
        brain_map2['loading'].values
    )[0, 1]

sn_factor_0 = all_brain_maps['volume_sn_voxels'][0]
sn_factor_1 = all_brain_maps['volume_sn_voxels'][1]

corr = spatial_correlation(sn_factor_0, sn_factor_1)
print(f"Spatial correlation: {corr:.3f}")
# Expected: ~0 (factors should be orthogonal)
```

## Summary

✅ **Spatial mapping is preserved** through selected feature indices
✅ **Helper functions** make remapping easy (`get_selected_voxel_positions`, etc.)
✅ **All factor loadings** can be mapped back to 3D brain coordinates
✅ **Dimension reduction** doesn't lose spatial information, just focuses on informative voxels

For questions or issues, see: [GitHub Issues](https://github.com/your-repo/issues)
