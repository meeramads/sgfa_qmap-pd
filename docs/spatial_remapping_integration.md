# Spatial Remapping Integration Guide

## Overview

Spatial remapping is now integrated into the experiment pipeline, allowing automatic conversion of factor loadings to 3D brain maps after feature selection.

## Configuration

### Enable in config.yaml

```yaml
experiments:
  # Spatial Remapping Configuration
  enable_spatial_remapping: true          # Enable automatic brain map generation
  save_brain_maps: true                   # Save as CSV files
  create_brain_map_plots: true            # Generate 3D visualizations
  brain_maps_output_dir: "brain_maps"     # Output subdirectory
```

### Preprocessing Configuration

Feature selection now applies to imaging data while preserving spatial mapping:

```yaml
preprocessing:
  strategy: "full_preprocessing_all_views"  # Feature selection on ALL views
  feature_selection_method: "variance"      # Removes uninformative voxels
  variance_threshold: 0.02                  # Keep voxels with >2% variance

  # Spatial remapping is preserved through selected voxel indices
```

## How It Works

### 1. Feature Selection (Automatic)

```
Original imaging data: 1,794 voxels
   ↓ (variance-based selection)
Selected informative voxels: ~600-1,000 voxels
   ↓ (indices stored: [0, 5, 12, 23, ...])
```

### 2. SGFA Training

```
Factor loadings (W): Shape (n_selected_voxels, K)
   - Each row corresponds to a selected voxel
   - Indices track which original voxel
```

### 3. Spatial Remapping (Automatic if enabled)

```
Position lookup: All 1,794 voxel positions
   ↓ (index by selected voxels)
Selected positions: ~600-1,000 positions
   ↓ (combine with factor loadings)
Brain maps: [x, y, z, loading]
```

## Usage in Experiments

### Automatic Integration (Recommended)

When `enable_spatial_remapping: true` in config, brain maps are saved automatically:

```bash
python run_experiments.py --select-rois volume_sn_voxels.tsv

# Output structure:
results/
└── experiment_run_20251013_123456/
    ├── sgfa_results/
    │   └── ...
    └── brain_maps/              # ← Automatically created
        ├── sn/
        │   ├── sn_factor0.csv
        │   ├── sn_factor1.csv
        │   └── ...
        └── putamen/
            ├── putamen_factor0.csv
            └── ...
```

### Manual Usage in Experiments

For custom experiments, use the integration helper:

```python
from experiments.spatial_remapping_integration import integrate_spatial_remapping_to_experiment

# At end of your experiment
def run_my_experiment(config):
    # ... run SGFA ...

    result = ExperimentResult(
        experiment_id="my_experiment",
        config=config,
        model_results=sgfa_results,
        ...
    )

    # Automatically save brain maps if enabled
    integrate_spatial_remapping_to_experiment(
        experiment_result=result,
        config=config,
        data_dir="qMAP-PD_data"
    )

    return result
```

### Direct Usage (Standalone)

For manual brain map generation:

```python
from experiments.spatial_remapping_integration import save_brain_maps_from_results

# After running SGFA
results = run_sgfa(...)

# Save all brain maps
brain_map_paths = save_brain_maps_from_results(
    results=results,
    output_dir=Path('my_output/brain_maps'),
    data_dir='qMAP-PD_data',
    save_csv=True,
    create_plots=True
)

# Returns: {view_name: {factor_k: Path}}
# e.g., {'volume_sn_voxels': {0: Path('sn/sn_factor0.csv'), ...}}
```

## Brain Map Format

CSV files contain:

| Column | Description |
|--------|-------------|
| x, y, z (or voxel_index) | Spatial coordinates |
| loading | Factor loading value |

Example `sn_factor0.csv`:
```csv
x,y,z,loading
45,-12,8,0.234
46,-12,8,0.198
44,-13,7,0.187
...
```

## Command-Line Flags

All experiments support spatial remapping when properly configured:

```bash
# Standard experiment with automatic brain maps
python run_experiments.py --select-rois volume_sn_voxels.tsv

# With custom confounds
python run_experiments.py \
    --select-rois volume_sn_voxels.tsv \
    --regress-confounds age sex tiv \
    --drop-confounds-from-clinical

# Results include brain maps automatically if enabled in config
```

## Integration Status

### ✅ Fully Integrated
- Configuration system (`config.yaml`)
- Helper utilities (`experiments/spatial_remapping_integration.py`)
- Core functions (`data/preprocessing.py`)
- Documentation (`docs/spatial_remapping_guide.md`)
- Example script (`examples/spatial_remapping_example.py`)

### ⚠️ Partial Integration (Manual Usage)
- Data validation experiment - use `integrate_spatial_remapping_to_experiment()`
- Model comparison experiment - use `save_brain_maps_from_results()`
- Other experiments - add helper function calls

### 🔄 Future Integration
- Automatic integration in all experiment types
- Brain map visualization dashboard
- Interactive 3D viewer

## Example Workflow

### 1. Configure

Edit `config.yaml`:
```yaml
experiments:
  enable_spatial_remapping: true
  save_brain_maps: true

preprocessing:
  feature_selection_method: "variance"
  variance_threshold: 0.02
```

### 2. Run Experiment

```bash
python run_experiments.py --select-rois volume_sn_voxels.tsv
```

### 3. Find Brain Maps

```bash
ls results/latest_run/brain_maps/sn/
# sn_factor0.csv
# sn_factor1.csv
# sn_factor2.csv
# ...
```

### 4. Analyze

```python
import pandas as pd

# Load brain map
brain_map = pd.read_csv('results/.../brain_maps/sn/sn_factor0.csv')

# Find peak voxel
peak = brain_map.loc[brain_map['loading'].abs().idxmax()]
print(f"Peak at ({peak['x']}, {peak['y']}, {peak['z']}): {peak['loading']:.3f}")

# Visualize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    brain_map['x'],
    brain_map['y'],
    brain_map['z'],
    c=brain_map['loading'].abs(),
    cmap='hot'
)
plt.colorbar(scatter)
plt.show()
```

## Troubleshooting

### Issue: No brain maps generated

**Check:**
1. Is `enable_spatial_remapping: true` in config?
2. Does `qMAP-PD_data/position_lookup/` directory exist?
3. Are position lookup files present (e.g., `position_sn_voxels.tsv`)?

### Issue: Dimension mismatch error

**Solution:** Ensure feature selection is enabled and working:
```yaml
preprocessing:
  enable_advanced_preprocessing: true
  feature_selection_method: "variance"
```

### Issue: Empty brain maps

**Check:** Feature selection may be too aggressive:
```yaml
preprocessing:
  variance_threshold: 0.01  # Lower threshold (less aggressive)
```

## Additional Resources

- [Spatial Remapping Guide](spatial_remapping_guide.md) - Detailed technical documentation
- [Example Script](../examples/spatial_remapping_example.py) - Standalone usage example
- [Preprocessing Guide](preprocessing.md) - Feature selection details

## Summary

✅ **Spatial remapping is integrated** into the experiment pipeline
✅ **Automatic brain map generation** when enabled in config
✅ **Preserved through feature selection** via selected voxel indices
✅ **Easy to use** - just enable in config.yaml

For questions, see: [GitHub Issues](https://github.com/your-repo/issues)
