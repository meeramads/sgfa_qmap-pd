# Filtered Position Lookup Save Implementation

## Summary

The masked position lookup vectors created during data preprocessing are now automatically saved in the data_validation experiment output directory.

## Changes Made

### 1. Updated `experiments/data_validation.py`

#### Change 1: Store experiment-specific output directory
**Location:** Lines 129-130

```python
# Store the experiment-specific output_dir for use in preprocessing
data_val_exp.base_output_dir = output_dir
```

This ensures the DataValidationExperiments instance knows where to save experiment-specific outputs.

#### Change 2: Use experiment-specific output_dir for PCA preprocessing
**Location:** Lines 283-294

```python
# Use experiment-specific output_dir if available, otherwise fall back to general output_dir
preprocessing_output_dir = getattr(self, 'base_output_dir', None) or get_output_dir(config)

X_list, preprocessing_info = apply_preprocessing_to_pipeline(
    config={"preprocessing": preprocessing_config, "data": {"data_dir": data_dir}},
    data_dir=data_dir,
    auto_select_strategy=False,
    preferred_strategy=preprocessing_config.get("strategy", "standard"),
    output_dir=preprocessing_output_dir,
)

logger.info(f"   ✅ Filtered position lookups saved to: {preprocessing_output_dir}/position_lookup_filtered")
```

#### Change 3: Use experiment-specific output_dir for position lookup creation
**Location:** Lines 329-336

```python
# Create filtered position lookups for imaging views
# This maps preprocessed voxels back to 3D brain coordinates
# Use experiment-specific output_dir if available
position_output_dir = getattr(self, 'base_output_dir', None) or get_output_dir(config)
filtered_position_paths = self._create_filtered_position_lookups(
    raw_data, preprocessed_data, data_dir, position_output_dir
)

if filtered_position_paths:
    logger.info(f"   ✅ Filtered position lookups saved to: {position_output_dir}/position_lookup_filtered")
```

#### Change 4: Use experiment-specific output_dir for strategy comparison
**Location:** Lines 1150-1158

```python
# Apply preprocessing with this strategy
# Use experiment-specific output_dir if available
strategy_output_dir = getattr(self, 'base_output_dir', None) or get_output_dir(config)
X_list_strategy, preprocessing_info = apply_preprocessing_to_pipeline(
    config=temp_config,
    data_dir=data_dir,
    auto_select_strategy=False,
    preferred_strategy=strategy_params.get("strategy", strategy_name),
    output_dir=strategy_output_dir
)
```

## How It Works

### Existing Infrastructure

The code already had the infrastructure to save filtered position lookups:

1. **`data/preprocessing.py` (NeuroImagingPreprocessor):**
   - `_filter_and_save_position_lookups()` method (lines 1229-1298)
   - Automatically filters position lookups based on feature masks
   - Saves to `output_dir/position_lookup_filtered/` if output_dir is provided

2. **`data/preprocessing_integration.py`:**
   - `apply_preprocessing_to_pipeline()` accepts `output_dir` parameter
   - Passes it to `imaging_preprocessor.fit_transform(output_dir=output_dir)`

### What Was Missing

The data_validation experiment was not passing its experiment-specific output directory to the preprocessing pipeline. Instead, it was using the general results directory from `get_output_dir(config)`.

### The Fix

Now the data_validation experiment:
1. Receives its experiment-specific `output_dir` from the framework
2. Stores it as `self.base_output_dir`
3. Passes it to all preprocessing calls
4. Filtered position lookups are saved in the experiment directory

## Output Location

Filtered position lookups are now saved to:

```
results/data_validation_<timestamp>/position_lookup_filtered/
├── position_sn_voxels_filtered.tsv
├── position_putamen_voxels_filtered.tsv
└── position_lentiform_voxels_filtered.tsv
```

Each experiment has its own filtered position lookups matching the exact preprocessing settings used.

## Benefits

1. **Reproducibility:** Each experiment has position lookups matching its exact preprocessing
2. **Isolation:** Different experiments with different preprocessing settings don't overwrite each other's position lookups
3. **Traceability:** Position lookups are saved alongside other experiment outputs
4. **Self-contained:** Everything needed to interpret the experiment results is in one directory

## Example Usage

When you run data validation:

```bash
python run_experiments.py --config config_convergence.yaml --experiments data_validation
```

You'll see log messages like:

```
✅ Filtered position lookups saved to: results/data_validation_20251020_143034/position_lookup_filtered
```

And the directory structure will be:

```
results/data_validation_20251020_143034/
├── config.yaml
├── plots/
│   ├── data_distribution_comparison.png
│   └── ...
├── position_lookup_filtered/          # ← NEW!
│   ├── position_sn_voxels_filtered.tsv
│   └── ...
├── result.json
└── summary.csv
```

## Testing

Run the test script to verify the flow:

```bash
python test_position_lookup_save.py
```

This will show the complete flow of how the output directory is passed through the system.
