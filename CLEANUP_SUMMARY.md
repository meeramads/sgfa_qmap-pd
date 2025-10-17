# Project Cleanup Summary

## Files to Remove

### 1. Test Output Files (Root Directory)
These are test artifacts that should not be in the root:
- `test_distributions_converged.png`
- `test_distributions_nonconverged.png`
- `test_hyperparameter_posteriors_converged.png`
- `test_hyperparameter_posteriors_nonconverged.png`
- `test_hyperparameter_traces_converged.png`
- `test_hyperparameter_traces_nonconverged.png`
- `test_trace_converged.png`
- `test_trace_nonconverged.png`
- `test_trace_plots.py`
- `test_variance_profile.py`
- `test_config_quick.py`
- `test_synthetic_config.py`
- `debug_experiments.py`

### 2. Obsolete Config Files
These settings are now attainable with command-line flags:
- `config_K2_no_sparsity.yaml` → Use main config with `--K 2 --no-sparsity`
- `config_K20_no_sparsity.yaml` → Use main config with `--K 20 --no-sparsity`
- `config_K50_no_sparsity.yaml` → Use main config with `--K 50 --no-sparsity`
- `config_K10_MAD5.0.yaml` → Use main config with `--K 10 --mad-threshold 5.0`
- `config_K10_full_voxels.yaml` → Use main config with `--K 10 --full-voxels`
- `config_K15_full_voxels.yaml` → Use main config with `--K 15 --full-voxels`
- `config_K20_MAD3.0.yaml` → Use main config with `--K 20 --mad-threshold 3.0`
- `config_K50_full_voxels.yaml` → Use main config with `--K 50 --full-voxels`
- `config_debug.yaml` → Use main config with `--debug` flag

### 3. Recommended Configs to Keep
- `config_K5_diagnostic.yaml` - Standard diagnostic config
- `config_K50.yaml` - Standard K=50 config
- `config_synthetic.yaml` - Synthetic data testing
- Main experiment config (if exists)

### 4. Directories to Clean
- `.mypy_cache/` - Type checking cache
- `__pycache__/` - Python bytecode cache (all directories)
- `test_outputs/` - Old test artifacts
- `logs/` - Old log files (>7 days)

### 5. System Files
- `.DS_Store` files (macOS metadata)

## Cleanup Actions

### Safe Cleanup (Recommended)
Moves old configs to `archived_configs/` for potential restore:
```bash
./cleanup_project.sh
```

### Aggressive Cleanup (After verifying you don't need archived files)
```bash
rm -rf archived_configs/ archived_test_outputs/
```

## Usage After Cleanup

Instead of multiple config files, use flags with your main config:

```bash
# Example: Run with K=20, no sparsity
python run_experiments.py --config config_main.yaml --K 20 --no-sparsity

# Example: Run with K=50, full voxels
python run_experiments.py --config config_main.yaml --K 50 --full-voxels

# Example: Debug mode with K=5
python run_experiments.py --config config_main.yaml --K 5 --debug --mad-threshold 5.0
```

## What Gets Preserved

- All source code (`*.py` files in proper directories)
- Active configs (K5_diagnostic, K50, synthetic)
- Test suite (`tests/` directory)
- Documentation (`*.md`, `docs/`)
- Results (`results/` directory)
- Data (`data/`, `qMAP-PD_data/`)
- Git history and configuration
