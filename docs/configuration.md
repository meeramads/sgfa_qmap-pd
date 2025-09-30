# Configuration Documentation

This document provides comprehensive documentation for configuring the SGFA qMAP-PD analysis pipeline.

## Overview

The SGFA pipeline uses a hierarchical YAML configuration system with automatic validation and defaults. Configuration files define data sources, model parameters, preprocessing options, and execution settings.

## Configuration Structure

```yaml
data:               # Data loading configuration (REQUIRED)
experiments:        # Experiment execution settings (REQUIRED)
model:             # Model parameters (REQUIRED)
preprocessing:     # Data preprocessing options (OPTIONAL)
cross_validation:  # Cross-validation settings (OPTIONAL)
monitoring:        # Logging and checkpointing (OPTIONAL)
system:           # Hardware and performance settings (OPTIONAL)
```

## Required Configuration Sections

### 1. Data Configuration (`data`)

**Purpose**: Specifies data sources and loading parameters.

```yaml
data:
  data_dir: "./qMAP-PD_data"                    # REQUIRED: Path to data directory
  clinical_file: "data_clinical/clinical.tsv"   # OPTIONAL: Clinical data file
  volume_dir: "volume_matrices"                  # OPTIONAL: Volume matrices directory
  imaging_as_single_view: true                   # OPTIONAL: Concatenate imaging data (default: true)
```

**Required Fields**:

- `data_dir`: Path to root data directory (must exist)

**Optional Fields**:

- `clinical_file`: Relative path to clinical data TSV file
- `volume_dir`: Relative path to volume matrices directory
- `imaging_as_single_view`: Whether to treat all imaging data as single view

### 2. Experiments Configuration (`experiments`)

**Purpose**: Controls experiment execution and output handling.

```yaml
experiments:
  base_output_dir: "./results"        # REQUIRED: Output directory
  save_intermediate: true             # OPTIONAL: Save intermediate results (default: true)
  generate_plots: true                # OPTIONAL: Generate visualization plots (default: true)
  max_parallel_jobs: 1               # OPTIONAL: Maximum parallel experiments (default: 1)
```

**Required Fields**:

- `base_output_dir`: Directory for saving all experiment results

**Optional Fields**:

- `save_intermediate`: Save intermediate analysis results
- `generate_plots`: Generate visualization plots
- `max_parallel_jobs`: Number of experiments to run in parallel (1-16)

### 3. Model Configuration (`model`)

**Purpose**: Defines model type and hyperparameters.

```yaml
model:
  model_type: "sparse_gfa"           # REQUIRED: Model type
  K: 5                               # REQUIRED: Number of factors
  num_samples: 1000                  # OPTIONAL: MCMC samples (default: 1000)
  num_warmup: 500                    # OPTIONAL: MCMC warmup steps (default: 500)
  num_chains: 2                      # OPTIONAL: MCMC chains (default: 2)
  sparsity_lambda: 0.1              # REQUIRED for sparse_gfa: Sparsity penalty
  group_lambda: 0.1                  # OPTIONAL: Group sparsity penalty
  random_seed: 42                    # OPTIONAL: Random seed for reproducibility
```

**Required Fields**:

- `model_type`: Model type (`"sparse_gfa"` or `"standard_gfa"`)
- `K`: Number of latent factors (1-50 recommended)

**Conditionally Required Fields**:

- `sparsity_lambda`: Required for `sparse_gfa` model type (≥0, typically 0.01-1.0)

**Optional Fields**:

- `num_samples`: Number of MCMC samples (10-50000, default: 1000)
- `num_warmup`: Number of warmup steps (10 to num_samples-1, default: 500)
- `num_chains`: Number of MCMC chains (1-8, default: 2)
- `group_lambda`: Group sparsity penalty (≥0, typically 0.01-1.0)
- `random_seed`: Random seed for reproducibility (≥0)

## Optional Configuration Sections

### 4. Preprocessing Configuration (`preprocessing`)

**Purpose**: Controls data preprocessing and feature selection.

```yaml
preprocessing:
  strategy: "standard"                        # Strategy name (default: "standard")
  enable_advanced_preprocessing: true         # Enable advanced preprocessing (default: true)
  enable_spatial_processing: true            # Enable spatial processing (default: true)
  imputation_strategy: "median"              # Missing data imputation (default: "median")
  feature_selection_method: "variance"       # Feature selection method (default: "variance")
  variance_threshold: 0.01                   # Variance threshold (default: 0.01)
  missing_threshold: 0.5                     # Missing data threshold (default: 0.5)
```

**All fields are optional with sensible defaults.**

**Field Options**:

- `strategy`: `"minimal"`, `"standard"`, `"advanced"`, `"clinical_focused"`
- `imputation_strategy`: `"mean"`, `"median"`, `"mode"`, `"drop"`
- `feature_selection_method`: `"variance"`, `"correlation"`, `"mutual_info"`, `"none"`
- `variance_threshold`: 0.0-1.0 (remove features with variance below threshold)
- `missing_threshold`: 0.0-1.0 (remove features with >threshold fraction missing)

### 5. Cross-Validation Configuration (`cross_validation`)

**Purpose**: Configures cross-validation for model evaluation.

```yaml
cross_validation:
  n_folds: 5                        # Number of CV folds (default: 5)
  n_repeats: 1                      # Number of CV repeats (default: 1)
  stratified: true                  # Use stratified CV (default: true)
  group_aware: false                # Use group-aware CV (default: false)
  random_seed: 42                   # Random seed for CV splits
```

**All fields are optional with sensible defaults.**

**Field Constraints**:

- `n_folds`: 2-20 (recommended: 5-10)
- `n_repeats`: 1-10 (recommended: 1-3)
- `random_seed`: ≥0

### 6. Monitoring Configuration (`monitoring`)

**Purpose**: Controls logging, checkpointing, and progress monitoring.

```yaml
monitoring:
  checkpoint_dir: "./results/checkpoints"     # Checkpoint directory (default: "./results/checkpoints")
  log_level: "INFO"                          # Logging level (default: "INFO")
  save_checkpoints: true                     # Save model checkpoints (default: true)
  checkpoint_interval: 100                   # Checkpoint every N samples (default: 100)
```

**All fields are optional with sensible defaults.**

**Field Options**:

- `log_level`: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`
- `checkpoint_interval`: ≥1 (how often to save checkpoints)

### 7. System Configuration (`system`)

**Purpose**: Hardware and performance optimization settings.

```yaml
system:
  use_gpu: true                      # Use GPU acceleration (default: true)
  memory_limit_gb: 16.0             # Memory limit in GB (default: auto-detect)
  n_cpu_cores: 8                    # Number of CPU cores to use (default: auto-detect)
```

**All fields are optional with automatic detection.**

**Field Constraints**:

- `memory_limit_gb`: >0 (reasonable values: 2-1000)
- `n_cpu_cores`: ≥1 (should not exceed system cores)

## Configuration Examples

### Minimal Configuration

```yaml
# Minimal working configuration
data:
  data_dir: "./qMAP-PD_data"

experiments:
  base_output_dir: "./results"

model:
  model_type: "sparse_gfa"
  K: 5
  sparsity_lambda: 0.1
```

### Full Research Configuration

```yaml
# Complete research-grade configuration
data:
  data_dir: "./qMAP-PD_data"
  clinical_file: "data_clinical/clinical.tsv"
  volume_dir: "volume_matrices"
  imaging_as_single_view: true

experiments:
  base_output_dir: "./results"
  save_intermediate: true
  generate_plots: true
  max_parallel_jobs: 2

model:
  model_type: "sparse_gfa"
  K: 8
  num_samples: 2000
  num_warmup: 1000
  num_chains: 4
  sparsity_lambda: 0.05
  group_lambda: 0.1
  random_seed: 42

preprocessing:
  strategy: "advanced"
  enable_advanced_preprocessing: true
  enable_spatial_processing: true
  imputation_strategy: "median"
  feature_selection_method: "variance"
  variance_threshold: 0.005

cross_validation:
  n_folds: 10
  n_repeats: 3
  stratified: true
  random_seed: 42

monitoring:
  log_level: "INFO"
  save_checkpoints: true
  checkpoint_interval: 200

system:
  use_gpu: true
  memory_limit_gb: 32.0
```

### Performance-Optimized Configuration

```yaml
# High-performance configuration for large datasets
data:
  data_dir: "./qMAP-PD_data"

experiments:
  base_output_dir: "./results"
  max_parallel_jobs: 4

model:
  model_type: "sparse_gfa"
  K: 6
  num_samples: 500      # Reduced for speed
  num_warmup: 250
  num_chains: 2
  sparsity_lambda: 0.1

preprocessing:
  strategy: "minimal"   # Faster preprocessing
  enable_advanced_preprocessing: false

system:
  use_gpu: true
  memory_limit_gb: 64.0
  n_cpu_cores: 16
```

## Configuration Validation

The pipeline automatically validates configurations and provides helpful error messages:

### Automatic Fixes

The system automatically applies these fixes:

1. **Missing Required Fields**: Adds sensible defaults where possible
2. **Relative Paths**: Converts to absolute paths
3. **Missing Sparsity**: Adds `sparsity_lambda=0.1` for `sparse_gfa` models
4. **Missing Sections**: Adds optional sections with default values

### Validation Errors

Common validation errors and solutions:

| Error | Solution |
|-------|----------|
| `data_dir does not exist` | Create directory or fix path |
| `K must be >= 1` | Set positive number of factors |
| `num_warmup must be < num_samples` | Reduce warmup or increase samples |
| `sparse_gfa requires sparsity_lambda` | Add sparsity parameter |
| `memory requirement exceeds limit` | Reduce K, samples, or increase memory limit |

### Performance Warnings

The system warns about potential performance issues:

- Large `num_samples` (>10000) may be slow
- Large `K` (>20) may cause overfitting
- GPU requested but not available
- Memory requirements exceed system limits

## Environment-Specific Configurations

### Development Configuration

```yaml
# Quick testing and development
model:
  K: 3
  num_samples: 100
  num_warmup: 50
  num_chains: 1

monitoring:
  log_level: "DEBUG"
```

### Production Configuration

```yaml
# Robust production settings
model:
  num_samples: 5000
  num_warmup: 2500
  num_chains: 4

cross_validation:
  n_folds: 10
  n_repeats: 5

monitoring:
  log_level: "INFO"
  save_checkpoints: true
```

### High-Memory Configuration

```yaml
# For large datasets requiring more memory
model:
  K: 15
  num_samples: 10000

system:
  memory_limit_gb: 128.0
  use_gpu: true

preprocessing:
  strategy: "advanced"
```

## Best Practices

### 1. Model Selection

- **Start with `K=5-8`** for initial exploration
- **Use `sparse_gfa`** for most neuroimaging applications
- **Set `sparsity_lambda=0.01-0.5`** for mild to moderate sparsity

### 2. MCMC Settings

- **Development**: 100-500 samples, 1-2 chains
- **Research**: 1000-5000 samples, 2-4 chains
- **High-quality results**: 5000+ samples, 4 chains

### 3. Cross-Validation

- **Quick validation**: 5-fold, 1 repeat
- **Robust validation**: 10-fold, 3-5 repeats
- **Use stratified CV** for imbalanced datasets

### 4. Hardware Optimization

- **Enable GPU** when available (10-100x speedup)
- **Set memory limits** to prevent system overload
- **Use parallel jobs** for independent experiments

### 5. Reproducibility

- **Always set `random_seed`** for reproducible results
- **Save intermediate results** for debugging
- **Use version control** for configuration files

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `K`, `num_samples`, or enable GPU
2. **Slow Convergence**: Increase `num_warmup` or check data quality
3. **Path Errors**: Use absolute paths or ensure directories exist
4. **GPU Not Found**: Install CUDA or set `use_gpu: false`

### Configuration Testing

Test your configuration before long runs:

```bash
# Validate configuration
python -c "from core.config_utils import validate_configuration; import yaml; validate_configuration(yaml.safe_load(open('config.yaml')))"

# Quick test run
python run_experiments.py --config config.yaml --experiments data_validation
```

## Advanced Configuration

### Experiment-Specific Settings

Each experiment can have specific settings:

```yaml
# Method comparison specific settings
method_comparison:
  models:
    - name: "sparse_gfa"
      sparsity_lambda: [0.01, 0.1, 0.5]
    - name: "standard_gfa"
  cross_validation:
    n_folds: 5

# Sensitivity analysis specific settings
sensitivity_analysis:
  parameter_ranges:
    K: [3, 5, 8, 10]
    sparsity_lambda: [0.01, 0.1, 0.5, 1.0]
```

### Shared Data Pipeline

Enable data sharing between experiments:

```yaml
experiments:
  enable_data_sharing: true  # Share preprocessed data between experiments
  shared_preprocessing: true # Use same preprocessing for all experiments
```

## Output Folder Structure

The pipeline creates an organized output structure to separate different types of analysis:

```text
results/
├── complete_run_YYYYMMDD_HHMMSS/
│   ├── plots/
│   │   ├── factors/                           # General factor analysis plots
│   │   │   ├── enhanced_loading_distributions.png  # Comprehensive loading analysis
│   │   │   ├── region_wise_analysis.png            # Multi-region comparison
│   │   │   ├── factor_loadings.png                 # Basic loading heatmaps
│   │   │   └── factor_scores.png                   # Factor score distributions
│   │   ├── brain_analysis/                    # Brain-specific visualizations
│   │   │   ├── brain_factor_loadings.png           # Interpretable brain plots
│   │   │   ├── [region_name]_brain.png             # Individual region analysis
│   │   │   └── spatial_coherence.png               # Spatial analysis (if enabled)
│   │   └── convergence/                       # Model convergence diagnostics
│   ├── summaries/
│   │   ├── hyperparameter_comparison_summary.json  # Performance-ranked parameters
│   │   ├── sensitivity_analysis_summary.json       # Parameter sensitivity analysis
│   │   └── optimal_parameter_recommendations.json  # Clinical deployment guidance
│   └── matrices/
│       ├── [variant]_factor_scores.csv             # Factor scores (subjects × factors)
│       ├── [variant]_factor_loadings.csv           # Factor loadings (features × factors)
│       └── [variant]_hyperparameter_performance.json  # Enhanced hyperparameter analysis
```

### Folder Organization Changes

- **General analyses** → `plots/factors/` (applies to all data types)
- **Brain-specific analyses** → `plots/brain_analysis/` (requires brain regions)
- **Enhanced summaries** → `summaries/` (scientifically interpretable results)
- **Raw matrices** → `matrices/` (numerical results with proper labeling)

This documentation provides complete guidance for configuring the SGFA qMAP-PD analysis pipeline. For additional help, consult the code documentation or example configurations in the `examples/` directory.
