# SGFA Codebase Architecture

**Last Updated**: October 21, 2025

## Overview

This document describes the architecture of the SGFA pipeline for neuroimaging biomarker discovery in Parkinson's disease. The pipeline implements sparse Bayesian factor analysis with regularized horseshoe priors using NumPyro/JAX.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                Entry Point: run_experiments.py               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Experiments (Production Pipeline)               │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │ data_validation│  │ robustness_  │  │ factor_stability│ │
│  │                │  │ testing      │  │                 │ │
│  └────────────────┘  └──────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Preprocessing Pipeline                       │
│         (preprocessing_integration.py)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. MAD-based QC (remove outlier voxels)              │  │
│  │ 2. Spatial imputation (fill missing with neighbors)  │  │
│  │ 3. Median imputation (fill remaining missing)        │  │
│  │ 4. Position tracking (voxel→brain coordinates)       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│             Model: sparse_gfa_fixed.py                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Regularized Horseshoe Priors:                        │  │
│  │ - W (loadings): τ₀, λ, slab regularization          │  │
│  │ - Z (factors): Optional regularized horseshoe        │  │
│  │ - Non-centered parameterization for better mixing    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              MCMC Sampling (NumPyro NUTS)                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ - Adaptive step size                                 │  │
│  │ - Tree depth control (max_tree_depth=13)            │  │
│  │ - Target acceptance probability (0.8)                │  │
│  │ - R-hat/ESS convergence diagnostics                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Analysis & Diagnostics                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ factor_stability.py: Cross-chain consensus           │  │
│  │ mcmc_diagnostics.py: R-hat, ESS, trace plots        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### Entry Point

**`run_experiments.py`**
- Command-line interface for experiments
- Config loading and validation
- Experiment orchestration
- Results directory management

### Configuration

**`config_convergence.yaml`**
- Model parameters (K, percW, slab_df, slab_scale)
- MCMC settings (samples, warmup, chains, tree depth)
- Preprocessing options (MAD threshold, imputation)
- Experiment-specific configs

### Data Pipeline

**`data/preprocessing.py`**
- `NeuroImagingPreprocessor`: Main preprocessing class
- MAD-based quality control
- Spatial and median imputation
- Position lookup tracking

**`data/preprocessing_integration.py`**
- `apply_preprocessing_to_pipeline()`: Main integration function
- Config merging (global overrides strategy-specific)
- Position file saving

**`data/qmap_pd.py`**
- qMAP-PD dataset loading
- Multi-view data handling (imaging + clinical)

### Model Implementation

**`models/sparse_gfa_fixed.py`**
- Regularized horseshoe priors with slab regularization
- Non-centered parameterization
- Data-dependent global scale (Piironen & Vehtari 2017)
- NumPyro MCMC implementation

**`models/factory.py`**
- Model instantiation
- Configuration validation

### Experiments

**`experiments/data_validation.py`**
- Data quality assessment
- Distribution analysis
- PCA dimensionality analysis

**`experiments/robustness_testing.py`**
- Seed reproducibility
- Perturbation analysis
- Initialization stability

**`experiments/framework.py`**
- Experiment base infrastructure
- Result standardization
- Plot saving utilities

### Analysis

**`analysis/factor_stability.py`**
- Multi-chain stability analysis (Ferreira et al. 2024)
- Cosine similarity factor matching
- Consensus loadings/scores computation
- CSV export for W, Z matrices

**`analysis/mcmc_diagnostics.py`**
- R-hat computation
- ESS calculation
- Trace plots
- Convergence diagnostics

## Data Flow

### Preprocessing Flow

```
Raw Data (1794 voxels)
    │
    ├─> MAD filtering (threshold=3.0)
    │       └─> ~531 voxels (70% removed)
    │
    ├─> Spatial imputation
    │       └─> Fill missing with neighbors
    │
    ├─> Median imputation
    │       └─> Fill remaining missing
    │
    └─> Position tracking
            └─> Save filtered position lookup
```

### Experiment Flow

```
run_experiments.py
    │
    ├─> Load config (config_convergence.yaml)
    │
    ├─> Apply preprocessing
    │       └─> apply_preprocessing_to_pipeline()
    │               ├─> Merge configs (global overrides)
    │               ├─> Create NeuroImagingPreprocessor
    │               └─> Return X_list + preprocessing_info
    │
    ├─> Run experiments
    │       ├─> data_validation
    │       ├─> robustness_testing
    │       └─> factor_stability
    │               ├─> Run 4 independent chains
    │               ├─> Compute cosine similarity
    │               ├─> Identify stable factors
    │               └─> Save consensus W/Z
    │
    └─> Save results
            ├─> Individual plots (PNG/PDF)
            ├─> CSV matrices (W, Z per chain + consensus)
            └─> JSON summaries
```

## Key Design Decisions

### 1. Config Merging Strategy

Preprocessing config uses **global-first merge**:
```python
global_preprocessing = config.get("preprocessing", {})
strategy_config = preprocessing_strategies.get(strategy, {})
merged_config = {**strategy_config, **global_preprocessing}  # Global wins
```

This allows command-line flags to override strategy-specific settings.

### 2. Non-Centered Parameterization

Model uses non-centered parameterization for better MCMC geometry:
```python
Z_raw ~ Normal(0, 1)
Z = Z_raw * sqrt(ξ)
```

### 3. Position Tracking

Cumulative mask tracking through preprocessing:
```python
cumulative_mask = np.ones(original_n_features, dtype=bool)
# Apply each filter
cumulative_mask &= mad_mask
cumulative_mask &= variance_mask
# Save filtered positions
filtered_positions = original_positions[cumulative_mask]
```

### 4. Factor Stability

Uses cosine similarity matching (Ferreira et al. 2024):
```python
similarity = 1 - cosine(W_chain1[:, k], W_chain2[:, k])
if similarity > threshold:
    match_found = True
```

## File Organization

```
sgfa_qmap-pd/
├── run_experiments.py          # Main entry point
├── config_convergence.yaml     # Production config
│
├── models/
│   ├── sparse_gfa_fixed.py     # Main model
│   └── factory.py              # Model factory
│
├── data/
│   ├── qmap_pd.py              # Data loader
│   ├── preprocessing.py        # Preprocessing
│   └── preprocessing_integration.py  # Config integration
│
├── experiments/
│   ├── framework.py            # Base infrastructure
│   ├── data_validation.py      # Data QC
│   ├── robustness_testing.py   # Reproducibility
│   └── train_sparse_gfa_fixed.py  # Standalone factor stability
│
├── analysis/
│   ├── factor_stability.py     # Stability analysis
│   └── mcmc_diagnostics.py     # Diagnostics
│
└── core/
    ├── config_utils.py         # Config helpers
    └── io_utils.py             # File I/O
```

## Output Directory Structure

### Run-Based Organization

All experiments follow a hierarchical directory structure:

```
results/
├── experiments.log                          # Global log (standalone scripts)
│
└── {experiment_name}_{params}_run_{timestamp}/   # Run directory
    ├── experiments.log                      # Run-specific log
    ├── README.md                            # Run metadata
    ├── summaries/                           # Cross-experiment summaries
    │
    ├── data_validation_{params}/            # Experiment subdirectory
    │   ├── config.yaml                      # Experiment config
    │   ├── result.json                      # Structured results
    │   ├── plots/                           # Summary plots (PNG + PDF)
    │   │   ├── mad_distribution_sn.png
    │   │   └── elbow_analysis_all_views.png
    │   └── position_lookup_filtered/        # Filtered voxel positions (TSV)
    │       └── position_sn_voxels_filtered.tsv
    │
    ├── robustness_tests_{params}/
    │   ├── config.yaml
    │   ├── result.json
    │   └── plots/
    │
    └── factor_stability_{params}/           # Main stability analysis
        ├── config.yaml
        ├── plots/                           # Summary plots
        │   ├── factor_stability_heatmap.png
        │   ├── hyperparameter_posteriors.png
        │   └── mcmc_trace_diagnostics.png
        │
        ├── individual_plots/                # Per-factor diagnostics
        │   └── trace_diagnostics/           # Individual factor traces
        │       ├── factor_00_trace.png
        │       ├── factor_01_trace.png
        │       └── ...
        │
        ├── chains/                          # Per-chain MCMC samples
        │   ├── chain_0_samples.npz
        │   ├── chain_1_samples.npz
        │   └── ...
        │
        └── stability_analysis/              # Consensus results
            ├── consensus_factor_loadings_SN.csv           # W matrix (features × K)
            ├── consensus_factor_scores.csv                # Z matrix (subjects × K)
            │
            ├── consensus_loadings_SN_factor_00.tsv        # Per-factor reconstruction (N × D)
            ├── consensus_loadings_SN_factor_01.tsv
            │
            ├── consensus_loadings_Clinical_factor_00.csv  # Clinical per-factor (N × features)
            ├── consensus_loadings_Clinical_factor_01.csv
            │
            └── factor_stability_summary.json              # Stability metrics
```

### Directory Management Patterns

#### 1. Run Directory Creation

The `ExperimentFramework` base class creates run directories:

```python
# config["experiments"]["base_output_dir"] → "results/"
self.base_output_dir = ConfigHelper.get_output_dir_safe(config)

# Creates: results/{semantic_run_name}/
# Example: results/all_rois-sn_conf-age+sex+tiv_K2_percW33_MAD3.0_run_20251021_204130/
```

#### 2. Experiment Subdirectories

**Within run_experiments.py pipeline:**
- Automatically created as subdirectories inside run directory
- Format: `{run_dir}/{experiment_name}_{params}/`

**Standalone scripts (train_sparse_gfa_fixed.py):**
- Must explicitly create experiment subdirectory inside run directory:
  ```python
  semantic_name = repro_exp._generate_semantic_experiment_name("factor_stability", exp_config)
  output_dir = repro_exp.base_output_dir / semantic_name  # Create inside run dir
  output_dir.mkdir(parents=True, exist_ok=True)
  ```

#### 3. Individual Plots Directory Resolution

**Critical pattern for ensuring plots go to the correct location:**

```python
# Priority chain in analysis/plotting code:
if hasattr(self, '_experiment_output_dir') and self._experiment_output_dir:
    experiment_output_dir = self._experiment_output_dir  # Set by run_factor_stability_analysis
elif output_dir:
    experiment_output_dir = Path(output_dir)             # From method parameter
elif hasattr(self, 'base_output_dir'):
    experiment_output_dir = Path(self.base_output_dir)   # Fallback (usually just "results/")
else:
    raise ValueError("No output directory available!")

individual_plots_dir = experiment_output_dir / "individual_plots"
```

**Common pitfall:** If `output_dir` is not passed to `run_factor_stability_analysis()`, plots will fall back to `base_output_dir` (typically just `"results/"`), creating a global `results/individual_plots/` directory instead of the experiment-specific location.

**Solution:** Always pass `output_dir` parameter:
```python
result = repro_exp.run_factor_stability_analysis(
    X_list=X_list,
    hypers=hypers,
    args=mcmc_args,
    output_dir=str(output_dir),  # ← CRITICAL: Pass experiment-specific directory
    # ...
)
```

### Logging Patterns

#### Two-Tier Logging System

1. **Global log (`results/experiments.log`)**:
   - Used by standalone scripts (train_sparse_gfa_fixed.py)
   - Accumulates across multiple runs
   - Logger name: `__main__`

2. **Run-specific log (`{run_dir}/experiments.log`)**:
   - Created by ExperimentFramework._setup_logging()
   - Scoped to single run
   - Captures all module loggers (experiments.*, analysis.*, etc.)

#### Log Handler Management

```python
# ExperimentFramework._setup_logging() adds handler to ROOT logger
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)

# All child loggers inherit, so logs from all modules go to run-specific file
# This includes: __main__, experiments.*, analysis.*, data.*, models.*
```

### Output File Naming Conventions

#### Consensus Results
- **W matrices (features × factors)**: `consensus_factor_loadings_{view_name}.csv`
  - Example: `consensus_factor_loadings_volume_sn_voxels.csv`

- **Z matrices (subjects × factors)**: `consensus_factor_scores.csv`
  - Rows: Subject IDs
  - Columns: Factor_00, Factor_01, ...

#### Per-Factor Reconstructions

**Volume data (TSV, no headers):**
- Format: `consensus_loadings_{ROI}_{factor_name}.tsv`
- Example: `consensus_loadings_SN_factor_00.tsv`
- Matrix: N subjects × D_m voxels
- Computation: `Z[:, k] @ W[:, k].T`

**Clinical data (CSV, with headers):**
- Format: `consensus_loadings_Clinical_{factor_name}.csv`
- Example: `consensus_loadings_Clinical_factor_00.csv`
- Matrix: N subjects × clinical features
- Columns: Feature names from original data

#### Position Lookup Files
- **Filtered positions**: `position_lookup_filtered/position_{roi}_filtered.tsv`
- **Format**: TSV with columns [x, y, z] for voxel brain coordinates
- **Purpose**: Map filtered feature indices back to brain space

## Dependencies

### External
- **NumPyro**: MCMC sampling
- **JAX**: Auto-differentiation
- **pandas/numpy**: Data manipulation
- **scikit-learn**: Preprocessing utilities
- **matplotlib/seaborn**: Visualization

### Internal Dependency Chain
```
core/ (foundation)
    ↓
data/ (preprocessing)
    ↓
models/ (model implementation)
    ↓
analysis/ (diagnostics)
    ↓
experiments/ (validation pipeline)
```

## Critical Implementation Details

### 1. MAD Filtering

Median Absolute Deviation for outlier detection:
```python
median = np.median(X, axis=0)
mad = np.median(np.abs(X - median), axis=0)
z_scores = np.abs(X - median) / mad
outliers = z_scores > threshold
```

### 2. Regularized Horseshoe Prior

```python
# Global scale (data-dependent)
τ₀ = (p₀/(D - p₀)) * (σ/√N)

# Local shrinkage
λ ~ Half-Cauchy(0, 1)

# Slab regularization
c² ~ InverseGamma(slab_df/2, slab_scale·slab_df/2)

# Regularized shrinkage
κ = (c² * λ²) / (c² + τ₀² * λ²)
W = W_raw * τ₀ * √κ
```

### 3. R-hat Computation

Gelman-Rubin diagnostic:
```python
# Between-chain variance
B = N * var(chain_means)

# Within-chain variance  
W = mean(chain_variances)

# R-hat
R_hat = sqrt((W + B/N) / W)
```

## Testing

```bash
# Unit tests
pytest tests/

# Specific modules
pytest tests/data/
pytest tests/models/
pytest tests/experiments/
```

## Common Operations

### Running Experiments
```bash
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --K 20 --qc-outlier-threshold 3.0
```

### Accessing Results
```python
# Load consensus loadings
W_consensus = pd.read_csv("results/.../consensus_factor_loadings.csv")

# Load per-view loadings
W_imaging = pd.read_csv("results/.../consensus_factor_loadings_volume_sn_voxels.csv")

# Load stability metrics
with open("results/.../factor_stability_summary.json") as f:
    stability = json.load(f)
```

## Performance Considerations

1. **Memory**: ~8-16GB RAM for typical datasets (86 subjects, 531 voxels)
2. **Compute**: 4-6 hours for full factor stability (4 chains × 13K samples)
3. **Storage**: ~100-500MB per experiment run (with all plots + matrices)

## References

- Ferreira et al. 2024: Factor stability methodology
- Piironen & Vehtari 2017: Regularized horseshoe priors
- qMAP-PD Study: https://qmaplab.com/qmap-pd
