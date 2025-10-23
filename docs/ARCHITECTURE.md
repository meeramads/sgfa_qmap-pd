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
│         Experiments (Default Pipeline: --experiments all)   │
│  ┌────────────────┐                    ┌─────────────────┐ │
│  │ data_validation│ ──── shared ────▶ │ factor_stability│ │
│  │                │      data          │                 │ │
│  └────────────────┘                    └─────────────────┘ │
│                                                             │
│  Note: robustness_testing available but not in default     │
│  pipeline (only 150 iterations, smoke test only)           │
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
│  │ - Tree depth control (max_tree_depth=10)            │  │
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

- Smoke test (150 iterations: 50 warmup + 100 samples)
- Not part of default pipeline
- Available for explicit use: `--experiments robustness_testing`
- Does not check convergence (just verifies code runs)

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

### Experiment Flow (Default Pipeline)

```
run_experiments.py --experiments all
    │
    ├─> Load config (config_convergence.yaml)
    │
    ├─> Initialize pipeline_context (shared data mode)
    │       └─> X_list: None, preprocessing_info: None
    │
    ├─> Run data_validation (stage 1/2)
    │       ├─> Apply preprocessing
    │       │       └─> apply_preprocessing_to_pipeline()
    │       │               ├─> Merge configs (global overrides)
    │       │               ├─> Create NeuroImagingPreprocessor
    │       │               └─> Return X_list + preprocessing_info
    │       │
    │       ├─> Store in pipeline_context
    │       │       └─> pipeline_context["X_list"] = X_list
    │       │           pipeline_context["preprocessing_info"] = preprocessing_info
    │       │
    │       └─> Save validation plots and metrics
    │
    ├─> Run factor_stability (stage 2/2)
    │       ├─> Check pipeline_context for shared data
    │       │       └─> if X_list available: use shared data ✓
    │       │           else: load fresh data
    │       │
    │       ├─> Run 4 independent MCMC chains (parallel)
    │       │       └─> 10,000 samples + 3,000 warmup per chain
    │       │
    │       ├─> Factor stability analysis
    │       │       ├─> Align factors across chains (sign/permutation)
    │       │       ├─> Compute aligned R-hat
    │       │       ├─> Cosine similarity matching (threshold=0.8)
    │       │       └─> Identify consensus factors (≥50% chain agreement)
    │       │
    │       └─> Save consensus W/Z
    │               ├─> Per-chain samples (NPZ)
    │               ├─> Consensus matrices (CSV)
    │               └─> Stability plots (PNG/PDF)
    │
    └─> Complete
            └─> All results in unified run directory
```

**Key Feature: Shared Data Flow**

The default pipeline (`--experiments all`) efficiently reuses preprocessed data:

1. **data_validation** preprocesses data once → stores in `pipeline_context`
2. **factor_stability** retrieves from `pipeline_context` → avoids re-preprocessing

This ensures:
- **Consistency**: Same data used across all stages
- **Efficiency**: Preprocessing happens only once
- **Provenance**: Shared preprocessing_info tracks all transformations

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

**Important**: PCA initialization is NOT recommended for non-centered parameterization (see [PCA_INITIALIZATION_GUIDE.md](PCA_INITIALIZATION_GUIDE.md)). Empirical testing shows PCA init actually slows convergence due to geometric mismatch between data space (PCA) and standardized space (non-centered parameterization).

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

### 5. PCA Initialization (Centered Parameterization Only)

PCA-based initialization available via `use_pca_initialization: true` in config:

```yaml
# config.yaml (centered parameterization)
preprocessing:
  use_pca_initialization: true  # OK for sparse_gfa (centered)

# config_convergence.yaml (non-centered parameterization)
preprocessing:
  use_pca_initialization: false  # NOT recommended for sparse_gfa_fixed
```

**Why non-centered + PCA is incompatible**:

- PCA operates in data space: `X ≈ ZW^T`
- Non-centered uses standardized space: `Z_raw ~ N(0,1)`, then `Z = Z_raw × scale`
- Geometric mismatch forces longer warmup (empirically verified: MAD 1000, K=5,12 faster WITHOUT PCA init)

**When to use PCA init**:

- ✅ Centered parameterization (`model_type: "sparseGFA"`)
- ✅ High-dimensional data where default init struggles
- ❌ Non-centered parameterization (`model_type: "sparse_gfa_fixed"`)

See [PCA_INITIALIZATION_GUIDE.md](PCA_INITIALIZATION_GUIDE.md) for details.

### 6. Shared Data Pipeline

Data preprocessed once and shared across experiments (default mode):

```python
# In run_experiments.py
pipeline_context = {
    "X_list": None,              # Preprocessed data
    "preprocessing_info": None,  # Transformation metadata
    "shared_mode": True          # Enable sharing
}

# data_validation populates context
pipeline_context["X_list"] = X_list

# factor_stability uses shared data
if pipeline_context["X_list"] is not None:
    X_list = pipeline_context["X_list"]  # Reuse
```

Override with `--no-shared-data` or `--independent-mode` for troubleshooting.

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

The `ExperimentFramework` base class creates run directories at initialization:

```python
# In run_experiments.py (lines 448-460):
if len(args.experiments) == 1:
    unified_dir = output_dir / f"{args.experiments[0]}{config_suffix}_run_{run_timestamp}"
else:
    unified_dir = output_dir / f"{'_'.join(args.experiments)}{config_suffix}_run_{run_timestamp}"

unified_dir.mkdir(parents=True, exist_ok=True)
config["experiments"]["base_output_dir"] = str(unified_dir)
```

**Example run directories:**

- `factor_stability_rois-sn_conf-age+sex+tiv_K2_percW33_MAD1000.0_run_20251022_000604/`
- `all_rois-sn_conf-age+sex+tiv_K2_percW33_MAD3.0_run_20251021_110602/`

#### 2. Experiment Subdirectories

**Within run_experiments.py pipeline:**

Subdirectories are created automatically for integrated experiments:

```python
# In ExperimentFramework (lines 730-741 of run_experiments.py):
repro_exp = RobustnessExperiments(experiment_config, logger)

# CRITICAL: Override base_output_dir with unified_dir (run directory)
if 'unified_dir' in locals() and unified_dir:
    repro_exp.base_output_dir = unified_dir
    logger.info(f"📁 Set base_output_dir to run directory: {unified_dir}")

# Create experiment subdirectory inside run directory
semantic_name = repro_exp._generate_semantic_experiment_name("factor_stability", experiment_config)
experiment_output_dir = repro_exp.base_output_dir / semantic_name
experiment_output_dir.mkdir(parents=True, exist_ok=True)
```

**Format:** `{run_dir}/{experiment_name}_{params}/`

**Examples:**

- `data_validation_tree10_20251022_000604/`
- `robustness_tests_fixed_K2_percW33_slab4_2_MAD1000.0_tree10_20251022_000622/`
- `03_factor_stability_fixed_K2_percW33_slab4_2_MAD1000.0_tree10/`

**Standalone scripts (train_sparse_gfa_fixed.py):**

Must explicitly create experiment subdirectory inside run directory:

```python
semantic_name = repro_exp._generate_semantic_experiment_name("factor_stability", exp_config)
output_dir = repro_exp.base_output_dir / semantic_name  # Create inside run dir
output_dir.mkdir(parents=True, exist_ok=True)

# CRITICAL: Update base_output_dir to experiment-specific directory
# This ensures all output (including individual_plots) goes to the right place
repro_exp.base_output_dir = output_dir
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

#### 4. Execution Modes: Integrated vs Standalone

**Two execution patterns exist in the codebase:**

**Integrated Execution (via run_experiments.py)**

```python
# run_experiments.py orchestrates experiments
if "factor_stability" in experiments_to_run:
    # 1. Create unified run directory
    unified_dir = output_dir / f"factor_stability_rois-sn...run_20251022_000604"

    # 2. Instantiate experiment class
    repro_exp = RobustnessExperiments(experiment_config, logger)

    # 3. Override base_output_dir (critical!)
    repro_exp.base_output_dir = unified_dir

    # 4. Create experiment subdirectory
    experiment_output_dir = unified_dir / "03_factor_stability_fixed_K2..."
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    # 5. Call experiment method with output_dir
    result = repro_exp.run_factor_stability_analysis(
        X_list=X_list,
        output_dir=str(experiment_output_dir),  # Explicit path
        ...
    )
```

**Characteristics:**

- Run directory created by framework
- Experiment subdirectories created inside run directory
- Logging goes to run-specific `{run_dir}/experiments.log`
- Multiple experiments can share data (via `pipeline_context`)

**Standalone Execution (direct script)**

```python
# experiments/train_sparse_gfa_fixed.py
if __name__ == "__main__":
    # 1. Create own run directory
    run_dir = output_dir / f"factor_stability_rois-sn...run_20251022_000604"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 2. Instantiate experiment class
    repro_exp = RobustnessExperiments(exp_config, logger)
    # repro_exp.base_output_dir initially points to "results/"

    # 3. Create experiment subdirectory
    semantic_name = repro_exp._generate_semantic_experiment_name("factor_stability", exp_config)
    experiment_dir = run_dir / semantic_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # 4. Update base_output_dir to experiment directory
    repro_exp.base_output_dir = experiment_dir

    # 5. Call experiment method
    result = repro_exp.run_factor_stability_analysis(
        X_list=X_list,
        # No output_dir needed - uses self.base_output_dir
        ...
    )
```

**Characteristics:**

- Script manages its own directory structure
- Logging goes to global `results/experiments.log`
- Single experiment per execution
- Must manually update `base_output_dir` after directory creation

#### 5. ExperimentFramework and ExperimentConfig Relationship

**Key architectural insight:**

```python
# ExperimentConfig is a dataclass (no base_output_dir field)
@dataclass
class ExperimentConfig:
    experiment_name: str
    K: int
    percW: float
    # ... but NO base_output_dir attribute

# ExperimentFramework is initialized with config OR path
class ExperimentFramework:
    def __init__(self, config_or_output_dir, ...):
        if hasattr(config_or_output_dir, "experiment_name"):
            # It's an ExperimentConfig
            self.config = config_or_output_dir
            # Gets base_output_dir from ConfigHelper
            self.base_output_dir = ConfigHelper.get_output_dir_safe(config_or_output_dir)
        else:
            # It's a direct path
            self.base_output_dir = Path(config_or_output_dir)
```

**This means:**

1. `ExperimentConfig` stores model/experiment parameters only
2. `base_output_dir` is set at `ExperimentFramework` initialization
3. Config dict's `["experiments"]["base_output_dir"]` doesn't automatically become `self.base_output_dir`
4. Must manually override `base_output_dir` when run directory is created outside the class

#### 6. Output Directory Parameter Flow

**Method signature pattern:**

```python
def run_factor_stability_analysis(
    self,
    X_list: List[np.ndarray],
    output_dir: str = None,  # Optional experiment-specific directory
    **kwargs
) -> ExperimentResult:
    # Store for use by plotting methods
    if output_dir:
        self._experiment_output_dir = Path(output_dir)
    else:
        self._experiment_output_dir = None
```

**Directory resolution hierarchy in plotting/analysis code:**

```python
# Priority chain (highest to lowest):
if hasattr(self, '_experiment_output_dir') and self._experiment_output_dir:
    base_dir = self._experiment_output_dir  # 1. Explicit output_dir parameter
elif hasattr(self, 'base_output_dir') and self.base_output_dir:
    base_dir = Path(self.base_output_dir)   # 2. Framework base_output_dir
else:
    base_dir = get_output_dir(config) / "factor_stability"  # 3. Global fallback
```

**This pattern allows:**

- Integrated execution: Pass explicit `output_dir` for full control
- Standalone execution: Rely on `base_output_dir` set by script
- Legacy support: Fallback to global directories if neither is set

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
# Default pipeline (recommended)
python run_experiments.py --config config_convergence.yaml \
  --experiments all \
  --select-rois volume_sn_voxels.tsv \
  --regress-confounds age sex tiv

# Specific stages
python run_experiments.py --config config_convergence.yaml \
  --experiments data_validation factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --K 20 --qc-outlier-threshold 3.0

# Factor stability only (if data already validated)
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
2. **Compute**: 4-6 hours for full factor stability (4 chains × 13K samples, max_tree_depth=10)
3. **Storage**: ~100-500MB per experiment run (with all plots + matrices)

**MCMC Settings**:
- `max_tree_depth: 10` → 2^10 = 1024 max leapfrog steps (sufficient for most cases)
- `max_tree_depth: 13` → 2^13 = 8192 max leapfrog steps (overkill, much slower)

## Related Documentation

For more specific information, see:

- **[../README.md](../README.md)** - Quick start guide and overview
- **[CONVERGENCE_TESTING_GUIDE.md](CONVERGENCE_TESTING_GUIDE.md)** - MCMC convergence diagnostics
- **[PCA_INITIALIZATION_GUIDE.md](PCA_INITIALIZATION_GUIDE.md)** - PCA initialization (centered only)
- **[configuration.md](configuration.md)** - Complete configuration reference
- **[TESTING.md](TESTING.md)** - Testing guide

## References

- Ferreira et al. 2024: Factor stability methodology
- Piironen & Vehtari 2017: Regularized horseshoe priors
- qMAP-PD Study: <https://qmaplab.com/qmap-pd>
