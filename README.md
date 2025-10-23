# Sparse Group Factor Analysis (SGFA) for qMAP-PD

Bayesian implementation of Sparse Group Factor Analysis using regularized horseshoe priors to identify latent disease factors in Parkinson's disease neuroimaging data. Uses NumPyro/JAX for efficient MCMC sampling with the NUTS algorithm.

## Overview

SGFA discovers latent factors that explain shared variation across multiple neuroimaging modalities while enforcing sparsity through regularized horseshoe priors. Designed for neuroimaging biomarker discovery in the qMAP-PD study dataset.

### Key Features

- **Sparse Bayesian Factor Analysis**: Regularized horseshoe priors with slab regularization
- **Multi-view Integration**: Joint analysis of imaging and clinical data  
- **Neuroimaging Preprocessing**: MAD-based QC, spatial imputation, position tracking
- **MCMC Convergence**: NUTS sampler with R-hat/ESS diagnostics
- **Factor Stability**: Cross-chain consensus (Ferreira et al. 2024)

## Main Pipeline

```bash
# Default pipeline (recommended)
python run_experiments.py --config config_convergence.yaml \
  --experiments all \
  --select-rois volume_sn_voxels.tsv \
  --regress-confounds age sex tiv

# Or specify stages explicitly
python run_experiments.py --config config_convergence.yaml \
  --experiments data_validation factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --K 20 --percW 33 --qc-outlier-threshold 3.0

# Factor stability only (if data already validated)
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --K 20 --percW 33 --qc-outlier-threshold 3.0
```

### Pipeline Stages

The default pipeline (`--experiments all`) runs two stages with shared preprocessed data:

| Experiment | Purpose | Outputs |
|------------|---------|---------|
| **data_validation** | Data quality, PCA analysis | Distribution plots, quality metrics, preprocessed data |
| **factor_stability** | Multi-chain consensus | Consensus W/Z, stability heatmap, R-hat |

**Note**: `robustness_testing` is available but not part of the default pipeline (only 150 iterations, primarily a smoke test). Run explicitly if needed: `--experiments robustness_testing`

### Optional Experiments

**SGFA Parameter Comparison** (`sgfa_configuration_comparison`)

Compare different SGFA hyperparameter configurations using aligned R-hat convergence diagnostics:

```bash
python run_experiments.py --config config_convergence.yaml \
  --experiments sgfa_configuration_comparison \
  --select-rois volume_sn_voxels.tsv
```

**Features**:
- Tests multiple SGFA variants (standard, sparse_only, group_only, basic_fa)
- Uses **aligned R-hat** as primary convergence metric (accounts for sign/permutation indeterminacy)
- Ranks variants by: (1) Convergence quality (R-hat < 1.1), (2) Speed, (3) Memory
- Generates convergence comparison plots showing max R-hat and convergence rate

**Why Aligned R-hat?** Standard R-hat incorrectly indicates poor convergence in factor models because chains can converge to equivalent solutions that differ only in sign flips or factor ordering. Aligned R-hat matches factors across chains using cosine similarity and aligns signs before computing R-hat.

**Configuration**:
```yaml
sgfa_configuration_comparison:
  num_chains: 2              # Minimum 2 chains for aligned R-hat
  num_samples: 1000          # Samples per chain
  num_warmup: 500            # Warmup per chain
```

See [docs/configuration.md](docs/configuration.md#sgfa-parameter-comparison-sgfa_configuration_comparison) for details.

## Configuration (`config_convergence.yaml`)

### Model Parameters

```yaml
model:
  model_type: "sparse_gfa_fixed"  # Convergence-tested implementation
  K: 20                          # Latent factors
  percW: 33                      # Sparsity %
  slab_df: 4                     # Slab degrees of freedom  
  slab_scale: 2                  # Slab scale
  num_samples: 10000            # MCMC samples
  num_warmup: 3000              # Warmup samples
  num_chains: 4                  # Parallel chains
```

### Preprocessing

```yaml
preprocessing:
  qc_outlier_threshold: 3.0      # MAD threshold
  variance_threshold: 0.0        # Disabled (preserve variance)
  roi_based_selection: false     # Disabled (preserve spatial clusters)
```

**MAD Threshold Guide**:

- `2.5-3.0`: Stringent QC, cleaner signal
- `3.0-4.0`: Moderate QC (default)
- `4.5-5.0`: Permissive, preserve heterogeneity for subtype discovery

### MCMC

```yaml
factor_stability:
  target_accept_prob: 0.8       # NUTS acceptance
  max_tree_depth: 10            # NUTS depth limit (2^10 = 1024 leapfrog steps)
  cosine_threshold: 0.8         # Factor matching
  min_match_rate: 0.5           # Chain agreement
```

## Command-Line Overrides

```bash
--K 20                          # Number of factors
--percW 33                      # Sparsity percentage
--qc-outlier-threshold 3.0      # MAD threshold
--max-tree-depth 10             # NUTS max depth (2^10 = 1024 leapfrog steps)
--select-rois <file>            # ROI selection
--regress-confounds age sex tiv # Confound regression
```

## Project Structure

```
sgfa_qmap-pd/
├── run_experiments.py          # Main entry point
├── config_convergence.yaml     # Production configuration (non-centered parameterization)
├── config.yaml                 # Original configuration (centered parameterization)
│
├── models/                     # Model implementations
│   ├── sparse_gfa_fixed.py     # Non-centered sparse GFA with regularized horseshoe
│   └── factory.py              # Model instantiation factory
│
├── data/                       # Data loading and preprocessing
│   ├── qmap_pd.py              # qMAP-PD dataset loader
│   ├── preprocessing.py        # Neuroimaging preprocessing (MAD QC, imputation)
│   └── preprocessing_integration.py  # Config merging and pipeline integration
│
├── experiments/                # Experimental validation pipeline
│   ├── framework.py            # Base infrastructure (logging, output management)
│   ├── data_validation.py      # Data quality assessment
│   ├── robustness_testing.py   # Optional smoke test (150 iterations)
│   └── train_sparse_gfa_fixed.py  # Standalone factor stability runner
│
├── analysis/                   # Post-hoc analysis and diagnostics
│   ├── factor_stability.py     # Multi-chain consensus (Ferreira et al. 2024)
│   └── mcmc_diagnostics.py     # R-hat, ESS, trace plots, convergence checks
│
├── visualization/              # Plotting utilities
│   ├── factor_plots.py         # Factor loading visualizations
│   ├── brain_plots.py          # Brain region mapping
│   ├── subtype_plots.py        # PD subtype discovery visualizations
│   └── neuroimaging_utils.py   # Neuroimaging-specific plotting
│
├── core/                       # Core utilities
│   ├── config_utils.py         # Configuration helpers
│   ├── io_utils.py             # File I/O (save plots, results)
│   ├── logger_utils.py         # Logging setup
│   ├── pca_initialization.py   # PCA-based MCMC initialization (centered only)
│   └── utils.py                # General utilities
│
└── docs/                       # Documentation
    ├── ARCHITECTURE.md         # Detailed architecture and design patterns
    ├── CONVERGENCE_TESTING_GUIDE.md  # MCMC convergence testing guide
    ├── PCA_INITIALIZATION_GUIDE.md   # PCA initialization (centered only)
    └── configuration.md        # Configuration reference
```

**See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.**

### Module Descriptions

**Core Infrastructure (`core/`)**

- Configuration management, file I/O, logging utilities
- Shared by all experiments and analysis code

**Data Pipeline (`data/`)**

- Load multi-view data (imaging + clinical)
- MAD-based quality control and spatial imputation
- Position tracking for brain coordinate mapping

**Models (`models/`)**

- Sparse Bayesian factor analysis with NumPyro/JAX
- Regularized horseshoe priors for structured sparsity

**Experiments (`experiments/`)**

- Orchestrates preprocessing → modeling → analysis
- Manages output directories and logging
- Runs validation pipeline stages

**Analysis (`analysis/`)**

- Factor stability across MCMC chains
- Convergence diagnostics (R-hat, ESS)
- Consensus loadings/scores computation

**Visualization (`visualization/`)**

- Factor loading plots, brain maps
- MCMC diagnostics (traces, posteriors)
- Summary visualizations

## Installation

```bash
git clone https://github.com/meeramads/sgfa_qmap-pd
cd sgfa_qmap-pd
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Results Structure

```
results/all_rois-sn_K20_percW33_MAD3.0_YYYYMMDD_HHMMSS/
├── 01_data_validation/individual_plots/
├── 02_robustness_testing/individual_plots/
└── 03_factor_stability_K20_percW33/
    ├── chains/               # W.csv, Z.csv per chain
    ├── stability_analysis/   # Consensus loadings/scores  
    └── plots/                # Stability heatmap, diagnostics
```

## Preprocessing Pipeline

1. **MAD QC**: Remove outlier voxels (threshold=3.0)
2. **Spatial Imputation**: Fill missing with neighbors
3. **Median Imputation**: Fill remaining missing
4. **Position Tracking**: Maintain voxel→brain coordinates

**Example (MAD 3.0)**:

- Raw: 1794 voxels → After QC: ~531 voxels (~70% removed)

## Model Priors

**Regularized Horseshoe for Loadings (W)**:

- Global scale: τ₀ (data-dependent, Piironen & Vehtari 2017)
- Local shrinkage: λ ~ Half-Cauchy(0, 1)
- Slab regularization: c² ~ InverseGamma(slab_df/2, slab_scale·slab_df/2)

**Factors (Z)**: Regularized horseshoe if `reghsZ: true`

**Noise**: σ_m⁻² ~ Gamma(1, 1) per view

## Convergence Diagnostics

- **R-hat**: < 1.1 (between-chain vs within-chain variance)
- **ESS**: Effective sample size (higher is better)
- **Acceptance**: Target 0.6-0.8 for NUTS
- **Tree Depth**: Monitor if max limit is reached

**Good convergence (K=2, MAD 3.0)**:

- R-hat < 1.1, acceptance ~0.79, 2/2 effective factors

**Failed convergence (K=2, MAD 100)**:

- R-hat = 1339, 0/2 effective factors (over-shrinkage from noise)

## Testing

```bash
pytest                    # All tests
pytest --cov=.           # With coverage
pytest tests/data/       # Specific module
```

## Documentation

For detailed information about specific aspects of the codebase:

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Complete architecture documentation, data flow diagrams, design patterns, and output directory structure
- **[CONVERGENCE_TESTING_GUIDE.md](docs/CONVERGENCE_TESTING_GUIDE.md)** - MCMC convergence diagnostics, R-hat interpretation, and troubleshooting
- **[PCA_INITIALIZATION_GUIDE.md](docs/PCA_INITIALIZATION_GUIDE.md)** - PCA initialization for MCMC (centered parameterization only; NOT recommended for non-centered)
- **[configuration.md](docs/configuration.md)** - Complete configuration reference and parameter descriptions
- **[TESTING.md](docs/TESTING.md)** - Testing guide and test suite documentation

## References

- qMAP-PD: <https://qmaplab.com/qmap-pd>
- Ferreira et al. 2024: Factor stability methodology
- Piironen & Vehtari 2017: Regularized horseshoe priors
- NumPyro: <https://num.pyro.ai/>
- JAX: <https://jax.readthedocs.io/>

## License

MIT
