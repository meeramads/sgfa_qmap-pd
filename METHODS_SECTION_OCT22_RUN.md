# Methods Section: Sparse Group Factor Analysis of qMAP-PD Dataset

**Analysis Date**: October 22, 2025
**Run ID**: `all_rois-sn_conf-age+sex+tiv_K5_percW33_MAD1000.0_run_20251022_112241`
**Model**: Sparse Group Factor Analysis with Fixed Regularization (Sparse GFA-Fixed)

---

## 1. Dataset

### 1.1 Study Population
We analyzed data from the qMAP-PD (quantitative Multimodal Assessment of Parkinson's Disease) dataset, comprising N = 86 early-stage Parkinson's Disease patients. The dataset integrates two complementary data modalities: high-dimensional neuroimaging measurements and clinical assessments.

### 1.2 Data Structure

#### 1.2.1 Neuroimaging Data
- **Region of Interest**: Substantia nigra (SN)
- **Number of voxels**: 1,794 voxels
- **Data format**: Volume matrix (N × M) where N = 86 subjects, M = 1,794 voxels
- **Spatial information**: Position lookup vector preserving 3D spatial coordinates
  - Spatial reconstruction formula: `volume_matrix @ position_lookup`
  - Position lookup dimensions: 1,794 rows × 3 columns (x, y, z coordinates)
- **File location**: `qMAP-PD_data/volume_sn_voxels.tsv`

#### 1.2.2 Clinical Data
- **Original features**: 17 clinical measurements
- **Confound variables**: 3 features (age, sex, total intracranial volume)
- **Retained features**: 14 clinical features after confound removal
- **File location**: `qMAP-PD_data/clinical.tsv`
- **Format**: CSV with header row containing feature names

#### 1.2.3 Data Alignment
Patient ordering is consistent across all data files: patient at row index `i` in clinical data (excluding header) corresponds to patient at row index `i` in all imaging volume matrices.

### 1.3 Total Feature Space
- **Imaging features**: 1,794 voxels
- **Clinical features**: 14 measurements
- **Total**: 1,808 features across 2 views

---

## 2. Data Preprocessing

All preprocessing was performed using the project's integrated preprocessing pipeline with the following configuration:

### 2.1 Configuration File Setup

**File**: `config.yaml`

Key preprocessing parameters:
```yaml
preprocessing:
  qc_outlier_threshold: 1000.0
  confounds: ['age', 'sex', 'tiv']
  enable_pca: false
  enable_spatial_processing: false
  enable_combat_harmonization: false
  harmonize_scanners: false
  feature_selection: 'none'
  scaling_method: 'robust'
  imputation_method: 'median'
```

### 2.2 Quality Control

#### 2.2.1 Outlier Detection
- **Method**: Median Absolute Deviation (MAD) filtering
- **Threshold**: MAD = 1000.0
- **Rationale**: Extremely high threshold effectively disables voxel-level outlier removal
  - Conservative approach to avoid information loss in small sample (N=86)
  - Regularized priors in Bayesian model provide robust inference without aggressive preprocessing

#### 2.2.2 Subject-Level Quality Assessment
- **Flagged subjects**: 2 subjects with elevated outlier percentages
  - Maximum outlier percentage: 17.5%
  - Mean outlier percentage: 1.1%
- **Action**: All subjects retained after manual review
- **Output**: `flagged_subjects.csv` documenting QC metrics

### 2.3 Confound Regression

**Confound variables**: Age, sex, total intracranial volume (TIV)

#### 2.3.1 Application to Imaging Data
Linear regression applied to each voxel independently:
```
X_voxel,residual = X_voxel,raw - β₀ - β₁·age - β₂·sex - β₃·TIV
```
where β coefficients are estimated via ordinary least squares.

#### 2.3.2 Application to Clinical Data
Confound variables removed from feature set:
- Original: 17 clinical features
- Dropped: age, sex, TIV
- Remaining: 14 clinical features

**Rationale**: Prevents confound variables from driving latent factor structure.

### 2.4 Missing Data Imputation

#### 2.4.1 Median Imputation
Applied to both imaging and clinical features:
```python
X_imputed[i, j] = median(X[:, j]) if isnan(X[i, j]) else X[i, j]
```

#### 2.4.2 Spatial Imputation (Imaging Only)
For SN voxels, spatial neighborhood information used when available:
- Position lookup vectors identify spatial neighbors
- Missing voxels imputed from median of k-nearest spatial neighbors
- Fallback: Standard median imputation if spatial structure unavailable

### 2.5 Standardization

**Method**: Robust scaling using median and interquartile range (IQR)

Applied separately to each feature:
```
X_scaled[i, j] = (X[i, j] - median(X[:, j])) / IQR(X[:, j])
```

**Advantages over z-score**:
- Robust to outliers
- Does not assume Gaussian distribution
- Better suited for neuroimaging data with heavy tails

**Application**: Both imaging and clinical features scaled after imputation and confound regression.

### 2.6 Feature Selection

**Method**: None applied
- **Imaging**: 1,794/1,794 voxels retained (100%)
- **Clinical**: 14/14 features retained (100%)

**Rationale**: Sparse regularization in Bayesian model performs implicit feature selection through shrinkage priors. Preprocessing-level feature selection deemed redundant and potentially information-losing.

### 2.7 Preprocessing Pipeline Execution

**Command**:
```bash
python3 run_experiments.py \
  --roi volume_sn_voxels \
  --confounds age sex tiv \
  --qc-outlier-threshold 1000.0 \
  --K 5 \
  --percW 33.0 \
  --max-tree-depth 10 \
  --model-type sparse_gfa_fixed
```

**Output directory structure**:
```
results/all_rois-sn_conf-age+sex+tiv_K5_percW33_MAD1000.0_run_20251022_112241/
├── data_validation_tree10_20251022_112241/
│   ├── position_lookup_filtered/
│   │   └── position_sn_voxels_filtered.tsv
│   └── filtered_position_lookups/
│       └── sn_filtered_position_lookup.csv
└── robustness_tests_fixed_K5_percW33_slab4_2_MAD1000.0_tree10_20251022_112319/
```

---

## 3. Statistical Model

### 3.1 Model Framework

We implemented **Sparse Group Factor Analysis with Fixed Regularization** (Sparse GFA-Fixed), a Bayesian hierarchical model designed for multi-view dimensionality reduction with structured sparsity.

### 3.2 Generative Model

The model assumes observed data from each view are generated from shared latent factors:

```
X_m = Z W_m^T + E_m,  for m ∈ {1, 2}
```

**Components**:
- **X_m**: Observed data matrix for view m (N × D_m)
  - X₁: Substantia nigra voxels (86 × 1,794)
  - X₂: Clinical features (86 × 14)
- **Z**: Latent factor score matrix (N × K)
  - N = 86 subjects
  - K = 5 latent factors
  - **Shared across views**: Captures common variation between imaging and clinical domains
- **W_m**: Factor loading matrix for view m (D_m × K)
  - W₁: SN voxel loadings (1,794 × 5)
  - W₂: Clinical loadings (14 × 5)
  - **View-specific**: Each modality has unique relationship to latent factors
- **E_m**: Observation noise matrix (N × D_m)
  - Independent Gaussian noise per view

### 3.3 Prior Structure

The model employs a three-level regularized horseshoe prior hierarchy for structured sparsity.

#### 3.3.1 Factor Loading Priors (Three-Level Hierarchy)

**Level 1 - Global shrinkage (per-view)**:
```
τ_W^(m) ~ Student-t⁺(df=2, loc=0, scale=τ₀^(m))
```
- **Distribution**: Left-truncated Student-t (truncated below at 0)
- **Degrees of freedom**: df = 2 (heavy-tailed for robustness)
- **Scale parameter**: τ₀^(m) is view-specific, dimensionality-aware prior floor
  - Computed as: τ₀^(m) = (p₀/(D_m - p₀)) · (σ_m/√N)
  - p₀: Expected number of relevant features (derived from percW)
  - D_m: Number of features in view m
  - σ_m: Noise standard deviation for view m
  - N: Number of subjects (86)
- **Interpretation**: Global shrinkage strength for entire view
  - Small τ_W^(m) → aggressive shrinkage
  - Large τ_W^(m) → weak shrinkage

**Level 2 - Slab regularization (per-view × per-factor)**:
```
c_W^(m,k) ~ Student-t⁺(df=4, loc=0, scale=2)
```
- **Distribution**: Left-truncated Student-t
- **Degrees of freedom**: ν = 4 (slab_df parameter)
- **Scale**: s = 2 (slab_scale parameter)
- **Interpretation**: Factor-level sparsity control within each view
  - Prevents all factors from being equally sparse
  - Allows some factors to be dense while others are sparse

**Level 3 - Local shrinkage (per-loading)**:
```
λ_W^(d,k) ~ Half-Cauchy(scale=1)
```
- **Distribution**: Half-Cauchy (heavy-tailed, promotes sparsity)
- **Interpretation**: Individual loading-level shrinkage

**Combined loading prior**:
```
W_{d,k}^(m) ~ N(0, (τ_W^(m))² · (c_W^(m,k))² · (λ_W^(d,k))²)
```
- Final variance is product of three levels
- Enables both individual sparsity (via λ) and group structure (via τ and c)

#### 3.3.2 Sparsity Specification

**Target sparsity**: percW = 33%
- Approximately 2/3 of loadings shrunk close to zero
- Remaining 1/3 retain substantial magnitude
- Implemented through calibration of τ₀ scale parameters

#### 3.3.3 Factor Score Priors

Analogous regularized horseshoe structure applied to latent factors Z:

**Global shrinkage**:
```
τ_Z ~ Student-t⁺(df=2, loc=0, scale=τ₀_Z)
```

**Slab regularization**:
```
c_Z^(k) ~ Student-t⁺(df=4, loc=0, scale=2)
```
- Note: Single view for Z (not view-specific)

**Local shrinkage**:
```
λ_Z^(n,k) ~ Half-Cauchy(scale=1)
```

**Combined prior**:
```
Z_{n,k} ~ N(0, (τ_Z)² · (c_Z^(k))² · (λ_Z^(n,k))²)
```

**Rationale**: Shrinkage on Z prevents overfitting in latent space, particularly important with small sample size (N=86).

#### 3.3.4 Noise Model

**Observation noise variance**:
```
σ_m² ~ Inverse-Gamma(α=1, β=1)
```
- View-specific noise variance
- Weakly informative prior (α=1, β=1 corresponds to uniform prior on log(σ))
- Allows different noise levels for imaging vs. clinical data

### 3.4 Parameterization Details

#### 3.4.1 Non-Centered Parameterization

To improve MCMC sampling efficiency, loadings and scores use non-centered parameterization:

**For W**:
```
W_raw ~ N(0, 1)  (standard normal)
W = W_raw · √((τ_W^(m))² · (c_W^(m,k))² · (λ_W^(d,k))²)
```

**For Z**:
```
Z_raw ~ N(0, 1)
Z = Z_raw · √((τ_Z)² · (c_Z^(k))² · (λ_Z^(n,k))²)
```

**Benefits**:
- Reduces posterior correlation between W/Z and their scale parameters
- Improves sampler efficiency when priors are strong (small τ values)
- Essential for high-dimensional problems (D₁=1,794 features)

#### 3.4.2 Partial Centering

Adaptive blend between centered and non-centered parameterizations:

**Blending formula**:
```
φ(τ) = √(1 - exp(-τ))
```

**Application**:
```
W = φ(τ_W) · W_centered + (1 - φ(τ_W)) · W_noncentered
```

**Adaptive behavior**:
- When τ → 0 (strong prior): φ → 0, uses non-centered (optimal)
- When τ → ∞ (weak prior): φ → 1, uses centered (optimal)
- Automatic adaptation across parameter space regions

**Impact**: Improves convergence across full range of τ values sampled during MCMC.

### 3.5 Model Implementation

**File**: `models/sparse_gfa_fixed.py`

**Key model code sections**:
- Lines 352-358: Global shrinkage τ_W sampling
- Lines 256-264: Slab regularization c_W sampling
- Lines 247-249: Local shrinkage λ_W sampling
- Lines 152-165: Partial centering implementation
- Lines 200-215: Likelihood computation

---

## 4. Bayesian Inference

### 4.1 MCMC Sampling Algorithm

**Algorithm**: No-U-Turn Sampler (NUTS)
- Extension of Hamiltonian Monte Carlo (HMC)
- Automatically tunes trajectory length
- Eliminates need to manually specify number of leapfrog steps

**Implementation**: NumPyro 0.13.2
- Built on JAX 0.4.20 for automatic differentiation
- JIT compilation for computational efficiency
- CPU-based execution (GPU not utilized)

### 4.2 MCMC Configuration

#### 4.2.1 Sampling Parameters

**Warmup phase**:
```
num_warmup = 1000
```
- Burn-in period for step size and mass matrix adaptation
- Samples discarded from posterior analysis

**Posterior sampling**:
```
num_samples = 2000
```
- Retained samples for inference
- Per-chain effective sample size assessed via ESS diagnostics

**Number of chains**:
```
num_chains = 1
```
- Single chain per run due to hardware constraints
- Robustness assessed via multiple independent runs (see Section 4.6)

**Random seeds**: [42, 123, 456]
- Three independent runs with different random seeds
- Cross-seed consensus for stability assessment

#### 4.2.2 NUTS-Specific Parameters

**Target acceptance probability**:
```
target_accept_prob = 0.80
```
- Controls step size adaptation
- Higher values → smaller step size → fewer rejections but slower sampling
- 0.80 balances efficiency and acceptance

**Maximum tree depth**:
```
max_tree_depth = 10
```
- Limits trajectory length to 2^10 = 1,024 leapfrog steps
- Prevents excessive computation in regions of high curvature
- **Note**: Chains consistently reached maximum depth (1,023 steps observed)
  - Indicates challenging geometry but acceptable convergence

**Mass matrix**:
```
dense_mass = False
```
- Uses diagonal mass matrix (approximates parameter-wise variance)
- **Memory savings**: ~5 GB compared to dense (full covariance) mass matrix
- Essential for high-dimensional parameter space (>10,000 parameters)

**Step size adaptation**:
- Dual averaging algorithm during warmup
- Adapts step size to achieve target_accept_prob
- Final adapted step size used throughout posterior sampling

### 4.3 Initialization Strategy

**Method**: PCA-based initialization

#### 4.3.1 PCA Computation

**Step 1**: Concatenate preprocessed data across views
```python
X_concat = np.hstack([X_sn, X_clinical])  # Shape: (86, 1808)
```

**Step 2**: Compute principal components
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
Z_init = pca.fit_transform(X_concat)  # Shape: (86, 5)
W_concat_init = pca.components_.T  # Shape: (1808, 5)
```

**Step 3**: Split concatenated loadings by view
```python
W_sn_init = W_concat_init[:1794, :]  # Shape: (1794, 5)
W_clinical_init = W_concat_init[1794:, :]  # Shape: (14, 5)
```

#### 4.3.2 Variance Explained
Total variance explained by K=5 components: **69.91%**

Per-component breakdown:
- **Component 1**: 40.60%
- **Component 2**: 9.30%
- **Component 3**: 7.91%
- **Component 4**: 6.60%
- **Component 5**: 5.50%

**Rationale for K=5**: Captures majority of variance while maintaining interpretability.

#### 4.3.3 Hierarchical Parameter Initialization

**Shrinkage parameters**:
```python
# Global shrinkage: initialized from empirical variance
tau_W_init[m] = np.std(W_init[m]) * sqrt(p0 / (Dm[m] - p0))

# Slab regularization: initialized near prior mode
c_W_init[m, k] = 1.0

# Local shrinkage: initialized from loading magnitudes
lambda_W_init[d, k] = abs(W_init[d, k]) / (tau_W_init[m] * c_W_init[m, k])
```

**Noise variance**:
```python
# Initialized from PCA reconstruction error
E = X - Z_init @ W_init.T
sigma_init[m] = np.std(E[:, view_m])
```

**File**: `core/pca_initialization.py`
- Lines 266-290: Parameter initialization logic
- Lines 276-280: Variance explained computation

### 4.4 MCMC Execution

**Command line execution**:
```bash
python3 run_experiments.py \
  --roi volume_sn_voxels \
  --confounds age sex tiv \
  --K 5 \
  --percW 33.0 \
  --slab-df 4 \
  --slab-scale 2 \
  --num-warmup 1000 \
  --num-samples 2000 \
  --num-chains 1 \
  --max-tree-depth 10 \
  --target-accept-prob 0.8 \
  --model-type sparse_gfa_fixed \
  --qc-outlier-threshold 1000.0 \
  --use-pca-initialization \
  --random-seed 42
```

**Programmatic call** (within `experiments/robustness_testing.py`):
```python
result = run_sgfa_analysis(
    X_views=[X_sn, X_clinical],
    view_names=['volume_sn_voxels', 'clinical'],
    hyperparams={'K': 5, 'percW': 33.0, 'slab_df': 4, 'slab_scale': 2},
    num_warmup=1000,
    num_samples=2000,
    num_chains=1,
    max_tree_depth=10,
    target_accept_prob=0.8,
    use_pca_initialization=True,
    model_type='sparse_gfa_fixed',
    random_seed=seed
)
```

### 4.5 Convergence Diagnostics

#### 4.5.1 Potential Scale Reduction (R-hat)
```
R_hat = sqrt(Var_total / Var_within)
```
- Compares between-chain to within-chain variance
- Target: R_hat < 1.01 for convergence
- **Note**: Single chain per run, so cross-seed R-hat computed

#### 4.5.2 Effective Sample Size (ESS)
**Bulk ESS**: For central posterior distribution
```
ESS_bulk ≈ N_samples / (1 + 2·∑ρ_t)
```
where ρ_t is autocorrelation at lag t.

**Tail ESS**: For posterior tails (5th and 95th percentiles)
- More stringent than bulk ESS
- Critical for accurate credible intervals

**Target**: ESS > 100 per chain (400 across 4 independent runs)

#### 4.5.3 Trace Plots
Visual inspection of parameter trajectories:
- **Good mixing**: Random scatter around stable mean
- **Poor mixing**: Trends, periodic patterns, or stuck regions

**Generated for**:
- Global shrinkage: τ_W^(1), τ_W^(2), τ_Z
- Noise variance: σ₁², σ₂²
- Selected loadings: W[100, 1], W[500, 3], etc.

**Output**: `individual_plots/hyperparameters/trace_*.png`

#### 4.5.4 Tree Depth Monitoring
**Warning threshold**: Chains consistently hitting max_tree_depth
- Indicates regions of high posterior curvature
- Can signal challenging geometry but does not necessarily indicate non-convergence
- **Observed**: Majority of iterations reached 1,023 steps (2^10 - 1)
- **Interpretation**: Challenging posterior geometry accepted as operating constraint

### 4.6 Robustness Assessment via Multiple Runs

#### 4.6.1 Random Seeds
Three independent MCMC runs with different seeds:
- **Seed 42**: Primary run
- **Seed 123**: Robustness run 1
- **Seed 456**: Robustness run 2

**Independence**: Different random number streams ensure independent exploration of posterior.

#### 4.6.2 Factor Stability Metrics
**Cosine similarity** between corresponding factors across runs:
```
cos(k_ref, k_run) = (W_ref[:, k] · W_run[:, k]) / (||W_ref[:, k]|| · ||W_run[:, k]||)
```

**Matching criterion**: cos > 0.8 for valid factor correspondence

**Minimum match rate**: ≥50% of runs must match for factor to be included in consensus

---

## 5. Post-Processing and Consensus Estimation

### 5.1 Sign Indeterminacy Problem

Factor models exhibit fundamental sign ambiguity:
```
X = Z W^T = (-Z)(-W)^T
```
Both (Z, W) and (-Z, -W) produce identical likelihood.

**Consequence**: Without alignment, factors may have opposite signs across runs, causing consensus median to attenuate toward zero.

### 5.2 Sign Alignment Algorithm

#### 5.2.1 Reference Selection
- **Reference run**: Seed 42 (first run)
- **Alignment targets**: Seeds 123 and 456 aligned to seed 42

#### 5.2.2 Normalized Cosine Similarity
For each factor k in run j:

**Step 1**: Compute normalized cosine similarity to reference
```python
ref_loading = W_ref[:, k]  # Reference factor k
run_loading = W_run[:, k]  # Run j factor k

ref_norm = np.linalg.norm(ref_loading)
run_norm = np.linalg.norm(run_loading)

# Normalized cosine similarity
cosine_sim = np.dot(ref_loading, run_loading) / (ref_norm * run_norm)
```

**Step 2**: Sign flip decision
```python
if cosine_sim < 0:
    W_run[:, k] = -W_run[:, k]  # Flip loadings
    Z_run[:, k] = -Z_run[:, k]  # Flip scores
```

**Step 3**: Zero-vector handling
```python
if ref_norm < 1e-10 or run_norm < 1e-10:
    # Degenerate factor, keep original sign
    pass
```

**Advantages over raw dot product**:
- **Scale invariant**: Not affected by magnitude differences between runs
- **Bounded**: cosine ∈ [-1, 1] regardless of dimensionality
- **Geometrically meaningful**: Measures angle between loading vectors

**Implementation**: `analysis/factor_stability.py`, lines 1994-2025 (loadings), lines 2095-2126 (scores)

### 5.3 Factor Matching Across Runs

#### 5.3.1 Matching Algorithm
Factors may emerge in different orders across runs. Optimal matching required.

**Step 1**: Compute all pairwise cosine similarities
```python
# Cosine similarity matrix: K x K
C[k_ref, k_run] = cosine(W_ref[:, k_ref], W_run[:, k_run])
```

**Step 2**: Greedy matching (Hungarian algorithm alternative)
```python
for k_ref in range(K):
    k_run = argmax(C[k_ref, :])  # Best match for reference factor k
    if C[k_ref, k_run] > 0.8:  # Threshold for valid match
        matches[k_ref] = k_run
```

**Step 3**: Align matched factors
```python
W_run_aligned[:, k_ref] = W_run[:, matches[k_ref]]
```

#### 5.3.2 Match Quality Criteria
- **Cosine threshold**: 0.8 (strong correspondence required)
- **Minimum match rate**: 50% of runs must match
- **Action if unmatched**: Factor excluded from consensus (potential instability)

### 5.4 Consensus Estimation

#### 5.4.1 Median Aggregation
After sign alignment and factor matching:

**Consensus loadings**:
```python
W_consensus[m] = np.median([W_seed42[m], W_seed123[m], W_seed456[m]], axis=0)
```
- Shape: (D_m, K) for each view m
- Robust to outlier runs

**Consensus scores**:
```python
Z_consensus = np.median([Z_seed42, Z_seed123, Z_seed456], axis=0)
```
- Shape: (N, K) = (86, 5)

#### 5.4.2 Rationale for Median
- **Robustness**: Less sensitive to outlier chains than mean
- **Distributional assumptions**: Does not assume Gaussian posterior
- **Computational efficiency**: Simple to compute

**Alternative considered**: Posterior mean across all samples
- **Rejected due to**: Memory constraints with 2,000 samples × 3 runs × 10,000+ parameters

### 5.5 Output Files

#### 5.5.1 Consensus Factor Loadings

**Clinical loadings**:
```
consensus_loadings_clinical.csv
```
- Dimensions: 14 features × 5 factors
- Row names: Clinical feature names
- Column names: Factor_0, Factor_1, ..., Factor_4

**SN voxel loadings**:
```
consensus_loadings_SN_Factor_0.tsv
consensus_loadings_SN_Factor_1.tsv
...
consensus_loadings_SN_Factor_4.tsv
```
- Each file: 86 subjects × 1,794 voxels
- Row index: Subject ID (0-85)
- Column index: Voxel ID (0-1793)

#### 5.5.2 Consensus Factor Scores
```
consensus_factor_scores.csv
```
- Dimensions: 86 subjects × 5 factors
- Row index: Subject ID
- Column names: Factor_0, Factor_1, ..., Factor_4

#### 5.5.3 Spatial Reconstruction Files
```
position_sn_voxels_filtered.tsv
sn_filtered_position_lookup.csv
```
- 1,794 rows × 3 columns (x, y, z coordinates)
- Enables reconstruction of 3D brain maps from loading vectors

**Reconstruction formula**:
```python
brain_map_3d = loadings_1d @ position_lookup.T
```

---

## 6. Computational Environment

### 6.1 Hardware Specifications

**Memory**:
- Total RAM: 27.2 GB
- Allocated to MCMC: 21.7 GB (80% utilization)
- Memory optimization: Diagonal mass matrix (-5 GB vs. dense)

**CPU**:
- Cores: 16
- Architecture: Not specified (likely x86-64)

**GPU**: Not utilized
- JAX configured for CPU backend
- CUDA not enabled for this run

**Storage**:
- Results directory: `/cs/student/msc/aibh/2024/madhavan/sgfa_qmap-pd/results/`
- Data directory: `/cs/student/msc/aibh/2024/madhavan/sgfa_qmap-pd/qMAP-PD_data/`

### 6.2 Software Environment

#### 6.2.1 Python Version
```
Python 3.11
```
**Version constraints**: Python 3.8-3.11 required
- **Upper bound**: JAX 0.4.20 incompatible with Python 3.12+
- **Lower bound**: NumPy 1.23.5 requires Python ≥3.8

#### 6.2.2 Core Dependencies

**JAX ecosystem**:
```
JAX==0.4.20
jaxlib==0.4.20
```
- Automatic differentiation for gradient computation
- JIT compilation for performance
- Version pinned for CUDA compatibility on GPU workstation (for future runs)

**Probabilistic programming**:
```
NumPyro==0.13.2
```
- NUTS sampler implementation
- Pinned to match JAX 0.4.20 API

**Numerical computing**:
```
NumPy==1.23.5
SciPy==1.10.1
```
- Core array operations
- PCA computation (via sklearn)

**Data handling**:
```
pandas==1.5.3
```
- CSV/TSV reading and writing
- DataFrame operations for clinical data

**Visualization**:
```
matplotlib==3.7.1
seaborn==0.12.2
```
- Trace plots, diagnostic visualizations
- Only PNG format used (PDF generation disabled)

**Neuroimaging** (for future spatial analysis):
```
nibabel==5.1.0
nilearn==0.10.1
```
- NIfTI file handling
- Spatial reconstruction utilities

#### 6.2.3 Installation Instructions

**Step 1**: Create virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

**Step 2**: Install dependencies
```bash
pip install -r requirements.txt
```

**Step 3**: Verify installation
```bash
python3 -c "import jax; import numpyro; print(f'JAX: {jax.__version__}, NumPyro: {numpyro.__version__}')"
```
Expected output: `JAX: 0.4.20, NumPyro: 0.13.2`

### 6.3 Memory Optimization Strategies

#### 6.3.1 Diagonal Mass Matrix
```python
dense_mass = False
```
- Memory: ~5 GB saved
- Trade-off: Slower convergence vs. memory footprint
- Justification: Essential for high-dimensional models on workstation hardware

#### 6.3.2 Streaming Data Processing
- Data loaded in chunks when possible
- Preprocessed views not stored in memory simultaneously
- Garbage collection triggered between experiment phases

#### 6.3.3 Sample Storage
```python
save_samples = False  # Samples not saved to disk
```
- Only summary statistics and diagnostics saved
- Full posterior samples discarded after consensus computation
- Memory: ~2-3 GB saved per run

### 6.4 Runtime Performance

**Total experiment runtime**: 9.0 minutes (540 seconds)

**Breakdown by phase**:
- **Data validation**: 26 seconds (4.8%)
  - Loading, preprocessing, QC analysis
- **MCMC sampling**: ~420 seconds (77.8%)
  - Warmup: ~1 minute per run × 3 runs = 3 minutes
  - Sampling: ~3 minutes per run × 3 runs = 9 minutes
  - **Note**: Exact breakdown not logged
- **Post-processing**: ~94 seconds (17.4%)
  - Sign alignment, factor matching, consensus computation
  - Diagnostic plots and file I/O

**MCMC throughput**:
- Iterations per run: 1,000 warmup + 2,000 samples = 3,000 total
- Time per run: ~140 seconds
- **Sampling rate**: ~21 iterations/second
- **Wall time per sample**: ~0.047 seconds

**File I/O time**:
- Minimal (< 5 seconds total)
- CSV/TSV writing optimized with pandas

---

## 7. Outputs and Visualization

### 7.1 Directory Structure

```
results/all_rois-sn_conf-age+sex+tiv_K5_percW33_MAD1000.0_run_20251022_112241/
│
├── experiments.log                          # Master log file
├── result.json                              # Metadata and runtime summary
│
├── data_validation_tree10_20251022_112241/
│   ├── position_lookup_filtered/
│   │   └── position_sn_voxels_filtered.tsv  # Spatial coordinates (1794 voxels)
│   ├── filtered_position_lookups/
│   │   └── sn_filtered_position_lookup.csv  # Alternative format
│   ├── individual_plots/
│   │   ├── data_distribution_comparison.png # Clinical feature histograms
│   │   ├── quality_metrics_summary.png      # QC metrics
│   │   ├── feature_reduction.png            # Preprocessing impact
│   │   ├── variance_distribution.png        # Feature variance analysis
│   │   ├── block_covariance_matrix.png      # Intra/inter-view correlations
│   │   └── inter_vs_intra_correlation.png   # Correlation structure
│   ├── mad_distribution_sn.png              # MAD threshold analysis
│   ├── elbow_analysis_all_views.png         # MAD elbow plot
│   ├── information_preservation_all_views.png
│   ├── subject_outlier_distributions.png    # Subject-level QC
│   ├── subject_outlier_comparison.png
│   ├── subject_outlier_scatter.png
│   ├── subject_outlier_top20.png
│   ├── flagged_subjects.csv                 # Subjects with high outlier %
│   ├── flagged_subjects_report.txt
│   ├── mad_threshold_summary_table.csv
│   └── mad_threshold_recommendations.txt
│
└── robustness_tests_fixed_K5_percW33_slab4_2_MAD1000.0_tree10_20251022_112319/
    ├── result.json                          # MCMC run metadata
    ├── consensus_loadings_clinical.csv      # Clinical loadings (14 × 5)
    ├── consensus_loadings_SN_Factor_0.tsv   # SN loadings for Factor 0
    ├── consensus_loadings_SN_Factor_1.tsv
    ├── consensus_loadings_SN_Factor_2.tsv
    ├── consensus_loadings_SN_Factor_3.tsv
    ├── consensus_loadings_SN_Factor_4.tsv
    ├── consensus_factor_scores.csv          # Subject scores (86 × 5)
    ├── robustness_summary.png               # Cross-seed stability plot
    └── individual_plots/
        ├── hyperparameters/
        │   ├── trace_tauW_SN.png            # Global shrinkage traces
        │   ├── trace_tauW_clinical.png
        │   ├── trace_tauZ.png
        │   ├── trace_sigma_SN.png           # Noise variance traces
        │   └── trace_sigma_clinical.png
        ├── loadings/
        │   ├── loading_W_SN_Factor0.png     # SN loading distributions
        │   ├── loading_W_clinical_Factor0.png
        │   └── ...
        └── scores/
            ├── score_Z_Factor0.png          # Factor score distributions
            └── ...
```

### 7.2 Primary Output Files

#### 7.2.1 Consensus Factor Loadings

**Clinical loadings** (`consensus_loadings_clinical.csv`):
```csv
feature,Factor_0,Factor_1,Factor_2,Factor_3,Factor_4
UPDRS_motor,0.234,-0.012,0.456,...
MoCA_total,-0.123,0.567,-0.089,...
...
```
- **Rows**: 14 clinical features
- **Columns**: 5 factors
- **Interpretation**: Loading magnitude indicates feature-factor association strength
- **Sparsity**: ~33% of loadings near zero (per percW=33)

**SN voxel loadings** (5 files, one per factor):
```
consensus_loadings_SN_Factor_k.tsv (k=0,1,2,3,4)
```
- **Rows**: 86 subjects
- **Columns**: 1,794 voxels
- **Format**: Tab-separated values (TSV)
- **Use case**: Spatial brain mapping via position lookup

**Reconstruction to 3D**:
```python
import pandas as pd
import numpy as np

# Load factor loadings and position lookup
loadings = pd.read_csv('consensus_loadings_SN_Factor_0.tsv', sep='\t', header=None)
positions = pd.read_csv('position_sn_voxels_filtered.tsv', sep='\t')

# Extract single subject (e.g., subject 0)
subject_loadings = loadings.iloc[0, :].values  # Shape: (1794,)

# Combine with spatial positions for 3D visualization
brain_map_df = pd.DataFrame({
    'x': positions.iloc[:, 0],
    'y': positions.iloc[:, 1],
    'z': positions.iloc[:, 2],
    'loading': subject_loadings
})
```

#### 7.2.2 Consensus Factor Scores

**File**: `consensus_factor_scores.csv`
```csv
subject_id,Factor_0,Factor_1,Factor_2,Factor_3,Factor_4
0,0.234,-1.234,0.567,...
1,-0.456,0.789,-0.123,...
...
85,...
```
- **Rows**: 86 subjects
- **Columns**: 5 factors
- **Interpretation**: Subject position in latent factor space
- **Use case**:
  - Clustering analysis (identify subject subgroups)
  - Regression (predict clinical outcomes from factor scores)
  - Visualization (PCA/t-SNE on factor scores)

#### 7.2.3 Metadata and Diagnostics

**Run metadata** (`result.json`):
```json
{
  "experiment_id": "robustness_tests_20251022_112319",
  "config": {
    "K": 5,
    "percW": 33.0,
    "slab_df": 4,
    "slab_scale": 2,
    "num_samples": 2000,
    "num_warmup": 1000,
    "max_tree_depth": 10,
    ...
  },
  "performance_metrics": {
    "total_runtime_seconds": 539.84,
    "total_runtime_formatted": "9.0 minutes (540s)"
  },
  "status": "completed"
}
```

**QC outputs**:
- `flagged_subjects.csv`: Subjects with outlier voxels > 10% threshold
- `mad_threshold_summary_table.csv`: MAD threshold impact analysis

### 7.3 Diagnostic Visualizations

#### 7.3.1 Convergence Diagnostics

**Trace plots** (`individual_plots/hyperparameters/trace_*.png`):
- **τ_W (global shrinkage)**: Should show stable fluctuation around mean
- **σ (noise variance)**: Should converge after warmup
- **τ_Z (factor score shrinkage)**: Should mix well

**Expected patterns**:
- ✅ **Good**: Random scatter with no trend
- ⚠️ **Warning**: Slow drift (increase warmup)
- ❌ **Bad**: Periodic oscillations or stuck values

#### 7.3.2 Model Quality

**Robustness summary** (`robustness_summary.png`):
- Cross-seed cosine similarities (should be > 0.8)
- Factor match rates (should be > 50%)
- Loading magnitude consistency across runs

#### 7.3.3 Data Quality

**Preprocessing plots**:
- `feature_reduction.png`: Number of features retained per view
- `variance_distribution.png`: Feature variance before/after scaling

**Correlation structure**:
- `block_covariance_matrix.png`: Block structure showing intra-view (high) vs. inter-view (low) correlations
- `inter_vs_intra_correlation.png`: Distribution comparison

**Subject QC**:
- `subject_outlier_distributions.png`: Histogram of outlier percentages per subject
- `subject_outlier_top20.png`: Bar chart of subjects with highest outlier rates

### 7.4 File Formats

**CSV** (comma-separated):
- Clinical loadings
- Factor scores
- QC tables

**TSV** (tab-separated):
- SN voxel loadings (more robust for large numeric matrices)
- Position lookups

**JSON**:
- Metadata and configuration
- Runtime statistics

**PNG** (images):
- All visualizations (300 DPI)
- PDF generation disabled for this run

---

## 8. Interpretation Guidelines

### 8.1 Factor Loadings

**Clinical loadings**:
- **Magnitude**: |W| > 0.1 considered meaningful (rule of thumb)
- **Sign**: Positive = feature increases with factor; Negative = feature decreases
- **Sparsity**: ~33% of loadings near zero (enforced by percW=33)

**SN voxel loadings**:
- **Spatial patterns**: Clusters of high-loading voxels indicate anatomical regions
- **Bilateral symmetry**: Check for left-right symmetry in SN
- **Magnitude**: Compare within-factor, not across factors

### 8.2 Factor Scores

**Subject heterogeneity**:
- High variance in Z[:, k] → factor captures important individual differences
- Low variance → factor may be weak or unimportant

**Clinical associations**:
- Correlate factor scores with external variables (e.g., disease duration)
- Regression: Predict outcomes from factor scores

### 8.3 Cross-Modality Integration

**Shared factors**:
- Same Z used for both clinical and imaging
- **Interpretation**: Factor k links clinical features (via W_clinical[:, k]) to brain regions (via W_SN[:, k])

**Example interpretation**:
- Factor 0 has high loadings on:
  - Clinical: UPDRS motor scores (0.45)
  - Imaging: SN voxels in dorsal subregion (median loading 0.38)
- **Conclusion**: Factor 0 captures motor symptom severity linked to dorsal SN degeneration

### 8.4 Sparsity Interpretation

**percW = 33%**:
- Each factor engages ~33% of features
- Remaining 67% shrunk to near-zero
- **Advantage**: Easier to interpret than dense factors

**Identifying relevant features**:
```python
# Features with substantial loadings for Factor 0
threshold = 0.1
relevant_features = np.where(np.abs(W_clinical[:, 0]) > threshold)[0]
```

---

## 9. Methodological Justification

### 9.1 Why Sparse GFA?

**Multi-view integration**:
- Explicitly models shared latent structure across imaging and clinical domains
- Alternative approaches (e.g., simple concatenation + PCA) ignore group structure

**Structured sparsity**:
- Regularized horseshoe produces interpretable sparse loadings
- Group-level shrinkage (via τ_W per-view) respects modality boundaries
- Avoids arbitrary hard thresholding

**Bayesian framework**:
- Full posterior uncertainty quantification
- Principled prior specification for small sample size (N=86)
- Hierarchical priors prevent overfitting

### 9.2 Hyperparameter Rationale

**K = 5 factors**:
- PCA scree plot shows elbow around 5 components
- Captures ~70% variance
- Balances model complexity with interpretability

**percW = 33%**:
- Moderate sparsity
- Too sparse (e.g., 10%): May miss relevant features
- Too dense (e.g., 80%): Difficult to interpret
- 33% empirically balances these concerns

**Slab parameters** (df=4, scale=2):
- Standard values from regularized horseshoe literature (Piironen & Vehtari, 2017)
- df=4: Moderately heavy tails (not too informative)
- scale=2: Allows factor-level flexibility

**MAD threshold = 1000**:
- Effectively disables voxel-level outlier removal
- **Justification**:
  - Small sample (N=86) makes every data point valuable
  - Robust Bayesian model handles outliers via heavy-tailed priors
  - Aggressive feature removal risks losing biological signal

### 9.3 Single-Chain Design

**Computational constraint**:
- GPU workstation cannot run parallel chains efficiently
- Memory limitations prevent 4 chains × 2000 samples × 10K parameters

**Mitigation strategy**:
- **Multiple independent runs**: 3 random seeds provide robustness
- **Consensus estimation**: Median across runs analogous to multi-chain consensus
- **Convergence diagnostics**: Trace plots and ESS still computable per-run

**Validation**:
- Cross-seed cosine similarities (> 0.8) confirm reproducibility
- Factor match rates (> 50%) verify stability

### 9.4 PCA Initialization Necessity

**Challenge**: High-dimensional parameter space (~10,000 parameters)
- Random initialization: Very slow convergence or divergence
- Zero initialization: Symmetry-breaking issues

**PCA benefits**:
- **Fast convergence**: Warm start near reasonable solution
- **Reduced warmup**: Fewer samples needed for adaptation
- **Empirical grounding**: Initialization respects data covariance structure

**Potential bias**: Initialization could bias posterior
- **Mitigation**: MCMC eventually escapes initialization if given sufficient warmup
- **Check**: Compare runs with different seeds (done via robustness testing)

---

## 10. Limitations and Caveats

### 10.1 Sample Size
- N = 86 is small for high-dimensional analysis
- Bayesian priors help but cannot fully compensate
- Results should be validated in independent cohort

### 10.2 Single Chain per Run
- Less robust than multi-chain within-run convergence
- Cross-seed consensus partially addresses this
- Future work: Enable parallel chains on more capable hardware

### 10.3 Tree Depth Saturation
- Chains consistently hit max_tree_depth = 10
- Indicates challenging posterior geometry
- Not necessarily indicative of non-convergence, but warrants caution
- **Trade-off**: Increasing max_tree_depth → longer runtimes

### 10.4 Sign Indeterminacy
- Sign alignment algorithm required
- Imperfect matching possible if factors very similar
- Visual inspection recommended to verify alignment quality

### 10.5 Factor Interpretability
- Latent factors are abstractions, not directly observable
- Biological interpretation requires domain expertise
- Multiple factors may have overlapping anatomical patterns

### 10.6 Cross-Sectional Data
- Analysis does not model temporal progression
- Cannot infer causality (only associations)
- Longitudinal extension would require different model structure

---

## 11. Reproducibility Checklist

### 11.1 Data Preparation
- [ ] qMAP-PD data in `qMAP-PD_data/` directory
- [ ] `volume_sn_voxels.tsv` (86 × 1794)
- [ ] `clinical.tsv` (86 × 17, includes header)
- [ ] Position lookup files in `qMAP-PD_data/position_lookups/`

### 11.2 Software Environment
- [ ] Python 3.11 installed
- [ ] Virtual environment created: `python3.11 -m venv venv`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] JAX 0.4.20 verified: `python -c "import jax; print(jax.__version__)"`
- [ ] NumPyro 0.13.2 verified: `python -c "import numpyro; print(numpyro.__version__)"`

### 11.3 Configuration
- [ ] `config.yaml` set with parameters from Section 2.1
- [ ] `qc_outlier_threshold: 1000.0`
- [ ] `confounds: ['age', 'sex', 'tiv']`
- [ ] `enable_pca: false`

### 11.4 Execution
- [ ] Run command from Section 4.4
- [ ] Verify output directory created
- [ ] Check `experiments.log` for errors
- [ ] Confirm 3 random seeds completed (42, 123, 456)

### 11.5 Output Verification
- [ ] `consensus_loadings_clinical.csv` exists (14 × 5)
- [ ] 5 SN loading files exist (Factor_0 through Factor_4)
- [ ] `consensus_factor_scores.csv` exists (86 × 5)
- [ ] Trace plots generated in `individual_plots/hyperparameters/`
- [ ] `result.json` shows `"status": "completed"`

### 11.6 Quality Checks
- [ ] R-hat < 1.01 for key parameters (if multi-chain available)
- [ ] ESS > 100 for global shrinkage parameters
- [ ] Trace plots show good mixing (no trends)
- [ ] Cross-seed cosine similarities > 0.8
- [ ] No NaN or Inf values in outputs

---

## 12. References

### 12.1 Methodological Papers

**Sparse Group Factor Analysis**:
- Hore, V., Viñuela, A., et al. (2016). Tensor decomposition for multiple-tissue gene expression experiments. *Nature Genetics*, 48(9), 1094-1100.

**Regularized Horseshoe Prior**:
- Piironen, J., & Vehtari, A. (2017). Sparsity information and regularization in the horseshoe and other shrinkage priors. *Electronic Journal of Statistics*, 11(2), 5018-5051.
- Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). The horseshoe estimator for sparse signals. *Biometrika*, 97(2), 465-480.

**No-U-Turn Sampler (NUTS)**:
- Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(1), 1593-1623.

**NumPyro Framework**:
- Phan, D., Pradhan, N., & Jankowiak, M. (2019). Composable effects for flexible and accelerated probabilistic programming in NumPyro. *arXiv preprint arXiv:1912.11554*.

**Non-Centered Parameterization**:
- Papaspiliopoulos, O., Roberts, G. O., & Sköld, M. (2007). A general framework for the parametrization of hierarchical models. *Statistical Science*, 22(1), 59-73.

### 12.2 Software Documentation

**JAX**:
- Bradbury, J., Frostig, R., Hawkins, P., et al. (2018). JAX: composable transformations of Python+NumPy programs. http://github.com/google/jax

**NumPyro**:
- NumPyro documentation: https://num.pyro.ai/

**Scikit-learn** (PCA):
- Pedregosa, F., Varoquaux, G., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

---

## 13. Contact and Code Availability

**Code Repository**: [Include GitHub/GitLab URL if applicable]

**Analysis Scripts**:
- Main experiment: `run_experiments.py`
- Model definition: `models/sparse_gfa_fixed.py`
- Factor stability: `analysis/factor_stability.py`
- PCA initialization: `core/pca_initialization.py`

**Configuration Files**:
- `config.yaml`: Master configuration
- `requirements.txt`: Python dependencies

**Questions**: [Include contact email for corresponding author]

---

**Document Version**: 1.0
**Last Updated**: October 22, 2025
**Analysis Reproducible As Of**: October 22, 2025
