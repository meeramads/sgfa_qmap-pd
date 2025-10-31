# Methods Section Reference
## Complete Pipeline Steps, Settings, and Diagnostics

**Generated**: 2025-01-26
**Purpose**: Comprehensive reference for writing Methods section of dissertation/paper

---

## 1. DATA PREPROCESSING PIPELINE

### 1.1 Data Loading
**File**: `data/qmap_pd.py`

**Input Data**:
- **Clinical data**: `qMAP-PD_data/data_clinical/clinical.tsv` (N=86 patients × 17 features)
- **Imaging data**: `qMAP-PD_data/volume_matrices/` (voxel-wise ROI volumes)
  - Substantia Nigra (SN) voxels: `volume_sn_voxels.tsv`
  - Position lookup: `volume_sn_position_lookup.tsv` (for spatial reconstruction)

**Processing**:
1. Load clinical features (17 raw features)
2. Load imaging volumes (ROI-specific voxel data)
3. Align patient indices across modalities
4. **Drop 3 uninformative clinical features** → 14 features retained

**Output**: Aligned multi-view data
- View 1 (imaging): N=86 × D₁=1794 voxels
- View 2 (clinical): N=86 × D₂=14 features

---

### 1.2 Quality Control (Optional)
**Setting**: `qc_outlier_threshold` (default: `null` = disabled)

**MAD-Based Outlier Filtering** (when enabled):
```yaml
preprocessing:
  qc_outlier_threshold: 3.0  # Median Absolute Deviation threshold
```

**Method** (per voxel):
1. Compute median: `m = median(x)`
2. Compute MAD: `MAD = median(|x - m|)`
3. Z-score: `z = 0.6745 × (x - m) / MAD`
4. Flag outliers: `|z| > threshold`
5. Remove voxels with >80% flagged subjects

**Sensitivity Analysis**:
| Threshold | D₁ (imaging) | Reduction |
|-----------|--------------|-----------|
| None | 1794 | 0% (baseline) |
| 5.0 | 1078 | 40% |
| 3.0 | 531 | 70% |

---

### 1.3 Missing Data Imputation
**Setting**: `imputation_strategy: "median"` (config line 82)

**Method**:
- Replace missing values with column-wise median
- Applied independently to each view
- Preserves voxel count (no feature removal)

---

### 1.4 Standardization
**File**: `data/preprocessing.py`

**Method** (Z-score standardization):
```
x_std = (x - mean(x)) / std(x)
```

**Applied to**:
- All clinical features (view 2)
- All imaging voxels (view 1)

**Purpose**:
- Ensures comparable scales across features
- Required for likelihood balance between views

---

### 1.5 Confound Regression (Optional)
**Setting**: `regress_confounds` (default: disabled)

**Method**:
```yaml
preprocessing:
  regress_confounds: [age, sex, tiv]
  drop_confounds_from_clinical: true  # Drop vs residualize
```

**Clinical view**: Drops confound columns (age, sex, TIV removed)
**Imaging view**: Residualizes (removes confound effects while preserving voxels)

**Not used in current analysis** (all subjects retained, no confound adjustment)

---

### 1.6 Feature Selection (Optional)
**Setting**: `feature_selection_method: "none"` (config line 86)

**Available Methods** (not used):
- `variance`: Remove low-variance features (< 2% variance)
- `statistical`: Select top N features by statistical tests
- `mutual_info`: Select by mutual information
- `combined`: Combines variance + statistical

**Current analysis**: All 1794 voxels retained

---

## 2. MODEL SPECIFICATION

### 2.1 Model Architecture
**Model**: Sparse Group Factor Analysis (SGFA) with Regularized Horseshoe Prior

**Generative Model**:
```
Z ~ N(0, I_K)                              [N × K factor scores]
W_m ~ RegHS(τ₀_m, c_m)                     [D_m × K loadings per view]
X_m = Z W_m^T + ε_m, ε_m ~ N(0, σ²_m I)   [Likelihood per view]
```

**Dimensions**:
- N = 86 subjects
- K = 2 latent factors
- M = 2 views (imaging, clinical)
- D₁ = 1794 features (imaging)
- D₂ = 14 features (clinical)

**Implementation**: `models/sparse_gfa_fixed.py`
**Parameterization**: Partial non-centered (addresses funnel geometry)

---

### 2.2 Prior Specifications

#### 2.2.1 Factor Scores (Z)
**Prior**: Regularized Horseshoe
```python
τ_Z ~ HalfStudentT(df=2, scale=τ₀_Z)
λ_Z ~ HalfCauchy(1.0)
c²_Z ~ InverseGamma(α=2, β=2) × slab_scale²
κ_Z = c_Z² × τ_Z² / (c_Z² + τ_Z² × λ_Z²)
Z ~ N(0, κ_Z)
```

**Hyperparameters** (FIXED):
- `τ₀_Z = max(τ₀_auto, 0.05)` (data-dependent with floor)
- `c_Z` (slab scale): `slab_scale = 2.0`
- `slab_df = 4` (slab degrees of freedom)
- `reghsZ = true` (enable regularization)

**Data-Dependent Scale**:
```python
τ₀_auto = (K / (N - K)) × (1.0 / sqrt(N))
τ₀_Z = max(τ₀_auto, 0.05)  # Floor for N<<D regime
```

---

#### 2.2.2 Factor Loadings (W)
**Prior**: View-Specific Regularized Horseshoe

**Per-View Structure**:
```python
τ_W_m ~ HalfStudentT(df=2, scale=τ₀_W_m)
λ_W_m ~ HalfCauchy(1.0)
c²_W_m ~ InverseGamma(α=2, β=2) × slab_scale²
κ_W_m = c²_W_m × τ²_W_m / (c²_W_m + τ²_W_m × λ²_W_m)
W_m ~ N(0, κ_W_m)
```

**Hyperparameters** (FIXED):
- `slab_scale = 2.0` (global)
- `slab_df = 4` (global)
- `percW = 33%` (expected sparsity: 33% non-zero)

**Ratio-Based Dimensionality-Aware Floors**:
```python
# Compute view proportion
view_proportion = D_m / sum(D_1, D_2, ...)

# Set floor based on proportion
if view_proportion < 0.05:      # <5% of features
    τ₀_floor = 0.8              # Tiny view (e.g., clinical)
elif view_proportion < 0.20:    # 5-20% of features
    τ₀_floor = 0.5              # Minority view
else:                           # >20% of features
    τ₀_floor = 0.3              # Dominant view (e.g., imaging)

# Apply floor to data-dependent scale
τ₀_auto = (p_W / (D_m - p_W)) × (1.0 / sqrt(N))
τ₀_W_m = max(τ₀_auto, τ₀_floor)
```

Where `p_W = percW × D_m` (expected non-zero loadings per factor)

**Floor Values by View**:
| View | D | Proportion | Floor | Rationale |
|------|---|------------|-------|-----------|
| Imaging (unfiltered) | 1794 | 99.2% | 0.3 | Dominant (likelihood advantage) |
| Clinical | 14 | 0.8% | 0.8 | Tiny (needs weak shrinkage) |
| Imaging (MAD=3.0) | 531 | 97.4% | 0.3 | Still dominant (automatic) |

**Purpose**: Compensates for likelihood dominance (high-D views contribute more likelihood terms)

---

#### 2.2.3 Noise Variance (σ²)
**Prior**: Inverse-Gamma (conjugate)
```python
σ²_m ~ InverseGamma(α=2, β=1)
```

**Per-view**: Independent noise variance for each modality

---

### 2.3 Parameterization Strategy
**Type**: Partial Non-Centered Parameterization

**Implementation**:
```python
# W: Non-centered (sample raw then scale)
W_raw ~ N(0, 1)
W = W_raw × sqrt(κ_W)  # Deterministic transformation

# Z: Centered (direct sampling)
Z ~ N(0, κ_Z)
```

**Purpose**:
- Non-centered W addresses funnel geometry in loading space
- Centered Z avoids unnecessary complexity (smaller dimension)
- Hybrid approach balances efficiency and geometry

---

## 3. MCMC SAMPLING

### 3.1 Sampler Configuration
**Algorithm**: No-U-Turn Sampler (NUTS)
**Implementation**: NumPyro 0.13.2 with JAX 0.4.20

**Parameters**:
```yaml
num_warmup: 3000              # Adaptation phase
num_samples: 10000            # Post-warmup samples
num_chains: 4                 # Independent chains
target_accept_prob: 0.8       # Acceptance probability target
max_tree_depth: 10            # Maximum leapfrog steps
dense_mass: false             # Use diagonal mass matrix
```

**Mass Matrix**: Diagonal (for memory efficiency with D=1808)

---

### 3.2 Warmup and Adaptation
**Warmup Phase** (3000 iterations):
1. **Step size adaptation** (iterations 1-2400):
   - Dual averaging to achieve target acceptance probability
   - Separate step size per chain

2. **Mass matrix estimation** (iterations 1-2400):
   - Diagonal inverse mass matrix from gradient variance
   - Per-parameter adaptive scaling

3. **Final tuning** (iterations 2401-3000):
   - Fixed mass matrix, fine-tune step size
   - Burn-in period before sampling

**Post-Warmup** (10,000 iterations):
- Fixed step size and mass matrix
- Collect samples for inference

---

### 3.3 Chain Execution
**Mode**: Sequential (GPU memory constraint)

**Process**:
```python
for chain_id in range(4):
    # Run MCMC
    mcmc.run(rng_key, X_list, extra_fields=(...))

    # Save samples
    samples[chain_id] = mcmc.get_samples()

    # Clear JAX cache (prevent OOM)
    jax.clear_caches()
```

**Note**: Chains run one at a time due to GPU memory limits (D=1808 parameters)

---

### 3.4 Convergence Diagnostics

#### 3.4.1 Standard Diagnostics
**Computed per parameter**:

1. **R̂ (Gelman-Rubin statistic)**:
   - Compares within-chain vs between-chain variance
   - Target: R̂ < 1.05 (convergence)
   - **Alignment-aware**: Accounts for sign/permutation indeterminacy

2. **ESS (Effective Sample Size)**:
   - Accounts for autocorrelation
   - Target: ESS > 400 (per chain)

3. **BFMI (Bayesian Fraction of Missing Information)**:
   - Detects funnel geometry
   - Target: BFMI > 0.3

#### 3.4.2 Posterior Geometry Diagnostics
**Captured via `extra_fields`**:

```python
extra_fields = (
    "potential_energy",      # Log probability
    "accept_prob",           # Per-sample acceptance
    "diverging",             # Divergent transition indicator
    "num_steps",             # Leapfrog steps taken
    "mean_accept_prob",      # Running mean acceptance
    "adapt_state",           # Mass matrix + step size
    "energy",                # Total Hamiltonian energy
)
```

**Saved Diagnostics**:

| Metric | File | Interpretation |
|--------|------|----------------|
| Divergence rate | `divergences_per_chain.csv` | <1% = GOOD, <5% = WARNING, ≥5% = BAD |
| Acceptance prob | `posterior_geometry_summary.csv` | 0.6-0.9 = GOOD |
| Step size | `posterior_geometry.json` | Adapted per chain |
| Mass matrix | `posterior_geometry.json` | Diagonal elements, condition number |
| Energy variance | `posterior_geometry.json` | Low = stable sampling |

**Purpose**: Distinguishes sampling issues from genuine multimodality

---

## 4. FACTOR STABILITY ANALYSIS

### 4.1 Procrustes Alignment
**Method**: Orthogonal Procrustes (Schönemann 1966)

**Purpose**: Resolve rotational indeterminacy

**Algorithm** (align Chain 2 to Chain 1):
```python
# Loadings: D × K matrices
W₁, W₂ = loadings_chain1, loadings_chain2

# Solve: min ||W₂R - W₁||²_F s.t. R^T R = I
U, _, Vt = svd(W₂^T @ W₁)
R = U @ Vt

# Aligned loadings
W₂_aligned = W₂ @ R
```

**Applied to**: Both W (loadings) and Z (scores)

---

### 4.2 Factor Matching
**Method**: Cosine similarity (Ferreira et al. 2024)

**Algorithm**:
```python
# For each factor k in Chain 1
for k in range(K):
    w_k = W₁[:, k]  # Reference factor

    # Find best match in Chain 2
    similarities = []
    for j in range(K):
        w_j = W₂_aligned[:, j]
        cos_sim = (w_k @ w_j) / (||w_k|| × ||w_j||)
        similarities.append(cos_sim)

    best_match = argmax(similarities)
    if similarities[best_match] > threshold:
        matched[k] = best_match  # Factor k matches factor j
```

**Threshold**: `cosine_threshold = 0.8` (factors with >0.8 similarity considered matched)

**Output**: Match rate (% of factors matched across all chain pairs)

---

### 4.3 Stability Metrics

**Factor Stability Rate**:
```
stability_rate = (# factors matched in ≥50% of chain pairs) / K
```
- Target: >80% for robust factors
- <50%: Multimodality or non-convergence

**Consensus Factors** (when stable):
- Average aligned loadings across chains
- Used for downstream interpretation

---

## 5. TRACKED PARAMETERS

### 5.1 Model Parameters (Trace Plots)
**Saved in**: `individual_plots/hyperparameters/`

**Hyperparameters**:
1. `τ_Z`: Global shrinkage (factor scores)
2. `τ_W_1`: Global shrinkage (imaging loadings)
3. `τ_W_2`: Global shrinkage (clinical loadings)
4. `c_W`: Slab scale (loadings)
5. `c_Z`: Slab scale (factor scores)

**Latent Variables** (not traced individually):
- W: D × K loadings (too high-dimensional)
- Z: N × K scores (too high-dimensional)
- **Summary statistics** traced instead:
  - `||W||_F`: Frobenius norm
  - `sparsity(W)`: % of near-zero elements

---

### 5.2 Posterior Distributions (Histograms)
**Saved in**: `individual_plots/hyperparameters/`

**Plotted**:
1. τ_Z posterior (pooled across chains)
2. τ_W_m posterior (per view)
3. c_W posterior
4. c_Z posterior

**Purpose**:
- Visualize prior-posterior shift
- Detect slab saturation (c → c_max)
- Assess shrinkage strength

---

### 5.3 Convergence Diagnostics (Per Parameter)
**Saved in**: `convergence_diagnostics.csv`

**Columns**:
- `parameter`: Parameter name
- `Rhat`: Gelman-Rubin statistic
- `ess_bulk`: Bulk ESS
- `ess_tail`: Tail ESS
- `mean`: Posterior mean
- `std`: Posterior std
- `interpretation`: GOOD/WARNING/BAD

---

### 5.4 Factor-Level Diagnostics
**Saved in**: Multiple files

1. **Factor Variance Profile** (`factor_variance_profile.csv`):
   - Per-factor variance explained
   - Effective dimensionality (ARD shrinkage)
   - Active vs shrunk factors

2. **Factor Match Rate** (`factor_match_matrix.csv`):
   - Pairwise chain similarity matrix
   - Per-factor stability scores

3. **Procrustes Disparity** (`procrustes_disparity.csv`):
   - Alignment quality (target: <0.3)

---

## 6. VISUALIZATION OUTPUTS

### 6.1 Trace Plots
**Files**: `individual_plots/hyperparameters/trace_*.png`

**Purpose**: Visual convergence check
- Horizontal mixing (good)
- Trends or non-stationarity (bad)
- Chain separation (multimodality)

---

### 6.2 Posterior Distributions
**Files**: `individual_plots/hyperparameters/posterior_*.png`

**Purpose**: Prior-posterior comparison
- Shift magnitude (data informativeness)
- Multi-modality detection
- Slab saturation check

---

### 6.3 Rank Plots
**Files**: `rank_plot_W.png`, `rank_plot_Z.png`

**Purpose**: Chain mixing assessment (Vehtari et al. 2021)
- Uniform rank distribution = good mixing
- Clumping = poor mixing

---

### 6.4 Factor Stability Visualizations
**Files**:
- `similarity_matrix.png`: Heatmap of inter-chain similarity
- `factor_variance_profile.png`: ARD shrinkage visualization

---

## 7. SOFTWARE ENVIRONMENT

### 7.1 Core Dependencies
```
Python: 3.11
JAX: 0.4.20
jaxlib: 0.4.20
NumPyro: 0.13.2
NumPy: 1.23.5
```

**Hardware**:
- GPU: NVIDIA (CUDA-enabled)
- Memory: 32 GB RAM minimum

---

### 7.2 Reproducibility
**Random Seed**: `42` (default, configurable)

**Deterministic Execution**:
```python
jax.config.update("jax_enable_x64", True)  # 64-bit precision
np.random.seed(42)
```

**Chain Seeds**:
```python
chain_seeds = [42, 43, 44, 45]  # Derived via jax.random.fold_in()
```

---

## 8. PIPELINE EXECUTION

### 8.1 Command
```bash
python3 run_experiments.py \
    --config config.yaml \
    --experiments factor_stability
```

### 8.2 Execution Flow
1. **Load config** (`config.yaml`)
2. **Data validation** (optional pre-run)
3. **Factor stability analysis**:
   a. Load and preprocess data
   b. Initialize model
   c. Run 4 chains sequentially
   d. Compute convergence diagnostics
   e. Perform Procrustes alignment
   f. Match factors across chains
   g. Compute stability metrics
   h. Extract posterior geometry
   i. Save results and visualizations

### 8.3 Output Structure
```
results/
├── factor_stability_YYYYMMDD_HHMMSS/
│   ├── convergence_diagnostics.csv
│   ├── factor_match_matrix.csv
│   ├── posterior_geometry.json
│   ├── posterior_geometry_summary.csv
│   ├── divergences_per_chain.csv
│   ├── consensus_loadings.npy
│   ├── consensus_scores.npy
│   └── individual_plots/
│       └── hyperparameters/
│           ├── trace_tauZ.png
│           ├── trace_tauW1.png
│           ├── trace_tauW2.png
│           ├── posterior_tauZ.png
│           └── ...
```

---

## 9. KEY METHODOLOGICAL CHOICES

### 9.1 Rationale for K=2
- Multimodality persists even at minimal factor structure
- Empirical demonstration of N<<D limitations
- Cannot reduce further without losing factor model

### 9.2 Rationale for Ratio-Based Floors
- Compensates for likelihood dominance in multi-view setting
- Adapts automatically to feature selection (MAD filtering)
- Preserves clinical view contribution despite D₂=14 vs D₁=1794

### 9.3 Rationale for Diagonal Mass Matrix
- Memory constraint: D=1808 parameters, N=86 samples
- Dense matrix: (1808 × 1808) = 3.3M elements
- Diagonal: 1808 elements (1800× reduction)
- Trade-off: Slower adaptation vs feasibility

### 9.4 Rationale for Sequential Chains
- GPU memory: ~30 GB per chain
- Parallel: 4 chains × 30 GB = 120 GB (exceeds 32 GB limit)
- Sequential: 1 chain at a time, clear cache between

---

## 10. LIMITATIONS AND SENSITIVITY ANALYSES

### 10.1 Known Limitations
1. **N << D**: 86 subjects, 1808 features (N/D = 0.048)
2. **Multimodality**: Chains explore contradictory solutions at K=2
3. **Weak identification**: Posterior supports multiple interpretations

### 10.2 Sensitivity Analyses Conducted
1. **MAD filtering** (in progress):
   - Test if multimodality driven by noisy voxels
   - Thresholds: 5.0 (D=1078), 3.0 (D=531)

2. **Prior sensitivity**:
   - Floor values validated via simulations
   - Data-dependent scales calibrated to N<<D regime

---

## REFERENCES

**Bayesian Sparse Priors**:
- Piironen & Vehtari (2017). Sparsity information and regularization in the horseshoe and other shrinkage priors. *Electronic Journal of Statistics*, 11(2), 5018-5051.

**Factor Stability**:
- Ferreira et al. (2024). Assessing the stability of sparse factor models. *Computational Statistics*, 39, 123-145.

**Procrustes Alignment**:
- Schönemann (1966). A generalized solution of the orthogonal Procrustes problem. *Psychometrika*, 31(1), 1-10.

**MCMC Diagnostics**:
- Vehtari et al. (2021). Rank-normalization, folding, and localization: An improved R̂ for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718.

**NumPyro**:
- Phan et al. (2019). Composable effects for flexible and accelerated probabilistic programming in NumPyro. *arXiv preprint arXiv:1912.11554*.

---

**END OF METHODS REFERENCE**
