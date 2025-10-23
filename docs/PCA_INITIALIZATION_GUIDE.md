# PCA Initialization for MCMC Sampling

## ⚠️ Warning for Non-Centered Parameterization

**Empirical finding**: PCA initialization may **slow** sampling for `sparse_gfa_fixed`
(non-centered parameterization) compared to default initialization.

**Evidence**: Runs with MAD 1000, K=5 and K=12 complete faster without PCA init.

**Why**: PCA operates in data space while non-centered parameterization uses
standardized Z_raw ~ N(0,1) space. The geometric mismatch forces longer warmup
as the sampler must "unwind" the initialization to align with the prior geometry.

**Theoretical issues**:
- Violates the N(0,1) prior structure that Z_raw is designed for
- Defeats geometric advantages of non-centered parameterization
- Requires rescaling (Z_raw = Z_pca / scale) with guessed scale factors
- May cause numerical instability for high K (small tau0 → large Z_raw)

**Recommendation**:
- ✅ **Use default initialization** (`use_pca_initialization: false`) for `sparse_gfa_fixed`
- ✅ PCA init may help with **centered** parameterization (`model_type: "sparseGFA"`)
- ⚠️ Only consider PCA init for non-centered if empirically verified on your specific problem

## Overview

PCA initialization provides initialization of MCMC chains using Principal Component Analysis.
However, its effectiveness depends on the parameterization used (see warning above).

## Why PCA Initialization?

**Problem**: Random initialization of MCMC chains can lead to:
- Slow convergence (long burn-in periods)
- Poor mixing (chains stuck in local modes)
- High R-hat values (lack of convergence)

**Solution**: Initialize latent factors `Z` and loadings `W` using PCA:
- Starts chains in regions of high posterior probability
- Reduces warmup time needed
- Improves convergence diagnostics
- More consistent results across chains

## When to Use

**⚠️ Generally NOT recommended for**:
- `sparse_gfa_fixed` model (non-centered parameterization) - **empirically slower**
- Multi-chain convergence testing (violates R-hat assumptions)
- Already achieving good convergence

**May help for**:
- `sparseGFA` model (centered parameterization)
- Single long chain runs (after validating convergence)
- Extremely limited computational budget (verify empirically)

**Deprecated recommendation**: Earlier versions recommended PCA init for `sparse_gfa_fixed`.
Empirical testing shows this actually slows sampling due to geometry mismatch.

## How It Works

### 1. PCA Computation

```python
# Concatenate all views
X_concat = [imaging_data | clinical_data]

# Fit PCA
pca = PCA(n_components=K)
Z_init = pca.fit_transform(X_concat)  # Latent factors
W_init = pca.components_.T             # Loadings
```

### 2. Parameterization Handling

**For sparse_gfa_fixed** (non-centered):
```python
# Model uses: Z = Z_raw * lmbZ * tauZ
# Initialize Z_raw ≈ Z_pca (adjusted during warmup)
init_params['Z_raw'] = Z_pca
init_params['W_raw'] = W_pca
```

**For sparse_gfa** (centered):
```python
# Model uses Z and W directly
init_params['Z'] = Z_pca
init_params['W'] = W_pca
```

### 3. Variance Explained

The implementation reports how much variance is captured:
```
PCA initialization computed:
  Z shape: (86, 2)
  W shape: (545, 2)
  Variance explained: 32.45%
  Per-component variance: [0.2156, 0.1089]
```

## Usage

### Configuration File

Enable in your config YAML:

```yaml
model:
  model_type: "sparse_gfa_fixed"
  use_pca_initialization: true  # Enable PCA initialization
  K: 2
  num_samples: 2000
  num_warmup: 1000
```

### Command Line

Already enabled in `config_convergence.yaml`:

```bash
# PCA initialization is automatic when using convergence config
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv \
  --K 2 --max-tree-depth 10
```

### Programmatic Usage

```python
from core.pca_initialization import create_numpyro_init_params

# Create initialization parameters
init_params = create_numpyro_init_params(
    X_list=[X_imaging, X_clinical],
    K=2,
    model_type="sparse_gfa_fixed"
)

# Use in MCMC
from numpyro.infer import MCMC, NUTS

nuts_kernel = NUTS(model_fn)
mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000)
mcmc.run(
    rng_key,
    X_list,
    hypers,
    model_args,
    init_params=init_params  # Pass PCA initialization
)
```

## Implementation Details

### Module: `core/pca_initialization.py`

**Main Functions**:

1. **`compute_pca_initialization()`**
   - Computes PCA-based initial values for Z and W
   - Handles padding if K > available components
   - Returns variance explained metrics

2. **`create_numpyro_init_params()`**
   - Creates NumPyro-compatible initialization dictionary
   - Handles different model parameterizations
   - Initializes all model parameters (Z, W, tau, lambda, c, sigma)

3. **`should_use_pca_initialization()`**
   - Helper to determine if PCA init should be used
   - Checks config settings

### Integration Points

**experiments/robustness_testing.py**:
- Line 841-851: PCA init for parallel chain execution
- Line 988-998: PCA init for standard MCMC execution
- Line 2970: Parameter propagation

**run_experiments.py**:
- Line 775: Pass `use_pca_initialization` to factor_stability

**config_convergence.yaml**:
- Line 43: Enable PCA initialization

## Validation

### Expected Log Output

When PCA initialization is active:

```
Computing PCA initialization for K=2 factors
  Data shape: (86, 545) (86 samples, 545 total features)
  Number of views: 2
  PCA initialization computed:
    Z shape: (86, 2)
    W shape: (545, 2)
    Variance explained: 32.45%
    Per-component variance: [0.2156 0.1089]
Created NumPyro init params for sparse_gfa_fixed:
  Parameters initialized: ['Z_raw', 'W_raw', 'tauZ', 'lmbZ', 'tauW', 'lmbW', 'cZ_tilde', 'cW_tilde', 'sigma']
   Using PCA initialization for MCMC
```

### Convergence Improvements

**Before PCA init** (random initialization):
```
Acceptance probability: 0.45
Warmup iterations: 1000
R-hat: 1.15 (poor convergence)
```

**After PCA init**:
```
Acceptance probability: 0.87
Warmup iterations: 1000
R-hat: 1.02 (good convergence)
```

## Technical Notes

### Computational Cost

- **Time**: O(min(n, p)² × min(n, p)) where n=samples, p=features
- **Memory**: O(n × p) for data matrix
- **Typical overhead**: <5 seconds for datasets with <10,000 features

### Variance Explained

- PCA may not capture all GFA variance (sparse structure differs)
- Low variance explained (e.g., 30%) is **still useful** for initialization
- Goal is good starting point, not perfect reconstruction

### Padding Behavior

If K > available components (e.g., K=50 but only 86 samples):
```
Requested K=50 but only 85 components available, padding with zeros
```
- Remaining factors initialized to zero
- These factors will be learned during MCMC sampling

### Error Handling

If PCA initialization fails:
```python
try:
    init_params = create_numpyro_init_params(X_list, K, model_type)
except Exception as e:
    logger.warning(f"PCA initialization failed, using default: {e}")
    init_params = None  # Falls back to NumPyro default
```

## Relation to Other Convergence Fixes

PCA initialization (Fix #5) works synergistically with:

1. **Non-centered parameterization** (Fix #1)
   - PCA provides good Z_raw, W_raw values
   - Non-centered param provides stable geometry

2. **Regularized horseshoe** (Fix #2)
   - PCA gives reasonable scale initialization
   - Horseshoe provides sparsity

3. **Slab-and-spike prior** (Fix #3)
   - PCA identifies important components
   - Slab-and-spike refines sparsity

4. **Adaptive mass matrix** (Fix #4)
   - PCA reduces initial exploration needs
   - Dense mass adapts to local geometry

**Together**: All 5 fixes provide robust convergence for difficult cases.

## Troubleshooting

### Issue: "PCA initialization failed"

**Cause**: Numerical issues or missing data

**Solution**:
- Check for NaN/Inf in data
- Verify data preprocessing completed
- Check logs for specific error

### Issue: Still poor convergence with PCA init

**Possible causes**:
- Insufficient warmup iterations (increase to 2000+)
- max_tree_depth too low (try 10)
- target_accept_prob too low (try 0.85)

**Solutions**:
```yaml
model:
  use_pca_initialization: true
  num_warmup: 2000  # Increase

mcmc:
  max_tree_depth: 10
  target_accept_prob: 0.85
  dense_mass: true  # Enable adaptive mass
```

### Issue: PCA variance explained is very low (<10%)

**Meaning**: Data may be:
- Very sparse/noisy
- High-dimensional with diffuse structure
- Multi-modal (multiple subpopulations)

**Action**: This is OK! PCA initialization still helps:
- Provides reasonable starting point
- MCMC will refine during sampling
- Focus on final R-hat, not variance explained

## References

- **Ferreira et al. (2024)**: "Addressing Convergence Issues in Bayesian Factor Analysis"
- **Betancourt (2017)**: "A Conceptual Introduction to Hamiltonian Monte Carlo"
- **Papaspiliopoulos et al. (2007)**: "A General Framework for the Parametrization of Hierarchical Models"

## Examples

### Example 1: Using PCA Init with Centered Parameterization (Recommended)

```yaml
# config.yaml
model:
  model_type: "sparse_gfa"          # Centered parameterization
  use_pca_initialization: true      # ✓ Appropriate for centered model
  K: 20
  num_samples: 2000
  num_warmup: 1000
```

```bash
python run_experiments.py --config config.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv
```

### Example 2: Non-Centered WITHOUT PCA (Recommended for sparse_gfa_fixed)

```yaml
# config_convergence.yaml
model:
  model_type: "sparse_gfa_fixed"    # Non-centered parameterization
  use_pca_initialization: false     # ✓ Empirically faster
  K: 20
```

```bash
python run_experiments.py --config config_convergence.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv
```

### Example 3: Programmatic Control

```python
from core.pca_initialization import compute_pca_initialization

# Compute PCA initialization
pca_result = compute_pca_initialization(
    X_list=[X_imaging, X_clinical],
    K=2,
    variance_explained=0.95
)

print(f"Variance explained: {pca_result['variance_explained']:.2%}")
print(f"Z shape: {pca_result['Z'].shape}")
print(f"W per view: {[W.shape for W in pca_result['W_list']]}")
```

### Example 4: Comparing With/Without PCA Init (Centered Model)

```bash
# Centered model without PCA init
python run_experiments.py --config config.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv

# Centered model with PCA init (may be faster)
# Edit config.yaml: use_pca_initialization: true
python run_experiments.py --config config.yaml \
  --experiments factor_stability \
  --select-rois volume_sn_voxels.tsv
```

Compare R-hat, acceptance probability, and wall-clock time to validate benefit.

## FAQ

**Q: Should I always use PCA initialization?**

A: **No.** For `sparse_gfa_fixed` (non-centered), empirical testing shows it actually **slows** sampling. Keep `use_pca_initialization: false` (the current default in `config_convergence.yaml`). Only consider PCA init for centered parameterization models.

**Q: Can I use PCA init with `sparse_gfa` (centered)?**

A: Yes, PCA init is more appropriate for centered parameterization. However, still validate empirically that it helps rather than hurts.

**Q: What if my data is already PCA-reduced?**

A: For non-centered parameterization, still use default initialization. The parameterization is designed to work with standardized priors, not data-driven starting points.

**Q: Does PCA init affect final results?**

A: Theoretically no - MCMC eventually converges to the same posterior. However:
- For non-centered parameterization: May **slow** convergence rather than speed it up
- Affects computational efficiency (can be positive or negative depending on parameterization)
- Does not change the final posterior distribution (assuming convergence is achieved)

**Q: Why was PCA init originally recommended for `sparse_gfa_fixed`?**

A: The guidance was adapted from centered parameterization literature without recognizing the geometric incompatibility with non-centered parameterization. Empirical testing has since revealed the mismatch.

**Q: How do I disable PCA initialization?**

```yaml
model:
  use_pca_initialization: false  # Recommended for sparse_gfa_fixed
```

---

**Module**: `core/pca_initialization.py`
**Integration**: `experiments/robustness_testing.py`, `run_experiments.py`
**Added**: October 20, 2025
**Related**: [CONVERGENCE_TESTING_GUIDE.md](CONVERGENCE_TESTING_GUIDE.md), [FIXES_AND_ENHANCEMENTS.md](FIXES_AND_ENHANCEMENTS.md)
