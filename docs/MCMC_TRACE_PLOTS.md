# MCMC Trace Plot Diagnostics

This document describes the MCMC trace plot functionality added to diagnose convergence issues in factor stability analysis.

## Overview

Trace plots are critical for assessing MCMC convergence. They visualize how chains explore the parameter space over iterations and help diagnose issues like:

- **Lack of convergence**: Chains exploring different regions
- **Poor mixing**: High autocorrelation between samples
- **Multimodality**: Chains stuck in different modes
- **Burn-in issues**: Insufficient warmup period

## Features

The `analysis/mcmc_diagnostics.py` module provides comprehensive MCMC diagnostics:

### 1. Trace Diagnostic Plots (`plot_trace_diagnostics`)

A multi-panel figure showing:

**Row 1: W (Factor Loadings) Traces**
- Trace plots for each factor showing mean loading across features
- Each chain shown in different color
- Helps identify if chains are converging to same values

**Row 2: Z (Factor Scores) Traces**
- Trace plots for each factor showing mean scores across subjects
- Visualizes chain agreement on factor scores

**Row 3: R-hat Evolution**
- How R-hat changes as more samples are collected
- Shows if more sampling would improve convergence
- Includes thresholds: 1.01 (excellent), 1.1 (acceptable)
- Log scale for better visualization of large R-hat values

**Row 4: Autocorrelation & ESS**
- Autocorrelation plots showing chain mixing quality
- Effective Sample Size (ESS) per factor
- Lower autocorrelation = better mixing = higher ESS

### 2. Parameter Distribution Plots (`plot_parameter_distributions`)

- Marginal posterior distributions for W and Z
- Overlaid across all chains
- Divergent distributions indicate non-convergence
- Helps identify multimodality issues

### 3. Hyperparameter Posterior Plots (`plot_hyperparameter_posteriors`) ⭐ NEW

**Critical for assessing prior specification!**

Shows posterior distributions of `tauW` (per view) and `tauZ` (per factor):

**What these parameters mean:**
- **tauW**: Precision (inverse variance) for factor loadings per view
  - Small tauW → Strong shrinkage → Sparse loadings
  - Large tauW → Weak shrinkage → Dense loadings
- **tauZ**: Precision for factor scores
  - Similar interpretation as tauW

**Key insights from these plots:**
- **Posteriors pushed against 0**: Prior is forcing excessive sparsity
- **Posteriors shifted away from 0**: Model learning appropriate regularization
- **Different tauW per view**: Different data views need different sparsity levels
- **Chains disagree on hyperparameters**: Convergence issues at hyperparameter level

**When to relax priors:**
- If all chains show tauW/tauZ concentrated near 0, priors may be too restrictive
- If posteriors look truncated at boundaries, consider wider prior support
- If you want less shrinkage, adjust the prior scale parameter

### 4. Hyperparameter Trace Plots (`plot_hyperparameter_traces`) ⭐ NEW

Trace plots specifically for tauW and tauZ hyperparameters:
- Shows convergence behavior of regularization parameters
- Helps diagnose if hyperparameters are well-identified
- Separated traces suggest multimodality in hyperparameter space

## Usage

### Automatic Integration

Trace plots are automatically generated during factor stability analysis when running:

```python
from experiments.robustness_testing import RobustnessTestingExperiment

experiment = RobustnessTestingExperiment(config)
result = experiment.run_factor_stability_analysis(
    X_list=X_list,
    hypers=hypers,
    args=mcmc_args,
    n_chains=4,
)
```

The plots will be saved in:

```text
results/{experiment_name}/03_factor_stability/plots/
  - mcmc_trace_diagnostics.png
  - mcmc_parameter_distributions.png
  - hyperparameter_posteriors.png
  - hyperparameter_traces.png
```

### Manual Usage

For custom analysis:

```python
from analysis.mcmc_diagnostics import (
    plot_trace_diagnostics,
    plot_parameter_distributions,
    plot_hyperparameter_posteriors,
    plot_hyperparameter_traces,
    compute_rhat,
    compute_ess,
)

# W_samples: (n_chains, n_samples, D, K)
# Z_samples: (n_chains, n_samples, N, K)
# samples_by_chain: List of dicts with full MCMC samples per chain

# Generate trace diagnostics
fig = plot_trace_diagnostics(
    W_samples=W_samples,
    Z_samples=Z_samples,
    save_path="trace_diagnostics.png",
    max_factors=4,  # Plot first 4 factors
    thin=10,  # Plot every 10th sample for readability
)

# Generate distribution plots
fig = plot_parameter_distributions(
    W_samples=W_samples,
    Z_samples=Z_samples,
    save_path="distributions.png",
    max_factors=4,
)

# Generate hyperparameter posterior plots
fig = plot_hyperparameter_posteriors(
    samples_by_chain=samples_by_chain,
    save_path="hyperparameter_posteriors.png",
    num_sources=2,  # Number of data views
)

# Generate hyperparameter trace plots
fig = plot_hyperparameter_traces(
    samples_by_chain=samples_by_chain,
    save_path="hyperparameter_traces.png",
    num_sources=2,
    thin=10,
)

# Compute diagnostics manually
rhat = compute_rhat(W_samples)
ess = compute_ess(W_samples)
print(f"R-hat: {rhat:.4f}")
print(f"ESS: {ess:.1f}")
```

## Interpreting Results

### R-hat (Gelman-Rubin Statistic)

- **R-hat < 1.01**: Excellent convergence
- **R-hat < 1.1**: Acceptable convergence
- **R-hat > 1.1**: Lack of convergence - need more samples or different approach

**What it measures**: Ratio of between-chain to within-chain variance. When chains converge, this ratio approaches 1.

### Effective Sample Size (ESS)

- **ESS ≈ n_samples**: Excellent mixing, independent samples
- **ESS > 0.1 × n_samples**: Acceptable
- **ESS < 0.1 × n_samples**: Poor mixing, high autocorrelation

**What it measures**: Number of "effectively independent" samples accounting for autocorrelation.

### Trace Plot Patterns

**Good convergence:**
```
Chain 0: ~~~~~~~~~~~~~~~
Chain 1: ~~~~~~~~~~~~~~~  ← All chains overlapping
Chain 2: ~~~~~~~~~~~~~~~
Chain 3: ~~~~~~~~~~~~~~~
```

**Poor convergence:**
```
Chain 0: _______________
Chain 1: ---------------  ← Chains separated
Chain 2: ~~~~~~~~~~~~~~~
Chain 3: ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
```

**Slow mixing (high autocorrelation):**
```
Chain: /\/\/\/\/\/\/\/\  ← Smooth, slow changes
```

**Good mixing (low autocorrelation):**
```
Chain: /‾\_/‾\‾/_/\_/‾\  ← Rapid, random fluctuations
```

### Autocorrelation Plots

- **Lag 0**: Always 1.0
- **Lag 1-5**: Should drop quickly
- **Lag > 20**: Should be near 0

High autocorrelation at large lags indicates:
- Parameters changing slowly
- Need more samples between iterations (thinning)
- Need better step size tuning

### Distribution Plots

**Converged chains:**
- All chain distributions overlap
- Single mode visible

**Non-converged chains:**
- Distributions separated
- Multiple modes suggest chains exploring different regions

### Hyperparameter Posteriors (tauW, tauZ)

**What to look for:**

1. **Posterior near 0 (strong shrinkage):**
   - Model learning sparse representation
   - Could be appropriate for high-dimensional data
   - If all chains are at ~0, priors may be too restrictive

2. **Posterior shifted from 0 (moderate shrinkage):**
   - Healthy - model learning optimal regularization
   - Different values per view indicate different sparsity needs

3. **Chains disagree on hyperparameters:**
   - Convergence problem at hyperparameter level
   - May indicate model misspecification
   - Check hyperparameter trace plots

4. **Truncated distributions:**
   - Hitting prior boundaries
   - Consider relaxing prior constraints

**Example interpretations:**

```text
tauW1 (View 1): Posterior centered at 0.1
→ View 1 needs strong shrinkage (sparse loadings)

tauW2 (View 2): Posterior centered at 0.8
→ View 2 needs weak shrinkage (dense loadings)

tauZ: Chains at different values
→ Convergence issue! Check trace plots
```

**When to relax priors:**

- All tauW/tauZ hugging zero across chains → Increase prior scale
- Want less sparsity → Use wider Cauchy scale parameter
- Different views need different constraints → Use view-specific priors

## Example: Diagnosing Your Results

Based on your recent run output:

```
R-hat (W): max=12.23, mean=5.47
R-hat (Z): max=15.34, mean=4.48
Stable factors: 0/2 (0%)
```

**Diagnosis:**

1. **Severe convergence failure**: R-hat >> 1.1
2. **Chains exploring different modes**: Zero stable factors
3. **Need trace plots** to understand:
   - Are chains stuck in different regions?
   - Is there insufficient burn-in?
   - Are step sizes too large/small?

**Look for in trace plots:**
- Separated traces → Different modes (multimodality)
- Drifting traces → Insufficient burn-in
- Noisy but overlapping → Just need more samples
- Smooth waves → Poor mixing (autocorrelation issue)

## Recommended Actions Based on Diagnostics

### If R-hat is high but traces are parallel:
- Increase warmup iterations
- Chains haven't reached stationary distribution yet

### If traces are separated:
- Multimodal posterior
- Consider stronger priors to regularize
- May need more informative initialization

### If autocorrelation is high:
- Adjust step size (usually decrease)
- Use thinning when extracting samples
- Consider different MCMC kernel

### If ESS is low:
- Increase number of samples
- Improve mixing with step size tuning
- Check for label switching in factors

## Testing

Test the trace plot functionality:

```bash
python test_trace_plots.py
```

This generates example plots for both converged and non-converged chains.

## References

- Vehtari et al. 2021: "Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC"
- Gelman & Rubin 1992: "Inference from Iterative Simulation Using Multiple Sequences"
- Brooks & Gelman 1998: "General Methods for Monitoring Convergence of Iterative Simulations"

## API Reference

### `plot_trace_diagnostics`

```python
def plot_trace_diagnostics(
    W_samples: np.ndarray,      # (n_chains, n_samples, D, K)
    Z_samples: np.ndarray,      # (n_chains, n_samples, N, K)
    save_path: Optional[str] = None,
    max_factors: int = 4,
    thin: int = 1,
) -> plt.Figure
```

### `plot_parameter_distributions`

```python
def plot_parameter_distributions(
    W_samples: np.ndarray,      # (n_chains, n_samples, D, K)
    Z_samples: np.ndarray,      # (n_chains, n_samples, N, K)
    save_path: Optional[str] = None,
    max_factors: int = 4,
) -> plt.Figure
```

### `compute_rhat`

```python
def compute_rhat(chains: np.ndarray) -> float
```

Returns maximum R-hat across all parameters (most conservative).

### `compute_ess`

```python
def compute_ess(samples: np.ndarray, axis: int = 0) -> float
```

Returns mean ESS across all parameters.

### `compute_autocorrelation`

```python
def compute_autocorrelation(
    samples: np.ndarray,
    max_lag: Optional[int] = None
) -> np.ndarray
```

Returns autocorrelation function up to max_lag.

### `compute_rhat_evolution`

```python
def compute_rhat_evolution(
    chains: np.ndarray,
    window_sizes: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray]
```

Returns (sample_points, rhat_values) showing how R-hat changes over time.
