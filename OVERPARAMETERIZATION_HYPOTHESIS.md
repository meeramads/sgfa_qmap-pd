# Overparameterization Hypothesis Testing

## Research Question

**Does overparameterization (high K) lead to stable factors emerging, even when appropriate K fails to converge?**

## Motivation

You observed that sometimes high K models produce stable factors despite convergence issues. This raises two competing hypotheses:

### Hypothesis 1: Real Phenomenon ✅
**Overparameterization improves stability through:**
1. **Better posterior exploration**: More parameters → richer geometry → easier mixing
2. **Automatic model selection**: ARD/sparsity shrinks unused factors
3. **Reduced multimodality**: Extra factors can "absorb" spurious modes
4. **Blessing of dimensionality**: High-dim space paradoxically easier to explore

### Hypothesis 2: Measurement Artifact ❌
**Stability is artifactual because:**
1. **Easier matching**: With 50 factors, random chance increases factor matches
2. **Lower threshold**: Cosine similarity easier to exceed by chance
3. **Selection bias**: Only analyzing "effective" factors (self-fulfilling)
4. **R-hat misleading**: Individual factors converge but combinations don't

## Test Design

We'll run 3 parallel experiments with identical settings except K:

| Config | K | Purpose | Expected Outcome |
|--------|---|---------|------------------|
| **config_K2_no_sparsity.yaml** | 2 | Baseline (minimal) | If fails: underparameterization |
| **config_K20_no_sparsity.yaml** | 20 | Moderate overparam | Test sweet spot |
| **config_K50_no_sparsity.yaml** | 50 | Extreme overparam | Test blessing/curse |

### Fixed Parameters (All Configs)
- Model: `standard_gfa` (ARD priors, NO horseshoe)
- Chains: 4 independent chains
- Samples: 1000 warmup + 2000 sampling
- NUTS: `max_tree_depth=13`
- Data: Same MAD-filtered voxels (qc_outlier_threshold=4.0)

## Predictions

### If Hypothesis 1 is True (Real Phenomenon)

**Expected Results:**

| Metric | K=2 | K=20 | K=50 |
|--------|-----|------|------|
| **R-hat** | High (>2) | Medium (1.1-1.5) | Low (<1.2) |
| **Effective Factors** | 2/2 (100%) | 5-8/20 (25-40%) | 3-10/50 (6-20%) |
| **Stable Factors** | 0/2 (0%) | 3-5 (>50% of effective) | 3-5 (most effective) |
| **Match Rate** | <30% | 60-80% | >80% |
| **Computation Time** | Baseline | 2-3x | 5-8x |

**Key Signature**:
- K=2 fails → K=20 improves → K=50 converges
- Stable factors are subset of effective factors
- R-hat improves with K
- Similar stable factors across K=20 and K=50

### If Hypothesis 2 is True (Measurement Artifact)

**Expected Results:**

| Metric | K=2 | K=20 | K=50 |
|--------|-----|------|------|
| **R-hat** | High (>2) | High (>2) | High (>2) |
| **Effective Factors** | 2/2 | 15-18/20 | 35-45/50 |
| **Stable Factors** | 0/2 | 5-10 (random) | 15-25 (random) |
| **Match Rate** | <30% | 40-60% | 50-70% |
| **Computation Time** | Baseline | 2-3x | 5-8x |

**Key Signature**:
- R-hat doesn't improve with K
- Many "effective" factors (ARD not shrinking)
- Stable factor count scales with K (not constant)
- Stable factors differ between K=20 and K=50

## Metrics to Analyze

### 1. Convergence Metrics

**R-hat (Gelman-Rubin)**
```python
# Per-factor R-hat
rhat_per_factor = [compute_rhat(W[:, :, :, k]) for k in range(K)]

# Overall convergence
rhat_max = np.max(rhat_per_factor)
rhat_mean = np.mean(rhat_per_factor)
```

**Interpretation:**
- R-hat < 1.01: Excellent convergence
- R-hat < 1.1: Acceptable convergence
- R-hat > 1.1: Convergence failure

**If H1**: R-hat improves with K
**If H2**: R-hat stays high across K

### 2. Effective Factor Count

**ARD Shrinkage** (alpha hyperparameters)
```python
# Per chain, count factors with large alpha (not shrunk)
alpha_median = np.median(alpha_samples, axis=0)  # (M, K)
effective = np.sum(alpha_median < threshold)  # Small alpha = large variance = active
```

**Interpretation:**
- Effective = factor has non-negligible variance
- ARD shrinks factors by increasing alpha (precision)
- Alpha >> 1 → factor shrunk to zero

**If H1**: Effective factors ~3-10 regardless of K (ARD selects)
**If H2**: Effective factors scales with K (ARD not working)

### 3. Factor Stability

**Cosine Similarity Matching**
```python
# Match factors across chains
for k in range(K):
    factor_k = W_chain0[:, k]  # Reference factor
    similarities = [cosine_similarity(factor_k, W_chain_j[:, k])
                    for j in range(1, n_chains)]
    is_stable = np.mean(similarities > threshold) > min_match_rate
```

**Interpretation:**
- Stable = factor matches across >50% of chains
- Cosine similarity > 0.8 = similar loading pattern
- Stable factors should be subset of effective factors

**If H1**: Stable factors are effective factors, count constant across K
**If H2**: Stable factor count increases with K (artifact)

### 4. Stability Rate vs K

**Critical Test:**
```python
stability_rate_K2 = n_stable_K2 / K_2
stability_rate_K20 = n_stable_K20 / K_20
stability_rate_K50 = n_stable_K50 / K_50
```

**If H1 (Real)**: Stability rate decreases with K
- K=2: 0% (0/2)
- K=20: 25% (5/20)
- K=50: 10% (5/50)
- Same ~5 factors stable across configs

**If H2 (Artifact)**: Stability rate constant or increases
- K=2: 0% (0/2)
- K=20: 50% (10/20)
- K=50: 50% (25/50)
- Different stable factors across configs

## Analysis Plan

### Step 1: Individual Analysis
For each K={2, 20, 50}:
1. Check R-hat convergence diagnostics
2. Count effective factors per chain (ARD analysis)
3. Identify stable factors (cross-chain matching)
4. Examine trace plots for separation

### Step 2: Comparative Analysis

**A. Convergence vs K**
```python
plt.plot([2, 20, 50], [rhat_K2, rhat_K20, rhat_K50])
plt.axhline(1.1, linestyle='--', label='Acceptable')
plt.xlabel('K (number of factors)')
plt.ylabel('Max R-hat')
```

**B. Effective Factors vs K**
```python
plt.bar([2, 20, 50], [eff_K2, eff_K20, eff_K50])
plt.plot([2, 20, 50], [2, 20, 50], 'r--', label='All factors')
plt.xlabel('K (number of factors)')
plt.ylabel('Effective factors')
```

**C. Stability Rate vs K**
```python
stability_rates = [stable/K for stable, K in zip([s2, s20, s50], [2, 20, 50])]
plt.plot([2, 20, 50], stability_rates, 'o-')
plt.xlabel('K (number of factors)')
plt.ylabel('Stability rate (stable/K)')
```

### Step 3: Hypothesis Test

**Critical Questions:**

1. **Does R-hat improve with K?**
   - Yes → Evidence for H1 (better exploration)
   - No → Evidence for H2 (artifact)

2. **Do effective factors saturate?**
   - Yes (plateau at ~5-10) → H1 (ARD selecting)
   - No (scales with K) → H2 (ARD failing)

3. **Are stable factors consistent across K?**
   - Yes (same ~5 factors) → H1 (real structure)
   - No (different each K) → H2 (random matching)

4. **Does stability rate decrease with K?**
   - Yes → H1 (fixed numerator, growing denominator)
   - No → H2 (both scale proportionally)

## Expected Timeline

| Config | K | Est. Runtime | Priority |
|--------|---|--------------|----------|
| K=2 | 2 | 4-6 hours | High (baseline) |
| K=20 | 20 | 8-12 hours | High (key test) |
| K=50 | 50 | 16-24 hours | Medium (extreme) |

**Total**: ~2-3 days for all three runs (sequential)

**Recommendation**: Start K=2 and K=20 first, then decide if K=50 needed based on results.

## Interpreting Results

### Scenario 1: H1 Confirmed (Real Phenomenon)

**Signature:**
- ✅ R-hat: 5.0 → 1.3 → 1.05 (improves with K)
- ✅ Effective: 2 → 6 → 5 (saturates, ARD working)
- ✅ Stable: 0 → 4 → 4 (consistent across K=20, K=50)
- ✅ Same 4 factors stable in both K=20 and K=50

**Conclusion**: Overparameterization helps! Use K=20 or K=30 in future.

**Next Steps**:
1. Use K=20 as standard
2. Analyze the ~4-6 stable factors biologically
3. Try K=20 with horseshoe priors (relaxed)
4. Publication angle: "Blessing of overparameterization for factor stability"

### Scenario 2: H2 Confirmed (Measurement Artifact)

**Signature:**
- ❌ R-hat: 5.0 → 4.8 → 4.5 (no improvement)
- ❌ Effective: 2 → 18 → 45 (scales with K, ARD not working)
- ❌ Stable: 0 → 10 → 25 (scales proportionally)
- ❌ Different stable factors in K=20 vs K=50

**Conclusion**: "Stable factors" are artifacts of measurement.

**Next Steps**:
1. Fix the real convergence problem (not K)
2. Use stricter stability criteria
3. Try different initialization strategies
4. Consider reparameterization
5. May need informative priors

### Scenario 3: Mixed Results (Partial Support)

**Signature:**
- ⚠️ R-hat: 5.0 → 1.8 → 1.5 (improves but not enough)
- ⚠️ Effective: 2 → 10 → 12 (partial saturation)
- ⚠️ Stable: 0 → 5 → 6 (mostly consistent)
- ⚠️ Overlap in stable factors but not complete

**Conclusion**: Overparameterization helps but insufficient.

**Next Steps**:
1. Combine: K=20 + relaxed priors + more MCMC samples
2. Try K=30 (middle ground)
3. Investigate which factors are consistently stable
4. Check if instability is factor label switching

## Key Insights from Literature

### Supporting H1 (Overparameterization Helps)

**Bhattacharya & Dunson (2011)**: "Sparse Bayesian infinite factor models"
- Overparameterization with sparsity priors improves model selection
- ARD effectively selects dimensionality

**Gao et al. (2020)**: "High-dimensional Bayesian inference"
- Blessing of dimensionality in some posterior geometries
- More parameters can improve HMC mixing

**Ghosh & Dunson (2009)**: "Default prior distributions"
- Automatic relevance determination robust to K misspecification
- Overspecified models shrink to true dimensionality

### Supporting H2 (Beware Artifacts)

**Stephens (2000)**: "Dealing with label switching"
- Multimodality increases with K
- Factor matching can find spurious similarities

**Minka (2000)**: "Automatic choice of dimensionality"
- High K can lead to overfitting without proper priors
- Need evidence lower bound, not just stability

**Ročková & George (2016)**: "Fast spike and slab"
- Model selection needs explicit penalty
- ARD alone may not be sufficient

## Statistical Power Considerations

### Sample Size (N=86 subjects)

**Rule of thumb**: Need N > 5K for stable factor recovery

- K=2: N=86, ratio=43 → ✅ Well-powered
- K=20: N=86, ratio=4.3 → ⚠️ Borderline
- K=50: N=86, ratio=1.7 → ❌ Under-powered

**Implication**: K=50 may be too high for N=86, even with ARD.

### Features (D≈21,000 voxels)

**Rule of thumb**: High D favors sparse methods

- Ratio D/N ≈ 244 → Extreme high-dimensional regime
- Sparsity essential for identifiability
- ARD may struggle without additional structure

### Practical Bounds

Based on N=86:
- **Conservative K**: 2-5 (N/K > 17)
- **Moderate K**: 10-15 (N/K > 5)
- **Aggressive K**: 20-30 (N/K > 3)
- **Extreme K**: 50+ (N/K < 2, likely issues)

**Prediction**: K=20 may be optimal, K=50 may be too high.

## Code Snippets for Analysis

### Extract Effective Factor Counts

```python
import json
import numpy as np

# Load alpha hyperparameters (ARD precisions)
for k_val in [2, 20, 50]:
    results_dir = f"results/robustness_testing_K{k_val}_*/03_factor_stability/"

    for chain_id in range(4):
        # Load chain results
        samples = load_chain_samples(f"{results_dir}/chains/chain_{chain_id}/")

        # Alpha hyperparameters (shape: n_samples, M, K)
        alpha = samples["alpha"]
        alpha_median = np.median(alpha, axis=0)  # (M, K)

        # Effective factors: small alpha = large variance = active
        # Threshold: alpha < 1.0 (arbitrary, adjust as needed)
        effective_per_view = np.sum(alpha_median < 1.0, axis=1)
        total_effective = np.sum(alpha_median < 1.0)

        print(f"K={k_val}, Chain {chain_id}: {total_effective}/{k_val} effective")
```

### Compare Stable Factors Across K

```python
# Load stability results
stability_K2 = load_json("results/.../K2/stability_results.json")
stability_K20 = load_json("results/.../K20/stability_results.json")
stability_K50 = load_json("results/.../K50/stability_results.json")

# Extract stable factor indices
stable_K2 = stability_K2["stable_factor_indices"]
stable_K20 = stability_K20["stable_factor_indices"]
stable_K50 = stability_K50["stable_factor_indices"]

print(f"K=2:  {len(stable_K2)}/2 stable ({100*len(stable_K2)/2:.1f}%)")
print(f"K=20: {len(stable_K20)}/20 stable ({100*len(stable_K20)/20:.1f}%)")
print(f"K=50: {len(stable_K50)}/50 stable ({100*len(stable_K50)/50:.1f}%)")

# Check if stable factors are consistent
# (requires loading actual factor loadings and comparing)
```

## Conclusion

This experiment will definitively answer whether overparameterization-induced stability is:
- **Real**: A useful property for factor discovery (use K=20-30)
- **Artifact**: A measurement issue (stick to K=2-5, fix convergence)

The three configs are ready to run. Start with K=2 and K=20, then decide on K=50 based on those results.
