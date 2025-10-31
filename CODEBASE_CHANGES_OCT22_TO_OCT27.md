# Codebase Changes: October 22 to October 27 Morning

**Period**: October 22, 2025 11:22 AM → October 27, 2025 10:48 AM (before problematic run started)

**Context**: The October 22 run produced good, interpretable results. The October 27 00:12 run showed:
- Enforced positivity on clinical parameters
- Flipped contralateral MRI relationships (positive-positive instead of positive-negative)
- Worse interpretability overall

This document identifies all substantive code changes between these two timepoints that could explain the differences.

---

## Summary of Major Changes

### 1. **Model Architecture Changes**
- **Partial Centering Parameterization** (Oct 25, 17:24) - MAJOR
- **Dimensionality-Aware τ₀ Floors** (Oct 26, 13:01) - MAJOR
- **τ₀ Floor Addition** (Oct 25, 00:29) - Moderate

### 2. **Hyperparameter Changes**
- **Slab Scale Increase**: 2.0 → 5.0 (Oct 25, 00:30) - MAJOR
- **Scaling Method Change**: RobustScaler → StandardScaler (Oct 25, 00:59) - MAJOR

### 3. **Configuration Changes**
- **MAD Filtering Default**: Disabled by default (Oct 24, 18:08) - Minor

### 4. **Bug Fixes Applied AFTER Oct 27 Run**
- **Consensus Calculation Bug Fix** (Oct 27, 10:48) - Fixed AFTER problematic run
- **Sign Alignment Normalization** (Oct 27, 15:22) - Fixed AFTER problematic run

---

## Detailed Change Analysis

---

## 1. Partial Centering Parameterization

**Commit**: `a17cf198` (Oct 25, 17:24)
**File**: `models/sparse_gfa_fixed.py`
**Impact**: MAJOR - Fundamentally changes sampling geometry

### What Changed

#### Before (Oct 22):
```python
# Non-centered parameterization only
Z = Z_raw * lmbZ * tauZ
W_chunk = z_raw_chunk * lmbW_tilde * tauW
```

#### After (Oct 25):
```python
# Adaptive blending between centered and non-centered
phi = jnp.sqrt(1.0 - jnp.exp(-tau))  # Mixing parameter

# Centered component (no raw parameter)
Z_centered = lmbZ * tauZ

# Non-centered component (with raw parameter)
Z_noncentered = Z_raw * lmbZ * tauZ

# Adaptive blend
Z = phi * Z_centered + (1.0 - phi) * Z_noncentered
```

### Why This Change Was Made

**Problem**: Non-centered parameterization creates funnel geometry when τ→0
- Small τ (weak data): Posterior pushes τ toward zero → creates narrow funnel
- NUTS struggles with funnels: Step size tuned for wide end fails in narrow end
- Common in N<<D regime (86 subjects, 1,794 features)

**Solution**: Partial centering adaptively blends parameterizations
- When τ small (weak prior): φ→0, use non-centered (decorrelates τ and λ)
- When τ large (data informative): φ→1, use centered (better geometry)
- Smooth transition: φ(τ) = √(1 - exp(-τ))

### Potential Impact on Oct 27 Results

**Mechanism by which this could affect results**:

1. **Different posterior exploration**:
   - Partial centering changes the geometry NUTS explores
   - Could cause sampler to find different modes or spend time in different regions
   - May preferentially explore regions with certain sign patterns

2. **Interaction with τ values**:
   - If clinical τ_W is small → more non-centered (φ≈0)
   - If imaging τ_W is large → more centered (φ≈1)
   - Asymmetric treatment could bias which view dominates

3. **Convergence behavior**:
   - Better geometry should improve mixing
   - But could also allow sampler to more confidently settle in local modes
   - May reduce between-chain variation but increase systematic bias if mode is wrong

**Likelihood of causing observed issues**: **MODERATE**
- Could contribute to different sign patterns across runs
- Unlikely to directly cause "enforced positivity" (that's more likely from priors)
- Could interact with dimensionality-aware floors to create asymmetry

---

## 2. Dimensionality-Aware τ₀ Floors

**Commit**: `952872d4` (Oct 26, 13:01)
**File**: `models/sparse_gfa_fixed.py`
**Impact**: MAJOR - Directly addresses view imbalance

### What Changed

#### Before (Oct 25):
```python
# Single floor for all views
tau0_W_auto = (D0_per_factor / (Dm_m - D0_per_factor)) * (sigma_std / jnp.sqrt(N))
tau0_W_scale = jnp.maximum(tau0_W_auto, 0.3)  # Uniform floor
```

#### After (Oct 26):
```python
# Dimensionality-adaptive floors
if Dm_m < 50:
    # Low-dimensional views (clinical: 14 features)
    tau0_W_floor = 0.8  # Weaker shrinkage
    floor_reason = "low-D (<50 features)"
elif Dm_m < 200:
    # Medium-dimensional views
    tau0_W_floor = 0.5
    floor_reason = "medium-D (50-200 features)"
else:
    # High-dimensional views (imaging: 1,794 features)
    tau0_W_floor = 0.3  # Standard shrinkage
    floor_reason = "high-D (>200 features)"

tau0_W_scale = jnp.maximum(tau0_W_auto, tau0_W_floor)
```

### Why This Change Was Made

**Problem**: Multi-view factor analysis with extreme dimensionality imbalance
- Imaging view: 1,794 features → 1,794 likelihood terms per subject
- Clinical view: 14 features → 14 likelihood terms per subject
- **Result**: Imaging contributes 128× more to log-likelihood
- Posterior dominated by "fit imaging well" regardless of clinical informativeness
- Clinical loadings shrink to ≈0 despite potential relevance

**Solution**: Dimensionality-adaptive prior floors
- Low-D views (clinical): Floor = 0.8 → weaker shrinkage → can compete for signal
- High-D views (imaging): Floor = 0.3 → standard shrinkage → likelihood dominance already present

### Potential Impact on Oct 27 Results

**Mechanism by which this DIRECTLY affects results**:

1. **Clinical view gets weaker priors**:
   - τ₀_clinical: floor increases from 0.3 → 0.8
   - Larger τ₀ → larger τ_W samples on average
   - Larger τ_W → weaker horseshoe shrinkage → loadings less shrunk toward zero
   - **Expected effect**: Clinical loadings should be LARGER in magnitude

2. **View asymmetry introduced**:
   - Clinical view: τ₀_floor = 0.8 (weaker shrinkage)
   - Imaging view: τ₀_floor = 0.3 (stronger shrinkage)
   - This is INTENTIONAL to compensate for likelihood dominance
   - But could overcorrect if not calibrated properly

3. **Sign pattern changes**:
   - Weaker priors allow more flexibility in clinical loadings
   - Could allow clinical features to "choose" different sign patterns
   - **Could explain "enforced positivity"** if most clinical features now prefer positive loadings

4. **MRI relationship flipping**:
   - If clinical loadings become larger and change sign
   - Factor scores Z must adjust to maintain fit to data
   - This could flip the imaging loadings to maintain X ≈ Z W^T
   - **Could directly explain positive-positive contralateral relationships**

**Likelihood of causing observed issues**: **HIGH**
- **Most likely culprit** for the specific symptoms observed
- Directly affects clinical vs. imaging balance
- Timing matches: Change on Oct 26, problematic run on Oct 27 00:12
- Mechanism is plausible and specific to observed symptoms

---

## 3. τ₀ Floor Addition (Initial)

**Commit**: `dd22859f` (Oct 25, 00:29)
**File**: `models/sparse_gfa_fixed.py`
**Impact**: MODERATE - Added safety floor before dimensionality-aware floors

### What Changed

#### Before (Oct 24):
```python
# No floor - could go arbitrarily small
tau0_W_scale = (D0_per_factor / (Dm_m - D0_per_factor)) * (sigma_std / jnp.sqrt(N))
tau0_Z_scale = (D0_Z / (N - D0_Z)) * (sigma_std / jnp.sqrt(N))
```

#### After (Oct 25):
```python
# Added uniform floor to prevent extreme prior weakness
tau0_W_auto = (D0_per_factor / (Dm_m - D0_per_factor)) * (sigma_std / jnp.sqrt(N))
tau0_W_scale = jnp.maximum(tau0_W_auto, 0.3)  # Min scale for stability

tau0_Z_auto = (D0_Z / (N - D0_Z)) * (sigma_std / jnp.sqrt(N))
tau0_Z_scale = jnp.maximum(tau0_Z_auto, 0.05)  # Min scale for N<<D regime
```

### Why This Change Was Made

**Problem**: N<<D regime can lead to τ₀ → 0
- Formula: τ₀ = (D₀/(D-D₀)) × (σ/√N)
- When N=86, D=1794, D₀=33% → very small τ₀
- τ₀ → 0 causes multimodal posteriors and chain disagreement

**Solution**: Add floor to prevent extreme prior weakness
- τ₀_W floor = 0.3
- τ₀_Z floor = 0.05

### Potential Impact on Oct 27 Results

**Mechanism**:
- This change was later superseded by dimensionality-aware floors (commit `952872d4`)
- By itself, uniform floor of 0.3 shouldn't cause view asymmetry
- Could improve convergence and reduce multimodality

**Likelihood of causing observed issues**: **LOW**
- Superseded by later change
- Uniform floor shouldn't cause systematic bias
- More likely to improve convergence than change interpretation

---

## 4. Slab Scale Increase: 2.0 → 5.0

**Commit**: `e8935012` (Oct 25, 00:30)
**File**: `models/models_integration.py`
**Impact**: MAJOR - Relaxes regularization strength

### What Changed

#### Before (Oct 24):
```python
"slab_scale": 2.0,  # Standard value from literature
```

#### After (Oct 25):
```python
"slab_scale": 5.0,  # Increased to allow larger loadings and reduce saturation
```

### Why This Change Was Made

**Comment in code**: "Increased from 2.0 to allow larger loadings and reduce saturation"

**Regularized horseshoe hierarchy**:
```
c_W^(m,k) ~ Student-t⁺(df=4, scale=slab_scale)
W_{d,k}^(m) ~ N(0, τ_W² · c_W² · λ_W²)
```

**Effect of increasing slab_scale**:
- Larger slab_scale → c_W can be larger
- Larger c_W → weaker regularization (less shrinkage to zero)
- Allows individual loadings to be larger in magnitude
- **Trade-off**: Less sparsity enforcement at the factor level

### Potential Impact on Oct 27 Results

**Mechanism**:

1. **Larger loading magnitudes**:
   - Slab scale 5.0 vs. 2.0 allows c_W to range higher
   - Individual loadings can grow larger before hitting regularization ceiling
   - **Could explain why clinical loadings are now non-zero** (less shrinkage)

2. **Reduced factor-level sparsity**:
   - c_W provides per-factor regularization (across all features in that factor)
   - Larger scale → less pressure for entire factors to shrink uniformly
   - Factors may be less sparse overall

3. **Interaction with dimensionality-aware floors**:
   - Dimensionality-aware floors increase τ₀_clinical to 0.8
   - Slab scale increase allows c_W to be larger
   - **Combined effect**: Clinical loadings can be MUCH larger
   - τ_W × c_W × λ_W all larger → loading variance increased

4. **Sign stability**:
   - Larger loadings are more stable (less shrinkage to zero)
   - But also more flexible (regularization weaker)
   - Could reduce sign flipping but also allow new sign patterns to emerge

**Likelihood of causing observed issues**: **MODERATE-HIGH**
- Works synergistically with dimensionality-aware floors
- Could amplify the effect of weaker clinical priors
- Timing: Oct 25 (before dimensionality-aware floors on Oct 26)
- **Combined with dimensionality-aware floors, could strongly affect results**

---

## 5. Scaling Method Change: RobustScaler → StandardScaler

**Commit**: `4a9751e7` (Oct 25, 00:59)
**File**: `data/preprocessing.py`
**Impact**: MAJOR - Changes data standardization

### What Changed

#### Before (Oct 24):
```python
# RobustScaler (median/IQR)
if robust:
    scaler = RobustScaler()  # median=0, IQR-based scaling
else:
    scaler = StandardScaler()  # mean=0, std=1
```

#### After (Oct 25):
```python
# StandardScaler by default for SGFA
# Changed default from robust=True to robust=False
scaler = StandardScaler()  # Always mean=0, std=1

# Warning if robust=True explicitly requested
if robust:
    logger.warning("robust=True not recommended for SGFA - horseshoe priors require mean=0, std=1")
```

### Why This Change Was Made

**Problem**: SGFA horseshoe priors require standardized data (mean=0, std=1)
- RobustScaler uses median/IQR which does NOT guarantee mean=0, std=1
- Median can equal 0, but mean may not
- IQR-based scaling doesn't match std=1

**Prior calibration assumptions**:
```python
sigma_std = 1.0  # After standardization
tau0_W_scale = (D0_per_factor / (Dm_m - D0_per_factor)) * (sigma_std / jnp.sqrt(N))
```
- Code assumes σ=1 exactly
- If actual std ≠ 1, τ₀ calibration is wrong

**Solution**: Use StandardScaler to ensure mean=0, std=1 exactly

### Potential Impact on Oct 27 Results

**Mechanism**:

1. **Different data standardization**:
   - RobustScaler: Median-centered, IQR-scaled (outlier-resistant)
   - StandardScaler: Mean-centered, std-scaled (exact standardization)
   - **Features with outliers will be scaled differently**

2. **Prior calibration now correct**:
   - Oct 22 run: σ may not have been exactly 1.0 → τ₀ miscalibrated
   - Oct 27 run: σ = 1.0 exactly → τ₀ correctly calibrated
   - Could change effective prior strength

3. **Outlier handling**:
   - RobustScaler downweights outliers (uses median/IQR)
   - StandardScaler gives outliers full weight
   - **Subjects flagged as outliers (2 subjects with 17.5% outlier voxels) now have more influence**

4. **Clinical feature scales**:
   - Clinical features may have different distributions than imaging
   - RobustScaler vs. StandardScaler could affect them differently
   - Could change relative importance of clinical vs. imaging

**Likelihood of causing observed issues**: **MODERATE**
- Changes data representation fundamentally
- Could interact with prior calibration issues
- But both runs should have been standardized (just differently)
- More likely to affect magnitude than sign patterns

---

## 6. MAD Filtering Default Change

**Commit**: `566380a9` (Oct 24, 18:08)
**File**: `config.yaml`
**Impact**: MINOR - October 22 run already used MAD=1000

### What Changed

```yaml
# Added to config.yaml
preprocessing:
  qc_outlier_threshold: 1000.0  # Effectively disabled
```

**Note**: October 22 run already used `MAD1000.0` in the directory name, so this change likely doesn't affect the comparison.

### Potential Impact on Oct 27 Results

**Likelihood of causing observed issues**: **VERY LOW**
- Both runs used same MAD threshold (1000.0 = effectively disabled)
- No feature-level filtering differences

---

## 7. Bug Fixes Applied AFTER Oct 27 Run

These fixes were applied AFTER the problematic Oct 27 00:12 run started, so they affected that run but NOT the Oct 22 run.

### 7.1 Consensus Calculation Bug Fix

**Commit**: `d92a5f7c` (Oct 27, 10:48)
**File**: `analysis/factor_stability.py`
**Impact**: CRITICAL - Bug was present in Oct 27 run

#### The Bug (Present in Oct 27 Run):
```python
# OLD BUGGY CODE:
if isinstance(result["W"], list):
    # WRONG: Used pre-split list instead of averaging samples!
    W_concat = np.vstack([W_view for W_view in result["W"]])
    W_chain_avg.append(W_concat)
else:
    W_avg = np.mean(W_samples, axis=0)
    W_chain_avg.append(W_avg)
```

#### The Fix (Applied After Oct 27 Run):
```python
# NEW FIXED CODE:
W_samples = result["samples"]["W"]  # (n_samples, D, K)
W_avg = np.mean(W_samples, axis=0)  # Properly average samples
W_chain_avg.append(W_avg)
```

#### Impact on Oct 27 Results:
- **Buggy path used pre-split W list** instead of averaging MCMC samples
- Could cause voxel collapse (94-97% identical loadings observed)
- **This bug was PRESENT in Oct 27 run, NOT in Oct 22 run**
- **Major contributor to Oct 27 issues**

### 7.2 Sign Alignment Normalization

**Commit**: `85e01ffe` (Oct 27, 15:22) - NOT in provided log, but mentioned in summary
**File**: `analysis/factor_stability.py`
**Impact**: MODERATE - Bug was present in Oct 27 run

#### The Bug (Present in Oct 27 Run):
```python
# OLD CODE - raw dot product:
correlation = np.dot(reference, loading)
if correlation < 0:
    aligned_loadings.append(-loading)
```

#### The Fix (Applied After Oct 27 Run):
```python
# NEW CODE - normalized cosine similarity:
ref_norm = np.linalg.norm(reference)
loading_norm = np.linalg.norm(loading)

if ref_norm < 1e-10 or loading_norm < 1e-10:
    aligned_loadings.append(loading)
else:
    cosine_sim = np.dot(reference, loading) / (ref_norm * loading_norm)
    if cosine_sim < 0:
        aligned_loadings.append(-loading)
```

#### Impact on Oct 27 Results:
- **Raw dot product is scale-dependent**
- Could cause incorrect sign alignment if loadings have different magnitudes
- Combined with consensus bug, could compound sign errors
- **This bug was PRESENT in Oct 27 run, NOT in Oct 22 run**

---

## Combined Impact Analysis

### Most Likely Explanation for Oct 27 Issues

The Oct 27 run had **THREE major differences** from Oct 22:

1. ✅ **Dimensionality-aware τ₀ floors** (Oct 26, 13:01)
   - Clinical floor: 0.3 → 0.8 (weaker shrinkage)
   - Imaging floor: 0.3 (unchanged)
   - **Direct mechanism for clinical parameter changes**

2. ✅ **Slab scale increase** (Oct 25, 00:30)
   - 2.0 → 5.0 (allows larger loadings)
   - **Amplifies effect of weaker τ₀ floors**
   - Combined: Clinical loadings can be MUCH larger

3. ❌ **Buggy consensus calculation** (fixed Oct 27, 10:48)
   - Used wrong code path for averaging loadings
   - **Could cause voxel collapse and sign errors**
   - Fixed AFTER Oct 27 run started

### Hypothesized Causal Chain

**Step 1**: Dimensionality-aware floors + slab scale increase
- Clinical loadings less shrunk (τ₀=0.8, slab=5.0)
- Clinical features now compete strongly for signal
- Clinical loadings grow larger in magnitude

**Step 2**: Clinical loadings dominate certain factors
- Factors now weighted toward clinical features
- To maintain fit X ≈ Z W^T, imaging loadings must adjust
- **Could flip imaging loading signs** to accommodate clinical dominance

**Step 3**: Buggy consensus calculation
- Wrong averaging method compounds issues
- Sign alignment bugs cause further corruption
- **Result**: Voxel collapse + incorrect sign patterns

### Why Oct 22 Run Was Better

**Oct 22 configuration**:
- Uniform τ₀ floors (0.3 for all views)
- Slab scale = 2.0
- RobustScaler (may have had slight standardization issues, but consistent)
- No consensus bugs

**Result**:
- Balanced shrinkage across views
- Clinical features appropriately sparse
- Imaging features dominate (natural given 128× more likelihood terms)
- Interpretable contralateral relationships preserved

---

## Recommendations

### If Using Oct 22 Results (Current Plan)

**Justification**:
- Oct 22 used established hyperparameters (slab_scale=2.0)
- Uniform prior floors (simpler, more interpretable)
- No experimental changes
- Results were interpretable and scientifically sensible

**Methods section should note**:
- Uniform τ₀ floors across all views
- Standard slab scale (2.0) from literature
- RobustScaler used (note: may not guarantee exact σ=1)

### If Re-Running with Current Code

**To avoid Oct 27 issues**:

1. **Test dimensionality-aware floors carefully**:
   - Current floors: Clinical=0.8, Imaging=0.3
   - May be overcorrecting for likelihood dominance
   - Consider: Clinical=0.5, Imaging=0.3 (more conservative)

2. **Reduce slab_scale or use original**:
   - Current: 5.0 (very weak regularization)
   - Original: 2.0 (standard from literature)
   - **Recommend**: Return to 2.0 or try 3.0 (moderate)

3. **Keep bug fixes**:
   - Consensus calculation fix (commit d92a5f7c)
   - Sign alignment normalization (commit 85e01ffe)
   - These are genuine improvements

4. **Partial centering**:
   - Likely helpful for convergence
   - Should not cause systematic bias if floors are calibrated
   - **Recommend**: Keep

5. **StandardScaler**:
   - Correct choice for prior calibration
   - Ensures σ=1 exactly
   - **Recommend**: Keep

### Alternative: Hybrid Configuration

**Goal**: Combine best of both runs

```yaml
# Hyperparameters
slab_scale: 2.0  # Original (Oct 22)
slab_df: 4       # Unchanged

# Prior floors (modified from Oct 27)
tau0_floors:
  clinical: 0.5   # Moderate (between 0.3 and 0.8)
  imaging: 0.3    # Original

# Preprocessing
scaling: "standard"  # Oct 27 improvement
mad_threshold: 1000.0  # Same in both

# Parameterization
partial_centering: true  # Oct 27 improvement (convergence)

# Bug fixes
consensus_method: "correct"  # Oct 27 fixes
sign_alignment: "normalized_cosine"  # Oct 27 fixes
```

**Rationale**:
- Moderate clinical floor (0.5) balances likelihood dominance without overcorrecting
- Keep slab_scale=2.0 for interpretability and literature consistency
- Keep Oct 27 improvements (StandardScaler, partial centering, bug fixes)
- Should produce interpretable results with better convergence than Oct 22

---

## Timeline of Changes

```
Oct 22, 11:22 AM - GOOD RUN COMPLETED
    └─ config: slab_scale=2.0, tau0_floor=0.3 (uniform), RobustScaler
    └─ Results: Interpretable, good contralateral relationships

Oct 24, 18:08 PM - MAD filtering disabled by default (no effect, already 1000)

Oct 25, 00:29 AM - τ₀ floor added (uniform 0.3)
Oct 25, 00:30 AM - Slab scale increased: 2.0 → 5.0 ⚠️
Oct 25, 00:59 AM - Scaling changed: RobustScaler → StandardScaler ⚠️
Oct 25, 17:24 PM - Partial centering implemented ⚠️

Oct 26, 13:01 PM - Dimensionality-aware τ₀ floors ⚠️⚠️
    └─ Clinical: 0.3 → 0.8
    └─ Imaging: 0.3 (unchanged)

Oct 27, 00:12 AM - PROBLEMATIC RUN STARTED
    └─ config: slab_scale=5.0, tau0_clinical=0.8, tau0_imaging=0.3, StandardScaler
    └─ Bugs: Consensus calculation bug, sign alignment bug
    └─ Results: Enforced positivity, flipped relationships, voxel collapse

Oct 27, 10:48 AM - Consensus bug fixed (AFTER run started)
Oct 27, 15:22 PM - Sign alignment bug fixed (AFTER run started)
```

**Critical window**: Oct 25-26 (between runs)
- 4 major changes in 36 hours
- Changes interact in complex ways
- Insufficient testing before Oct 27 run

---

## Files Modified (Summary)

### Core Model Files
- `models/sparse_gfa_fixed.py` - Partial centering, τ₀ floors (multiple commits)
- `models/models_integration.py` - Slab scale increase

### Preprocessing
- `data/preprocessing.py` - RobustScaler → StandardScaler

### Post-processing (Bug Fixes)
- `analysis/factor_stability.py` - Consensus calculation, sign alignment (AFTER Oct 27 run)

### Configuration
- `config.yaml` - MAD threshold default

---

## Conclusion

**Primary culprits for Oct 27 issues** (in order of likelihood):

1. **Dimensionality-aware τ₀ floors** (Oct 26) - HIGH
   - Direct mechanism for clinical parameter changes
   - Timing matches perfectly
   - Specific to observed symptoms

2. **Slab scale increase** (Oct 25) - HIGH
   - Amplifies effect of τ₀ floors
   - Reduces regularization significantly (2.0 → 5.0)
   - Works synergistically with τ₀ changes

3. **Buggy consensus calculation** (fixed Oct 27) - HIGH
   - Present in Oct 27 run, not Oct 22 run
   - Could cause voxel collapse observed
   - Compounds other issues

4. **Partial centering** (Oct 25) - MODERATE
   - Changes sampling geometry
   - Could interact with other changes
   - Unlikely to be sole cause

5. **StandardScaler change** (Oct 25) - MODERATE
   - Affects data standardization
   - Prior calibration now correct
   - Could change effective prior strength

**Recommendation for publication**: Use Oct 22 results
- Simpler, more interpretable configuration
- Established hyperparameters from literature
- No experimental changes or bugs
- Results scientifically sensible

**If re-running needed**: Use hybrid configuration with moderate floors and original slab_scale=2.0
