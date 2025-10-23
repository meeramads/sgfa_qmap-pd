# Future Work

This document lists features and experiments that have **framework infrastructure** in place but require additional implementation work before they can be used in production.

## Model Comparison Experiments

**Status**: ⏳ **Blocked - Awaiting Suitable Comparison Metric**

### Description

Compare SGFA against traditional dimensionality reduction methods (PCA, ICA, Factor Analysis) to validate the sparse Bayesian approach.

### Framework Available

- `experiments/model_comparison.py`: Experiment orchestration
- Model implementations for PCA, ICA, FA
- Factor map export (W, Z matrices) for all methods

### Blocker: Comparison Metric

**Problem**: No suitable metric exists for comparing models in this pipeline.

**Why Standard Metrics Don't Work**:
- **Log-likelihood**: Not comparable across different model classes (Bayesian vs frequentist)
- **ELBO**: Not computed in NumPyro NUTS sampling (requires variational inference)
- **Cross-validation**: No held-out test set in current design
- **Reconstruction error**: Not meaningful without ground truth

**What's Needed**:
- Define domain-appropriate comparison metric (e.g., factor interpretability score, clinical association strength, spatial coherence)
- Or: Redesign experiment to use cross-validation framework
- Or: Implement variational inference for ELBO computation

**Estimated Effort**: 2-4 weeks (metric definition + validation)

### Files

- `experiments/model_comparison.py`
- Infrastructure complete, needs comparison metric implementation

---

## SGFA Hyperparameter Comparison

**Status**: ✅ **Implemented - Aligned R-hat Comparison Metric**

### Description

Systematic comparison of SGFA configurations using aligned R-hat convergence diagnostics:
- K (number of factors): 2, 5, 10, 15, 20
- percW (sparsity %): 10%, 25%, 33%, 50%
- slab parameters: (df=4, scale=2) vs other configurations

### Implementation Complete

**Framework**: `experiments/sgfa_configuration_comparison.py`
- ✅ Multi-chain MCMC (minimum 2 chains per variant)
- ✅ Aligned R-hat convergence diagnostics (accounts for sign/permutation indeterminacy)
- ✅ Automatic factor matching across chains using cosine similarity
- ✅ Convergence comparison plots (max R-hat, convergence rate)
- ✅ Variant ranking by: (1) Convergence quality, (2) Speed, (3) Memory

**Configuration**: `config_convergence.yaml` or `config.yaml`
```yaml
sgfa_configuration_comparison:
  num_chains: 2              # Minimum 2 for aligned R-hat
  num_samples: 1000          # Samples per chain
  num_warmup: 500            # Warmup per chain
```

### Comparison Metric: Aligned R-hat

**Why This Solves the Problem**:

Standard R-hat fails for factor models because chains can converge to equivalent solutions that differ only in sign flips or factor ordering, incorrectly indicating poor convergence.

**Aligned R-hat Solution**:
1. Match factors across chains using cosine similarity
2. Align signs to reference chain
3. Compute R-hat on aligned samples

**Ranking Criteria**:
- **Primary**: Aligned R-hat < 1.1 (good convergence)
- **Secondary**: Execution time (faster is better)
- **Tertiary**: Memory usage (lower is better)

This provides objective, automatic comparison of SGFA variants based on convergence quality.

### Usage

```bash
python run_experiments.py --config config_convergence.yaml \
  --experiments sgfa_configuration_comparison \
  --select-rois volume_sn_voxels.tsv
```

### Output

- **Convergence plots**: Max R-hat and convergence rate by variant
- **Performance metrics**: Aligned R-hat, execution time, memory usage
- **Recommendations**: Best converged variant automatically identified

### Performance Note

⚠️ **Computationally Intensive**: This experiment is time-consuming because chains run sequentially on remote workstations (memory-safe but slow).

- **Runtime estimate**: ~(num_variants × num_chains × chain_runtime)
- **Example**: 4 variants × 2 chains × 30 min/chain = ~4 hours total
- **Recommendation**: Use reduced sampling for initial comparison; full sampling only for final validation

### Files

- `experiments/sgfa_configuration_comparison.py` - Main implementation
- `analysis/factor_stability.py` - Aligned R-hat computation
- `config_convergence.yaml`, `config.yaml` - Configuration sections

---

## NeuroGFA: Neuroimaging-Specific Priors

**Status**: 🧠 **Requires Neuroimaging Expertise**

### Description

Specialized GFA variant with neuroimaging-specific spatial priors:
- Ising model priors for spatial coherence (neighboring voxels should have similar loadings)
- Anatomical connectivity integration (use brain atlases to inform factor structure)
- Hemisphere symmetry constraints (left/right brain should have symmetric factors)
- ROI-aware factor allocation (factors should respect anatomical boundaries)

### Why It's Blocked

Requires **deep neuroimaging domain knowledge** to:
1. Design appropriate spatial prior structures
2. Integrate anatomical atlases (AAL, Harvard-Oxford, etc.)
3. Define meaningful spatial coherence constraints
4. Validate that priors improve biological interpretability

### Framework Available

- `models/variants/neuroimaging_gfa.py`: Stub implementation exists
- Spatial utilities in `data/preprocessing.py`
- Position lookup infrastructure for spatial coordinates

### What's Needed

**Phase 1: Spatial Prior Design** (3-4 weeks)
- Research literature on spatial priors for neuroimaging (e.g., Ising models, CAR models)
- Design prior structure that respects brain anatomy
- Implement in NumPyro (may be computationally intensive)

**Phase 2: Anatomical Integration** (2-3 weeks)
- Load brain atlases (AAL, Harvard-Oxford)
- Map voxels to anatomical regions
- Design ROI-aware factor initialization

**Phase 3: Validation** (2-3 weeks)
- Compare against standard SGFA on interpretability
- Assess computational cost
- Validate on held-out data or external cohorts

**Dependencies**:
- `nibabel`, `nilearn`: Neuroimaging libraries
- Brain atlases (AAL, Harvard-Oxford, etc.)
- Advanced Bayesian modeling expertise
- Neuroimaging researcher familiar with spatial statistics

**Estimated Effort**: 2-3 months for experienced neuroimaging researcher

### Files

- `models/variants/neuroimaging_gfa.py` (stub exists)
- Would need new spatial prior utilities

---

## Advanced Neuroimaging Cross-Validation

**Status**: 🧠 **Requires Neuroimaging Expertise**

### Description

Neuroimaging-aware cross-validation with:
- Clinical-stratified splitting (balance diagnoses, subtypes)
- Scanner/site effect correction
- Spatial structure preservation across folds
- Neuroimaging-specific evaluation metrics

### Why It's Blocked

Requires **neuroimaging + machine learning expertise** to:
1. Design CV splits that respect clinical structure
2. Implement scanner harmonization (ComBat, neuroCombat)
3. Define neuroimaging-specific metrics (spatial coherence, anatomical plausibility)
4. Validate that CV properly estimates generalization

### Framework Available

- `analysis/cross_validation_library.py`: Basic CV infrastructure
- `analysis/cv_fallbacks.py`: Fallback utilities
- Clinical data loading in `data/qmap_pd.py`

### What's Needed

**Phase 1: Clinical-Aware Splitting** (1-2 weeks)
```python
class ClinicalAwareSplitter:
    def split(self, X, y, groups, clinical_data):
        # Multi-variable stratification (diagnosis, age, sex, site)
        # Site/scanner awareness
        # Group preservation (same subject not in train/test)
        # Demographic balance validation
```

**Phase 2: Neuroimaging Metrics** (1-2 weeks)
```python
class NeuroImagingMetrics:
    def calculate_fold_metrics(self, train_result, test_metrics, clinical_data):
        # Factor interpretability scoring
        # Cross-view consistency
        # Clinical association strength
        # Spatial coherence measures
```

**Phase 3: Hyperparameter Optimization** (2-3 weeks)
```python
class NeuroImagingHyperOptimizer:
    def optimize_hyperparameters(self, X_data, clinical_data, search_space):
        # Bayesian optimization (optuna/hyperopt)
        # Clinical-aware objective functions
        # Multi-objective optimization (accuracy + interpretability)
```

**Dependencies**:
- `optuna` or `hyperopt`: Bayesian optimization
- `neuroCombat`: Scanner harmonization
- Domain knowledge of neuroimaging evaluation

**Estimated Effort**: 1-2 months for experienced ML researcher with neuroimaging background

### Files

- `analysis/cross_validation_library.py` (framework exists)
- Would need implementations of:
  - `ClinicalAwareSplitter`
  - `NeuroImagingMetrics`
  - `NeuroImagingHyperOptimizer`

---

## Implementation Priority Recommendations

### High Priority (If Metrics Defined)
1. **Model Comparison** - Validate SGFA against baselines (2-4 weeks)
2. **SGFA Hyperparameter Comparison** - Optimize K/percW (2-3 weeks)

### Medium Priority (Requires Expertise)
3. **NeuroGFA** - Spatial priors for better interpretability (2-3 months)
4. **Advanced Neuro CV** - Proper generalization testing (1-2 months)

### Comparison Metric Options to Explore

**Option 1: Multi-Objective Composite Score**
```python
score = (
    0.4 * stability_rate +           # From factor_stability
    0.3 * clinical_association +     # From clinical_validation
    0.2 * spatial_coherence +        # New: spatial clustering of loadings
    0.1 * (1 - shrinkage_rate)       # Penalize over-shrinkage
)
```

**Option 2: Cross-Validated Reconstruction**
- Split data into train/test
- Train model on train set
- Measure reconstruction error on test set
- Compare across models

**Option 3: External Validation**
- Train on qMAP-PD cohort
- Test on external PD cohort (if available)
- Measure factor transferability

**Option 4: Clinical Endpoint Prediction**
- Use discovered factors to predict clinical outcomes (UPDRS, disease progression)
- Compare predictive performance across models
- Most clinically relevant but requires longitudinal data

---

## Notes

- **Framework vs Implementation**: "Framework available" means experiment orchestration, file I/O, and plotting infrastructure exists. The actual scientific implementation (metrics, priors, etc.) is missing.

- **Why Not Remove?**: These experiments are kept as stubs because:
  1. The infrastructure is useful for understanding the codebase architecture
  2. Future researchers may have the domain expertise to complete them
  3. They demonstrate the intended scope of the project

- **Current Production Pipeline**: The fully implemented and tested pipeline is:
  - `data_validation` → `robustness_testing` → `factor_stability`
  - Using `sparse_gfa_fixed` model with `config_convergence.yaml`

---

## References

- Model comparison metrics: Held-out likelihood, ELBO, WAIC
- Neuroimaging priors: Ising models, CAR models, spatial Bayesian methods
- CV strategies: Clinical-stratified CV, site-aware CV, neuroCombat harmonization
- Hyperparameter optimization: Bayesian optimization (Optuna, Hyperopt)
