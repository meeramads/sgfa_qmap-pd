# Synthetic Data Demonstration

This guide explains how to run the complete SGFA factor stability pipeline on synthetic data to generate demonstrable results with ground truth validation.

## Why Run on Synthetic Data?

1. **Complete Ground Truth**: Know the true number of factors (K=3), sparsity patterns, and noise levels
2. **Faster Execution**: Synthetic data is smaller and runs faster than real qMAP-PD data
3. **Memory Safe**: No GPU memory issues with N=150, D=120 total features
4. **Pipeline Validation**: Verify all recent changes (patient IDs, feature names) work end-to-end
5. **Demonstrable Results**: Generate publication-quality results for presentations/papers

## Synthetic Data Properties

The synthetic data generator creates:

- **N = 150 samples** (vs. N=86 in real qMAP-PD data)
- **3 views** (mimicking clinical + 2 imaging views)
  - View 1: 60 features
  - View 2: 40 features
  - View 3: 20 features
  - Total: D = 120 features
- **K_true = 3** latent factors (ground truth)
- **percW_true = 33%** sparsity (ground truth)
- **Known noise levels** per view
- **Structured sparsity patterns** in latent factors Z
- **Subject IDs**: `subj_000` through `subj_149`
- **Feature names**: `view_1_feat_000`, etc.

## Configuration

The synthetic data configuration is in [`config_synthetic.yaml`](config_synthetic.yaml):

### Key Parameters

```yaml
data:
  dataset: "synthetic"  # Use synthetic data generator
  num_sources: 3        # Number of views
  K_true: 3             # TRUE number of factors
  percW_true: 33.0      # TRUE sparsity percentage

factor_stability:
  K: 10                 # Infer with K > K_true to test shrinkage
  num_chains: 4         # Multiple chains for stability
  num_samples: 3000     # MCMC samples per chain
  num_warmup: 500       # Warmup samples
  percW: 25.0           # Infer with slightly different sparsity
```

### Expected Behavior

1. **Automatic Shrinkage**: Model should shrink from K=10 to ~3-5 effective factors
2. **High Stability**: Clean synthetic data should show >80% factor stability
3. **Fast Runtime**: ~15-30 minutes total (vs. hours for real data)

## Running the Demo

### Option 1: Using the Shell Script (Recommended)

```bash
./run_synthetic_demo.sh
```

This runs the factor stability experiment with unified results output.

### Option 2: Direct Python Command

```bash
python run_experiments.py \
    --config config_synthetic.yaml \
    --experiments factor_stability \
    --unified-results
```

### Option 3: Run Full Pipeline

To run all experiments (data validation, robustness testing, stability, clinical validation):

```bash
python run_experiments.py \
    --config config_synthetic.yaml \
    --experiments all \
    --unified-results
```

## Output Structure

Results will be saved to:

```
results/factor_stability_run_YYYYMMDD_HHMMSS/
├── 03_factor_stability/
│   ├── chains/
│   │   ├── chain_0/
│   │   │   ├── W_view_0.csv          # Factor loadings (view 1)
│   │   │   ├── W_view_1.csv          # Factor loadings (view 2)
│   │   │   ├── W_view_2.csv          # Factor loadings (view 3)
│   │   │   ├── Z.csv                 # Factor scores (with subject IDs)
│   │   │   └── metadata.json         # Convergence info
│   │   ├── chain_1/ ...
│   │   ├── chain_2/ ...
│   │   └── chain_3/ ...
│   ├── stability_analysis/
│   │   ├── consensus_W.csv           # Consensus factor loadings
│   │   ├── consensus_Z.csv           # Consensus factor scores
│   │   ├── factor_stability_summary.json
│   │   ├── similarity_matrix.csv
│   │   └── similarity_heatmap.png
│   └── plots/
│       ├── factor_stability_heatmap.png
│       ├── effective_factors_barplot.png
│       └── chain_similarity_distributions.png
└── summaries/
    └── factor_stability_summary.txt
```

## Interpreting Results

### Factor Stability Summary

Check `factor_stability_summary.json`:

```json
{
  "stability": {
    "n_stable_factors": 3,      // Should be ~3 (matching K_true)
    "total_factors": 10,
    "stability_rate": 0.3,       // 30% stable (3/10)
    "threshold": 0.8,
    "min_match_rate": 0.5,
    "n_chains": 4
  },
  "effectiveness": {
    "n_effective": 4,            // Should be ~3-5
    "total_factors": 10,
    "shrinkage_rate": 0.6        // 60% shrinkage (good!)
  }
}
```

### Success Criteria

✅ **Good Results:**
- n_effective ≈ 3-5 (close to K_true=3)
- shrinkage_rate > 0.5 (model successfully shrinks from K=10)
- stability_rate > 0.2 (at least 2-3 stable factors)
- All chains converge (check `metadata.json` per chain)

⚠️ **Needs Investigation:**
- n_effective = 10 (no shrinkage - try higher percW)
- stability_rate = 0 (no stable factors - increase num_samples)
- convergence = false (try longer warmup)

### Factor Loadings

Check `consensus_W.csv` - should show:
- **Feature Names**: `view_1_feat_000`, `view_2_feat_010`, etc. (not generic Feature_0)
- **Sparse Patterns**: Most loadings near zero, few large values
- **View Structure**: Clear separation between views

### Factor Scores

Check `consensus_Z.csv` - should show:
- **Patient IDs**: `subj_000` through `subj_149` (not generic Subject_0)
- **Column Names**: `Factor_0`, `Factor_1`, ... (indexed from 0)
- **Values**: Standardized scores (mean ≈ 0, std ≈ 1)

## Ground Truth Validation

The synthetic data includes ground truth for validation. After running, you can compare:

1. **True K vs. Inferred K**: Compare n_effective to K_true=3
2. **True Sparsity vs. Inferred**: Compare learned loadings to ground truth patterns
3. **Factor Stability**: High stability confirms reliable inference

To access ground truth (for advanced analysis):

```python
from data.synthetic import generate_synthetic_data

# Generate same data
data = generate_synthetic_data(num_sources=3, K=3, percW=33.0)

# Ground truth
Z_true = data["ground_truth"]["Z"]        # True factor scores
W_true = data["ground_truth"]["W"]        # True factor loadings
sigma_true = data["ground_truth"]["sigma"] # True noise levels
```

## Comparison with Real qMAP-PD Data

| Property | Synthetic | Real qMAP-PD |
|----------|-----------|--------------|
| N (samples) | 150 | 86 |
| D (features) | 120 | 1,793 (voxels) |
| K_true | 3 (known) | Unknown |
| Runtime | 15-30 min | 2-5 hours |
| Memory | Low | High (GPU limited) |
| Ground truth | Yes | No |
| Stability | High (>80%) | Low (~5-20%) |

## Troubleshooting

### Error: "data_dir must be provided"

**Solution**: The synthetic data generator doesn't need a data directory. The code should handle this automatically. If you see this error, it means the dataset type wasn't detected. Check that `config_synthetic.yaml` has:

```yaml
data:
  dataset: "synthetic"
```

### Error: "Out of memory"

**Solution**: This shouldn't happen with synthetic data (very small). If it does, reduce:

```yaml
factor_stability:
  num_samples: 2000  # Reduce from 3000
  K: 8               # Reduce from 10
```

### Low Factor Stability

**Possible causes**:
1. Not enough MCMC samples - increase `num_samples`
2. K too high - reduce K closer to K_true=3
3. Random variation - try different `random_seed`

### No Shrinkage (n_effective = K)

**Solution**: Increase sparsity:

```yaml
factor_stability:
  percW: 15.0  # Stronger sparsity (lower = more aggressive)
```

## Next Steps

After successfully running on synthetic data:

1. **Verify Patient IDs**: Check Z.csv files have `subj_000` etc. (not Subject_0)
2. **Verify Feature Names**: Check W.csv files have `view_1_feat_000` etc. (not Feature_0)
3. **Compare to Ground Truth**: Validate that n_effective ≈ K_true
4. **Use for Presentations**: Clean results with known ground truth
5. **Apply to Real Data**: Once validated, run on actual qMAP-PD data

## Questions?

- Configuration issues: Check [config_synthetic.yaml](config_synthetic.yaml)
- Data generation: See [data/synthetic.py](data/synthetic.py)
- Pipeline flow: See [run_experiments.py](run_experiments.py)
- Factor stability: See [analysis/factor_stability.py](analysis/factor_stability.py)
