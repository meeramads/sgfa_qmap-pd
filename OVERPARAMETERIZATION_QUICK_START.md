# Overparameterization Testing: Quick Start Guide

## What We're Testing

**Question**: Does high K (overparameterization) lead to stable factors, or is stability an artifact?

## Three Configs Ready to Run

| Config File | K | Purpose | Runtime |
|-------------|---|---------|---------|
| **config_K2_no_sparsity.yaml** | 2 | Baseline (minimal) | 4-6 hours |
| **config_K20_no_sparsity.yaml** | 20 | Moderate overparameterization | 8-12 hours |
| **config_K50_no_sparsity.yaml** | 50 | Extreme overparameterization | 16-24 hours |

All use:
- `standard_gfa` (simple ARD priors, NO horseshoe)
- 4 chains × (1000 warmup + 2000 samples)
- `max_tree_depth=13`
- Same MAD-filtered data

## How to Run

```bash
cd /Users/meera/Desktop/sgfa_qmap-pd

# Option 1: Run K=2 (baseline)
python run_experiments.py --config config_K2_no_sparsity.yaml --experiments robustness_testing

# Option 2: Run K=20 (key test)
python run_experiments.py --config config_K20_no_sparsity.yaml --experiments robustness_testing

# Option 3: Run K=50 (extreme test)
python run_experiments.py --config config_K50_no_sparsity.yaml --experiments robustness_testing
```

**Recommendation**: Start with K=2 and K=20 first, then decide if K=50 is needed.

## What to Check After Each Run

### 1. R-hat Convergence

```bash
# Check convergence diagnostics
cat results/robustness_testing_K*_*/03_factor_stability/stability_analysis/rhat_convergence_diagnostics.json

# Look for:
# "rhat_W_max": should decrease with K if H1 true
# "rhat_Z_max": should decrease with K if H1 true
```

### 2. Effective Factors

From the log output, look for:
```
INFO: Chain 0: X/K effective factors
INFO: Chain 1: X/K effective factors
...
```

**If H1 (real)**: Effective factors ~5-10 regardless of K (ARD selecting)
**If H2 (artifact)**: Effective factors scales with K

### 3. Stable Factors

From stability results:
```
INFO: ✅ Stable factors: X/K
INFO: Stability rate: X%
```

**Critical metric**: Count of stable factors (not rate)
- **H1 (real)**: ~4-6 stable factors across K=20 and K=50
- **H2 (artifact)**: Stable factors increases with K

### 4. Trace Plots

Open the trace diagnostic plots:
```
results/robustness_testing_K*_*/03_factor_stability/plots/mcmc_trace_diagnostics.png
```

Look for:
- **Converged**: Overlapping traces across chains
- **Not converged**: Separated traces

## Expected Results

### Hypothesis 1: Overparameterization Helps (Real)

| K | R-hat | Effective | Stable | Interpretation |
|---|-------|-----------|--------|----------------|
| 2 | >3.0 | 2/2 (100%) | 0/2 (0%) | Underparameterized |
| 20 | ~1.3 | 6/20 (30%) | 4/20 (20%) | Better exploration |
| 50 | <1.1 | 5/50 (10%) | 4/50 (8%) | ARD selects ~5 factors |

**Signature**: Same ~4-5 stable factors in K=20 and K=50

### Hypothesis 2: Measurement Artifact

| K | R-hat | Effective | Stable | Interpretation |
|---|-------|-----------|--------|----------------|
| 2 | >3.0 | 2/2 (100%) | 0/2 (0%) | Convergence issues |
| 20 | >2.5 | 18/20 (90%) | 10/20 (50%) | Spurious matches |
| 50 | >2.0 | 45/50 (90%) | 25/50 (50%) | Random matching |

**Signature**: Stable factor count scales with K (not constant)

## Quick Decision Tree

After K=2 and K=20 complete:

```
Does R-hat improve from K=2 to K=20?
│
├─ YES (e.g., 5.0 → 1.5)
│  └─ Does K=20 have ~5-10 effective factors?
│     ├─ YES → H1 likely true
│     │  └─ Run K=50 to confirm (optional)
│     │     • If same stable factors → CONFIRMED H1
│     │     • Use K=20-30 in future
│     │
│     └─ NO (e.g., 18/20 effective) → Mixed evidence
│        └─ Check stability consistency
│
└─ NO (e.g., 5.0 → 4.8)
   └─ H2 likely true (artifact)
      • Don't bother with K=50
      • Focus on fixing convergence
      • Use K=2-5 only
```

## Next Steps Based on Results

### If H1 Confirmed (Overparameterization Helps)

✅ **Actions:**
1. Use K=20 or K=30 as standard
2. Analyze the ~4-6 stable factors biologically
3. Try K=20 with relaxed horseshoe priors
4. Paper angle: "Blessing of overparameterization"

### If H2 Confirmed (Artifact)

❌ **Actions:**
1. Stick to K=2-5 (low K)
2. Fix underlying convergence issues:
   - More MCMC samples
   - Better initialization
   - Reparameterization
   - Informative priors
3. Use stricter stability criteria (cosine > 0.9, min_match_rate > 0.75)

## Files Generated Per Run

```
results/robustness_testing_K{2,20,50}_no_sparsity_run_{timestamp}/
└── 03_factor_stability/
    ├── plots/
    │   ├── mcmc_trace_diagnostics.png          ⭐ Visual convergence check
    │   ├── mcmc_parameter_distributions.png
    │   ├── hyperparameter_posteriors.png       (shows alpha not tauW)
    │   ├── hyperparameter_traces.png
    │   └── factor_stability_heatmap.png
    ├── chains/
    │   ├── chain_0/ ... chain_3/
    │   │   ├── W_view_*.csv
    │   │   ├── Z.csv
    │   │   └── metadata.json
    ├── stability_analysis/
    │   ├── rhat_convergence_diagnostics.json   ⭐ Numerical R-hat values
    │   ├── stability_results.json              ⭐ Stable factor info
    │   └── consensus_W_*.npy (if any stable)
    └── summaries/
        └── complete_experiment_summary.yaml
```

## Computational Considerations

### Memory

- K=2: ~1-2 GB
- K=20: ~3-5 GB (10x more parameters)
- K=50: ~8-12 GB (25x more parameters)

All should fit in your system (16 GB available).

### Time Per Iteration

- K=2: Baseline
- K=20: ~2-3x slower (more parameters)
- K=50: ~5-8x slower (many parameters)

### Total Runtime Estimates

| Config | Per Chain | 4 Chains Total |
|--------|-----------|----------------|
| K=2 | 1-1.5 hrs | 4-6 hours |
| K=20 | 2-3 hrs | 8-12 hours |
| K=50 | 4-6 hrs | 16-24 hours |

**Plan**: Start K=2 and K=20 overnight, check results in morning.

## Key Metrics Summary Table

After all runs complete, create this comparison:

```python
import pandas as pd

results = {
    'K': [2, 20, 50],
    'R-hat (max)': [rhat_K2, rhat_K20, rhat_K50],
    'R-hat (mean)': [rhat_mean_K2, rhat_mean_K20, rhat_mean_K50],
    'Effective factors': [eff_K2, eff_K20, eff_K50],
    'Stable factors': [stable_K2, stable_K20, stable_K50],
    'Stability rate (%)': [100*s/k for s,k in zip([stable_K2, stable_K20, stable_K50], [2,20,50])],
    'Runtime (hours)': [time_K2, time_K20, time_K50],
}

df = pd.DataFrame(results)
print(df.to_markdown())
```

Example output:
```
| K  | R-hat (max) | Effective | Stable | Stability rate | Runtime |
|----|-------------|-----------|--------|----------------|---------|
| 2  | 5.2         | 2         | 0      | 0%             | 5.1     |
| 20 | 1.4         | 6         | 4      | 20%            | 10.3    |
| 50 | 1.1         | 5         | 4      | 8%             | 19.7    |
```

→ This would support H1 (same 4 stable factors, R-hat improves)

## Documentation

- **Full analysis**: [OVERPARAMETERIZATION_HYPOTHESIS.md](OVERPARAMETERIZATION_HYPOTHESIS.md)
- **Config details**: [CONVERGENCE_DIAGNOSTIC_RUN.md](CONVERGENCE_DIAGNOSTIC_RUN.md)
- **MCMC diagnostics**: [docs/MCMC_TRACE_PLOTS.md](docs/MCMC_TRACE_PLOTS.md)

## Contact

If results are ambiguous or unexpected, check:
1. Trace diagnostic plots (visual)
2. R-hat values (numerical)
3. Effective factor counts per chain
4. Factor loading magnitudes

Good luck with the experiments! 🎉
