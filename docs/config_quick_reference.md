# Configuration Quick Reference

## Essential Fields (Required)

```yaml
data:
  data_dir: "./qMAP-PD_data"           # REQUIRED: Data directory path

experiments:
  base_output_dir: "./results"         # REQUIRED: Results directory

model:
  model_type: "sparse_gfa"             # REQUIRED: "sparse_gfa" or "standard_gfa"
  K: 5                                 # REQUIRED: Number of factors (1-50)
  sparsity_lambda: 0.1                 # REQUIRED for sparse_gfa (≥0)
```

## Common Optional Fields

```yaml
model:
  num_samples: 1000                    # MCMC samples (default: 1000)
  num_warmup: 500                      # MCMC warmup (default: 500)
  num_chains: 2                        # MCMC chains (default: 2)
  random_seed: 42                      # Reproducibility seed

preprocessing:
  strategy: "standard"                 # "minimal"|"standard"|"advanced"
  imputation_strategy: "median"        # "mean"|"median"|"mode"|"drop"

cross_validation:
  n_folds: 5                          # CV folds (default: 5)
  n_repeats: 1                        # CV repeats (default: 1)

system:
  use_gpu: true                       # GPU acceleration (default: true)
  memory_limit_gb: 16.0               # Memory limit
```

## Model Type Cheat Sheet

| Model Type | Required Fields | Best For |
|------------|----------------|----------|
| `sparse_gfa` | `K`, `sparsity_lambda` | Neuroimaging, interpretable factors |
| `standard_gfa` | `K` | General factor analysis |

## Parameter Ranges

| Parameter | Range | Recommended | Notes |
|-----------|-------|-------------|-------|
| `K` | 1-50 | 5-10 | More factors = more complexity |
| `sparsity_lambda` | 0.0-10.0 | 0.01-0.5 | Higher = more sparse |
| `num_samples` | 10-50000 | 1000-5000 | More = better estimates |
| `num_chains` | 1-8 | 2-4 | More = better convergence check |

## Quick Start Configurations

### Development (Fast)

```yaml
model:
  K: 3
  num_samples: 100
  num_chains: 1
```

### Research (Balanced)

```yaml
model:
  K: 5
  num_samples: 1000
  num_chains: 2
  sparsity_lambda: 0.1
```

### Production (Robust)

```yaml
model:
  K: 8
  num_samples: 5000
  num_chains: 4
  sparsity_lambda: 0.05
cross_validation:
  n_folds: 10
  n_repeats: 3
```

## Validation Shortcuts

```bash
# Test configuration
python -c "from core.config_utils import validate_configuration; import yaml; print('Valid' if validate_configuration(yaml.safe_load(open('config.yaml'))) else 'Invalid')"

# Get defaults
python -c "from core.config_utils import get_default_configuration; import yaml; print(yaml.dump(get_default_configuration()))"
```

## Common Fixes

| Issue | Fix |
|-------|-----|
| "data_dir does not exist" | Create directory or check path |
| "sparse_gfa requires sparsity_lambda" | Add `sparsity_lambda: 0.1` |
| "num_warmup must be < num_samples" | Set `num_warmup < num_samples` |
| Out of memory | Reduce `K` or `num_samples`, enable GPU |
| Slow convergence | Increase `num_warmup` |
