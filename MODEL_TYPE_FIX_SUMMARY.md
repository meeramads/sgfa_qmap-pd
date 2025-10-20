# Model Type Configuration Fix

## Problem

The code was using `sparseGFA` instead of `sparse_gfa_fixed` even though `config_convergence.yaml` specifies `model_type: "sparse_gfa_fixed"`.

## Root Cause

The `model_type` from the YAML config was not being propagated through the system:

1. **ExperimentConfig dataclass** didn't have a `model_type` field
2. **run_experiments.py** wasn't reading `model_type` from the YAML and passing it to ExperimentConfig
3. **robustness_testing.py** wasn't reading `model_type` from config when creating ExperimentConfig
4. **models_integration.py** expected `config_dict["model"]["model_type"]` but the structure wasn't created correctly

## Solution

### Changes Made

#### 1. Added `model_type` field to ExperimentConfig
**File:** `experiments/framework.py:65`

```python
# Model configuration
model_type: str = "sparseGFA"  # Model type: sparseGFA, sparse_gfa_fixed, neuroGFA, etc.
```

#### 2. Updated run_experiments.py for factor_stability
**File:** `run_experiments.py:646-655`

```python
# Get model configuration
model_config = exp_config.get("model", {})
model_type = model_config.get("model_type", "sparseGFA")

experiment_config = ExperimentConfig(
    experiment_name="factor_stability_analysis",
    description="Factor stability analysis with fixed parameters (Ferreira et al. 2024)",
    dataset="qmap_pd",
    data_dir=get_data_dir(exp_config),
    model_type=model_type,  # Use model_type from config
    ...
)
```

#### 3. Updated robustness_testing.py
**File:** `experiments/robustness_testing.py:2901-2918`

```python
# Get model type from config
model_type = model_config.get("model_type", "sparseGFA")
logger.info(f"   Using model type: {model_type}")

# Create ExperimentConfig with model parameters for semantic naming
exp_config = ExperimentConfig(
    experiment_name="robustness_tests",
    description="Robustness testing for SGFA",
    dataset="qmap_pd",
    data_dir=get_data_dir(config),
    model_type=model_type,  # Use model_type from config
    ...
)
```

#### 4. Fixed model config structuring in run_sgfa
**File:** `experiments/robustness_testing.py:675-693`

```python
# Debug logging
if verbose:
    self.logger.info(f"🔍 DEBUG: config_dict.get('model_type') = {config_dict.get('model_type', 'NOT FOUND')}")
    self.logger.info(f"🔍 DEBUG: 'model' in config_dict = {'model' in config_dict}")
    if "model_type" in args:
        self.logger.info(f"🔍 DEBUG: args['model_type'] = {args['model_type']}")

# Ensure model configuration is structured correctly for integration
if "model" not in config_dict:
    config_dict["model"] = {}

# Priority: args > ExperimentConfig.model_type > default
if "model_type" in args:
    config_dict["model"]["model_type"] = args["model_type"]
elif "model_type" not in config_dict["model"]:
    config_dict["model"]["model_type"] = config_dict.get("model_type", "sparseGFA")
```

**Key fix:** Create `config_dict["model"]` BEFORE trying to set `config_dict["model"]["model_type"]`

## Data Flow

Here's how `model_type` flows through the system now:

```
config_convergence.yaml
├── model:
│   └── model_type: "sparse_gfa_fixed"
    ↓
run_experiments.py
├── Reads YAML → exp_config dict
├── Extracts: model_type = exp_config["model"]["model_type"]
└── Creates: ExperimentConfig(model_type=model_type)
    ↓
robustness_testing.py (run_robustness_testing)
├── Reads config dict
├── Extracts: model_type = config_dict["model"]["model_type"]
└── Creates: ExperimentConfig(model_type=model_type)
    ↓
robustness_testing.py (run_sgfa)
├── Converts: config_dict = ConfigHelper.to_dict(self.config)
│   └── Now has: config_dict["model_type"] = "sparse_gfa_fixed"
├── Structures: config_dict["model"]["model_type"] = config_dict["model_type"]
└── Passes to: integrate_models_with_pipeline(config=config_dict)
    ↓
models_integration.py
├── Reads: model_type = config_dict["model"]["model_type"]
├── Validates: model_type in ModelFactory.list_models()
└── Creates: ModelFactory.create_model(model_type="sparse_gfa_fixed")
    ↓
models/factory.py
└── Returns: SparseGFAFixedModel instance ✅
```

## Verification

Run the test script to verify:

```bash
python test_model_type_robustness.py
```

Expected output:
```
✅ SUCCESS! Model type flows correctly: sparse_gfa_fixed
```

## Usage

Now when you run:

```bash
python run_experiments.py --config config_convergence.yaml \
  --experiments robustness_testing --select-rois volume_sn_voxels.tsv \
  --K 2 --percW 33 --max-tree-depth 10
```

You'll see in the logs:

```
INFO:experiments.robustness_testing:   Using model type: sparse_gfa_fixed
INFO:experiments.robustness_testing:🔍 DEBUG: config_dict.get('model_type') = sparse_gfa_fixed
INFO:models.models_integration:🔍 DEBUG: model_type from config: 'sparse_gfa_fixed'
INFO:models.models_integration:✅ Selected model type: sparse_gfa_fixed
INFO:models.models_integration:   Model class: SparseGFAFixedModel
```

## Testing

All existing tests still pass, and the new behavior is verified by:

1. `test_model_type.py` - Tests basic flow
2. `test_model_type_robustness.py` - Tests robustness testing flow
3. Running actual experiments confirms correct model is used

## Benefits

1. **Correctness:** Uses the model specified in config
2. **Transparency:** Clear logging shows which model is selected
3. **Flexibility:** Can easily switch models by changing config
4. **Debugging:** Added debug logs to trace model selection
5. **Consistency:** All experiments use the same model selection logic
