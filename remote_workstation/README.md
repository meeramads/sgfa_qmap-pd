# Remote Computing Quick Start Guide

This directory contains setups for running the SGFA framework on remote computing resources:
- **University GPU Workstation** (`setup.sh` + `run_experiments.py`)
- **Google Colab** (`colab_experiments.ipynb`)

## University GPU Workstation Setup

## Initial Setup (First Time Only)

1. **Connect to remote  workstation** and clone the repository:
   ```bash
   git clone https://github.com/meeramads/sgfa_qmap-pd
   cd sgfa_qmap-pd
   ```

2. **Run the setup script**:
   ```bash
   ./setup_remote_workstation.sh
   ```
   This will:
   - Create Python virtual environment
   - Install all dependencies including JAX with CUDA support
   - Verify GPU availability
   - Test framework imports

3. **Update data path** in `config/remote_workstation_config.yaml`:
   ```yaml
   data:
     data_dir: "/path/to/your/qMAP-PD_data"  # Update this!
   ```

## Running Experiments

### Full Experimental Suite
```bash
# Run all experiments (recommended)
python run_remote_workstation_experiments.py

# With custom config
python run_remote_workstation_experiments.py --config config/remote_workstation_config.yaml
```

### Individual Experiment Types
```bash
# Data validation only
python run_remote_workstation_experiments.py --experiments data_validation

# Method comparison only  
python run_remote_workstation_experiments.py --experiments method_comparison

# Performance benchmarks only
python run_remote_workstation_experiments.py --experiments performance_benchmarks

# Multiple specific experiments
python run_remote_workstation_experiments.py --experiments data_validation method_comparison
```

### With Custom Data Directory
```bash
python run_remote_workstation_experiments.py --data-dir /custom/path/to/qMAP-PD_data
```

## Monitoring Progress

### View Live Logs
```bash
tail -f logs/remote_workstation_experiments.log
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### View Results
Results are saved in `remote_workstation_experiment_results/` with timestamps:
- `data_validation_<timestamp>/`
- `method_comparison_<timestamp>/`  
- `performance_benchmarks_<timestamp>/`
- `sensitivity_analysis_<timestamp>/`

## Configuration Options

Edit `config/remote_workstation_config.yaml` to customize:

- **GPU settings**: `system.batch_size`, `system.memory_limit_gb`
- **Model parameters**: `method_comparison.models`
- **Preprocessing**: `data_validation.preprocessing_strategies`
- **Performance tests**: `performance_benchmarks.benchmark_configs`

## Troubleshooting

### GPU Not Detected
```bash
nvidia-smi  # Check if GPU is available
python -c "import jax; print(jax.devices())"
```

### Memory Issues
Reduce batch sizes in `config/remote_workstation_config.yaml`:
```yaml
system:
  batch_size: 32  # Reduce from 64
  memory_limit_gb: 16  # Reduce from 32
```

### Import Errors
Re-activate environment:
```bash
source venv/bin/activate
python -c "from experiments.framework import ExperimentFramework"
```

## Expected Runtime

On remote GPU workstation (estimated):
- **Data validation**: 30-60 minutes
- **Method comparison**: 2-4 hours  
- **Performance benchmarks**: 1-2 hours
- **Sensitivity analysis**: 3-6 hours
- **Complete suite**: 6-12 hours

## Output Structure

```
remote_workstation_experiment_results/
├── data_validation_20231207_143022/
│   ├── quality_assessment/
│   ├── preprocessing_comparison/
│   └── diagnostics/
├── method_comparison_20231207_150315/
│   ├── model_results/
│   ├── cross_validation/
│   └── performance_metrics/
├── performance_benchmarks_20231207_154521/
└── experiment_summary.yaml
```

## Recovery from Interruption

If experiments are interrupted:
1. Check `experiment_summary.yaml` to see what completed
2. Re-run specific experiments:
   ```bash
   python run_remote_workstation_experiments.py --experiments method_comparison
   ```
3. Results are timestamped, so no data is lost

---

## Quick Commands Reference

```bash
# Setup (first time)
./setup_remote_workstation.sh

# Run everything  
python run_remote_workstation_experiments.py

# Monitor
tail -f logs/remote_workstation_experiments.log

# Check GPU
nvidia-smi

# Activate environment
source venv/bin/activate
```

---

## Google Colab Setup

### Quick Start
1. **Open the notebook**: Upload `colab_experiments.ipynb` to Google Colab or open from GitHub
2. **Enable GPU**: Runtime → Change runtime type → Hardware accelerator: GPU
3. **Run setup cells**: Execute the initial setup and data mounting cells
4. **Run experiments**: Choose from data validation, method comparison, or performance benchmarks

### Features
- **GPU acceleration** with CUDA support
- **Persistent storage** via Google Drive mounting
- **Interactive development** with Jupyter interface
- **Quick prototyping** without local setup

### Data Setup for Colab
```python
# Mount Google Drive (run in Colab cell)
from google.colab import drive
drive.mount('/content/drive')

# Upload qMAP-PD data to Google Drive at:
# /content/drive/MyDrive/qMAP-PD_data/
```

### Runtime Limits
- **Free tier**: 12-hour sessions, shared GPU resources
- **Pro/Pro+**: Longer sessions, better GPU access
- **Recommendation**: Use for prototyping, not production runs