# Remote Workstation Guide for SGFA qMAP-PD

This directory contains scripts and configurations for running the SGFA framework on university GPU workstations with quota-limited environments.

## Table of Contents
- [Initial Setup](#initial-setup)
- [Running Experiments](#running-experiments)
- [Using tmux for Long Runs](#using-tmux-for-long-runs)
- [Monitoring Progress](#monitoring-progress)
- [Configuration Options](#configuration-options)
- [Troubleshooting](#troubleshooting)
- [Managing Disk Space](#managing-disk-space)

---

## Initial Setup

### First Time Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/meeramads/sgfa_qmap-pd
   cd sgfa_qmap-pd
   ```

2. **Run the setup script**:
   ```bash
   chmod +x remote_workstation/setup.sh
   ./remote_workstation/setup.sh
   ```
   
   Choose option 1 for full setup, which will:
   - Check disk quota and space availability
   - Configure CUDA paths for system GPU
   - Create Python virtual environment
   - Install dependencies with `jax[cuda12_local]` (saves ~4GB vs full CUDA)
   - Verify JAX can access GPU
   - Create necessary directories
   - Configure Git for HTTPS (if needed)

3. **Update data path** in `config.yaml`:
   ```bash
   # The setup script will prompt you, or manually edit:
   vim remote_workstation/config.yaml
   # Update: data_dir: "/path/to/your/qMAP-PD_data"
   ```

### Quick Setup Commands

```bash
# Full setup (interactive menu)
./remote_workstation/setup.sh

# Direct setup without menu
./remote_workstation/setup.sh setup

# Check if setup is complete
./remote_workstation/setup.sh status
```

---

## Running Experiments

After setup is complete, you have several options for running experiments:

### Option 1: Using tmux (Recommended for Long Runs)

```bash
# Start a new tmux session
tmux new -s sgfa_exp

# Inside tmux, activate environment and run
source venv/bin/activate
python remote_workstation/run_experiments.py --experiments all

# Detach from tmux (experiments keep running)
# Press: Ctrl+B, then D

# Later, reattach to check progress
tmux attach -t sgfa_exp

# List all tmux sessions
tmux ls
```

### Option 2: Using nohup (Background Process)

```bash
# Activate environment
source venv/bin/activate

# Run in background with nohup
nohup python remote_workstation/run_experiments.py --experiments all > experiment_output.log 2>&1 &

# Save the process ID
echo $! > experiment.pid

# Check if still running
ps -p $(cat experiment.pid)

# Monitor output
tail -f experiment_output.log
```

### Option 3: Direct Run (Stay Attached)

```bash
# For shorter experiments or testing
source venv/bin/activate
python remote_workstation/run_experiments.py --experiments all
```

### Running Specific Experiments

```bash
# Run individual experiment types
python remote_workstation/run_experiments.py --experiments data_validation

python remote_workstation/run_experiments.py --experiments method_comparison

python remote_workstation/run_experiments.py --experiments performance_benchmarks

python remote_workstation/run_experiments.py --experiments sensitivity_analysis

# Run multiple specific experiments
python remote_workstation/run_experiments.py --experiments data_validation method_comparison

# With custom data directory
python remote_workstation/run_experiments.py --data-dir /custom/path/to/data
```

---

## Using tmux for Long Runs

### Essential tmux Commands

| Command | Description |
|---------|-------------|
| `tmux new -s name` | Create new session |
| `tmux ls` | List all sessions |
| `tmux attach -t name` | Attach to session |
| `Ctrl+B, D` | Detach from session |
| `Ctrl+B, [` | Enter scroll mode (use arrow keys) |
| `q` | Exit scroll mode |
| `Ctrl+B, C` | Create new window |
| `Ctrl+B, N` | Next window |
| `Ctrl+B, P` | Previous window |
| `tmux kill-session -t name` | Kill a session |

### Example Workflow

```bash
# 1. Start experiments in tmux
tmux new -s experiments
source venv/bin/activate
python remote_workstation/run_experiments.py --experiments all

# 2. Detach (Ctrl+B, D) and logout
exit

# 3. Next day: SSH back in and reattach
ssh username@workstation
cd sgfa_qmap-pd
tmux attach -t experiments
```

---

## Monitoring Progress

### Live Monitoring Commands

```bash
# Watch experiment logs
tail -f logs/remote_workstation_experiments.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check specific experiment progress
ls -la remote_workstation_experiment_results/

# View latest results
ls -lt remote_workstation_experiment_results/ | head -10

# Check disk usage
df -h ~/
du -sh remote_workstation_experiment_results/

# Monitor memory usage
htop  # or top
```

### Creating a Monitoring Script

Save as `monitor.sh`:

```bash
#!/bin/bash
clear
echo "=== SGFA Experiment Monitor ==="
echo "Time: $(date)"
echo ""
echo "=== Process Status ==="
pgrep -f "run_experiments.py" && echo "✓ Running" || echo "✗ Not running"
echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
echo ""
echo "=== Latest Log Entries ==="
tail -n 5 logs/remote_workstation_experiments.log
echo ""
echo "=== Recent Results ==="
ls -lt remote_workstation_experiment_results/ | head -3
```

Make executable and run:
```bash
chmod +x monitor.sh
watch -n 10 ./monitor.sh  # Update every 10 seconds
```

---

## Configuration Options

### Edit Experiment Configuration

```bash
vim remote_workstation/config.yaml
```

Key settings to adjust:

```yaml
# System settings
system:
  batch_size: 64  # Reduce if OOM errors
  memory_limit_gb: 32  # Adjust based on GPU
  
# Data settings  
data:
  data_dir: "/path/to/qMAP-PD_data"  # Must update!
  
# Experiment selection
experiments:
  run_experiments:
    - "data_validation"
    - "method_comparison"
    - "sensitivity_analysis"
    - "performance_benchmarks"

# Model parameters
method_comparison:
  models:
    - name: "sparseGFA"
      n_factors: [5, 10, 15]  # Reduce for faster tests
      sparsity_lambda: [0.1, 0.5, 1.0]
```

### Preprocessing Strategies

Configure different preprocessing approaches:

```yaml
data_validation:
  preprocessing_strategies:
    minimal:
      enable_advanced_preprocessing: false
    standard:
      variance_threshold: 0.01
    aggressive:
      n_top_features: 1000
```

---

## Troubleshooting

### Common Issues and Solutions

#### GPU Not Detected
```bash
# Check CUDA configuration
echo $CUDA_HOME
echo $XLA_FLAGS

# Reconfigure CUDA
./remote_workstation/setup.sh cuda

# Test JAX GPU access
source venv/bin/activate
python -c "import jax; print(jax.devices())"
```

#### Out of Memory Errors
```bash
# Reduce batch size in config.yaml
# system:
#   batch_size: 32  # Reduced from 64

# Or set memory fraction
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
```

#### Disk Quota Exceeded
```bash
# Check quota
quota -s

# Clean up
./remote_workstation/setup.sh clean

# Remove old results
find remote_workstation_experiment_results -mtime +7 -delete
```

#### Import Errors
```bash
# Reactivate environment
source venv/bin/activate

# Verify installation
python -c "from experiments.framework import ExperimentFramework"

# Reinstall if needed
pip install -r requirements.txt
```

#### Experiments Interrupted
```bash
# Check if checkpoint exists
ls checkpoints/

# Resume from checkpoint (if implemented)
python remote_workstation/run_experiments.py --resume-from checkpoints/latest.pkl
```

---

## Managing Disk Space

### Space-Saving Tips

1. **Use system CUDA** instead of pip packages:
   ```bash
   pip uninstall nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 -y
   ```

2. **Clean pip cache**:
   ```bash
   pip cache purge
   rm -rf ~/.cache/pip
   ```

3. **Remove old results**:
   ```bash
   # Keep only last 7 days
   find remote_workstation_experiment_results -mtime +7 -type d -delete
   ```

4. **Compress results**:
   ```bash
   tar -czf results_backup_$(date +%Y%m%d).tar.gz remote_workstation_experiment_results/
   ```

5. **Check space usage**:
   ```bash
   # Top space users in home
   du -sh ~/* | sort -rh | head -10
   
   # Project space breakdown
   du -sh */ | sort -rh
   ```

---

## Expected Runtimes

Approximate runtimes on GPU workstation (86 subjects, 30K features):

| Experiment | Duration | GPU Memory |
|------------|----------|------------|
| Data Validation | 30-60 min | ~4GB |
| Method Comparison | 2-4 hours | ~8GB |
| Performance Benchmarks | 1-2 hours | ~6GB |
| Sensitivity Analysis | 3-6 hours | ~8GB |
| **Full Suite** | **6-12 hours** | **~8GB peak** |

---

## Output Structure

```
remote_workstation_experiment_results/
├── data_validation_20250913_194434/
│   ├── quality_assessment/
│   ├── preprocessing_comparison/
│   └── diagnostics/
├── method_comparison_20250913_201523/
│   ├── model_results/
│   ├── cross_validation/
│   └── performance_metrics/
├── performance_benchmarks_20250913_214521/
│   ├── scalability_tests/
│   └── memory_profiles/
├── sensitivity_analysis_20250913_223012/
│   └── parameter_sweeps/
└── experiment_summary.yaml
```

---

## Quick Reference Card

```bash
# Setup
./remote_workstation/setup.sh         # Run setup

# Start experiments (tmux)
tmux new -s exp                       # Create session
source venv/bin/activate              # Activate env
python remote_workstation/run_experiments.py --experiments all
Ctrl+B, D                             # Detach

# Monitor
tmux attach -t exp                    # Reattach
tail -f logs/*.log                    # View logs
watch nvidia-smi                      # GPU usage

# Status checks
./remote_workstation/setup.sh status  # System status
tmux ls                               # List sessions
ps aux | grep run_experiments         # Check process

# Cleanup
./remote_workstation/setup.sh clean   # Free space
tmux kill-session -t exp             # End session
```

---

## Support

For issues specific to the remote workstation setup, check:
1. This README
2. The setup script help: `./remote_workstation/setup.sh help`
3. Experiment logs in `logs/`
4. The main project README: `../README.md`

For qMAP-PD data questions, refer to the dataset documentation.