#!/bin/bash
# Remote Workstation Setup Script
# Run this when you connect to the remote GPU workstation

set -e  # Exit on any error

echo "Setting up SGFA qMAP-PD Framework on Remote Workstation"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo " Error: Run this script from the sgfa_qmap-pd directory"
    exit 1
fi

# Setup Python environment (assuming Python 3.8+ available)
echo " Setting up Python environment..."

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install JAX with CUDA support (for remote GPUs)
echo "Installing JAX with CUDA support..."
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify installations
echo ""
echo " Verifying installations..."
python -c "
import jax
import numpyro
print(f' JAX version: {jax.__version__}')
print(f' JAX backend: {jax.lib.xla_bridge.get_backend().platform}')
print(f' JAX devices: {jax.devices()}')
print(f' NumPyro version: {numpyro.__version__}')
"

# Test framework imports
echo ""
echo " Testing framework imports..."
python -c "
import sys
sys.path.insert(0, '.')
try:
    from experiments.framework import ExperimentFramework
    from experiments.data_validation import DataValidationExperiments
    from experiments.method_comparison import MethodComparisonExperiments  
    from experiments.performance_benchmarks import PerformanceBenchmarkExperiments
    from models import create_model
    from data.qmap_pd import load_qmap_pd
    print(' All framework imports successful!')
except Exception as e:
    print(f' Import error: {e}')
    exit(1)
"

# Create output directories
echo ""
echo " Creating output directories..."
mkdir -p remote_workstation_experiment_results
mkdir -p checkpoints
mkdir -p logs

# Check GPU availability
echo ""
echo " GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader,nounits
else
    echo "nvidia-smi not available - check GPU setup"
fi

# Set up data directory (you'll need to update this path)
echo ""
echo " Data Directory Setup:"
if [ ! -d "/data/qMAP-PD_data" ]; then
    echo "  Warning: Update the data path in remote_workstation/config.yaml"
    echo "   Current config expects: /data/qMAP-PD_data"
    echo "   Update this to your actual data location"
else
    echo " Data directory found: /data/qMAP-PD_data"
fi

echo ""
echo " Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update data path in remote_workstation/config.yaml if needed"
echo "2. Run experiments with: python remote_workstation/run_experiments.py"
echo "3. Monitor progress in: tail -f logs/remote_workstation_experiments.log"
echo ""
echo "To activate environment later: source venv/bin/activate"