#!/bin/bash
# Lightweight test script for parallel chain execution
# Run this on the GPU-enabled workstation

echo "=========================================="
echo "Testing Parallel Chain Execution"
echo "=========================================="
echo ""

# Show GPU/device info
echo "Checking available devices..."
python -c "import jax; print(f'JAX devices: {jax.devices()}'); print(f'Device count: {jax.device_count()}')"
echo ""

# Run lightweight factor stability test
echo "Running factor stability with 4 parallel chains..."
echo "Configuration: K=5, samples=100, warmup=50"
echo ""

time python run_experiments.py \
    --config config_test_parallel.yaml \
    --select-rois volume_sn_voxels.tsv \
    --experiments factor_stability \
    2>&1 | tee test_parallel_output.log

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
echo ""
echo "Check test_parallel_output.log for full output"
echo "Look for warnings about 'not enough devices to run parallel chains'"
