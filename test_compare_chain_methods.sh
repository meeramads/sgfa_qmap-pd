#!/bin/bash
# Compare parallel vs sequential chain execution
# Run this on the GPU-enabled workstation

echo "=========================================="
echo "Comparing Chain Execution Methods"
echo "=========================================="
echo ""

# Show GPU/device info
echo "Checking available devices..."
python -c "import jax; print(f'JAX devices: {jax.devices()}'); print(f'Device count: {jax.device_count()}')"
echo ""

# Test 1: Sequential
echo "=========================================="
echo "TEST 1: Sequential Chain Method"
echo "=========================================="
echo "Configuration: K=5, samples=100, warmup=50, 4 chains, chain_method='sequential'"
echo ""

time python run_experiments.py \
    --config config_test_sequential.yaml \
    --select-rois volume_sn_voxels.tsv \
    --experiments factor_stability \
    2>&1 | tee test_sequential_output.log

echo ""
echo "Sequential test complete. Pausing for 10 seconds..."
sleep 10
echo ""

# Test 2: Parallel
echo "=========================================="
echo "TEST 2: Parallel Chain Method"
echo "=========================================="
echo "Configuration: K=5, samples=100, warmup=50, 4 chains, chain_method='parallel'"
echo ""

time python run_experiments.py \
    --config config_test_parallel.yaml \
    --select-rois volume_sn_voxels.tsv \
    --experiments factor_stability \
    2>&1 | tee test_parallel_output.log

echo ""
echo "=========================================="
echo "Tests Complete"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - test_sequential_output.log (chain_method='sequential')"
echo "  - test_parallel_output.log (chain_method='parallel')"
echo ""
echo "Look for:"
echo "  1. Device warnings about 'not enough devices to run parallel chains'"
echo "  2. Execution time differences between the two methods"
echo "  3. Whether both methods complete successfully with 4 chains"
