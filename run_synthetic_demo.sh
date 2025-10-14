#!/bin/bash
# Run factor stability analysis on synthetic data for demonstration
# This provides ground truth validation of the pipeline with known K=3 factors

echo "=========================================="
echo "SGFA Factor Stability - Synthetic Data Demo"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Dataset: Synthetic (N=150, K_true=3)"
echo "  - Model K: 10 (should shrink to ~3)"
echo "  - Chains: 4 (for stability assessment)"
echo "  - MCMC: 3000 samples, 500 warmup"
echo ""
echo "Expected outcomes:"
echo "  1. Model should shrink from K=10 to ~3-5 effective factors"
echo "  2. High factor stability (>80%) due to clean synthetic data"
echo "  3. Ground truth available for validation"
echo ""
echo "Starting experiment..."
echo ""

python run_experiments.py \
    --config config_synthetic.yaml \
    --experiments factor_stability \
    --unified-results

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "Check results/ directory for outputs"
echo "=========================================="
