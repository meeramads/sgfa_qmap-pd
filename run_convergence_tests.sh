#!/bin/bash
# =============================================================================
# Multi-K Convergence Test Suite
# =============================================================================
# Tests sparse_gfa_fixed model with different K values
# Uses single config_convergence.yaml with command-line flag overrides
#
# Usage:
#   bash run_convergence_tests.sh           # Run all tests (K=2,5,8,20,50)
#   bash run_convergence_tests.sh 2 5 8     # Run specific K values
# =============================================================================

# Test configuration
CONFIG_FILE="config_convergence.yaml"
MAX_TREE_DEPTH=10
MAD_THRESHOLD=3.0
TARGET_ACCEPT=0.95
PERCW=33

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to run a single test
run_test() {
    local K_value=$1

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running Convergence Test: K=${K_value}${NC}"
    echo -e "${BLUE}Config: $CONFIG_FILE${NC}"
    echo -e "${BLUE}Flags: --K $K_value --max-tree-depth $MAX_TREE_DEPTH --qc-outlier-threshold $MAD_THRESHOLD${NC}"
    echo -e "${BLUE}========================================${NC}"

    python run_experiments.py \
        --config "$CONFIG_FILE" \
        --experiments factor_stability \
        --select-rois volume_sn_voxels.tsv \
        --K "$K_value" \
        --percW "$PERCW" \
        --max-tree-depth "$MAX_TREE_DEPTH" \
        --qc-outlier-threshold "$MAD_THRESHOLD" \
        --target-accept-prob "$TARGET_ACCEPT"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Test K=$K_value completed successfully${NC}\n"
    else
        echo -e "${YELLOW}✗ Test K=$K_value failed${NC}\n"
        return 1
    fi
}

# Main execution
if [ $# -eq 0 ]; then
    # No arguments: run all tests
    echo -e "${GREEN}Running ALL convergence tests (K=2,5,8,20,50)${NC}"
    echo -e "${GREEN}Settings: max_tree_depth=$MAX_TREE_DEPTH, MAD=$MAD_THRESHOLD, percW=$PERCW${NC}\n"
    for K in 2 5 8 20 50; do
        run_test "$K"
    done
else
    # Run specified K values
    echo -e "${GREEN}Running selected tests: $@${NC}\n"
    for K_value in "$@"; do
        # Remove "K" prefix if present
        K_value="${K_value#K}"

        if [[ "$K_value" =~ ^[0-9]+$ ]]; then
            run_test "$K_value"
        else
            echo -e "${YELLOW}Warning: Invalid K value '$K_value'${NC}"
        fi
    done
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All requested tests completed!${NC}"
echo -e "${GREEN}========================================${NC}"
