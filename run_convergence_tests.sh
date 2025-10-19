#!/bin/bash
# =============================================================================
# Multi-K Convergence Test Suite
# =============================================================================
# Tests sparse_gfa_fixed model with different K values
# All tests use max_tree_depth=10 and all 5 convergence fixes
#
# Usage:
#   bash run_convergence_tests.sh           # Run all tests sequentially
#   bash run_convergence_tests.sh K2        # Run only K=2 test
#   bash run_convergence_tests.sh K5 K8     # Run K=5 and K=8 tests
# =============================================================================

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to get config file for a test name (compatible with older bash)
get_config() {
    case "$1" in
        K2)  echo "config_convergence_K2_tree10.yaml" ;;
        K5)  echo "config_convergence_K5_tree10.yaml" ;;
        K8)  echo "config_convergence_K8_tree10.yaml" ;;
        K20) echo "config_convergence_K20_tree10.yaml" ;;
        K50) echo "config_convergence_K50_tree10.yaml" ;;
        *)   echo "" ;;
    esac
}

# Function to run a single test
run_test() {
    local test_name=$1
    local config_file=$(get_config "$test_name")

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running Convergence Test: $test_name${NC}"
    echo -e "${BLUE}Config: $config_file${NC}"
    echo -e "${BLUE}========================================${NC}"

    python run_experiments.py \
        --config "$config_file" \
        --experiments factor_stability \
        --select-rois volume_sn_voxels.tsv

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Test $test_name completed successfully${NC}\n"
    else
        echo -e "${YELLOW}✗ Test $test_name failed${NC}\n"
        return 1
    fi
}

# Main execution
if [ $# -eq 0 ]; then
    # No arguments: run all tests
    echo -e "${GREEN}Running ALL convergence tests (K=2,5,8,20,50)${NC}\n"
    for test_name in K2 K5 K8 K20 K50; do
        run_test "$test_name"
    done
else
    # Run specified tests
    echo -e "${GREEN}Running selected tests: $@${NC}\n"
    for test_name in "$@"; do
        config_file=$(get_config "$test_name")
        if [ -n "$config_file" ]; then
            run_test "$test_name"
        else
            echo -e "${YELLOW}Warning: Unknown test '$test_name'. Available: K2, K5, K8, K20, K50${NC}"
        fi
    done
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All requested tests completed!${NC}"
echo -e "${GREEN}========================================${NC}"
