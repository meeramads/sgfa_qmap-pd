#!/usr/bin/env python3
"""
Test script for the new subtype logging functionality.
"""

import numpy as np
import logging
from analysis.cross_validation_library import NeuroImagingCVConfig, NeuroImagingCrossValidator

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def test_subtype_logging():
    """Test the automatic subtype determination with clear logging."""
    print("Testing automatic subtype determination...")

    # Create mock data
    np.random.seed(42)
    n_subjects = 50
    n_factors = 10
    Z_test = np.random.randn(n_subjects, n_factors)

    # Add some structure to make clustering meaningful
    # Simulate 3 groups with different means
    Z_test[:15, :3] += 2.0    # Group 1: high on first 3 factors
    Z_test[15:30, 3:6] += 2.0  # Group 2: high on factors 3-6
    Z_test[30:, 6:9] += 2.0    # Group 3: high on factors 6-9

    # Test the new method
    config = NeuroImagingCVConfig()
    config.auto_optimize_subtypes = True
    config.subtype_candidate_range = [2, 3, 4]

    # Create a mock validator to test the method
    validator = NeuroImagingCrossValidator(config)

    # Test the optimal subtype finding
    candidate_clusters = [2, 3, 4]
    optimal_n, metrics = validator._find_optimal_subtypes(Z_test, candidate_clusters, fold_id=1)

    print(f"\nTest Results:")
    print(f"   Optimal number of subtypes: {optimal_n}")
    print(f"   Best silhouette score: {metrics['best_silhouette']:.4f}")
    print(f"   Cluster sizes: {metrics['cluster_sizes']}")

    return optimal_n, metrics

if __name__ == "__main__":
    try:
        optimal_n, metrics = test_subtype_logging()
        print(f"\nTest completed successfully!")
        print(f"   Found {optimal_n} optimal subtypes")
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        import traceback
        traceback.print_exc()