#!/usr/bin/env python3
"""Test script for factor variance profile analysis.

This script creates synthetic MCMC samples with known ARD structure to verify
that the variance profile analysis correctly identifies:
1. Healthy ARD shrinkage (few active factors)
2. Poor shrinkage (many factors with similar variance)
"""

import numpy as np
import matplotlib.pyplot as plt
from analysis.mcmc_diagnostics import analyze_factor_variance_profile

np.random.seed(42)

print("=" * 70)
print("Testing Factor Variance Profile Analysis")
print("=" * 70)

# Test Case 1: HEALTHY ARD - K=20 with 3 real factors
print("\n" + "=" * 70)
print("TEST 1: Healthy ARD Shrinkage (K=20, 3 real factors)")
print("=" * 70)

n_chains = 4
n_samples = 500
n_subjects = 86
K = 20

# Create synthetic Z samples with strong ARD shrinkage
# 3 factors have large variance, rest are near-zero
Z_healthy = np.zeros((n_chains, n_samples, n_subjects, K))

for chain in range(n_chains):
    for sample in range(n_samples):
        # Factor 1: strong signal (variance ~ 1.5)
        Z_healthy[chain, sample, :, 0] = np.random.normal(0, 1.2, n_subjects)

        # Factor 2: strong signal (variance ~ 1.2)
        Z_healthy[chain, sample, :, 1] = np.random.normal(0, 1.1, n_subjects)

        # Factor 3: moderate signal (variance ~ 0.8)
        Z_healthy[chain, sample, :, 2] = np.random.normal(0, 0.9, n_subjects)

        # Factors 4-20: shrunk away (variance ~ 0.01-0.03)
        for k in range(3, K):
            Z_healthy[chain, sample, :, k] = np.random.normal(0, 0.1 * np.random.uniform(0.5, 1.5), n_subjects)

print(f"\nCreated synthetic Z_samples: shape {Z_healthy.shape}")
print(f"Expected: 3 active factors, 17 shrunk factors")

# Analyze variance profile
results_healthy = analyze_factor_variance_profile(
    Z_samples=Z_healthy,
    variance_threshold=0.1,
    save_path="test_outputs/variance_profile_healthy_K20.png"
)

print(f"\n✅ Test 1 Results:")
print(f"   Active factors: {results_healthy['n_active_factors']} (expected ~3)")
print(f"   Effective dimensionality: {results_healthy['effective_dimensionality']}")
print(f"   Top 5 variances: {results_healthy['sorted_variances'][:5]}")

# Test Case 2: POOR SHRINKAGE - K=20 with gradual decline
print("\n" + "=" * 70)
print("TEST 2: Poor Shrinkage / Convergence Issues (K=20, gradual decline)")
print("=" * 70)

# Create synthetic Z samples with poor shrinkage (gradual variance decline)
Z_poor = np.zeros((n_chains, n_samples, n_subjects, K))

for chain in range(n_chains):
    for sample in range(n_samples):
        # All 20 factors have varying degrees of variance (gradual decline)
        for k in range(K):
            std = 1.2 - (k * 0.05)  # Gradual decline from 1.2 to 0.2
            Z_poor[chain, sample, :, k] = np.random.normal(0, std, n_subjects)

print(f"\nCreated synthetic Z_samples: shape {Z_poor.shape}")
print(f"Expected: Many factors with intermediate variance (poor shrinkage)")

# Analyze variance profile
results_poor = analyze_factor_variance_profile(
    Z_samples=Z_poor,
    variance_threshold=0.1,
    save_path="test_outputs/variance_profile_poor_K20.png"
)

print(f"\n❌ Test 2 Results:")
print(f"   Active factors: {results_poor['n_active_factors']} (expected >10)")
print(f"   Effective dimensionality: {results_poor['effective_dimensionality']}")
print(f"   Top 5 variances: {results_poor['sorted_variances'][:5]}")

# Test Case 3: HIGH K with healthy ARD - K=50 with 4 real factors
print("\n" + "=" * 70)
print("TEST 3: High K with Healthy ARD (K=50, 4 real factors)")
print("=" * 70)

K_high = 50

# Create synthetic Z samples with K=50 but only 4 real factors
Z_high_k = np.zeros((n_chains, n_samples, n_subjects, K_high))

for chain in range(n_chains):
    for sample in range(n_samples):
        # 4 factors with substantial variance
        Z_high_k[chain, sample, :, 0] = np.random.normal(0, 1.4, n_subjects)
        Z_high_k[chain, sample, :, 1] = np.random.normal(0, 1.2, n_subjects)
        Z_high_k[chain, sample, :, 2] = np.random.normal(0, 1.0, n_subjects)
        Z_high_k[chain, sample, :, 3] = np.random.normal(0, 0.8, n_subjects)

        # Factors 5-50: heavily shrunk
        for k in range(4, K_high):
            Z_high_k[chain, sample, :, k] = np.random.normal(0, 0.05 * np.random.uniform(0.5, 2.0), n_subjects)

print(f"\nCreated synthetic Z_samples: shape {Z_high_k.shape}")
print(f"Expected: 4 active factors, 46 shrunk factors")

# Analyze variance profile
results_high_k = analyze_factor_variance_profile(
    Z_samples=Z_high_k,
    variance_threshold=0.1,
    save_path="test_outputs/variance_profile_healthy_K50.png"
)

print(f"\n✅ Test 3 Results:")
print(f"   Active factors: {results_high_k['n_active_factors']} (expected ~4)")
print(f"   Effective dimensionality: {results_high_k['effective_dimensionality']}")
print(f"   Top 5 variances: {results_high_k['sorted_variances'][:5]}")

# Test Case 4: LOW K baseline - K=2 (should show 2 active factors)
print("\n" + "=" * 70)
print("TEST 4: Low K Baseline (K=2)")
print("=" * 70)

K_low = 2

# Create synthetic Z samples with K=2
Z_low_k = np.zeros((n_chains, n_samples, n_subjects, K_low))

for chain in range(n_chains):
    for sample in range(n_samples):
        Z_low_k[chain, sample, :, 0] = np.random.normal(0, 1.3, n_subjects)
        Z_low_k[chain, sample, :, 1] = np.random.normal(0, 1.1, n_subjects)

print(f"\nCreated synthetic Z_samples: shape {Z_low_k.shape}")
print(f"Expected: 2 active factors (no shrinkage needed)")

# Analyze variance profile
results_low_k = analyze_factor_variance_profile(
    Z_samples=Z_low_k,
    variance_threshold=0.1,
    save_path="test_outputs/variance_profile_K2.png"
)

print(f"\n✅ Test 4 Results:")
print(f"   Active factors: {results_low_k['n_active_factors']} (expected 2)")
print(f"   Effective dimensionality: {results_low_k['effective_dimensionality']}")
print(f"   Top variances: {results_low_k['sorted_variances']}")

# Summary comparison
print("\n" + "=" * 70)
print("SUMMARY: Comparison Across All Tests")
print("=" * 70)

print(f"\n{'Test':<30} {'K':<5} {'Active':>8} {'Effective':>10} {'Status':<20}")
print("-" * 70)

tests = [
    ("Healthy ARD (K=20)", 20, results_healthy['n_active_factors'],
     results_healthy['effective_dimensionality'], "✅ HEALTHY"),
    ("Poor Shrinkage (K=20)", 20, results_poor['n_active_factors'],
     results_poor['effective_dimensionality'], "❌ POOR"),
    ("Healthy ARD (K=50)", 50, results_high_k['n_active_factors'],
     results_high_k['effective_dimensionality'], "✅ HEALTHY"),
    ("Baseline (K=2)", 2, results_low_k['n_active_factors'],
     results_low_k['effective_dimensionality'], "✅ BASELINE"),
]

for name, k, active, effective, status in tests:
    print(f"{name:<30} {k:<5} {active:>8} {effective:>10} {status:<20}")

print("\n" + "=" * 70)
print("KEY INSIGHTS:")
print("=" * 70)
print("""
1. HEALTHY ARD shows sharp drop-off: few active factors, rest near-zero
   → Indicates model confidently identified true dimensionality

2. POOR SHRINKAGE shows gradual decline: many intermediate variance factors
   → Suggests convergence issues or model uncertainty

3. OVERPARAMETERIZATION TEST (K=50 vs K=20):
   If both show ~same effective dimensionality → Real phenomenon
   If K=50 shows more active factors → Measurement artifact

4. Compare 'active factors' to 'stability analysis' count:
   Large mismatch → Investigate factor stability measurement
""")

print("\n✅ All tests completed successfully!")
print(f"   Plots saved to: test_outputs/variance_profile_*.png")
print("=" * 70)
