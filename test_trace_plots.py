"""Test script for MCMC trace plot generation.

This script creates synthetic MCMC samples to test the trace plot
functionality without requiring full experimental data.
"""

import numpy as np
import matplotlib.pyplot as plt
from analysis.mcmc_diagnostics import (
    plot_trace_diagnostics,
    plot_parameter_distributions,
    plot_hyperparameter_posteriors,
    plot_hyperparameter_traces,
    compute_rhat,
    compute_ess,
)

def generate_synthetic_mcmc_samples(
    n_chains=4,
    n_samples=1000,
    D=100,
    K=2,
    N=50,
    converged=False,
):
    """Generate synthetic MCMC samples for testing.

    Parameters
    ----------
    n_chains : int
        Number of MCMC chains
    n_samples : int
        Number of samples per chain
    D : int
        Number of features
    K : int
        Number of factors
    N : int
        Number of subjects
    converged : bool
        If True, generate converged chains. If False, generate
        chains with convergence issues (different modes).

    Returns
    -------
    W_samples : np.ndarray
        Shape (n_chains, n_samples, D, K)
    Z_samples : np.ndarray
        Shape (n_chains, n_samples, N, K)
    """
    np.random.seed(42)

    W_samples = []
    Z_samples = []

    for chain_idx in range(n_chains):
        if converged:
            # All chains sample from the same distribution (converged)
            W_mean = np.random.randn(D, K) * 0.5
            Z_mean = np.random.randn(N, K) * 0.5
        else:
            # Each chain samples from different modes (not converged)
            W_mean = np.random.randn(D, K) * 0.5 + chain_idx * 2.0
            Z_mean = np.random.randn(N, K) * 0.5 + chain_idx * 1.5

        # Generate samples with some autocorrelation
        W_chain = []
        Z_chain = []

        W_current = W_mean.copy()
        Z_current = Z_mean.copy()

        for i in range(n_samples):
            # Random walk with drift back to mean (autocorrelated)
            W_current = 0.95 * W_current + 0.05 * W_mean + np.random.randn(D, K) * 0.1
            Z_current = 0.95 * Z_current + 0.05 * Z_mean + np.random.randn(N, K) * 0.1

            W_chain.append(W_current.copy())
            Z_chain.append(Z_current.copy())

        W_samples.append(np.array(W_chain))
        Z_samples.append(np.array(Z_chain))

    W_samples = np.array(W_samples)  # (n_chains, n_samples, D, K)
    Z_samples = np.array(Z_samples)  # (n_chains, n_samples, N, K)

    # Generate synthetic hyperparameter samples
    samples_by_chain = []
    for chain_idx in range(n_chains):
        chain_samples = {}

        # tauW for 2 views (imaging and clinical)
        if converged:
            # Converged: all chains sample similar values
            tauW1_mean = 0.5
            tauW2_mean = 0.3
        else:
            # Non-converged: different values per chain
            tauW1_mean = 0.5 + chain_idx * 0.2
            tauW2_mean = 0.3 + chain_idx * 0.15

        # Generate autocorrelated samples
        tauW1_samples = []
        tauW2_samples = []
        tauZ_samples = []

        tauW1_current = tauW1_mean
        tauW2_current = tauW2_mean
        tauZ_current = np.ones(K) * 0.4 if converged else np.ones(K) * (0.4 + chain_idx * 0.2)

        for i in range(n_samples):
            # Random walk for hyperparameters
            tauW1_current = 0.95 * tauW1_current + 0.05 * tauW1_mean + np.random.randn() * 0.05
            tauW2_current = 0.95 * tauW2_current + 0.05 * tauW2_mean + np.random.randn() * 0.05
            tauZ_current = 0.95 * tauZ_current + 0.05 * (0.4 if converged else 0.4 + chain_idx * 0.2) + np.random.randn(K) * 0.05

            # Ensure positive values (truncated)
            tauW1_current = max(0.01, tauW1_current)
            tauW2_current = max(0.01, tauW2_current)
            tauZ_current = np.maximum(0.01, tauZ_current)

            tauW1_samples.append(tauW1_current)
            tauW2_samples.append(tauW2_current)
            tauZ_samples.append(tauZ_current.copy())

        chain_samples["tauW1"] = np.array(tauW1_samples)
        chain_samples["tauW2"] = np.array(tauW2_samples)
        chain_samples["tauZ"] = np.array(tauZ_samples).reshape(n_samples, 1, K)
        chain_samples["W"] = W_samples[chain_idx]
        chain_samples["Z"] = Z_samples[chain_idx]

        samples_by_chain.append(chain_samples)

    return W_samples, Z_samples, samples_by_chain


def test_trace_plots():
    """Test trace plot generation with synthetic data."""
    print("=" * 80)
    print("Testing MCMC Trace Plot Generation")
    print("=" * 80)

    # Test 1: Converged chains
    print("\nTest 1: Generating converged chains...")
    W_conv, Z_conv, samples_conv = generate_synthetic_mcmc_samples(
        n_chains=4,
        n_samples=1000,
        D=100,
        K=2,
        N=50,
        converged=True,
    )

    print(f"  W_samples shape: {W_conv.shape}")
    print(f"  Z_samples shape: {Z_conv.shape}")

    # Compute R-hat
    rhat_w = compute_rhat(W_conv)
    rhat_z = compute_rhat(Z_conv)
    print(f"  R-hat (W): {rhat_w:.4f}")
    print(f"  R-hat (Z): {rhat_z:.4f}")

    # Compute ESS
    ess_w = compute_ess(W_conv[:, :, :, 0])
    ess_z = compute_ess(Z_conv[:, :, :, 0])
    print(f"  ESS (W, factor 0): {ess_w:.1f}")
    print(f"  ESS (Z, factor 0): {ess_z:.1f}")

    print("\nCreating trace diagnostic plots for converged chains...")
    fig1 = plot_trace_diagnostics(
        W_samples=W_conv,
        Z_samples=Z_conv,
        save_path="/Users/meera/Desktop/sgfa_qmap-pd/test_trace_converged.png",
        max_factors=2,
        thin=1,
    )
    print("  ✓ Saved to test_trace_converged.png")

    print("\nCreating parameter distribution plots for converged chains...")
    fig2 = plot_parameter_distributions(
        W_samples=W_conv,
        Z_samples=Z_conv,
        save_path="/Users/meera/Desktop/sgfa_qmap-pd/test_distributions_converged.png",
        max_factors=2,
    )
    print("  ✓ Saved to test_distributions_converged.png")

    print("\nCreating hyperparameter posterior plots for converged chains...")
    fig3 = plot_hyperparameter_posteriors(
        samples_by_chain=samples_conv,
        save_path="/Users/meera/Desktop/sgfa_qmap-pd/test_hyperparameter_posteriors_converged.png",
        num_sources=2,
    )
    print("  ✓ Saved to test_hyperparameter_posteriors_converged.png")

    print("\nCreating hyperparameter trace plots for converged chains...")
    fig4 = plot_hyperparameter_traces(
        samples_by_chain=samples_conv,
        save_path="/Users/meera/Desktop/sgfa_qmap-pd/test_hyperparameter_traces_converged.png",
        num_sources=2,
        thin=1,
    )
    print("  ✓ Saved to test_hyperparameter_traces_converged.png")

    # Test 2: Non-converged chains
    print("\n" + "=" * 80)
    print("Test 2: Generating non-converged chains...")
    W_nonconv, Z_nonconv, samples_nonconv = generate_synthetic_mcmc_samples(
        n_chains=4,
        n_samples=1000,
        D=100,
        K=2,
        N=50,
        converged=False,
    )

    print(f"  W_samples shape: {W_nonconv.shape}")
    print(f"  Z_samples shape: {Z_nonconv.shape}")

    # Compute R-hat
    rhat_w = compute_rhat(W_nonconv)
    rhat_z = compute_rhat(Z_nonconv)
    print(f"  R-hat (W): {rhat_w:.4f} {'⚠️ High!' if rhat_w > 1.1 else '✓ Good'}")
    print(f"  R-hat (Z): {rhat_z:.4f} {'⚠️ High!' if rhat_z > 1.1 else '✓ Good'}")

    # Compute ESS
    ess_w = compute_ess(W_nonconv[:, :, :, 0])
    ess_z = compute_ess(Z_nonconv[:, :, :, 0])
    print(f"  ESS (W, factor 0): {ess_w:.1f}")
    print(f"  ESS (Z, factor 0): {ess_z:.1f}")

    print("\nCreating trace diagnostic plots for non-converged chains...")
    fig3 = plot_trace_diagnostics(
        W_samples=W_nonconv,
        Z_samples=Z_nonconv,
        save_path="/Users/meera/Desktop/sgfa_qmap-pd/test_trace_nonconverged.png",
        max_factors=2,
        thin=1,
    )
    print("  ✓ Saved to test_trace_nonconverged.png")

    print("\nCreating parameter distribution plots for non-converged chains...")
    fig6 = plot_parameter_distributions(
        W_samples=W_nonconv,
        Z_samples=Z_nonconv,
        save_path="/Users/meera/Desktop/sgfa_qmap-pd/test_distributions_nonconverged.png",
        max_factors=2,
    )
    print("  ✓ Saved to test_distributions_nonconverged.png")

    print("\nCreating hyperparameter posterior plots for non-converged chains...")
    fig7 = plot_hyperparameter_posteriors(
        samples_by_chain=samples_nonconv,
        save_path="/Users/meera/Desktop/sgfa_qmap-pd/test_hyperparameter_posteriors_nonconverged.png",
        num_sources=2,
    )
    print("  ✓ Saved to test_hyperparameter_posteriors_nonconverged.png")

    print("\nCreating hyperparameter trace plots for non-converged chains...")
    fig8 = plot_hyperparameter_traces(
        samples_by_chain=samples_nonconv,
        save_path="/Users/meera/Desktop/sgfa_qmap-pd/test_hyperparameter_traces_nonconverged.png",
        num_sources=2,
        thin=1,
    )
    print("  ✓ Saved to test_hyperparameter_traces_nonconverged.png")

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  Converged chains:")
    print("    - test_trace_converged.png")
    print("    - test_distributions_converged.png")
    print("    - test_hyperparameter_posteriors_converged.png")
    print("    - test_hyperparameter_traces_converged.png")
    print("\n  Non-converged chains (shows convergence issues):")
    print("    - test_trace_nonconverged.png")
    print("    - test_distributions_nonconverged.png")
    print("    - test_hyperparameter_posteriors_nonconverged.png")
    print("    - test_hyperparameter_traces_nonconverged.png")

    plt.close('all')


if __name__ == "__main__":
    test_trace_plots()
