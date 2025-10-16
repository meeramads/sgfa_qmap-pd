"""MCMC diagnostic plotting utilities for convergence assessment.

This module provides trace plots and diagnostic visualizations for assessing
MCMC chain convergence, including:
- Parameter traces across iterations
- R-hat (Gelman-Rubin) evolution over time
- Effective Sample Size (ESS) diagnostics
- Autocorrelation plots for assessing chain mixing

References:
- Vehtari et al. 2021: "Rank-Normalization, Folding, and Localization:
  An Improved R-hat for Assessing Convergence of MCMC"
- Gelman & Rubin 1992: "Inference from Iterative Simulation Using
  Multiple Sequences"
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)


def compute_ess(samples: np.ndarray, axis: int = 0) -> float:
    """Compute Effective Sample Size using autocorrelation.

    Parameters
    ----------
    samples : np.ndarray
        MCMC samples, shape (n_samples, ...)
    axis : int, default=0
        Axis along which to compute ESS

    Returns
    -------
    float
        Effective sample size
    """
    n_samples = samples.shape[axis]

    # Move sample axis to front
    samples = np.moveaxis(samples, axis, 0)
    original_shape = samples.shape
    samples = samples.reshape(n_samples, -1)

    # Compute autocorrelation for each parameter
    ess_values = []
    for param_idx in range(samples.shape[1]):
        chain = samples[:, param_idx]

        # Demean
        chain = chain - np.mean(chain)

        # Compute autocorrelation
        autocorr = np.correlate(chain, chain, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]

        # Find cutoff where autocorrelation becomes negligible
        # Use first negative autocorrelation as cutoff
        cutoff = np.where(autocorr < 0)[0]
        if len(cutoff) > 0:
            cutoff = cutoff[0]
        else:
            cutoff = len(autocorr)

        # Compute ESS
        if cutoff > 1:
            rho_sum = 1 + 2 * np.sum(autocorr[1:cutoff])
            ess = n_samples / rho_sum if rho_sum > 0 else n_samples
        else:
            ess = n_samples

        ess_values.append(ess)

    return np.mean(ess_values)


def compute_autocorrelation(samples: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """Compute autocorrelation function for MCMC samples.

    Parameters
    ----------
    samples : np.ndarray
        MCMC samples, shape (n_samples,)
    max_lag : int, optional
        Maximum lag to compute. If None, uses n_samples // 2

    Returns
    -------
    np.ndarray
        Autocorrelation values, shape (max_lag + 1,)
    """
    n_samples = len(samples)
    if max_lag is None:
        max_lag = min(n_samples // 2, 100)

    # Demean
    samples = samples - np.mean(samples)

    # Compute autocorrelation
    autocorr = np.correlate(samples, samples, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]

    return autocorr[:max_lag + 1]


def compute_rhat_evolution(
    chains: np.ndarray,
    window_sizes: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute R-hat at different points during sampling.

    This shows how R-hat evolves as more samples are collected,
    helping diagnose convergence issues.

    Parameters
    ----------
    chains : np.ndarray
        MCMC samples from multiple chains, shape (n_chains, n_samples, ...)
    window_sizes : List[int], optional
        Sample sizes at which to compute R-hat. If None, uses
        [100, 250, 500, 1000, 2500, 5000, full_length]

    Returns
    -------
    sample_points : np.ndarray
        Sample sizes at which R-hat was computed
    rhat_values : np.ndarray
        R-hat values at each sample point
    """
    n_chains, n_samples = chains.shape[:2]

    if window_sizes is None:
        # Default windows
        max_samples = n_samples
        window_sizes = [100, 250, 500, 1000, 2500, 5000, max_samples]
        window_sizes = [w for w in window_sizes if w <= max_samples]
        if max_samples not in window_sizes:
            window_sizes.append(max_samples)

    rhat_values = []
    sample_points = []

    for window in window_sizes:
        if window > n_samples:
            continue

        # Take last 'window' samples from each chain
        chain_subset = chains[:, -window:]

        # Compute R-hat
        rhat = compute_rhat(chain_subset)

        rhat_values.append(rhat)
        sample_points.append(window)

    return np.array(sample_points), np.array(rhat_values)


def compute_rhat(chains: np.ndarray) -> float:
    """Compute Gelman-Rubin R-hat statistic for MCMC convergence.

    R-hat < 1.01 indicates excellent convergence
    R-hat < 1.1 is typically acceptable
    R-hat > 1.1 suggests lack of convergence

    Parameters
    ----------
    chains : np.ndarray
        MCMC samples from multiple chains, shape (n_chains, n_samples, ...)

    Returns
    -------
    float
        R-hat statistic (maximum across all parameters)
    """
    n_chains, n_samples = chains.shape[:2]

    # Reshape to (n_chains, n_samples, n_params)
    original_shape = chains.shape
    chains = chains.reshape(n_chains, n_samples, -1)
    n_params = chains.shape[2]

    rhat_per_param = []

    for param_idx in range(n_params):
        param_chains = chains[:, :, param_idx]  # (n_chains, n_samples)

        # Within-chain variance
        W = np.mean(np.var(param_chains, axis=1, ddof=1))

        # Between-chain variance
        chain_means = np.mean(param_chains, axis=1)
        overall_mean = np.mean(chain_means)
        B = n_samples * np.var(chain_means, ddof=1)

        # Pooled variance estimate
        var_plus = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B

        # R-hat
        if W > 0:
            rhat = np.sqrt(var_plus / W)
        else:
            rhat = 1.0

        rhat_per_param.append(rhat)

    # Return maximum R-hat (most conservative)
    return np.max(rhat_per_param)


def plot_trace_diagnostics(
    W_samples: np.ndarray,
    Z_samples: np.ndarray,
    save_path: Optional[str] = None,
    max_factors: int = 4,
    thin: int = 1,
) -> plt.Figure:
    """Create comprehensive MCMC trace diagnostic plots.

    Generates a multi-panel figure with:
    1. Trace plots for W (factor loadings) - first few factors
    2. Trace plots for Z (factor scores) - first few subjects
    3. R-hat evolution over sampling iterations
    4. Autocorrelation plots for chain mixing assessment

    Parameters
    ----------
    W_samples : np.ndarray
        Factor loading samples, shape (n_chains, n_samples, D, K)
    Z_samples : np.ndarray
        Factor score samples, shape (n_chains, n_samples, N, K)
    save_path : str, optional
        Path to save figure. If None, returns figure without saving.
    max_factors : int, default=4
        Maximum number of factors to plot (for readability)
    thin : int, default=1
        Thinning factor for trace plots (plot every nth sample)

    Returns
    -------
    fig : plt.Figure
        The generated figure
    """
    logger.info("Creating MCMC trace diagnostic plots...")

    n_chains, n_samples, D, K = W_samples.shape
    N = Z_samples.shape[2]

    # Limit number of factors to plot
    n_factors_to_plot = min(K, max_factors)

    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

    fig.suptitle(
        f"MCMC Trace Diagnostics ({n_chains} chains, {n_samples} samples)",
        fontsize=16,
        fontweight='bold'
    )

    # Color palette for chains
    colors = sns.color_palette("husl", n_chains)

    # ============================================================================
    # Row 1: W (Factor Loadings) Trace Plots
    # ============================================================================
    logger.info(f"  Creating W trace plots for {n_factors_to_plot} factors...")

    for k in range(n_factors_to_plot):
        ax = fig.add_subplot(gs[0, k % 3])

        # Plot mean loading across features for each chain
        for chain_idx in range(n_chains):
            # Average across features for this factor
            w_trace = np.mean(W_samples[chain_idx, ::thin, :, k], axis=1)
            iterations = np.arange(0, n_samples, thin)

            ax.plot(
                iterations,
                w_trace,
                color=colors[chain_idx],
                alpha=0.7,
                linewidth=1,
                label=f'Chain {chain_idx}'
            )

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Loading')
        ax.set_title(f'W Trace: Factor {k}')
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, loc='best')

    # ============================================================================
    # Row 2: Z (Factor Scores) Trace Plots
    # ============================================================================
    logger.info(f"  Creating Z trace plots for {n_factors_to_plot} factors...")

    for k in range(n_factors_to_plot):
        ax = fig.add_subplot(gs[1, k % 3])

        # Plot mean score across subjects for each chain
        for chain_idx in range(n_chains):
            # Average across subjects for this factor
            z_trace = np.mean(Z_samples[chain_idx, ::thin, :, k], axis=1)
            iterations = np.arange(0, n_samples, thin)

            ax.plot(
                iterations,
                z_trace,
                color=colors[chain_idx],
                alpha=0.7,
                linewidth=1,
                label=f'Chain {chain_idx}'
            )

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Score')
        ax.set_title(f'Z Trace: Factor {k}')
        ax.grid(True, alpha=0.3)

    # ============================================================================
    # Row 3: R-hat Evolution Over Time
    # ============================================================================
    logger.info("  Computing R-hat evolution...")

    # R-hat for W
    ax_w = fig.add_subplot(gs[2, 0])

    # Compute R-hat at different sample sizes
    sample_points, rhat_w_evolution = compute_rhat_evolution(
        W_samples[:, :, :, 0]  # Use first factor for efficiency
    )

    ax_w.plot(sample_points, rhat_w_evolution, 'o-', linewidth=2, markersize=6)
    ax_w.axhline(1.1, color='orange', linestyle='--', label='R-hat = 1.1 (threshold)', linewidth=2)
    ax_w.axhline(1.01, color='green', linestyle='--', label='R-hat = 1.01 (excellent)', linewidth=2)
    ax_w.set_xlabel('Number of Samples')
    ax_w.set_ylabel('R-hat')
    ax_w.set_title('R-hat Evolution: W (Factor 0)')
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(fontsize=8)
    ax_w.set_yscale('log')

    # R-hat for Z
    ax_z = fig.add_subplot(gs[2, 1])

    sample_points, rhat_z_evolution = compute_rhat_evolution(
        Z_samples[:, :, :, 0]  # Use first factor for efficiency
    )

    ax_z.plot(sample_points, rhat_z_evolution, 'o-', linewidth=2, markersize=6)
    ax_z.axhline(1.1, color='orange', linestyle='--', label='R-hat = 1.1 (threshold)', linewidth=2)
    ax_z.axhline(1.01, color='green', linestyle='--', label='R-hat = 1.01 (excellent)', linewidth=2)
    ax_z.set_xlabel('Number of Samples')
    ax_z.set_ylabel('R-hat')
    ax_z.set_title('R-hat Evolution: Z (Factor 0)')
    ax_z.grid(True, alpha=0.3)
    ax_z.legend(fontsize=8)
    ax_z.set_yscale('log')

    # Summary panel: R-hat per factor
    ax_summary = fig.add_subplot(gs[2, 2])

    rhat_per_factor_w = []
    rhat_per_factor_z = []
    for k in range(K):
        rhat_w = compute_rhat(W_samples[:, :, :, k])
        rhat_z = compute_rhat(Z_samples[:, :, :, k])
        rhat_per_factor_w.append(rhat_w)
        rhat_per_factor_z.append(rhat_z)

    x_pos = np.arange(K)
    width = 0.35

    ax_summary.bar(x_pos - width/2, rhat_per_factor_w, width, label='W (loadings)', alpha=0.7)
    ax_summary.bar(x_pos + width/2, rhat_per_factor_z, width, label='Z (scores)', alpha=0.7)
    ax_summary.axhline(1.1, color='orange', linestyle='--', linewidth=2)
    ax_summary.axhline(1.01, color='green', linestyle='--', linewidth=2)
    ax_summary.set_xlabel('Factor Index')
    ax_summary.set_ylabel('R-hat')
    ax_summary.set_title('Final R-hat per Factor')
    ax_summary.set_xticks(x_pos)
    ax_summary.legend(fontsize=8)
    ax_summary.grid(True, alpha=0.3, axis='y')
    ax_summary.set_yscale('log')

    # ============================================================================
    # Row 4: Autocorrelation Plots
    # ============================================================================
    logger.info("  Creating autocorrelation plots...")

    max_lag = min(100, n_samples // 4)

    # Autocorrelation for W (first factor, first chain)
    ax_acf_w = fig.add_subplot(gs[3, 0])

    w_flat = np.mean(W_samples[0, :, :, 0], axis=1)  # Average across features
    acf_w = compute_autocorrelation(w_flat, max_lag=max_lag)

    ax_acf_w.bar(range(len(acf_w)), acf_w, alpha=0.7)
    ax_acf_w.axhline(0, color='black', linewidth=0.8)
    ax_acf_w.set_xlabel('Lag')
    ax_acf_w.set_ylabel('Autocorrelation')
    ax_acf_w.set_title('Autocorrelation: W (Chain 0, Factor 0)')
    ax_acf_w.grid(True, alpha=0.3)

    # Autocorrelation for Z (first factor, first chain)
    ax_acf_z = fig.add_subplot(gs[3, 1])

    z_flat = np.mean(Z_samples[0, :, :, 0], axis=1)  # Average across subjects
    acf_z = compute_autocorrelation(z_flat, max_lag=max_lag)

    ax_acf_z.bar(range(len(acf_z)), acf_z, alpha=0.7)
    ax_acf_z.axhline(0, color='black', linewidth=0.8)
    ax_acf_z.set_xlabel('Lag')
    ax_acf_z.set_ylabel('Autocorrelation')
    ax_acf_z.set_title('Autocorrelation: Z (Chain 0, Factor 0)')
    ax_acf_z.grid(True, alpha=0.3)

    # ESS summary
    ax_ess = fig.add_subplot(gs[3, 2])

    ess_per_factor_w = []
    ess_per_factor_z = []
    for k in range(K):
        # Compute ESS for each chain then average
        ess_w_chains = [compute_ess(W_samples[c, :, :, k]) for c in range(n_chains)]
        ess_z_chains = [compute_ess(Z_samples[c, :, :, k]) for c in range(n_chains)]
        ess_per_factor_w.append(np.mean(ess_w_chains))
        ess_per_factor_z.append(np.mean(ess_z_chains))

    x_pos = np.arange(K)
    width = 0.35

    ax_ess.bar(x_pos - width/2, ess_per_factor_w, width, label='W (loadings)', alpha=0.7)
    ax_ess.bar(x_pos + width/2, ess_per_factor_z, width, label='Z (scores)', alpha=0.7)
    ax_ess.axhline(n_samples, color='green', linestyle='--', label='Total samples', linewidth=2)
    ax_ess.axhline(n_samples * 0.1, color='orange', linestyle='--', label='10% of samples', linewidth=2)
    ax_ess.set_xlabel('Factor Index')
    ax_ess.set_ylabel('Effective Sample Size')
    ax_ess.set_title('ESS per Factor (averaged across chains)')
    ax_ess.set_xticks(x_pos)
    ax_ess.legend(fontsize=8)
    ax_ess.grid(True, alpha=0.3, axis='y')

    logger.info("  ✓ Trace diagnostic plots completed")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  ✓ Saved to {save_path}")

    return fig


def plot_parameter_distributions(
    W_samples: np.ndarray,
    Z_samples: np.ndarray,
    save_path: Optional[str] = None,
    max_factors: int = 4,
) -> plt.Figure:
    """Plot posterior distributions for selected parameters.

    Shows marginal posterior distributions across chains to visualize
    convergence and multimodality issues.

    Parameters
    ----------
    W_samples : np.ndarray
        Factor loading samples, shape (n_chains, n_samples, D, K)
    Z_samples : np.ndarray
        Factor score samples, shape (n_chains, n_samples, N, K)
    save_path : str, optional
        Path to save figure
    max_factors : int, default=4
        Maximum number of factors to plot

    Returns
    -------
    fig : plt.Figure
        The generated figure
    """
    logger.info("Creating parameter distribution plots...")

    n_chains, n_samples, D, K = W_samples.shape
    n_factors_to_plot = min(K, max_factors)

    fig, axes = plt.subplots(2, n_factors_to_plot, figsize=(5*n_factors_to_plot, 8))
    if n_factors_to_plot == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(
        "Posterior Distributions by Chain",
        fontsize=16,
        fontweight='bold'
    )

    colors = sns.color_palette("husl", n_chains)

    # W distributions
    for k in range(n_factors_to_plot):
        ax = axes[0, k]

        for chain_idx in range(n_chains):
            # Flatten W samples for this chain and factor
            w_flat = W_samples[chain_idx, :, :, k].flatten()

            ax.hist(
                w_flat,
                bins=50,
                alpha=0.4,
                color=colors[chain_idx],
                label=f'Chain {chain_idx}',
                density=True
            )

        ax.set_xlabel('Loading Value')
        ax.set_ylabel('Density')
        ax.set_title(f'W Distribution: Factor {k}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Z distributions
    for k in range(n_factors_to_plot):
        ax = axes[1, k]

        for chain_idx in range(n_chains):
            # Flatten Z samples for this chain and factor
            z_flat = Z_samples[chain_idx, :, :, k].flatten()

            ax.hist(
                z_flat,
                bins=50,
                alpha=0.4,
                color=colors[chain_idx],
                label=f'Chain {chain_idx}',
                density=True
            )

        ax.set_xlabel('Score Value')
        ax.set_ylabel('Density')
        ax.set_title(f'Z Distribution: Factor {k}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  ✓ Saved to {save_path}")

    return fig


def plot_hyperparameter_posteriors(
    samples_by_chain: List[Dict],
    save_path: Optional[str] = None,
    num_sources: Optional[int] = None,
) -> plt.Figure:
    """Plot posterior distributions of hyperparameters (tauW, tauZ).

    This visualizes the learned precision parameters to assess whether
    priors are too restrictive or if the model is learning appropriate
    regularization levels.

    Key insights:
    - tauW close to 0: Strong shrinkage (sparse loadings)
    - tauW large: Weak shrinkage (dense loadings)
    - Different tauW per view: Different sparsity levels needed
    - tauZ behavior: Similar interpretation for factor scores

    If posteriors are pushed against prior boundaries, consider relaxing priors.

    Parameters
    ----------
    samples_by_chain : List[Dict]
        List of sample dictionaries from each chain, each containing:
        - "tauW1", "tauW2", ... : tauW samples per view, shape (n_samples, K)
        - "tauZ" : tauZ samples, shape (n_samples, 1, K)
    save_path : str, optional
        Path to save figure
    num_sources : int, optional
        Number of data sources (views). If None, inferred from samples.

    Returns
    -------
    fig : plt.Figure
        The generated figure with hyperparameter posterior distributions
    """
    logger.info("Creating hyperparameter posterior plots...")

    n_chains = len(samples_by_chain)
    colors = sns.color_palette("husl", n_chains)

    # Infer number of sources and factors from samples
    if num_sources is None:
        # Count tauW parameters
        num_sources = sum(
            1 for key in samples_by_chain[0].keys() if key.startswith("tauW")
        )

    # Get number of factors from tauZ shape
    if "tauZ" in samples_by_chain[0]:
        tauZ_shape = samples_by_chain[0]["tauZ"].shape
        K = tauZ_shape[-1] if len(tauZ_shape) > 1 else 1
    else:
        K = 1

    logger.info(f"  Detected {num_sources} data sources, {K} factors")

    # Create figure layout: one row for tauW per view, one row for tauZ
    n_rows = num_sources + 1
    fig, axes = plt.subplots(n_rows, K, figsize=(4*K, 3*n_rows))

    # Handle single factor case
    if K == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(
        "Hyperparameter Posterior Distributions\n"
        "(tauW controls loading sparsity, tauZ controls score sparsity)",
        fontsize=14,
        fontweight='bold'
    )

    # ========================================================================
    # Plot tauW for each view
    # ========================================================================
    for view_idx in range(num_sources):
        param_name = f"tauW{view_idx + 1}"

        for k in range(K):
            ax = axes[view_idx, k]

            # Check if this parameter exists in samples
            has_param = param_name in samples_by_chain[0]

            if has_param:
                for chain_idx in range(n_chains):
                    samples = samples_by_chain[chain_idx][param_name]

                    # Handle different shapes: (n_samples,) or (n_samples, K)
                    if len(samples.shape) == 1:
                        tau_samples = samples
                    else:
                        tau_samples = samples[:, k] if samples.shape[1] > k else samples[:, 0]

                    ax.hist(
                        tau_samples,
                        bins=50,
                        alpha=0.4,
                        color=colors[chain_idx],
                        label=f'Chain {chain_idx}',
                        density=True
                    )

                ax.set_xlabel(r'$\tau_W$ Value')
                ax.set_ylabel('Density')
                ax.set_title(f'tauW (View {view_idx + 1}, Factor {k})', fontsize=10)
                ax.grid(True, alpha=0.3)

                if k == 0:
                    ax.legend(fontsize=7, loc='best')

                # Add vertical line at prior mean/mode if known
                # For TruncatedCauchy, mode is at 0
                ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Prior mode')

            else:
                ax.text(0.5, 0.5, f'{param_name}\nnot found',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

    # ========================================================================
    # Plot tauZ
    # ========================================================================
    for k in range(K):
        ax = axes[num_sources, k]

        if "tauZ" in samples_by_chain[0]:
            for chain_idx in range(n_chains):
                samples = samples_by_chain[chain_idx]["tauZ"]

                # Handle different shapes: (n_samples, 1, K) or (n_samples, K)
                if len(samples.shape) == 3:
                    tau_samples = samples[:, 0, k]
                elif len(samples.shape) == 2:
                    tau_samples = samples[:, k] if samples.shape[1] > k else samples[:, 0]
                else:
                    tau_samples = samples

                ax.hist(
                    tau_samples,
                    bins=50,
                    alpha=0.4,
                    color=colors[chain_idx],
                    label=f'Chain {chain_idx}',
                    density=True
                )

            ax.set_xlabel(r'$\tau_Z$ Value')
            ax.set_ylabel('Density')
            ax.set_title(f'tauZ (Factor {k})', fontsize=10)
            ax.grid(True, alpha=0.3)

            if k == 0:
                ax.legend(fontsize=7, loc='best')

            # Add vertical line at prior mode
            ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Prior mode')

        else:
            ax.text(0.5, 0.5, 'tauZ\nnot found',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  ✓ Saved to {save_path}")

    logger.info("  ✓ Hyperparameter posterior plots completed")

    return fig


def plot_hyperparameter_traces(
    samples_by_chain: List[Dict],
    save_path: Optional[str] = None,
    num_sources: Optional[int] = None,
    thin: int = 1,
) -> plt.Figure:
    """Plot trace plots for hyperparameters (tauW, tauZ).

    Trace plots show how hyperparameters evolve across MCMC iterations,
    helping diagnose convergence issues specific to hyperparameters.

    Parameters
    ----------
    samples_by_chain : List[Dict]
        List of sample dictionaries from each chain
    save_path : str, optional
        Path to save figure
    num_sources : int, optional
        Number of data sources (views). If None, inferred from samples.
    thin : int, default=1
        Thinning factor (plot every nth sample)

    Returns
    -------
    fig : plt.Figure
        The generated figure with hyperparameter trace plots
    """
    logger.info("Creating hyperparameter trace plots...")

    n_chains = len(samples_by_chain)
    colors = sns.color_palette("husl", n_chains)

    # Infer number of sources
    if num_sources is None:
        num_sources = sum(
            1 for key in samples_by_chain[0].keys() if key.startswith("tauW")
        )

    # Get number of factors and samples
    if "tauZ" in samples_by_chain[0]:
        tauZ_shape = samples_by_chain[0]["tauZ"].shape
        K = tauZ_shape[-1] if len(tauZ_shape) > 1 else 1
        n_samples = tauZ_shape[0]
    else:
        K = 1
        n_samples = 1000  # fallback

    logger.info(f"  Detected {num_sources} views, {K} factors, {n_samples} samples")

    # Create figure: all tauW in top rows, tauZ in bottom row
    n_rows = num_sources + 1
    fig, axes = plt.subplots(n_rows, K, figsize=(5*K, 2.5*n_rows))

    if K == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(
        "Hyperparameter Trace Plots (Convergence Assessment)",
        fontsize=14,
        fontweight='bold'
    )

    iterations = np.arange(0, n_samples, thin)

    # ========================================================================
    # Plot tauW traces for each view
    # ========================================================================
    for view_idx in range(num_sources):
        param_name = f"tauW{view_idx + 1}"

        for k in range(K):
            ax = axes[view_idx, k]

            if param_name in samples_by_chain[0]:
                for chain_idx in range(n_chains):
                    samples = samples_by_chain[chain_idx][param_name]

                    # Extract samples for this factor
                    if len(samples.shape) == 1:
                        tau_trace = samples[::thin]
                    else:
                        tau_trace = samples[::thin, k] if samples.shape[1] > k else samples[::thin, 0]

                    ax.plot(
                        iterations[:len(tau_trace)],
                        tau_trace,
                        color=colors[chain_idx],
                        alpha=0.7,
                        linewidth=1,
                        label=f'Chain {chain_idx}'
                    )

                ax.set_xlabel('Iteration')
                ax.set_ylabel(r'$\tau_W$ Value')
                ax.set_title(f'tauW Trace (View {view_idx + 1}, Factor {k})', fontsize=10)
                ax.grid(True, alpha=0.3)

                if k == 0 and view_idx == 0:
                    ax.legend(fontsize=7, loc='best')

            else:
                ax.text(0.5, 0.5, f'{param_name}\nnot found',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

    # ========================================================================
    # Plot tauZ traces
    # ========================================================================
    for k in range(K):
        ax = axes[num_sources, k]

        if "tauZ" in samples_by_chain[0]:
            for chain_idx in range(n_chains):
                samples = samples_by_chain[chain_idx]["tauZ"]

                # Extract samples for this factor
                if len(samples.shape) == 3:
                    tau_trace = samples[::thin, 0, k]
                elif len(samples.shape) == 2:
                    tau_trace = samples[::thin, k] if samples.shape[1] > k else samples[::thin, 0]
                else:
                    tau_trace = samples[::thin]

                ax.plot(
                    iterations[:len(tau_trace)],
                    tau_trace,
                    color=colors[chain_idx],
                    alpha=0.7,
                    linewidth=1,
                    label=f'Chain {chain_idx}'
                )

            ax.set_xlabel('Iteration')
            ax.set_ylabel(r'$\tau_Z$ Value')
            ax.set_title(f'tauZ Trace (Factor {k})', fontsize=10)
            ax.grid(True, alpha=0.3)

        else:
            ax.text(0.5, 0.5, 'tauZ\nnot found',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  ✓ Saved to {save_path}")

    logger.info("  ✓ Hyperparameter trace plots completed")

    return fig
