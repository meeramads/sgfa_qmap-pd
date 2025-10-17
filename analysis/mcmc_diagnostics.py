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
    """Plot posterior distributions of hyperparameters (tauW, tauZ, sigma, cW, cZ).

    This visualizes the learned precision and regularization parameters to assess whether
    priors are too restrictive or if the model is learning appropriate
    regularization levels.

    Key insights:
    - tauW close to 0: Strong shrinkage (sparse loadings)
    - tauW large: Weak shrinkage (dense loadings)
    - Different tauW per view: Different sparsity levels needed
    - tauZ behavior: Similar interpretation for factor scores
    - sigma: Noise precision parameter (1/variance) per view
      - High sigma: Low noise (high precision)
      - Low sigma: High noise (low precision)
    - cW: Slab scale for loadings (regularization strength per view/factor)
    - cZ: Slab scale for factors (regularization strength per factor)

    If posteriors are pushed against prior boundaries, consider relaxing priors.

    Parameters
    ----------
    samples_by_chain : List[Dict]
        List of sample dictionaries from each chain, each containing:
        - "tauW1", "tauW2", ... : tauW samples per view, shape (n_samples, K)
        - "tauZ" : tauZ samples, shape (n_samples, 1, K)
        - "sigma" : sigma samples, shape (n_samples, 1, M) where M is number of views
        - "cW" : cW samples (slab scale), shape (n_samples, M, K)
        - "cZ" : cZ samples (slab scale), shape (n_samples, 1, K) [optional]
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

    # Check if optional parameters are available
    has_sigma = "sigma" in samples_by_chain[0]
    has_cW = "cW" in samples_by_chain[0]
    has_cZ = "cZ" in samples_by_chain[0]

    if has_cW:
        logger.info("  Including cW (slab scale for loadings) posteriors...")
    if has_cZ:
        logger.info("  Including cZ (slab scale for factors) posteriors...")

    # Calculate number of rows needed
    # Base: num_sources rows for tauW + 1 row for tauZ
    # Optional: +1 for sigma, +num_sources for cW, +1 for cZ
    n_rows = num_sources + 1
    if has_sigma:
        n_rows += 1
    if has_cW:
        n_rows += num_sources
    if has_cZ:
        n_rows += 1

    # Determine column count
    n_cols = max(K, num_sources)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))

    # Handle single factor/view cases
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)

    title_text = "Hyperparameter Posterior Distributions\n"
    title_text += "(tauW/tauZ: global shrinkage, sigma: noise precision"
    if has_cW or has_cZ:
        title_text += ", cW/cZ: slab regularization"
    title_text += ")"

    fig.suptitle(
        title_text,
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

    # ========================================================================
    # Plot sigma (noise precision) for each view
    # ========================================================================
    if has_sigma:
        logger.info("  Including sigma (noise precision) posteriors...")
        for m in range(num_sources):
            ax_idx = num_sources + 1  # Row index for sigma plots
            col_idx = m  # Column index (one per view)

            ax = axes[ax_idx, col_idx] if axes.ndim == 2 else axes[ax_idx]

            for chain_idx in range(n_chains):
                samples = samples_by_chain[chain_idx]["sigma"]

                # Handle different shapes: (n_samples, 1, M) or (n_samples, M)
                if len(samples.shape) == 3:
                    sigma_samples = samples[:, 0, m]
                elif len(samples.shape) == 2:
                    sigma_samples = samples[:, m] if samples.shape[1] > m else samples[:, 0]
                else:
                    # Single view case
                    sigma_samples = samples.flatten()

                ax.hist(
                    sigma_samples,
                    bins=50,
                    alpha=0.4,
                    color=colors[chain_idx],
                    label=f'Chain {chain_idx}',
                    density=True
                )

            ax.set_xlabel(r'$\sigma$ (Noise Precision)')
            ax.set_ylabel('Density')
            ax.set_title(f'Sigma (View {m + 1})', fontsize=10)
            ax.grid(True, alpha=0.3)

            if m == 0:
                ax.legend(fontsize=7, loc='best')

            # Add note about prior (Gamma with a_sigma, b_sigma)
            # Mean of Gamma(a, b) is a/b
            ax.text(0.98, 0.98, 'Gamma prior',
                   transform=ax.transAxes, fontsize=8, va='top', ha='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # Hide any unused subplots in the sigma row
        if axes.ndim == 2:
            n_cols = axes.shape[1]
            for col_idx in range(num_sources, n_cols):
                axes[ax_idx, col_idx].axis('off')

    # ========================================================================
    # Plot cW (slab scale for loadings) for each view and factor
    # ========================================================================
    if has_cW:
        logger.info("  Plotting cW (slab scale for loadings) posteriors...")
        # Calculate row offset for cW plots
        row_offset = num_sources + 1  # After tauW and tauZ
        if has_sigma:
            row_offset += 1  # After sigma

        for view_idx in range(num_sources):
            for k in range(K):
                ax_idx = row_offset + view_idx
                col_idx = k

                ax = axes[ax_idx, col_idx] if axes.ndim == 2 else axes[ax_idx]

                for chain_idx in range(n_chains):
                    samples = samples_by_chain[chain_idx]["cW"]

                    # Handle different shapes: (n_samples, M, K)
                    if len(samples.shape) == 3:
                        cW_samples = samples[:, view_idx, k]
                    elif len(samples.shape) == 2:
                        # Fallback for unexpected shapes
                        cW_samples = samples[:, k] if samples.shape[1] > k else samples[:, 0]
                    else:
                        cW_samples = samples.flatten()

                    ax.hist(
                        cW_samples,
                        bins=50,
                        alpha=0.4,
                        color=colors[chain_idx],
                        label=f'Chain {chain_idx}',
                        density=True
                    )

                ax.set_xlabel(r'$c_W$ Value')
                ax.set_ylabel('Density')
                ax.set_title(f'cW (View {view_idx + 1}, Factor {k})', fontsize=10)
                ax.grid(True, alpha=0.3)

                if k == 0 and view_idx == 0:
                    ax.legend(fontsize=7, loc='best')

                # Add note about prior
                ax.text(0.98, 0.98, 'InvGamma prior',
                       transform=ax.transAxes, fontsize=8, va='top', ha='right',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

            # Hide unused columns for this view
            for col_idx in range(K, n_cols):
                axes[row_offset + view_idx, col_idx].axis('off')

    # ========================================================================
    # Plot cZ (slab scale for factors)
    # ========================================================================
    if has_cZ:
        logger.info("  Plotting cZ (slab scale for factors) posteriors...")
        # Calculate row offset for cZ plots
        row_offset = num_sources + 1  # After tauW and tauZ
        if has_sigma:
            row_offset += 1
        if has_cW:
            row_offset += num_sources

        for k in range(K):
            ax = axes[row_offset, k] if axes.ndim == 2 else axes[row_offset]

            for chain_idx in range(n_chains):
                samples = samples_by_chain[chain_idx]["cZ"]

                # Handle different shapes: (n_samples, 1, K) or (n_samples, K)
                if len(samples.shape) == 3:
                    cZ_samples = samples[:, 0, k]
                elif len(samples.shape) == 2:
                    cZ_samples = samples[:, k] if samples.shape[1] > k else samples[:, 0]
                else:
                    cZ_samples = samples.flatten()

                ax.hist(
                    cZ_samples,
                    bins=50,
                    alpha=0.4,
                    color=colors[chain_idx],
                    label=f'Chain {chain_idx}',
                    density=True
                )

            ax.set_xlabel(r'$c_Z$ Value')
            ax.set_ylabel('Density')
            ax.set_title(f'cZ (Factor {k})', fontsize=10)
            ax.grid(True, alpha=0.3)

            if k == 0:
                ax.legend(fontsize=7, loc='best')

            # Add note about prior
            ax.text(0.98, 0.98, 'InvGamma prior',
                   transform=ax.transAxes, fontsize=8, va='top', ha='right',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        # Hide unused columns in cZ row
        for col_idx in range(K, n_cols):
            axes[row_offset, col_idx].axis('off')

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
    """Plot trace plots for hyperparameters (tauW, tauZ, sigma, cW, cZ).

    Trace plots show how hyperparameters evolve across MCMC iterations,
    helping diagnose convergence issues specific to hyperparameters.

    Parameters
    ----------
    samples_by_chain : List[Dict]
        List of sample dictionaries from each chain, containing:
        - "tauW1", "tauW2", ... : tauW samples per view
        - "tauZ" : tauZ samples
        - "sigma" : sigma (noise precision) samples
        - "cW" : cW (slab scale for loadings) samples [optional]
        - "cZ" : cZ (slab scale for factors) samples [optional]
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

    # Check if optional parameters are available
    has_sigma = "sigma" in samples_by_chain[0]
    has_cW = "cW" in samples_by_chain[0]
    has_cZ = "cZ" in samples_by_chain[0]

    logger.info(f"  Detected {num_sources} views, {K} factors, {n_samples} samples")
    if has_sigma:
        logger.info(f"  Including sigma (noise precision) traces")
    if has_cW:
        logger.info(f"  Including cW (slab scale for loadings) traces")
    if has_cZ:
        logger.info(f"  Including cZ (slab scale for factors) traces")

    # Calculate number of rows
    n_rows = num_sources + 1  # tauW + tauZ
    if has_sigma:
        n_rows += 1
    if has_cW:
        n_rows += num_sources
    if has_cZ:
        n_rows += 1

    # Determine column count
    n_cols = max(K, num_sources)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 2.5*n_rows))

    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)

    title_text = "Hyperparameter Trace Plots (Convergence Assessment)\n"
    title_text += "(tauW, tauZ"
    if has_sigma:
        title_text += ", sigma"
    if has_cW or has_cZ:
        title_text += ", cW/cZ"
    title_text += ")"

    fig.suptitle(
        title_text,
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

    # ========================================================================
    # Plot sigma traces (noise precision per view)
    # ========================================================================
    if has_sigma:
        for m in range(num_sources):
            ax_idx = num_sources + 1  # Row index for sigma traces
            col_idx = m  # Column index (one per view)

            ax = axes[ax_idx, col_idx] if axes.ndim == 2 else axes[ax_idx]

            for chain_idx in range(n_chains):
                samples = samples_by_chain[chain_idx]["sigma"]

                # Handle different shapes: (n_samples, 1, M) or (n_samples, M)
                if len(samples.shape) == 3:
                    sigma_trace = samples[::thin, 0, m]
                elif len(samples.shape) == 2:
                    sigma_trace = samples[::thin, m] if samples.shape[1] > m else samples[::thin, 0]
                else:
                    # Single view case
                    sigma_trace = samples[::thin].flatten()

                ax.plot(
                    iterations[:len(sigma_trace)],
                    sigma_trace,
                    color=colors[chain_idx],
                    alpha=0.7,
                    linewidth=1,
                    label=f'Chain {chain_idx}'
                )

            ax.set_xlabel('Iteration')
            ax.set_ylabel(r'$\sigma$ (Noise Precision)')
            ax.set_title(f'Sigma Trace (View {m + 1})', fontsize=10)
            ax.grid(True, alpha=0.3)

            if m == 0:
                ax.legend(fontsize=7, loc='best')

        # Hide any unused subplots in the sigma row
        if axes.ndim == 2:
            n_cols = axes.shape[1]
            for col_idx in range(num_sources, n_cols):
                axes[ax_idx, col_idx].axis('off')

    # ========================================================================
    # Plot cW traces (slab scale for loadings per view and factor)
    # ========================================================================
    if has_cW:
        # Calculate row offset for cW traces
        row_offset = num_sources + 1  # After tauW and tauZ
        if has_sigma:
            row_offset += 1  # After sigma

        for view_idx in range(num_sources):
            for k in range(K):
                ax_idx = row_offset + view_idx
                col_idx = k

                ax = axes[ax_idx, col_idx] if axes.ndim == 2 else axes[ax_idx]

                for chain_idx in range(n_chains):
                    samples = samples_by_chain[chain_idx]["cW"]

                    # Handle different shapes: (n_samples, M, K)
                    if len(samples.shape) == 3:
                        cW_trace = samples[::thin, view_idx, k]
                    elif len(samples.shape) == 2:
                        cW_trace = samples[::thin, k] if samples.shape[1] > k else samples[::thin, 0]
                    else:
                        cW_trace = samples[::thin].flatten()

                    ax.plot(
                        iterations[:len(cW_trace)],
                        cW_trace,
                        color=colors[chain_idx],
                        alpha=0.7,
                        linewidth=1,
                        label=f'Chain {chain_idx}'
                    )

                ax.set_xlabel('Iteration')
                ax.set_ylabel(r'$c_W$ Value')
                ax.set_title(f'cW Trace (View {view_idx + 1}, Factor {k})', fontsize=10)
                ax.grid(True, alpha=0.3)

                if k == 0 and view_idx == 0:
                    ax.legend(fontsize=7, loc='best')

            # Hide unused columns for this view
            for col_idx in range(K, n_cols):
                axes[row_offset + view_idx, col_idx].axis('off')

    # ========================================================================
    # Plot cZ traces (slab scale for factors)
    # ========================================================================
    if has_cZ:
        # Calculate row offset for cZ traces
        row_offset = num_sources + 1  # After tauW and tauZ
        if has_sigma:
            row_offset += 1
        if has_cW:
            row_offset += num_sources

        for k in range(K):
            ax = axes[row_offset, k] if axes.ndim == 2 else axes[row_offset]

            for chain_idx in range(n_chains):
                samples = samples_by_chain[chain_idx]["cZ"]

                # Handle different shapes: (n_samples, 1, K) or (n_samples, K)
                if len(samples.shape) == 3:
                    cZ_trace = samples[::thin, 0, k]
                elif len(samples.shape) == 2:
                    cZ_trace = samples[::thin, k] if samples.shape[1] > k else samples[::thin, 0]
                else:
                    cZ_trace = samples[::thin].flatten()

                ax.plot(
                    iterations[:len(cZ_trace)],
                    cZ_trace,
                    color=colors[chain_idx],
                    alpha=0.7,
                    linewidth=1,
                    label=f'Chain {chain_idx}'
                )

            ax.set_xlabel('Iteration')
            ax.set_ylabel(r'$c_Z$ Value')
            ax.set_title(f'cZ Trace (Factor {k})', fontsize=10)
            ax.grid(True, alpha=0.3)

            if k == 0:
                ax.legend(fontsize=7, loc='best')

        # Hide unused columns in cZ row
        for col_idx in range(K, n_cols):
            axes[row_offset, col_idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  ✓ Saved to {save_path}")

    logger.info("  ✓ Hyperparameter trace plots completed")

    return fig


def analyze_factor_variance_profile(
    Z_samples: np.ndarray,
    variance_threshold: float = 0.1,
    save_path: Optional[str] = None,
) -> Dict[str, Union[np.ndarray, int, List[int]]]:
    """Analyze factor variance profile to assess ARD shrinkage effectiveness.

    This function computes the variance of each factor across subjects to identify
    which factors capture real signal vs. which are shrunk to near-zero by ARD priors.
    Critical for understanding effective dimensionality in overparameterized models.

    Parameters
    ----------
    Z_samples : np.ndarray
        Factor score samples, shape (n_chains, n_samples, n_subjects, K)
    variance_threshold : float, default=0.1
        Minimum variance for a factor to be considered "active"
        Factors below this are considered shrunk away by ARD
    save_path : str, optional
        Path to save diagnostic plot

    Returns
    -------
    dict
        Dictionary containing:
        - 'factor_variances': np.ndarray, variance of each factor (length K)
        - 'sorted_variances': np.ndarray, variances in descending order
        - 'sorted_indices': np.ndarray, factor indices in variance order
        - 'n_active_factors': int, count of factors above threshold
        - 'active_factor_indices': list, indices of active factors
        - 'variance_ratios': np.ndarray, each variance / max variance
        - 'cumulative_variance_explained': np.ndarray, cumulative proportion
        - 'effective_dimensionality': int, factors explaining 90% variance

    Notes
    -----
    Healthy ARD shrinkage shows a "spiky" variance profile:
    - Few factors with variance > 1.0 (capturing real signal)
    - Sharp drop-off to near-zero for remaining factors
    - Example: [1.5, 1.2, 0.8, 0.03, 0.01, 0.01, ...] suggests 3 real factors

    Poor shrinkage shows gradual decline:
    - Many factors with intermediate variance (0.2-0.6)
    - Suggests model uncertainty or convergence issues
    - Example: [1.2, 0.9, 0.7, 0.5, 0.4, 0.3, ...] suggests overfitting

    References
    ----------
    - Archambeau & Bach 2008: "Sparse Bayesian Factor Analysis"
    - Tipping 2001: "Sparse Bayesian Learning and the Relevance Vector Machine"
    """
    logger.info("📊 Analyzing factor variance profile (ARD shrinkage assessment)...")

    # Z_samples shape: (n_chains, n_samples, n_subjects, K)
    n_chains, n_samples, n_subjects, K = Z_samples.shape

    logger.info(f"   Z_samples shape: {n_chains} chains × {n_samples} samples × {n_subjects} subjects × {K} factors")

    # Calculate variance of each factor across subjects
    # For each chain and sample, compute variance across subjects, then average
    # This preserves the scale of variance while accounting for MCMC uncertainty
    factor_variances = np.zeros(K)
    for k in range(K):
        # Extract this factor across all chains, samples, subjects
        # Shape: (n_chains, n_samples, n_subjects)
        factor_k = Z_samples[:, :, :, k]

        # Compute variance across subjects for each (chain, sample) combination
        # Shape: (n_chains, n_samples)
        variances_per_sample = factor_k.var(axis=2)

        # Average across chains and samples
        factor_variances[k] = variances_per_sample.mean()

    # Sort variances in descending order
    sorted_indices = np.argsort(factor_variances)[::-1]
    sorted_variances = factor_variances[sorted_indices]

    # Identify active factors (above threshold)
    active_mask = factor_variances >= variance_threshold
    n_active_factors = active_mask.sum()
    active_factor_indices = np.where(active_mask)[0].tolist()

    # Compute variance ratios (normalized by max)
    max_variance = factor_variances.max()
    variance_ratios = factor_variances / max_variance if max_variance > 0 else factor_variances

    # Cumulative variance explained
    cumulative_variance = np.cumsum(sorted_variances) / sorted_variances.sum()

    # Effective dimensionality (factors needed to explain 90% variance)
    effective_dim = np.searchsorted(cumulative_variance, 0.90) + 1

    # Log summary
    logger.info(f"   Total factors (K): {K}")
    logger.info(f"   Active factors (var > {variance_threshold}): {n_active_factors}")
    logger.info(f"   Effective dimensionality (90% variance): {effective_dim}")
    logger.info(f"   Max factor variance: {max_variance:.4f}")
    logger.info(f"   Top 5 factor variances: {sorted_variances[:5]}")

    # Check for healthy vs poor shrinkage
    if K >= 10:
        top_5_mean = sorted_variances[:5].mean()
        next_10_mean = sorted_variances[5:15].mean() if K >= 15 else sorted_variances[5:].mean()
        shrinkage_ratio = top_5_mean / (next_10_mean + 1e-10)

        if shrinkage_ratio > 10:
            logger.info(f"   ✅ HEALTHY ARD SHRINKAGE: Sharp drop-off detected (ratio={shrinkage_ratio:.1f})")
        elif shrinkage_ratio > 3:
            logger.info(f"   ⚠️  MODERATE SHRINKAGE: Some drop-off but gradual (ratio={shrinkage_ratio:.1f})")
        else:
            logger.info(f"   ❌ POOR SHRINKAGE: Variance spread across many factors (ratio={shrinkage_ratio:.1f})")
            logger.info(f"       → Suggests model uncertainty or convergence issues")

    # Create comprehensive visualization
    if save_path or True:  # Always create figure
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Variance profile (bar plot, sorted)
        ax1 = fig.add_subplot(gs[0, :2])
        colors = ['#2ecc71' if v >= variance_threshold else '#e74c3c' for v in sorted_variances]
        ax1.bar(range(K), sorted_variances, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.axhline(variance_threshold, color='#e74c3c', linestyle='--', linewidth=2,
                   label=f'Threshold ({variance_threshold})')
        ax1.set_xlabel('Factor Rank (sorted by variance)', fontsize=11)
        ax1.set_ylabel('Variance Across Subjects', fontsize=11)
        ax1.set_title(f'Factor Variance Profile (K={K}): {n_active_factors} Active Factors',
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add text annotation
        ax1.text(0.98, 0.95, f'Active: {n_active_factors}/{K}\nEffective: {effective_dim}',
                transform=ax1.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

        # 2. Log-scale variance (better for seeing drop-off)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.semilogy(range(K), sorted_variances, 'o-', color='#3498db',
                    markersize=5, linewidth=2, alpha=0.7)
        ax2.axhline(variance_threshold, color='#e74c3c', linestyle='--', linewidth=2)
        ax2.set_xlabel('Factor Rank', fontsize=10)
        ax2.set_ylabel('Variance (log scale)', fontsize=10)
        ax2.set_title('Log-Scale Variance\n(Sharp drop = good ARD)', fontsize=10)
        ax2.grid(True, alpha=0.3, which='both')

        # 3. Cumulative variance explained
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(range(1, K+1), cumulative_variance * 100, 'o-',
                color='#9b59b6', markersize=4, linewidth=2)
        ax3.axhline(90, color='#e74c3c', linestyle='--', linewidth=1.5, label='90% threshold')
        ax3.axvline(effective_dim, color='#e74c3c', linestyle='--', linewidth=1.5)
        ax3.set_xlabel('Number of Factors', fontsize=10)
        ax3.set_ylabel('Cumulative Variance (%)', fontsize=10)
        ax3.set_title(f'Cumulative Variance\n(90% @ {effective_dim} factors)', fontsize=10)
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3)

        # 4. Variance ratios (normalized by max)
        ax4 = fig.add_subplot(gs[1, 1])
        sorted_ratios = variance_ratios[sorted_indices]
        ax4.bar(range(K), sorted_ratios, color='#f39c12', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('Factor Rank', fontsize=10)
        ax4.set_ylabel('Variance Ratio (vs Max)', fontsize=10)
        ax4.set_title('Normalized Variance Profile', fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Top 15 factors (detailed view)
        ax5 = fig.add_subplot(gs[1, 2])
        n_top = min(15, K)
        top_colors = ['#2ecc71' if sorted_variances[i] >= variance_threshold else '#e74c3c'
                     for i in range(n_top)]
        ax5.barh(range(n_top), sorted_variances[:n_top], color=top_colors,
                alpha=0.7, edgecolor='black', linewidth=0.5)
        ax5.set_yticks(range(n_top))
        ax5.set_yticklabels([f'F{sorted_indices[i]+1}' for i in range(n_top)], fontsize=8)
        ax5.invert_yaxis()
        ax5.set_xlabel('Variance', fontsize=10)
        ax5.set_title(f'Top {n_top} Factors', fontsize=10)
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. Histogram of variances
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.hist(factor_variances, bins=30, color='#16a085', alpha=0.7, edgecolor='black')
        ax6.axvline(variance_threshold, color='#e74c3c', linestyle='--', linewidth=2,
                   label=f'Threshold ({variance_threshold})')
        ax6.set_xlabel('Variance Value', fontsize=10)
        ax6.set_ylabel('Count', fontsize=10)
        ax6.set_title('Distribution of Factor Variances', fontsize=10)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')

        # 7. Scatter: Factor index vs variance
        ax7 = fig.add_subplot(gs[2, 1])
        colors_scatter = ['#2ecc71' if v >= variance_threshold else '#e74c3c'
                         for v in factor_variances]
        ax7.scatter(range(K), factor_variances, c=colors_scatter, s=50, alpha=0.6, edgecolor='black')
        ax7.axhline(variance_threshold, color='#e74c3c', linestyle='--', linewidth=2)
        ax7.set_xlabel('Original Factor Index', fontsize=10)
        ax7.set_ylabel('Variance', fontsize=10)
        ax7.set_title('Variance by Original Factor Order', fontsize=10)
        ax7.grid(True, alpha=0.3)

        # 8. Diagnostic text summary
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')

        # Compute diagnostics
        if K >= 10:
            top_5_mean = sorted_variances[:5].mean()
            next_10_mean = sorted_variances[5:15].mean() if K >= 15 else sorted_variances[5:].mean()
            shrinkage_ratio = top_5_mean / (next_10_mean + 1e-10)
            shrinkage_status = "HEALTHY ✅" if shrinkage_ratio > 10 else \
                             "MODERATE ⚠️" if shrinkage_ratio > 3 else "POOR ❌"
        else:
            shrinkage_ratio = np.nan
            shrinkage_status = "N/A (K < 10)"

        # Build top variances text
        top_vars_text = ""
        for i in range(min(3, K)):
            top_vars_text += f"\n  #{sorted_indices[i]+1}: {sorted_variances[i]:.4f}"

        summary_text = f"""
VARIANCE PROFILE SUMMARY
{'=' * 25}

Total Factors (K):     {K}
Active Factors:        {n_active_factors}
Effective Dim (90%):   {effective_dim}

Max Variance:          {max_variance:.4f}
Mean Variance:         {factor_variances.mean():.4f}
Median Variance:       {np.median(factor_variances):.4f}

Top {min(3, K)} Variances:{top_vars_text}

ARD Shrinkage:         {shrinkage_status}
Shrinkage Ratio:       {shrinkage_ratio:.2f}

INTERPRETATION:
{'─' * 25}
""".strip()

        if shrinkage_ratio > 10:
            interp = "Sharp drop-off indicates\nhealthy ARD shrinkage.\nModel confidently identified\nfew real factors."
        elif shrinkage_ratio > 3:
            interp = "Moderate shrinkage.\nModel somewhat uncertain.\nCheck convergence."
        else:
            interp = "Poor shrinkage suggests:\n• Convergence issues\n• Overfitting\n• Need longer sampling"

        summary_text += f"\n{interp}"

        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

        plt.suptitle(f'Factor Variance Analysis: ARD Shrinkage Assessment (K={K})',
                    fontsize=14, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"  ✓ Saved variance profile plot to {save_path}")

    logger.info("  ✓ Factor variance analysis completed")

    return {
        'factor_variances': factor_variances,
        'sorted_variances': sorted_variances,
        'sorted_indices': sorted_indices,
        'n_active_factors': n_active_factors,
        'active_factor_indices': active_factor_indices,
        'variance_ratios': variance_ratios,
        'cumulative_variance_explained': cumulative_variance,
        'effective_dimensionality': effective_dim,
        'max_variance': max_variance,
        'mean_variance': factor_variances.mean(),
        'median_variance': np.median(factor_variances),
    }
