"""Visualization functions for factor analysis diagnostics.

This module provides plotting functions for the comprehensive factor analysis
diagnostics introduced in Aspects 2, 4, 11, 16, 17, and 18.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)

# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)


def plot_scale_indeterminacy_diagnostics(
    scale_diag: Dict,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Visualize scale indeterminacy diagnostics (Aspect 4).

    Creates a comprehensive 2x2 plot showing:
    1. ||W|| across chains
    2. ||Z|| across chains
    3. tau_W across chains
    4. tau_Z across chains

    Parameters
    ----------
    scale_diag : Dict
        Output from diagnose_scale_indeterminacy()
    output_path : Path, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    n_chains = len(scale_diag['per_chain'])
    chain_ids = list(range(n_chains))

    # Extract metrics
    W_norms = [c['W_norm'] for c in scale_diag['per_chain']]
    Z_norms = [c['Z_norm'] for c in scale_diag['per_chain']]
    tau_W_vals = [c['tau_W'] for c in scale_diag['per_chain']]
    tau_Z_vals = [c['tau_Z'] for c in scale_diag['per_chain']]

    # Plot 1: ||W|| across chains
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(chain_ids, W_norms, alpha=0.7, color='steelblue')
    ax1.axhline(scale_diag['summary']['W_mean'], color='red', linestyle='--',
                label=f"Mean: {scale_diag['summary']['W_mean']:.2f}")
    ax1.set_xlabel('Chain ID')
    ax1.set_ylabel('||W||_F')
    ax1.set_title(f"Loading Matrix Frobenius Norm\nCV = {scale_diag['summary']['W_cv']:.3f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: ||Z|| across chains
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(chain_ids, Z_norms, alpha=0.7, color='seagreen')
    ax2.axhline(scale_diag['summary']['Z_mean'], color='red', linestyle='--',
                label=f"Mean: {scale_diag['summary']['Z_mean']:.2f}")
    ax2.set_xlabel('Chain ID')
    ax2.set_ylabel('||Z||_F')
    ax2.set_title(f"Factor Score Frobenius Norm\nCV = {scale_diag['summary']['Z_cv']:.3f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: tau_W across chains
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(chain_ids, tau_W_vals, alpha=0.7, color='coral')
    ax3.axhline(scale_diag['summary']['tau_W_mean'], color='red', linestyle='--',
                label=f"Mean: {scale_diag['summary']['tau_W_mean']:.4f}")
    ax3.set_xlabel('Chain ID')
    ax3.set_ylabel('tau_W')
    ax3.set_title(f"Loading Global Scale Parameter\nCV = {scale_diag['summary']['tau_W_cv']:.3f}")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: tau_Z across chains
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(chain_ids, tau_Z_vals, alpha=0.7, color='mediumpurple')
    ax4.axhline(scale_diag['summary']['tau_Z_mean'], color='red', linestyle='--',
                label=f"Mean: {scale_diag['summary']['tau_Z_mean']:.4f}")
    ax4.set_xlabel('Chain ID')
    ax4.set_ylabel('tau_Z')
    ax4.set_title(f"Factor Score Global Scale Parameter\nCV = {scale_diag['summary']['tau_Z_cv']:.3f}")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Overall title with verdict
    verdict = scale_diag['verdict']
    color = 'green' if verdict == 'well_constrained' else 'orange' if verdict == 'moderate_drift' else 'red'
    fig.suptitle(f"Scale Indeterminacy Diagnostics (Aspect 4)\nVerdict: {verdict.replace('_', ' ').title()}",
                 fontsize=14, fontweight='bold', color=color)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved scale indeterminacy plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_slab_saturation_diagnostics(
    slab_diag: Dict,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Visualize slab saturation diagnostics (Aspect 11).

    Creates a figure showing:
    1. Loading magnitude distribution
    2. Saturation at ceiling detection
    3. Per-chain diagnostics

    Parameters
    ----------
    slab_diag : Dict
        Output from diagnose_slab_saturation()
    output_path : Path, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    slab_scale = slab_diag['slab_scale']
    n_chains = len(slab_diag['per_chain'])

    # Plot 1: Loading magnitude histogram (all chains)
    ax1 = fig.add_subplot(gs[0, :2])
    all_loadings = []
    for chain in slab_diag['per_chain']:
        all_loadings.extend(chain['loading_magnitudes'])

    ax1.hist(all_loadings, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(slab_scale, color='red', linestyle='--', linewidth=2, label=f'Slab scale c = {slab_scale:.2f}')
    ax1.axvline(2 * slab_scale, color='orange', linestyle='--', linewidth=2, label=f'2c = {2*slab_scale:.2f}')
    ax1.set_xlabel('|W_jk|')
    ax1.set_ylabel('Count')
    ax1.set_title('Loading Magnitude Distribution (All Chains)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, min(max(all_loadings), 10 * slab_scale)])

    # Plot 2: Saturation summary per chain
    ax2 = fig.add_subplot(gs[0, 2])
    chain_ids = list(range(n_chains))
    saturation_pcts = [chain['saturation_pct'] * 100 for chain in slab_diag['per_chain']]

    colors = ['red' if pct > 10 else 'green' for pct in saturation_pcts]
    ax2.bar(chain_ids, saturation_pcts, alpha=0.7, color=colors)
    ax2.axhline(10, color='orange', linestyle='--', label='10% threshold')
    ax2.set_xlabel('Chain ID')
    ax2.set_ylabel('Saturation %')
    ax2.set_title('Loadings Saturating at Ceiling')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Max loading per chain
    ax3 = fig.add_subplot(gs[1, 0])
    max_loadings = [chain['max_loading'] for chain in slab_diag['per_chain']]
    colors = ['red' if ml > 2 * slab_scale else 'orange' if ml > slab_scale else 'green'
              for ml in max_loadings]

    ax3.bar(chain_ids, max_loadings, alpha=0.7, color=colors)
    ax3.axhline(slab_scale, color='orange', linestyle='--', label=f'c = {slab_scale:.2f}')
    ax3.axhline(2 * slab_scale, color='red', linestyle='--', label=f'2c = {2*slab_scale:.2f}')
    ax3.set_xlabel('Chain ID')
    ax3.set_ylabel('max |W_jk|')
    ax3.set_title('Maximum Loading Magnitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Data scale check (boxplot of loadings per chain)
    ax4 = fig.add_subplot(gs[1, 1:])
    loading_data = [chain['loading_magnitudes'] for chain in slab_diag['per_chain']]
    bp = ax4.boxplot(loading_data, labels=[f'Chain {i}' for i in chain_ids], patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax4.axhline(slab_scale, color='orange', linestyle='--', linewidth=2, label=f'c = {slab_scale:.2f}')
    ax4.axhline(2 * slab_scale, color='red', linestyle='--', linewidth=2, label=f'2c = {2*slab_scale:.2f}')
    ax4.set_xlabel('Chain')
    ax4.set_ylabel('|W_jk|')
    ax4.set_title('Loading Distribution per Chain')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Overall title with verdict
    issues = slab_diag['issues']
    if issues['data_preprocessing_failure']:
        title_color = 'red'
        verdict = 'CRITICAL: Data Preprocessing Failure'
    elif issues['saturation_at_ceiling']:
        title_color = 'orange'
        verdict = 'WARNING: Saturation Detected'
    elif issues['goldilocks_violation']:
        title_color = 'orange'
        verdict = 'WARNING: Slab Scale Out of Range'
    else:
        title_color = 'green'
        verdict = 'HEALTHY: No Issues Detected'

    fig.suptitle(f"Slab Saturation Diagnostics (Aspect 11)\n{verdict}",
                 fontsize=14, fontweight='bold', color=title_color)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved slab saturation plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_prior_posterior_shift(
    shift_diag: Dict,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Visualize prior-posterior shift diagnostics (Aspect 18).

    Creates a figure showing shift factors for all hyperparameters.

    Parameters
    ----------
    shift_diag : Dict
        Output from diagnose_prior_posterior_shift()
    output_path : Path, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Extract parameters
    params = ['tau_W', 'tau_Z', 'cW', 'cZ']
    param_labels = ['tau_W (Loading Scale)', 'tau_Z (Score Scale)',
                    'c_W (Loading Slab)', 'c_Z (Score Slab)']

    for idx, (param, label) in enumerate(zip(params, param_labels)):
        if param in shift_diag:
            ax = axes[idx]
            data = shift_diag[param]

            # Bar plot of shift factor
            shift = data['shift_factor']
            classification = data['classification']

            # Color based on classification
            if classification == 'healthy':
                color = 'green'
            elif classification == 'moderate':
                color = 'orange'
            elif classification == 'weak':
                color = 'red'
            else:  # severe
                color = 'darkred'

            ax.bar(['Shift Factor'], [shift], color=color, alpha=0.7, width=0.5)

            # Add reference lines
            ax.axhline(2.0, color='orange', linestyle='--', linewidth=1.5, label='Moderate threshold')
            ax.axhline(5.0, color='red', linestyle='--', linewidth=1.5, label='Weak threshold')
            ax.axhline(10.0, color='darkred', linestyle='--', linewidth=1.5, label='Severe threshold')

            # Add value text
            ax.text(0, shift + 0.5, f'{shift:.2f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')

            ax.set_ylabel('Shift Factor (posterior/prior)')
            ax.set_title(f'{label}\n{classification.upper()}')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

            # Log scale if shift > 10
            if shift > 10:
                ax.set_yscale('log')
                ax.set_ylabel('Shift Factor (log scale)')

    # Overall title
    overall_classification = shift_diag.get('overall_classification', 'unknown')
    if overall_classification == 'healthy':
        title_color = 'green'
    elif overall_classification == 'moderate':
        title_color = 'orange'
    elif overall_classification in ['weak', 'severe']:
        title_color = 'red'
    else:
        title_color = 'black'

    fig.suptitle(f"Prior-Posterior Shift Diagnostics (Aspect 18)\nOverall: {overall_classification.upper()}",
                 fontsize=14, fontweight='bold', color=title_color)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved prior-posterior shift plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_factor_type_classification(
    factor_types: Dict,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Visualize factor type classification (Aspect 16).

    Creates a figure showing:
    1. Factor type distribution (pie chart)
    2. Per-factor activity heatmap

    Parameters
    ----------
    factor_types : Dict
        Output from classify_factor_types()
    output_path : Path, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    # Plot 1: Factor type distribution (pie chart)
    ax1 = fig.add_subplot(gs[0, 0])

    summary = factor_types['summary']
    sizes = [summary['n_shared'], summary['n_view_specific'], summary['n_background']]
    labels = [f"Shared ({summary['pct_shared']:.0f}%)",
              f"View-Specific ({summary['pct_view_specific']:.0f}%)",
              f"Background ({summary['pct_background']:.0f}%)"]
    colors = ['#2ecc71', '#3498db', '#95a5a6']
    explode = (0.05, 0.05, 0.05)

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%d',
            shadow=True, startangle=90)
    ax1.set_title('Factor Type Distribution')

    # Plot 2: Per-factor activity heatmap
    ax2 = fig.add_subplot(gs[0, 1])

    # Build activity matrix
    n_factors = len(factor_types['per_factor'])
    n_views = len(factor_types['view_names'])
    activity_matrix = np.zeros((n_factors, n_views))

    for k, factor_info in enumerate(factor_types['per_factor']):
        for view_idx in factor_info['active_views']:
            activity_matrix[k, view_idx] = 1

    # Create heatmap
    im = ax2.imshow(activity_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax2.set_xticks(range(n_views))
    ax2.set_xticklabels(factor_types['view_names'], rotation=45, ha='right')
    ax2.set_yticks(range(n_factors))
    ax2.set_yticklabels([f"Factor {k}" for k in range(n_factors)])

    # Add factor type labels on the right
    for k, factor_info in enumerate(factor_types['per_factor']):
        ftype = factor_info['type']
        if ftype == 'shared':
            label = 'S'
            color = '#2ecc71'
        elif ftype == 'view_specific':
            label = 'V'
            color = '#3498db'
        else:
            label = 'B'
            color = '#95a5a6'

        ax2.text(n_views + 0.2, k, label, va='center', ha='left',
                fontsize=10, fontweight='bold', color=color,
                bbox=dict(boxstyle='circle', facecolor='white', edgecolor=color))

    ax2.set_xlabel('View')
    ax2.set_ylabel('Factor')
    ax2.set_title('Factor Activity Matrix\n(S=Shared, V=View-Specific, B=Background)')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Active', rotation=270, labelpad=15)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Inactive', 'Active'])

    fig.suptitle('Factor Type Classification (Aspect 16)', fontsize=14, fontweight='bold')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved factor type classification plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_cross_view_correlation(
    corr_results: Dict,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Visualize cross-view correlation matrix (Aspect 17).

    Parameters
    ----------
    corr_results : Dict
        Output from compute_cross_view_correlation()
    output_path : Path, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract correlation matrix
    view_names = corr_results['view_names']
    n_views = len(view_names)
    corr_matrix = np.zeros((n_views, n_views))

    # Fill correlation matrix
    for i in range(n_views):
        corr_matrix[i, i] = 1.0
        for j in range(i + 1, n_views):
            pair_key = f"{view_names[i]}_vs_{view_names[j]}"
            if pair_key in corr_results['pairwise']:
                corr = corr_results['pairwise'][pair_key]['correlation']
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(range(n_views))
    ax.set_xticklabels(view_names, rotation=45, ha='right')
    ax.set_yticks(range(n_views))
    ax.set_yticklabels(view_names)

    # Add correlation values
    for i in range(n_views):
        for j in range(n_views):
            if i != j:
                text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Canonical Correlation', rotation=270, labelpad=20)

    # Title with interpretation
    mean_corr = corr_results['summary']['mean_correlation']
    level = corr_results['interpretation']['correlation_level']

    if level == 'HIGH':
        color = 'green'
    elif level == 'MODERATE':
        color = 'orange'
    else:
        color = 'blue'

    ax.set_title(f'Cross-View Canonical Correlation (Aspect 17)\n'
                f'Mean correlation: {mean_corr:.3f} ({level})',
                fontsize=12, fontweight='bold')

    # Add text box with interpretation
    textstr = corr_results['interpretation']['recommendation']
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.5, -0.15, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, ha='center')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved cross-view correlation plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_procrustes_alignment(
    procrustes_results: Dict,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Visualize Procrustes alignment diagnostics (Aspect 2).

    Shows disparities and rotation angles across chains.

    Parameters
    ----------
    procrustes_results : Dict
        Output from assess_factor_stability_procrustes()
    output_path : Path, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_chains = len(procrustes_results['per_chain'])
    chain_ids = list(range(n_chains))

    # Plot 1: Disparities
    ax1 = axes[0]
    disparities = [c['disparity'] for c in procrustes_results['per_chain']]
    threshold = procrustes_results.get('disparity_threshold', 0.3)

    colors = ['green' if d < threshold else 'red' for d in disparities]
    ax1.bar(chain_ids, disparities, alpha=0.7, color=colors)
    ax1.axhline(threshold, color='orange', linestyle='--', linewidth=2,
                label=f'Threshold = {threshold:.2f}')
    ax1.set_xlabel('Chain ID')
    ax1.set_ylabel('Procrustes Disparity')
    ax1.set_title(f'Alignment Quality\n(lower = better alignment)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Maximum rotation angles
    ax2 = axes[1]
    max_angles = [c['max_rotation_angle_deg'] for c in procrustes_results['per_chain']]

    ax2.bar(chain_ids, max_angles, alpha=0.7, color='steelblue')
    ax2.set_xlabel('Chain ID')
    ax2.set_ylabel('Max Rotation Angle (degrees)')
    ax2.set_title('Maximum Factor Rotation\n(relative to reference)')
    ax2.grid(True, alpha=0.3)

    # Overall title
    alignment_rate = procrustes_results.get('alignment_rate', 0)
    color = 'green' if alignment_rate > 0.8 else 'orange' if alignment_rate > 0.5 else 'red'

    fig.suptitle(f'Procrustes Alignment Diagnostics (Aspect 2)\n'
                f'Alignment Rate: {alignment_rate:.1%}',
                fontsize=14, fontweight='bold', color=color)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved Procrustes alignment plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def create_comprehensive_diagnostic_report(
    diagnostics: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """
    Create a comprehensive PDF report with all diagnostic plots.

    Parameters
    ----------
    diagnostics : Dict[str, Dict]
        Dictionary mapping diagnostic names to their results:
        - 'scale_indeterminacy': from diagnose_scale_indeterminacy()
        - 'slab_saturation': from diagnose_slab_saturation()
        - 'prior_posterior_shift': from diagnose_prior_posterior_shift()
        - 'factor_types': from classify_factor_types()
        - 'cross_view_correlation': from compute_cross_view_correlation()
        - 'procrustes_alignment': from assess_factor_stability_procrustes()
    output_dir : Path
        Directory to save individual plots and summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Creating comprehensive diagnostic report...")
    logger.info("=" * 80)

    plot_count = 0

    # Generate each diagnostic plot
    if 'scale_indeterminacy' in diagnostics:
        logger.info("  Plotting scale indeterminacy diagnostics (Aspect 4)...")
        plot_scale_indeterminacy_diagnostics(
            diagnostics['scale_indeterminacy'],
            output_path=output_dir / 'aspect_04_scale_indeterminacy.png'
        )
        plot_count += 1

    if 'slab_saturation' in diagnostics:
        logger.info("  Plotting slab saturation diagnostics (Aspect 11)...")
        plot_slab_saturation_diagnostics(
            diagnostics['slab_saturation'],
            output_path=output_dir / 'aspect_11_slab_saturation.png'
        )
        plot_count += 1

    if 'prior_posterior_shift' in diagnostics:
        logger.info("  Plotting prior-posterior shift diagnostics (Aspect 18)...")
        plot_prior_posterior_shift(
            diagnostics['prior_posterior_shift'],
            output_path=output_dir / 'aspect_18_prior_posterior_shift.png'
        )
        plot_count += 1

    if 'factor_types' in diagnostics:
        logger.info("  Plotting factor type classification (Aspect 16)...")
        plot_factor_type_classification(
            diagnostics['factor_types'],
            output_path=output_dir / 'aspect_16_factor_types.png'
        )
        plot_count += 1

    if 'cross_view_correlation' in diagnostics:
        logger.info("  Plotting cross-view correlation (Aspect 17)...")
        plot_cross_view_correlation(
            diagnostics['cross_view_correlation'],
            output_path=output_dir / 'aspect_17_cross_view_correlation.png'
        )
        plot_count += 1

    if 'procrustes_alignment' in diagnostics:
        logger.info("  Plotting Procrustes alignment (Aspect 2)...")
        plot_procrustes_alignment(
            diagnostics['procrustes_alignment'],
            output_path=output_dir / 'aspect_02_procrustes_alignment.png'
        )
        plot_count += 1

    logger.info(f"  Generated {plot_count} diagnostic plots")
    logger.info(f"  Saved to: {output_dir}")
    logger.info("=" * 80)
