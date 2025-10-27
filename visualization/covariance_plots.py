"""Covariance and correlation matrix visualization for multi-view data.

This module provides functions to analyze and visualize:
1. Intra-modality covariance (within imaging, within clinical)
2. Inter-modality covariance (between imaging and clinical)
3. Block-structured covariance matrices
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)


def compute_covariance_structure(
    X_list: List[np.ndarray],
    view_names: Optional[List[str]] = None,
) -> Dict:
    """Compute intra-modality and inter-modality covariance matrices.

    Parameters
    ----------
    X_list : List[np.ndarray]
        List of data matrices, one per view. Each should be (n_samples, n_features).
    view_names : Optional[List[str]]
        Names for each view (e.g., ['imaging', 'clinical'])

    Returns
    -------
    Dict with keys:
        'full_cov': Full covariance matrix (concatenated features)
        'full_corr': Full correlation matrix (concatenated features)
        'intra_cov': Dict of within-view covariance matrices
        'intra_corr': Dict of within-view correlation matrices
        'inter_cov': Dict of between-view covariance matrices
        'inter_corr': Dict of between-view correlation matrices
        'view_dims': List of feature dimensions per view
        'view_names': List of view names
    """
    if view_names is None:
        view_names = [f"view_{i}" for i in range(len(X_list))]

    # Validate inputs
    n_samples = X_list[0].shape[0]
    for i, X in enumerate(X_list):
        if X.shape[0] != n_samples:
            raise ValueError(f"All views must have same number of samples. "
                           f"View 0: {n_samples}, View {i}: {X.shape[0]}")

    # Get dimensions
    view_dims = [X.shape[1] for X in X_list]

    # Concatenate all views
    X_concat = np.concatenate(X_list, axis=1)

    # Compute full covariance and correlation
    full_cov = np.cov(X_concat.T)
    full_corr = np.corrcoef(X_concat.T)

    # Compute intra-modality covariance (within each view)
    intra_cov = {}
    intra_corr = {}
    for i, (X, name) in enumerate(zip(X_list, view_names)):
        intra_cov[name] = np.cov(X.T)
        intra_corr[name] = np.corrcoef(X.T)
        logger.info(f"Intra-modality covariance for {name}: {X.shape[1]}x{X.shape[1]}")

    # Compute inter-modality covariance (between pairs of views)
    inter_cov = {}
    inter_corr = {}
    for i in range(len(X_list)):
        for j in range(i + 1, len(X_list)):
            name_i, name_j = view_names[i], view_names[j]
            pair_name = f"{name_i}_vs_{name_j}"

            # Concatenate the two views
            X_pair = np.concatenate([X_list[i], X_list[j]], axis=1)
            cov_pair = np.cov(X_pair.T)
            corr_pair = np.corrcoef(X_pair.T)

            # Extract the off-diagonal block (cross-view covariance)
            dim_i = X_list[i].shape[1]
            dim_j = X_list[j].shape[1]
            inter_cov[pair_name] = cov_pair[:dim_i, dim_i:]
            inter_corr[pair_name] = corr_pair[:dim_i, dim_i:]

            logger.info(f"Inter-modality covariance for {pair_name}: {dim_i}x{dim_j}")

    return {
        'full_cov': full_cov,
        'full_corr': full_corr,
        'intra_cov': intra_cov,
        'intra_corr': intra_corr,
        'inter_cov': inter_cov,
        'inter_corr': inter_corr,
        'view_dims': view_dims,
        'view_names': view_names,
    }


def analyze_covariance_statistics(cov_structure: Dict) -> Dict:
    """Compute summary statistics for covariance structure.

    Parameters
    ----------
    cov_structure : Dict
        Output from compute_covariance_structure

    Returns
    -------
    Dict with summary statistics
    """
    stats = {
        'view_names': cov_structure['view_names'],
        'view_dims': cov_structure['view_dims'],
        'intra_stats': {},
        'inter_stats': {},
    }

    # Intra-modality statistics
    for name in cov_structure['view_names']:
        cov = cov_structure['intra_cov'][name]
        corr = cov_structure['intra_corr'][name]

        # Get off-diagonal elements (exclude diagonal)
        mask = ~np.eye(cov.shape[0], dtype=bool)
        off_diag_cov = cov[mask]
        off_diag_corr = corr[mask]

        stats['intra_stats'][name] = {
            'cov_mean': np.mean(off_diag_cov),
            'cov_std': np.std(off_diag_cov),
            'cov_median': np.median(off_diag_cov),
            'corr_mean': np.mean(off_diag_corr),
            'corr_std': np.std(off_diag_corr),
            'corr_median': np.median(off_diag_corr),
            'corr_max': np.max(np.abs(off_diag_corr)),
        }

    # Inter-modality statistics
    for pair_name, cov in cov_structure['inter_cov'].items():
        corr = cov_structure['inter_corr'][pair_name]

        stats['inter_stats'][pair_name] = {
            'cov_mean': np.mean(cov),
            'cov_std': np.std(cov),
            'cov_median': np.median(cov),
            'corr_mean': np.mean(corr),
            'corr_std': np.std(corr),
            'corr_median': np.median(corr),
            'corr_max': np.max(np.abs(corr)),
        }

    return stats


def plot_block_covariance_matrix(
    cov_structure: Dict,
    output_path: Optional[Path] = None,
    use_correlation: bool = True,
    subsample: Optional[int] = None,
) -> plt.Figure:
    """Plot block-structured covariance matrix showing inter/intra-modality structure.

    Parameters
    ----------
    cov_structure : Dict
        Output from compute_covariance_structure
    output_path : Optional[Path]
        Path to save figure
    use_correlation : bool
        If True, plot correlation matrix instead of covariance
    subsample : Optional[int]
        If provided, subsample features to this number for visualization

    Returns
    -------
    matplotlib Figure
    """
    matrix = cov_structure['full_corr'] if use_correlation else cov_structure['full_cov']
    view_dims = cov_structure['view_dims']
    view_names = cov_structure['view_names']

    # Subsample if requested (for large matrices)
    if subsample is not None and matrix.shape[0] > subsample:
        logger.info(f"Subsampling covariance matrix from {matrix.shape[0]} to {subsample} features")
        # Subsample proportionally from each view
        indices = []
        start_idx = 0
        for dim in view_dims:
            n_sample = max(1, int(subsample * dim / sum(view_dims)))
            view_indices = np.linspace(start_idx, start_idx + dim - 1, n_sample, dtype=int)
            indices.extend(view_indices)
            start_idx += dim

        matrix = matrix[np.ix_(indices, indices)]
        # Update view_dims for plotting
        view_dims_sampled = []
        start_idx = 0
        for dim in view_dims:
            n_sample = max(1, int(subsample * dim / sum(view_dims)))
            view_dims_sampled.append(n_sample)
        view_dims = view_dims_sampled

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 0.3], width_ratios=[1, 0.05])

    # Main heatmap
    ax_main = fig.add_subplot(gs[0, 0])

    # Determine color scale
    if use_correlation:
        vmin, vmax = -1, 1
        cmap = 'RdBu_r'
        label = 'Correlation'
    else:
        vmax = np.max(np.abs(matrix))
        vmin = -vmax
        cmap = 'RdBu_r'
        label = 'Covariance'

    # Plot heatmap
    im = ax_main.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    # Add block boundaries
    boundaries = np.cumsum([0] + view_dims)
    for boundary in boundaries[1:-1]:
        ax_main.axhline(y=boundary - 0.5, color='black', linewidth=2)
        ax_main.axvline(x=boundary - 0.5, color='black', linewidth=2)

    # Add view labels
    for i, (start, end, name) in enumerate(zip(boundaries[:-1], boundaries[1:], view_names)):
        mid = (start + end) / 2
        ax_main.text(mid, -0.05 * matrix.shape[0], name,
                    ha='center', va='top', fontsize=12, fontweight='bold',
                    transform=ax_main.transData)
        ax_main.text(-0.05 * matrix.shape[1], mid, name,
                    ha='right', va='center', fontsize=12, fontweight='bold',
                    rotation=90, transform=ax_main.transData)

    ax_main.set_title(f'Block-Structured {"Correlation" if use_correlation else "Covariance"} Matrix\n'
                     f'({len(view_names)} views, total {matrix.shape[0]} features)',
                     fontsize=14, fontweight='bold', pad=20)
    ax_main.set_xlabel('Features (by view)', fontsize=12)
    ax_main.set_ylabel('Features (by view)', fontsize=12)

    # Colorbar
    ax_cbar = fig.add_subplot(gs[0, 1])
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label(label, fontsize=12)

    # Statistics panel
    ax_stats = fig.add_subplot(gs[1, :])
    ax_stats.axis('off')

    # Compute and display statistics
    stats = analyze_covariance_statistics(cov_structure)

    stats_text = "Summary Statistics:\n\n"
    stats_text += "Intra-modality (within-view):\n"
    for name in view_names:
        s = stats['intra_stats'][name]
        stats_text += f"  {name}: mean_corr={s['corr_mean']:.3f}, max_|corr|={s['corr_max']:.3f}\n"

    stats_text += "\nInter-modality (between-view):\n"
    for pair_name, s in stats['inter_stats'].items():
        stats_text += f"  {pair_name}: mean_corr={s['corr_mean']:.3f}, max_|corr|={s['corr_max']:.3f}\n"

    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if output_path:
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved block covariance matrix plot to {output_path}")
        except (AttributeError, OSError) as e:
            # Fallback: try saving with lower DPI or different format
            logger.warning(f"Failed to save with dpi=300: {e}. Trying fallback...")
            try:
                plt.savefig(output_path, dpi=150, bbox_inches='tight', format='png')
                logger.info(f"Saved with fallback settings to {output_path}")
            except Exception as e2:
                logger.error(f"Failed to save covariance plot: {e2}")
                # Don't crash - just skip this plot

    return fig


def plot_inter_vs_intra_comparison(
    cov_structure: Dict,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot comparison of inter-modality vs intra-modality correlations.

    Parameters
    ----------
    cov_structure : Dict
        Output from compute_covariance_structure
    output_path : Optional[Path]
        Path to save figure

    Returns
    -------
    matplotlib Figure
    """
    stats = analyze_covariance_statistics(cov_structure)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Distribution of correlations
    ax = axes[0]

    # Collect all intra-modality correlations
    intra_corrs = []
    intra_labels = []
    for name in cov_structure['view_names']:
        corr = cov_structure['intra_corr'][name]
        mask = ~np.eye(corr.shape[0], dtype=bool)
        intra_corrs.append(corr[mask].flatten())
        intra_labels.append(f"{name}\n(intra)")

    # Collect all inter-modality correlations
    inter_corrs = []
    inter_labels = []
    for pair_name, corr in cov_structure['inter_corr'].items():
        inter_corrs.append(corr.flatten())
        inter_labels.append(f"{pair_name}\n(inter)")

    # Combine
    all_corrs = intra_corrs + inter_corrs
    all_labels = intra_labels + inter_labels

    # Violin plot
    positions = np.arange(len(all_labels))
    parts = ax.violinplot(all_corrs, positions=positions, showmeans=True, showmedians=True)

    # Color intra vs inter differently
    n_intra = len(intra_corrs)
    for i, pc in enumerate(parts['bodies']):
        if i < n_intra:
            pc.set_facecolor('skyblue')
            pc.set_alpha(0.7)
        else:
            pc.set_facecolor('salmon')
            pc.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.set_title('Distribution of Correlations:\nIntra-modality vs Inter-modality',
                fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    # Right plot: Bar chart of mean absolute correlations
    ax = axes[1]

    mean_abs_corrs = []
    labels = []
    colors = []

    for name in cov_structure['view_names']:
        s = stats['intra_stats'][name]
        mean_abs_corrs.append(np.abs(s['corr_mean']))
        labels.append(f"{name}\n(intra)")
        colors.append('skyblue')

    for pair_name, s in stats['inter_stats'].items():
        mean_abs_corrs.append(np.abs(s['corr_mean']))
        labels.append(f"{pair_name}\n(inter)")
        colors.append('salmon')

    bars = ax.bar(range(len(labels)), mean_abs_corrs, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Mean |Correlation|', fontsize=12)
    ax.set_title('Mean Absolute Correlation:\nIntra-modality vs Inter-modality',
                fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', edgecolor='black', alpha=0.7, label='Intra-modality'),
        Patch(facecolor='salmon', edgecolor='black', alpha=0.7, label='Inter-modality')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved inter/intra comparison plot to {output_path}")

    return fig


def create_covariance_report(
    X_list: List[np.ndarray],
    view_names: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    subsample: int = 500,
) -> Dict:
    """Create comprehensive covariance analysis report.

    Parameters
    ----------
    X_list : List[np.ndarray]
        List of data matrices, one per view
    view_names : Optional[List[str]]
        Names for each view
    output_dir : Optional[Path]
        Directory to save plots
    subsample : int
        Number of features to subsample for visualization

    Returns
    -------
    Dict with covariance structure and statistics
    """
    logger.info("=" * 80)
    logger.info("COVARIANCE STRUCTURE ANALYSIS")
    logger.info("=" * 80)

    # Compute covariance structure
    cov_structure = compute_covariance_structure(X_list, view_names)
    stats = analyze_covariance_statistics(cov_structure)

    # Log summary
    logger.info(f"\nData structure:")
    logger.info(f"  Number of views: {len(X_list)}")
    logger.info(f"  Number of samples: {X_list[0].shape[0]}")
    for name, dim in zip(stats['view_names'], stats['view_dims']):
        logger.info(f"  {name}: {dim} features")

    logger.info(f"\nIntra-modality correlation statistics:")
    for name, s in stats['intra_stats'].items():
        logger.info(f"  {name}:")
        logger.info(f"    Mean correlation: {s['corr_mean']:.3f}")
        logger.info(f"    Std correlation: {s['corr_std']:.3f}")
        logger.info(f"    Max |correlation|: {s['corr_max']:.3f}")

    logger.info(f"\nInter-modality correlation statistics:")
    for pair_name, s in stats['inter_stats'].items():
        logger.info(f"  {pair_name}:")
        logger.info(f"    Mean correlation: {s['corr_mean']:.3f}")
        logger.info(f"    Std correlation: {s['corr_std']:.3f}")
        logger.info(f"    Max |correlation|: {s['corr_max']:.3f}")

    # Create plots if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nGenerating covariance plots...")

        # Block covariance matrix
        plot_block_covariance_matrix(
            cov_structure,
            output_path=output_dir / "block_covariance_matrix.png",
            use_correlation=True,
            subsample=subsample,
        )

        # Inter vs intra comparison
        plot_inter_vs_intra_comparison(
            cov_structure,
            output_path=output_dir / "inter_vs_intra_correlation.png",
        )

        logger.info(f"Plots saved to {output_dir}")

    logger.info("=" * 80)

    return {
        'cov_structure': cov_structure,
        'stats': stats,
    }
