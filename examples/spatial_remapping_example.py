#!/usr/bin/env python3
"""
Spatial Remapping Example: Map SGFA Factor Loadings to Brain Space

This script demonstrates how to remap factor loadings back to 3D brain
coordinates after feature selection, enabling visualization and interpretation
of factors in anatomical space.

Usage:
    python examples/spatial_remapping_example.py --results-dir results/my_experiment
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing import (
    get_selected_voxel_positions,
    create_brain_map_from_factors,
    remap_all_factors_to_brain
)


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_sgfa_results(results_dir: Path):
    """Load SGFA results from experiment directory."""
    logger = logging.getLogger(__name__)

    # Load results (adapt to your actual save format)
    # This is a placeholder - adjust based on how you save SGFA results
    try:
        import pickle
        with open(results_dir / 'sgfa_results.pkl', 'rb') as f:
            results = pickle.load(f)
        logger.info(f"✓ Loaded SGFA results from {results_dir}")
        return results
    except Exception as e:
        logger.error(f"Could not load results: {e}")
        logger.info("This example assumes results are saved as sgfa_results.pkl")
        logger.info("Adjust load_sgfa_results() function for your format")
        return None


def plot_brain_map_3d(brain_map: pd.DataFrame, output_path: Path, title: str = "Brain Map"):
    """
    Create 3D scatter plot of brain map.

    Parameters
    ----------
    brain_map : pd.DataFrame
        Brain map with columns: ['x', 'y', 'z', 'loading']
    output_path : Path
        Where to save the plot
    title : str
        Plot title
    """
    logger = logging.getLogger(__name__)

    # Check if we have 3D coordinates
    if not all(col in brain_map.columns for col in ['x', 'y', 'z']):
        logger.warning("Brain map missing x/y/z coordinates, skipping 3D plot")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Extract data
    x = brain_map['x'].values
    y = brain_map['y'].values
    z = brain_map['z'].values
    loadings = brain_map['loading'].values

    # Color by loading magnitude
    colors = np.abs(loadings)

    # Plot
    scatter = ax.scatter(
        x, y, z,
        c=colors,
        cmap='hot',
        s=50,
        alpha=0.6,
        edgecolors='k',
        linewidth=0.5
    )

    # Labels
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_zlabel('Z Coordinate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('|Factor Loading|', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  ✓ Saved 3D plot: {output_path}")


def plot_brain_map_slices(brain_map: pd.DataFrame, output_path: Path, title: str = "Brain Map"):
    """
    Create 2D slice plots (axial, coronal, sagittal).

    Parameters
    ----------
    brain_map : pd.DataFrame
        Brain map with columns: ['x', 'y', 'z', 'loading']
    output_path : Path
        Where to save the plot
    title : str
        Plot title
    """
    logger = logging.getLogger(__name__)

    if not all(col in brain_map.columns for col in ['x', 'y', 'z']):
        logger.warning("Brain map missing x/y/z coordinates, skipping slice plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = brain_map['x'].values
    y = brain_map['y'].values
    z = brain_map['z'].values
    loadings = brain_map['loading'].values
    colors = np.abs(loadings)

    # Axial (z-plane, view from top)
    scatter1 = axes[0].scatter(x, y, c=colors, cmap='hot', s=30, alpha=0.6)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('Axial View')
    axes[0].set_aspect('equal')

    # Coronal (y-plane, view from front)
    scatter2 = axes[1].scatter(x, z, c=colors, cmap='hot', s=30, alpha=0.6)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].set_title('Coronal View')
    axes[1].set_aspect('equal')

    # Sagittal (x-plane, view from side)
    scatter3 = axes[2].scatter(y, z, c=colors, cmap='hot', s=30, alpha=0.6)
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    axes[2].set_title('Sagittal View')
    axes[2].set_aspect('equal')

    # Shared colorbar
    fig.colorbar(scatter1, ax=axes, label='|Factor Loading|', shrink=0.8)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  ✓ Saved slice plots: {output_path}")


def analyze_brain_map(brain_map: pd.DataFrame, factor_name: str):
    """Print analysis of brain map."""
    logger = logging.getLogger(__name__)

    logger.info(f"\n=== Analysis: {factor_name} ===")
    logger.info(f"Total voxels: {len(brain_map)}")

    # Loading statistics
    loadings = brain_map['loading'].values
    logger.info(f"Loading range: [{loadings.min():.3f}, {loadings.max():.3f}]")
    logger.info(f"Loading mean: {loadings.mean():.3f}")
    logger.info(f"Loading std: {loadings.std():.3f}")

    # Top voxels
    logger.info("\nTop 5 voxels (by absolute loading):")
    top5 = brain_map.nlargest(5, 'loading', keep='first')
    for idx, row in top5.iterrows():
        if all(col in brain_map.columns for col in ['x', 'y', 'z']):
            logger.info(f"  ({row['x']:.0f}, {row['y']:.0f}, {row['z']:.0f}): {row['loading']:.3f}")
        else:
            logger.info(f"  Voxel {idx}: {row['loading']:.3f}")

    # Center of mass (if coordinates available)
    if all(col in brain_map.columns for col in ['x', 'y', 'z']):
        weights = np.abs(loadings)
        weights = weights / weights.sum()

        com_x = np.average(brain_map['x'].values, weights=weights)
        com_y = np.average(brain_map['y'].values, weights=weights)
        com_z = np.average(brain_map['z'].values, weights=weights)

        logger.info(f"\nCenter of mass: ({com_x:.1f}, {com_y:.1f}, {com_z:.1f})")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Spatial Remapping Example for SGFA Factor Loadings"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/example_experiment',
        help='Directory containing SGFA results'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='qMAP-PD_data',
        help='Data directory with position lookups'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='brain_maps',
        help='Output directory for brain maps and plots'
    )

    args = parser.parse_args()
    logger = setup_logging()

    # Setup directories
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Spatial Remapping Example")
    logger.info("=" * 80)

    # Load SGFA results
    logger.info(f"\n1. Loading SGFA results from: {results_dir}")
    results = load_sgfa_results(results_dir)

    if results is None:
        logger.error("Failed to load results. Exiting.")
        return 1

    # Extract components
    W_list = results.get('W')
    view_names = results.get('view_names')
    preprocessor = results.get('preprocessor')

    if not all([W_list, view_names, preprocessor]):
        logger.error("Results missing required components (W, view_names, preprocessor)")
        return 1

    logger.info(f"✓ Found {len(W_list)} views, {W_list[0].shape[1]} factors")

    # Remap all factors to brain space
    logger.info("\n2. Remapping all factors to brain space...")
    all_brain_maps = remap_all_factors_to_brain(
        W_list=W_list,
        view_names=view_names,
        preprocessor=preprocessor,
        data_dir=args.data_dir
    )

    logger.info(f"✓ Created brain maps for {len(all_brain_maps)} imaging views")

    # Save brain maps and create visualizations
    logger.info("\n3. Saving brain maps and creating visualizations...")

    for view_name, factor_maps in all_brain_maps.items():
        roi = view_name.replace('volume_', '').replace('_voxels', '')
        logger.info(f"\nProcessing view: {view_name} ({roi})")

        # Create subdirectory for this ROI
        roi_dir = output_dir / roi
        roi_dir.mkdir(exist_ok=True)

        for k, brain_map in factor_maps.items():
            factor_name = f"{roi}_factor{k}"

            # Save CSV
            csv_path = roi_dir / f"{factor_name}.csv"
            brain_map.to_csv(csv_path, index=False)
            logger.info(f"  ✓ Saved: {csv_path}")

            # Analyze
            analyze_brain_map(brain_map, factor_name)

            # Create 3D plot
            plot_path_3d = roi_dir / f"{factor_name}_3d.png"
            plot_brain_map_3d(brain_map, plot_path_3d, f"{roi.upper()} Factor {k}")

            # Create slice plots
            plot_path_slices = roi_dir / f"{factor_name}_slices.png"
            plot_brain_map_slices(brain_map, plot_path_slices, f"{roi.upper()} Factor {k}")

    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Complete! Brain maps saved to: {output_dir}")
    logger.info("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
