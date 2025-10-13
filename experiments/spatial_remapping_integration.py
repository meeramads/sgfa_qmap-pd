"""
Spatial Remapping Integration for Experiments

This module provides integration between SGFA experiments and spatial remapping utilities,
making it easy to automatically save brain maps after running experiments.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from data.preprocessing import remap_all_factors_to_brain

logger = logging.getLogger(__name__)


def save_brain_maps_from_results(
    results: Dict,
    output_dir: Path,
    data_dir: str,
    save_csv: bool = True,
    create_plots: bool = False
) -> Dict[str, Dict[int, Path]]:
    """
    Automatically save brain maps from SGFA results.

    Parameters
    ----------
    results : Dict
        SGFA results dictionary containing 'W', 'view_names', 'preprocessor'
    output_dir : Path
        Directory to save brain maps
    data_dir : str
        Data directory containing position lookup files
    save_csv : bool
        Save brain maps as CSV files
    create_plots : bool
        Create visualization plots (requires matplotlib)

    Returns
    -------
    Dict[str, Dict[int, Path]]
        Nested dict: {view_name: {factor_k: csv_path}}

    Examples
    --------
    >>> results = run_sgfa(X_list, hypers, args)
    >>> brain_map_paths = save_brain_maps_from_results(
    ...     results,
    ...     output_dir=Path('results/brain_maps'),
    ...     data_dir='qMAP-PD_data'
    ... )
    """
    logger.info("🧠 Saving brain maps from SGFA results...")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract components
    W_list = results.get('W')
    view_names = results.get('view_names')
    preprocessor = results.get('preprocessor')

    if not all([W_list, view_names, preprocessor]):
        logger.warning("Results missing required components for spatial remapping")
        return {}

    # Remap all factors
    try:
        all_brain_maps = remap_all_factors_to_brain(
            W_list=W_list,
            view_names=view_names,
            preprocessor=preprocessor,
            data_dir=data_dir
        )
    except Exception as e:
        logger.error(f"Failed to remap factors to brain space: {e}")
        return {}

    # Save brain maps
    saved_paths = {}

    for view_name, factor_maps in all_brain_maps.items():
        roi = view_name.replace('volume_', '').replace('_voxels', '')
        logger.info(f"  Processing {roi}: {len(factor_maps)} factors")

        # Create subdirectory for this ROI
        roi_dir = output_dir / roi
        roi_dir.mkdir(exist_ok=True)

        view_paths = {}

        for k, brain_map in factor_maps.items():
            factor_name = f"{roi}_factor{k}"

            # Save CSV
            if save_csv:
                csv_path = roi_dir / f"{factor_name}.csv"
                brain_map.to_csv(csv_path, index=False)
                view_paths[k] = csv_path

            # Create plots (optional)
            if create_plots:
                try:
                    plot_path = roi_dir / f"{factor_name}_viz.png"
                    _create_brain_map_plot(brain_map, plot_path, factor_name)
                except Exception as e:
                    logger.warning(f"Could not create plot for {factor_name}: {e}")

        saved_paths[view_name] = view_paths

    total_maps = sum(len(v) for v in saved_paths.values())
    logger.info(f"✓ Saved {total_maps} brain maps to {output_dir}")

    return saved_paths


def _create_brain_map_plot(brain_map, output_path: Path, title: str):
    """Create simple visualization of brain map."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        if not all(col in brain_map.columns for col in ['x', 'y', 'z']):
            logger.debug("Brain map missing coordinates, skipping plot")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        x = brain_map['x'].values
        y = brain_map['y'].values
        z = brain_map['z'].values
        loadings = brain_map['loading'].values
        colors = np.abs(loadings)

        scatter = ax.scatter(x, y, z, c=colors, cmap='hot', s=30, alpha=0.6)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

        plt.colorbar(scatter, label='|Loading|')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    except ImportError:
        logger.debug("matplotlib not available for plotting")
    except Exception as e:
        logger.warning(f"Plot creation failed: {e}")


def integrate_spatial_remapping_to_experiment(
    experiment_result,
    config,
    data_dir: str
) -> None:
    """
    Integrate spatial remapping into experiment results.

    This function can be called at the end of any experiment to automatically
    save brain maps if spatial remapping is enabled in config.

    Parameters
    ----------
    experiment_result : ExperimentResult
        Experiment result object containing model_results
    config : ExperimentConfig
        Experiment configuration
    data_dir : str
        Data directory

    Examples
    --------
    >>> # In your experiment
    >>> result = ExperimentResult(...)
    >>> integrate_spatial_remapping_to_experiment(result, config, data_dir)
    """
    # Check if spatial remapping is enabled
    experiments_config = getattr(config, 'experiments', {})
    if isinstance(experiments_config, dict):
        enable_remapping = experiments_config.get('enable_spatial_remapping', False)
        save_brain_maps = experiments_config.get('save_brain_maps', True)
        create_plots = experiments_config.get('create_brain_map_plots', False)
        brain_maps_subdir = experiments_config.get('brain_maps_output_dir', 'brain_maps')
    else:
        enable_remapping = getattr(experiments_config, 'enable_spatial_remapping', False)
        save_brain_maps = getattr(experiments_config, 'save_brain_maps', True)
        create_plots = getattr(experiments_config, 'create_brain_map_plots', False)
        brain_maps_subdir = getattr(experiments_config, 'brain_maps_output_dir', 'brain_maps')

    if not enable_remapping:
        logger.debug("Spatial remapping disabled in config")
        return

    # Get output directory from config
    try:
        from core.config_utils import get_output_dir
        base_output_dir = get_output_dir(config)
    except:
        logger.warning("Could not get output directory from config")
        return

    output_dir = base_output_dir / brain_maps_subdir

    # Get SGFA results from experiment
    model_results = experiment_result.model_results if hasattr(experiment_result, 'model_results') else {}

    if 'sgfa_result' in model_results:
        results = model_results['sgfa_result']
    elif all(k in model_results for k in ['W', 'view_names', 'preprocessor']):
        results = model_results
    else:
        logger.debug("No SGFA results found for spatial remapping")
        return

    # Save brain maps
    try:
        brain_map_paths = save_brain_maps_from_results(
            results=results,
            output_dir=output_dir,
            data_dir=data_dir,
            save_csv=save_brain_maps,
            create_plots=create_plots
        )

        # Store paths in experiment result
        if hasattr(experiment_result, 'metadata'):
            experiment_result.metadata['brain_map_paths'] = brain_map_paths

        logger.info(f"✓ Spatial remapping complete: {len(brain_map_paths)} views")

    except Exception as e:
        logger.error(f"Spatial remapping integration failed: {e}")
