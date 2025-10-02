"""
Refactored visualization module - delegates to specialized visualization modules.
This is the main interface for all visualization functionality.
"""

import contextlib
import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from visualization.brain_plots import BrainVisualizer
from visualization.cv_plots import CrossValidationVisualizer
from visualization.factor_plots import FactorVisualizer

# Specialized visualization modules
from visualization.manager import VisualizationManager
from visualization.preprocessing_plots import PreprocessingVisualizer
from visualization.report_generator import ReportGenerator

logging.captureWarnings(True)
logger = logging.getLogger(__name__)


# === UTILITY FUNCTIONS ===
@contextlib.contextmanager
def safe_plotting_context(figsize=None, dpi=300):
    """Context manager for safe plotting with automatic cleanup."""
    import matplotlib.pyplot as plt

    plt.get_backend()

    if figsize:
        plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi

    try:
        yield plt
    except Exception as e:
        logger.error(f"Plotting error: {e}")
        raise
    finally:
        plt.close("all")
        gc.collect()


def save_plot_safely(fig, filepath, formats=["png", "pdf"], **kwargs):
    """Safely save plot in multiple formats with error handling."""
    filepath = Path(filepath)
    saved_files = []

    for fmt in formats:
        try:
            output_path = filepath.with_suffix(f".{fmt}")
            fig.savefig(output_path, format=fmt, **kwargs)
            saved_files.append(output_path)
            logger.debug(f"Saved plot: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save {filepath}.{fmt}: {e}")

    return saved_files


def setup_plotting_style():
    """Setup consistent plotting style across all visualizations."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set style
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    # Update rcParams
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
        }
    )


def visualization_with_error_handling(func):
    """Decorator for safe visualization with error handling."""

    def wrapper(*args, **kwargs):
        try:
            with safe_plotting_context():
                return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Visualization function {func.__name__} failed: {e}")
            logger.exception("Full traceback:")
            return None

    return wrapper


# === MAIN VISUALIZATION FUNCTIONS ===


class VisualizationConfig:
    """Configuration for visualization settings."""

    def __init__(
        self,
        dpi: int = 300,
        figure_size: tuple = (10, 6),
        style: str = "seaborn-v0_8-darkgrid",
        save_formats: List[str] = ["png"],
        color_palette: str = "husl",
        dataset: str = "qmap_pd",
        K: int = 5,
        percW: float = 25.0,
        cv_type: str = "standard",
        cv_folds: int = 5,
    ):
        self.dpi = dpi
        self.figure_size = figure_size
        self.style = style
        self.save_formats = save_formats
        self.color_palette = color_palette

        # Analysis configuration attributes required by ReportGenerator
        self.dataset = dataset
        self.K = K
        self.percW = percW
        self.cv_type = cv_type
        self.cv_folds = cv_folds


@visualization_with_error_handling
def synthetic_data(res_dir: str, true_params: Dict, args: Any, hypers: Dict):
    """
    Create visualizations for synthetic data analysis.

    Parameters:
    -----------
    res_dir : str
        Results directory path
    true_params : dict
        True parameter values for synthetic data
    args : object
        Analysis arguments
    hypers : dict
        Hyperparameters
    """
    logger.info(f"Creating synthetic data visualizations in {res_dir}")

    config = VisualizationConfig()

    # Initialize specialized visualizers
    FactorVisualizer(config)
    report_gen = ReportGenerator(config)

    plot_path = Path(res_dir) / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load results
        rob_params, brun = _load_results(res_dir, args)
        if rob_params is None:
            logger.warning("No robust parameters found for visualization")
            return

        # Create synthetic data specific plots
        _plot_ground_truth_components(true_params, plot_path, args, hypers)
        _plot_inferred_components(rob_params, true_params, plot_path, args, hypers)
        _plot_factor_comparison(true_params, rob_params, plot_path, args)
        _plot_subgroup_analysis(true_params, rob_params, plot_path, args)

        # Create comprehensive report
        report_data = {
            "analysis_type": "synthetic_data",
            "true_params": true_params,
            "inferred_params": rob_params,
            "args": vars(args),
            "hypers": hypers,
            "best_run": brun,
        }

        # Generate HTML report using the correct method
        plot_dir = Path(res_dir) / "plots"
        report_gen.generate_html_report(plot_dir, report_data, analysis_results=report_data)

        logger.info("Synthetic data visualization completed")

    except Exception as e:
        logger.error(f"Synthetic data visualization failed: {e}")
        raise


@visualization_with_error_handling
def qmap_pd(data: Dict, res_dir: str, args: Any, hypers: Dict, topk: int = 20):
    """
    Create visualizations for qMAP-PD data analysis.

    Parameters:
    -----------
    data : dict
        Loaded qMAP-PD data
    res_dir : str
        Results directory path
    args : object
        Analysis arguments
    hypers : dict
        Hyperparameters
    topk : int
        Number of top features to highlight
    """
    logger.info(f"Creating qMAP-PD visualizations in {res_dir}")

    # Create config with actual analysis parameters using ParameterResolver
    from core.parameter_resolver import ParameterResolver

    params = ParameterResolver(args, hypers)
    config = VisualizationConfig(
        dataset="qmap_pd",
        K=params.get('K', default=5),
        percW=params.get('percW', default=25.0),
        cv_type=params.get('cv_type', default='standard'),
        cv_folds=params.get('cv_folds', default=5)
    )

    # Initialize specialized visualizers
    FactorVisualizer(config)
    brain_viz = BrainVisualizer(config)
    report_gen = ReportGenerator(config)

    plot_path = Path(res_dir) / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load results
        rob_params, brun = _load_results(res_dir, args)
        if rob_params is None:
            logger.warning("No robust parameters found for visualization")
            return

        # Extract data components
        X_list = data.get("X_list", [])
        view_names = data.get(
            "view_names", [f"View_{i + 1}" for i in range(len(X_list))]
        )
        feat_names = data.get("feature_names", {})
        sub_ids = data.get("subject_ids", [])

        # Calculate dimensions
        Dm = [X.shape[1] for X in X_list] if X_list else hypers.get("Dm", [])

        # Create qMAP-PD specific visualizations
        if "W" in rob_params and "Z" in rob_params:
            W, Z = rob_params["W"], rob_params["Z"]

            # Multi-view factor loadings
            _plot_multiview_loadings(W, Dm, view_names, feat_names, plot_path, topk)

            # Subject factor scores
            _plot_subject_scores(Z, sub_ids, plot_path)

            # Latent factor summary
            _plot_latent_factor_summary(W, Z, Dm, view_names, plot_path)

        # Brain-specific visualizations
        brain_summary = brain_viz.create_brain_visualization_summary(res_dir)

        # Create comprehensive report
        report_data = {
            "analysis_type": "qmap_pd",
            "data_summary": {
                "n_subjects": X_list[0].shape[0] if X_list else 0,
                "n_views": len(X_list),
                "view_names": view_names,
                "dimensions": Dm,
            },
            "results": rob_params,
            "brain_analysis": brain_summary,
            "args": vars(args),
            "hypers": hypers,
            "best_run": brun,
        }

        # Generate HTML report using the correct method
        plot_dir = Path(res_dir) / "plots"
        report_gen.generate_html_report(plot_dir, report_data, analysis_results=report_data)

        logger.info("qMAP-PD visualization completed")

    except Exception as e:
        logger.error(f"qMAP-PD visualization failed: {e}")
        raise


# === DELEGATED FUNCTIONS ===


def plot_preprocessing_summary(
    preprocessing_results: Dict, plot_path: str, view_names: List[str]
):
    """Delegate to preprocessing visualizer."""
    config = VisualizationConfig()
    prep_viz = PreprocessingVisualizer(config)
    return prep_viz.plot_preprocessing_summary(
        preprocessing_results, plot_path, view_names
    )


def plot_cv_results(
    cv_results: Dict, plot_path: str, run_name: str = "cross_validation"
):
    """Delegate to CV visualizer."""
    config = VisualizationConfig()
    cv_viz = CrossValidationVisualizer(config)
    return cv_viz.plot_cv_results(cv_results, plot_path, run_name)


def plot_consensus_subtypes(
    centroids_data: Dict, probabilities_data: Dict, plot_path: str
):
    """Delegate to CV visualizer."""
    config = VisualizationConfig()
    cv_viz = CrossValidationVisualizer(config)
    return cv_viz.plot_consensus_subtypes(centroids_data, probabilities_data, plot_path)


def create_brain_visualization_summary(
    results_dir: str, include_reconstructions: bool = True
):
    """Delegate to brain visualizer."""
    config = VisualizationConfig()
    brain_viz = BrainVisualizer(config)
    return brain_viz.create_brain_visualization_summary(
        results_dir, include_reconstructions
    )


def create_all_visualizations(
    results_dir: str,
    data: Dict,
    run_name: str,
    cv_results: Optional[Dict] = None,
    neuroimaging_metrics: Optional[Dict] = None,
):
    """
    Create comprehensive visualizations using all available modules.

    Parameters:
    -----------
    results_dir : str
        Directory containing analysis results
    data : dict
        Analysis data
    run_name : str
        Name for the analysis run
    cv_results : dict, optional
        Cross-validation results
    neuroimaging_metrics : dict, optional
        Neuroimaging-specific metrics
    """
    logger.info(f"Creating comprehensive visualizations for {run_name}")

    config = VisualizationConfig()
    viz_manager = VisualizationManager(config)

    return viz_manager.create_comprehensive_analysis(
        results_dir, data, run_name, cv_results, neuroimaging_metrics
    )


# === UTILITY FUNCTIONS (kept for backward compatibility) ===


def find_bestrun(res_dir: str, args: Any, ofile: str = None) -> int:
    """Find the best run based on log density."""
    from core.utils import get_model_files, safe_pickle_load

    best_run = 1
    best_logdens = -np.inf

    for run_id in range(1, args.num_runs + 1):
        files = get_model_files(res_dir, run_id)
        mcmc_samples = safe_pickle_load(
            files["model_params"], description=f"MCMC samples run {run_id}"
        )

        if mcmc_samples and "exp_logdensity" in mcmc_samples:
            logdens = mcmc_samples["exp_logdensity"]
            if logdens > best_logdens:
                best_logdens = logdens
                best_run = run_id

    if ofile:
        with open(Path(res_dir) / ofile, "w") as f:
            f.write(f"Best run: {best_run}\n")
            f.write(f"Log density: {best_logdens}\n")

    return best_run


def _load_results(res_dir: str, args: Any):
    """Load analysis results for visualization."""
    from core.utils import get_model_files, safe_pickle_load

    # Find best run
    brun = find_bestrun(res_dir, args, "results.txt")

    # Load robust parameters
    files = get_model_files(res_dir, brun)
    rob_params = safe_pickle_load(files["robust_params"], description="Robust parameters")

    return rob_params, brun


# === HELPER FUNCTIONS (simplified versions of original functions) ===


def _plot_ground_truth_components(
    true_params: Dict, plot_path: Path, args: Any, hypers: Dict
):
    """Plot ground truth components for synthetic data."""
    # Simplified implementation - delegate to factor visualizer
    config = VisualizationConfig()
    factor_viz = FactorVisualizer(config)

    if "W_true" in true_params and "Z_true" in true_params:
        factor_viz.plot_factor_loadings(
            true_params["W_true"],
            title="Ground Truth Factor Loadings",
            save_path=plot_path / "ground_truth_loadings.png",
        )

        factor_viz.plot_factor_scores(
            true_params["Z_true"],
            title="Ground Truth Factor Scores",
            save_path=plot_path / "ground_truth_scores.png",
        )


def _plot_inferred_components(
    rob_params: Dict, true_params: Dict, plot_path: Path, args: Any, hypers: Dict
):
    """Plot inferred components for synthetic data."""
    config = VisualizationConfig()
    factor_viz = FactorVisualizer(config)

    if "W" in rob_params and "Z" in rob_params:
        factor_viz.plot_factor_loadings(
            rob_params["W"],
            title="Inferred Factor Loadings",
            save_path=plot_path / "inferred_loadings.png",
        )

        factor_viz.plot_factor_scores(
            rob_params["Z"],
            title="Inferred Factor Scores",
            save_path=plot_path / "inferred_scores.png",
        )


def _plot_factor_comparison(
    true_params: Dict, rob_params: Dict, plot_path: Path, args: Any
):
    """Compare true vs inferred factors."""
    config = VisualizationConfig()
    factor_viz = FactorVisualizer(config)

    if all(key in true_params for key in ["W_true", "Z_true"]) and all(
        key in rob_params for key in ["W", "Z"]
    ):
        factor_viz.plot_factor_comparison(
            true_params["W_true"],
            rob_params["W"],
            true_params["Z_true"],
            rob_params["Z"],
            save_path=plot_path / "factor_comparison.png",
        )


def _plot_subgroup_analysis(
    true_params: Dict, rob_params: Dict, plot_path: Path, args: Any
):
    """Plot subgroup analysis for synthetic data."""
    # Placeholder for subgroup analysis
    logger.info("Subgroup analysis visualization - placeholder implementation")


def _plot_multiview_loadings(
    W: np.ndarray,
    Dm: List[int],
    view_names: List[str],
    feat_names: Dict,
    plot_path: Path,
    topk: int,
):
    """Plot multi-view factor loadings."""
    config = VisualizationConfig()
    factor_viz = FactorVisualizer(config)

    factor_viz.plot_multiview_loadings(
        W,
        Dm,
        view_names,
        feat_names,
        topk,
        save_path=plot_path / "multiview_loadings.png",
    )


def _plot_subject_scores(Z: np.ndarray, sub_ids: List, plot_path: Path):
    """Plot subject factor scores."""
    config = VisualizationConfig()
    factor_viz = FactorVisualizer(config)

    factor_viz.plot_factor_scores(
        Z, subject_ids=sub_ids, save_path=plot_path / "subject_scores.png"
    )


def _plot_latent_factor_summary(
    W: np.ndarray, Z: np.ndarray, Dm: List[int], view_names: List[str], plot_path: Path
):
    """Plot latent factor summary."""
    config = VisualizationConfig()
    factor_viz = FactorVisualizer(config)

    factor_viz.plot_factor_summary(
        W, Z, Dm, view_names, save_path=plot_path / "latent_factor_summary.png"
    )


# === BACKWARD COMPATIBILITY ===
# Keep some functions for backward compatibility but log deprecation warnings


def plot_param(*args, **kwargs):
    """Deprecated: Use FactorVisualizer.plot_factor_loadings instead."""
    logger.warning(
        "plot_param is deprecated. Use FactorVisualizer.plot_factor_loadings instead."
    )
    # Implement minimal backward compatibility or raise NotImplementedError


def plot_X(*args, **kwargs):
    """Deprecated: Use FactorVisualizer.plot_data_reconstruction instead."""
    logger.warning(
        "plot_X is deprecated. Use FactorVisualizer.plot_data_reconstruction instead."
    )
    # Implement minimal backward compatibility or raise NotImplementedError


def plot_model_comparison(*args, **kwargs):
    """Deprecated: Use ReportGenerator.create_model_comparison_report instead."""
    logger.warning(
        "plot_model_comparison is deprecated. Use ReportGenerator.create_model_comparison_report instead."
    )
    # Implement minimal backward compatibility or raise NotImplementedError
