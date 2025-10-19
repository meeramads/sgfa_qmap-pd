# visualization/preprocessing_plots.py
"""Preprocessing visualization module."""

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from core.io_utils import save_plot

logger = logging.getLogger(__name__)


def _format_view_name(view_name: str) -> str:
    """Convert technical view names to human-readable format."""
    # Handle ROI-specific names first (preferred)
    name_map = {
        "volume_sn_voxels": "Substantia Nigra",
        "volume_putamen_voxels": "Putamen",
        "volume_lentiform_voxels": "Lentiform Nucleus",
        "volume_bg-all_voxels": "All ROIs (Basal Ganglia)",
        "imaging": "Imaging Data",
        "clinical": "Clinical Data"
    }

    if view_name in name_map:
        return name_map[view_name]

    # Handle generic view_N names as fallback
    if view_name.startswith("view_"):
        try:
            view_idx = view_name.split('_')[1]
            return f"View {view_idx}"
        except (IndexError, ValueError):
            pass

    # Fallback: Clean up the name by replacing underscores and title-casing
    return view_name.replace("_", " ").replace("voxels", "").replace("volume", "").strip().title()


class PreprocessingVisualizer:
    """Creates preprocessing visualizations."""

    def __init__(self, config):
        self.config = config

    def create_plots(self, preprocessing_results: Dict, plot_dir: Path):
        """Create preprocessing visualization plots."""
        logger.info("Creating preprocessing plots")

        save_dir = plot_dir / "preprocessing"
        save_dir.mkdir(exist_ok=True)

        # Feature reduction plot
        if "feature_reduction" in preprocessing_results:
            self._plot_feature_reduction(
                preprocessing_results["feature_reduction"], save_dir
            )

        # Variance distribution plot (shows why features were retained/removed)
        if "variance_analysis" in preprocessing_results:
            self._plot_variance_distribution(
                preprocessing_results["variance_analysis"], save_dir
            )

        # Source validation plot
        if "source_validation" in preprocessing_results:
            self._plot_source_validation(
                preprocessing_results["source_validation"], save_dir
            )

        # Optimization results
        if "optimization" in preprocessing_results:
            self._plot_optimization_results(
                preprocessing_results["optimization"], save_dir
            )

    def _plot_feature_reduction(self, feature_reduction: Dict, save_dir: Path):
        """Plot feature reduction summary."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Extract data and format view names
        views = list(feature_reduction.keys())
        formatted_views = [_format_view_name(v) for v in views]
        original = [feature_reduction[v]["original"] for v in views]
        processed = [feature_reduction[v]["processed"] for v in views]
        retention = [feature_reduction[v]["reduction_ratio"] for v in views]

        # Bar plot of feature counts
        x = np.arange(len(views))
        width = 0.35

        ax1.bar(x - width / 2, original, width, label="Original", alpha=0.7)
        ax1.bar(x + width / 2, processed, width, label="Processed", alpha=0.7)
        ax1.set_xlabel("View")
        ax1.set_ylabel("Number of Features")
        ax1.set_title("Feature Count Reduction")
        ax1.set_xticks(x)
        ax1.set_xticklabels(formatted_views, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Retention ratio plot
        x_pos = np.arange(len(views))
        ax2.bar(
            x_pos,
            retention,
            alpha=0.7,
            color=["green" if r > 0.5 else "orange" for r in retention],
        )
        ax2.set_xlabel("View")
        ax2.set_ylabel("Retention Ratio")
        ax2.set_title("Feature Retention Ratios")
        ax2.set_ylim(0, 1.1)
        ax2.axhline(y=0.5, color="r", linestyle="--", alpha=0.5)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(formatted_views, rotation=45, ha="right")
        ax2.grid(True, alpha=0.3)

        # Add percentage labels
        for i, (v, r) in enumerate(zip(views, retention)):
            ax2.text(i, r + 0.02, f"{r:.1%}", ha="center")

        plt.tight_layout()

        # Save
        save_path = save_dir / "feature_reduction.png"
        save_plot(save_path)
        logger.info(f"Saved: {save_path}")

    def _plot_variance_distribution(self, variance_analysis: Dict, save_dir: Path):
        """Plot feature variance distributions to show why features were retained/removed."""
        n_views = len(variance_analysis)
        fig, axes = plt.subplots(1, n_views, figsize=(6 * n_views, 5))

        if n_views == 1:
            axes = [axes]

        for ax, (view_name, var_data) in zip(axes, variance_analysis.items()):
            variances = var_data["variances"]
            threshold = var_data.get("threshold", 0.0)
            n_retained = var_data.get("n_retained", len(variances))
            n_total = len(variances)

            # Create histogram
            ax.hist(variances, bins=50, alpha=0.7, edgecolor='black')

            # Add threshold line if present
            if threshold > 0:
                ax.axvline(threshold, color='r', linestyle='--', linewidth=2,
                          label=f'Threshold: {threshold:.3f}')

            # Add text annotation
            retention_pct = (n_retained / n_total) * 100
            ax.text(0.95, 0.95, f'{n_retained}/{n_total}\nfeatures retained\n({retention_pct:.1f}%)',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel('Feature Variance')
            ax.set_ylabel('Count')
            # Format view name to be human-readable
            formatted_name = _format_view_name(view_name)
            ax.set_title(f'{formatted_name}\nVariance Distribution', fontsize=10)
            ax.grid(True, alpha=0.3)
            if threshold > 0:
                ax.legend(fontsize=8, loc='upper right', framealpha=0.9, ncol=1)

        plt.tight_layout()

        # Save
        save_path = save_dir / "variance_distribution.png"
        save_plot(save_path)
        logger.info(f"Saved: {save_path}")

    def _plot_source_validation(self, source_validation: Dict, save_dir: Path):
        """Plot source combination validation results."""
        # Sort by performance
        sorted_items = sorted(
            source_validation.items(), key=lambda x: x[1]["rmse_mean"]
        )

        fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_items) * 0.3)))

        # Extract data
        names = [item[0] for item in sorted_items]
        means = [item[1]["rmse_mean"] for item in sorted_items]
        stds = [item[1]["rmse_std"] for item in sorted_items]

        # Horizontal bar plot
        y_pos = np.arange(len(names))
        ax.barh(y_pos, means, xerr=stds, alpha=0.7, capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel("RMSE")
        ax.set_title("Source Combination Performance", fontsize=10)
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        # Save
        save_path = save_dir / "source_validation.png"
        save_plot(save_path)
        logger.info(f"Saved: {save_path}")

    def _plot_optimization_results(self, optimization: Dict, save_dir: Path):
        """Plot preprocessing optimization results."""
        if "all_results" not in optimization:
            return

        results = optimization["all_results"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Extract data
        imputation_methods = [r["params"]["imputation_strategy"] for r in results]
        scores = [r["score"] for r in results]
        feature_counts = [r["n_features_total"] for r in results]

        # Plot 1: Score by imputation method
        unique_methods = list(set(imputation_methods))
        method_scores = {m: [] for m in unique_methods}
        for method, score in zip(imputation_methods, scores):
            method_scores[method].append(score)

        axes[0, 0].boxplot(method_scores.values(), labels=method_scores.keys())
        axes[0, 0].set_xlabel("Imputation Method")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].set_title("Performance by Imputation")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Score vs feature count
        axes[0, 1].scatter(feature_counts, scores, alpha=0.6)
        axes[0, 1].set_xlabel("Total Features")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].set_title("Score vs Feature Count")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Parameter combinations heatmap
        # ... (implementation similar to original)

        plt.suptitle(
            "Preprocessing Optimization Results", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        # Save
        save_path = save_dir / "optimization_results.png"
        save_plot(save_path)
        logger.info(f"Saved: {save_path}")
