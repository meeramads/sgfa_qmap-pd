"""
Brain visualization module for neuroimaging data.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from core.io_utils import save_plot

logger = logging.getLogger(__name__)


class BrainVisualizer:
    """Creates brain-specific visualizations for neuroimaging data."""

    def __init__(self, config):
        self.config = config
        self.setup_style()

        # Check for factor mapping availability
        self.factor_mapping_available = self._check_factor_mapping()

    def setup_style(self):
        """Setup consistent plotting style."""
        plt.style.use("seaborn-v0_8-darkgrid")
        plt.rcParams.update(
            {
                "figure.figsize": (14, 10),
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            }
        )

    def _check_factor_mapping(self) -> bool:
        """Check if factor mapping module is available."""
        try:
            from .neuroimaging_utils import add_to_qmap_visualization

            return True
        except ImportError:
            logger.warning("Factor mapping module not available")
            return False

    def create_brain_visualization_summary(
        self, results_dir: str, include_reconstructions: bool = True
    ) -> Dict:
        """
        Create comprehensive brain visualization summary.

        Parameters:
        -----------
        results_dir : str
            Directory containing analysis results
        include_reconstructions : bool
            Whether to include subject reconstructions

        Returns:
        --------
        dict
            Summary of created visualizations
        """
        results_dir = Path(results_dir)
        summary = {
            "brain_plots_created": [],
            "factor_maps": {},
            "reconstruction_summary": {},
            "spatial_analysis": {},
        }

        logger.info(f"Creating brain visualization summary for {results_dir}")

        # Create brain plots directory
        brain_plots_dir = results_dir / "brain_visualizations"
        brain_plots_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Factor loading brain maps
            if self.factor_mapping_available:
                factor_maps = self._create_factor_brain_maps(
                    results_dir, brain_plots_dir
                )
                summary["factor_maps"] = factor_maps

            # 2. Spatial coherence analysis
            spatial_analysis = self._analyze_spatial_coherence(
                results_dir, brain_plots_dir
            )
            summary["spatial_analysis"] = spatial_analysis

            # 3. Region-wise factor analysis
            region_analysis = self._create_region_factor_analysis(
                results_dir, brain_plots_dir
            )
            summary["region_analysis"] = region_analysis

            # 4. Subject reconstructions if requested
            if include_reconstructions:
                reconstruction_summary = self._create_reconstruction_summary(
                    results_dir, brain_plots_dir
                )
                summary["reconstruction_summary"] = reconstruction_summary

            # 5. Create comprehensive brain report
            self._create_brain_analysis_report(summary, brain_plots_dir)

        except Exception as e:
            logger.error(f"Error creating brain visualizations: {e}")
            summary["error"] = str(e)

        return summary

    def _create_factor_brain_maps(self, results_dir: Path, plot_dir: Path) -> Dict:
        """Create brain maps for factor loadings."""
        if not self.factor_mapping_available:
            logger.warning("Factor mapping not available - skipping brain maps")
            return {}

        # Load robust parameters
        try:
            from core.utils import get_model_files, safe_pickle_load

            # Find best run
            best_run = self._find_best_run(results_dir)
            files = get_model_files(results_dir, best_run)
            rob_params = safe_pickle_load(files["robust_params"], "Robust parameters")

            if not rob_params or "W" not in rob_params:
                logger.warning("No factor loadings found for brain mapping")
                return {}

            W = rob_params["W"]
            n_factors = W.shape[1]

            # Create brain maps for top factors
            factor_maps = {}
            n_maps_to_create = min(10, n_factors)  # Create maps for top 10 factors

            for factor_idx in range(n_maps_to_create):
                try:
                    # Create factor-specific brain map
                    factor_loadings = W[:, factor_idx]

                    # Create statistical summary plot
                    self._plot_factor_loading_distribution(
                        factor_loadings, factor_idx, plot_dir
                    )

                    factor_maps[f"factor_{factor_idx + 1}"] = {
                        "mean_loading": float(np.mean(np.abs(factor_loadings))),
                        "max_loading": float(np.max(np.abs(factor_loadings))),
                        "sparsity": float(
                            np.sum(np.abs(factor_loadings) > 0.01)
                            / len(factor_loadings)
                        ),
                    }

                except Exception as e:
                    logger.warning(
                        f"Failed to create brain map for factor {factor_idx + 1}: {e}"
                    )
                    continue

            logger.info(f"Created brain maps for {len(factor_maps)} factors")
            return factor_maps

        except Exception as e:
            logger.error(f"Error in factor brain mapping: {e}")
            return {}

    def _plot_factor_loading_distribution(
        self, loadings: np.ndarray, factor_idx: int, plot_dir: Path
    ):
        """Plot distribution of factor loadings."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Factor {factor_idx + 1} Loading Analysis", fontsize=16, fontweight="bold"
        )

        # Distribution of loadings
        axes[0, 0].hist(
            loadings, bins=50, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 0].axvline(x=0, color="red", linestyle="--", alpha=0.7)
        axes[0, 0].set_title("Loading Distribution")
        axes[0, 0].set_xlabel("Loading Value")
        axes[0, 0].set_ylabel("Frequency")

        # Absolute loadings
        axes[0, 1].hist(
            np.abs(loadings), bins=50, alpha=0.7, color="lightcoral", edgecolor="black"
        )
        axes[0, 1].set_title("Absolute Loading Distribution")
        axes[0, 1].set_xlabel("|Loading Value|")
        axes[0, 1].set_ylabel("Frequency")

        # Top loadings
        top_indices = np.argsort(np.abs(loadings))[-20:]
        top_loadings = loadings[top_indices]

        axes[1, 0].barh(
            range(len(top_loadings)),
            top_loadings,
            color=["red" if x < 0 else "blue" for x in top_loadings],
            alpha=0.7,
        )
        axes[1, 0].set_title("Top 20 Loadings")
        axes[1, 0].set_xlabel("Loading Value")
        axes[1, 0].set_ylabel("Feature Rank")

        # Cumulative explained variance proxy
        sorted_abs_loadings = np.sort(np.abs(loadings))[::-1]
        cumulative_loadings = np.cumsum(sorted_abs_loadings**2)
        cumulative_loadings = cumulative_loadings / cumulative_loadings[-1]

        axes[1, 1].plot(
            range(1, len(cumulative_loadings) + 1),
            cumulative_loadings,
            "b-",
            linewidth=2,
        )
        axes[1, 1].axhline(
            y=0.8, color="red", linestyle="--", alpha=0.7, label="80% threshold"
        )
        axes[1, 1].set_title("Cumulative Loading Contribution")
        axes[1, 1].set_xlabel("Feature Index (sorted by loading magnitude)")
        axes[1, 1].set_ylabel("Cumulative Proportion")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            plot_dir / f"factor_{factor_idx + 1}_loading_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _analyze_spatial_coherence(self, results_dir: Path, plot_dir: Path) -> Dict:
        """Analyze spatial coherence of factors."""
        spatial_analysis = {
            "coherence_metrics": {},
            "region_specificity": {},
            "spatial_patterns": {},
        }

        try:
            # Load data and results
            from core.utils import safe_pickle_load

            # Check if preprocessing results contain spatial information
            prep_path = results_dir / "preprocessing_results.dictionary"
            preprocessing_results = safe_pickle_load(
                prep_path, "Preprocessing results"
            )

            if preprocessing_results and "metadata" in preprocessing_results:
                metadata = preprocessing_results["metadata"]

                if metadata.get("spatial_processing_applied", False):
                    # Create spatial coherence plots
                    self._plot_spatial_processing_summary(metadata, plot_dir)

                    spatial_analysis["spatial_processing"] = {
                        "applied": True,
                        "position_lookups_loaded": metadata.get(
                            "position_lookups_loaded", []
                        ),
                        "harmonization_applied": metadata.get(
                            "harmonization_applied", False
                        ),
                    }

            # If no spatial processing, create basic spatial analysis
            if not spatial_analysis.get("spatial_processing", {}).get(
                "applied", False
            ):
                spatial_analysis = self._basic_spatial_analysis(results_dir, plot_dir)

        except Exception as e:
            logger.warning(f"Spatial coherence analysis failed: {e}")
            spatial_analysis["error"] = str(e)

        return spatial_analysis

    def _plot_spatial_processing_summary(self, metadata: Dict, plot_dir: Path):
        """Plot spatial preprocessing summary."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Spatial Processing Summary", fontsize=16, fontweight="bold")

        # Feature reduction by view
        if "feature_reduction" in metadata:
            views = list(metadata["feature_reduction"].keys())
            original_features = [
                metadata["feature_reduction"][v]["original"] for v in views
            ]
            processed_features = [
                metadata["feature_reduction"][v]["processed"] for v in views
            ]

            x = np.arange(len(views))
            width = 0.35

            axes[0, 0].bar(
                x - width / 2, original_features, width, label="Original", alpha=0.7
            )
            axes[0, 0].bar(
                x + width / 2, processed_features, width, label="Processed", alpha=0.7
            )
            axes[0, 0].set_title("Feature Reduction by View")
            axes[0, 0].set_xlabel("View")
            axes[0, 0].set_ylabel("Number of Features")
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(views, rotation=45)
            axes[0, 0].legend()

        # Reduction ratios
        if "feature_reduction" in metadata:
            reduction_ratios = [
                metadata["feature_reduction"][v]["reduction_ratio"] for v in views
            ]

            axes[0, 1].bar(views, reduction_ratios, alpha=0.7, color="lightcoral")
            axes[0, 1].set_title("Feature Retention Ratios")
            axes[0, 1].set_xlabel("View")
            axes[0, 1].set_ylabel("Retention Ratio")
            axes[0, 1].tick_params(axis="x", rotation=45)

        # Position data availability
        if metadata.get("position_lookups_loaded"):
            position_views = metadata["position_lookups_loaded"]
            axes[1, 0].bar(
                range(len(position_views)),
                [1] * len(position_views),
                alpha=0.7,
                color="lightgreen",
            )
            axes[1, 0].set_title("Views with Spatial Position Data")
            axes[1, 0].set_ylabel("Available")
            axes[1, 0].set_xticks(range(len(position_views)))
            axes[1, 0].set_xticklabels(position_views, rotation=45)

        # Processing summary text
        axes[1, 1].text(
            0.1,
            0.7,
            f"Spatial Processing: {'Applied' if metadata.get('spatial_processing_applied') else 'Not Applied'}",
            fontsize=12,
            fontweight="bold",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].text(
            0.1,
            0.5,
            f"Scanner Harmonization: {'Applied' if metadata.get('harmonization_applied') else 'Not Applied'}",
            fontsize=12,
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].text(
            0.1,
            0.3,
            f"Position Data Views: {len(metadata.get('position_lookups_loaded', []))}",
            fontsize=12,
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis("off")
        axes[1, 1].set_title("Processing Summary")

        plt.tight_layout()
        save_plot(plot_dir / "spatial_processing_summary.png")

    def _basic_spatial_analysis(self, results_dir: Path, plot_dir: Path) -> Dict:
        """Perform basic spatial analysis without full preprocessing."""
        spatial_analysis = {
            "basic_analysis": True,
            "coherence_metrics": {},
            "spatial_processing": {"applied": False},
        }

        # Create placeholder spatial analysis plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "Basic Spatial Analysis\n(Full spatial processing not applied)",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.3,
            "For enhanced spatial analysis, enable spatial processing\nwith --enable_spatial_processing",
            ha="center",
            va="center",
            fontsize=12,
            style="italic",
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Spatial Analysis Summary")

        plt.tight_layout()
        save_plot(plot_dir / "basic_spatial_analysis.png")

        return spatial_analysis

    def _create_region_factor_analysis(
        self, results_dir: Path, plot_dir: Path
    ) -> Dict:
        """Create region-wise factor analysis."""
        region_analysis = {
            "region_loadings": {},
            "factor_specificity": {},
            "cross_region_correlations": {},
        }

        try:
            # This would require region-specific data organization
            # For now, create a placeholder analysis
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "Region-wise Factor Analysis\n(Requires region-specific data organization)",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                transform=ax.transAxes,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.set_title("Region-wise Analysis")

            plt.tight_layout()
            save_plot(plot_dir / "region_factor_analysis.png")

        except Exception as e:
            logger.warning(f"Region factor analysis failed: {e}")
            region_analysis["error"] = str(e)

        return region_analysis

    def _create_reconstruction_summary(
        self, results_dir: Path, plot_dir: Path
    ) -> Dict:
        """Create summary of subject reconstructions."""
        reconstruction_summary = {
            "reconstructions_available": False,
            "n_subjects_reconstructed": 0,
            "reconstruction_quality": {},
        }

        # Check for reconstruction directory
        reconstruction_dir = results_dir / "subject_reconstructions"
        if reconstruction_dir.exists():
            reconstruction_files = list(reconstruction_dir.glob("*.nii*"))
            reconstruction_summary["reconstructions_available"] = True
            reconstruction_summary["n_subjects_reconstructed"] = len(
                reconstruction_files
            )

            # Create reconstruction summary plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(
                "Subject Reconstruction Summary", fontsize=16, fontweight="bold"
            )

            # Number of reconstructions
            axes[0, 0].bar(
                ["Reconstructed"],
                [len(reconstruction_files)],
                alpha=0.7,
                color="lightblue",
            )
            axes[0, 0].set_title("Number of Subject Reconstructions")
            axes[0, 0].set_ylabel("Count")

            # File size distribution (as proxy for data completeness)
            if reconstruction_files:
                file_sizes = [
                    f.stat().st_size / 1024 / 1024 for f in reconstruction_files[:50]
                ]  # MB, limit to 50
                axes[0, 1].hist(file_sizes, bins=20, alpha=0.7, color="lightgreen")
                axes[0, 1].set_title("Reconstruction File Sizes (MB)")
                axes[0, 1].set_xlabel("File Size (MB)")
                axes[0, 1].set_ylabel("Count")

            # Reconstruction availability over time (if timestamps available)
            axes[1, 0].text(
                0.5,
                0.5,
                f"Reconstructions Created\n{len(reconstruction_files)} subjects",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                transform=axes[1, 0].transAxes,
            )
            axes[1, 0].set_xlim(0, 1)
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].axis("off")

            # Summary statistics
            summary_text = f"""Reconstruction Statistics:
Total files: {len(reconstruction_files)}
Directory: {reconstruction_dir.name}
File types: {set(f.suffix for f in reconstruction_files)}
"""
            axes[1, 1].text(
                0.1,
                0.5,
                summary_text,
                fontsize=11,
                transform=axes[1, 1].transAxes,
                verticalalignment="center",
            )
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis("off")
            axes[1, 1].set_title("Summary Statistics")

            plt.tight_layout()
            save_plot(plot_dir / "subject_reconstruction_summary.png")

        return reconstruction_summary

    def _create_brain_analysis_report(self, summary: Dict, plot_dir: Path):
        """Create comprehensive brain analysis report."""
        report_path = plot_dir / "brain_analysis_report.json"

        # Create JSON report
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Create visual summary
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Brain Analysis Summary Report", fontsize=16, fontweight="bold")

        # Factor maps summary
        if summary.get("factor_maps"):
            n_maps = len(summary["factor_maps"])
            axes[0, 0].bar(["Factor Maps"], [n_maps], alpha=0.7, color="lightblue")
            axes[0, 0].set_title("Brain Visualizations Created")
            axes[0, 0].set_ylabel("Count")

        # Spatial analysis status
        spatial_status = (
            "Applied"
            if summary.get("spatial_analysis", {})
            .get("spatial_processing", {})
            .get("applied", False)
            else "Basic"
        )
        status_colors = {"Applied": "green", "Basic": "orange"}
        axes[0, 1].bar(
            ["Spatial Analysis"],
            [1],
            alpha=0.7,
            color=status_colors.get(spatial_status, "gray"),
        )
        axes[0, 1].set_title("Spatial Processing Status")
        axes[0, 1].set_ylabel("Status")
        axes[0, 1].text(
            0, 0.5, spatial_status, ha="center", va="center", fontweight="bold"
        )

        # Reconstruction summary
        n_reconstructions = summary.get("reconstruction_summary", {}).get(
            "n_subjects_reconstructed", 0
        )
        axes[1, 0].bar(
            ["Reconstructions"], [n_reconstructions], alpha=0.7, color="lightgreen"
        )
        axes[1, 0].set_title("Subject Reconstructions")
        axes[1, 0].set_ylabel("Count")

        # Overall summary text
        summary_text = f"""Brain Analysis Summary:
• Factor Maps: {len(summary.get('factor_maps', { }))} created
• Spatial Processing: {spatial_status}
• Subject Reconstructions: {n_reconstructions}
• Analysis Complete: {'Yes' if not summary.get('error') else 'With Errors'}
"""
        axes[1, 1].text(
            0.1,
            0.5,
            summary_text,
            fontsize=11,
            transform=axes[1, 1].transAxes,
            verticalalignment="center",
        )
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis("off")
        axes[1, 1].set_title("Summary")

        plt.tight_layout()
        plt.savefig(
            plot_dir / "brain_analysis_overview.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        logger.info(f"Brain analysis report saved to {report_path}")

    def _find_best_run(self, results_dir: Path) -> int:
        """Find the best run based on results.txt or default to 1."""
        results_file = results_dir / "results.txt"
        if results_file.exists():
            try:
                with open(results_file, "r") as f:
                    for line in f:
                        if line.startswith("Best run:"):
                            return int(line.split(":")[1].strip())
            except BaseException:
                pass
        return 1  # Default to run 1
