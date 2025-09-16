"""
Cross-validation visualization module.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.io_utils import save_plot

logger = logging.getLogger(__name__)


class CrossValidationVisualizer:
    """Creates cross-validation analysis visualizations."""

    def __init__(self, config):
        self.config = config
        self.setup_style()

    def setup_style(self):
        """Setup consistent plotting style."""
        plt.style.use("seaborn-v0_8-darkgrid")
        plt.rcParams.update(
            {
                "figure.figsize": (12, 8),
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            }
        )

    def plot_cv_results(
        self, cv_results: Dict, plot_path: str, run_name: str = "cross_validation"
    ):
        """
        Create comprehensive cross-validation results visualizations.

        Parameters:
        -----------
        cv_results : dict
            Cross-validation results dictionary
        plot_path : str
            Directory path for saving plots
        run_name : str
            Name for the analysis run
        """
        plot_path = Path(plot_path)
        plot_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating CV visualizations for {run_name}")

        # 1. Overall performance metrics
        self._plot_cv_performance_summary(cv_results, plot_path, run_name)

        # 2. Factor stability analysis
        if "fold_results" in cv_results:
            self._plot_factor_stability_heatmap(cv_results["fold_results"], plot_path)

        # 3. Hyperparameter optimization results
        if "hyperparameter_optimization" in cv_results:
            self._plot_hyperparameter_optimization(
                cv_results["hyperparameter_optimization"], plot_path
            )

        # 4. Neuroimaging-specific plots
        if "spatial_coherence_scores" in cv_results:
            self._plot_spatial_coherence(cv_results, plot_path)

        # 5. Subtype analysis if available
        if "subtype_validation" in cv_results:
            self._plot_subtype_validation(cv_results["subtype_validation"], plot_path)

        logger.info(f"CV visualizations saved to {plot_path}")

    def _plot_cv_performance_summary(
        self, cv_results: Dict, plot_path: Path, run_name: str
    ):
        """Plot overall CV performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Cross-Validation Performance Summary: {run_name}",
            fontsize=16,
            fontweight="bold",
        )

        # Performance metrics across folds
        if "fold_scores" in cv_results:
            fold_scores = cv_results["fold_scores"]

            # Box plot of fold scores
            axes[0, 0].boxplot(fold_scores, patch_artist=True)
            axes[0, 0].set_title("Performance Across Folds")
            axes[0, 0].set_ylabel("Score")
            axes[0, 0].set_xlabel("Folds")

            # Performance trend
            axes[0, 1].plot(
                range(1, len(fold_scores) + 1),
                fold_scores,
                "bo-",
                linewidth=2,
                markersize=6,
            )
            axes[0, 1].axhline(
                y=np.mean(fold_scores),
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Mean: {np.mean(fold_scores):.4f}",
            )
            axes[0, 1].fill_between(
                range(1, len(fold_scores) + 1),
                np.mean(fold_scores) - np.std(fold_scores),
                np.mean(fold_scores) + np.std(fold_scores),
                alpha=0.2,
                color="red",
                label=f"±1 SD: {np.std(fold_scores):.4f}",
            )
            axes[0, 1].set_title("Performance Trend Across Folds")
            axes[0, 1].set_xlabel("Fold Number")
            axes[0, 1].set_ylabel("Score")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Model complexity analysis
        if "model_complexity" in cv_results:
            complexity = cv_results["model_complexity"]
            axes[1, 0].bar(
                range(len(complexity)), complexity, alpha=0.7, color="skyblue"
            )
            axes[1, 0].set_title("Model Complexity per Fold")
            axes[1, 0].set_xlabel("Fold Number")
            axes[1, 0].set_ylabel("Effective Number of Parameters")

        # Training vs validation performance
        if "train_scores" in cv_results and "val_scores" in cv_results:
            folds = range(1, len(cv_results["train_scores"]) + 1)
            axes[1, 1].plot(
                folds, cv_results["train_scores"], "go-", label="Training", linewidth=2
            )
            axes[1, 1].plot(
                folds, cv_results["val_scores"], "ro-", label="Validation", linewidth=2
            )
            axes[1, 1].set_title("Training vs Validation Performance")
            axes[1, 1].set_xlabel("Fold Number")
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_plot(plot_path / f"{run_name}_cv_performance_summary.png")

    def _plot_factor_stability_heatmap(self, fold_results: Dict, plot_path: Path):
        """Plot factor stability across CV folds."""
        if not fold_results:
            logger.warning("No fold results available for stability analysis")
            return

        # Extract stability metrics
        n_folds = len(fold_results)
        n_factors = max(
            [len(fold_results[f].get("factor_loadings", [])) for f in fold_results]
        )

        # Create stability matrix
        stability_matrix = np.zeros((n_factors, n_folds))

        for fold_idx, (fold_name, fold_data) in enumerate(fold_results.items()):
            if "factor_stability" in fold_data:
                stability_scores = fold_data["factor_stability"]
                for factor_idx, stability in enumerate(stability_scores[:n_factors]):
                    stability_matrix[factor_idx, fold_idx] = stability

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            stability_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlBu_r",
            xticklabels=[f"Fold {i + 1}" for i in range(n_folds)],
            yticklabels=[f"Factor {i + 1}" for i in range(n_factors)],
            cbar_kws={"label": "Stability Score"},
        )

        plt.title("Factor Stability Across CV Folds", fontsize=14, fontweight="bold")
        plt.xlabel("CV Fold")
        plt.ylabel("Factor Index")
        plt.tight_layout()
        save_plot(plot_path / "factor_stability_heatmap.png")

    def _plot_hyperparameter_optimization(self, hp_results: Dict, plot_path: Path):
        """Plot hyperparameter optimization results."""
        if "search_results" not in hp_results:
            return

        search_results = hp_results["search_results"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Hyperparameter Optimization Results", fontsize=16, fontweight="bold"
        )

        # Parameter importance
        if "param_importance" in hp_results:
            params = list(hp_results["param_importance"].keys())
            importance = list(hp_results["param_importance"].values())

            axes[0, 0].barh(params, importance, alpha=0.7, color="lightcoral")
            axes[0, 0].set_title("Parameter Importance")
            axes[0, 0].set_xlabel("Importance Score")

        # Best parameters visualization
        if "best_params" in hp_results:
            best_params = hp_results["best_params"]
            param_names = list(best_params.keys())
            param_values = list(best_params.values())

            axes[0, 1].bar(param_names, param_values, alpha=0.7, color="lightgreen")
            axes[0, 1].set_title("Best Parameters")
            axes[0, 1].set_ylabel("Parameter Value")
            axes[0, 1].tick_params(axis="x", rotation=45)

        # Optimization convergence
        if "convergence_history" in hp_results:
            history = hp_results["convergence_history"]
            axes[1, 0].plot(history, "b-", linewidth=2)
            axes[1, 0].set_title("Optimization Convergence")
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].set_ylabel("Best Score")
            axes[1, 0].grid(True, alpha=0.3)

        # Parameter correlation matrix if available
        if len(search_results) > 10:  # Only if we have enough samples
            param_df = pd.DataFrame(search_results)
            if len(param_df.columns) > 1:
                corr_matrix = param_df.corr()
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    center=0,
                    ax=axes[1, 1],
                )
                axes[1, 1].set_title("Parameter Correlation Matrix")

        plt.tight_layout()
        save_plot(plot_path / "hyperparameter_optimization.png")

    def _plot_spatial_coherence(self, cv_results: Dict, plot_path: Path):
        """Plot neuroimaging spatial coherence results."""
        spatial_scores = cv_results["spatial_coherence_scores"]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Spatial Coherence Analysis", fontsize=16, fontweight="bold")

        # Overall spatial coherence
        if "mean" in spatial_scores:
            coherence_values = [spatial_scores["mean"]]
            if "std" in spatial_scores:
                err_values = [spatial_scores["std"]]
            else:
                err_values = [0]

            axes[0].bar(
                ["Overall"],
                coherence_values,
                yerr=err_values,
                capsize=10,
                alpha=0.7,
                color="lightblue",
            )
            axes[0].set_title("Overall Spatial Coherence")
            axes[0].set_ylabel("Coherence Score")

        # Per-region coherence if available
        if "region_scores" in spatial_scores:
            region_scores = spatial_scores["region_scores"]
            regions = list(region_scores.keys())
            scores = list(region_scores.values())

            axes[1].barh(regions, scores, alpha=0.7, color="lightcoral")
            axes[1].set_title("Spatial Coherence by Brain Region")
            axes[1].set_xlabel("Coherence Score")

        plt.tight_layout()
        save_plot(plot_path / "spatial_coherence_analysis.png")

    def _plot_subtype_validation(self, subtype_results: Dict, plot_path: Path):
        """Plot subtype validation results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Subtype Validation Results", fontsize=16, fontweight="bold")

        # Silhouette scores
        if "silhouette_scores" in subtype_results:
            silhouette_data = subtype_results["silhouette_scores"]
            if isinstance(silhouette_data, dict) and "mean" in silhouette_data:
                axes[0, 0].bar(
                    ["Mean Silhouette"],
                    [silhouette_data["mean"]],
                    alpha=0.7,
                    color="gold",
                )
                axes[0, 0].set_title("Silhouette Score")
                axes[0, 0].set_ylabel("Score")

        # Subtype stability
        if "subtype_stability" in subtype_results:
            stability_scores = subtype_results["subtype_stability"]
            n_subtypes = len(stability_scores)

            axes[0, 1].bar(
                range(1, n_subtypes + 1),
                stability_scores,
                alpha=0.7,
                color="lightgreen",
            )
            axes[0, 1].set_title("Subtype Stability")
            axes[0, 1].set_xlabel("Subtype")
            axes[0, 1].set_ylabel("Stability Score")

        # Clinical associations if available
        if "clinical_associations" in subtype_results:
            clinical_data = subtype_results["clinical_associations"]
            if isinstance(clinical_data, dict):
                variables = list(clinical_data.keys())
                p_values = [
                    -np.log10(clinical_data[var].get("p_value", 1.0))
                    for var in variables
                ]

                axes[1, 0].barh(variables, p_values, alpha=0.7, color="lightcoral")
                axes[1, 0].axvline(
                    x=-np.log10(0.05),
                    color="red",
                    linestyle="--",
                    label="p=0.05 threshold",
                )
                axes[1, 0].set_title("Clinical Variable Associations")
                axes[1, 0].set_xlabel("-log10(p-value)")
                axes[1, 0].legend()

        # Subtype sizes
        if "subtype_sizes" in subtype_results:
            sizes = subtype_results["subtype_sizes"]
            subtypes = [f"Subtype {i + 1}" for i in range(len(sizes))]

            axes[1, 1].pie(sizes, labels=subtypes, autopct="%1.1f%%", startangle=90)
            axes[1, 1].set_title("Subtype Distribution")

        plt.tight_layout()
        save_plot(plot_path / "subtype_validation.png")

    def plot_consensus_subtypes(
        self, centroids_data: Dict, probabilities_data: Dict, plot_path: str
    ):
        """
        Plot consensus subtype analysis results.

        Parameters:
        -----------
        centroids_data : dict
            Subtype centroids data
        probabilities_data : dict
            Subject assignment probabilities
        plot_path : str
            Directory path for saving plots
        """
        plot_path = Path(plot_path)
        plot_path.mkdir(parents=True, exist_ok=True)

        # Plot centroids heatmap
        if "centroids" in centroids_data:
            self._plot_centroids_heatmap(centroids_data["centroids"], plot_path)

        # Plot centroids radar chart
        if "centroids" in centroids_data:
            self._plot_centroids_radar(centroids_data["centroids"], plot_path)

        # Plot assignment probabilities
        if "probabilities" in probabilities_data:
            self._plot_subtype_probabilities(probabilities_data, plot_path)
            self._plot_subtype_assignments(probabilities_data, plot_path)

    def _plot_centroids_heatmap(self, C: np.ndarray, plot_path: Path):
        """Plot subtype centroids as heatmap."""
        plt.figure(figsize=(12, 8))

        # Create heatmap
        sns.heatmap(
            C.T,
            annot=False,
            cmap="RdBu_r",
            center=0,
            xticklabels=[f"Subtype {i + 1}" for i in range(C.shape[0])],
            cbar_kws={"label": "Factor Loading"},
        )

        plt.title("Subtype Centroids Heatmap", fontsize=14, fontweight="bold")
        plt.xlabel("Subtype")
        plt.ylabel("Factor")
        plt.tight_layout()
        save_plot(plot_path / "subtype_centroids_heatmap.png")

    def _plot_centroids_radar(self, C: np.ndarray, plot_path: Path):
        """Plot subtype centroids as radar chart."""
        n_subtypes, n_factors = C.shape

        if n_factors < 3:
            logger.warning("Insufficient factors for radar plot")
            return

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, n_factors, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        colors = plt.cm.Set3(np.linspace(0, 1, n_subtypes))

        for i in range(n_subtypes):
            values = C[i, :].tolist()
            values += values[:1]  # Complete the circle

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=f"Subtype {i + 1}",
                color=colors[i],
            )
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f"Factor {i + 1}" for i in range(n_factors)])
        ax.set_title(
            "Subtype Centroids Radar Chart", size=14, fontweight="bold", y=1.08
        )
        ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

        plt.tight_layout()
        save_plot(plot_path / "subtype_centroids_radar.png")

    def _plot_subtype_probabilities(self, prob_data: Dict, plot_path: Path):
        """Plot subtype assignment probabilities."""
        if "probabilities" not in prob_data:
            return

        prob_df = pd.DataFrame(prob_data["probabilities"])
        n_subtypes = prob_df.shape[1]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Subtype Assignment Analysis", fontsize=16, fontweight="bold")

        # Distribution of assignment probabilities
        for i in range(n_subtypes):
            axes[0, 0].hist(
                prob_df.iloc[:, i],
                bins=30,
                alpha=0.7,
                label=f"Subtype {i + 1}",
                density=True,
            )

        axes[0, 0].set_title("Assignment Probability Distributions")
        axes[0, 0].set_xlabel("Probability")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].legend()

        # Maximum probability distribution
        max_probs = prob_df.max(axis=1)
        axes[0, 1].hist(max_probs, bins=30, alpha=0.7, color="gold")
        axes[0, 1].axvline(
            x=0.5, color="red", linestyle="--", label="Uncertain threshold"
        )
        axes[0, 1].set_title("Maximum Assignment Probabilities")
        axes[0, 1].set_xlabel("Maximum Probability")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].legend()

        # Subtype assignments
        assignments = prob_df.idxmax(axis=1)
        assignment_counts = assignments.value_counts().sort_index()

        axes[1, 0].bar(
            range(len(assignment_counts)),
            assignment_counts.values,
            alpha=0.7,
            color="lightcoral",
        )
        axes[1, 0].set_title("Subtype Assignment Counts")
        axes[1, 0].set_xlabel("Subtype")
        axes[1, 0].set_ylabel("Number of Subjects")
        axes[1, 0].set_xticks(range(len(assignment_counts)))
        axes[1, 0].set_xticklabels(
            [f"Subtype {i + 1}" for i in assignment_counts.index]
        )

        # Assignment certainty
        certainty = 1 - (
            prob_df.apply(lambda row: np.sum(np.sort(row)[-2:]), axis=1)
            - prob_df.max(axis=1)
        )
        axes[1, 1].hist(certainty, bins=30, alpha=0.7, color="lightgreen")
        axes[1, 1].set_title("Assignment Certainty")
        axes[1, 1].set_xlabel("Certainty Score")
        axes[1, 1].set_ylabel("Count")

        plt.tight_layout()
        save_plot(plot_path / "subtype_probabilities_analysis.png")

    def _plot_subtype_assignments(self, prob_data: Dict, plot_path: Path):
        """Plot final subtype assignments."""
        if "probabilities" not in prob_data:
            return

        prob_df = pd.DataFrame(prob_data["probabilities"])
        assignments = prob_df.idxmax(axis=1)
        max_probs = prob_df.max(axis=1)

        plt.figure(figsize=(12, 8))

        # Create scatter plot of assignments vs certainty
        colors = plt.cm.Set3(np.linspace(0, 1, prob_df.shape[1]))

        for subtype in range(prob_df.shape[1]):
            mask = assignments == subtype
            plt.scatter(
                np.where(mask)[0],
                max_probs[mask],
                c=[colors[subtype]],
                label=f"Subtype {subtype + 1}",
                alpha=0.6,
                s=50,
            )

        plt.axhline(
            y=0.5,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Uncertainty threshold",
        )
        plt.title("Subtype Assignments by Certainty", fontsize=14, fontweight="bold")
        plt.xlabel("Subject Index")
        plt.ylabel("Assignment Probability")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        save_plot(plot_path / "subtype_assignments_scatter.png")
