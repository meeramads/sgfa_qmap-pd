# visualization/factor_plots.py
"""Factor analysis visualization module."""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy

from core.io_utils import save_plot

logger = logging.getLogger(__name__)


class FactorVisualizer:
    """Creates factor analysis visualizations."""

    def __init__(self, config):
        self.config = config
        self.setup_style()

    def setup_style(self):
        """Setup consistent plotting style."""
        plt.style.use("seaborn-v0_8-darkgrid")
        plt.rcParams.update(
            {
                "figure.figsize": (10, 6),
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.dpi": 100,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
            }
        )

    def create_plots(self, results: Dict, data: Dict, plot_dir: Path):
        """Create all factor analysis plots."""
        logger.info("Creating factor analysis plots")

        # Extract best run
        best_run = self._find_best_run(results)
        if not best_run:
            logger.warning("No converged runs found")
            return

        W = best_run.get("W")
        Z = best_run.get("Z")

        if W is None or Z is None:
            logger.error("Missing W or Z matrices")
            return

        # Create various plots
        self._plot_factor_loadings(W, data, plot_dir / "factors")
        self._plot_factor_scores(Z, data, plot_dir / "factors")
        self._plot_factor_heatmap(W, Z, plot_dir / "factors")
        self._plot_view_contributions(W, data, plot_dir / "factors")

    def _plot_factor_loadings(self, W: np.ndarray, data: Dict, save_dir: Path):
        """Plot factor loadings for each view."""
        view_names = data.get("view_names", [])
        Dm = [X.shape[1] for X in data.get("X_list", [])]

        # Create figure with subplots for each view
        n_views = len(view_names)
        fig, axes = plt.subplots(1, n_views, figsize=(5 * n_views, 6))
        if n_views == 1:
            axes = [axes]

        d = 0
        for m, (view_name, dim) in enumerate(zip(view_names, Dm)):
            W_view = W[d : d + dim, :]

            # Plot heatmap
            im = axes[m].imshow(
                W_view.T,
                aspect="auto",
                cmap="RdBu_r",
                vmin=-np.max(np.abs(W_view)),
                vmax=np.max(np.abs(W_view)),
            )

            axes[m].set_title(f"{view_name}\n({dim} features)")
            axes[m].set_xlabel("Features")
            axes[m].set_ylabel("Factors")
            axes[m].set_yticks(range(W_view.shape[1]))
            axes[m].set_yticklabels([f"F{i + 1}" for i in range(W_view.shape[1])])

            # Add colorbar
            plt.colorbar(im, ax=axes[m], fraction=0.046)

            d += dim

        plt.suptitle("Factor Loadings by View", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save
        save_path = save_dir / "factor_loadings_by_view.png"
        save_plot(save_path)
        plt.close()
        logger.info(f"Saved: {save_path}")

    def _plot_factor_scores(self, Z: np.ndarray, data: Dict, save_dir: Path):
        """Plot factor scores heatmap."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create heatmap
        im = ax.imshow(
            Z.T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-np.max(np.abs(Z)),
            vmax=np.max(np.abs(Z)),
        )

        ax.set_xlabel("Subjects")
        ax.set_ylabel("Factors")
        ax.set_yticks(range(Z.shape[1]))
        ax.set_yticklabels([f"Factor {i + 1}" for i in range(Z.shape[1])])
        ax.set_title("Factor Scores Across Subjects", fontsize=14, fontweight="bold")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label("Factor Score", rotation=270, labelpad=15)

        plt.tight_layout()

        # Save
        save_path = save_dir / "factor_scores.png"
        save_plot(save_path)
        plt.close()
        logger.info(f"Saved: {save_path}")

    def _plot_factor_heatmap(self, W: np.ndarray, Z: np.ndarray, save_dir: Path):
        """Plot comprehensive factor analysis heatmap."""
        # Calculate factor statistics
        self._calculate_factor_statistics(W, Z)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # Ensure axes is always a 2D array
        if axes.ndim == 1:
            axes = axes.reshape(2, 2)

        # 1. Loading magnitude per factor
        loading_mags = np.mean(np.abs(W), axis=0)
        axes[0, 0].bar(range(len(loading_mags)), loading_mags)
        axes[0, 0].set_xlabel("Factor")
        axes[0, 0].set_ylabel("Mean |Loading|")
        axes[0, 0].set_title("Factor Loading Magnitudes")
        axes[0, 0].set_xticks(range(len(loading_mags)))
        axes[0, 0].set_xticklabels([f"F{i + 1}" for i in range(len(loading_mags))])

        # 2. Factor score variance
        score_vars = np.var(Z, axis=0)
        axes[0, 1].bar(range(len(score_vars)), score_vars)
        axes[0, 1].set_xlabel("Factor")
        axes[0, 1].set_ylabel("Variance")
        axes[0, 1].set_title("Factor Score Variance")
        axes[0, 1].set_xticks(range(len(score_vars)))
        axes[0, 1].set_xticklabels([f"F{i + 1}" for i in range(len(score_vars))])

        # 3. Factor correlation matrix
        factor_corr = np.corrcoef(Z.T)
        im = axes[1, 0].imshow(factor_corr, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1, 0].set_title("Factor Correlation Matrix")
        axes[1, 0].set_xlabel("Factor")
        axes[1, 0].set_ylabel("Factor")
        plt.colorbar(im, ax=axes[1, 0])

        # 4. Explained variance (approximation)
        total_var = np.sum(np.var(Z @ W.T, axis=0))
        factor_vars = []
        for k in range(Z.shape[1]):
            Z_k = Z[:, k : k + 1]
            W_k = W[:, k : k + 1]
            factor_var = np.sum(np.var(Z_k @ W_k.T, axis=0))
            factor_vars.append(factor_var / total_var * 100)

        axes[1, 1].bar(range(len(factor_vars)), factor_vars)
        axes[1, 1].set_xlabel("Factor")
        axes[1, 1].set_ylabel("% Variance Explained")
        axes[1, 1].set_title("Approximate Variance Explained")
        axes[1, 1].set_xticks(range(len(factor_vars)))
        axes[1, 1].set_xticklabels([f"F{i + 1}" for i in range(len(factor_vars))])

        plt.suptitle("Factor Analysis Summary", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save
        save_path = save_dir / "factor_summary.png"
        save_plot(save_path)
        plt.close()
        logger.info(f"Saved: {save_path}")

    def _plot_view_contributions(self, W: np.ndarray, data: Dict, save_dir: Path):
        """Plot contribution of each view to factors."""
        view_names = data.get("view_names", [])
        Dm = [X.shape[1] for X in data.get("X_list", [])]
        n_factors = W.shape[1]

        # Calculate view contributions
        view_contributions = np.zeros((len(view_names), n_factors))

        d = 0
        for m, dim in enumerate(Dm):
            W_view = W[d : d + dim, :]
            # Use mean absolute loading as contribution measure
            view_contributions[m, :] = np.mean(np.abs(W_view), axis=0)
            d += dim

        # Create stacked bar plot
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(n_factors)
        width = 0.8

        # Stack bars for each view
        bottom = np.zeros(n_factors)
        colors = plt.cm.Set3(np.linspace(0, 1, len(view_names)))

        for m, view_name in enumerate(view_names):
            ax.bar(
                x,
                view_contributions[m, :],
                width,
                bottom=bottom,
                label=view_name,
                color=colors[m],
            )
            bottom += view_contributions[m, :]

        ax.set_xlabel("Factor")
        ax.set_ylabel("Mean |Loading|")
        ax.set_title(
            "View Contributions to Each Factor", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{i + 1}" for i in range(n_factors)])
        ax.legend()

        plt.tight_layout()

        # Save
        save_path = save_dir / "view_contributions.png"
        save_plot(save_path)
        plt.close()
        logger.info(f"Saved: {save_path}")

    def _find_best_run(self, results: Dict) -> Optional[Dict]:
        """Find the best run from results (modernized to handle various formats)."""
        # Handle modern experiment results format
        if not results:
            logger.warning("No results provided to visualization")
            return None

        # Check if results is already a direct result (single model case)
        if "W" in results and "Z" in results:
            logger.info("Found direct W/Z matrices in results")
            return results

        # Check for modern experiment format with model results
        for key, value in results.items():
            if isinstance(value, dict):
                # Look for W and Z matrices directly
                if "W" in value and "Z" in value:
                    logger.info(f"Found W/Z matrices in results['{key}']")
                    return value

                # Look for robust parameters (legacy format)
                if "robust" in value and isinstance(value["robust"], dict):
                    robust = value["robust"]
                    if "W" in robust and "Z" in robust:
                        logger.info(f"Found W/Z matrices in results['{key}']['robust']")
                        return robust

                # Look for nested model results
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict) and "W" in subvalue and "Z" in subvalue:
                        logger.info(f"Found W/Z matrices in results['{key}']['{subkey}']")
                        return subvalue

        logger.warning("No W/Z matrices found in any result format")
        return None

    def _calculate_factor_statistics(self, W: np.ndarray, Z: np.ndarray) -> Dict:
        """Calculate various factor statistics."""
        return {
            "n_factors": W.shape[1],
            "n_features": W.shape[0],
            "n_subjects": Z.shape[0],
            "loading_sparsity": np.mean(np.abs(W) < 0.01),
            "score_kurtosis": np.mean(
                [np.abs(scipy.stats.kurtosis(Z[:, k])) for k in range(Z.shape[1])]
            ),
            "mean_factor_correlation": np.mean(
                np.abs(np.corrcoef(Z.T)[np.triu_indices(Z.shape[1], k=1)])
            ),
        }

    # === MISSING METHODS REQUIRED BY CORE.VISUALIZATION ===

    def plot_factor_loadings(self, W, title="Factor Loadings", save_path=None):
        """Plot factor loadings matrix as heatmap."""
        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(W.T, cmap='RdBu_r', aspect='auto')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Features')
        ax.set_ylabel('Factors')

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Loading Value')

        plt.tight_layout()

        if save_path:
            save_plot(save_path)
            plt.close()
            logger.info(f"Saved factor loadings plot: {save_path}")
        else:
            plt.show()

    def plot_factor_scores(self, Z, subject_ids=None, title="Factor Scores", save_path=None):
        """Plot factor scores for subjects."""
        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(Z.T, cmap='RdBu_r', aspect='auto')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Subjects')
        ax.set_ylabel('Factors')

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Score Value')

        plt.tight_layout()

        if save_path:
            save_plot(save_path)
            plt.close()
            logger.info(f"Saved factor scores plot: {save_path}")
        else:
            plt.show()

    def plot_factor_comparison(self, W_true, W_est, Z_true, Z_est, save_path=None):
        """Compare true vs estimated factors."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        # Ensure axes is always a 2D array
        if axes.ndim == 1:
            axes = axes.reshape(2, 2)

        # True vs estimated loadings
        im1 = axes[0, 0].imshow(W_true.T, cmap='RdBu_r', aspect='auto')
        axes[0, 0].set_title('True Factor Loadings')
        axes[0, 0].set_xlabel('Features')
        axes[0, 0].set_ylabel('Factors')
        plt.colorbar(im1, ax=axes[0, 0])

        im2 = axes[0, 1].imshow(W_est.T, cmap='RdBu_r', aspect='auto')
        axes[0, 1].set_title('Estimated Factor Loadings')
        axes[0, 1].set_xlabel('Features')
        axes[0, 1].set_ylabel('Factors')
        plt.colorbar(im2, ax=axes[0, 1])

        # True vs estimated scores
        im3 = axes[1, 0].imshow(Z_true.T, cmap='RdBu_r', aspect='auto')
        axes[1, 0].set_title('True Factor Scores')
        axes[1, 0].set_xlabel('Subjects')
        axes[1, 0].set_ylabel('Factors')
        plt.colorbar(im3, ax=axes[1, 0])

        im4 = axes[1, 1].imshow(Z_est.T, cmap='RdBu_r', aspect='auto')
        axes[1, 1].set_title('Estimated Factor Scores')
        axes[1, 1].set_xlabel('Subjects')
        axes[1, 1].set_ylabel('Factors')
        plt.colorbar(im4, ax=axes[1, 1])

        plt.suptitle('Factor Comparison: True vs Estimated', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            save_plot(save_path)
            plt.close()
            logger.info(f"Saved factor comparison plot: {save_path}")
        else:
            plt.show()

    def plot_multiview_loadings(self, W, Dm, view_names, feat_names, topk, save_path=None):
        """Plot multi-view factor loadings."""
        n_factors = W.shape[1]
        n_views = len(Dm)

        fig, axes = plt.subplots(n_factors, n_views, figsize=(4*n_views, 3*n_factors))
        if n_factors == 1:
            axes = axes.reshape(1, -1)
        if n_views == 1:
            axes = axes.reshape(-1, 1)

        # Split loadings by view
        view_start = 0
        for view_idx, view_dim in enumerate(Dm):
            view_end = view_start + view_dim
            W_view = W[view_start:view_end, :]

            for factor_idx in range(n_factors):
                ax = axes[factor_idx, view_idx]

                # Get top k loadings for this factor and view
                loadings = W_view[:, factor_idx]
                top_indices = np.argsort(np.abs(loadings))[-topk:]
                top_loadings = loadings[top_indices]

                # Create feature names for this view
                if feat_names and view_names[view_idx] in feat_names:
                    feature_labels = [f"{feat_names[view_names[view_idx]][i]}" for i in top_indices]
                else:
                    feature_labels = [f"F{i}" for i in top_indices]

                # Bar plot
                colors = ['red' if x < 0 else 'blue' for x in top_loadings]
                bars = ax.barh(range(len(top_loadings)), top_loadings, color=colors, alpha=0.7)

                ax.set_yticks(range(len(feature_labels)))
                ax.set_yticklabels(feature_labels, fontsize=8)
                ax.set_xlabel('Loading Value')
                ax.set_title(f'{view_names[view_idx]}\nFactor {factor_idx+1}', fontsize=10)
                ax.grid(True, alpha=0.3)

            view_start = view_end

        plt.suptitle(f'Multi-View Factor Loadings (Top {topk} per factor)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            save_plot(save_path)
            plt.close()
            logger.info(f"Saved multi-view loadings plot: {save_path}")
        else:
            plt.show()

    def plot_factor_summary(self, W, Z, Dm, view_names, save_path=None):
        """Plot comprehensive factor summary."""
        n_factors = W.shape[1]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        # Ensure axes is always a 2D array
        if axes.ndim == 1:
            axes = axes.reshape(2, 2)

        # 1. Factor loading magnitudes by view
        view_start = 0
        view_contributions = []
        for view_idx, view_dim in enumerate(Dm):
            view_end = view_start + view_dim
            W_view = W[view_start:view_end, :]
            view_contrib = np.mean(np.abs(W_view), axis=0)
            view_contributions.append(view_contrib)
            view_start = view_end

        view_contributions = np.array(view_contributions)

        x = np.arange(n_factors)
        width = 0.8 / len(view_names)

        for i, view_name in enumerate(view_names):
            axes[0, 0].bar(x + i*width, view_contributions[i], width,
                          label=view_name, alpha=0.7)

        axes[0, 0].set_xlabel('Factor')
        axes[0, 0].set_ylabel('Mean |Loading|')
        axes[0, 0].set_title('Factor Contributions by View')
        axes[0, 0].set_xticks(x + width * (len(view_names)-1) / 2)
        axes[0, 0].set_xticklabels([f'F{i+1}' for i in range(n_factors)])
        axes[0, 0].legend()

        # 2. Factor score distributions
        for i in range(min(n_factors, 4)):  # Show up to 4 factors
            axes[0, 1].hist(Z[:, i], bins=20, alpha=0.6, label=f'Factor {i+1}')
        axes[0, 1].set_xlabel('Score Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Factor Score Distributions')
        axes[0, 1].legend()

        # 3. Factor correlation matrix
        if n_factors > 1:
            factor_corr = np.corrcoef(Z.T)
            im = axes[1, 0].imshow(factor_corr, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1, 0].set_title('Inter-Factor Correlations')
            axes[1, 0].set_xlabel('Factor')
            axes[1, 0].set_ylabel('Factor')
            plt.colorbar(im, ax=axes[1, 0])
        else:
            axes[1, 0].text(0.5, 0.5, 'Single Factor\n(No Correlations)',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Inter-Factor Correlations')

        # 4. Explained variance approximation
        total_var = np.sum(np.var(Z @ W.T, axis=0))
        if total_var > 0:
            factor_vars = []
            for k in range(n_factors):
                Z_k = Z[:, k:k+1]
                W_k = W[:, k:k+1]
                factor_var = np.sum(np.var(Z_k @ W_k.T, axis=0))
                factor_vars.append(factor_var / total_var * 100)

            axes[1, 1].bar(range(n_factors), factor_vars, alpha=0.7)
            axes[1, 1].set_xlabel('Factor')
            axes[1, 1].set_ylabel('% Variance Explained')
            axes[1, 1].set_title('Approximate Variance Explained')
            axes[1, 1].set_xticks(range(n_factors))
            axes[1, 1].set_xticklabels([f'F{i+1}' for i in range(n_factors)])
        else:
            axes[1, 1].text(0.5, 0.5, 'Cannot compute\nvariance explained',
                           ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.suptitle('Factor Analysis Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            save_plot(save_path)
            plt.close()
            logger.info(f"Saved factor summary plot: {save_path}")
        else:
            plt.show()
