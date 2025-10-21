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
        self.enable_spatial_analysis = getattr(config, 'enable_spatial_analysis', True)
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

        # Create enhanced visualizations with proper labeling
        logger.info("Creating enhanced factor loading distribution plots")
        self.plot_enhanced_factor_loading_distributions(
            W, data, save_path=str(plot_dir / "factors" / "enhanced_loading_distributions.png")
        )

        # Create clinical-specific visualization if clinical data present
        view_names = data.get("view_names", [])
        has_clinical_data = any('clinical' in view_name.lower() for view_name in view_names)

        if has_clinical_data:
            logger.info("Creating detailed clinical factor loading analysis")
            self.plot_clinical_factor_loadings(
                W, data, save_path=str(plot_dir / "factors" / "clinical_factor_loadings.png")
            )

        # Create region-wise analysis for brain data (if brain regions detected)
        brain_regions = ['lentiform', 'sn', 'putamen', 'caudate', 'thalamus', 'hippocampus',
                        'amygdala', 'cortical', 'subcortical', 'frontal', 'parietal', 'temporal',
                        'occipital', 'cerebellum', 'brainstem', 'roi', 'region']
        view_names = data.get("view_names", [])

        has_brain_data = any(any(region in view_name.lower() for region in brain_regions)
                            for view_name in view_names)

        if has_brain_data:
            logger.info("Creating region-wise factor analysis")
            self.plot_region_wise_factor_analysis(
                W, data, save_path=str(plot_dir / "factors" / "region_wise_analysis.png")
            )

            # Create brain-specific visualizations in brain folder
            brain_plot_dir = plot_dir / "brain_analysis"
            brain_plot_dir.mkdir(exist_ok=True)

            logger.info("Creating interpretable brain loading visualizations")
            self.plot_interpretable_brain_loadings(
                W, data, save_path=str(brain_plot_dir / "brain_factor_loadings.png")
            )

    def _plot_factor_loadings(self, W: np.ndarray, data: Dict, save_dir: Path):
        """Plot factor loadings for each view."""
        view_names = data.get("view_names", [])
        feature_names = data.get("feature_names", {})
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
            axes[m].set_ylabel("Factors")
            axes[m].set_yticks(range(W_view.shape[1]))
            axes[m].set_yticklabels([f"F{i + 1}" for i in range(W_view.shape[1])])

            # Add feature names on x-axis if available
            if view_name in feature_names and len(feature_names[view_name]) == dim:
                feat_labels = feature_names[view_name]
                # Show labels if not too many features
                if dim <= 30:
                    axes[m].set_xticks(range(dim))
                    axes[m].set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=8)
                    axes[m].set_xlabel("")
                else:
                    # Too many features - just show count
                    axes[m].set_xlabel("Features")
            else:
                axes[m].set_xlabel("Features")

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
        ax.set_title("Factor Scores Across Subjects", fontsize=10)

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
        axes[0, 0].set_title("Factor Loading Magnitude Distributions")
        axes[0, 0].set_xticks(range(len(loading_mags)))
        axes[0, 0].set_xticklabels([f"F{i + 1}" for i in range(len(loading_mags))])

        # 2. Factor score variance
        score_vars = np.var(Z, axis=0)
        axes[0, 1].bar(range(len(score_vars)), score_vars)
        axes[0, 1].set_xlabel("Factor")
        axes[0, 1].set_ylabel("Variance")
        axes[0, 1].set_title("Inter-Subject Factor Score Variance")
        axes[0, 1].set_xticks(range(len(score_vars)))
        axes[0, 1].set_xticklabels([f"F{i + 1}" for i in range(len(score_vars))])

        # 3. Factor correlation matrix
        factor_corr = np.corrcoef(Z.T)
        im = axes[1, 0].imshow(factor_corr, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1, 0].set_title("Inter-Factor Correlation Matrix")
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
        axes[1, 1].set_title("Data Variance Explained by Factor")
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

        ax.set_xlabel("Latent Factor")
        ax.set_ylabel("Mean Absolute Loading")
        ax.set_title(
            "View Contributions to Each Factor", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{i + 1}" for i in range(n_factors)])
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9, ncol=1)

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
        ax.set_title(title, fontsize=10)
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
        ax.set_title(title, fontsize=10)
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
        axes[0, 0].set_xlabel('Feature Index')
        axes[0, 0].set_ylabel('Latent Factor (F1-FK)')
        plt.colorbar(im1, ax=axes[0, 0])

        im2 = axes[0, 1].imshow(W_est.T, cmap='RdBu_r', aspect='auto')
        axes[0, 1].set_title('Estimated Factor Loadings')
        axes[0, 1].set_xlabel('Feature Index')
        axes[0, 1].set_ylabel('Latent Factor (F1-FK)')
        plt.colorbar(im2, ax=axes[0, 1])

        # True vs estimated scores
        im3 = axes[1, 0].imshow(Z_true.T, cmap='RdBu_r', aspect='auto')
        axes[1, 0].set_title('True Factor Scores')
        axes[1, 0].set_xlabel('Subject ID')
        axes[1, 0].set_ylabel('Latent Factor (F1-FK)')
        plt.colorbar(im3, ax=axes[1, 0])

        im4 = axes[1, 1].imshow(Z_est.T, cmap='RdBu_r', aspect='auto')
        axes[1, 1].set_title('Estimated Factor Scores')
        axes[1, 1].set_xlabel('Subject ID')
        axes[1, 1].set_ylabel('Latent Factor (F1-FK)')
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
                    try:
                        view_feat_names = feat_names[view_names[view_idx]]
                        feature_labels = [view_feat_names[i] if i < len(view_feat_names) else f"feature_{i}" for i in top_indices]
                    except (IndexError, KeyError) as e:
                        logger.warning(f"Could not use feature names for view {view_names[view_idx]}: {e}")
                        feature_labels = [f"{view_names[view_idx]}_feature_{i}" for i in top_indices]
                else:
                    # Create meaningful labels based on view name and feature index
                    view_name = view_names[view_idx]

                    # For neuroimaging data, create more descriptive labels
                    if any(region in view_name.lower() for region in ['sn', 'substantia', 'putamen', 'lentiform']):
                        # Brain region names
                        brain_region_map = {
                            'sn': 'SubstantiaNigra',
                            'substantia': 'SubstantiaNigra',
                            'putamen': 'Putamen',
                            'lentiform': 'Lentiform'
                        }
                        region_name = next((brain_region_map[k] for k in brain_region_map.keys()
                                          if k in view_name.lower()), view_name)
                        feature_labels = [f"{region_name}_voxel_{i}" for i in top_indices]
                    elif 'clinical' in view_name.lower():
                        # Use feature names from data if available
                        if feat_names and view_names[view_idx] in feat_names:
                            try:
                                view_feat_names = feat_names[view_names[view_idx]]
                                feature_labels = [view_feat_names[i] if i < len(view_feat_names) else f"Clinical_var_{i}" for i in top_indices]
                            except (IndexError, KeyError) as e:
                                logger.warning(f"Could not use feature names for clinical view: {e}")
                                feature_labels = [f"Clinical_var_{i}" for i in top_indices]
                        else:
                            feature_labels = [f"Clinical_var_{i}" for i in top_indices]
                    else:
                        feature_labels = [f"{view_name}_feature_{i}" for i in top_indices]

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

        axes[0, 0].set_xlabel('Latent Factor')
        axes[0, 0].set_ylabel('Mean Absolute Loading')
        axes[0, 0].set_title(f'Factor Contributions by View (K={n_factors})')
        axes[0, 0].set_xticks(x + width * (len(view_names)-1) / 2)
        axes[0, 0].set_xticklabels([f'F{i+1}' for i in range(n_factors)])
        axes[0, 0].legend()

        # 2. Factor score distributions
        for i in range(n_factors):  # Show all factors consistently
            axes[0, 1].hist(Z[:, i], bins=20, alpha=0.6, label=f'Factor {i+1}')
        axes[0, 1].set_xlabel('Factor Score Value')
        axes[0, 1].set_ylabel('Subject Frequency')
        axes[0, 1].set_title(f'Factor Score Distributions (K={n_factors})')
        axes[0, 1].legend()

        # 3. Factor correlation matrix
        if n_factors > 1:
            factor_corr = np.corrcoef(Z.T)
            im = axes[1, 0].imshow(factor_corr, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1, 0].set_title(f'Inter-Factor Correlations (K={n_factors})')
            axes[1, 0].set_xlabel('Latent Factor')
            axes[1, 0].set_ylabel('Latent Factor')
            axes[1, 0].set_xticks(range(n_factors))
            axes[1, 0].set_xticklabels([f'F{i+1}' for i in range(n_factors)])
            axes[1, 0].set_yticks(range(n_factors))
            axes[1, 0].set_yticklabels([f'F{i+1}' for i in range(n_factors)])
            plt.colorbar(im, ax=axes[1, 0])
        else:
            axes[1, 0].text(0.5, 0.5, 'Single Factor\n(No Correlations)',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title(f'Inter-Factor Correlations (K={n_factors})')

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
            axes[1, 1].set_xlabel('Latent Factor')
            axes[1, 1].set_ylabel('% Data Variance Explained')
            axes[1, 1].set_title(f'Approximate Variance Explained (K={n_factors})')
            axes[1, 1].set_xticks(range(n_factors))
            axes[1, 1].set_xticklabels([f'F{i+1}' for i in range(n_factors)])
        else:
            axes[1, 1].text(0.5, 0.5, 'Cannot compute\nvariance explained',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title(f'Approximate Variance Explained (K={n_factors})')

        plt.suptitle(f'Factor Analysis Summary (K={n_factors})', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            save_plot(save_path)
            plt.close()
            logger.info(f"Saved factor summary plot: {save_path}")
        else:
            plt.show()

    def plot_enhanced_factor_loading_distributions(self, W: np.ndarray, data: Dict, save_path: str = None):
        """Create enhanced factor loading distribution plots with proper labeling."""
        view_names = data.get("view_names", [f"View_{i+1}" for i in range(len(data.get("X_list", [])))])
        feature_names = data.get("feature_names", {})
        Dm = [X.shape[1] for X in data.get("X_list", [])]

        if W is None or W.size == 0:
            logger.warning("No factor loadings available for distribution plotting")
            return

        n_factors = W.shape[1]
        n_views = len(view_names)

        # Limit plot complexity for large K to avoid matplotlib memory errors
        if n_factors > 30:
            logger.warning(f"Skipping enhanced_loading_distributions plot: K={n_factors} > 30 factors (would create oversized plot)")
            return

        # Create comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))

        # 1. Overall loading distribution
        ax1 = fig.add_subplot(3, 3, 1)
        ax1.hist(W.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Overall Factor Loading Distribution', fontweight='bold')
        ax1.set_xlabel('Loading Value')
        ax1.set_ylabel('Frequency')
        ax1.axvline(0, color='red', linestyle='--', alpha=0.7, label='Zero')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Absolute loading distribution
        ax2 = fig.add_subplot(3, 3, 2)
        abs_loadings = np.abs(W)
        ax2.hist(abs_loadings.flatten(), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_title('Absolute Factor Loading Distribution', fontweight='bold')
        ax2.set_xlabel('|Loading Value|')
        ax2.set_ylabel('Frequency')

        # Add sparsity information
        sparsity_threshold = 0.1
        sparse_percent = np.mean(abs_loadings < sparsity_threshold) * 100
        ax2.axvline(sparsity_threshold, color='orange', linestyle='--', alpha=0.7,
                   label=f'Threshold (0.1)\n{sparse_percent:.1f}% below')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Loading distribution by factor
        ax3 = fig.add_subplot(3, 3, 3)
        factor_colors = plt.cm.tab10(np.linspace(0, 1, n_factors))
        for k in range(n_factors):
            ax3.hist(W[:, k], bins=30, alpha=0.6, label=f'Factor {k+1}',
                    color=factor_colors[k], edgecolor='black', linewidth=0.5)
        ax3.set_title('Loading Distribution by Factor', fontweight='bold')
        ax3.set_xlabel('Loading Value')
        ax3.set_ylabel('Frequency')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)

        # 4. Loading distribution by view
        ax4 = fig.add_subplot(3, 3, 4)
        d = 0
        view_colors = plt.cm.Set3(np.linspace(0, 1, n_views))
        for m, (view_name, dim) in enumerate(zip(view_names, Dm)):
            W_view = W[d:d+dim, :]
            ax4.hist(W_view.flatten(), bins=30, alpha=0.6, label=view_name,
                    color=view_colors[m], edgecolor='black', linewidth=0.5)
            d += dim
        ax4.set_title('Loading Distribution by View', fontweight='bold')
        ax4.set_xlabel('Loading Value')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Sparsity analysis by factor
        ax5 = fig.add_subplot(3, 3, 5)
        sparsity_levels = []
        for k in range(n_factors):
            sparsity = np.mean(np.abs(W[:, k]) < sparsity_threshold) * 100
            sparsity_levels.append(sparsity)

        bars = ax5.bar(range(n_factors), sparsity_levels, color=factor_colors, alpha=0.7, edgecolor='black')
        ax5.set_title(f'Sparsity by Factor (% < {sparsity_threshold})', fontweight='bold', fontsize=10)
        ax5.set_xlabel('Factor', fontsize=9)
        ax5.set_ylabel('Sparsity (%)', fontsize=9)
        ax5.set_xticks(range(n_factors))
        ax5.set_xticklabels([f'F{k+1}' for k in range(n_factors)], fontsize=8)
        ax5.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, sparsity_levels)):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)

        # 6. Maximum absolute loading by factor
        ax6 = fig.add_subplot(3, 3, 6)
        max_loadings = np.max(np.abs(W), axis=0)
        bars = ax6.bar(range(n_factors), max_loadings, color=factor_colors, alpha=0.7, edgecolor='black')
        ax6.set_title('Maximum |Loading| by Factor', fontweight='bold')
        ax6.set_xlabel('Factor')
        ax6.set_ylabel('Max |Loading|')
        ax6.set_xticks(range(n_factors))
        ax6.set_xticklabels([f'F{k+1}' for k in range(n_factors)])
        ax6.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, max_loadings)):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

        # 7. Loading variance by view
        ax7 = fig.add_subplot(3, 3, 7)
        d = 0
        view_variances = []
        for m, (view_name, dim) in enumerate(zip(view_names, Dm)):
            W_view = W[d:d+dim, :]
            view_var = np.var(W_view.flatten())
            view_variances.append(view_var)
            d += dim

        bars = ax7.bar(range(n_views), view_variances, color=view_colors, alpha=0.7, edgecolor='black')
        ax7.set_title('Loading Variance by View', fontweight='bold')
        ax7.set_xlabel('View')
        ax7.set_ylabel('Loading Variance')
        ax7.set_xticks(range(n_views))
        ax7.set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in view_names], rotation=45, ha='right')
        ax7.grid(True, alpha=0.3, axis='y')

        # 8. Cross-factor loading correlation
        ax8 = fig.add_subplot(3, 3, 8)
        if n_factors > 1:
            loading_corr = np.corrcoef(W.T)
            im = ax8.imshow(loading_corr, cmap='RdBu_r', vmin=-1, vmax=1)
            ax8.set_title('Inter-Factor Loading Correlations', fontweight='bold')
            ax8.set_xlabel('Factor')
            ax8.set_ylabel('Factor')

            # Add correlation values as text
            for i in range(n_factors):
                for j in range(n_factors):
                    ax8.text(j, i, f'{loading_corr[i, j]:.2f}',
                            ha='center', va='center',
                            color='white' if abs(loading_corr[i, j]) > 0.5 else 'black',
                            fontweight='bold', fontsize=8)

            ax8.set_xticks(range(n_factors))
            ax8.set_xticklabels([f'F{k+1}' for k in range(n_factors)])
            ax8.set_yticks(range(n_factors))
            ax8.set_yticklabels([f'F{k+1}' for k in range(n_factors)])
            plt.colorbar(im, ax=ax8, shrink=0.8)
        else:
            ax8.text(0.5, 0.5, 'Single Factor\nNo Correlations',
                    ha='center', va='center', transform=ax8.transAxes,
                    fontsize=12, fontweight='bold')
            ax8.set_title('Inter-Factor Loading Correlations', fontweight='bold')
            ax8.axis('off')

        # 9. Summary statistics table
        ax9 = fig.add_subplot(3, 3, 9)
        ax9.axis('off')

        # Calculate summary statistics
        stats_data = [
            ['Statistic', 'Value'],
            ['Mean |Loading|', f'{np.mean(np.abs(W)):.4f}'],
            ['Std |Loading|', f'{np.std(np.abs(W)):.4f}'],
            ['Max |Loading|', f'{np.max(np.abs(W)):.4f}'],
            ['Min |Loading|', f'{np.min(np.abs(W)):.4f}'],
            ['Overall Sparsity', f'{np.mean(np.abs(W) < 0.1)*100:.1f}%'],
            ['# Factors', f'{n_factors}'],
            ['# Features', f'{W.shape[0]}'],
            ['# Views', f'{n_views}']
        ]

        table = ax9.table(cellText=stats_data, cellLoc='center', loc='center',
                         colWidths=[0.5, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style the header row
        for i in range(len(stats_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        fig.suptitle('Enhanced Factor Loading Analysis', fontsize=16, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96], h_pad=3.0, w_pad=2.0)

        if save_path:
            save_plot(save_path)
            plt.close()
            logger.info(f"Saved enhanced factor loading distribution plot: {save_path}")
        else:
            plt.show()

        return fig

    def plot_interpretable_brain_loadings(self, W: np.ndarray, data: Dict, save_path: str = None):
        """Create interpretable brain loading visualizations with anatomical context."""
        view_names = data.get("view_names", [])
        atlas_info = data.get("atlas_info", {})
        region_names = data.get("region_names", {})

        if W is None or W.size == 0:
            logger.warning("No factor loadings available for brain visualization")
            return

        # Find brain/neuroimaging views - specifically look for your region names
        brain_regions = ['lentiform', 'sn', 'putamen', 'caudate', 'thalamus', 'hippocampus',
                        'amygdala', 'cortical', 'subcortical', 'frontal', 'parietal', 'temporal',
                        'occipital', 'cerebellum', 'brainstem', 'roi', 'region']

        brain_views = []
        for i, view_name in enumerate(view_names):
            # Check if view name contains any brain region keywords
            if any(region in view_name.lower() for region in brain_regions):
                brain_views.append((i, view_name))

        if not brain_views:
            logger.warning("No brain/neuroimaging views identified for brain loading visualization")
            logger.info(f"Available view names: {view_names}")
            logger.info("Consider updating brain region keywords if your regions aren't recognized")
            return

        n_factors = W.shape[1]
        Dm = [X.shape[1] for X in data.get("X_list", [])]

        # Create figure for each brain view
        for view_idx, view_name in brain_views:
            # Extract loadings for this view
            d = sum(Dm[:view_idx])
            dim = Dm[view_idx]
            W_brain = W[d:d+dim, :]

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Brain Factor Loadings: {view_name}', fontsize=16, fontweight='bold')

            # 1. Heatmap of all factor loadings
            ax = axes[0, 0]
            im = ax.imshow(W_brain.T, aspect='auto', cmap='RdBu_r',
                          vmin=-np.max(np.abs(W_brain)), vmax=np.max(np.abs(W_brain)))
            ax.set_title('Factor Loading Heatmap', fontweight='bold')
            ax.set_xlabel('Brain Regions/Features')
            ax.set_ylabel('Factors')
            ax.set_yticks(range(n_factors))
            ax.set_yticklabels([f'Factor {i+1}' for i in range(n_factors)])
            plt.colorbar(im, ax=ax, shrink=0.8, label='Loading Strength')

            # 2. Top loadings per factor
            ax = axes[0, 1]
            n_top = min(10, dim)  # Show top 10 or all if fewer

            factor_colors = plt.cm.tab10(np.linspace(0, 1, n_factors))
            bar_width = 0.8 / n_factors

            for k in range(n_factors):
                # Get top absolute loadings for this factor
                abs_loadings = np.abs(W_brain[:, k])
                top_indices = np.argsort(abs_loadings)[-n_top:][::-1]
                top_values = W_brain[top_indices, k]

                x_pos = np.arange(n_top) + k * bar_width
                bars = ax.bar(x_pos, top_values, bar_width,
                             label=f'Factor {k+1}', color=factor_colors[k], alpha=0.7)

                # Add region names if available
                if view_name in region_names:
                    regions = region_names[view_name]
                    if len(regions) >= max(top_indices) + 1:
                        ax.set_xticks(np.arange(n_top) + bar_width * (n_factors-1) / 2)
                        ax.set_xticklabels([regions[i][:8] + '...' if len(regions[i]) > 8 else regions[i]
                                          for i in top_indices], rotation=45, ha='right')

            ax.set_title(f'Top {n_top} Loadings per Factor', fontweight='bold')
            ax.set_ylabel('Loading Strength')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.axhline(0, color='black', linestyle='-', alpha=0.3)

            # 3. Loading magnitude distribution
            ax = axes[0, 2]
            for k in range(n_factors):
                ax.hist(np.abs(W_brain[:, k]), bins=20, alpha=0.6,
                       label=f'Factor {k+1}', color=factor_colors[k])
            ax.set_title('Loading Magnitude Distribution', fontweight='bold')
            ax.set_xlabel('|Loading|')
            ax.set_ylabel('Number of Regions')
            ax.legend(fontsize=8, loc='upper right', framealpha=0.9, ncol=1)

            # 4. Factor loading variability (replacing spatial coherence when no spatial info)
            ax = axes[1, 0]
            if (self.enable_spatial_analysis and
                view_name in atlas_info and 'adjacency_matrix' in atlas_info[view_name]):
                # Calculate spatial coherence for each factor (requires MRI reference)
                adj_matrix = atlas_info[view_name]['adjacency_matrix']
                coherence_scores = []

                for k in range(n_factors):
                    loadings = W_brain[:, k]
                    coherence = self._calculate_spatial_coherence(loadings, adj_matrix)
                    coherence_scores.append(coherence)

                bars = ax.bar(range(n_factors), coherence_scores, color=factor_colors, alpha=0.7)
                ax.set_title('Spatial Coherence by Factor', fontweight='bold')
                ax.set_xlabel('Factor')
                ax.set_ylabel('Spatial Coherence (Moran\'s I)')
                ax.set_xticks(range(n_factors))
                ax.set_xticklabels([f'F{k+1}' for k in range(n_factors)])

                # Add interpretation guide
                ax.axhline(0, color='black', linestyle='--', alpha=0.5, label='Random')
                ax.legend(fontsize=8, loc='upper right', framealpha=0.9, ncol=1)
            else:
                # Alternative analysis: Factor loading variability (doesn't require spatial info)
                logger.info(f"No spatial information available for {view_name}. Using loading variability analysis instead.")

                variability_scores = []
                for k in range(n_factors):
                    loadings = W_brain[:, k]
                    # Use coefficient of variation as variability measure
                    cv = np.std(loadings) / (np.abs(np.mean(loadings)) + 1e-8)
                    variability_scores.append(cv)

                bars = ax.bar(range(n_factors), variability_scores, color=factor_colors, alpha=0.7)
                ax.set_title('Factor Loading Variability', fontweight='bold')
                ax.set_xlabel('Factor')
                ax.set_ylabel('Coefficient of Variation')
                ax.set_xticks(range(n_factors))
                ax.set_xticklabels([f'F{k+1}' for k in range(n_factors)])

                # Add value labels
                for i, (bar, score) in enumerate(zip(bars, variability_scores)):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

                # Add explanatory text
                ax.text(0.02, 0.98, 'Note: Spatial analysis requires\nMRI reference image',
                       transform=ax.transAxes, fontsize=8, va='top', ha='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

            # 5. Factor interpretability scores
            ax = axes[1, 1]
            interpretability_scores = []
            for k in range(n_factors):
                # Calculate interpretability as combination of sparsity and magnitude
                abs_loadings = np.abs(W_brain[:, k])
                sparsity = np.mean(abs_loadings < 0.1)  # Fraction of small loadings
                max_loading = np.max(abs_loadings)
                # Interpretability increases with sparsity and strong peak loadings
                interpretability = sparsity * max_loading
                interpretability_scores.append(interpretability)

            bars = ax.bar(range(n_factors), interpretability_scores, color=factor_colors, alpha=0.7)
            ax.set_title('Factor Interpretability Scores', fontweight='bold')
            ax.set_xlabel('Factor')
            ax.set_ylabel('Interpretability')
            ax.set_xticks(range(n_factors))
            ax.set_xticklabels([f'F{k+1}' for k in range(n_factors)])

            # Add value labels
            for i, (bar, score) in enumerate(zip(bars, interpretability_scores)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

            # 6. Clinical relevance indicators
            ax = axes[1, 2]
            ax.axis('off')

            # Create summary table for this view
            summary_data = [
                ['Metric', 'Value'],
                ['Brain Regions', f'{dim}'],
                ['Avg |Loading|', f'{np.mean(np.abs(W_brain)):.4f}'],
                ['Max |Loading|', f'{np.max(np.abs(W_brain)):.4f}'],
                ['Sparsity %', f'{np.mean(np.abs(W_brain) < 0.1)*100:.1f}%'],
                ['Most Interpretable', f'Factor {np.argmax(interpretability_scores)+1}'],
                ['Strongest Factor', f'Factor {np.argmax(np.max(np.abs(W_brain), axis=0))+1}']
            ]

            table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                           colWidths=[0.6, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2.5)

            # Style the header row
            for i in range(len(summary_data[0])):
                table[(0, i)].set_facecolor('#2196F3')
                table[(0, i)].set_text_props(weight='bold', color='white')

            plt.tight_layout()

            if save_path:
                brain_save_path = save_path.replace('.png', f'_{view_name.replace(" ", "_")}_brain.png')
                save_plot(brain_save_path)
                plt.close()
                logger.info(f"Saved brain loading plot: {brain_save_path}")
            else:
                plt.show()

    def _calculate_spatial_coherence(self, loadings: np.ndarray, adjacency_matrix: np.ndarray) -> float:
        """Calculate spatial coherence (Moran's I) for factor loadings."""
        try:
            n = len(loadings)
            if adjacency_matrix.shape != (n, n):
                return 0.0

            # Calculate Moran's I
            W = adjacency_matrix
            W_sum = np.sum(W)

            if W_sum == 0:
                return 0.0

            mean_loading = np.mean(loadings)
            num = 0
            den = 0

            for i in range(n):
                for j in range(n):
                    if i != j:
                        num += W[i, j] * (loadings[i] - mean_loading) * (loadings[j] - mean_loading)
                den += (loadings[i] - mean_loading) ** 2

            if den == 0:
                return 0.0

            moran_i = (n / W_sum) * (num / den)
            return moran_i

        except Exception as e:
            logger.warning(f"Could not calculate spatial coherence: {e}")
            return 0.0

    def plot_clinical_factor_loadings(self, W: np.ndarray, data: Dict, save_path: str = None):
        """Create detailed clinical variable factor loading visualization."""
        view_names = data.get("view_names", [])
        feature_names = data.get("feature_names", {})
        Dm = [X.shape[1] for X in data.get("X_list", [])]

        # Find clinical view
        clinical_view_idx = None
        clinical_view_name = None
        for i, view_name in enumerate(view_names):
            if 'clinical' in view_name.lower():
                clinical_view_idx = i
                clinical_view_name = view_name
                break

        if clinical_view_idx is None:
            logger.warning("No clinical view found for clinical factor loading visualization")
            return

        # Extract clinical loadings
        d = sum(Dm[:clinical_view_idx])
        dim = Dm[clinical_view_idx]
        W_clinical = W[d:d+dim, :]
        n_factors = W_clinical.shape[1]

        # Get clinical feature names
        if clinical_view_name in feature_names:
            clinical_feat_names = feature_names[clinical_view_name]
        else:
            clinical_feat_names = [f"Clinical_var_{i}" for i in range(dim)]

        # Create comprehensive figure
        fig = plt.figure(figsize=(18, 12))

        # 1. Heatmap of all clinical loadings
        ax1 = plt.subplot(2, 3, 1)
        im = ax1.imshow(W_clinical.T, aspect='auto', cmap='RdBu_r',
                       vmin=-np.max(np.abs(W_clinical)), vmax=np.max(np.abs(W_clinical)))
        ax1.set_title('Clinical Feature Loadings Heatmap', fontweight='bold')
        ax1.set_ylabel('Factors')
        ax1.set_yticks(range(n_factors))
        ax1.set_yticklabels([f'Factor {i+1}' for i in range(n_factors)])

        if dim <= 30:
            ax1.set_xticks(range(dim))
            ax1.set_xticklabels(clinical_feat_names, rotation=45, ha='right', fontsize=8)
        else:
            ax1.set_xlabel(f'Clinical Features (n={dim})')

        plt.colorbar(im, ax=ax1, shrink=0.8, label='Loading Strength')

        # 2. Top loadings per factor (bar plot)
        ax2 = plt.subplot(2, 3, 2)
        n_top = min(10, dim)
        factor_colors = plt.cm.tab10(np.linspace(0, 1, n_factors))

        for k in range(n_factors):
            abs_loadings = np.abs(W_clinical[:, k])
            top_indices = np.argsort(abs_loadings)[-n_top:][::-1]
            top_values = W_clinical[top_indices, k]
            top_names = [clinical_feat_names[i] for i in top_indices]

            offset = k * (n_top + 2)
            colors = ['red' if x < 0 else 'blue' for x in top_values]
            ax2.barh(np.arange(offset, offset + n_top), top_values,
                    color=colors, alpha=0.7, label=f'Factor {k+1}')

            # Add feature names
            for i, (val, name) in enumerate(zip(top_values, top_names)):
                ax2.text(0, offset + i, f'  {name}', va='center', fontsize=7, fontweight='bold')

        ax2.set_title(f'Top {n_top} Clinical Features per Factor', fontweight='bold')
        ax2.set_xlabel('Loading Value')
        ax2.set_yticks([])
        ax2.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax2.legend(loc='lower right')

        # 3. Feature importance across all factors
        ax3 = plt.subplot(2, 3, 3)
        feature_importance = np.max(np.abs(W_clinical), axis=1)
        sorted_indices = np.argsort(feature_importance)[-15:][::-1]  # Top 15

        bars = ax3.barh(range(len(sorted_indices)),
                       feature_importance[sorted_indices],
                       color=plt.cm.viridis(np.linspace(0, 1, len(sorted_indices))))
        ax3.set_yticks(range(len(sorted_indices)))
        ax3.set_yticklabels([clinical_feat_names[i] for i in sorted_indices], fontsize=8)
        ax3.set_xlabel('Max |Loading| Across Factors')
        ax3.set_title('Most Important Clinical Features', fontweight='bold')

        # 4. Loading distribution by feature category (if identifiable)
        ax4 = plt.subplot(2, 3, 4)

        # Categorize clinical features
        categories = {}
        for i, name in enumerate(clinical_feat_names):
            if any(kw in name.lower() for kw in ['age', 'sex', 'tiv', 'demographic']):
                categories.setdefault('Demographics', []).append(i)
            elif any(kw in name.lower() for kw in ['rigidity', 'rigid']):
                categories.setdefault('Rigidity', []).append(i)
            elif any(kw in name.lower() for kw in ['tremor']):
                categories.setdefault('Tremor', []).append(i)
            elif any(kw in name.lower() for kw in ['bradykinesia', 'brady']):
                categories.setdefault('Bradykinesia', []).append(i)
            elif any(kw in name.lower() for kw in ['mirror', 'movement']):
                categories.setdefault('Mirror Movement', []).append(i)
            else:
                categories.setdefault('Other', []).append(i)

        if len(categories) > 1:
            category_loadings = []
            category_names = []
            for cat_name, indices in categories.items():
                if indices:
                    mean_loading = np.mean(np.abs(W_clinical[indices, :]))
                    category_loadings.append(mean_loading)
                    category_names.append(f"{cat_name} (n={len(indices)})")

            bars = ax4.barh(range(len(category_names)), category_loadings,
                          color=plt.cm.Set3(np.linspace(0, 1, len(category_names))))
            ax4.set_yticks(range(len(category_names)))
            ax4.set_yticklabels(category_names, fontsize=9)
            ax4.set_xlabel('Mean |Loading|')
            ax4.set_title('Loading Strength by Feature Category', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Feature categories\nnot identified',
                    ha='center', va='center', transform=ax4.transAxes, fontsize=10)
            ax4.set_title('Loading Strength by Feature Category', fontweight='bold')

        # 5. Factor interpretability for clinical features
        ax5 = plt.subplot(2, 3, 5)
        interpretability_scores = []
        for k in range(n_factors):
            sparsity = np.mean(np.abs(W_clinical[:, k]) < 0.1)
            max_loading = np.max(np.abs(W_clinical[:, k]))
            interpretability = sparsity * max_loading
            interpretability_scores.append(interpretability)

        bars = ax5.bar(range(n_factors), interpretability_scores,
                      color=factor_colors, alpha=0.7)
        ax5.set_xlabel('Factor')
        ax5.set_ylabel('Interpretability Score')
        ax5.set_title('Clinical Factor Interpretability', fontweight='bold')
        ax5.set_xticks(range(n_factors))
        ax5.set_xticklabels([f'F{k+1}' for k in range(n_factors)])

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, interpretability_scores)):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

        # 6. Summary table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        # Find most important feature per factor
        most_important_per_factor = []
        for k in range(n_factors):
            max_idx = np.argmax(np.abs(W_clinical[:, k]))
            feat_name = clinical_feat_names[max_idx]
            loading_val = W_clinical[max_idx, k]
            most_important_per_factor.append(f"{feat_name[:20]}... ({loading_val:.3f})"
                                            if len(feat_name) > 20
                                            else f"{feat_name} ({loading_val:.3f})")

        summary_data = [
            ['Metric', 'Value'],
            ['Clinical Features', f'{dim}'],
            ['Number of Factors', f'{n_factors}'],
            ['Avg |Loading|', f'{np.mean(np.abs(W_clinical)):.4f}'],
            ['Max |Loading|', f'{np.max(np.abs(W_clinical)):.4f}'],
            ['Sparsity %', f'{np.mean(np.abs(W_clinical) < 0.1)*100:.1f}%'],
            ['Most Interpretable Factor', f'Factor {np.argmax(interpretability_scores)+1}'],
        ]

        # Add top feature per factor
        for k in range(min(3, n_factors)):  # Show top 3 factors
            summary_data.append([f'Top Feature F{k+1}', most_important_per_factor[k]])

        table = ax6.table(cellText=summary_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Style header row
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.suptitle('Clinical Factor Loading Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            clinical_save_path = save_path.replace('.png', '_clinical_detailed.png')
            save_plot(clinical_save_path)
            plt.close()
            logger.info(f"Saved clinical factor loading plot: {clinical_save_path}")
        else:
            plt.show()

    def plot_region_wise_factor_analysis(self, W: np.ndarray, data: Dict, save_path: str = None):
        """Create comprehensive region-wise factor analysis for brain data organized as separate views."""
        view_names = data.get("view_names", [])
        Dm = [X.shape[1] for X in data.get("X_list", [])]

        if W is None or W.size == 0:
            logger.warning("No factor loadings available for region-wise analysis")
            return

        # Identify all brain region views
        brain_regions = ['lentiform', 'sn', 'putamen', 'caudate', 'thalamus', 'hippocampus',
                        'amygdala', 'cortical', 'subcortical', 'frontal', 'parietal', 'temporal',
                        'occipital', 'cerebellum', 'brainstem', 'roi', 'region']

        brain_views = []
        for i, view_name in enumerate(view_names):
            if any(region in view_name.lower() for region in brain_regions):
                brain_views.append((i, view_name))

        if not brain_views:
            logger.warning("No brain regions found for region-wise analysis")
            return

        logger.info(f"Found {len(brain_views)} brain regions: {[name for _, name in brain_views]}")

        n_factors = W.shape[1]
        n_regions = len(brain_views)

        # Create comprehensive region-wise visualization
        fig = plt.figure(figsize=(20, 16))

        # 1. Factor loading heatmap across all brain regions
        ax1 = plt.subplot(3, 4, 1)

        # Extract all brain region loadings
        brain_loadings = []
        region_names = []
        region_boundaries = [0]

        for view_idx, view_name in brain_views:
            d = sum(Dm[:view_idx])
            dim = Dm[view_idx]
            W_region = W[d:d+dim, :]
            brain_loadings.append(W_region)
            region_names.append(view_name)
            region_boundaries.append(region_boundaries[-1] + dim)

        # Concatenate all brain loadings
        W_all_brain = np.vstack(brain_loadings)

        im = ax1.imshow(W_all_brain.T, aspect='auto', cmap='RdBu_r',
                       vmin=-np.max(np.abs(W_all_brain)), vmax=np.max(np.abs(W_all_brain)))
        ax1.set_title('Factor Loadings Across All Brain Regions', fontweight='bold')
        ax1.set_xlabel('Brain Features (by Region)')
        ax1.set_ylabel('Factors')
        ax1.set_yticks(range(n_factors))
        ax1.set_yticklabels([f'F{i+1}' for i in range(n_factors)])

        # Add vertical lines to separate regions
        for boundary in region_boundaries[1:-1]:
            ax1.axvline(boundary - 0.5, color='white', linewidth=2, alpha=0.8)

        # Add region labels
        region_centers = [(region_boundaries[i] + region_boundaries[i+1]) / 2
                         for i in range(len(region_boundaries)-1)]
        ax1.set_xticks(region_centers)
        ax1.set_xticklabels([name[:8] + '...' if len(name) > 8 else name
                            for name in region_names], rotation=45, ha='right')

        plt.colorbar(im, ax=ax1, shrink=0.6, label='Loading Strength')

        # 2. Average absolute loading by region and factor
        ax2 = plt.subplot(3, 4, 2)

        region_factor_loadings = np.zeros((n_regions, n_factors))
        for i, (view_idx, view_name) in enumerate(brain_views):
            d = sum(Dm[:view_idx])
            dim = Dm[view_idx]
            W_region = W[d:d+dim, :]
            region_factor_loadings[i, :] = np.mean(np.abs(W_region), axis=0)

        im2 = ax2.imshow(region_factor_loadings, aspect='auto', cmap='viridis')
        ax2.set_title('Average |Loading| by Region & Factor', fontweight='bold')
        ax2.set_xlabel('Factors')
        ax2.set_ylabel('Brain Regions')
        ax2.set_xticks(range(n_factors))
        ax2.set_xticklabels([f'F{i+1}' for i in range(n_factors)])
        ax2.set_yticks(range(n_regions))
        ax2.set_yticklabels([name[:12] + '...' if len(name) > 12 else name
                            for _, name in brain_views])

        # Add values to heatmap
        for i in range(n_regions):
            for j in range(n_factors):
                ax2.text(j, i, f'{region_factor_loadings[i, j]:.3f}',
                        ha='center', va='center',
                        color='white' if region_factor_loadings[i, j] > np.max(region_factor_loadings)/2 else 'black',
                        fontweight='bold', fontsize=8)

        plt.colorbar(im2, ax=ax2, shrink=0.6, label='Avg |Loading|')

        # 3. Region-wise factor contributions (stacked bar)
        ax3 = plt.subplot(3, 4, 3)

        factor_colors = plt.cm.tab10(np.linspace(0, 1, n_factors))
        bottom = np.zeros(n_regions)

        for k in range(n_factors):
            values = region_factor_loadings[:, k]
            ax3.bar(range(n_regions), values, bottom=bottom,
                   label=f'Factor {k+1}', color=factor_colors[k], alpha=0.8)
            bottom += values

        ax3.set_title('Factor Contributions by Region', fontweight='bold')
        ax3.set_xlabel('Brain Regions')
        ax3.set_ylabel('Cumulative |Loading|')
        ax3.set_xticks(range(n_regions))
        ax3.set_xticklabels([name[:8] + '...' if len(name) > 8 else name
                            for _, name in brain_views], rotation=45, ha='right')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 4. Regional sparsity analysis
        ax4 = plt.subplot(3, 4, 4)

        sparsity_threshold = 0.1
        region_sparsity = []

        for view_idx, view_name in brain_views:
            d = sum(Dm[:view_idx])
            dim = Dm[view_idx]
            W_region = W[d:d+dim, :]
            sparsity = np.mean(np.abs(W_region) < sparsity_threshold) * 100
            region_sparsity.append(sparsity)

        bars = ax4.bar(range(n_regions), region_sparsity, alpha=0.7,
                      color=plt.cm.viridis(np.linspace(0, 1, n_regions)))
        ax4.set_title(f'Sparsity by Region\n(% loadings < {sparsity_threshold})', fontweight='bold')
        ax4.set_xlabel('Brain Regions')
        ax4.set_ylabel('Sparsity (%)')
        ax4.set_xticks(range(n_regions))
        ax4.set_xticklabels([name[:8] + '...' if len(name) > 8 else name
                            for _, name in brain_views], rotation=45, ha='right')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, region_sparsity)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)

        # 5. Factor-wise regional distribution
        ax5 = plt.subplot(3, 4, 5)

        # Show which regions contribute most to each factor
        max_contrib_regions = []
        for k in range(n_factors):
            max_region_idx = np.argmax(region_factor_loadings[:, k])
            max_contrib_regions.append(brain_views[max_region_idx][1])

        factor_positions = np.arange(n_factors)
        max_values = [np.max(region_factor_loadings[:, k]) for k in range(n_factors)]

        bars = ax5.bar(factor_positions, max_values, color=factor_colors, alpha=0.7)
        ax5.set_title('Peak Regional Contribution by Factor', fontweight='bold')
        ax5.set_xlabel('Factors')
        ax5.set_ylabel('Max Regional |Loading|')
        ax5.set_xticks(factor_positions)
        ax5.set_xticklabels([f'F{k+1}' for k in range(n_factors)])

        # Add region labels on bars
        for i, (bar, region) in enumerate(zip(bars, max_contrib_regions)):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    region[:8] + '...' if len(region) > 8 else region,
                    ha='center', va='bottom', fontweight='bold', fontsize=8, rotation=45)

        # 6. Cross-regional correlation matrix
        ax6 = plt.subplot(3, 4, 6)

        if n_regions > 1:
            region_corr = np.corrcoef(region_factor_loadings)
            im3 = ax6.imshow(region_corr, cmap='RdBu_r', vmin=-1, vmax=1)
            ax6.set_title('Inter-Regional Loading Correlations', fontweight='bold')
            ax6.set_xlabel('Brain Regions')
            ax6.set_ylabel('Brain Regions')
            ax6.set_xticks(range(n_regions))
            ax6.set_xticklabels([name[:8] + '...' if len(name) > 8 else name
                                for _, name in brain_views], rotation=45, ha='right')
            ax6.set_yticks(range(n_regions))
            ax6.set_yticklabels([name[:8] + '...' if len(name) > 8 else name
                                for _, name in brain_views])

            # Add correlation values
            for i in range(n_regions):
                for j in range(n_regions):
                    ax6.text(j, i, f'{region_corr[i, j]:.2f}',
                            ha='center', va='center',
                            color='white' if abs(region_corr[i, j]) > 0.5 else 'black',
                            fontweight='bold', fontsize=8)

            plt.colorbar(im3, ax=ax6, shrink=0.6, label='Correlation')
        else:
            ax6.text(0.5, 0.5, 'Single Region\nNo Correlations',
                    ha='center', va='center', transform=ax6.transAxes,
                    fontsize=12, fontweight='bold')
            ax6.set_title('Inter-Regional Loading Correlations', fontweight='bold')

        # 7. Regional factor variance explained
        ax7 = plt.subplot(3, 4, 7)

        region_variance = []
        for view_idx, view_name in brain_views:
            d = sum(Dm[:view_idx])
            dim = Dm[view_idx]
            W_region = W[d:d+dim, :]
            variance = np.var(W_region, axis=0)  # Variance per factor
            region_variance.append(variance)

        # Create stacked bar chart of variance by factor and region
        bottom = np.zeros(n_factors)
        for i, (view_idx, view_name) in enumerate(brain_views):
            ax7.bar(range(n_factors), region_variance[i], bottom=bottom,
                   label=view_name[:10] + '...' if len(view_name) > 10 else view_name,
                   alpha=0.7)
            bottom += region_variance[i]

        ax7.set_title('Loading Variance by Factor & Region', fontweight='bold')
        ax7.set_xlabel('Factors')
        ax7.set_ylabel('Cumulative Variance')
        ax7.set_xticks(range(n_factors))
        ax7.set_xticklabels([f'F{k+1}' for k in range(n_factors)])
        ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # 8. Regional interpretability scores
        ax8 = plt.subplot(3, 4, 8)

        region_interpretability = []
        for view_idx, view_name in brain_views:
            d = sum(Dm[:view_idx])
            dim = Dm[view_idx]
            W_region = W[d:d+dim, :]

            # Calculate interpretability as sparsity * peak loading
            sparsity = np.mean(np.abs(W_region) < 0.1)
            max_loading = np.max(np.abs(W_region))
            interpretability = sparsity * max_loading
            region_interpretability.append(interpretability)

        bars = ax8.bar(range(n_regions), region_interpretability,
                      color=plt.cm.plasma(np.linspace(0, 1, n_regions)), alpha=0.7)
        ax8.set_title('Regional Interpretability Scores', fontweight='bold')
        ax8.set_xlabel('Brain Regions')
        ax8.set_ylabel('Interpretability')
        ax8.set_xticks(range(n_regions))
        ax8.set_xticklabels([name[:8] + '...' if len(name) > 8 else name
                            for _, name in brain_views], rotation=45, ha='right')

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, region_interpretability)):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

        # 9. Summary statistics table
        ax9 = plt.subplot(3, 4, 9)
        ax9.axis('off')

        # Calculate summary statistics
        most_active_region = brain_views[np.argmax([np.mean(np.abs(W[sum(Dm[:i]):sum(Dm[:i])+Dm[i], :]))
                                                   for i, _ in enumerate(brain_views)
                                                   for view_idx, _ in [brain_views[i]] if i < len(Dm)])][1]
        most_sparse_region = brain_views[np.argmax(region_sparsity)][1]
        most_interpretable_region = brain_views[np.argmax(region_interpretability)][1]

        summary_data = [
            ['Metric', 'Value'],
            ['Brain Regions', f'{n_regions}'],
            ['Total Brain Features', f'{sum([Dm[i] for i, _ in brain_views])}'],
            ['Most Active Region', most_active_region[:15] + '...' if len(most_active_region) > 15 else most_active_region],
            ['Most Sparse Region', most_sparse_region[:15] + '...' if len(most_sparse_region) > 15 else most_sparse_region],
            ['Most Interpretable', most_interpretable_region[:15] + '...' if len(most_interpretable_region) > 15 else most_interpretable_region],
            ['Avg Regional Sparsity', f'{np.mean(region_sparsity):.1f}%'],
            ['Cross-Regional Corr', f'{np.mean(np.abs(region_corr[np.triu_indices(n_regions, k=1)])):.3f}' if n_regions > 1 else 'N/A']
        ]

        table = ax9.table(cellText=summary_data, cellLoc='center', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style the header row
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor('#FF9800')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # 10-12. Individual region highlights (top 3 most interesting)
        top_regions_indices = np.argsort(region_interpretability)[-3:][::-1]

        for plot_idx, region_rank in enumerate(top_regions_indices):
            ax = plt.subplot(3, 4, 10 + plot_idx)
            view_idx, view_name = brain_views[region_rank]
            d = sum(Dm[:view_idx])
            dim = Dm[view_idx]
            W_region = W[d:d+dim, :]

            # Create mini heatmap for this region
            im = ax.imshow(W_region.T, aspect='auto', cmap='RdBu_r',
                          vmin=-np.max(np.abs(W_region)), vmax=np.max(np.abs(W_region)))
            ax.set_title(f'{view_name}\n(Rank #{plot_idx+1} Interpretability)', fontweight='bold', fontsize=10)
            ax.set_xlabel('Features')
            ax.set_ylabel('Factors')
            ax.set_yticks(range(n_factors))
            ax.set_yticklabels([f'F{i+1}' for i in range(n_factors)])

        plt.suptitle('Comprehensive Region-Wise Factor Analysis', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])

        if save_path:
            region_save_path = save_path.replace('.png', '_region_wise_analysis.png')
            save_plot(region_save_path)
            plt.close()
            logger.info(f"Saved region-wise factor analysis: {region_save_path}")
        else:
            plt.show()
