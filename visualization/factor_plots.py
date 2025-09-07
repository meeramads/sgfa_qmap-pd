# visualization/factor_plots.py
"""Factor analysis visualization module."""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FactorVisualizer:
    """Creates factor analysis visualizations."""
    
    def __init__(self, config):
        self.config = config
        self.setup_style()
        
    def setup_style(self):
        """Setup consistent plotting style."""
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def create_plots(self, results: Dict, data: Dict, plot_dir: Path):
        """Create all factor analysis plots."""
        logger.info("Creating factor analysis plots")
        
        # Extract best run
        best_run = self._find_best_run(results)
        if not best_run:
            logger.warning("No converged runs found")
            return
            
        W = best_run.get('W')
        Z = best_run.get('Z')
        
        if W is None or Z is None:
            logger.error("Missing W or Z matrices")
            return
            
        # Create various plots
        self._plot_factor_loadings(W, data, plot_dir / 'factors')
        self._plot_factor_scores(Z, data, plot_dir / 'factors')
        self._plot_factor_heatmap(W, Z, plot_dir / 'factors')
        self._plot_view_contributions(W, data, plot_dir / 'factors')
        
    def _plot_factor_loadings(self, W: np.ndarray, data: Dict, save_dir: Path):
        """Plot factor loadings for each view."""
        view_names = data.get('view_names', [])
        Dm = [X.shape[1] for X in data.get('X_list', [])]
        
        # Create figure with subplots for each view
        n_views = len(view_names)
        fig, axes = plt.subplots(1, n_views, figsize=(5*n_views, 6))
        if n_views == 1:
            axes = [axes]
            
        d = 0
        for m, (view_name, dim) in enumerate(zip(view_names, Dm)):
            W_view = W[d:d+dim, :]
            
            # Plot heatmap
            im = axes[m].imshow(
                W_view.T, 
                aspect='auto', 
                cmap='RdBu_r',
                vmin=-np.max(np.abs(W_view)),
                vmax=np.max(np.abs(W_view))
            )
            
            axes[m].set_title(f'{view_name}\n({dim} features)')
            axes[m].set_xlabel('Features')
            axes[m].set_ylabel('Factors')
            axes[m].set_yticks(range(W_view.shape[1]))
            axes[m].set_yticklabels([f'F{i+1}' for i in range(W_view.shape[1])])
            
            # Add colorbar
            plt.colorbar(im, ax=axes[m], fraction=0.046)
            
            d += dim
            
        plt.suptitle('Factor Loadings by View', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = save_dir / 'factor_loadings_by_view.png'
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved: {save_path}")
        
    def _plot_factor_scores(self, Z: np.ndarray, data: Dict, save_dir: Path):
        """Plot factor scores heatmap."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        im = ax.imshow(
            Z.T,
            aspect='auto',
            cmap='RdBu_r',
            vmin=-np.max(np.abs(Z)),
            vmax=np.max(np.abs(Z))
        )
        
        ax.set_xlabel('Subjects')
        ax.set_ylabel('Factors')
        ax.set_yticks(range(Z.shape[1]))
        ax.set_yticklabels([f'Factor {i+1}' for i in range(Z.shape[1])])
        ax.set_title('Factor Scores Across Subjects', fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label('Factor Score', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        # Save
        save_path = save_dir / 'factor_scores.png'
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved: {save_path}")
        
    def _plot_factor_heatmap(self, W: np.ndarray, Z: np.ndarray, save_dir: Path):
        """Plot comprehensive factor analysis heatmap."""
        # Calculate factor statistics
        factor_stats = self._calculate_factor_statistics(W, Z)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Loading magnitude per factor
        loading_mags = np.mean(np.abs(W), axis=0)
        axes[0, 0].bar(range(len(loading_mags)), loading_mags)
        axes[0, 0].set_xlabel('Factor')
        axes[0, 0].set_ylabel('Mean |Loading|')
        axes[0, 0].set_title('Factor Loading Magnitudes')
        axes[0, 0].set_xticks(range(len(loading_mags)))
        axes[0, 0].set_xticklabels([f'F{i+1}' for i in range(len(loading_mags))])
        
        # 2. Factor score variance
        score_vars = np.var(Z, axis=0)
        axes[0, 1].bar(range(len(score_vars)), score_vars)
        axes[0, 1].set_xlabel('Factor')
        axes[0, 1].set_ylabel('Variance')
        axes[0, 1].set_title('Factor Score Variance')
        axes[0, 1].set_xticks(range(len(score_vars)))
        axes[0, 1].set_xticklabels([f'F{i+1}' for i in range(len(score_vars))])
        
        # 3. Factor correlation matrix
        factor_corr = np.corrcoef(Z.T)
        im = axes[1, 0].imshow(factor_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 0].set_title('Factor Correlation Matrix')
        axes[1, 0].set_xlabel('Factor')
        axes[1, 0].set_ylabel('Factor')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Explained variance (approximation)
        total_var = np.sum(np.var(Z @ W.T, axis=0))
        factor_vars = []
        for k in range(Z.shape[1]):
            Z_k = Z[:, k:k+1]
            W_k = W[:, k:k+1]
            factor_var = np.sum(np.var(Z_k @ W_k.T, axis=0))
            factor_vars.append(factor_var / total_var * 100)
            
        axes[1, 1].bar(range(len(factor_vars)), factor_vars)
        axes[1, 1].set_xlabel('Factor')
        axes[1, 1].set_ylabel('% Variance Explained')
        axes[1, 1].set_title('Approximate Variance Explained')
        axes[1, 1].set_xticks(range(len(factor_vars)))
        axes[1, 1].set_xticklabels([f'F{i+1}' for i in range(len(factor_vars))])
        
        plt.suptitle('Factor Analysis Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = save_dir / 'factor_summary.png'
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved: {save_path}")
        
    def _plot_view_contributions(self, W: np.ndarray, data: Dict, save_dir: Path):
        """Plot contribution of each view to factors."""
        view_names = data.get('view_names', [])
        Dm = [X.shape[1] for X in data.get('X_list', [])]
        n_factors = W.shape[1]
        
        # Calculate view contributions
        view_contributions = np.zeros((len(view_names), n_factors))
        
        d = 0
        for m, dim in enumerate(Dm):
            W_view = W[d:d+dim, :]
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
            ax.bar(x, view_contributions[m, :], width, bottom=bottom,
                  label=view_name, color=colors[m])
            bottom += view_contributions[m, :]
            
        ax.set_xlabel('Factor')
        ax.set_ylabel('Mean |Loading|')
        ax.set_title('View Contributions to Each Factor', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'F{i+1}' for i in range(n_factors)])
        ax.legend()
        
        plt.tight_layout()
        
        # Save
        save_path = save_dir / 'view_contributions.png'
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved: {save_path}")
        
    def _find_best_run(self, results: Dict) -> Optional[Dict]:
        """Find the best run from results."""
        best_score = -np.inf
        best_run = None
        
        for run_id, run_data in results.items():
            if isinstance(run_data, dict) and 'robust' in run_data:
                # Use robust parameters if available
                if 'exp_logdensity' in run_data:
                    score = run_data['exp_logdensity']
                    if score > best_score:
                        best_score = score
                        best_run = run_data['robust']
                        
        return best_run
    
    def _calculate_factor_statistics(self, W: np.ndarray, Z: np.ndarray) -> Dict:
        """Calculate various factor statistics."""
        return {
            'n_factors': W.shape[1],
            'n_features': W.shape[0],
            'n_subjects': Z.shape[0],
            'loading_sparsity': np.mean(np.abs(W) < 0.01),
            'score_kurtosis': np.mean([np.abs(scipy.stats.kurtosis(Z[:, k])) 
                                      for k in range(Z.shape[1])]),
            'mean_factor_correlation': np.mean(np.abs(np.corrcoef(Z.T)[np.triu_indices(Z.shape[1], k=1)]))
        }