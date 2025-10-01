"""
PD Subtype Discovery Visualization Module.

Provides specialized visualizations for Parkinson's Disease subtype discovery,
including clinical validation, subtype characterization, and research decision support.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from core.io_utils import save_plot

logger = logging.getLogger(__name__)


class PDSubtypeVisualizer:
    """Creates PD subtype discovery and validation visualizations."""

    def __init__(self, config):
        self.config = config
        self.setup_style()

    def setup_style(self):
        """Setup consistent plotting style for PD research."""
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "figure.figsize": (12, 8),
                "font.size": 11,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "font.weight": "normal",
                "axes.titleweight": "bold",
            }
        )

    def create_pd_subtype_plots(
        self,
        results: Dict,
        Z_sgfa: np.ndarray,
        clinical_data: Optional[Union[Dict, pd.DataFrame]],
        plot_dir: Path
    ):
        """
        Create comprehensive PD subtype discovery visualizations.

        Parameters
        ----------
        results : Dict
            Subtype discovery results containing clustering and validation data
        Z_sgfa : np.ndarray
            SGFA factor scores [n_subjects x n_factors]
        clinical_data : Optional[Union[Dict, pd.DataFrame]]
            Clinical measures for validation
        plot_dir : Path
            Directory for saving plots
        """
        logger.info("Creating PD subtype discovery visualizations")

        plot_dir.mkdir(parents=True, exist_ok=True)

        # 1. Clustering quality and optimization
        if "subtype_discovery" in results:
            self._plot_clustering_quality(
                results["subtype_discovery"],
                plot_dir / "clustering_quality.png"
            )

        # 2. Subtype visualization in factor space
        if "subtype_discovery" in results:
            self._plot_subtype_factor_space(
                Z_sgfa,
                results["subtype_discovery"]["best_solution"]["labels"],
                plot_dir / "subtype_factor_space.png"
            )

        # 3. Clinical validation bubble plot
        if "clinical_validation" in results and clinical_data is not None:
            self._plot_clinical_validation_bubble(
                results["clinical_validation"]["clinical_measures"],
                plot_dir / "clinical_validation.png"
            )

        # 4. PD subtype characterization panel
        if "clinical_validation" in results and clinical_data is not None:
            self._plot_pd_subtype_characterization(
                results,
                Z_sgfa,
                clinical_data,
                plot_dir / "pd_subtype_characterization.png"
            )

        # 5. Subtype discovery pipeline summary
        self._plot_subtype_discovery_summary(
            results,
            plot_dir / "subtype_discovery_summary.png"
        )

        logger.info(f"PD subtype plots saved to {plot_dir}")

    def create_performance_subtype_plots(
        self,
        integrated_results: Dict,
        plot_dir: Path
    ):
        """
        Create performance vs subtype discovery trade-off visualizations.

        Parameters
        ----------
        integrated_results : Dict
            Integrated benchmark results with performance and subtype metrics
        plot_dir : Path
            Directory for saving plots
        """
        logger.info("Creating performance-subtype trade-off visualizations")

        plot_dir.mkdir(parents=True, exist_ok=True)

        # 1. Performance overview across K values
        self._plot_integrated_performance_overview(
            integrated_results,
            plot_dir / "performance_overview.png"
        )

        # 2. Trade-off analysis scatter plots
        self._plot_performance_discovery_tradeoffs(
            integrated_results,
            plot_dir / "trade_off_analysis.png"
        )

        # 3. PD research decision matrix
        self._plot_pd_research_decision_matrix(
            integrated_results,
            plot_dir / "pd_research_decision_matrix.png"
        )

        logger.info(f"Performance-subtype plots saved to {plot_dir}")

    def _plot_clustering_quality(self, subtype_discovery: Dict, plot_path: Path):
        """Plot clustering quality metrics across different K values."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        k_values = subtype_discovery["cluster_range"]
        silhouette_scores = subtype_discovery["silhouette_scores"]
        calinski_scores = subtype_discovery["calinski_scores"]
        optimal_k = subtype_discovery["optimal_k"]

        # Silhouette scores
        ax1.plot(k_values, silhouette_scores, 'bo-', linewidth=2, markersize=8, label='Silhouette Score')
        ax1.axvline(optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Clustering Quality vs Number of Clusters')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Calinski-Harabasz scores
        ax2.plot(k_values, calinski_scores, 'go-', linewidth=2, markersize=8, label='Calinski-Harabasz Score')
        ax2.axvline(optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Calinski-Harabasz Score')
        ax2.set_title('Calinski-Harabasz Score vs Number of Clusters')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_plot(plot_path)

    def _plot_subtype_factor_space(self, Z_sgfa: np.ndarray, cluster_labels: np.ndarray, plot_path: Path):
        """Plot subtype visualization in factor space using PCA and t-SNE."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

        # PCA visualization
        pca = PCA(n_components=2)
        Z_pca = pca.fit_transform(Z_sgfa)

        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            ax1.scatter(Z_pca[mask, 0], Z_pca[mask, 1], c=[colors[i]],
                       label=f'Subtype {label+1}', alpha=0.7, s=50)

        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.set_title('PD Subtypes in PCA Space')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # t-SNE visualization
        if Z_sgfa.shape[0] > 30:  # Only run t-SNE if enough samples
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, Z_sgfa.shape[0]//4))
            Z_tsne = tsne.fit_transform(Z_sgfa)

            for i, label in enumerate(unique_labels):
                mask = cluster_labels == label
                ax2.scatter(Z_tsne[mask, 0], Z_tsne[mask, 1], c=[colors[i]],
                           label=f'Subtype {label+1}', alpha=0.7, s=50)

            ax2.set_xlabel('t-SNE Component 1')
            ax2.set_ylabel('t-SNE Component 2')
            ax2.set_title('PD Subtypes in t-SNE Space')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Insufficient samples\nfor t-SNE visualization',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('t-SNE Visualization (Insufficient Data)')

        plt.tight_layout()
        save_plot(plot_path)

    def _plot_clinical_validation_bubble(self, clinical_measures: Dict, plot_path: Path):
        """Plot clinical validation results as bubble plot."""
        if not clinical_measures:
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        measures = list(clinical_measures.keys())
        p_values = [clinical_measures[measure]["anova_p"] for measure in measures]
        effect_sizes = [clinical_measures[measure]["eta_squared"] for measure in measures]

        # Create bubble plot
        colors = ['red' if p < 0.05 else 'blue' for p in p_values]
        sizes = [max(50, es * 1000) for es in effect_sizes]

        scatter = ax.scatter(range(len(measures)), [-np.log10(p) for p in p_values],
                           c=colors, s=sizes, alpha=0.6)

        ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p = 0.05')
        ax.set_xlabel('Clinical Measures')
        ax.set_ylabel('-log10(p-value)')
        ax.set_title('Clinical Validation of Discovered PD Subtypes')
        ax.set_xticks(range(len(measures)))
        ax.set_xticklabels(measures, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_plot(plot_path)

    def _plot_pd_subtype_characterization(
        self,
        results: Dict,
        Z_sgfa: np.ndarray,
        clinical_data: Union[Dict, pd.DataFrame],
        plot_path: Path
    ):
        """Create comprehensive PD subtype characterization panel."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        cluster_labels = results["subtype_discovery"]["best_solution"]["labels"]
        unique_labels = np.unique(cluster_labels)

        # Convert clinical_data to DataFrame if needed
        if isinstance(clinical_data, dict):
            clinical_df = pd.DataFrame(clinical_data)
        else:
            clinical_df = clinical_data

        # Align data sizes
        n_imaging = len(cluster_labels)
        n_clinical = len(clinical_df)
        clinical_aligned = clinical_df.iloc[:n_imaging] if n_clinical > n_imaging else clinical_df

        # Plot 1: Age distribution by subtype
        age_col = 'age' if 'age' in clinical_aligned.columns else clinical_aligned.columns[0]
        if age_col in clinical_aligned.columns:
            age_data = [clinical_aligned[age_col][cluster_labels == label].dropna()
                       for label in unique_labels]
            axes[0].boxplot(age_data, labels=[f'Subtype {label+1}' for label in unique_labels])
            axes[0].set_title('Age Distribution by PD Subtype')
            axes[0].set_ylabel('Age (years)')
            axes[0].grid(True, alpha=0.3)

        # Plot 2: Clinical severity by subtype (if UPDRS available)
        motor_cols = [col for col in clinical_aligned.columns
                     if 'updrs' in col.lower() or 'motor' in col.lower()]
        if motor_cols:
            motor_col = motor_cols[0]
            motor_data = [clinical_aligned[motor_col][cluster_labels == label].dropna()
                         for label in unique_labels]
            axes[1].boxplot(motor_data, labels=[f'Subtype {label+1}' for label in unique_labels])
            axes[1].set_title(f'{motor_col} by PD Subtype')
            axes[1].set_ylabel('Clinical Score')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No motor scores\navailable', ha='center', va='center',
                        transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title('Motor Scores by Subtype (N/A)')

        # Plot 3: Subtype stability heatmap
        if "stability_analysis" in results and "error" not in results["stability_analysis"]:
            stability = results["stability_analysis"]
            ari_scores = stability.get("ari_scores", [])
            if ari_scores:
                # Create pseudo heatmap of stability
                stability_matrix = np.full((len(unique_labels), len(unique_labels)), np.mean(ari_scores))
                np.fill_diagonal(stability_matrix, 1.0)

                im = axes[2].imshow(stability_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
                axes[2].set_title(f'Subtype Stability (Mean ARI: {np.mean(ari_scores):.3f})')
                axes[2].set_xlabel('Subtype')
                axes[2].set_ylabel('Subtype')
                axes[2].set_xticks(range(len(unique_labels)))
                axes[2].set_yticks(range(len(unique_labels)))
                axes[2].set_xticklabels([f'S{label+1}' for label in unique_labels])
                axes[2].set_yticklabels([f'S{label+1}' for label in unique_labels])
                plt.colorbar(im, ax=axes[2])
        else:
            axes[2].text(0.5, 0.5, 'Stability analysis\nnot available', ha='center', va='center',
                        transform=axes[2].transAxes, fontsize=12)
            axes[2].set_title('Subtype Stability (N/A)')

        # Plot 4: Factor importance by subtype
        if "factor_interpretation" in results:
            factor_interp = results["factor_interpretation"]

            # Create factor importance matrix
            n_factors = Z_sgfa.shape[1]
            importance_matrix = np.zeros((len(unique_labels), n_factors))

            for i, label in enumerate(unique_labels):
                key = f"subtype_{label}"
                if key in factor_interp:
                    importance = factor_interp[key].get("factor_importance", [])
                    if len(importance) == n_factors:
                        importance_matrix[i] = importance

            if np.any(importance_matrix):
                im = axes[3].imshow(importance_matrix, cmap='viridis', aspect='auto')
                axes[3].set_title('Factor Importance by Subtype')
                axes[3].set_xlabel('Factor')
                axes[3].set_ylabel('Subtype')
                axes[3].set_xticks(range(n_factors))
                axes[3].set_yticks(range(len(unique_labels)))
                axes[3].set_xticklabels([f'F{i+1}' for i in range(n_factors)])
                axes[3].set_yticklabels([f'S{label+1}' for label in unique_labels])
                plt.colorbar(im, ax=axes[3])

        plt.tight_layout()
        save_plot(plot_path)

    def _plot_subtype_discovery_summary(self, results: Dict, plot_path: Path):
        """Create subtype discovery pipeline summary visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

        subtype_discovery = results["subtype_discovery"]
        optimal_k = subtype_discovery["optimal_k"]
        best_silhouette = subtype_discovery["best_solution"]["silhouette_score"]
        cluster_labels = subtype_discovery["best_solution"]["labels"]
        unique_labels = np.unique(cluster_labels)

        # Plot 1: Discovery pipeline flow
        ax1.text(0.5, 0.8, 'PD Subtype Discovery Pipeline', ha='center', va='center',
                transform=ax1.transAxes, fontsize=16, fontweight='bold')
        ax1.text(0.5, 0.6, f'SGFA Factors → Clustering → {optimal_k} Subtypes', ha='center', va='center',
                transform=ax1.transAxes, fontsize=12)
        ax1.text(0.5, 0.4, f'Silhouette Score: {best_silhouette:.3f}', ha='center', va='center',
                transform=ax1.transAxes, fontsize=12)

        if "clinical_validation" in results:
            cv_score = results["clinical_validation"]["validation_score"]
            ax1.text(0.5, 0.2, f'Clinical Validation: {cv_score:.3f}', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=12)

        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')

        # Plot 2: Subtype sizes
        subtype_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        wedges, texts, autotexts = ax2.pie(subtype_sizes, labels=[f'Subtype {label+1}' for label in unique_labels],
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('PD Subtype Distribution')

        # Plot 3: Quality metrics comparison
        metrics = ['Silhouette', 'Calinski-H']
        scores = [best_silhouette, subtype_discovery["best_solution"].get("calinski_score", 0)]

        bars = ax3.bar(metrics, scores, color=['skyblue', 'lightcoral'])
        ax3.set_title('Clustering Quality Metrics')
        ax3.set_ylabel('Score')
        ax3.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{score:.3f}', ha='center', va='bottom')

        # Plot 4: Clinical significance summary
        if "clinical_validation" in results:
            cv = results["clinical_validation"]
            sig_measures = cv["significant_measures"]
            total_measures = cv["total_measures"]

            # Create donut chart for clinical validation
            sizes = [sig_measures, total_measures - sig_measures]
            colors = ['green', 'lightgray']
            labels = [f'Significant ({sig_measures})', f'Non-significant ({total_measures - sig_measures})']

            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.0f', colors=colors,
                                              startangle=90, wedgeprops=dict(width=0.5))
            ax4.set_title(f'Clinical Validation\n({sig_measures}/{total_measures} measures significant)')

        plt.tight_layout()
        save_plot(plot_path)

    def _plot_integrated_performance_overview(self, integrated_results: Dict, plot_path: Path):
        """Plot integrated performance overview across K values."""
        successful_runs = {k: v for k, v in integrated_results.items() if "error" not in v}
        if not successful_runs:
            return

        # Extract data for plotting
        k_values = [int(k[1:]) for k in successful_runs.keys()]  # Remove 'K' prefix
        sgfa_times = [successful_runs[f"K{k}"]["sgfa_time_seconds"] for k in k_values]
        silhouette_scores = [successful_runs[f"K{k}"]["best_silhouette_score"] for k in k_values]
        clinical_scores = [successful_runs[f"K{k}"]["clinical_separation_score"] for k in k_values]
        memory_usage = [successful_runs[f"K{k}"]["sgfa_memory_mb"] for k in k_values]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # SGFA Performance vs K
        ax1.plot(k_values, sgfa_times, 'bo-', linewidth=2, markersize=8, label='SGFA Time')
        ax1.set_xlabel('Number of Factors (K)')
        ax1.set_ylabel('SGFA Time (seconds)')
        ax1.set_title('SGFA Performance Scaling')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Subtype Discovery Quality vs K
        ax2.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8, label='Silhouette Score')
        ax2.set_xlabel('Number of Factors (K)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('PD Subtype Discovery Quality')
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Good Quality Threshold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Clinical Translation vs K
        ax3.plot(k_values, clinical_scores, 'ro-', linewidth=2, markersize=8, label='Clinical Separation')
        ax3.set_xlabel('Number of Factors (K)')
        ax3.set_ylabel('Clinical Separation Score')
        ax3.set_title('Clinical Translation Quality')
        ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Strong Validation Threshold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Memory Usage vs K
        ax4.plot(k_values, memory_usage, 'mo-', linewidth=2, markersize=8, label='Memory Usage')
        ax4.set_xlabel('Number of Factors (K)')
        ax4.set_ylabel('Memory Usage (MB)')
        ax4.set_title('Memory Scaling')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()
        save_plot(plot_path)

    def _plot_performance_discovery_tradeoffs(self, integrated_results: Dict, plot_path: Path):
        """Plot performance vs discovery quality trade-offs."""
        successful_runs = {k: v for k, v in integrated_results.items() if "error" not in v}
        if not successful_runs:
            return

        k_values = [int(k[1:]) for k in successful_runs.keys()]
        sgfa_times = [successful_runs[f"K{k}"]["sgfa_time_seconds"] for k in k_values]
        silhouette_scores = [successful_runs[f"K{k}"]["best_silhouette_score"] for k in k_values]
        clinical_scores = [successful_runs[f"K{k}"]["clinical_separation_score"] for k in k_values]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Speed vs Quality Scatter
        ax1.scatter(sgfa_times, silhouette_scores, c=k_values, s=100, cmap='viridis', alpha=0.7)
        for i, k in enumerate(k_values):
            ax1.annotate(f'K={k}', (sgfa_times[i], silhouette_scores[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax1.set_xlabel('SGFA Time (seconds)')
        ax1.set_ylabel('Subtype Discovery Quality (Silhouette)')
        ax1.set_title('Speed vs Quality Trade-off')
        ax1.grid(True, alpha=0.3)

        # Quality vs Clinical Validation Scatter
        ax2.scatter(silhouette_scores, clinical_scores, c=k_values, s=100, cmap='viridis', alpha=0.7)
        for i, k in enumerate(k_values):
            ax2.annotate(f'K={k}', (silhouette_scores[i], clinical_scores[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax2.set_xlabel('Subtype Discovery Quality (Silhouette)')
        ax2.set_ylabel('Clinical Validation Score')
        ax2.set_title('Discovery Quality vs Clinical Translation')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_plot(plot_path)

    def _plot_pd_research_decision_matrix(self, integrated_results: Dict, plot_path: Path):
        """Create PD research decision matrix for optimal K selection."""
        successful_runs = {k: v for k, v in integrated_results.items() if "error" not in v}
        if not successful_runs:
            return

        k_values = [int(k[1:]) for k in successful_runs.keys()]
        sgfa_times = [successful_runs[f"K{k}"]["sgfa_time_seconds"] for k in k_values]
        silhouette_scores = [successful_runs[f"K{k}"]["best_silhouette_score"] for k in k_values]
        clinical_scores = [successful_runs[f"K{k}"]["clinical_separation_score"] for k in k_values]

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Normalize metrics for comparison
        norm_times = np.array(sgfa_times) / np.max(sgfa_times)  # Lower is better
        norm_silhouette = np.array(silhouette_scores)  # Higher is better
        norm_clinical = np.array(clinical_scores)  # Higher is better

        # Calculate composite scores
        speed_scores = 1 - norm_times  # Convert to "higher is better"
        research_quality_scores = (norm_silhouette + norm_clinical) / 2

        # Create quadrant plot
        ax.scatter(speed_scores, research_quality_scores, s=200, c=k_values,
                  cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)

        # Add K labels
        for i, k in enumerate(k_values):
            ax.annotate(f'K={k}', (speed_scores[i], research_quality_scores[i]),
                       xytext=(8, 8), textcoords='offset points', fontsize=12, fontweight='bold')

        # Add quadrant lines
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

        # Add quadrant labels
        ax.text(0.25, 0.75, 'High Quality\nSlow Performance', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax.text(0.75, 0.75, 'HIGH QUALITY\nFAST PERFORMANCE\n(OPTIMAL)', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7), fontweight='bold')
        ax.text(0.25, 0.25, 'Low Quality\nSlow Performance', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        ax.text(0.75, 0.25, 'Low Quality\nFast Performance', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

        ax.set_xlabel('Performance Efficiency (Normalized)', fontsize=12)
        ax.set_ylabel('PD Research Quality (Subtype + Clinical)', fontsize=12)
        ax.set_title('PD Research Decision Matrix:\nChoosing Optimal K for Subtype Discovery',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add colorbar for K values
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Number of Factors (K)', fontsize=12)

        # Add research recommendation
        best_idx = np.argmax(speed_scores + research_quality_scores)  # Simple composite score
        best_k = k_values[best_idx]
        ax.text(0.02, 0.98, f'🎯 Recommended for PD Research: K={best_k}',
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="gold", alpha=0.8))

        plt.tight_layout()
        save_plot(plot_path)