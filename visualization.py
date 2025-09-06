import pickle
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import gcf
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway, ttest_ind
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging
import contextlib
from pathlib import Path
import gc

try:
    from qmap_gfa_weight2mri_python import add_to_qmap_visualization
    FACTOR_MAPPING_AVAILABLE = True
except ImportError:
    FACTOR_MAPPING_AVAILABLE = False

logging.captureWarnings(True)

# == UTILITY FUNCTIONS ==
@contextlib.contextmanager
def safe_plotting_context(figsize=None, dpi=300):
    """Context manager for safe plotting with automatic cleanup."""
    import matplotlib.pyplot as plt
    
    original_backend = plt.get_backend()
    
    if figsize:
        plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    
    try:
        yield plt
    except Exception as e:
        logging.error(f"Plotting error: {e}")
        raise
    finally:
        plt.close('all')
        gc.collect()
        try:
            plt.switch_backend(original_backend)
        except:
            pass

def save_plot_safely(fig, filepath, formats=['png', 'pdf'], **kwargs):
    """Save plot in multiple formats with error handling."""
    from pathlib import Path
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    save_params = {
        'dpi': 300,
        'bbox_inches': 'tight',
        'pad_inches': 0.1,
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    save_params.update(kwargs)
    
    saved_files = []
    for fmt in formats:
        try:
            save_path = filepath.with_suffix(f'.{fmt}')
            fig.savefig(save_path, format=fmt, **save_params)
            saved_files.append(str(save_path))
            logging.debug(f"Saved plot: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save plot as {fmt}: {e}")
    
    return saved_files

def setup_plotting_style():
    """Set up consistent plotting style across all visualizations."""
    import matplotlib.pyplot as plt
    
    style_params = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "figure.subplot.hspace": 0.8,  
        "figure.subplot.wspace": 0.5,
        "figure.constrained_layout.use": True,
        "axes.formatter.use_mathtext": True,
    }
    
    plt.rcParams.update(style_params)

def visualization_with_error_handling(func):
    """Decorator for visualization functions to add error handling."""
    def wrapper(*args, **kwargs):
        try:
            setup_plotting_style()
            result = func(*args, **kwargs)
            gc.collect()
            return result
        except Exception as e:
            logging.error(f"Visualization function {func.__name__} failed: {e}")
            logging.exception("Full traceback:")
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
                gc.collect()
            except:
                pass
            raise
    return wrapper

setup_plotting_style()

# Color schemes
COLORS = {
    'primary': '#2E86C1',
    'secondary': '#E74C3C', 
    'tertiary': '#28B463',
    'quaternary': '#F39C12',
    'neutral': '#5D6D7E',
    'light_gray': '#BDC3C7',
    'groups': ['#3498DB', '#E74C3C', '#28B463', '#F39C12', '#9B59B6', '#E67E22']
}

# == PREPROCESSING VISUALIZATION ==

def plot_preprocessing_summary(preprocessing_results, plot_path, view_names):
    """
    Create comprehensive preprocessing summary plots.
    """
    os.makedirs(f"{plot_path}/preprocessing", exist_ok=True)
    
    if 'feature_reduction' not in preprocessing_results:
        logging.warning("No feature reduction data found in preprocessing results")
        return
    
    # Feature reduction bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    views = list(preprocessing_results['feature_reduction'].keys())
    original_counts = [preprocessing_results['feature_reduction'][v]['original'] for v in views]
    processed_counts = [preprocessing_results['feature_reduction'][v]['processed'] for v in views]
    
    x = np.arange(len(views))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, original_counts, width, label='Original', 
                    color=COLORS['primary'], alpha=0.7)
    bars2 = ax1.bar(x + width/2, processed_counts, width, label='After Preprocessing',
                    color=COLORS['secondary'], alpha=0.7)
    
    ax1.set_xlabel('Data Views')
    ax1.set_ylabel('Number of Features')
    ax1.set_title('Feature Count Before/After Preprocessing', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(views, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Reduction ratio plot
    reduction_ratios = [preprocessing_results['feature_reduction'][v]['reduction_ratio'] for v in views]
    colors = [COLORS['tertiary'] if ratio > 0.5 else COLORS['quaternary'] for ratio in reduction_ratios]
    
    bars = ax2.bar(views, reduction_ratios, color=colors, alpha=0.7)
    ax2.set_xlabel('Data Views')
    ax2.set_ylabel('Feature Retention Ratio')
    ax2.set_title('Feature Retention After Preprocessing', fontweight='bold')
    ax2.set_xticklabels(views, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # Add percentage labels
    for i, (bar, ratio) in enumerate(zip(bars, reduction_ratios)):
        ax2.text(bar.get_x() + bar.get_width()/2., ratio + 0.02,
                f'{ratio:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Preprocessing Impact Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{plot_path}/preprocessing/feature_reduction_summary.png")
    plt.savefig(f"{plot_path}/preprocessing/feature_reduction_summary.pdf")
    plt.close()
    
    # Source combination validation plot (if available)
    if 'source_validation' in preprocessing_results:
        _plot_source_validation_results(preprocessing_results['source_validation'], plot_path)
    
    # Optimization results plot (if available)
    if 'optimization' in preprocessing_results:
        _plot_optimization_results(preprocessing_results['optimization'], plot_path)

def _plot_source_validation_results(source_validation, plot_path):
    """Plot cross-validation results for different source combinations."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by RMSE performance
    sorted_combos = sorted(source_validation.items(), key=lambda x: x[1]['rmse_mean'])
    
    combo_names = [combo for combo, _ in sorted_combos]
    rmse_means = [results['rmse_mean'] for _, results in sorted_combos]
    rmse_stds = [results['rmse_std'] for _, results in sorted_combos]
    n_features = [results['n_features'] for _, results in sorted_combos]
    
    # Create color map based on number of features
    norm = plt.Normalize(min(n_features), max(n_features))
    colors = plt.cm.viridis(norm(n_features))
    
    # Horizontal bar plot with error bars
    bars = ax.barh(range(len(combo_names)), rmse_means, xerr=rmse_stds,
                   color=colors, alpha=0.7, capsize=3)
    
    ax.set_yticks(range(len(combo_names)))
    ax.set_yticklabels(combo_names, fontsize=9)
    ax.set_xlabel('RMSE (Cross-Validation)')
    ax.set_title('Source Combination Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add feature count annotations
    for i, (bar, n_feat) in enumerate(zip(bars, n_features)):
        width = bar.get_width()
        ax.text(width + rmse_stds[i] + max(rmse_means)*0.01, bar.get_y() + bar.get_height()/2,
                f'{n_feat} feat.', ha='left', va='center', fontsize=8)
    
    # Add colorbar for feature count
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Number of Features', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(f"{plot_path}/preprocessing/source_validation_results.png")
    plt.savefig(f"{plot_path}/preprocessing/source_validation_results.pdf")
    plt.close()

def _plot_optimization_results(optimization_results, plot_path):
    """Plot preprocessing parameter optimization results."""
    if 'all_results' not in optimization_results:
        return
    
    results = optimization_results['all_results']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    # Extract parameter values and scores
    imputation_methods = [r['params']['imputation_strategy'] for r in results]
    selection_methods = [r['params']['feature_selection_method'] for r in results]
    scores = [r['score'] for r in results]
    feature_counts = [r['n_features_total'] for r in results]
    
    # Plot 1: Score by imputation method
    imputation_unique = list(set(imputation_methods))
    imputation_scores = {method: [] for method in imputation_unique}
    for method, score in zip(imputation_methods, scores):
        imputation_scores[method].append(score)
    
    box_data = [imputation_scores[method] for method in imputation_unique]
    bp1 = axes[0].boxplot(box_data, labels=imputation_unique, patch_artist=True)
    for patch, color in zip(bp1['boxes'], COLORS['groups']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0].set_xlabel('Imputation Method')
    axes[0].set_ylabel('Optimization Score')
    axes[0].set_title('Performance by Imputation Strategy', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Score by feature selection method
    selection_unique = list(set(selection_methods))
    selection_scores = {method: [] for method in selection_unique}
    for method, score in zip(selection_methods, scores):
        selection_scores[method].append(score)
    
    box_data = [selection_scores[method] for method in selection_unique]
    bp2 = axes[1].boxplot(box_data, labels=selection_unique, patch_artist=True)
    for patch, color in zip(bp2['boxes'], COLORS['groups']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1].set_xlabel('Feature Selection Method')
    axes[1].set_ylabel('Optimization Score')
    axes[1].set_title('Performance by Feature Selection', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Score vs feature count
    scatter = axes[2].scatter(feature_counts, scores, c=scores, cmap='viridis', 
                             alpha=0.7, s=50)
    axes[2].set_xlabel('Total Number of Features')
    axes[2].set_ylabel('Optimization Score')
    axes[2].set_title('Score vs Feature Count Trade-off', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=axes[2])
    cbar.set_label('Score')
    
    # Plot 4: Parameter combination heatmap (simplified)
    param_matrix = np.zeros((len(imputation_unique), len(selection_unique)))
    for i, imp_method in enumerate(imputation_unique):
        for j, sel_method in enumerate(selection_unique):
            matching_scores = [score for imp, sel, score in zip(imputation_methods, selection_methods, scores)
                             if imp == imp_method and sel == sel_method]
            if matching_scores:
                param_matrix[i, j] = np.mean(matching_scores)
            else:
                param_matrix[i, j] = np.nan
    
    im = axes[3].imshow(param_matrix, aspect='auto', cmap='viridis')
    axes[3].set_xticks(range(len(selection_unique)))
    axes[3].set_xticklabels(selection_unique, rotation=45, ha='right')
    axes[3].set_yticks(range(len(imputation_unique)))
    axes[3].set_yticklabels(imputation_unique)
    axes[3].set_xlabel('Feature Selection Method')
    axes[3].set_ylabel('Imputation Method')
    axes[3].set_title('Parameter Combination Heatmap', fontweight='bold')
    
    # Add text annotations
    for i in range(len(imputation_unique)):
        for j in range(len(selection_unique)):
            if not np.isnan(param_matrix[i, j]):
                axes[3].text(j, i, f'{param_matrix[i, j]:.2f}', 
                           ha='center', va='center', fontsize=8, color='white')
    
    plt.suptitle('Preprocessing Parameter Optimization Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{plot_path}/preprocessing/optimization_results.png")
    plt.savefig(f"{plot_path}/preprocessing/optimization_results.pdf")
    plt.close()

# == CROSS-VALIDATION VISUALIZATION ==

def plot_cv_results(cv_results, plot_path, run_name="cross_validation"):
    """
    Create comprehensive cross-validation results visualization.
    """
    os.makedirs(f"{plot_path}/cross_validation", exist_ok=True)
    
    if 'fold_results' not in cv_results:
        logging.warning("No fold results found in CV results")
        return
    
    fold_results = [r for r in cv_results['fold_results'] if r.get('converged', False)]
    if not fold_results:
        logging.warning("No converged folds found")
        return
    
    # Extract metrics across folds
    fold_scores = [r['recon_metrics']['overall']['mean_r2'] for r in fold_results]
    fold_ids = [r['fold_id'] for r in fold_results]
    fit_times = [r.get('fit_time', 0) for r in fold_results]
    
    # Create comprehensive CV summary
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    # Plot 1: CV scores across folds
    bars = axes[0].bar(fold_ids, fold_scores, color=COLORS['primary'], alpha=0.7)
    axes[0].axhline(np.mean(fold_scores), color=COLORS['secondary'], 
                    linestyle='--', linewidth=2, label=f'Mean: {np.mean(fold_scores):.3f}')
    axes[0].set_xlabel('Fold ID')
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Cross-Validation Scores by Fold', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, fold_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2., score + max(fold_scores)*0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Score distribution
    axes[1].hist(fold_scores, bins=min(len(fold_scores), 10), color=COLORS['tertiary'], 
                 alpha=0.7, density=True)
    axes[1].axvline(np.mean(fold_scores), color=COLORS['secondary'], 
                    linestyle='--', linewidth=2, label='Mean')
    axes[1].axvline(np.median(fold_scores), color=COLORS['quaternary'], 
                    linestyle=':', linewidth=2, label='Median')
    axes[1].set_xlabel('R² Score')
    axes[1].set_ylabel('Density')
    axes[1].set_title('CV Score Distribution', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Computational time
    axes[2].bar(fold_ids, fit_times, color=COLORS['quaternary'], alpha=0.7)
    axes[2].set_xlabel('Fold ID')
    axes[2].set_ylabel('Fit Time (seconds)')
    axes[2].set_title('Computational Time per Fold', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: View-specific performance (if available)
    if fold_results[0]['recon_metrics'] and len([k for k in fold_results[0]['recon_metrics'].keys() if k.startswith('view_')]) > 1:
        view_names = [k for k in fold_results[0]['recon_metrics'].keys() if k.startswith('view_')]
        view_scores = {view: [] for view in view_names}
        
        for result in fold_results:
            for view in view_names:
                if view in result['recon_metrics']:
                    view_scores[view].append(result['recon_metrics'][view]['r2'])
        
        # Box plot of view-specific scores
        box_data = [view_scores[view] for view in view_names]
        bp = axes[3].boxplot(box_data, labels=[v.replace('view_', 'View ') for v in view_names], 
                            patch_artist=True)
        for patch, color in zip(bp['boxes'], COLORS['groups']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[3].set_xlabel('Data Views')
        axes[3].set_ylabel('R² Score')
        axes[3].set_title('View-Specific Performance', fontweight='bold')
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].text(0.5, 0.5, 'View-specific\nmetrics not available', 
                    ha='center', va='center', transform=axes[3].transAxes,
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[3].set_title('View-Specific Performance', fontweight='bold')
    
    # Plot 5: Stability metrics (if available)
    if 'stability_metrics' in cv_results and cv_results['stability_metrics']:
        stability = cv_results['stability_metrics']
        metrics = ['loading_stability_mean', 'score_stability_mean']
        values = [stability.get(m, 0) for m in metrics]
        errors = [stability.get(m.replace('_mean', '_std'), 0) for m in metrics]
        labels = ['Loading\nStability', 'Score\nStability']
        
        bars = axes[4].bar(labels, values, yerr=errors, capsize=5,
                          color=[COLORS['primary'], COLORS['secondary']], alpha=0.7)
        axes[4].set_ylabel('Correlation')
        axes[4].set_title('Factor Stability Across Folds', fontweight='bold')
        axes[4].set_ylim(0, 1)
        axes[4].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val, err in zip(bars, values, errors):
            axes[4].text(bar.get_x() + bar.get_width()/2., val + err + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        axes[4].text(0.5, 0.5, 'Stability metrics\nnot available', 
                    ha='center', va='center', transform=axes[4].transAxes,
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[4].set_title('Factor Stability', fontweight='bold')
    
    # Plot 6: Summary statistics
    stats_text = f"""CV Summary Statistics:
    
Mean CV Score: {np.mean(fold_scores):.4f}
Std CV Score: {np.std(fold_scores):.4f}
Min CV Score: {np.min(fold_scores):.4f}
Max CV Score: {np.max(fold_scores):.4f}

Converged Folds: {len(fold_results)}/{len(cv_results.get('fold_results', []))}
Total Time: {cv_results.get('total_time', 0):.1f}s
Mean Fit Time: {np.mean(fit_times):.1f}s
"""
    
    axes[5].text(0.1, 0.9, stats_text, transform=axes[5].transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    axes[5].set_xlim(0, 1)
    axes[5].set_ylim(0, 1)
    axes[5].axis('off')
    axes[5].set_title('Summary Statistics', fontweight='bold')
    
    plt.suptitle(f'Cross-Validation Results: {run_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{plot_path}/cross_validation/cv_summary.png")
    plt.savefig(f"{plot_path}/cross_validation/cv_summary.pdf")
    plt.close()
    
    # Create factor stability heatmap if data available
    if len(fold_results) >= 2:
        _plot_factor_stability_heatmap(fold_results, plot_path)

def _plot_factor_stability_heatmap(fold_results, plot_path):
    """Create heatmap showing factor stability across CV folds."""
    # Extract factor loadings from each fold
    W_matrices = [r['W'] for r in fold_results if 'W' in r]
    
    if len(W_matrices) < 2:
        return
    
    # Compute pairwise correlations between folds
    n_folds = len(W_matrices)
    n_factors = W_matrices[0].shape[1] if W_matrices else 0
    
    if n_factors == 0:
        return
    
    # Create correlation matrix for each factor
    fig, axes = plt.subplots(1, min(n_factors, 4), figsize=(4*min(n_factors, 4), 4))
    if min(n_factors, 4) == 1:
        axes = [axes]
    
    for k in range(min(n_factors, 4)):  # Show first 4 factors max
        corr_matrix = np.ones((n_folds, n_folds))
        
        for i in range(n_folds):
            for j in range(i+1, n_folds):
                if (W_matrices[i].shape == W_matrices[j].shape and 
                    k < W_matrices[i].shape[1] and k < W_matrices[j].shape[1]):
                    corr = np.corrcoef(W_matrices[i][:, k], W_matrices[j][:, k])[0, 1]
                    corr = abs(corr) if not np.isnan(corr) else 0
                    corr_matrix[i, j] = corr_matrix[j, i] = corr
        
        im = axes[k].imshow(corr_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[k].set_title(f'Factor {k+1}\nStability', fontweight='bold')
        axes[k].set_xlabel('Fold')
        axes[k].set_ylabel('Fold')
        
        # Add correlation values as text
        for i in range(n_folds):
            for j in range(n_folds):
                text = axes[k].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="white" if corr_matrix[i, j] < 0.5 else "black")
    
    # Add colorbar
    if len(axes) > 0:
        cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.1)
        cbar.set_label('Absolute Correlation', rotation=270, labelpad=15)
    
    plt.suptitle('Factor Loading Stability Across CV Folds', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{plot_path}/cross_validation/factor_stability_heatmap.png")
    plt.savefig(f"{plot_path}/cross_validation/factor_stability_heatmap.pdf")
    plt.close()

# == CONSENSUS SUBTYPE VISUALIZATION ==

def plot_consensus_subtypes(centroids_data, probabilities_data, plot_path):
    """
    Create consensus subtype visualization plots.
    """
    os.makedirs(f"{plot_path}/consensus", exist_ok=True)
    
    # Load centroids data
    if isinstance(centroids_data, (str, Path)):
        with open(centroids_data, 'r') as f:
            centroids_obj = json.load(f)
    else:
        centroids_obj = centroids_data
    
    C = np.array(centroids_obj.get('centroids', []))
    if C.size == 0:
        logging.warning("No centroids data available")
        return
    
    # Load probabilities data
    if isinstance(probabilities_data, (str, Path)):
        prob_df = pd.read_csv(probabilities_data)
    else:
        prob_df = probabilities_data
    
    # Create comprehensive consensus plots
    _plot_centroids_heatmap(C, plot_path)
    _plot_centroids_radar(C, plot_path)
    _plot_subtype_probabilities(prob_df, plot_path)
    _plot_subtype_assignments(prob_df, plot_path)

def _plot_centroids_heatmap(C, plot_path):
    """Create centroids heatmap."""
    fig, ax = plt.subplots(figsize=(max(8, C.shape[1]*0.6), max(6, C.shape[0]*0.8)))
    
    # Create heatmap with diverging colormap
    vmax = np.max(np.abs(C))
    im = ax.imshow(C, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    
    ax.set_xlabel("Latent Factors")
    ax.set_ylabel("Consensus Subtypes")
    ax.set_xticks(range(C.shape[1]))
    ax.set_xticklabels([f"LF{j+1}" for j in range(C.shape[1])])
    ax.set_yticks(range(C.shape[0]))
    ax.set_yticklabels([f"Subtype {j+1}" for j in range(C.shape[0])])
    ax.set_title("Consensus Subtype Centroids", fontsize=14, fontweight='bold')
    
    # Add colorbar
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Centroid Value (Standardized)", rotation=270, labelpad=15)
    
    # Add text annotations for significant values
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if abs(C[i, j]) > vmax * 0.5:  # Show values > 50% of max
                ax.text(j, i, f'{C[i, j]:.2f}', ha='center', va='center',
                       color='white' if abs(C[i, j]) > vmax * 0.7 else 'black',
                       fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{plot_path}/consensus/centroids_heatmap.png")
    plt.savefig(f"{plot_path}/consensus/centroids_heatmap.pdf")
    plt.close()

def _plot_centroids_radar(C, plot_path):
    """Create radar plots for each subtype."""
    n_subtypes = C.shape[0]
    n_factors = C.shape[1]
    
    # Create subplot for each subtype
    cols = min(3, n_subtypes)
    rows = (n_subtypes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), subplot_kw=dict(projection='polar'))
    
    if n_subtypes == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.ravel()
    
    # Angles for radar chart
    angles = np.linspace(0, 2*np.pi, n_factors, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i in range(n_subtypes):
        ax = axes[i]
        values = C[i].tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, color=COLORS['groups'][i % len(COLORS['groups'])], 
                label=f'Subtype {i+1}')
        ax.fill(angles, values, alpha=0.25, color=COLORS['groups'][i % len(COLORS['groups'])])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f'LF{j+1}' for j in range(n_factors)])
        ax.set_title(f'Subtype {i+1}', fontweight='bold', pad=20)
        ax.grid(True)
        
        # Set consistent y-axis limits
        ax.set_ylim(-np.max(np.abs(C)), np.max(np.abs(C)))
    
    # Hide empty subplots
    for i in range(n_subtypes, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Consensus Subtype Profiles', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{plot_path}/consensus/centroids_radar.png")
    plt.savefig(f"{plot_path}/consensus/centroids_radar.pdf")
    plt.close()

def _plot_subtype_probabilities(prob_df, plot_path):
    """Create subtype probability distribution plots."""
    prob_cols = [c for c in prob_df.columns if c.startswith("prob_subtype_")]
    n_subtypes = len(prob_cols)
    
    if n_subtypes == 0:
        logging.warning("No subtype probability columns found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    # Plot 1: Assignment confidence (max probability)
    max_probs = prob_df[prob_cols].max(axis=1).values
    axes[0].hist(max_probs, bins=30, color=COLORS['primary'], alpha=0.7, density=True)
    axes[0].axvline(np.mean(max_probs), color=COLORS['secondary'], 
                    linestyle='--', linewidth=2, label=f'Mean: {np.mean(max_probs):.3f}')
    axes[0].axvline(np.median(max_probs), color=COLORS['tertiary'], 
                    linestyle=':', linewidth=2, label=f'Median: {np.median(max_probs):.3f}')
    axes[0].set_xlabel('Maximum Subtype Probability')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Subtype Assignment Confidence', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Per-subtype probability distributions
    for k in range(min(n_subtypes, 4)):  # Show first 4 subtypes
        probs = prob_df[f"prob_subtype_{k}"].values
        axes[1].hist(probs, bins=20, alpha=0.6, 
                    color=COLORS['groups'][k % len(COLORS['groups'])], 
                    label=f'Subtype {k+1}', density=True)
    
    axes[1].set_xlabel('Subtype Probability')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Per-Subtype Probability Distributions', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Subtype assignment pie chart
    if 'hard_label' in prob_df.columns:
        hard_labels = prob_df['hard_label'].values
        counts = np.bincount(hard_labels, minlength=n_subtypes)
        colors = [COLORS['groups'][i % len(COLORS['groups'])] for i in range(n_subtypes)]
        
        wedges, texts, autotexts = axes[2].pie(counts, 
                                              labels=[f'Subtype {i+1}' for i in range(n_subtypes)],
                                              colors=colors,
                                              autopct='%1.1f%%',
                                              startangle=90)
        axes[2].set_title('Hard Subtype Assignments', fontweight='bold')
    
    # Plot 4: Uncertainty analysis
    entropy = -np.sum(prob_df[prob_cols].values * np.log(prob_df[prob_cols].values + 1e-10), axis=1)
    axes[3].scatter(max_probs, entropy, alpha=0.6, color=COLORS['quaternary'])
    axes[3].set_xlabel('Maximum Probability')
    axes[3].set_ylabel('Assignment Entropy')
    axes[3].set_title('Confidence vs Uncertainty', fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(max_probs, entropy)[0, 1]
    axes[3].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[3].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.suptitle('Subtype Assignment Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{plot_path}/consensus/subtype_probabilities.png")
    plt.savefig(f"{plot_path}/consensus/subtype_probabilities.pdf")
    plt.close()

def _plot_subtype_assignments(prob_df, plot_path):
    """Create stacked bar plot of subtype assignments."""
    prob_cols = [c for c in prob_df.columns if c.startswith("prob_subtype_")]
    probs_matrix = prob_df[prob_cols].values
    
    # Sort subjects by assignment confidence
    max_probs = np.max(probs_matrix, axis=1)
    sort_idx = np.argsort(max_probs)[::-1]
    probs_sorted = probs_matrix[sort_idx]
    
    # Sample subjects for visualization if too many
    n_show = min(100, len(probs_sorted))
    if len(probs_sorted) > n_show:
        step = len(probs_sorted) // n_show
        probs_show = probs_sorted[::step][:n_show]
    else:
        probs_show = probs_sorted
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Create stacked bar plot
    bottom = np.zeros(len(probs_show))
    colors = [COLORS['groups'][i % len(COLORS['groups'])] for i in range(len(prob_cols))]
    
    for k in range(len(prob_cols)):
        ax.bar(range(len(probs_show)), probs_show[:, k], 
               bottom=bottom, color=colors[k], alpha=0.8, 
               label=f'Subtype {k+1}')
        bottom += probs_show[:, k]
    
    ax.set_xlabel('Subjects (sorted by assignment confidence)')
    ax.set_ylabel('Subtype Probability')
    ax.set_title('Individual Subtype Assignments', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{plot_path}/consensus/subtype_assignments.png")
    plt.savefig(f"{plot_path}/consensus/subtype_assignments.pdf")
    plt.close()

def create_brain_visualization_summary(results_dir: str, include_reconstructions: bool = True):
    """
    Create a summary plot of all brain visualizations.
    """
    results_dir = Path(results_dir)
    
    # Look for brain visualization outputs
    brain_viz_dir = results_dir / "comprehensive_brain_visualization"
    factor_maps_dir = results_dir / "factor_maps"
    subject_recon_dir = results_dir / "subject_reconstructions"
    
    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Factor maps overview
    if factor_maps_dir.exists():
        factor_files = list(factor_maps_dir.glob("*.nii"))
        axes[0, 0].bar(['Factor Maps'], [len(factor_files)], color=COLORS['primary'])
        axes[0, 0].set_title('Factor Maps Created', fontweight='bold')
        axes[0, 0].set_ylabel('Number of Files')
        
        # Add text annotation
        axes[0, 0].text(0, len(factor_files) + 1, f'{len(factor_files)} NIfTI files',
                       ha='center', fontweight='bold')
    
    # Plot 2: Subject reconstructions overview  
    if subject_recon_dir.exists():
        recon_files = list(subject_recon_dir.glob("*.nii"))
        overlay_files = list((subject_recon_dir / "overlays").glob("*.png"))
        
        categories = ['Reconstructions', 'Overlays']
        counts = [len(recon_files), len(overlay_files)]
        
        axes[0, 1].bar(categories, counts, color=[COLORS['secondary'], COLORS['tertiary']])
        axes[0, 1].set_title('Subject Reconstructions', fontweight='bold')
        axes[0, 1].set_ylabel('Number of Files')
        
        for i, count in enumerate(counts):
            axes[0, 1].text(i, count + 1, str(count), ha='center', fontweight='bold')
    
    # Plot 3: File size analysis
    total_size = 0
    file_types = {}
    
    for viz_dir in [brain_viz_dir, factor_maps_dir, subject_recon_dir]:
        if viz_dir.exists():
            for file_path in viz_dir.rglob("*.*"):
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    total_size += size_mb
                    
                    ext = file_path.suffix.lower()
                    file_types[ext] = file_types.get(ext, 0) + size_mb
    
    if file_types:
        axes[1, 0].pie(file_types.values(), labels=file_types.keys(), autopct='%1.1f%%')
        axes[1, 0].set_title(f'File Types Distribution\n(Total: {total_size:.1f} MB)', fontweight='bold')
    
    # Plot 4: Processing summary
    summary_text = f"""Brain Visualization Summary:

Total Output Size: {total_size:.1f} MB

Directories Created:
- Factor Maps: {'✓' if factor_maps_dir.exists() else '✗'}
- Subject Reconstructions: {'✓' if subject_recon_dir.exists() else '✗'}
- Comprehensive Viz: {'✓' if brain_viz_dir.exists() else '✗'}

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
"""
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    axes[1, 1].axis('off')
    
    plt.suptitle('Brain Visualization Summary Report', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    summary_file = results_dir / "brain_visualization_summary.png"
    plt.savefig(summary_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Brain visualization summary saved: {summary_file}")
    return str(summary_file)

# == MAIN FUNCTIONS ==

def synthetic_data(res_dir, true_params, args, hypers):
    """Generate publication-ready plots for synthetic data analysis."""
    logging.info(f"Starting visualization for {res_dir}")
    
    with open(f'{res_dir}/results.txt','w') as ofile:
        # Find best initialisation
        exp_logs, _ = find_bestrun(res_dir, args, ofile)
        brun = np.nanargmax(exp_logs) + 1
        print(f'Best run: {brun}', file=ofile) 

    # Create plot directories
    plot_path = f'{res_dir}/plots_{brun}'
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(f'{plot_path}/publication', exist_ok=True)
    
    # Load robust parameters
    rparams_path = f'{res_dir}/[{brun}]Robust_params.dictionary'
    if not os.path.exists(rparams_path) or os.stat(rparams_path).st_size <= 5:
        logging.error("No robust parameters found")
        return
        
    with open(rparams_path, 'rb') as f:
        rob_params = pickle.load(f)
    
    # Generate publication plots
    _plot_ground_truth_components(true_params, plot_path, args, hypers)
    _plot_inferred_components(rob_params, true_params, plot_path, args, hypers)
    _plot_factor_comparison(true_params, rob_params, plot_path, args)
    _plot_subgroup_analysis(true_params, rob_params, plot_path, args)

def qmap_pd(data, res_dir, args, hypers, topk=20):
    """
    Create publication-ready plots for qMAP-PD multi-view analysis with preprocessing support.
    """
    # Find best run
    with open(f"{res_dir}/results.txt", "w") as ofile:
        exp_logs, _ = find_bestrun(res_dir, args, ofile)
        brun = int(np.nanargmax(exp_logs) + 1)
        print(f"Best run: {brun}", file=ofile)

    # Create directories
    plot_path = f"{res_dir}/plots_{brun}"
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(f"{plot_path}/publication", exist_ok=True)

    # Load results
    W, Z = _load_results(res_dir, brun)
    if W is None or Z is None:
        raise RuntimeError("Could not load W/Z from results")

    # Extract data info
    Dm = np.array(hypers["Dm"], dtype=int)
    view_names = data.get("view_names", [f"View {i+1}" for i in range(len(Dm))])
    feat_names = data.get("feature_names", {})
    sub_ids = data.get("subject_ids", None)

    # Generate publication plots
    _plot_multiview_loadings(W, Dm, view_names, feat_names, plot_path, topk)
    _plot_subject_scores(Z, sub_ids, plot_path)
    _plot_latent_factor_summary(W, Z, Dm, view_names, plot_path)
    
    # Add preprocessing visualization if available
    if 'preprocessing' in data:
        plot_preprocessing_summary(data['preprocessing'], plot_path, view_names)
        logging.info("Added preprocessing visualization plots")

    # Add factor-to-MRI mapping visualization if available
    if FACTOR_MAPPING_AVAILABLE:
        
        try:
            factor_maps = add_to_qmap_visualization(
                data, res_dir, args, hypers, 
                base_dir=getattr(args, 'data_dir', None), 
                brun=brun
            )
            
            if factor_maps:
                logging.info("Factor-to-MRI mapping added to visualization")
                
                # Create overlay images for better visualization
                from qmap_gfa_weight2mri_python import create_factor_overlay_images
                
                # Find reference MRI
                base_dir = getattr(args, 'data_dir', 'qMAP-PD_data')
                ref_mri = os.path.join(base_dir, 'mri_reference', 'average_space-qmap-shoot-384_pdw-brain.nii')
                
                if os.path.exists(ref_mri):
                    create_factor_overlay_images(
                        factor_maps, ref_mri, 
                        output_dir=f"{plot_path}/factor_overlays"
                    )
                    logging.info("Factor overlay images created")
                
        except Exception as e:
            logging.warning(f"Could not add factor-to-MRI mapping: {e}")

# == LEGACY FUNCTIONS (BACKWARDS COMPATIBILITY) ==

def find_bestrun(res_dir, args, ofile):
    """Find the best run based on log density."""
    exp_logs = np.full(args.num_runs, np.nan)
    
    for r in range(args.num_runs):
        res_path = f'{res_dir}/[{r+1}]Model_params.dictionary'
        if os.path.exists(res_path) and os.stat(res_path).st_size > 5:
            try:
                with open(res_path, 'rb') as f:
                    mcmc_samples = pickle.load(f)
                exp_logs[r] = mcmc_samples['exp_logdensity']
                print(f'Run {r+1}: Log density = {exp_logs[r]:.2f}', file=ofile)
            except Exception as e:
                logging.warning(f"Could not load run {r+1}: {e}")
    
    # Check if any valid runs exist
    if np.all(np.isnan(exp_logs)):
        raise RuntimeError("No successful MCMC runs found. All runs failed.")
        
    return exp_logs, ofile

def plot_param(params, paths, args, cids=None, tr_vals=False):
    """
    Plot parameter matrices (W, Z, lambda, etc.) with professional styling.
    This function is called by run_analysis.py and must work.
    """
    
    lcomps = list(range(1, params['W'].shape[1]+1))
    
    # Plot W (loading matrix)
    if 'W' in params:
        W = params['W']
        pathW = paths['W']
        
        plt.figure(figsize=(max(8, W.shape[1]*0.8), max(6, W.shape[0]*0.03)))
        
        # Use improved colormap and styling
        sns.heatmap(W, 
                   vmin=-np.max(np.abs(W)), 
                   vmax=np.max(np.abs(W)), 
                   cmap="RdBu_r",
                   yticklabels=False, 
                   xticklabels=[f'F{i}' for i in lcomps],
                   cbar_kws={'label': 'Loading Weight'})
        
        plt.xlabel('Latent Factors', fontsize=12)
        plt.ylabel('Features', fontsize=12) 
        plt.title('Factor Loading Matrix (W)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{pathW}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot lambda W (sparsity parameters)
    if 'lmbW' in params:
        if cids is not None:
            lmbW = params['lmbW'][:,cids]
        else:
            lmbW = params['lmbW']
        pathlmbW = paths['lmbW'] 
        
        plt.figure(figsize=(max(8, lmbW.shape[1]*0.8), max(6, lmbW.shape[0]*0.03)))
        sns.heatmap(lmbW, 
                   vmin=0, 
                   vmax=np.max(lmbW), 
                   cmap="viridis",
                   yticklabels=False, 
                   xticklabels=[f'F{i}' for i in lcomps],
                   cbar_kws={'label': 'Sparsity Parameter'})
        
        plt.xlabel('Latent Factors', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title('Sparsity Parameters (λW)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{pathlmbW}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot Z (latent factors)
    if 'Z' in params:
        Z = params['Z']
        pathZ = paths['Z']
        pathZ_svg = paths['Z_svg']
        
        plt.figure(figsize=(max(8, Z.shape[1]*0.8), max(6, Z.shape[0]*0.03)))
        sns.heatmap(Z, 
                   vmin=-np.max(np.abs(Z)), 
                   vmax=np.max(np.abs(Z)), 
                   cmap="RdBu_r",
                   yticklabels=False, 
                   xticklabels=[f'F{i}' for i in lcomps],
                   cbar_kws={'label': 'Factor Score'})
        
        plt.xlabel('Latent Factors', fontsize=12)
        plt.ylabel('Subjects', fontsize=12) 
        plt.title('Latent Factor Scores (Z)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{pathZ}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{pathZ_svg}.svg', bbox_inches='tight')
        plt.close()
    
    # [Additional parameter plots continued as in original...]

def plot_X(data, args, hypers, path, true_data=False):
    """
    Plot data matrices X with improved styling.
    This function is called by run_analysis.py and must work.
    """
    
    if true_data:
        X = np.dot(data['Z'], data['W'].T)
        K = data['Z'].shape[1]
    else:    
        X = np.zeros((data[0][0].shape[0], data[0][0].shape[1]))
        K = len(data)
    
    for k in range(K):
        if true_data:
            z = np.reshape(data['Z'][:,k], (data['Z'].shape[0], 1)) 
            w = np.reshape(data['W'][:,k], (data['W'].shape[0], 1))
            X_k = np.dot(z,w.T)
        else:
            X_k = data[k][0]    
            X += X_k
        
        # Create subplot for each data source
        fig, axes = plt.subplots(1, args.num_sources, figsize=(4*args.num_sources, 5))
        if args.num_sources == 1:
            axes = [axes]
        
        Dm = hypers['Dm']
        d = 0
        view_names = ['Neuroimaging', 'Cognitive', 'Clinical'][:args.num_sources]
        
        for m in range(args.num_sources):
            X_m = X_k[:,d:d+Dm[m]]
            
            im = axes[m].imshow(X_m.T, aspect='auto', cmap="RdBu_r", 
                               vmin=np.min(X_k), vmax=np.max(X_k))
            axes[m].set_xlabel('Subjects')
            axes[m].set_ylabel('Features')
            axes[m].set_title(f'{view_names[m]}\n({Dm[m]} features)', fontweight='bold')
            
            if m == args.num_sources - 1:  # Add colorbar to last subplot
                cbar = fig.colorbar(im, ax=axes[m])
                cbar.set_label('Value')
            
            d += Dm[m]
        
        plt.suptitle(f'Factor {k+1} - Data Space Reconstruction', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{path}_comp{k+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot complete data matrix
    plt.figure(figsize=(10, 6))
    im = plt.imshow(X.T, aspect='auto', cmap="RdBu_r", 
                    vmin=np.min(X), vmax=np.max(X))
    plt.xlabel('Subjects', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Complete Data Matrix', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im)
    cbar.set_label('Value')
    
    plt.tight_layout()
    plt.savefig(f'{path}.png', dpi=300, bbox_inches='tight')
    plt.close()

def define_box_properties(plot_name, color_code, label):
    """Helper function for box plot styling - kept for legacy compatibility."""
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
    plt.plot([], c=color_code, label=label)
    plt.legend()

def _plot_ground_truth_components(true_params, plot_path, args, hypers):
    """Plot ground truth components with professional styling."""
    
    # Ground truth factor loadings (W)
    W_true = true_params['W']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    Dm = hypers['Dm']
    d = 0
    view_names = ['Neuroimaging', 'Cognitive', 'Clinical']
    
    for m in range(args.num_sources):
        W_view = W_true[d:d+Dm[m], :]
        im = axes[m].imshow(W_view, aspect='auto', cmap='RdBu_r', 
                           vmin=-np.max(np.abs(W_true)), vmax=np.max(np.abs(W_true)))
        axes[m].set_title(f'{view_names[m]}\n({Dm[m]} features)', fontweight='bold')
        axes[m].set_xlabel('Latent Factors')
        axes[m].set_ylabel('Features')
        axes[m].set_xticks(range(W_true.shape[1]))
        axes[m].set_xticklabels([f'F{i+1}' for i in range(W_true.shape[1])])
        d += Dm[m]
    
    # Add colorbar with proper positioning
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.1, aspect=20)
    cbar.set_label('Loading Weight', rotation=270, labelpad=15)
    
    plt.suptitle('Ground Truth Factor Loadings', fontsize=14, fontweight='bold')
    plt.subplots_adjust(top=0.75, hspace=0.3, wspace=0.4, right=0.88)
    plt.savefig(f'{plot_path}/publication/ground_truth_loadings.png')
    plt.savefig(f'{plot_path}/publication/ground_truth_loadings.pdf')
    plt.close()
    
    # Ground truth latent factors (Z)
    Z_true = true_params['Z']
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(Z_true.T, aspect='auto', cmap='RdBu_r', 
                   vmin=-np.max(np.abs(Z_true)), vmax=np.max(np.abs(Z_true)))
    ax.set_xlabel('Subjects')
    ax.set_ylabel('Latent Factors')
    ax.set_yticks(range(Z_true.shape[1]))
    ax.set_yticklabels([f'Factor {i+1}' for i in range(Z_true.shape[1])])
    ax.set_title('Ground Truth Latent Factor Scores', fontsize=14, fontweight='bold')
    
    cbar = fig.colorbar(im, fraction=0.02, pad=0.04)
    cbar.set_label('Factor Score', rotation=270, labelpad=15)
    
    plt.subplots_adjust(bottom=0.15, right=0.85, top=0.80)
    plt.savefig(f'{plot_path}/publication/ground_truth_factors.png')
    plt.savefig(f'{plot_path}/publication/ground_truth_factors.pdf')
    plt.close()

def _plot_inferred_components(rob_params, true_params, plot_path, args, hypers):
    """Plot inferred components with matched ground truth."""
    
    Z_inf = rob_params['Z']
    W_inf = rob_params['W']
    Z_true = true_params['Z']
    
    # Match factors using cosine similarity
    if Z_inf.shape[1] == Z_true.shape[1]:
        Z_matched, W_matched = _match_factors(Z_inf, W_inf, Z_true, true_params['W'])
    else:
        Z_matched, W_matched = Z_inf, W_inf
    
    # Plot matched inferred loadings
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    Dm = hypers['Dm']
    d = 0
    view_names = ['Neuroimaging', 'Cognitive', 'Clinical']
    
    for m in range(args.num_sources):
        W_view = W_matched[d:d+Dm[m], :]
        im = axes[m].imshow(W_view, aspect='auto', cmap='RdBu_r', 
                           vmin=-np.max(np.abs(W_matched)), vmax=np.max(np.abs(W_matched)))
        axes[m].set_title(f'{view_names[m]}\n({Dm[m]} features)', fontweight='bold')
        axes[m].set_xlabel('Latent Factors')
        axes[m].set_ylabel('Features')
        axes[m].set_xticks(range(W_matched.shape[1]))
        axes[m].set_xticklabels([f'F{i+1}' for i in range(W_matched.shape[1])])
        d += Dm[m]
    
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.1, aspect=20)
    cbar.set_label('Loading Weight', rotation=270, labelpad=15)
    
    plt.suptitle('Inferred Factor Loadings', fontsize=14, fontweight='bold')
    plt.subplots_adjust(top=0.75, hspace=0.3, wspace=0.4, right=0.88)
    plt.savefig(f'{plot_path}/publication/inferred_loadings.png')
    plt.savefig(f'{plot_path}/publication/inferred_loadings.pdf')
    plt.close()
    
    # Plot inferred latent factors
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(Z_matched.T, aspect='auto', cmap='RdBu_r', 
                   vmin=-np.max(np.abs(Z_matched)), vmax=np.max(np.abs(Z_matched)))
    ax.set_xlabel('Subjects')
    ax.set_ylabel('Latent Factors')
    ax.set_yticks(range(Z_matched.shape[1]))
    ax.set_yticklabels([f'Factor {i+1}' for i in range(Z_matched.shape[1])])
    ax.set_title('Inferred Latent Factor Scores', fontsize=14, fontweight='bold')
    
    cbar = fig.colorbar(im, fraction=0.02, pad=0.04)
    cbar.set_label('Factor Score', rotation=270, labelpad=15)
    
    plt.subplots_adjust(bottom=0.15, right=0.85, top=0.80)
    plt.savefig(f'{plot_path}/publication/inferred_factors.png')
    plt.savefig(f'{plot_path}/publication/inferred_factors.pdf')
    plt.close()

def _match_factors(Z_inf, W_inf, Z_true, W_true):
    """Match inferred factors to ground truth using cosine similarity."""
    sim_matrix = cosine_similarity(Z_true.T, Z_inf.T)
    
    Z_matched = np.zeros_like(Z_true)
    W_matched = np.zeros_like(W_true)
    
    for k in range(Z_true.shape[1]):
        best_match = np.argmax(np.abs(sim_matrix[k, :]))
        if sim_matrix[k, best_match] > 0:
            Z_matched[:, k] = Z_inf[:, best_match]
            W_matched[:, k] = W_inf[:, best_match]
        else:
            Z_matched[:, k] = -Z_inf[:, best_match]
            W_matched[:, k] = -W_inf[:, best_match]
    
    return Z_matched, W_matched

def _plot_factor_comparison(true_params, rob_params, plot_path, args):
    """Create factor correlation and reconstruction plots."""
    
    Z_true = true_params['Z']
    Z_inf = rob_params['Z']
    
    if Z_inf.shape[1] != Z_true.shape[1]:
        logging.warning("Cannot compare factors: different number of latent factors")
        return
    
    Z_matched, _ = _match_factors(Z_inf, rob_params['W'], Z_true, true_params['W'])
    
    # Factor correlation plot
    fig, axes = plt.subplots(1, Z_true.shape[1], figsize=(4*Z_true.shape[1], 3))
    if Z_true.shape[1] == 1:
        axes = [axes]
    
    for k in range(Z_true.shape[1]):
        r = np.corrcoef(Z_true[:, k], Z_matched[:, k])[0, 1]
        scatter = axes[k].scatter(Z_true[:, k], Z_matched[:, k], alpha=0.6, 
                       color=COLORS['primary'], s=20, label='Observed')
        line = axes[k].plot([Z_true[:, k].min(), Z_true[:, k].max()], 
                    [Z_true[:, k].min(), Z_true[:, k].max()], 
                    '--', color=COLORS['secondary'], alpha=0.8, linewidth=2, label='Perfect Recovery')
        axes[k].set_xlabel('True Factor Scores')
        axes[k].set_ylabel('Inferred Factor Scores')
        axes[k].set_title(f'Factor {k+1}\nr = {r:.3f}', fontweight='bold')
        axes[k].grid(True, alpha=0.3)
        
        # Add legend to first plot only
        if k == 0:
            axes[k].legend(loc='upper left', fontsize=8)
    
    plt.suptitle('Factor Recovery Performance', fontsize=14, fontweight='bold')
    plt.subplots_adjust(top=0.75, hspace=0.4, wspace=0.3)
    plt.savefig(f'{plot_path}/publication/factor_correlation.png')
    plt.savefig(f'{plot_path}/publication/factor_correlation.pdf')
    plt.close()

def _plot_subgroup_analysis(true_params, rob_params, plot_path, args):
    """Analyze subgroup-specific factor patterns."""
    
    Z_true = true_params['Z']
    Z_inf = rob_params['Z']
    
    if Z_inf.shape[1] != Z_true.shape[1]:
        return
    
    Z_matched, _ = _match_factors(Z_inf, rob_params['W'], Z_true, true_params['W'])
    
    # Assuming 3 equal-sized groups as in synthetic data
    N = Z_true.shape[0]
    group_size = N // 3
    group_labels = ['Group 1', 'Group 2', 'Group 3']
    
    # Calculate group-specific factor scores
    fig, axes = plt.subplots(2, Z_true.shape[1], figsize=(4*Z_true.shape[1], 10))
    
    for k in range(Z_true.shape[1]):
        # True scores by group
        for g in range(3):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size if g < 2 else N
            
            true_scores = np.abs(Z_true[start_idx:end_idx, k])
            inf_scores = np.abs(Z_matched[start_idx:end_idx, k])
            
            box1 = axes[0, k].boxplot([true_scores], positions=[g], widths=0.6,
                             patch_artist=True, 
                             boxprops=dict(facecolor=COLORS['groups'][g], alpha=0.7))
            box2 = axes[1, k].boxplot([inf_scores], positions=[g], widths=0.6,
                             patch_artist=True,
                             boxprops=dict(facecolor=COLORS['groups'][g], alpha=0.7))
        
        axes[0, k].set_title(f'Factor {k+1}', fontweight='bold')
        axes[0, k].set_ylabel('True |Factor Score|')
        axes[1, k].set_ylabel('Inferred |Factor Score|')
        axes[1, k].set_xlabel('Subgroups')
        
        for ax in axes[:, k]:
            ax.set_xticks(range(3))
            ax.set_xticklabels(group_labels)
            ax.grid(True, alpha=0.3)
            
        # Add legend to first factor only to avoid repetition
        if k == 0:
            legend_elements = [mpatches.Patch(color=COLORS['groups'][g], alpha=0.7, label=group_labels[g]) 
                             for g in range(3)]
            axes[0, k].legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    axes[0, 0].set_ylabel('True |Factor Score|')
    axes[1, 0].set_ylabel('Inferred |Factor Score|')
    
    plt.suptitle('Subgroup-Specific Factor Analysis', fontsize=14, fontweight='bold')
    plt.subplots_adjust(top=0.83, hspace=0.4, wspace=0.3)
    plt.savefig(f'{plot_path}/publication/subgroup_analysis.png')
    plt.savefig(f'{plot_path}/publication/subgroup_analysis.pdf')
    plt.close()

def _load_results(res_dir, brun):
    """Load W and Z matrices from results."""
    rob_path = f"{res_dir}/[{brun}]Robust_params.dictionary"
    mdl_path = f"{res_dir}/[{brun}]Model_params.dictionary"

    # Try robust parameters first
    if os.path.exists(rob_path) and os.stat(rob_path).st_size > 5:
        with open(rob_path, "rb") as f:
            rp = pickle.load(f)
        return rp.get("W"), rp.get("Z")

    # Fallback to model samples
    if os.path.exists(mdl_path) and os.stat(mdl_path).st_size > 5:
        with open(mdl_path, "rb") as f:
            smp = pickle.load(f)
        W = np.asarray(smp["W"])
        Z = np.asarray(smp["Z"])
        if W.ndim > 2: W = W.mean(axis=0)
        if Z.ndim > 2: Z = Z.mean(axis=0)
        return W, Z

    return None, None

def _shorten_imaging_labels(feature_names):
    """Shorten long imaging feature names for better readability."""
    shortened = []
    for name in feature_names:
        if "::" in name:
            # Extract region and voxel number: "volume_putamen_voxels::v1234" -> "putamen::v1234" (improves readability)
            parts = name.split("::")
            if len(parts) == 2:
                region_part = parts[0]
                voxel_part = parts[1]
                
                # Extract region name
                if "putamen" in region_part:
                    region = "putamen"
                elif "lentiform" in region_part:
                    region = "lentiform"
                elif "_sn_" in region_part:
                    region = "sn"
                else:
                    region = region_part.replace("volume_", "").replace("_voxels", "")
                
                shortened.append(f"{region}::{voxel_part}")
            else:
                shortened.append(name)
        else:
            shortened.append(name)
    return shortened

def _plot_multiview_loadings(W, Dm, view_names, feat_names, plot_path, topk):
    """Create loading plots for each view with improved error handling."""
    d = 0
    plot_path = Path(plot_path)
    
    for m, (vname, dim) in enumerate(zip(view_names, Dm)):
        try:
            Wv = W[d:d+dim, :]
            features = feat_names.get(vname, [f"Feature {i+1}" for i in range(dim)])
            
            # Shorten imaging labels for readability
            if 'imaging' in vname or any('volume_' in f for f in features[:5]):
                features = _shorten_imaging_labels(features)
            
            with safe_plotting_context() as plt:
                # Create figure with subplots for each component
                n_comp = Wv.shape[1]
                height_per_comp = 5 if ('clinical' in vname or 'imaging' in vname) else 3
                fig, axes = plt.subplots(n_comp, 1, figsize=(10, height_per_comp*n_comp))
                
                if n_comp == 1:
                    axes = [axes]
                
                for j in range(n_comp):
                    w = Wv[:, j]
                    top_idx = np.argsort(np.abs(w))[::-1][:topk]
                    top_weights = w[top_idx]
                    top_features = [features[i] for i in top_idx]
                    
                    colors = [COLORS['primary'] if x >= 0 else COLORS['secondary'] for x in top_weights]
                    bars = axes[j].barh(range(len(top_weights)), top_weights, color=colors, alpha=0.8)
                    
                    axes[j].set_yticks(range(len(top_weights)))
                    label_fontsize = 7 if ('clinical' in vname or 'imaging' in vname) else 8
                    axes[j].set_yticklabels(top_features, fontsize=label_fontsize)
                    axes[j].set_xlabel('Loading Weight')
                    axes[j].set_title(f'Latent Factor {j+1}', fontweight='bold')
                    axes[j].axvline(0, color='black', linewidth=0.8)
                    axes[j].grid(True, alpha=0.3, axis='x')
                    axes[j].invert_yaxis()
                
                plt.suptitle(f'{vname.title()} - Top {topk} Features by Absolute Loading Weight', 
                            fontsize=14, fontweight='bold')
                plt.subplots_adjust(top=0.83, hspace=0.4)
                
                # Save using safe save function
                base_path = plot_path / "publication" / f"loadings_{vname.lower().replace(' ', '_')}"
                save_plot_safely(fig, base_path)
                
        except Exception as e:
            logging.error(f"Failed to create loading plot for {vname}: {e}")
            continue
        finally:
            d += dim

def _plot_subject_scores(Z, sub_ids, plot_path):
    """Create professional subject scores heatmap."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    im = ax.imshow(Z.T, aspect='auto', cmap='RdBu_r', 
                   vmin=-np.max(np.abs(Z)), vmax=np.max(np.abs(Z)))
    
    # Labels and formatting
    ax.set_xlabel('Subjects')
    ax.set_ylabel('Latent Factors')
    ax.set_yticks(range(Z.shape[1]))
    ax.set_yticklabels([f'LF{i+1}' for i in range(Z.shape[1])])
    ax.set_title('Subject Latent Factor Scores', fontsize=14, fontweight='bold')
    
    # Add subject IDs if reasonable number
    if sub_ids is not None and len(sub_ids) <= 50:
        ax.set_xticks(range(0, len(sub_ids), max(1, len(sub_ids)//10)))
        ax.set_xticklabels([sub_ids[i] for i in range(0, len(sub_ids), max(1, len(sub_ids)//10))], 
                          rotation=45, ha='right', fontsize=8)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('Latent Factor Score', rotation=270, labelpad=15)
    
    plt.subplots_adjust(bottom=0.15, right=0.85, top=0.80)
    plt.savefig(f"{plot_path}/publication/subject_scores.png")
    plt.savefig(f"{plot_path}/publication/subject_scores.pdf")
    plt.close()

def _plot_latent_factor_summary(W, Z, Dm, view_names, plot_path):
    """Create latent factor summary visualization."""
    n_comp = W.shape[1]
    fig, axes = plt.subplots(2, n_comp, figsize=(3*n_comp, 8))
    
    if n_comp == 1:
        axes = axes.reshape(-1, 1)
    
    d = 0
    colors = COLORS['groups'][:len(view_names)]
    
    for j in range(n_comp):
        # Top panel: View-wise loading magnitudes
        view_magnitudes = []
        d_temp = 0
        for m, dim in enumerate(Dm):
            Wv = W[d_temp:d_temp+dim, j]
            view_magnitudes.append(np.mean(np.abs(Wv)))
            d_temp += dim
        
        bars = axes[0, j].bar(range(len(view_names)), view_magnitudes, 
                             color=colors, alpha=0.8)
        axes[0, j].set_xticks(range(len(view_names)))
        axes[0, j].set_xticklabels(view_names, rotation=0, ha='center')
        axes[0, j].set_ylabel('Mean |Loading|')
        axes[0, j].set_title(f'Latent Factor {j+1}', fontweight='bold')
        axes[0, j].grid(True, alpha=0.3, axis='y')
        
        # Add legend to first subplot only to avoid repetition
        if j == 0:
            legend_elements = [mpatches.Patch(color=colors[i], alpha=0.8, label=view_names[i]) 
                             for i in range(len(view_names))]
            axes[0, j].legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # Bottom panel: Subject score distribution
        axes[1, j].hist(Z[:, j], bins=20, color=COLORS['primary'], alpha=0.7, density=True)
        axes[1, j].axvline(0, color='black', linestyle='--', alpha=0.8)
        axes[1, j].set_xlabel('Latent Factor Score')
        axes[1, j].set_ylabel('Density')
        axes[1, j].grid(True, alpha=0.3)
        
        # Reduce number of x-axis ticks to prevent crowding
        axes[1, j].locator_params(axis='x', nbins=5)
    
    plt.suptitle('Latent Factor Summary Statistics', fontsize=16, fontweight='bold')
    plt.subplots_adjust(top=0.80, hspace=0.4, wspace=0.3)
    plt.savefig(f"{plot_path}/publication/latent_factor_summary.png")
    plt.savefig(f"{plot_path}/publication/latent_factor_summary.pdf")
    plt.close()

# == COMPREHENSIVE VISUALIZATION WRAPPER ==

def create_all_visualizations(
    results_dir: str,
    data: dict = None,
    cv_results: dict = None,
    centroids_json: str = None,
    probabilities_csv: str = None,
    run_name: str = "analysis",
    **kwargs
):
    """
    Create comprehensive visualizations for GFA results including preprocessing and CV.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing GFA results
    data : dict, optional
        Data dictionary from loader (for preprocessing plots)
    cv_results : dict, optional
        Cross-validation results dictionary
    centroids_json : str, optional
        Path to consensus centroids JSON
    probabilities_csv : str, optional
        Path to subtype probabilities CSV
    run_name : str
        Name for the analysis run
    **kwargs
        Additional arguments for visualization functions
    """
    
    logging.info(f"Creating comprehensive visualizations in {results_dir}")
    
    # Create main plots directory
    plots_dir = os.path.join(results_dir, "comprehensive_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    generated_plots = {}
    
    # 1. Create preprocessing visualization if available
    if data and 'preprocessing' in data:
        try:
            view_names = data.get('view_names', ['View 1', 'View 2'])
            plot_preprocessing_summary(data['preprocessing'], plots_dir, view_names)
            generated_plots['preprocessing'] = f"{plots_dir}/preprocessing"
            logging.info("✓ Preprocessing visualizations created")
        except Exception as e:
            logging.error(f"Failed to create preprocessing plots: {e}")
    
    # 2. Create cross-validation visualization if available
    if cv_results:
        try:
            plot_cv_results(cv_results, plots_dir, run_name)
            generated_plots['cross_validation'] = f"{plots_dir}/cross_validation"
            logging.info("✓ Cross-validation visualizations created")
        except Exception as e:
            logging.error(f"Failed to create CV plots: {e}")
    
    # 3. Create consensus subtype visualization if available
    if centroids_json and probabilities_csv:
        if (os.path.exists(centroids_json) and os.path.exists(probabilities_csv)):
            try:
                plot_consensus_subtypes(centroids_json, probabilities_csv, plots_dir)
                generated_plots['consensus'] = f"{plots_dir}/consensus"
                logging.info("✓ Consensus subtype visualizations created")
            except Exception as e:
                logging.error(f"Failed to create consensus plots: {e}")
        else:
            logging.warning("Consensus files not found, skipping consensus plots")
    
    # 4. Create factor analysis summary report
    try:
        _create_analysis_summary_report(results_dir, plots_dir, data, cv_results, run_name)
        generated_plots['summary_report'] = f"{plots_dir}/analysis_summary.html"
        logging.info("✓ Analysis summary report created")
    except Exception as e:
        logging.error(f"Failed to create summary report: {e}")
    
    logging.info(f"Generated {len(generated_plots)} visualization categories")
    return generated_plots

def _create_analysis_summary_report(results_dir, plots_dir, data, cv_results, run_name):
    """Create an HTML summary report combining all analysis results."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GFA Analysis Report: {run_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1 {{ color: #2E86C1; border-bottom: 2px solid #2E86C1; padding-bottom: 10px; }}
            h2 {{ color: #34495E; margin-top: 30px; }}
            .metric {{ background-color: #F8F9FA; padding: 10px; margin: 10px 0; border-left: 4px solid #2E86C1; }}
            .section {{ margin-bottom: 40px; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        </style>
    </head>
    <body>
        <h1>Sparse Bayesian Group Factor Analysis Report</h1>
        <p><strong>Analysis:</strong> {run_name}</p>
        <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Data Overview</h2>
    """
    
    if data:
        X_list = data.get('X_list', [])
        view_names = data.get('view_names', [])
        n_subjects = X_list[0].shape[0] if X_list else 'N/A'
        
        html_content += f"""
            <div class="metric">
                <strong>Number of Subjects:</strong> {n_subjects}<br>
                <strong>Number of Views:</strong> {len(view_names)}<br>
                <strong>Views:</strong> {', '.join(view_names)}<br>
                <strong>Feature Counts:</strong> {[X.shape[1] for X in X_list] if X_list else 'N/A'}
            </div>
        """
        
        if 'preprocessing' in data:
            prep = data['preprocessing']
            html_content += f"""
            <h3>Preprocessing Summary</h3>
            <div class="metric">
                <strong>Feature Reduction:</strong><br>
            """
            for view, stats in prep['feature_reduction'].items():
                html_content += f"• {view}: {stats['original']} → {stats['processed']} features ({stats['reduction_ratio']:.1%} retained)<br>"
            html_content += "</div>"
    
    if cv_results:
        html_content += f"""
        <div class="section">
            <h2>Cross-Validation Results</h2>
            <div class="metric">
                <strong>Mean CV Score:</strong> {cv_results.get('mean_cv_score', 'N/A'):.4f}<br>
                <strong>CV Standard Deviation:</strong> {cv_results.get('std_cv_score', 'N/A'):.4f}<br>
                <strong>Converged Folds:</strong> {cv_results.get('n_converged_folds', 'N/A')}<br>
                <strong>Total Time:</strong> {cv_results.get('total_time', 'N/A'):.1f} seconds
            </div>
        </div>
        """
    
    # Add image references
    html_content += """
        <div class="section">
            <h2>Visualizations</h2>
            <p>The following plots have been generated for this analysis:</p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(f"{plots_dir}/analysis_summary.html", 'w') as f:
        f.write(html_content)

# == ADDITIONAL UTILITY FUNCTIONS ==

def plot_model_comparison(results_dict, output_path):
    """
    Compare multiple model results side by side.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and results as values
    output_path : str
        Path to save comparison plots
    """
    
    os.makedirs(output_path, exist_ok=True)
    
    model_names = list(results_dict.keys())
    n_models = len(model_names)
    
    # Compare CV scores
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cv_means = []
    cv_stds = []
    
    for model_name in model_names:
        results = results_dict[model_name]
        if 'cv_scores' in results:
            scores = results['cv_scores']
            cv_means.append(np.mean(scores))
            cv_stds.append(np.std(scores))
        else:
            cv_means.append(0)
            cv_stds.append(0)
    
    bars = ax.bar(model_names, cv_means, yerr=cv_stds, capsize=5,
                  color=COLORS['groups'][:n_models], alpha=0.7)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Mean CV Score (R²)')
    ax.set_title('Model Comparison: Cross-Validation Performance', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean, std in zip(bars, cv_means, cv_stds):
        ax.text(bar.get_x() + bar.get_width()/2., mean + std + max(cv_means)*0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_path}/model_comparison.png")
    plt.savefig(f"{output_path}/model_comparison.pdf")
    plt.close()
    
    logging.info(f"Model comparison plot saved to {output_path}")

# Apply error handling to main visualization functions
synthetic_data = visualization_with_error_handling(synthetic_data)
qmap_pd = visualization_with_error_handling(qmap_pd)

# Export key functions for backward compatibility
__all__ = [
    'synthetic_data', 'qmap_pd', 'plot_param', 'plot_X', 'find_bestrun',
    'define_box_properties', 'plot_preprocessing_summary', 'plot_cv_results',
    'plot_consensus_subtypes', 'create_all_visualizations', 'plot_model_comparison', 'safe_plotting_context', 'save_plot_safely', 'setup_plotting_style'
]