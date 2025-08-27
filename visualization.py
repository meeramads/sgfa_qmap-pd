import pickle
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import gcf
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway, ttest_ind
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.captureWarnings(True)

# Publication-ready matplotlib settings with improved spacing
plt.rcParams.update({
    "font.family": "DejaVu Sans",  # Available in Colab
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
    # Remove constrained_layout to avoid conflicts with subplots_adjust
    # "figure.constrained_layout.use": True,  
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    # Add default spacing parameters for better plot separation
    "figure.subplot.hspace": 0.8,  # Increased vertical spacing between subplots
    "figure.subplot.wspace": 0.5,  # Increased horizontal spacing between subplots
})

# Professional color schemes
COLORS = {
    'primary': '#2E86C1',
    'secondary': '#E74C3C', 
    'tertiary': '#28B463',
    'quaternary': '#F39C12',
    'neutral': '#5D6D7E',
    'light_gray': '#BDC3C7',
    'groups': ['#3498DB', '#E74C3C', '#28B463', '#F39C12', '#9B59B6', '#E67E22']
}

def synthetic_data(res_dir, true_params, args, hypers):
    """Generate publication-ready plots for synthetic data analysis."""
    logging.info(f"Starting improved visualization for {res_dir}")
    
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
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label('Loading Weight', rotation=270, labelpad=15)
    
    plt.suptitle('Ground Truth Factor Loadings', fontsize=14, fontweight='bold')
    plt.subplots_adjust(top=0.85, hspace=0.3, wspace=0.4)
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
    
    plt.subplots_adjust(bottom=0.15, right=0.85)
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
    
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label('Loading Weight', rotation=270, labelpad=15)
    
    plt.suptitle('Inferred Factor Loadings', fontsize=14, fontweight='bold')
    plt.subplots_adjust(top=0.85, hspace=0.3, wspace=0.4)
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
    
    plt.subplots_adjust(bottom=0.15, right=0.85)
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
    plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.3)
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
    plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
    plt.savefig(f'{plot_path}/publication/subgroup_analysis.png')
    plt.savefig(f'{plot_path}/publication/subgroup_analysis.pdf')
    plt.close()

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

def qmap_pd(data, res_dir, args, hypers, topk=20):
    """
    Create publication-ready plots for qMAP-PD multi-view analysis.
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
            # Extract region and voxel number: "volume_putamen_voxels::v1234" -> "putamen::v1234"
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
    """Create professional loading plots for each view."""
    d = 0
    
    for m, (vname, dim) in enumerate(zip(view_names, Dm)):
        Wv = W[d:d+dim, :]
        features = feat_names.get(vname, [f"Feature {i+1}" for i in range(dim)])
        
        # Shorten imaging labels for readability
        if 'imaging' in vname or any('volume_' in f for f in features[:5]):
            features = _shorten_imaging_labels(features)
        
        # Create figure with subplots for each component
        n_comp = Wv.shape[1]
        # Increase height per component for better spacing
        height_per_comp = 5 if ('clinical' in vname or 'imaging' in vname) else 3
        fig, axes = plt.subplots(n_comp, 1, figsize=(10, height_per_comp*n_comp))
        
        if n_comp == 1:
            axes = [axes]
        
        for j in range(n_comp):
            w = Wv[:, j]
            # Get top features by absolute weight
            top_idx = np.argsort(np.abs(w))[::-1][:topk]
            top_weights = w[top_idx]
            top_features = [features[i] for i in top_idx]
            
            # Create horizontal bar plot
            colors = [COLORS['primary'] if x >= 0 else COLORS['secondary'] for x in top_weights]
            bars = axes[j].barh(range(len(top_weights)), top_weights, color=colors, alpha=0.8)
            
            axes[j].set_yticks(range(len(top_weights)))
            # Smaller font for clinical and imaging features due to long names
            label_fontsize = 7 if ('clinical' in vname or 'imaging' in vname) else 8
            axes[j].set_yticklabels(top_features, fontsize=label_fontsize)
            axes[j].set_xlabel('Loading Weight')
            axes[j].set_title(f'Latent Factor {j+1}', fontweight='bold')
            axes[j].axvline(0, color='black', linewidth=0.8)
            axes[j].grid(True, alpha=0.3, axis='x')
            
            # Invert y-axis so highest weights are at top
            axes[j].invert_yaxis()
        
        plt.suptitle(f'{vname.title()} - Top {topk} Features by Absolute Loading Weight', 
                    fontsize=14, fontweight='bold')
        plt.subplots_adjust(top=0.92, hspace=0.4)
        plt.savefig(f"{plot_path}/publication/loadings_{vname.lower().replace(' ', '_')}.png")
        plt.savefig(f"{plot_path}/publication/loadings_{vname.lower().replace(' ', '_')}.pdf")
        plt.close()
        
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
    
    plt.subplots_adjust(bottom=0.15, right=0.85)
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
        for m, dim in enumerate(Dm):
            Wv = W[d:d+dim, j]
            view_magnitudes.append(np.mean(np.abs(Wv)))
            d_temp = d + dim if m < len(Dm)-1 else d + dim
            d = d_temp if m == len(Dm)-1 else d
        d = 0
        
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
        
        # Update d for next iteration
        d = sum(Dm[:m+1]) if m < len(Dm)-1 else 0
    
    plt.suptitle('Latent Factor Summary Statistics', fontsize=16, fontweight='bold')
    plt.subplots_adjust(top=0.9, hspace=0.4, wspace=0.3)
    plt.savefig(f"{plot_path}/publication/latent_factor_summary.png")
    plt.savefig(f"{plot_path}/publication/latent_factor_summary.pdf")
    plt.close()

def define_box_properties(plot_name, color_code, label):
    """Helper function for box plot styling - kept for legacy compatibility."""
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
    plt.plot([], c=color_code, label=label)
    plt.legend()

# Simplified legacy functions for backward compatibility
def plot_param(params, paths, args, cids=None, tr_vals=False):
    """Legacy function - use new plotting functions instead."""
    logging.warning("plot_param is deprecated - use new visualization functions")
    pass

def plot_X(data, args, hypers, path, true_data=False):
    """Legacy function - use new plotting functions instead."""
    logging.warning("plot_X is deprecated - use new visualization functions")
    pass