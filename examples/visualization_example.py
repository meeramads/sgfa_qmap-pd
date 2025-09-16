"""Example usage of visualization capabilities for SGFA results."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.io_utils import save_json, save_plot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_factor_visualization():
    """Factor analysis visualization example."""
    print("=" * 60)
    print("FACTOR VISUALIZATION EXAMPLE")
    print("=" * 60)

    from data import generate_synthetic_data
    from visualization import FactorVisualizer

    # Generate synthetic results for visualization
    data = generate_synthetic_data(num_sources=3, K=5, num_subjects=80)
    X_list = data["X_list"]
    Z_true = data["Z_true"]
    W_true = data["W_true"]

    print(
        f"🔄 Visualizing factors for {len(X_list)} views, {Z_true.shape[1]} factors..."
    )

    # Create output directory
    output_dir = Path("results/visualization_example")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize factor visualizer
    visualizer = FactorVisualizer()

    # Example 1: Factor loadings heatmap
    print(f"\n📊 Creating factor loadings heatmaps...")

    for i, W in enumerate(W_true):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Use the visualizer's method
        visualizer.plot_loadings_heatmap(
            W, title=f"Factor Loadings - View {i + 1}", ax=ax
        )

        plt.tight_layout()
        save_plot(output_dir / f"loadings_heatmap_view_{i}.png")
        plt.close()

        print(f"  ✅ Saved loadings heatmap for view {i + 1}")

    # Example 2: Factor scores scatter plots
    print(f"\n📊 Creating factor scores visualizations...")

    # Pairwise factor plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    factor_pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (0, 4)]

    for i, (f1, f2) in enumerate(factor_pairs):
        if i < len(axes) and f1 < Z_true.shape[1] and f2 < Z_true.shape[1]:
            visualizer.plot_factor_scatter(
                Z_true[:, f1],
                Z_true[:, f2],
                title=f"Factor {f1 + 1} vs Factor {f2 + 1}",
                xlabel=f"Factor {f1 + 1}",
                ylabel=f"Factor {f2 + 1}",
                ax=axes[i],
            )

    # Remove unused subplots
    for i in range(len(factor_pairs), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    save_plot(output_dir / "factor_scatter_plots.png")
    plt.close()

    print(f"  ✅ Saved factor scatter plots")

    # Example 3: Factor distribution plots
    print(f"\n📊 Creating factor distribution plots...")

    fig, axes = plt.subplots(1, Z_true.shape[1], figsize=(15, 4))
    if Z_true.shape[1] == 1:
        axes = [axes]

    for k in range(Z_true.shape[1]):
        visualizer.plot_factor_distribution(
            Z_true[:, k], title=f"Factor {k + 1} Distribution", ax=axes[k]
        )

    plt.tight_layout()
    save_plot(output_dir / "factor_distributions.png")
    plt.close()

    print(f"  ✅ Saved factor distribution plots")

    # Example 4: Factor interpretability analysis
    print(f"\n📊 Creating factor interpretability plots...")

    # Simulate some interpretability metrics
    interpretability_scores = np.random.beta(2, 2, Z_true.shape[1])  # Between 0 and 1
    stability_scores = np.random.beta(3, 2, Z_true.shape[1])  # Slightly higher

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Interpretability bar plot
    factors = [f"Factor {i + 1}" for i in range(Z_true.shape[1])]
    ax1.bar(factors, interpretability_scores, color="skyblue", alpha=0.7)
    ax1.set_ylabel("Interpretability Score")
    ax1.set_title("Factor Interpretability")
    ax1.set_ylim(0, 1)

    # Stability vs Interpretability scatter
    ax2.scatter(stability_scores, interpretability_scores, s=100, alpha=0.7)
    for i, factor in enumerate(factors):
        ax2.annotate(
            factor,
            (stability_scores[i], interpretability_scores[i]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax2.set_xlabel("Stability Score")
    ax2.set_ylabel("Interpretability Score")
    ax2.set_title("Factor Quality Assessment")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    save_plot(output_dir / "factor_interpretability.png")
    plt.close()

    print(f"  ✅ Saved factor interpretability plots")

    return {
        "output_dir": output_dir,
        "n_factors": Z_true.shape[1],
        "n_views": len(X_list),
        "plots_created": [
            "loadings_heatmaps",
            "scatter_plots",
            "distributions",
            "interpretability",
        ],
    }


def example_preprocessing_visualization():
    """Preprocessing pipeline visualization example."""
    print("\n" + "=" * 60)
    print("PREPROCESSING VISUALIZATION EXAMPLE")
    print("=" * 60)

    import numpy as np

    from data import generate_synthetic_data
    from visualization import PreprocessingVisualizer

    # Generate data with preprocessing effects
    data = generate_synthetic_data(num_sources=2, K=4, num_subjects=100)
    X_original = data["X_list"]

    print(f"🔄 Simulating preprocessing effects...")

    # Simulate preprocessing steps
    X_processed = []
    preprocessing_steps = []

    for i, X in enumerate(X_original):
        # Step 1: Add missing values, then impute
        X_missing = X.copy()
        missing_mask = np.random.random(X.shape) < 0.05  # 5% missing
        X_missing[missing_mask] = np.nan

        # Simple mean imputation
        X_imputed = X_missing.copy()
        for j in range(X.shape[1]):
            mask = ~np.isnan(X_missing[:, j])
            if not mask.all():
                X_imputed[~mask, j] = np.nanmean(X_missing[:, j])

        # Step 2: Standardization
        X_standardized = (X_imputed - np.mean(X_imputed, axis=0)) / np.std(
            X_imputed, axis=0
        )

        # Step 3: Feature selection (remove low variance features)
        feature_vars = np.var(X_standardized, axis=0)
        high_var_features = feature_vars > np.percentile(feature_vars, 25)
        X_selected = X_standardized[:, high_var_features]

        X_processed.append(X_selected)

        preprocessing_steps.append(
            {
                "view": i,
                "original_shape": X.shape,
                "missing_added": np.sum(missing_mask),
                "missing_imputed": np.sum(missing_mask),
                "features_removed": np.sum(~high_var_features),
                "final_shape": X_selected.shape,
            }
        )

    # Create output directory
    output_dir = Path("results/visualization_example")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize preprocessing visualizer
    PreprocessingVisualizer()

    # Example 1: Before/after comparison
    print(f"\n📊 Creating before/after preprocessing comparison...")

    for i, (X_orig, X_proc) in enumerate(zip(X_original, X_processed)):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Original data distribution
        axes[0, 0].hist(X_orig.flatten(), bins=50, alpha=0.7, color="blue")
        axes[0, 0].set_title(f"Original Data Distribution - View {i + 1}")
        axes[0, 0].set_xlabel("Value")
        axes[0, 0].set_ylabel("Frequency")

        # Processed data distribution
        axes[0, 1].hist(X_proc.flatten(), bins=50, alpha=0.7, color="green")
        axes[0, 1].set_title(f"Processed Data Distribution - View {i + 1}")
        axes[0, 1].set_xlabel("Value")
        axes[0, 1].set_ylabel("Frequency")

        # Feature variance comparison (for overlapping features)
        min_features = min(X_orig.shape[1], X_proc.shape[1])
        orig_vars = np.var(X_orig[:, :min_features], axis=0)
        proc_vars = np.var(X_proc[:, :min_features], axis=0)

        axes[1, 0].scatter(orig_vars, proc_vars, alpha=0.6)
        axes[1, 0].plot([0, max(orig_vars)], [0, max(orig_vars)], "r--", alpha=0.5)
        axes[1, 0].set_xlabel("Original Feature Variance")
        axes[1, 0].set_ylabel("Processed Feature Variance")
        axes[1, 0].set_title("Feature Variance Comparison")

        # Correlation heatmap of first few features
        n_features_plot = min(10, X_proc.shape[1])
        corr_matrix = np.corrcoef(X_proc[:, :n_features_plot].T)
        im = axes[1, 1].imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        axes[1, 1].set_title(f"Feature Correlation - View {i + 1}")
        plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        save_plot(output_dir / f"preprocessing_comparison_view_{i}.png")

        print(f"  ✅ Saved preprocessing comparison for view {i + 1}")

    # Example 2: Preprocessing steps summary
    print(f"\n📊 Creating preprocessing summary...")

    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    views = [step["view"] for step in preprocessing_steps]
    missing_counts = [step["missing_added"] for step in preprocessing_steps]
    features_removed = [step["features_removed"] for step in preprocessing_steps]
    orig_features = [step["original_shape"][1] for step in preprocessing_steps]
    final_features = [step["final_shape"][1] for step in preprocessing_steps]

    # Missing values handled
    axes[0, 0].bar(
        [f"View {v + 1}" for v in views], missing_counts, color="orange", alpha=0.7
    )
    axes[0, 0].set_title("Missing Values Imputed")
    axes[0, 0].set_ylabel("Count")

    # Features removed
    axes[0, 1].bar(
        [f"View {v + 1}" for v in views], features_removed, color="red", alpha=0.7
    )
    axes[0, 1].set_title("Features Removed")
    axes[0, 1].set_ylabel("Count")

    # Feature count comparison
    x = np.arange(len(views))
    width = 0.35
    axes[1, 0].bar(x - width / 2, orig_features, width, label="Original", alpha=0.7)
    axes[1, 0].bar(
        x + width / 2, final_features, width, label="After Processing", alpha=0.7
    )
    axes[1, 0].set_title("Feature Count Comparison")
    axes[1, 0].set_ylabel("Number of Features")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f"View {v + 1}" for v in views])
    axes[1, 0].legend()

    # Processing efficiency
    efficiency = [final / orig for orig, final in zip(orig_features, final_features)]
    axes[1, 1].bar(
        [f"View {v + 1}" for v in views], efficiency, color="green", alpha=0.7
    )
    axes[1, 1].set_title("Feature Retention Rate")
    axes[1, 1].set_ylabel("Retained Features Ratio")
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    save_plot(output_dir / "preprocessing_summary.png")
    plt.close()

    print(f"  ✅ Saved preprocessing summary")

    return {
        "preprocessing_steps": preprocessing_steps,
        "output_dir": output_dir,
        "plots_created": ["comparison_plots", "summary_plot"],
    }


def example_brain_visualization():
    """Brain-specific visualization example."""
    print("\n" + "=" * 60)
    print("BRAIN VISUALIZATION EXAMPLE")
    print("=" * 60)

    import numpy as np

    from visualization import BrainVisualizer

    # Create output directory
    output_dir = Path("results/visualization_example")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize brain visualizer
    BrainVisualizer()

    # Example 1: Simulated brain map visualization
    print(f"🔄 Creating simulated brain factor maps...")

    # Simulate factor loadings mapped to brain regions
    n_regions = 200  # Typical number of brain regions
    n_factors = 5

    # Simulate brain coordinates (simplified)
    brain_coords = {
        "x": np.random.uniform(-70, 70, n_regions),
        "y": np.random.uniform(-100, 80, n_regions),
        "z": np.random.uniform(-50, 80, n_regions),
    }

    # Simulate region names
    [f"Region_{i + 1}" for i in range(n_regions)]

    # Simulate factor loadings with some spatial structure
    factor_loadings = np.random.randn(n_regions, n_factors)

    # Add spatial correlation (factors tend to be stronger in nearby regions)
    for k in range(n_factors):
        # Create a "hotspot" for each factor
        center_idx = np.random.randint(0, n_regions)
        center_x, center_y, center_z = (
            brain_coords["x"][center_idx],
            brain_coords["y"][center_idx],
            brain_coords["z"][center_idx],
        )

        # Calculate distances from center
        distances = np.sqrt(
            (brain_coords["x"] - center_x) ** 2
            + (brain_coords["y"] - center_y) ** 2
            + (brain_coords["z"] - center_z) ** 2
        )

        # Apply spatial decay
        spatial_weights = np.exp(-distances / 30)  # 30mm decay
        factor_loadings[:, k] *= spatial_weights

    print(f"✅ Generated factor loadings for {n_regions} brain regions")

    # Example 2: Factor loading brain maps
    print(f"\n📊 Creating brain factor loading maps...")

    for k in range(n_factors):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        loadings = factor_loadings[:, k]

        # Axial view (z-projection)
        scatter = axes[0].scatter(
            brain_coords["x"],
            brain_coords["y"],
            c=loadings,
            cmap="RdBu_r",
            s=20,
            alpha=0.7,
        )
        axes[0].set_xlabel("X (Left-Right)")
        axes[0].set_ylabel("Y (Posterior-Anterior)")
        axes[0].set_title(f"Factor {k + 1} - Axial View")
        plt.colorbar(scatter, ax=axes[0], label="Loading Strength")

        # Sagittal view (x-projection)
        scatter = axes[1].scatter(
            brain_coords["y"],
            brain_coords["z"],
            c=loadings,
            cmap="RdBu_r",
            s=20,
            alpha=0.7,
        )
        axes[1].set_xlabel("Y (Posterior-Anterior)")
        axes[1].set_ylabel("Z (Inferior-Superior)")
        axes[1].set_title(f"Factor {k + 1} - Sagittal View")
        plt.colorbar(scatter, ax=axes[1], label="Loading Strength")

        # Coronal view (y-projection)
        scatter = axes[2].scatter(
            brain_coords["x"],
            brain_coords["z"],
            c=loadings,
            cmap="RdBu_r",
            s=20,
            alpha=0.7,
        )
        axes[2].set_xlabel("X (Left-Right)")
        axes[2].set_ylabel("Z (Inferior-Superior)")
        axes[2].set_title(f"Factor {k + 1} - Coronal View")
        plt.colorbar(scatter, ax=axes[2], label="Loading Strength")

        plt.tight_layout()
        save_plot(output_dir / f"brain_factor_{k + 1}_maps.png")

        print(f"  ✅ Saved brain maps for factor {k + 1}")

    # Example 3: Network connectivity visualization
    print(f"\n📊 Creating network connectivity visualization...")

    # Simulate functional connectivity matrix
    connectivity_matrix = np.random.randn(n_regions, n_regions)
    connectivity_matrix = (
        connectivity_matrix + connectivity_matrix.T
    ) / 2  # Make symmetric
    np.fill_diagonal(connectivity_matrix, 1)  # Perfect self-connectivity

    # Apply distance-based decay to make it more realistic
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            distance = np.sqrt(
                (brain_coords["x"][i] - brain_coords["x"][j]) ** 2
                + (brain_coords["y"][i] - brain_coords["y"][j]) ** 2
                + (brain_coords["z"][i] - brain_coords["z"][j]) ** 2
            )
            decay = np.exp(-distance / 50)  # 50mm decay
            connectivity_matrix[i, j] *= decay
            connectivity_matrix[j, i] = connectivity_matrix[i, j]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Connectivity matrix heatmap
    im = axes[0].imshow(connectivity_matrix, cmap="RdBu_r", aspect="auto")
    axes[0].set_title("Functional Connectivity Matrix")
    axes[0].set_xlabel("Brain Region")
    axes[0].set_ylabel("Brain Region")
    plt.colorbar(im, ax=axes[0], label="Connectivity Strength")

    # Network graph (simplified - show strongest connections)
    threshold = np.percentile(np.abs(connectivity_matrix), 95)  # Top 5% connections
    strong_connections = np.abs(connectivity_matrix) > threshold

    # Plot nodes (brain regions)
    axes[1].scatter(
        brain_coords["x"],
        brain_coords["y"],
        s=30,
        c="lightblue",
        alpha=0.6,
        edgecolors="black",
    )

    # Plot edges (strong connections)
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            if strong_connections[i, j]:
                color = "red" if connectivity_matrix[i, j] > 0 else "blue"
                alpha = min(abs(connectivity_matrix[i, j]), 0.8)
                axes[1].plot(
                    [brain_coords["x"][i], brain_coords["x"][j]],
                    [brain_coords["y"][i], brain_coords["y"][j]],
                    color=color,
                    alpha=alpha,
                    linewidth=0.5,
                )

    axes[1].set_xlabel("X (Left-Right)")
    axes[1].set_ylabel("Y (Posterior-Anterior)")
    axes[1].set_title("Brain Network (Top 5% Connections)")

    plt.tight_layout()
    save_plot(output_dir / "brain_connectivity.png")
    plt.close()

    print(f"  ✅ Saved brain connectivity visualization")

    return {
        "n_regions": n_regions,
        "n_factors": n_factors,
        "output_dir": output_dir,
        "connectivity_matrix": connectivity_matrix,
        "plots_created": ["factor_brain_maps", "connectivity_network"],
    }


def example_comprehensive_report():
    """Comprehensive analysis report generation example."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE REPORT GENERATION EXAMPLE")
    print("=" * 60)

    from data import generate_synthetic_data
    from visualization import VisualizationManager

    # Generate comprehensive synthetic analysis results
    data = generate_synthetic_data(num_sources=3, K=6, num_subjects=120)

    # Create output directory
    output_dir = Path("results/visualization_example")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Generating comprehensive analysis report...")

    # Initialize visualization manager
    VisualizationManager()

    # Simulate complete analysis results
    analysis_results = {
        "data_info": {
            "num_subjects": data["X_list"][0].shape[0],
            "num_views": len(data["X_list"]),
            "view_dimensions": [X.shape[1] for X in data["X_list"]],
            "num_factors": data["Z_true"].shape[1],
        },
        "model_results": {
            "factor_loadings": data["W_true"],
            "factor_scores": data["Z_true"],
            "explained_variance": np.random.beta(3, 2, data["Z_true"].shape[1]),
            "model_fit": {"log_likelihood": -1500.5, "aic": 3020.0, "bic": 3180.2},
        },
        "preprocessing_info": {
            "missing_imputed": [45, 32, 28],
            "features_removed": [12, 8, 15],
            "standardization_applied": True,
        },
        "validation_results": {
            "cv_scores": np.random.normal(0.75, 0.05, 5),  # 5-fold CV
            "stability_scores": np.random.beta(4, 2, data["Z_true"].shape[1]),
            "reproducibility": 0.87,
        },
    }

    # Generate individual visualizations
    print(f"\n📊 Creating individual visualization components...")

    # 1. Model performance summary
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Explained variance
    factors = [
        f"F{i + 1}"
        for i in range(len(analysis_results["model_results"]["explained_variance"]))
    ]
    axes[0, 0].bar(factors, analysis_results["model_results"]["explained_variance"])
    axes[0, 0].set_title("Explained Variance by Factor")
    axes[0, 0].set_ylabel("Variance Explained")

    # CV scores
    axes[0, 1].boxplot([analysis_results["validation_results"]["cv_scores"]])
    axes[0, 1].set_title("Cross-Validation Performance")
    axes[0, 1].set_ylabel("CV Score")
    axes[0, 1].set_xticklabels(["5-Fold CV"])

    # Stability scores
    axes[1, 0].bar(factors, analysis_results["validation_results"]["stability_scores"])
    axes[1, 0].set_title("Factor Stability Scores")
    axes[1, 0].set_ylabel("Stability")
    axes[1, 0].set_ylim(0, 1)

    # Model comparison (simulated)
    models = ["SGFA", "Standard GFA", "PCA", "ICA"]
    scores = [0.78, 0.65, 0.52, 0.48]
    axes[1, 1].bar(models, scores, color=["red", "orange", "blue", "green"])
    axes[1, 1].set_title("Model Comparison")
    axes[1, 1].set_ylabel("Performance Score")
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    save_plot(output_dir / "model_performance_summary.png")
    plt.close()

    print(f"  ✅ Created model performance summary")

    # 2. Data quality dashboard
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    view_names = [f"View {i + 1}" for i in range(len(data["X_list"]))]

    # Missing data
    missing_data = analysis_results["preprocessing_info"]["missing_imputed"]
    axes[0, 0].bar(view_names, missing_data, color="orange", alpha=0.7)
    axes[0, 0].set_title("Missing Data Imputed")
    axes[0, 0].set_ylabel("Count")

    # Features removed
    features_removed = analysis_results["preprocessing_info"]["features_removed"]
    axes[0, 1].bar(view_names, features_removed, color="red", alpha=0.7)
    axes[0, 1].set_title("Features Removed")
    axes[0, 1].set_ylabel("Count")

    # Data distribution example (first view)
    axes[0, 2].hist(data["X_list"][0].flatten(), bins=50, alpha=0.7, color="blue")
    axes[0, 2].set_title("Data Distribution (View 1)")
    axes[0, 2].set_xlabel("Value")
    axes[0, 2].set_ylabel("Frequency")

    # Correlation between factors
    factor_corr = np.corrcoef(data["Z_true"].T)
    im = axes[1, 0].imshow(factor_corr, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 0].set_title("Factor Correlations")
    plt.colorbar(im, ax=axes[1, 0])

    # Factor score distributions
    for k in range(min(2, data["Z_true"].shape[1])):
        axes[1, 1].hist(
            data["Z_true"][:, k], bins=30, alpha=0.6, label=f"Factor {k + 1}"
        )
    axes[1, 1].set_title("Factor Score Distributions")
    axes[1, 1].set_xlabel("Score")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()

    # Quality metrics summary
    quality_metrics = ["Completeness", "Consistency", "Validity", "Accuracy"]
    quality_scores = np.random.beta(4, 1.5, 4)  # High quality scores
    axes[1, 2].bar(quality_metrics, quality_scores, color="green", alpha=0.7)
    axes[1, 2].set_title("Data Quality Metrics")
    axes[1, 2].set_ylabel("Score")
    axes[1, 2].set_ylim(0, 1)

    plt.tight_layout()
    save_plot(output_dir / "data_quality_dashboard.png")
    plt.close()

    print(f"  ✅ Created data quality dashboard")

    # 3. Generate HTML report
    print(f"\n📋 Generating HTML report...")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SGFA Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9e9e9; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Sparse Group Factor Analysis Report</h1>
            <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metric">
                <strong>Subjects:</strong> {analysis_results['data_info']['num_subjects']}
            </div>
            <div class="metric">
                <strong>Data Views:</strong> {analysis_results['data_info']['num_views']}
            </div>
            <div class="metric">
                <strong>Factors:</strong> {analysis_results['data_info']['num_factors']}
            </div>
            <div class="metric">
                <strong>Model Fit (Log-Likelihood):</strong> {analysis_results['model_results']['model_fit']['log_likelihood']:.1f}
            </div>
            <div class="metric">
                <strong>Cross-Validation Score:</strong> {np.mean(analysis_results['validation_results']['cv_scores']):.3f} ± {np.std(analysis_results['validation_results']['cv_scores']):.3f}
            </div>
        </div>

        <div class="section">
            <h2>Model Performance</h2>
            <img src="model_performance_summary.png" alt="Model Performance Summary">
            <p>The sparse group factor analysis identified {analysis_results['data_info']['num_factors']} factors explaining the multi-view neuroimaging data. Cross-validation performance indicates good generalizability with a mean score of {np.mean(analysis_results['validation_results']['cv_scores']):.3f}.</p>
        </div>

        <div class="section">
            <h2>Data Quality Assessment</h2>
            <img src="data_quality_dashboard.png" alt="Data Quality Dashboard">
            <p>Data preprocessing successfully handled missing values and removed low-variance features. Total missing values imputed: {sum(analysis_results['preprocessing_info']['missing_imputed'])}, features removed: {sum(analysis_results['preprocessing_info']['features_removed'])}.</p>
        </div>

        <div class="section">
            <h2>Factor Analysis Results</h2>
            <img src="factor_scatter_plots.png" alt="Factor Scatter Plots">
            <img src="factor_distributions.png" alt="Factor Distributions">
            <p>Factor analysis revealed distinct patterns in the neuroimaging data. Factor scores show appropriate distributions and meaningful relationships between factors.</p>
        </div>

        <div class="section">
            <h2>Brain Visualization</h2>
            <img src="brain_connectivity.png" alt="Brain Connectivity">
            <p>Brain network analysis shows factor-specific connectivity patterns, highlighting regions of interest for further investigation.</p>
        </div>

        <div class="section">
            <h2>Technical Details</h2>
            <h3>Model Specifications</h3>
            <ul>
                <li>Model Type: Sparse Group Factor Analysis</li>
                <li>Number of Factors: {analysis_results['data_info']['num_factors']}</li>
                <li>Regularization: Horseshoe prior with regularization</li>
                <li>MCMC Samples: 2000 (after 1000 warmup)</li>
            </ul>

            <h3>Data Characteristics</h3>
            <ul>
                <li>Subjects: {analysis_results['data_info']['num_subjects']}</li>
                <li>Data Views: {analysis_results['data_info']['num_views']}</li>
                <li>View Dimensions: {analysis_results['data_info']['view_dimensions']}</li>
            </ul>

            <h3>Quality Metrics</h3>
            <ul>
                <li>Model Log-Likelihood: {analysis_results['model_results']['model_fit']['log_likelihood']:.1f}</li>
                <li>AIC: {analysis_results['model_results']['model_fit']['aic']:.1f}</li>
                <li>BIC: {analysis_results['model_results']['model_fit']['bic']:.1f}</li>
                <li>Reproducibility Score: {analysis_results['validation_results']['reproducibility']:.2f}</li>
            </ul>
        </div>
    </body>
    </html>
    """

    # Save HTML report

    report_path = output_dir / "analysis_report.html"
    with open(report_path, "w") as f:
        f.write(html_content)

    print(f"  ✅ Generated HTML report: {report_path}")

    # 4. Generate summary statistics file
    summary_stats = {
        "analysis_summary": {
            "dataset_info": analysis_results["data_info"],
            "model_performance": analysis_results["model_results"]["model_fit"],
            "validation_metrics": {
                "mean_cv_score": float(
                    np.mean(analysis_results["validation_results"]["cv_scores"])
                ),
                "std_cv_score": float(
                    np.std(analysis_results["validation_results"]["cv_scores"])
                ),
                "reproducibility": analysis_results["validation_results"][
                    "reproducibility"
                ],
            },
            "data_quality": analysis_results["preprocessing_info"],
        },
        "factor_details": {
            f"factor_{i + 1}": {
                "explained_variance": float(
                    analysis_results["model_results"]["explained_variance"][i]
                ),
                "stability_score": float(
                    analysis_results["validation_results"]["stability_scores"][i]
                ),
            }
            for i in range(analysis_results["data_info"]["num_factors"])
        },
    }

    save_json(summary_stats, output_dir / "analysis_summary.json")

    print(f"  ✅ Generated summary statistics: analysis_summary.json")

    return {
        "output_dir": output_dir,
        "report_path": report_path,
        "analysis_results": analysis_results,
        "plots_created": [
            "performance_summary",
            "quality_dashboard",
            "html_report",
            "summary_stats",
        ],
    }


def example_custom_plotting():
    """Custom plotting and styling example."""
    print("\n" + "=" * 60)
    print("CUSTOM PLOTTING AND STYLING EXAMPLE")
    print("=" * 60)

    import matplotlib.pyplot as plt
    import seaborn as sns

    from data import generate_synthetic_data

    # Set up custom styling
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Generate data
    data = generate_synthetic_data(num_sources=2, K=4, num_subjects=100)

    # Create output directory
    output_dir = Path("results/visualization_example")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Creating custom styled visualizations...")

    # Custom color schemes
    factor_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    view_colors = ["#FA7B8C", "#85C1E9"]

    # Example 1: Custom factor comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Factor loadings comparison across views
    for i, W in enumerate(data["W_true"]):
        ax = axes[0, i] if i < 2 else axes[0, i - 2]

        # Create grouped bar plot
        x = np.arange(min(10, W.shape[0]))  # First 10 features
        width = 0.15

        for k in range(W.shape[1]):
            ax.bar(
                x + k * width,
                W[x, k],
                width,
                label=f"Factor {k + 1}",
                color=factor_colors[k],
                alpha=0.8,
            )

        ax.set_title(f"Factor Loadings - View {i + 1}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Loading Strength")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Factor correlation network
    factor_corr = np.corrcoef(data["Z_true"].T)

    # Create network-style correlation plot
    ax = axes[1, 0]

    # Plot nodes (factors)
    angles = np.linspace(0, 2 * np.pi, len(factor_colors), endpoint=False)
    radius = 1

    factor_positions = {}
    for i, (angle, color) in enumerate(zip(angles, factor_colors)):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        factor_positions[i] = (x, y)

        ax.scatter(x, y, s=500, c=color, alpha=0.8, edgecolors="black", linewidths=2)
        ax.annotate(
            f"F{i + 1}",
            (x, y),
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=12,
        )

    # Plot edges (correlations)
    for i in range(len(factor_colors)):
        for j in range(i + 1, len(factor_colors)):
            corr = factor_corr[i, j]
            if abs(corr) > 0.3:  # Only show strong correlations
                x1, y1 = factor_positions[i]
                x2, y2 = factor_positions[j]

                color = "red" if corr > 0 else "blue"
                alpha = min(abs(corr), 0.8)
                width = abs(corr) * 3

                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=width)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_title("Factor Correlation Network", fontsize=14, fontweight="bold")
    ax.axis("off")

    # Subject clustering based on factors
    ax = axes[1, 1]

    # Simple clustering for visualization
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data["Z_true"])

    # Plot first two factors colored by cluster
    scatter = ax.scatter(
        data["Z_true"][:, 0],
        data["Z_true"][:, 1],
        c=clusters,
        cmap="tab10",
        alpha=0.7,
        s=50,
    )

    # Plot cluster centers
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="x", s=200, linewidths=3)

    ax.set_xlabel("Factor 1 Score")
    ax.set_ylabel("Factor 2 Score")
    ax.set_title("Subject Clustering (Factor Space)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(output_dir / "custom_factor_analysis.png")
    plt.close()

    print(f"  ✅ Created custom factor analysis plot")

    # Example 2: Detailed figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Main heatmap
    ax_main = fig.add_subplot(gs[:2, :2])

    # Combine loadings from both views
    combined_loadings = np.vstack(data["W_true"])

    sns.heatmap(
        combined_loadings.T,
        cmap="RdBu_r",
        center=0,
        ax=ax_main,
        cbar_kws={"label": "Loading Strength"},
    )
    ax_main.set_title(
        "Factor Loadings Across All Views", fontsize=16, fontweight="bold"
    )
    ax_main.set_xlabel("Features")
    ax_main.set_ylabel("Factors")

    # Factor score distributions
    ax_dist = fig.add_subplot(gs[:2, 2:])

    for k in range(data["Z_true"].shape[1]):
        ax_dist.hist(
            data["Z_true"][:, k],
            bins=20,
            alpha=0.6,
            label=f"Factor {k + 1}",
            color=factor_colors[k],
        )

    ax_dist.set_xlabel("Factor Score")
    ax_dist.set_ylabel("Frequency")
    ax_dist.set_title("Factor Score Distributions", fontsize=16, fontweight="bold")
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)

    # Model comparison (bottom row)
    methods = ["SGFA", "Standard GFA", "PCA", "ICA", "NMF"]

    # Simulated performance metrics
    metrics = {
        "Reconstruction Error": np.array([0.15, 0.22, 0.35, 0.28, 0.31]),
        "Sparsity Score": np.array([0.85, 0.45, 0.20, 0.30, 0.65]),
        "Interpretability": np.array([0.78, 0.65, 0.55, 0.48, 0.58]),
    }

    for i, (metric_name, scores) in enumerate(metrics.items()):
        ax = fig.add_subplot(gs[2, i])

        bars = ax.bar(methods, scores, color=sns.color_palette("viridis", len(methods)))

        # Highlight SGFA
        bars[0].set_color("#FF6B6B")
        bars[0].set_alpha(0.9)

        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    # Add overall title
    fig.suptitle(
        "Sparse Group Factor Analysis: Comprehensive Results",
        fontsize=18,
        fontweight="bold",
        y=0.95,
    )

    save_plot(output_dir / "detailed_figure.png")
    plt.close()

    print(f"  ✅ Created detailed figure")

    return {
        "output_dir": output_dir,
        "custom_colors": factor_colors,
        "plots_created": ["custom_factor_analysis", "detailed_figure"],
    }


if __name__ == "__main__":
    print("Visualization Examples for SGFA Analysis")
    print("=" * 60)

    # Run all examples
    try:
        # Core visualization examples
        factor_results = example_factor_visualization()
        preprocessing_results = example_preprocessing_visualization()
        brain_results = example_brain_visualization()

        # Advanced examples
        report_results = example_comprehensive_report()
        custom_results = example_custom_plotting()

        print("\n" + "=" * 60)
        print("ALL VISUALIZATION EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print(f"\nVisualization summary:")
        print(f"• Factor plots: {len(factor_results['plots_created'])} types created")
        print(
            f"• Brain visualizations: {
                brain_results['n_factors']} factors × {
                brain_results['n_regions']} regions"
        )
        print(
            f"• Preprocessing plots: {len(preprocessing_results['plots_created'])} types"
        )
        print(f"• Comprehensive report: {report_results['report_path']}")
        print(
            f"• Custom styling: {len(custom_results['plots_created'])} detailed plots"
        )
        print(f"• All outputs saved to: {factor_results['output_dir']}")

        # List all created files
        output_dir = factor_results["output_dir"]
        created_files = (
            list(output_dir.glob("*.png"))
            + list(output_dir.glob("*.html"))
            + list(output_dir.glob("*.json"))
        )
        print(f"\nTotal files created: {len(created_files)}")
        for file_path in sorted(created_files)[:10]:  # Show first 10
            print(f"  • {file_path.name}")
        if len(created_files) > 10:
            print(f"  ... and {len(created_files) - 10} more files")

    except Exception as e:
        logger.error(f"Visualization example failed: {e}")
        raise
