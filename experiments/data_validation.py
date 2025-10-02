"""Data validation and preprocessing experiments."""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from core.config_utils import get_data_dir
from core.experiment_utils import (
    experiment_handler,
)
from core.io_utils import save_csv, save_plot
from core.validation_utils import (
    ResultValidator,
    validate_data_types,
)
from data.qmap_pd import load_qmap_pd
from experiments.framework import (
    ExperimentConfig,
    ExperimentFramework,
    ExperimentResult,
)
from optimization.experiment_mixins import performance_optimized_experiment

logger = logging.getLogger(__name__)


def _log_preprocessing_summary(preprocessing_info):
    """Log a concise preprocessing summary instead of full details."""
    if not preprocessing_info:
        logger.info("✅ Preprocessing info: None")
        return

    # Extract key information
    status = preprocessing_info.get("status", "unknown")

    # Handle different strategy formats
    strategy = preprocessing_info.get("strategy", "unknown")
    if isinstance(strategy, dict):
        strategy = strategy.get("name", "unknown")

    # Check for preprocessing results nested structure
    if "preprocessing_results" in preprocessing_info:
        nested_info = preprocessing_info["preprocessing_results"]
        status = nested_info.get("status", status)
        strategy = nested_info.get("strategy", strategy)
        preprocessor_type = nested_info.get("preprocessor_type", "unknown")
    else:
        preprocessor_type = preprocessing_info.get("preprocessor_type", "unknown")

    logger.info(f"✅ Preprocessing: {status} ({preprocessor_type})")
    logger.info(f"   Strategy: {strategy}")

    # Get the actual preprocessing data (could be nested)
    data_source = preprocessing_info.get("preprocessing_results", preprocessing_info)

    # Log feature reduction if available
    if "feature_reduction" in data_source:
        fr = data_source["feature_reduction"]
        logger.info(
            f" Features: { fr['total_before']:,} → { fr['total_after']:,} ({ fr['reduction_ratio']:.3f} ratio)"
        )

    # Log steps applied
    steps = data_source.get("steps_applied", [])
    if steps:
        logger.info(f"   Steps: {', '.join(steps)}")

    # Log basic shapes summary
    original_shapes = data_source.get("original_shapes", [])
    processed_shapes = data_source.get("processed_shapes", [])
    if original_shapes and processed_shapes:
        total_orig_features = sum(shape[1] for shape in original_shapes)
        total_proc_features = sum(shape[1] for shape in processed_shapes)
        logger.info(
            f" Data: { len(original_shapes)} views, { total_orig_features:,} → { total_proc_features:,} features"
        )


def run_data_validation(config):
    """Run data validation experiments with remote workstation integration."""
    logger.info("Starting Data Validation Experiments")

    try:
        # Self-contained data validation - no external framework dependencies
        import os
        import sys

        # Add project root for basic imports only
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(
            os.path.dirname(current_file)
        )  # Go up from experiments/ to project root
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Run comprehensive data validation with EDA plots
        logger.info("Running comprehensive data validation with EDA plots...")

        # Initialize experiment framework
        from experiments.framework import ExperimentConfig, ExperimentFramework

        exp_config = ExperimentConfig(
            experiment_name="data_validation",
            description="Data quality assessment and EDA",
            dataset="qmap_pd",
            data_dir=get_data_dir(config),
        )

        # Create data validation experiment instance
        data_val_exp = DataValidationExperiments(exp_config, logger)

        # Run comprehensive data quality assessment (includes EDA plots)
        result = data_val_exp.run_data_quality_assessment(X_list=None)

        # Save plots if they were generated
        if result and result.plots:
            logger.info(f"📊 Generated {len(result.plots)} EDA plots")
            output_dir = get_output_dir(config)
            plots_dir = Path(output_dir) / "plots"
            plots_dir.mkdir(exist_ok=True, parents=True)

            import matplotlib.pyplot as plt
            for plot_name, fig in result.plots.items():
                if fig is not None:
                    plot_path = plots_dir / f"{plot_name}.png"
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    logger.info(f"  ✅ Saved: {plot_name}.png")
                    plt.close(fig)

        logger.info("✅ Data validation with EDA completed successfully")

        # Final memory cleanup
        jax.clear_caches()
        gc.collect()
        logger.info("🧹 Memory cleanup completed")

        return {
            "status": "completed",
            "plots_generated": len(result.plots) if result and result.plots else 0,
            "analysis": result.diagnostics if result else {},
        }

    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Cleanup memory on failure
        jax.clear_caches()
        gc.collect()
        logger.info("🧹 Memory cleanup on failure completed")

        return None


@performance_optimized_experiment()
class DataValidationExperiments(ExperimentFramework):
    """Comprehensive data validation and preprocessing experiments."""

    def __init__(
        self, config: ExperimentConfig, logger: Optional[logging.Logger] = None
    ):
        """Initialize data validation experiments."""
        super().__init__(config, None, logger)

    @experiment_handler("data_quality_assessment")
    @validate_data_types(X_list=(list, type(None)))
    def run_data_quality_assessment(
        self, X_list: List[np.ndarray] = None, **kwargs
    ) -> ExperimentResult:
        """
        Run comprehensive data quality assessment.

        Parameters
        ----------
        X_list : List[np.ndarray], optional
            Pre-loaded data. If None, will load from config.
        **kwargs : Additional arguments

        Returns
        -------
        ExperimentResult : Data quality assessment results.
        """
        logger.info("Running data quality assessment")

        # Load data if not provided
        if X_list is None:
            logger.info("Loading qMAP-PD data for quality assessment...")

            # Load raw data
            config_dict = (
                self.config.to_dict()
                if hasattr(self.config, "to_dict")
                else self.config.__dict__
            )
            # Get data_dir directly from config (ExperimentConfig has it as top-level attr)
            data_dir = config_dict.get("data_dir") or "./qMAP-PD_data"
            raw_data = load_qmap_pd(
                data_dir=data_dir,
                enable_advanced_preprocessing=False,
                **config_dict.get("preprocessing_config", {}),
            )

            # Load preprocessed data for comparison
            preprocessed_data = load_qmap_pd(
                data_dir=data_dir,
                enable_advanced_preprocessing=True,
                **config_dict.get("preprocessing_config", {}),
            )

            X_list = preprocessed_data["X_list"]
        else:
            # Validate inputs if provided
            ResultValidator.validate_data_matrices(X_list)
            # Use provided data and create basic raw data structure for comparison
            raw_data = {
                "X_list": X_list,
                "view_names": [f"view_{i}" for i in range(len(X_list))],
            }
            preprocessed_data = (
                raw_data  # For now, assume provided data is preprocessed
            )

        # Analyze data quality
        results = {
            "data_summary": self._analyze_data_structure(
                raw_data, preprocessed_data
            ),
            "quality_metrics": self._assess_data_quality(
                raw_data, preprocessed_data
            ),
            "preprocessing_effects": self._analyze_preprocessing_effects(
                raw_data, preprocessed_data
            ),
        }

        # Analyze data validation results
        analysis = self._analyze_data_validation_results(
            results, raw_data, preprocessed_data
        )

        # Generate basic plots (convert existing diagnostic plots to return figures)
        plots = self._plot_data_validation_results(
            raw_data, preprocessed_data, results
        )

        # Add comprehensive data validation visualizations (focus on preprocessing
        # quality)
        advanced_plots = self._create_comprehensive_data_validation_visualizations(
            X_list, results, "data_quality_assessment"
        )
        plots.update(advanced_plots)

        return ExperimentResult(
            experiment_id="data_quality_assessment",
            config=self.config,
            model_results=results,
            diagnostics=analysis,
            plots=plots,
            status="completed",
        )

    def _analyze_data_structure(
        self, raw_data: Dict, preprocessed_data: Dict
    ) -> Dict[str, Any]:
        """Analyze basic data structure and dimensions."""
        analysis = {"raw_data": {}, "preprocessed_data": {}, "comparison": {}}

        for label, data in [
            ("raw_data", raw_data),
            ("preprocessed_data", preprocessed_data),
        ]:
            if not data or "X_list" not in data:
                analysis[label] = {
                    "n_subjects": 0,
                    "n_views": 0,
                    "features_per_view": [],
                    "total_features": 0,
                    "clinical_variables": [],
                    "n_clinical_variables": 0,
                    "data_types": [],
                    "view_names": [],
                }
                continue

            X_list = data["X_list"]
            clinical = data.get("clinical", pd.DataFrame())

            analysis[label] = {
                "n_subjects": X_list[0].shape[0] if X_list else 0,
                "n_views": len(X_list),
                "features_per_view": [X.shape[1] for X in X_list],
                "total_features": sum(X.shape[1] for X in X_list),
                "clinical_variables": (
                    list(clinical.columns) if not clinical.empty else []
                ),
                "n_clinical_variables": (
                    len(clinical.columns) if not clinical.empty else 0
                ),
                "data_types": [str(X.dtype) for X in X_list],
                "view_names": data.get("view_names", []),
            }

        # Compare raw vs preprocessed
        raw_features = analysis["raw_data"]["total_features"]
        proc_features = analysis["preprocessed_data"]["total_features"]

        analysis["comparison"] = {
            "feature_reduction_ratio": (
                (raw_features - proc_features) / raw_features if raw_features > 0 else 0
            ),
            "features_removed": raw_features - proc_features,
            "subjects_unchanged": analysis["raw_data"]["n_subjects"]
            == analysis["preprocessed_data"]["n_subjects"],
        }

        logger.info(
            f"Data structure analysis completed. "
            f"Raw: {raw_features} features, Processed: {proc_features} features"
        )

        return analysis

    def _assess_data_quality(
        self, raw_data: Dict, preprocessed_data: Dict
    ) -> Dict[str, Any]:
        """Assess various data quality metrics."""
        quality_metrics = {
            "raw_data": {},
            "preprocessed_data": {},
            "improvement_metrics": {},
        }

        for label, data in [
            ("raw_data", raw_data),
            ("preprocessed_data", preprocessed_data),
        ]:
            if not data or "X_list" not in data:
                quality_metrics[label] = {}
                continue

            X_list = data["X_list"]
            metrics = {}

            for view_idx, X in enumerate(X_list):
                view_name = data.get("view_names", [f"view_{view_idx}"])[view_idx]

                # Missing data analysis
                missing_ratio = np.isnan(X).mean()

                # Feature variance analysis
                feature_variances = np.nanvar(X, axis=0)
                low_variance_features = np.sum(feature_variances < 1e-8)

                # Outlier detection (simple Z-score based)
                z_scores = np.abs((X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0))
                outlier_ratio = np.mean(z_scores > 3)

                # Dynamic range analysis
                feature_ranges = np.nanmax(X, axis=0) - np.nanmin(X, axis=0)

                # Correlation analysis
                valid_mask = ~np.isnan(X).any(axis=0)
                if np.sum(valid_mask) > 1:
                    X_valid = X[:, valid_mask]
                    corr_matrix = np.corrcoef(X_valid.T)
                    high_corr_pairs = np.sum(np.abs(corr_matrix) > 0.95) - len(
                        corr_matrix
                    )  # Exclude diagonal
                else:
                    high_corr_pairs = 0

                metrics[view_name] = {
                    "missing_data_ratio": float(missing_ratio),
                    "low_variance_features": int(low_variance_features),
                    "low_variance_ratio": float(low_variance_features / X.shape[1]),
                    "outlier_ratio": float(outlier_ratio),
                    "mean_feature_range": float(np.nanmean(feature_ranges)),
                    "std_feature_range": float(np.nanstd(feature_ranges)),
                    "highly_correlated_pairs": int(high_corr_pairs),
                    "condition_number": (
                        float(np.linalg.cond(X_valid.T @ X_valid))
                        if np.sum(valid_mask) > 0
                        else np.inf
                    ),
                }

            quality_metrics[label] = metrics

        # Calculate improvement metrics
        improvement = {}
        for view_name in quality_metrics["raw_data"].keys():
            if view_name in quality_metrics["preprocessed_data"]:
                raw_metrics = quality_metrics["raw_data"][view_name]
                proc_metrics = quality_metrics["preprocessed_data"][view_name]

                improvement[view_name] = {
                    "missing_data_improvement": raw_metrics["missing_data_ratio"]
                    - proc_metrics["missing_data_ratio"],
                    "low_variance_reduction": raw_metrics["low_variance_features"]
                    - proc_metrics["low_variance_features"],
                    "outlier_reduction": raw_metrics["outlier_ratio"]
                    - proc_metrics["outlier_ratio"],
                    "condition_number_improvement": (
                        raw_metrics["condition_number"]
                        / proc_metrics["condition_number"]
                        if proc_metrics["condition_number"] > 0
                        else np.inf
                    ),
                }

        quality_metrics["improvement_metrics"] = improvement

        return quality_metrics

    def _analyze_preprocessing_effects(
        self, raw_data: Dict, preprocessed_data: Dict
    ) -> Dict[str, Any]:
        """Analyze the effects of preprocessing on data distribution and structure."""
        effects = {
            "distribution_changes": {},
            "dimensionality_reduction": {},
            "feature_selection": {},
            "normalization_effects": {},
        }

        if (
            not raw_data
            or "X_list" not in raw_data
            or not preprocessed_data
            or "X_list" not in preprocessed_data
        ):
            return effects

        raw_X_list = raw_data["X_list"]
        proc_X_list = preprocessed_data["X_list"]

        for view_idx, (raw_X, proc_X) in enumerate(zip(raw_X_list, proc_X_list)):
            view_name = raw_data.get("view_names", [f"view_{view_idx}"])[view_idx]

            # Distribution changes
            raw_mean = np.nanmean(raw_X, axis=0)
            raw_std = np.nanstd(raw_X, axis=0)
            proc_mean = np.nanmean(proc_X, axis=0)
            proc_std = np.nanstd(proc_X, axis=0)

            effects["distribution_changes"][view_name] = {
                "mean_shift": float(
                    np.nanmean(np.abs(proc_mean - raw_mean[: len(proc_mean)]))
                ),
                "std_change_ratio": (
                    float(np.nanmean(proc_std / raw_std[: len(proc_std)]))
                    if len(proc_std) > 0
                    else 0
                ),
                "skewness_change": self._calculate_skewness_change(raw_X, proc_X),
            }

            # Dimensionality reduction analysis
            effects["dimensionality_reduction"][view_name] = {
                "original_features": raw_X.shape[1],
                "processed_features": proc_X.shape[1],
                "reduction_ratio": (raw_X.shape[1] - proc_X.shape[1]) / raw_X.shape[1],
                "features_removed": raw_X.shape[1] - proc_X.shape[1],
            }

            # Feature selection analysis (if applicable)
            if "preprocessing" in preprocessed_data:
                preprocessing_info = preprocessed_data["preprocessing"]
                if "feature_selection" in preprocessing_info:
                    effects["feature_selection"][view_name] = preprocessing_info[
                        "feature_selection"
                    ].get(view_name, {})

            # Normalization effects
            effects["normalization_effects"][view_name] = {
                "mean_after_processing": float(np.nanmean(proc_X)),
                "std_after_processing": float(np.nanstd(proc_X)),
                "range_after_processing": float(np.nanmax(proc_X) - np.nanmin(proc_X)),
            }

        return effects

    def _calculate_skewness_change(
        self, raw_X: np.ndarray, proc_X: np.ndarray
    ) -> float:
        """Calculate change in data skewness due to preprocessing."""
        from scipy.stats import skew

        # Calculate skewness for features present in both raw and processed data
        min_features = min(raw_X.shape[1], proc_X.shape[1])

        raw_skew = skew(raw_X[:, :min_features], axis=0, nan_policy="omit")
        proc_skew = skew(proc_X[:, :min_features], axis=0, nan_policy="omit")

        return float(np.nanmean(np.abs(proc_skew - raw_skew)))

    def _generate_data_diagnostics(
        self, raw_data: Dict, preprocessed_data: Dict, output_dir: Path
    ) -> Dict[str, Any]:
        """Generate comprehensive diagnostic plots and analyses."""
        diagnostics = {
            "plots_generated": [],
            "statistical_tests": {},
            "visualizations": {},
        }

        # Feature distribution comparison plots
        self._plot_feature_distributions(raw_data, preprocessed_data, output_dir)
        diagnostics["plots_generated"].append("feature_distributions.png")

        # Missing data heatmaps
        self._plot_missing_data_patterns(raw_data, preprocessed_data, output_dir)
        diagnostics["plots_generated"].append("missing_data_patterns.png")

        # Correlation matrices
        self._plot_correlation_matrices(raw_data, preprocessed_data, output_dir)
        diagnostics["plots_generated"].append("correlation_matrices.png")

        # Dimensionality reduction visualization (PCA)
        pca_results = self._perform_pca_analysis(
            raw_data, preprocessed_data, output_dir
        )
        diagnostics["statistical_tests"]["pca_analysis"] = pca_results
        diagnostics["plots_generated"].append("pca_visualization.png")

        # Quality metrics visualization
        self._plot_quality_metrics(raw_data, preprocessed_data, output_dir)
        diagnostics["plots_generated"].append("quality_metrics.png")

        return diagnostics

    def _plot_feature_distributions(
        self, raw_data: Dict, preprocessed_data: Dict, output_dir: Path
    ):
        """Plot feature distribution comparisons."""
        if "X_list" not in raw_data or "X_list" not in preprocessed_data:
            logger.warning("Cannot plot feature distributions - missing X_list data")
            return

        fig, axes = plt.subplots(2, len(raw_data["X_list"]), figsize=(15, 10))
        if len(raw_data["X_list"]) == 1:
            axes = axes.reshape(-1, 1)

        for view_idx, (raw_X, proc_X) in enumerate(
            zip(raw_data["X_list"], preprocessed_data["X_list"])
        ):
            view_name = raw_data.get("view_names", [f"View {view_idx}"])[view_idx]

            # Raw data distribution
            feature_means_raw = np.nanmean(raw_X, axis=0)
            axes[0, view_idx].hist(feature_means_raw, bins=50, alpha=0.7, label="Raw")
            axes[0, view_idx].set_title(f"{view_name} - Raw Feature Means")
            axes[0, view_idx].set_xlabel("Feature Mean Value")
            axes[0, view_idx].set_ylabel("Frequency")

            # Preprocessed data distribution
            feature_means_proc = np.nanmean(proc_X, axis=0)
            axes[1, view_idx].hist(
                feature_means_proc,
                bins=50,
                alpha=0.7,
                label="Processed",
                color="orange",
            )
            axes[1, view_idx].set_title(f"{view_name} - Processed Feature Means")
            axes[1, view_idx].set_xlabel("Feature Mean Value")
            axes[1, view_idx].set_ylabel("Frequency")

        plt.tight_layout()
        save_plot(
            output_dir / "feature_distributions.png",
            dpi=300,
            bbox_inches="tight",
            close_after=True,
        )

    def _plot_missing_data_patterns(
        self, raw_data: Dict, preprocessed_data: Dict, output_dir: Path
    ):
        """Plot missing data patterns."""
        if "X_list" not in raw_data or "X_list" not in preprocessed_data:
            logger.warning("Cannot plot missing data patterns - missing X_list data")
            return

        fig, axes = plt.subplots(2, len(raw_data["X_list"]), figsize=(15, 8))
        if len(raw_data["X_list"]) == 1:
            axes = axes.reshape(-1, 1)

        for view_idx, (raw_X, proc_X) in enumerate(
            zip(raw_data["X_list"], preprocessed_data["X_list"])
        ):
            view_name = raw_data.get("view_names", [f"View {view_idx}"])[view_idx]

            # Sample data for visualization (to avoid memory issues)
            sample_size = min(500, raw_X.shape[0])
            sample_features = min(100, raw_X.shape[1])

            raw_sample = raw_X[:sample_size, :sample_features]
            proc_sample = proc_X[:sample_size, : min(sample_features, proc_X.shape[1])]

            # Raw data missing pattern
            missing_raw = np.isnan(raw_sample)
            axes[0, view_idx].imshow(missing_raw.T, cmap="RdYlBu", aspect="auto")
            axes[0, view_idx].set_title(f"{view_name} - Raw Missing Data")
            axes[0, view_idx].set_xlabel("Subjects")
            axes[0, view_idx].set_ylabel("Features")

            # Preprocessed data missing pattern
            missing_proc = np.isnan(proc_sample)
            axes[1, view_idx].imshow(missing_proc.T, cmap="RdYlBu", aspect="auto")
            axes[1, view_idx].set_title(f"{view_name} - Processed Missing Data")
            axes[1, view_idx].set_xlabel("Subjects")
            axes[1, view_idx].set_ylabel("Features")

        plt.tight_layout()
        save_plot(
            output_dir / "missing_data_patterns.png",
            dpi=300,
            bbox_inches="tight",
            close_after=True,
        )

    def _plot_correlation_matrices(
        self, raw_data: Dict, preprocessed_data: Dict, output_dir: Path
    ):
        """Plot correlation matrices for each view."""
        if "X_list" not in raw_data or "X_list" not in preprocessed_data:
            logger.warning("Cannot plot correlation matrices - missing X_list data")
            return

        n_views = len(raw_data["X_list"])
        fig, axes = plt.subplots(2, n_views, figsize=(5 * n_views, 10))
        if n_views == 1:
            axes = axes.reshape(-1, 1)

        for view_idx, (raw_X, proc_X) in enumerate(
            zip(raw_data["X_list"], preprocessed_data["X_list"])
        ):
            view_name = raw_data.get("view_names", [f"View {view_idx}"])[view_idx]

            # Sample features for correlation analysis
            max_features = 50  # Limit for computational efficiency

            # Raw data correlation
            raw_sample = raw_X[:, : min(max_features, raw_X.shape[1])]
            valid_mask_raw = ~np.isnan(raw_sample).any(axis=0)
            if np.sum(valid_mask_raw) > 1:
                raw_corr = np.corrcoef(raw_sample[:, valid_mask_raw].T)
                im1 = axes[0, view_idx].imshow(raw_corr, cmap="RdBu", vmin=-1, vmax=1)
                axes[0, view_idx].set_title(f"{view_name} - Raw Correlations")
                plt.colorbar(im1, ax=axes[0, view_idx])

            # Processed data correlation
            proc_sample = proc_X[:, : min(max_features, proc_X.shape[1])]
            valid_mask_proc = ~np.isnan(proc_sample).any(axis=0)
            if np.sum(valid_mask_proc) > 1:
                proc_corr = np.corrcoef(proc_sample[:, valid_mask_proc].T)
                im2 = axes[1, view_idx].imshow(proc_corr, cmap="RdBu", vmin=-1, vmax=1)
                axes[1, view_idx].set_title(f"{view_name} - Processed Correlations")
                plt.colorbar(im2, ax=axes[1, view_idx])

        plt.tight_layout()
        save_plot(
            output_dir / "correlation_matrices.png",
            dpi=300,
            bbox_inches="tight",
            close_after=True,
        )

    def _perform_pca_analysis(
        self, raw_data: Dict, preprocessed_data: Dict, output_dir: Path
    ) -> Dict[str, Any]:
        """Perform PCA analysis and visualization."""
        pca_results = {"raw_data": {}, "preprocessed_data": {}, "comparison": {}}

        if "X_list" not in raw_data or "X_list" not in preprocessed_data:
            logger.warning("Cannot perform PCA analysis - missing X_list data")
            return pca_results

        fig, axes = plt.subplots(2, len(raw_data["X_list"]), figsize=(15, 10))
        if len(raw_data["X_list"]) == 1:
            axes = axes.reshape(-1, 1)

        for view_idx, (raw_X, proc_X) in enumerate(
            zip(raw_data["X_list"], preprocessed_data["X_list"])
        ):
            view_name = raw_data.get("view_names", [f"View {view_idx}"])[view_idx]

            # PCA on raw data
            raw_X_clean = raw_X[
                ~np.isnan(raw_X).any(axis=1)
            ]  # Remove subjects with any NaN
            if raw_X_clean.shape[0] > 10 and raw_X_clean.shape[1] > 2:
                pca_raw = PCA(n_components=min(10, raw_X_clean.shape[1]))
                pca_raw.fit(raw_X_clean)

                pca_results["raw_data"][view_name] = {
                    "explained_variance_ratio": pca_raw.explained_variance_ratio_.tolist(),
                    "cumulative_variance": np.cumsum(
                        pca_raw.explained_variance_ratio_
                    ).tolist(),
                    "n_components_90_var": int(
                        np.argmax(np.cumsum(pca_raw.explained_variance_ratio_) > 0.9)
                        + 1
                    ),
                }

                # Plot explained variance
                axes[0, view_idx].plot(
                    range(1, len(pca_raw.explained_variance_ratio_) + 1),
                    np.cumsum(pca_raw.explained_variance_ratio_),
                    "o-",
                )
                axes[0, view_idx].set_title(f"{view_name} - Raw Data PCA")
                axes[0, view_idx].set_xlabel("Principal Component")
                axes[0, view_idx].set_ylabel("Cumulative Explained Variance")
                axes[0, view_idx].grid(True)

            # PCA on processed data
            proc_X_clean = proc_X[~np.isnan(proc_X).any(axis=1)]
            if proc_X_clean.shape[0] > 10 and proc_X_clean.shape[1] > 2:
                pca_proc = PCA(n_components=min(10, proc_X_clean.shape[1]))
                pca_proc.fit(proc_X_clean)

                pca_results["preprocessed_data"][view_name] = {
                    "explained_variance_ratio": pca_proc.explained_variance_ratio_.tolist(),
                    "cumulative_variance": np.cumsum(
                        pca_proc.explained_variance_ratio_
                    ).tolist(),
                    "n_components_90_var": int(
                        np.argmax(np.cumsum(pca_proc.explained_variance_ratio_) > 0.9)
                        + 1
                    ),
                }

                # Plot explained variance
                axes[1, view_idx].plot(
                    range(1, len(pca_proc.explained_variance_ratio_) + 1),
                    np.cumsum(pca_proc.explained_variance_ratio_),
                    "o-",
                    color="orange",
                )
                axes[1, view_idx].set_title(f"{view_name} - Processed Data PCA")
                axes[1, view_idx].set_xlabel("Principal Component")
                axes[1, view_idx].set_ylabel("Cumulative Explained Variance")
                axes[1, view_idx].grid(True)

        plt.tight_layout()
        save_plot(
            output_dir / "pca_visualization.png",
            dpi=300,
            bbox_inches="tight",
            close_after=True,
        )

        return pca_results

    def _plot_quality_metrics(
        self, raw_data: Dict, preprocessed_data: Dict, output_dir: Path
    ):
        """Plot data quality metrics comparison."""
        if "X_list" not in raw_data or "X_list" not in preprocessed_data:
            logger.warning("Cannot plot quality metrics - missing X_list data")
            return

        quality_raw = self._assess_data_quality(raw_data, {})["raw_data"]
        quality_proc = self._assess_data_quality({}, preprocessed_data)[
            "preprocessed_data"
        ]

        metrics = ["missing_data_ratio", "low_variance_ratio", "outlier_ratio"]
        len(raw_data["X_list"])

        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

        for metric_idx, metric in enumerate(metrics):
            view_names = []
            raw_values = []
            proc_values = []

            for view_name in quality_raw.keys():
                if view_name in quality_proc:
                    view_names.append(view_name)
                    raw_values.append(quality_raw[view_name][metric])
                    proc_values.append(quality_proc[view_name][metric])

            x = np.arange(len(view_names))
            width = 0.35

            axes[metric_idx].bar(
                x - width / 2, raw_values, width, label="Raw", alpha=0.7
            )
            axes[metric_idx].bar(
                x + width / 2, proc_values, width, label="Processed", alpha=0.7
            )

            axes[metric_idx].set_xlabel("Views")
            axes[metric_idx].set_ylabel(metric.replace("_", " ").title())
            axes[metric_idx].set_title(
                f'Data Quality: {metric.replace("_", " ").title()}'
            )
            axes[metric_idx].set_xticks(x)
            axes[metric_idx].set_xticklabels(view_names)
            axes[metric_idx].legend()

        plt.tight_layout()
        save_plot(
            output_dir / "quality_metrics.png",
            dpi=300,
            bbox_inches="tight",
            close_after=True,
        )

    def _save_data_summaries(
        self, raw_data: Dict, preprocessed_data: Dict, output_dir: Path
    ):
        """Save detailed data summaries to files."""
        if "X_list" not in raw_data or "X_list" not in preprocessed_data:
            logger.warning("Cannot save data summaries - missing X_list data")
            return

        # Raw data summary
        raw_summary = self._create_detailed_summary(raw_data, "raw")
        save_csv(raw_summary, output_dir / "raw_data_summary.csv", index=False)

        # Preprocessed data summary
        proc_summary = self._create_detailed_summary(preprocessed_data, "preprocessed")
        save_csv(
            proc_summary, output_dir / "preprocessed_data_summary.csv", index=False
        )

        # Comparison summary
        comparison_df = self._create_comparison_summary(raw_data, preprocessed_data)
        save_csv(comparison_df, output_dir / "data_comparison_summary.csv", index=False)

    def _create_detailed_summary(self, data: Dict, data_type: str) -> pd.DataFrame:
        """Create detailed summary DataFrame for data."""
        if "X_list" not in data:
            return pd.DataFrame()

        summaries = []

        for view_idx, X in enumerate(data["X_list"]):
            view_name = data.get("view_names", [f"view_{view_idx}"])[view_idx]

            summary = {
                "data_type": data_type,
                "view_name": view_name,
                "n_subjects": X.shape[0],
                "n_features": X.shape[1],
                "missing_ratio": np.isnan(X).mean(),
                "mean_value": np.nanmean(X),
                "std_value": np.nanstd(X),
                "min_value": np.nanmin(X),
                "max_value": np.nanmax(X),
                "dtype": str(X.dtype),
            }

            summaries.append(summary)

        return pd.DataFrame(summaries)

    def _create_comparison_summary(
        self, raw_data: Dict, preprocessed_data: Dict
    ) -> pd.DataFrame:
        """Create comparison summary between raw and preprocessed data."""
        if "X_list" not in raw_data or "X_list" not in preprocessed_data:
            return pd.DataFrame()

        comparisons = []

        for view_idx, (raw_X, proc_X) in enumerate(
            zip(raw_data["X_list"], preprocessed_data["X_list"])
        ):
            view_name = raw_data.get("view_names", [f"view_{view_idx}"])[view_idx]

            comparison = {
                "view_name": view_name,
                "features_raw": raw_X.shape[1],
                "features_processed": proc_X.shape[1],
                "feature_reduction_count": raw_X.shape[1] - proc_X.shape[1],
                "feature_reduction_ratio": (raw_X.shape[1] - proc_X.shape[1])
                / raw_X.shape[1],
                "missing_data_improvement": np.isnan(raw_X).mean()
                - np.isnan(proc_X).mean(),
                "mean_change": np.nanmean(proc_X)
                - np.nanmean(raw_X[:, : proc_X.shape[1]]),
                "std_change": np.nanstd(proc_X)
                - np.nanstd(raw_X[:, : proc_X.shape[1]]),
            }

            comparisons.append(comparison)

        return pd.DataFrame(comparisons)

    @experiment_handler("preprocessing_comparison")
    @validate_data_types(X_list=(list, type(None)))
    def run_preprocessing_comparison(
        self, X_list: List[np.ndarray] = None, **kwargs
    ) -> ExperimentResult:
        """
        Compare different preprocessing strategies.

        Parameters
        ----------
        X_list : List[np.ndarray], optional
            Pre-loaded data. If None, will load from config.
        **kwargs : Additional arguments

        Returns
        -------
        ExperimentResult : Preprocessing comparison results.
        """
        logger.info("Running preprocessing strategy comparison")

        try:
            from data.preprocessing_integration import apply_preprocessing_to_pipeline
            from core.config_utils import ConfigHelper, ConfigAccessor

            # Get strategy configurations from config
            config_dict = ConfigHelper.to_dict(self.config)
            # Get data_dir directly from config (ExperimentConfig has it as top-level attr)
            data_dir = config_dict.get("data_dir") or "./qMAP-PD_data"
            strategies_config = config_dict.get("data_validation", {}).get("preprocessing_strategies", {})

            if not strategies_config:
                logger.warning("No preprocessing strategies found in config")
                strategies_config = {
                    "minimal": {"enable_advanced_preprocessing": False},
                    "standard": {"enable_advanced_preprocessing": True, "feature_selection_method": "variance"},
                    "statistical": {"enable_advanced_preprocessing": True, "feature_selection_method": "statistical"},
                    "mutual_info": {"enable_advanced_preprocessing": True, "feature_selection_method": "mutual_info"},
                    "combined": {"enable_advanced_preprocessing": True, "feature_selection_method": "combined"}
                }

            logger.info(f"Comparing {len(strategies_config)} preprocessing strategies: {list(strategies_config.keys())}")

            strategy_results = {}
            strategy_data = {}

            # Test each preprocessing strategy
            for strategy_name, strategy_params in strategies_config.items():
                logger.info(f"🧠 Testing preprocessing strategy: {strategy_name}")

                try:
                    # Create a temporary config with this strategy
                    temp_config_dict = config_dict.copy()
                    temp_config_dict["preprocessing"] = strategy_params
                    temp_config = ConfigAccessor(temp_config_dict)

                    # Apply preprocessing with this strategy
                    X_list_strategy, preprocessing_info = apply_preprocessing_to_pipeline(
                        config=temp_config,
                        data_dir=data_dir,
                        auto_select_strategy=False,
                        preferred_strategy=strategy_params.get("strategy", strategy_name)
                    )

                    # Store the data and preprocessing info
                    strategy_data[strategy_name] = {
                        "X_list": X_list_strategy,
                        "preprocessing_info": preprocessing_info
                    }

                    # Analyze this strategy's results
                    strategy_results[strategy_name] = {
                        "data_structure": {
                            "n_subjects": X_list_strategy[0].shape[0] if X_list_strategy else 0,
                            "n_views": len(X_list_strategy),
                            "features_per_view": [X.shape[1] for X in X_list_strategy],
                            "total_features": sum(X.shape[1] for X in X_list_strategy)
                        },
                        "quality_metrics": self._assess_strategy_quality(X_list_strategy),
                        "preprocessing_info": preprocessing_info
                    }

                    logger.info(f"✅ Strategy {strategy_name}: {strategy_results[strategy_name]['data_structure']['total_features']} total features")

                except Exception as e:
                    logger.error(f"❌ Strategy {strategy_name} failed: {e}")
                    strategy_results[strategy_name] = {"error": str(e)}
                    continue

            # Compare strategies
            comparison_analysis = self._compare_preprocessing_strategies(strategy_data)
            recommendations = self._generate_preprocessing_recommendations(strategy_results)

            # Generate comparison plots
            plots = {}
            try:
                self._plot_strategy_comparison(strategy_data, Path("/tmp"))
                plots["strategy_comparison"] = "preprocessing_strategy_comparison.png"
            except Exception as e:
                logger.warning(f"Could not generate comparison plots: {e}")

            return ExperimentResult(
                experiment_id="preprocessing_comparison",
                config=self.config,
                model_results={
                    "strategy_results": strategy_results,
                    "strategy_comparison": comparison_analysis,
                    "tested_strategies": list(strategies_config.keys())
                },
                diagnostics={
                    "comparison_summary": comparison_analysis,
                    "recommendations": recommendations,
                    "best_strategy": recommendations.get("best_strategy"),
                    "strategies_tested": len(strategy_results)
                },
                plots=plots,
                status="completed",
            )

        except Exception as e:
            logger.error(f"Preprocessing comparison failed: {e}")
            return ExperimentResult(
                experiment_id="preprocessing_comparison",
                config=self.config,
                model_results={"error": str(e)},
                diagnostics={"error": str(e)},
                plots={},
                status="failed",
                error_message=str(e),
            )

    def _assess_strategy_quality(self, X_list: List[np.ndarray]) -> Dict[str, Any]:
        """Assess quality metrics for a specific preprocessing strategy."""
        quality = {}

        try:
            for view_idx, X in enumerate(X_list):
                view_name = f"view_{view_idx}"

                # Missing data analysis
                missing_ratio = np.isnan(X).mean()

                # Feature variance analysis
                feature_variances = np.nanvar(X, axis=0)
                low_variance_features = np.sum(feature_variances < 1e-8)

                # Outlier detection (simple Z-score based)
                z_scores = np.abs((X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0))
                outlier_ratio = np.mean(z_scores > 3)

                # Condition number (numerical stability)
                valid_mask = ~np.isnan(X).any(axis=0)
                if np.sum(valid_mask) > 1:
                    X_valid = X[:, valid_mask]
                    try:
                        condition_number = float(np.linalg.cond(X_valid.T @ X_valid))
                    except:
                        condition_number = np.inf
                else:
                    condition_number = np.inf

                quality[view_name] = {
                    "missing_data_ratio": float(missing_ratio),
                    "low_variance_features": int(low_variance_features),
                    "low_variance_ratio": float(low_variance_features / X.shape[1]),
                    "outlier_ratio": float(outlier_ratio),
                    "condition_number": condition_number,
                    "mean_variance": float(np.nanmean(feature_variances)),
                    "n_features": X.shape[1]
                }

        except Exception as e:
            logger.warning(f"Error assessing strategy quality: {e}")

        return quality

    def _compare_preprocessing_strategies(self, strategy_data: Dict) -> Dict[str, Any]:
        """Compare different preprocessing strategies."""
        comparison = {}

        if not strategy_data:
            logger.warning("No strategy data available for comparison")
            return comparison

        # Get strategy names
        strategy_names = list(strategy_data.keys())
        logger.info(
            f"Comparing { len(strategy_names)} preprocessing strategies: {strategy_names}"
        )

        # Compare data dimensions
        comparison["data_dimensions"] = {}
        for strategy_name, data in strategy_data.items():
            try:
                if "X_list" not in data:
                    logger.error(
                        f"Strategy {strategy_name} missing X_list. Keys: { list( data.keys())}"
                    )
                    continue

                X_list = data["X_list"]
                comparison["data_dimensions"][strategy_name] = {
                    "total_features": sum(X.shape[1] for X in X_list),
                    "features_per_view": [X.shape[1] for X in X_list],
                }
            except Exception as e:
                logger.error(f"Error processing dimensions for {strategy_name}: {e}")
                continue

        # Compare data quality metrics
        comparison["quality_comparison"] = {}
        if strategy_data:
            # Find a strategy that has X_list to determine number of views
            valid_strategy_data = [
                (name, data) for name, data in strategy_data.items() if "X_list" in data
            ]

            if valid_strategy_data:
                num_views = len(valid_strategy_data[0][1]["X_list"])

                for view_idx in range(num_views):
                    view_name = f"view_{view_idx}"
                    comparison["quality_comparison"][view_name] = {}

                    for strategy_name, data in strategy_data.items():
                        try:
                            if "X_list" not in data:
                                continue

                            X = data["X_list"][view_idx]
                            comparison["quality_comparison"][view_name][
                                strategy_name
                            ] = {
                                "missing_ratio": float(np.isnan(X).mean()),
                                "feature_variance_mean": float(
                                    np.nanvar(X, axis=0).mean()
                                ),
                                "condition_number": float(np.linalg.cond(X.T @ X)),
                            }
                        except Exception as e:
                            logger.error(
                                f"Error computing quality metrics for {strategy_name}, view {view_idx}: {e}"
                            )
                            continue
            else:
                logger.warning("No valid strategy data found for quality comparison")

        return comparison

    def _generate_preprocessing_recommendations(
        self, strategy_results: Dict
    ) -> Dict[str, Any]:
        """Generate preprocessing recommendations based on results."""
        recommendations = {
            "best_strategy": None,
            "rationale": [],
            "specific_recommendations": {},
        }

        # Simple scoring system (would be more sophisticated in practice)
        scores = {}

        for strategy_name, results in strategy_results.items():
            if "error" in results:
                scores[strategy_name] = 0
                continue

            score = 0

            # Reward strategies that preserve more data
            data_structure = results.get("data_structure", {})
            if data_structure:
                score += data_structure.get("n_subjects", 0) * 0.1
                score += sum(data_structure.get("features_per_view", [])) * 0.001

            # Reward strategies with better quality metrics
            quality_metrics = results.get("quality_metrics", {})
            for view_metrics in quality_metrics.values():
                score -= view_metrics.get("missing_data_ratio", 1.0) * 100
                score -= view_metrics.get("low_variance_ratio", 1.0) * 50
                score -= view_metrics.get("outlier_ratio", 1.0) * 25

            scores[strategy_name] = score

        if scores:
            best_strategy = max(scores.keys(), key=lambda k: scores[k])
            recommendations["best_strategy"] = best_strategy

            recommendations["rationale"] = [
                f"Strategy '{best_strategy}' achieved the highest overall score ({ scores[best_strategy]:.2f})",
                "Scoring based on data preservation, missing data handling, and feature quality",
            ]

        return recommendations

    def _plot_strategy_comparison(self, strategy_data: Dict, output_dir: Path):
        """Create comparison plots for different preprocessing strategies."""
        if not strategy_data:
            logger.warning("No strategy data available for plotting")
            return

        # Filter to valid strategies only
        valid_strategies = {
            name: data for name, data in strategy_data.items() if "X_list" in data
        }
        if not valid_strategies:
            logger.warning("No valid strategy data with X_list for plotting")
            return

        len(valid_strategies)
        strategy_names = list(valid_strategies.keys())

        # Feature count comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Total features per strategy
        total_features = []
        for strategy_name in strategy_names:
            data = valid_strategies[strategy_name]
            total_features.append(sum(X.shape[1] for X in data["X_list"]))

        axes[0, 0].bar(strategy_names, total_features)
        axes[0, 0].set_title("Total Features by Strategy")
        axes[0, 0].set_ylabel("Number of Features")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. Missing data comparison
        if len(valid_strategies) > 0:
            first_data = list(valid_strategies.values())[0]
            n_views = len(first_data["X_list"])

            missing_ratios = {strategy: [] for strategy in strategy_names}

            for view_idx in range(n_views):
                for strategy_name in strategy_names:
                    X = valid_strategies[strategy_name]["X_list"][view_idx]
                    missing_ratios[strategy_name].append(np.isnan(X).mean())

            x = np.arange(n_views)
            width = 0.8 / len(strategy_names)

            for i, strategy_name in enumerate(strategy_names):
                axes[0, 1].bar(
                    x + i * width,
                    missing_ratios[strategy_name],
                    width,
                    label=strategy_name,
                    alpha=0.7,
                )

            axes[0, 1].set_title("Missing Data by View and Strategy")
            axes[0, 1].set_xlabel("View")
            axes[0, 1].set_ylabel("Missing Data Ratio")
            axes[0, 1].set_xticks(x + width * (len(strategy_names) - 1) / 2)
            axes[0, 1].set_xticklabels([f"View {i}" for i in range(n_views)])
            axes[0, 1].legend()

        # 3. Data variance comparison
        variance_data = {strategy: [] for strategy in strategy_names}
        for strategy_name in strategy_names:
            data = strategy_data[strategy_name]
            for X in data["X_list"]:
                variance_data[strategy_name].append(np.nanvar(X))

        axes[1, 0].boxplot(
            [variance_data[strategy] for strategy in strategy_names],
            labels=strategy_names,
        )
        axes[1, 0].set_title("Data Variance Distribution by Strategy")
        axes[1, 0].set_ylabel("Variance")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # 4. Feature reduction comparison
        if "minimal" in strategy_data:
            baseline_features = sum(
                X.shape[1] for X in strategy_data["minimal"]["X_list"]
            )
            reduction_ratios = []

            for strategy_name in strategy_names:
                current_features = sum(
                    X.shape[1] for X in strategy_data[strategy_name]["X_list"]
                )
                reduction_ratio = (
                    baseline_features - current_features
                ) / baseline_features
                reduction_ratios.append(reduction_ratio)

            axes[1, 1].bar(strategy_names, reduction_ratios)
            axes[1, 1].set_title("Feature Reduction vs Minimal Strategy")
            axes[1, 1].set_ylabel("Reduction Ratio")
            axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        save_plot(
            output_dir / "preprocessing_strategy_comparison.png",
            dpi=300,
            bbox_inches="tight",
            close_after=True,
        )

        logger.info("Preprocessing strategy comparison plots saved")

    def _analyze_data_validation_results(
        self, results: Dict, raw_data: Dict, preprocessed_data: Dict
    ) -> Dict:
        """Analyze data validation results to provide summary insights."""
        analysis = {
            "overall_quality_score": 0.0,
            "preprocessing_improvement": {},
            "data_completeness": {},
            "quality_recommendations": [],
        }

        try:
            # Calculate overall quality score based on completeness and variance
            if "quality_metrics" in results:
                quality_metrics = results["quality_metrics"]
                if "preprocessed_data" in quality_metrics:
                    completeness = quality_metrics["preprocessed_data"].get(
                        "completeness", 0
                    )
                    analysis["overall_quality_score"] = completeness

            # Assess preprocessing improvement
            if "preprocessing_effects" in results:
                effects = results["preprocessing_effects"]
                analysis["preprocessing_improvement"] = {
                    "feature_reduction": effects.get("dimensionality_reduction", {}),
                    "normalization_impact": effects.get("normalization_effects", {}),
                    "missing_data_handled": effects.get("feature_selection", {}),
                }

            # Data completeness assessment
            if preprocessed_data and "X_list" in preprocessed_data:
                total_elements = sum(X.size for X in preprocessed_data["X_list"])
                missing_elements = sum(
                    np.isnan(X).sum() for X in preprocessed_data["X_list"]
                )
                analysis["data_completeness"] = {
                    "completeness_ratio": 1 - (missing_elements / total_elements),
                    "total_features": sum(
                        X.shape[1] for X in preprocessed_data["X_list"]
                    ),
                    "total_subjects": (
                        preprocessed_data["X_list"][0].shape[0]
                        if preprocessed_data["X_list"]
                        else 0
                    ),
                }

            # Generate quality recommendations
            if analysis["overall_quality_score"] < 0.8:
                analysis["quality_recommendations"].append(
                    "Consider additional preprocessing steps"
                )
            if analysis["data_completeness"].get("completeness_ratio", 1) < 0.95:
                analysis["quality_recommendations"].append(
                    "Address missing data issues"
                )

        except Exception as e:
            logger.warning(f"Failed to analyze data validation results: {e}")

        return analysis

    def _plot_data_validation_results(
        self, raw_data: Dict, preprocessed_data: Dict, results: Dict
    ) -> Dict:
        """Generate basic data validation plots and return as matplotlib figures."""
        logger.info("📊 Generating data validation plots...")
        plots = {}

        try:
            # Data distribution comparison plot (only for clinical data, skip imaging)
            if raw_data.get("X_list") and preprocessed_data.get("X_list"):
                view_names = raw_data.get("view_names", [])

                # Only plot clinical view (last view, typically smallest)
                clinical_views = []
                for view_idx, (raw_X, proc_X) in enumerate(
                    zip(raw_data["X_list"], preprocessed_data["X_list"])
                ):
                    view_name = view_names[view_idx] if view_idx < len(view_names) else f"view_{view_idx}"
                    # Skip large imaging views - only plot clinical data
                    if raw_X.shape[1] < 100 or "clinical" in view_name.lower():
                        clinical_views.append((view_idx, view_name, raw_X, proc_X))

                if clinical_views:
                    n_views = len(clinical_views)
                    logger.info(f"   Creating distribution histograms for {n_views} clinical view(s), skipping {len(raw_data['X_list']) - n_views} imaging views")
                    fig, axes = plt.subplots(2, n_views, figsize=(5*n_views, 10))
                    if n_views == 1:
                        axes = axes.reshape(-1, 1)
                    fig.suptitle("Data Distribution: Raw vs Preprocessed (Clinical Data Only)", fontsize=16)

                    for plot_idx, (view_idx, view_name, raw_X, proc_X) in enumerate(clinical_views):
                        # Raw data distribution
                        axes[0, plot_idx].hist(
                            raw_X.flatten(), bins=50, alpha=0.7, color="red"
                        )
                        axes[0, plot_idx].set_title(f"Raw {view_name}")
                        axes[0, plot_idx].set_ylabel("Frequency")

                        # Preprocessed data distribution
                        axes[1, plot_idx].hist(
                            proc_X.flatten(), bins=50, alpha=0.7, color="blue"
                        )
                        axes[1, plot_idx].set_title(f"Preprocessed {view_name}")
                        axes[1, plot_idx].set_ylabel("Frequency")

                    plt.tight_layout()
                    plots["data_distribution_comparison"] = fig
                    logger.info(f"   ✅ Distribution histograms created")

            # Quality metrics summary plot
            if "quality_metrics" in results:
                logger.info("   Creating quality metrics summary plot...")
                fig, ax = plt.subplots(figsize=(10, 6))

                metrics_names = []
                metrics_values = []

                for data_type, metrics in results["quality_metrics"].items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                metrics_names.append(f"{data_type}_{metric_name}")
                                metrics_values.append(value)

                if metrics_names:
                    ax.bar(metrics_names, metrics_values)
                    ax.set_title("Data Quality Metrics Summary")
                    ax.set_ylabel("Metric Value")
                    ax.tick_params(axis="x", rotation=45)
                    plt.tight_layout()
                    plots["quality_metrics_summary"] = fig
                    logger.info(f"   ✅ Quality metrics plot created ({len(metrics_names)} metrics)")

        except Exception as e:
            logger.warning(f"Failed to create data validation plots: {e}")

        logger.info(f"📊 Basic data validation plots completed: {len(plots)} plots generated")
        return plots

    def _create_comprehensive_data_validation_visualizations(
        self, X_list: List[np.ndarray], results: Dict, experiment_name: str
    ) -> Dict:
        """Create comprehensive data validation visualizations focusing on preprocessing quality."""
        advanced_plots = {}

        try:
            logger.info(
                f"🎨 Creating comprehensive data validation visualizations for {experiment_name}"
            )

            # Import visualization system
            from core.config_utils import ConfigAccessor
            from visualization.manager import VisualizationManager

            # Create a data validation focused config for visualization
            viz_config = ConfigAccessor(
                {
                    "visualization": {
                        "create_brain_viz": False,  # Focus on preprocessing, not brain maps
                        "output_format": ["png", "pdf"],
                        "dpi": 300,
                        "data_validation_focus": True,
                    },
                    "output_dir": f"/tmp/data_validation_viz_{experiment_name}",
                }
            )

            # Initialize visualization manager
            viz_manager = VisualizationManager(viz_config)

            # Prepare data validation structure for visualizations
            data = {
                "X_list": X_list,
                "view_names": [f"view_{i}" for i in range(len(X_list))],
                "n_subjects": X_list[0].shape[0],
                "view_dimensions": [X.shape[1] for X in X_list],
                "preprocessing": {
                    "status": "completed",
                    "strategy": "comprehensive_validation",
                    "quality_results": results,
                },
            }

            # Create mock analysis results for visualization (data validation doesn't
            # have factor analysis)
            analysis_results = {
                "data_validation": True,
                "quality_assessment": results,
                "preprocessing_validation": True,
            }

            # Focus on preprocessing visualizations
            logger.info("   Creating preprocessing quality visualizations...")
            viz_manager.preprocessing_viz.create_plots(
                data["preprocessing"], viz_manager.plot_dir
            )
            logger.info("   ✅ Preprocessing visualizations created")

            # Extract the generated plots and convert to matplotlib figures
            logger.info("   Loading and converting generated plots to figures...")
            if hasattr(viz_manager, "plot_dir") and viz_manager.plot_dir.exists():
                plot_files = list(viz_manager.plot_dir.glob("**/*.png"))
                logger.info(f"   Found {len(plot_files)} plot files to convert")

                for plot_file in plot_files:
                    plot_name = f"data_validation_{plot_file.stem}"

                    try:
                        import matplotlib.image as mpimg
                        import matplotlib.pyplot as plt

                        fig, ax = plt.subplots(figsize=(12, 8))
                        img = mpimg.imread(str(plot_file))
                        ax.imshow(img)
                        ax.axis("off")
                        ax.set_title(f"Data Validation: {plot_name}", fontsize=14)

                        advanced_plots[plot_name] = fig

                    except Exception as e:
                        logger.warning(
                            f"Could not load data validation plot {plot_name}: {e}"
                        )

                logger.info(
                    f"   ✅ Created {len(plot_files)} comprehensive data validation visualizations"
                )

            else:
                logger.warning(
                    "Data validation visualization manager did not create plot directory"
                )

        except Exception as e:
            logger.warning(
                f"Failed to create comprehensive data validation visualizations: {e}"
            )

        logger.info(f"🎨 Comprehensive data validation plots completed: {len(advanced_plots)} advanced plots generated")
        return advanced_plots

    def _create_failure_result(
        self, experiment_name: str, error_message: str
    ) -> ExperimentResult:
        """Create a failure result for data validation experiments."""
        return ExperimentResult(
            experiment_id=experiment_name,
            config=self.config,
            model_results={},
            diagnostics={"error": error_message},
            plots={},
            status="failed",
            error_message=error_message,
        )
