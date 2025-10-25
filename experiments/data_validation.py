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

from core.config_utils import get_data_dir, get_output_dir
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
from visualization.preprocessing_plots import _format_view_name

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
            f"   Features: {fr['total_before']:,} → {fr['total_after']:,} ({fr['reduction_ratio']:.3f} ratio)"
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
            preprocessing_config=config.get("preprocessing", {}),  # Pass preprocessing config!
            max_tree_depth=config.get("mcmc", {}).get("max_tree_depth"),  # For semantic naming
        )

        # Initialize experiment framework
        framework = ExperimentFramework(get_output_dir(config))

        # Define the experiment function (similar to other experiments)
        def data_validation_experiment(config, output_dir, **kwargs):
            logger.info("Running comprehensive data validation...")

            # Create data validation experiment instance
            data_val_exp = DataValidationExperiments(exp_config, logger)

            # Store the experiment-specific output_dir for use in preprocessing
            data_val_exp.base_output_dir = output_dir

            # Run comprehensive data quality assessment (includes EDA plots)
            assessment_result = data_val_exp.run_data_quality_assessment(X_list=None)

            logger.info("✅ Data validation with EDA completed!")
            logger.info(f"   Plots generated: {len(assessment_result.plots) if assessment_result.plots else 0}")

            # Return results in framework-compatible format
            return {
                "status": "completed",
                "model_results": assessment_result.model_results,
                "diagnostics": assessment_result.diagnostics,
                "plots": assessment_result.plots,
            }

        # Run experiment using framework (auto-saves plots and results)
        result = framework.run_experiment(
            experiment_function=data_validation_experiment,
            config=exp_config,
            model_results={},
        )

        logger.info("✅ Data validation completed successfully")

        # Final memory cleanup
        jax.clear_caches()
        gc.collect()
        logger.info("🧹 Memory cleanup completed")

        return result

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

            # Extract ROI selection, confound regression, and feature selection from preprocessing_config
            # (This comes from ExperimentConfig.preprocessing_config field, which was set from main config["preprocessing"])
            preprocessing_config = config_dict.get("preprocessing_config", {})
            select_rois = preprocessing_config.get("select_rois")
            regress_confounds = preprocessing_config.get("regress_confounds")
            feature_selection_method = preprocessing_config.get("feature_selection_method")

            # Log preprocessing settings
            if select_rois:
                self.logger.info(f"   ROI selection: {select_rois}")
            if regress_confounds:
                self.logger.info(f"   Confound regression: {regress_confounds}")
            if feature_selection_method:
                self.logger.info(f"   Feature selection: {feature_selection_method}")

            # Extract only the parameters that load_qmap_pd() accepts
            # (filter out experiment-level config like 'strategy', 'select_rois', etc.)
            # NOTE: PCA parameters are NOT included here because load_qmap_pd() doesn't support them.
            # PCA is handled by apply_preprocessing_to_pipeline() which is called by downstream experiments.
            valid_load_params = {
                "enable_advanced_preprocessing",
                "enable_spatial_processing",
                "imputation_strategy",
                "feature_selection_method",
                "n_top_features",
                "missing_threshold",
                "variance_threshold",
                "target_variable",
                "cross_validate_sources",
                "optimize_preprocessing",
                "spatial_imputation",
                "roi_based_selection",
                "harmonize_scanners",
                "scanner_info_col",
                "qc_outlier_threshold",
                "spatial_neighbor_radius",
                "min_voxel_distance",
                "select_rois",
                "regress_confounds",
                "drop_confounds_from_clinical",
            }

            # Filter preprocessing config to only include valid parameters
            raw_config = {k: v for k, v in preprocessing_config.items() if k in valid_load_params}
            raw_config["enable_advanced_preprocessing"] = False

            raw_data = load_qmap_pd(
                data_dir=data_dir,
                **raw_config,  # Pass filtered preprocessing config with advanced preprocessing disabled
            )

            # Load preprocessed data for comparison
            # Check if PCA is requested - if so, use apply_preprocessing_to_pipeline
            # because load_qmap_pd() doesn't support PCA parameters
            use_pca = preprocessing_config.get("enable_pca", False)
            logger.info(f"   PCA enabled in preprocessing_config: {use_pca}")
            if use_pca:
                logger.info(f"   PCA variance threshold: {preprocessing_config.get('pca_variance_threshold')}")
                logger.info(f"   PCA n_components: {preprocessing_config.get('pca_n_components')}")

            if use_pca:
                # Use apply_preprocessing_to_pipeline for full preprocessing including PCA
                from data.preprocessing_integration import apply_preprocessing_to_pipeline
                logger.debug("   Using apply_preprocessing_to_pipeline for PCA support...")

                # Use experiment-specific output_dir if available, otherwise fall back to general output_dir
                preprocessing_output_dir = getattr(self, 'base_output_dir', None) or get_output_dir(config)

                X_list, preprocessing_info = apply_preprocessing_to_pipeline(
                    config={"preprocessing": preprocessing_config, "data": {"data_dir": data_dir}},
                    data_dir=data_dir,
                    auto_select_strategy=False,
                    preferred_strategy=preprocessing_config.get("strategy", "standard"),
                    output_dir=preprocessing_output_dir,
                )

                logger.info(f"   ✅ Filtered position lookups saved to: {preprocessing_output_dir}/position_lookup_filtered")

                # Convert to load_qmap_pd format for compatibility
                preprocessed_data = {
                    "X_list": X_list,
                    "view_names": preprocessing_info.get("data_summary", {}).get("view_names", [f"view_{i}" for i in range(len(X_list))]),
                    "feature_names": preprocessing_info.get("data_summary", {}).get("original_data", {}).get("feature_names", {}),
                    "preprocessing_applied": True,
                    "preprocessing_info": preprocessing_info,
                }
            else:
                # Use load_qmap_pd for standard preprocessing without PCA
                preprocessed_config = {k: v for k, v in preprocessing_config.items() if k in valid_load_params}
                preprocessed_config["enable_advanced_preprocessing"] = True

                # Use experiment-specific output_dir if available, otherwise fall back to general output_dir
                preprocessing_output_dir = getattr(self, 'base_output_dir', None) or get_output_dir(config)

                preprocessed_data = load_qmap_pd(
                    data_dir=data_dir,
                    output_dir=preprocessing_output_dir,
                    **preprocessed_config,  # Pass filtered preprocessing config with advanced preprocessing enabled
                )

                logger.info(f"   ✅ Filtered position lookups saved to: {preprocessing_output_dir}/position_lookup_filtered")

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

        # Create filtered position lookups for imaging views
        # This maps preprocessed voxels back to 3D brain coordinates
        # Use experiment-specific output_dir if available
        position_output_dir = getattr(self, 'base_output_dir', None) or get_output_dir(config)
        logger.info(f"   Creating filtered position lookups with output_dir: {position_output_dir}")
        filtered_position_paths = self._create_filtered_position_lookups(
            raw_data, preprocessed_data, data_dir, position_output_dir
        )

        if filtered_position_paths:
            logger.info(f"   ✅ Filtered position lookups saved to: {position_output_dir}/filtered_position_lookups")
            logger.info(f"   Files created: {list(filtered_position_paths.values())}")

        # Analyze data quality
        results = {
            "data_summary": self._analyze_data_structure(
                raw_data, preprocessed_data
            ),
            "quality_metrics": self._assess_data_quality(
                raw_data, preprocessed_data
            ),
            "snr_analysis": self._estimate_snr(
                preprocessed_data.get("X_list", []),
                preprocessed_data.get("view_names", [])
            ) if preprocessed_data.get("X_list") else (
                logger.warning("⚠️ Skipping SNR analysis - preprocessed_data['X_list'] is empty or None") or {}
            ),
            "preprocessing_effects": self._analyze_preprocessing_effects(
                raw_data, preprocessed_data
            ),
            # Include preprocessed data for downstream experiments
            "preprocessed_data": {
                "X_list": preprocessed_data.get("X_list"),
                "preprocessing_info": {
                    "strategy": "advanced" if preprocessed_data.get("preprocessing_applied") else "basic",
                    "select_rois": select_rois,
                    "regress_confounds": regress_confounds,
                    "feature_selection_method": feature_selection_method,
                    "variance_threshold": preprocessing_config.get("variance_threshold"),
                    "data_summary": {
                        "view_names": preprocessed_data.get("view_names", []),
                        "original_data": {
                            "feature_names": preprocessed_data.get("feature_names", {}),
                        },
                    },
                    "filtered_position_lookups": filtered_position_paths,
                },
            },
        }

        # =====================================================================
        # MAD THRESHOLD EXPLORATORY DATA ANALYSIS
        # =====================================================================
        # Run MAD threshold analysis on raw imaging data BEFORE preprocessing
        # This informs optimal threshold selection for outlier detection
        mad_analysis_results = self._run_mad_threshold_eda(
            raw_data, output_dir=self.base_output_dir
        )
        results["mad_threshold_analysis"] = mad_analysis_results

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

        # Add SNR analysis plots
        if results.get("snr_analysis"):
            logger.info("📊 Creating SNR analysis plots...")
            snr_plots = self._plot_snr_analysis(
                X_list,
                preprocessed_data.get("view_names", [f"view_{i}" for i in range(len(X_list))]),
                results["snr_analysis"]
            )
            advanced_plots.update(snr_plots)

        # Log what plots we have before and after merging
        logger.debug(f"Basic plots keys: {list(plots.keys())}")
        logger.debug(f"Advanced plots keys: {list(advanced_plots.keys())}")

        plots.update(advanced_plots)

        logger.debug(f"Merged plots keys: {list(plots.keys())}")
        logger.info(f"📊 Total plots ready for saving: {len(plots)}")

        # Save all plots as individual files
        try:
            from core.io_utils import save_all_plots_individually

            # Use experiment-specific output directory
            # When running via experiment framework, self.base_output_dir is already set
            # to the experiment-specific directory (e.g., results/.../data_validation_tree10.../)
            if hasattr(self, 'base_output_dir') and self.base_output_dir:
                base_dir = Path(self.base_output_dir)
                logger.info(f"📁 Using experiment output dir: {base_dir}")
            else:
                # Fallback to global directory
                from core.config_utils import ConfigHelper
                config_dict = ConfigHelper.to_dict(self.config)
                base_dir = get_output_dir(config_dict) / "data_validation"
                logger.warning(f"⚠️  Falling back to global data_validation dir: {base_dir}")

            output_dir = base_dir / "individual_plots"
            save_all_plots_individually(plots, output_dir, dpi=300)
            logger.info(f"✅ Saved {len(plots)} individual plots to {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to save individual plots: {e}")

        # Save SNR analysis results
        if results.get("snr_analysis"):
            try:
                logger.info("📊 Saving SNR analysis results...")
                self._save_snr_results(results["snr_analysis"], base_dir)
            except Exception as e:
                logger.warning(f"Failed to save SNR results: {e}")

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
        logger.info("🔍 Assessing data quality metrics...")
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
                logger.info(f"   Analyzing quality metrics for {view_name} ({X.shape[0]}×{X.shape[1]})...")

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

                # Correlation analysis (skip for large imaging views to avoid memory/time issues)
                valid_mask = ~np.isnan(X).any(axis=0)
                n_valid = np.sum(valid_mask)

                # Only compute correlation for views with < 1000 features (clinical data)
                # For imaging data (>1000 features), skip correlation analysis
                if n_valid > 1 and X.shape[1] < 1000:
                    logger.info(f"      Computing correlation matrix for {view_name}...")
                    X_valid = X[:, valid_mask]
                    corr_matrix = np.corrcoef(X_valid.T)
                    high_corr_pairs = np.sum(np.abs(corr_matrix) > 0.95) - len(
                        corr_matrix
                    )  # Exclude diagonal

                    # Compute condition number
                    if n_valid > 0:
                        condition_number = float(np.linalg.cond(X_valid.T @ X_valid))
                    else:
                        condition_number = np.inf
                else:
                    # Skip correlation computation for large imaging views
                    high_corr_pairs = -1  # -1 indicates skipped
                    condition_number = np.inf  # Cannot compute without correlation

                metrics[view_name] = {
                    "missing_data_ratio": float(missing_ratio),
                    "low_variance_features": int(low_variance_features),
                    "low_variance_ratio": float(low_variance_features / X.shape[1]),
                    "outlier_ratio": float(outlier_ratio),
                    "mean_feature_range": float(np.nanmean(feature_ranges)),
                    "std_feature_range": float(np.nanstd(feature_ranges)),
                    "highly_correlated_pairs": int(high_corr_pairs),
                    "condition_number": condition_number,
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

    def _estimate_snr(
        self, X_list: List[np.ndarray], view_names: List[str]
    ) -> Dict[str, Any]:
        """
        Estimate Signal-to-Noise Ratio for imaging data.

        Critical for understanding:
        1. Whether data has sufficient signal for K factors
        2. Why certain priors/regularization strengths are needed
        3. Expected convergence behavior in MCMC

        Methods:
        - PCA-based: Signal subspace vs noise subspace eigenvalues
        - Variance-based: Between-subject var / within-subject var
        - Effective dimensionality: Number of PCs to explain 95% variance

        Returns
        -------
        dict
            SNR metrics per view with interpretation
        """
        from sklearn.decomposition import PCA

        logger.info("📊 Estimating Signal-to-Noise Ratio (SNR)...")
        snr_metrics = {}

        for view_idx, (X, view_name) in enumerate(zip(X_list, view_names)):
            logger.info(f"   Analyzing SNR for {view_name} ({X.shape})...")

            # Handle missing data
            if np.isnan(X).any():
                # Simple imputation for SNR estimation only
                X_clean = X.copy()
                col_means = np.nanmean(X, axis=0)
                for col in range(X.shape[1]):
                    X_clean[np.isnan(X[:, col]), col] = col_means[col]
            else:
                X_clean = X

            N, D = X_clean.shape
            metrics = {}

            # === METHOD 1: PCA Eigenvalue Spectrum ===
            # Signal = top K eigenvalues, Noise = remaining eigenvalues
            try:
                if D > N:
                    # High-dimensional: Use randomized PCA
                    n_components = min(50, N - 1)
                    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
                else:
                    pca = PCA()

                pca.fit(X_clean)
                eigenvalues = pca.explained_variance_
                explained_var_ratio = pca.explained_variance_ratio_
                cumsum_var = np.cumsum(explained_var_ratio)

                # Effective dimensionality (95% variance threshold)
                n_effective_99 = int(np.argmax(cumsum_var >= 0.99) + 1)
                n_effective_95 = int(np.argmax(cumsum_var >= 0.95) + 1)
                n_effective_90 = int(np.argmax(cumsum_var >= 0.90) + 1)
                n_effective_80 = int(np.argmax(cumsum_var >= 0.80) + 1)

                # SNR estimate: Signal components vs noise components
                # Conservative: Assume first sqrt(N) components are signal
                n_signal_conservative = max(5, int(np.sqrt(N)))
                n_signal_conservative = min(n_signal_conservative, len(eigenvalues) - 1)

                signal_eigenvalues = eigenvalues[:n_signal_conservative]
                noise_eigenvalues = eigenvalues[n_signal_conservative:]

                if len(noise_eigenvalues) > 0:
                    snr_pca = np.sum(signal_eigenvalues) / np.sum(noise_eigenvalues)
                    mean_signal_eigenvalue = np.mean(signal_eigenvalues)
                    mean_noise_eigenvalue = np.mean(noise_eigenvalues)
                    snr_pca_mean = mean_signal_eigenvalue / mean_noise_eigenvalue
                else:
                    snr_pca = np.inf
                    snr_pca_mean = np.inf

                # Scree plot interpretation: Where does eigenvalue spectrum elbow?
                # Use second derivative to find elbow
                if len(eigenvalues) > 3:
                    diff2 = np.diff(np.diff(eigenvalues))
                    elbow_idx = int(np.argmax(diff2) + 2)  # +2 because of double diff
                    elbow_idx = min(elbow_idx, len(eigenvalues) - 1)
                else:
                    elbow_idx = 1

                metrics["pca_snr"] = {
                    "snr_estimate": float(snr_pca),
                    "snr_mean_ratio": float(snr_pca_mean),
                    "n_effective_99pct": int(n_effective_99),
                    "n_effective_95pct": int(n_effective_95),
                    "n_effective_90pct": int(n_effective_90),
                    "n_effective_80pct": int(n_effective_80),
                    "scree_elbow": int(elbow_idx),
                    "variance_explained_by_top5": float(cumsum_var[min(4, len(cumsum_var) - 1)]),
                    "variance_explained_by_top10": float(cumsum_var[min(9, len(cumsum_var) - 1)]),
                    "first_eigenvalue": float(eigenvalues[0]),
                    "ratio_first_to_second": float(eigenvalues[0] / eigenvalues[1]) if len(eigenvalues) > 1 else np.inf,
                }

                logger.info(f"      PCA SNR: {snr_pca:.2f} (signal={n_signal_conservative} PCs)")
                logger.info(f"      Effective dim (95%): {n_effective_95} PCs")
                logger.info(f"      Scree elbow at PC: {elbow_idx}")

            except Exception as e:
                logger.warning(f"      PCA SNR estimation failed: {e}")
                metrics["pca_snr"] = None

            # === METHOD 2: Between-Subject vs Within-Subject Variance ===
            # For imaging: Between-subject signal vs within-subject noise
            try:
                # Between-subject variance (signal)
                feature_means = np.mean(X_clean, axis=0)
                between_var = np.var(feature_means)

                # Within-subject variance (noise estimate)
                # Use residuals from subject means
                residuals = X_clean - np.mean(X_clean, axis=0, keepdims=True)
                within_var = np.mean(np.var(residuals, axis=0))

                if within_var > 0:
                    snr_variance = between_var / within_var
                else:
                    snr_variance = np.inf

                metrics["variance_snr"] = {
                    "snr_estimate": float(snr_variance),
                    "between_subject_var": float(between_var),
                    "within_subject_var": float(within_var),
                }

                logger.info(f"      Variance SNR: {snr_variance:.2f}")

            except Exception as e:
                logger.warning(f"      Variance SNR estimation failed: {e}")
                metrics["variance_snr"] = None

            # === METHOD 3: Feature-wise SNR Distribution ===
            try:
                # For each feature: SNR = mean / std
                feature_means = np.mean(X_clean, axis=0)
                feature_stds = np.std(X_clean, axis=0, ddof=1)

                # Avoid division by zero
                nonzero_std = feature_stds > 1e-10
                snr_per_feature = np.zeros_like(feature_means)
                snr_per_feature[nonzero_std] = np.abs(feature_means[nonzero_std]) / feature_stds[nonzero_std]

                metrics["feature_snr"] = {
                    "mean_snr": float(np.mean(snr_per_feature)),
                    "median_snr": float(np.median(snr_per_feature)),
                    "std_snr": float(np.std(snr_per_feature)),
                    "min_snr": float(np.min(snr_per_feature)),
                    "max_snr": float(np.max(snr_per_feature)),
                    "pct_high_snr": float(np.mean(snr_per_feature > 1.0)),  # SNR > 1
                    "pct_low_snr": float(np.mean(snr_per_feature < 0.5)),   # SNR < 0.5
                }

                logger.info(f"      Feature SNR: median={metrics['feature_snr']['median_snr']:.2f}")

            except Exception as e:
                logger.warning(f"      Feature SNR estimation failed: {e}")
                metrics["feature_snr"] = None

            # === INTERPRETATION ===
            interpretation = self._interpret_snr(metrics, N, D)
            metrics["interpretation"] = interpretation

            # Log interpretation summary
            logger.info(f"   📋 SNR Interpretation for {view_name}:")
            logger.info(f"      Signal quality: {interpretation.get('signal_quality', 'Unknown')}")
            logger.info(f"      Recommended K: {interpretation.get('recommended_K_range', 'Unknown')}")
            logger.info(f"      Prior strength: {interpretation.get('prior_strength_recommendation', 'Unknown')}")
            logger.info(f"      Convergence: {interpretation.get('convergence_expectation', 'Unknown')}")

            snr_metrics[view_name] = metrics

        logger.info("✅ SNR estimation complete for all views")
        return snr_metrics

    def _interpret_snr(self, metrics: Dict, N: int, D: int) -> Dict[str, str]:
        """Interpret SNR metrics and provide recommendations."""
        interpretation = {
            "signal_quality": "Unknown",
            "recommended_K_range": "Unknown",
            "prior_strength_recommendation": "Unknown",
            "convergence_expectation": "Unknown",
        }

        if metrics.get("pca_snr") is None:
            return interpretation

        pca_metrics = metrics["pca_snr"]
        snr = pca_metrics["snr_estimate"]
        n_eff = pca_metrics["n_effective_95pct"]
        elbow = pca_metrics["scree_elbow"]

        # Signal quality assessment
        if snr > 10:
            interpretation["signal_quality"] = "High (SNR > 10): Strong signal, low noise"
        elif snr > 3:
            interpretation["signal_quality"] = "Moderate (SNR 3-10): Adequate signal, moderate noise"
        elif snr > 1:
            interpretation["signal_quality"] = "Low (SNR 1-3): Weak signal, high noise"
        else:
            interpretation["signal_quality"] = "Very Low (SNR < 1): Dominated by noise"

        # K recommendation based on effective dimensionality
        K_conservative = min(elbow, n_eff)
        K_aggressive = min(n_eff, int(N / 10))  # Rule of thumb: N/K >= 10

        if snr > 5:
            interpretation["recommended_K_range"] = f"{K_conservative}-{K_aggressive} (signal supports more factors)"
        elif snr > 2:
            interpretation["recommended_K_range"] = f"{max(3, K_conservative-2)}-{K_conservative} (moderate signal)"
        else:
            interpretation["recommended_K_range"] = f"3-{max(3, K_conservative)} (weak signal, be conservative)"

        # Prior strength recommendation
        if snr < 2:
            interpretation["prior_strength_recommendation"] = "STRONG priors needed (SNR < 2): Data has weak signal, increase τ₀ floor"
        elif snr < 5:
            interpretation["prior_strength_recommendation"] = "MODERATE priors (SNR 2-5): Use τ₀ floor (current: 0.3)"
        else:
            interpretation["prior_strength_recommendation"] = "WEAK priors ok (SNR > 5): Data-dependent τ₀ may suffice"

        # Convergence expectation
        if snr > 5 and n_eff < int(N / 5):
            interpretation["convergence_expectation"] = "GOOD: High SNR + well-defined subspace"
        elif snr > 2:
            interpretation["convergence_expectation"] = "MODERATE: May need longer chains or stronger priors"
        else:
            interpretation["convergence_expectation"] = "CHALLENGING: Low SNR → expect slow convergence, need strong priors"

        return interpretation

    def _save_snr_results(self, snr_analysis: Dict[str, Any], output_dir: Path) -> None:
        """
        Save SNR analysis results to CSV files and text report.

        Args:
            snr_analysis: SNR analysis results from _estimate_snr()
            output_dir: Directory to save results
        """
        if not snr_analysis:
            logger.debug("No SNR analysis to save")
            return

        logger.info("📊 Saving SNR analysis results...")
        snr_dir = output_dir / "snr_analysis"
        snr_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"   SNR output directory: {snr_dir}")

        # Save per-view metrics to CSV
        view_metrics = []
        for view_name, metrics in snr_analysis.items():
            if isinstance(metrics, dict) and "snr_pca" in metrics:
                view_metrics.append({
                    "view_name": view_name,
                    "snr_pca": metrics["snr_pca"],
                    "snr_variance": metrics["snr_variance"],
                    "n_effective_95pct": metrics["n_effective_95"],
                    "n_effective_99pct": metrics["n_effective_99"],
                    "signal_quality": metrics["interpretation"]["signal_quality"],
                    "recommended_K_min": metrics["interpretation"]["recommended_K_range"].split("-")[0],
                    "recommended_K_max": metrics["interpretation"]["recommended_K_range"].split("-")[1],
                    "prior_strength_recommendation": metrics["interpretation"]["prior_strength_recommendation"],
                    "convergence_expectation": metrics["interpretation"]["convergence_expectation"],
                })

        if view_metrics:
            from core.io_utils import save_csv
            df_metrics = pd.DataFrame(view_metrics)
            save_csv(df_metrics, snr_dir / "snr_summary.csv", index=False)
            logger.info(f"   ✅ Saved SNR summary CSV: {snr_dir / 'snr_summary.csv'}")
            logger.debug(f"      {len(view_metrics)} views included in summary")

        # Generate text report
        report_lines = [
            "=" * 80,
            "SIGNAL-TO-NOISE RATIO (SNR) ANALYSIS REPORT",
            "=" * 80,
            "",
        ]

        for view_name, metrics in snr_analysis.items():
            if isinstance(metrics, dict) and "snr_pca" in metrics:
                report_lines.extend([
                    f"View: {view_name}",
                    "-" * 80,
                    f"  SNR (PCA-based):           {metrics['snr_pca']:.2f}",
                    f"  SNR (Variance-based):      {metrics['snr_variance']:.2f}",
                    f"  Effective dimensions (95%): {metrics['n_effective_95']}",
                    f"  Effective dimensions (99%): {metrics['n_effective_99']}",
                    "",
                    "Interpretation:",
                    f"  Signal Quality:             {metrics['interpretation']['signal_quality']}",
                    f"  Recommended K range:        {metrics['interpretation']['recommended_K_range']}",
                    f"  Prior Strength:             {metrics['interpretation']['prior_strength_recommendation']}",
                    f"  Convergence Expectation:    {metrics['interpretation']['convergence_expectation']}",
                    "",
                ])

        report_lines.extend([
            "=" * 80,
            "KEY INSIGHTS:",
            "=" * 80,
            "- PCA SNR: Ratio of signal subspace variance to noise variance",
            "- Variance SNR: Between-subject variance / within-subject variance",
            "- Effective dimensions: Number of PCs explaining 95%/99% of variance",
            "- High SNR (>10): Strong signal, expect fast convergence",
            "- Moderate SNR (3-10): Moderate signal, may need longer chains",
            "- Low SNR (<3): Weak signal, requires strong priors and careful tuning",
            "=" * 80,
        ])

        report_path = snr_dir / "snr_report.txt"
        report_path.write_text("\n".join(report_lines))
        logger.info(f"   ✅ Saved SNR text report: {report_path}")
        logger.info(f"✅ SNR results saved successfully")

    def _plot_snr_analysis(
        self, X_list: List[np.ndarray], view_names: List[str], snr_analysis: Dict[str, Any]
    ) -> Dict[str, plt.Figure]:
        """
        Create SNR visualization plots: scree plot, cumulative variance, and interpretation.

        Args:
            X_list: List of data matrices (one per view)
            view_names: Names of each view
            snr_analysis: SNR analysis results from _estimate_snr()

        Returns:
            Dictionary of figure names to matplotlib Figure objects
        """
        logger.info("📊 Creating SNR analysis visualizations...")
        plots = {}

        for view_idx, (X, view_name) in enumerate(zip(X_list, view_names)):
            if view_name not in snr_analysis or "pca_snr" not in snr_analysis[view_name]:
                logger.debug(f"   Skipping {view_name} (no SNR data)")
                continue

            logger.info(f"   Generating 3-panel SNR plot for {view_name}...")

            metrics = snr_analysis[view_name]
            pca_metrics = metrics.get("pca_snr")

            # Skip if PCA SNR estimation failed
            if pca_metrics is None or not isinstance(pca_metrics, dict):
                logger.warning(f"   Skipping SNR plot for {view_name} (PCA metrics unavailable)")
                continue

            # Re-compute PCA to get eigenvalues (they're in the metrics but not as array)
            # We need the full eigenvalue spectrum for plotting
            N, D = X.shape
            X_clean = X[~np.isnan(X).any(axis=1)]

            try:
                if D > N:
                    pca = PCA(n_components=min(N, 50), svd_solver="randomized")
                else:
                    pca = PCA()
                pca.fit(X_clean)
                eigenvalues = pca.explained_variance_
                explained_var_ratio = pca.explained_variance_ratio_
                cumsum_var = np.cumsum(explained_var_ratio)
            except Exception as e:
                logger.warning(f"Failed to recompute PCA for {view_name}: {e}")
                continue

            # Get key metrics
            n_effective_95 = pca_metrics.get("n_effective_95pct", 5)
            n_effective_99 = pca_metrics.get("n_effective_99pct", 10)
            scree_elbow = pca_metrics.get("scree_elbow", n_effective_95)
            snr_pca = pca_metrics.get("snr_estimate", 1.0)
            signal_quality = metrics.get("interpretation", {}).get("signal_quality", "Unknown")
            recommended_K = metrics.get("interpretation", {}).get("recommended_K_range", "Unknown")

            # Signal/noise cutoff (conservative: sqrt(N))
            n_signal = max(5, int(np.sqrt(N)))
            n_signal = min(n_signal, len(eigenvalues) - 1)

            # Create figure with 3 subplots
            fig = plt.figure(figsize=(15, 5))
            gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

            # === SUBPLOT 1: Scree Plot (Eigenvalue Spectrum) ===
            ax1 = fig.add_subplot(gs[0, 0])
            n_components = len(eigenvalues)
            component_idx = np.arange(1, n_components + 1)

            ax1.plot(component_idx, eigenvalues, 'o-', linewidth=2, markersize=4, color='steelblue', label='Eigenvalues')

            # Mark signal vs noise cutoff
            ax1.axvline(n_signal, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Signal/Noise cutoff (n={n_signal})')

            # Mark scree elbow
            if scree_elbow < len(eigenvalues):
                ax1.axvline(scree_elbow, color='orange', linestyle=':', linewidth=2, alpha=0.7, label=f'Scree elbow (n={scree_elbow})')

            # Mark effective dimensionality
            ax1.axvline(n_effective_95, color='green', linestyle='-.', linewidth=1.5, alpha=0.7, label=f'95% variance (n={n_effective_95})')

            ax1.set_xlabel('Component Index', fontsize=12)
            ax1.set_ylabel('Eigenvalue', fontsize=12)
            ax1.set_title(f'Scree Plot - {_format_view_name(view_name)}', fontsize=13, fontweight='bold')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=9, loc='best')

            # Add SNR annotation
            ax1.text(0.98, 0.98, f'SNR (PCA): {snr_pca:.2f}', transform=ax1.transAxes,
                    fontsize=11, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # === SUBPLOT 2: Cumulative Variance Explained ===
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(component_idx, cumsum_var * 100, 'o-', linewidth=2, markersize=4, color='darkgreen')

            # Mark key thresholds
            ax2.axhline(95, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='95% variance')
            ax2.axhline(99, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='99% variance')

            # Mark effective dimensions
            ax2.axvline(n_effective_95, color='green', linestyle='-.', linewidth=1.5, alpha=0.7)
            ax2.axvline(n_effective_99, color='blue', linestyle='-.', linewidth=1.5, alpha=0.7)

            # Annotate points
            if n_effective_95 < len(cumsum_var):
                ax2.plot(n_effective_95, cumsum_var[n_effective_95 - 1] * 100, 'o', markersize=10, color='green', alpha=0.6)
            if n_effective_99 < len(cumsum_var):
                ax2.plot(n_effective_99, cumsum_var[n_effective_99 - 1] * 100, 'o', markersize=10, color='blue', alpha=0.6)

            ax2.set_xlabel('Number of Components', fontsize=12)
            ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
            ax2.set_title(f'Cumulative Variance - {_format_view_name(view_name)}', fontsize=13, fontweight='bold')
            ax2.set_ylim([0, 105])
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=9, loc='lower right')

            # === SUBPLOT 3: SNR Interpretation Summary ===
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.axis('off')

            # Create text summary
            snr_variance = metrics.get("snr_variance", {}).get("snr_estimate", 0.0) if isinstance(metrics.get("snr_variance"), dict) else 0.0
            prior_strength = metrics.get("interpretation", {}).get("prior_strength_recommendation", "Unknown")
            convergence = metrics.get("interpretation", {}).get("convergence_expectation", "Unknown")

            summary_text = [
                f"SNR Analysis Summary",
                f"{'=' * 35}",
                f"",
                f"Signal Quality:",
                f"  {signal_quality}",
                f"",
                f"SNR Estimates:",
                f"  PCA-based: {snr_pca:.2f}",
                f"  Variance-based: {snr_variance:.2f}",
                f"",
                f"Effective Dimensionality:",
                f"  95% variance: {n_effective_95} PCs",
                f"  99% variance: {n_effective_99} PCs",
                f"  Scree elbow: {scree_elbow}",
                f"",
                f"Recommendations:",
                f"  K range: {recommended_K}",
                f"  Prior strength:",
                f"    {prior_strength}",
                f"",
                f"Convergence:",
                f"  {convergence}",
            ]

            # Add text to plot
            ax3.text(0.05, 0.95, '\n'.join(summary_text), transform=ax3.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='left',
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

            plt.suptitle(f'Signal-to-Noise Ratio Analysis - {_format_view_name(view_name)}',
                        fontsize=14, fontweight='bold', y=1.02)

            plots[f"snr_analysis_{view_name}"] = fig

            logger.info(f"      ✅ Created SNR plot: snr_analysis_{view_name}.png")

        logger.info(f"✅ Generated {len(plots)} SNR visualization(s)")
        return plots

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
            view_name = raw_data.get("view_names", [f"view_{view_idx}"])[view_idx]
            formatted_view_name = _format_view_name(view_name)

            # Raw data distribution
            feature_means_raw = np.nanmean(raw_X, axis=0)
            axes[0, view_idx].hist(feature_means_raw, bins=50, alpha=0.7, label="Raw")
            axes[0, view_idx].set_title(f"{formatted_view_name} - Raw Feature Means")
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
            axes[1, view_idx].set_title(f"{formatted_view_name} - Processed Feature Means")
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
            view_name = raw_data.get("view_names", [f"view_{view_idx}"])[view_idx]
            formatted_view_name = _format_view_name(view_name)

            # Sample data for visualization (to avoid memory issues)
            sample_size = min(500, raw_X.shape[0])
            sample_features = min(100, raw_X.shape[1])

            raw_sample = raw_X[:sample_size, :sample_features]
            proc_sample = proc_X[:sample_size, : min(sample_features, proc_X.shape[1])]

            # Raw data missing pattern
            missing_raw = np.isnan(raw_sample)
            axes[0, view_idx].imshow(missing_raw.T, cmap="RdYlBu", aspect="auto")
            axes[0, view_idx].set_title(f"{formatted_view_name} - Raw Missing Data")
            axes[0, view_idx].set_xlabel("Subjects")
            axes[0, view_idx].set_ylabel("Features")

            # Preprocessed data missing pattern
            missing_proc = np.isnan(proc_sample)
            axes[1, view_idx].imshow(missing_proc.T, cmap="RdYlBu", aspect="auto")
            axes[1, view_idx].set_title(f"{formatted_view_name} - Processed Missing Data")
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
            view_name = raw_data.get("view_names", [f"view_{view_idx}"])[view_idx]
            formatted_view_name = _format_view_name(view_name)

            # Sample features for correlation analysis
            max_features = 50  # Limit for computational efficiency

            # Raw data correlation
            raw_sample = raw_X[:, : min(max_features, raw_X.shape[1])]
            valid_mask_raw = ~np.isnan(raw_sample).any(axis=0)
            if np.sum(valid_mask_raw) > 1:
                raw_corr = np.corrcoef(raw_sample[:, valid_mask_raw].T)
                im1 = axes[0, view_idx].imshow(raw_corr, cmap="RdBu", vmin=-1, vmax=1)
                axes[0, view_idx].set_title(f"{formatted_view_name} - Raw Correlations")
                plt.colorbar(im1, ax=axes[0, view_idx])

            # Processed data correlation
            proc_sample = proc_X[:, : min(max_features, proc_X.shape[1])]
            valid_mask_proc = ~np.isnan(proc_sample).any(axis=0)
            if np.sum(valid_mask_proc) > 1:
                proc_corr = np.corrcoef(proc_sample[:, valid_mask_proc].T)
                im2 = axes[1, view_idx].imshow(proc_corr, cmap="RdBu", vmin=-1, vmax=1)
                axes[1, view_idx].set_title(f"{formatted_view_name} - Processed Correlations")
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
            view_name = raw_data.get("view_names", [f"view_{view_idx}"])[view_idx]
            formatted_view_name = _format_view_name(view_name)

            # PCA on raw data
            raw_X_clean = raw_X[
                ~np.isnan(raw_X).any(axis=1)
            ]  # Remove subjects with any NaN
            if raw_X_clean.shape[0] > 10 and raw_X_clean.shape[1] > 2:
                pca_raw = PCA(n_components=min(10, raw_X_clean.shape[1]))
                pca_raw.fit(raw_X_clean)

                pca_results["raw_data"][formatted_view_name] = {
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
                axes[0, view_idx].set_title(f"{formatted_view_name} - Raw Data PCA")
                axes[0, view_idx].set_xlabel("Principal Component")
                axes[0, view_idx].set_ylabel("Cumulative Explained Variance")
                axes[0, view_idx].grid(True)

            # PCA on processed data
            proc_X_clean = proc_X[~np.isnan(proc_X).any(axis=1)]
            if proc_X_clean.shape[0] > 10 and proc_X_clean.shape[1] > 2:
                pca_proc = PCA(n_components=min(10, proc_X_clean.shape[1]))
                pca_proc.fit(proc_X_clean)

                pca_results["preprocessed_data"][formatted_view_name] = {
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
                axes[1, view_idx].set_title(f"{formatted_view_name} - Processed Data PCA")
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

                    # Preserve command-line overrides from original config
                    original_preprocessing = config_dict.get("preprocessing", {})
                    merged_preprocessing = strategy_params.copy()

                    # Preserve all command-line overrides (ROI selection, confounds, MAD, PCA, etc.)
                    preserve_keys = [
                        "select_rois",
                        "regress_confounds",
                        "drop_confounds_from_clinical",
                        "enable_spatial_processing",  # Required for MAD and spatial features
                        "qc_outlier_threshold",       # MAD threshold
                        "enable_pca",                 # PCA settings
                        "pca_n_components",
                        "pca_variance_threshold",
                        "pca_whiten",
                    ]
                    for key in preserve_keys:
                        if key in original_preprocessing:
                            merged_preprocessing[key] = original_preprocessing[key]

                    temp_config_dict["preprocessing"] = merged_preprocessing
                    temp_config = ConfigAccessor(temp_config_dict)

                    # Apply preprocessing with this strategy
                    # Use experiment-specific output_dir if available
                    strategy_output_dir = getattr(self, 'base_output_dir', None) or get_output_dir(config)
                    X_list_strategy, preprocessing_info = apply_preprocessing_to_pipeline(
                        config=temp_config,
                        data_dir=data_dir,
                        auto_select_strategy=False,
                        preferred_strategy=strategy_params.get("strategy", strategy_name),
                        output_dir=strategy_output_dir
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

            # Get view names from first valid strategy (all should have same views)
            first_strategy_data = valid_strategies[strategy_names[0]]
            view_names = first_strategy_data.get("view_names", [f"view_{i}" for i in range(n_views)])
            formatted_view_names = [_format_view_name(vn) for vn in view_names]

            axes[0, 1].set_title("Missing Data by View and Strategy")
            axes[0, 1].set_xlabel("View")
            axes[0, 1].set_ylabel("Missing Data Ratio")
            axes[0, 1].set_xticks(x + width * (len(strategy_names) - 1) / 2)
            axes[0, 1].set_xticklabels(formatted_view_names, rotation=45, ha='right')
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

                    # Create figure explicitly to ensure proper reference
                    fig_dist = plt.figure(figsize=(5*n_views, 10))
                    axes = fig_dist.subplots(2, n_views)
                    if n_views == 1:
                        axes = axes.reshape(-1, 1)
                    fig_dist.suptitle("Data Distribution: Raw vs Preprocessed (Clinical Data Only)", fontsize=16)

                    for plot_idx, (view_idx, view_name, raw_X, proc_X) in enumerate(clinical_views):
                        formatted_view_name = _format_view_name(view_name)
                        # Raw data distribution
                        axes[0, plot_idx].hist(
                            raw_X.flatten(), bins=50, alpha=0.7, color="red", edgecolor="black"
                        )
                        axes[0, plot_idx].set_title(f"Raw {formatted_view_name}")
                        axes[0, plot_idx].set_xlabel("Feature Value")
                        axes[0, plot_idx].set_ylabel("Frequency")

                        # Preprocessed data distribution
                        axes[1, plot_idx].hist(
                            proc_X.flatten(), bins=50, alpha=0.7, color="blue", edgecolor="black"
                        )
                        axes[1, plot_idx].set_title(f"Preprocessed {formatted_view_name}")
                        axes[1, plot_idx].set_xlabel("Standardized Value")
                        axes[1, plot_idx].set_ylabel("Frequency")

                    fig_dist.tight_layout()
                    plots["data_distribution_comparison"] = fig_dist
                    logger.info(f"   ✅ Distribution histograms created (stored as data_distribution_comparison)")

            # Quality metrics summary plot
            if "quality_metrics" in results:
                logger.debug("   Creating quality metrics summary plot...")

                # Create figure explicitly to ensure proper reference
                fig_quality = plt.figure(figsize=(10, 6))
                ax = fig_quality.add_subplot(111)

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
                    fig_quality.tight_layout()
                    plots["quality_metrics_summary"] = fig_quality
                    logger.info(f"   ✅ Quality metrics plot created ({len(metrics_names)} metrics, stored as quality_metrics_summary)")

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

            # Setup plot directory before using it
            viz_manager.plot_dir = viz_manager._setup_plot_directory()

            # Prepare data validation structure for visualizations
            # Build feature_reduction directly from actual loaded views (not from preprocessing_effects)
            # This ensures we only show data for views that were actually loaded (respects --select-rois)
            feature_reduction = {}

            # Get view names from preprocessed_data section of data_summary
            view_names_actual = results.get("data_summary", {}).get("preprocessed_data", {}).get("view_names", [])
            if not view_names_actual:
                # Fallback to generic names if not found
                view_names_actual = [f"view_{i}" for i in range(len(X_list))]

            for view_idx, X in enumerate(X_list):
                view_name = view_names_actual[view_idx] if view_idx < len(view_names_actual) else f"view_{view_idx}"
                # For data validation, original = processed since we're showing the loaded data
                feature_reduction[view_name] = {
                    "original": X.shape[1],
                    "processed": X.shape[1],
                    "reduction_ratio": 1.0  # No reduction in data validation (just showing loaded data)
                }

            # Collect variance analysis for visualization
            variance_analysis = {}
            variance_threshold = getattr(self.config.preprocessing_config, 'variance_threshold', 0.0)

            for view_idx, X in enumerate(X_list):
                view_name = view_names_actual[view_idx] if view_idx < len(view_names_actual) else f"view_{view_idx}"
                feature_variances = np.nanvar(X, axis=0)
                n_retained = np.sum(feature_variances >= variance_threshold) if variance_threshold > 0 else len(feature_variances)

                variance_analysis[view_name] = {
                    "variances": feature_variances,
                    "threshold": variance_threshold,
                    "n_retained": n_retained,
                    "n_total": len(feature_variances)
                }

            data = {
                "X_list": X_list,
                "view_names": view_names_actual,  # Use actual loaded view names, not generic ones
                "n_subjects": X_list[0].shape[0],
                "view_dimensions": [X.shape[1] for X in X_list],
                "preprocessing": {
                    "status": "completed",
                    "strategy": "comprehensive_validation",
                    "quality_results": results,
                    "feature_reduction": feature_reduction,  # Add transformed data
                    "variance_analysis": variance_analysis,  # Add variance data
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
            logger.debug("   Creating preprocessing quality visualizations...")
            viz_manager.preprocessing_viz.create_plots(
                data["preprocessing"], viz_manager.plot_dir
            )
            logger.debug("   ✅ Preprocessing visualizations created")

            # Add covariance structure analysis
            logger.debug("   Creating covariance structure analysis...")
            try:
                from visualization.covariance_plots import create_covariance_report

                cov_report = create_covariance_report(
                    X_list=X_list,
                    view_names=view_names_actual,
                    output_dir=viz_manager.plot_dir,
                    subsample=500  # Subsample for visualization of large matrices
                )
                # Don't add cov_report to advanced_plots - it's a dict, not a Figure
                # The function already saves its plots to disk
                logger.debug("   ✅ Covariance analysis completed")
            except Exception as e:
                logger.warning(f"   ⚠️  Failed to create covariance analysis: {e}")

            # Extract the generated plots and convert to matplotlib figures
            logger.debug("   Loading and converting generated plots to figures...")
            if hasattr(viz_manager, "plot_dir") and viz_manager.plot_dir.exists():
                plot_files = list(viz_manager.plot_dir.glob("**/*.png"))
                logger.info(f"   Found {len(plot_files)} plot files to convert")

                for plot_file in plot_files:
                    plot_name = f"data_validation_{plot_file.stem}"

                    try:
                        import matplotlib.image as mpimg
                        import matplotlib.pyplot as plt

                        # Create a new figure explicitly and make it current
                        fig = plt.figure(figsize=(12, 8))
                        ax = fig.add_subplot(111)

                        # Load and display the image
                        img = mpimg.imread(str(plot_file))
                        ax.imshow(img)
                        ax.axis("off")

                        # Don't add extra title - the image already has its own title baked in

                        # Store the figure immediately before creating the next one
                        advanced_plots[plot_name] = fig

                        logger.debug(f"   Loaded plot: {plot_name} from {plot_file.name}")

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

    # =========================================================================
    # MAD THRESHOLD EXPLORATORY DATA ANALYSIS
    # =========================================================================

    def _run_mad_threshold_eda(self, raw_data: Dict, output_dir: str) -> Dict[str, Any]:
        """
        Run comprehensive MAD threshold exploratory data analysis.

        This analysis helps determine optimal MAD threshold values for outlier detection
        in imaging data preprocessing. Analysis runs on RAW data before preprocessing.

        Parameters
        ----------
        raw_data : Dict
            Raw data dictionary containing X_list and view_names
        output_dir : str
            Directory to save analysis outputs

        Returns
        -------
        Dict[str, Any]
            MAD threshold analysis results including recommendations
        """
        logger.info("=" * 80)
        logger.info("🔍 MAD THRESHOLD EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 80)
        logger.info("Analyzing optimal MAD threshold values for outlier detection...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        X_list = raw_data.get("X_list", [])
        view_names = raw_data.get("view_names", [f"View_{i}" for i in range(len(X_list))])

        # Filter to imaging views only (exclude clinical data)
        imaging_views = []
        imaging_names = []
        for X, name in zip(X_list, view_names):
            if "volume_" in name.lower():  # Imaging views have 'volume_' prefix
                imaging_views.append(X)
                imaging_names.append(name)

        if not imaging_views:
            logger.warning("No imaging views found for MAD threshold analysis")
            return {"status": "skipped", "reason": "no_imaging_views"}

        logger.info(f"Analyzing {len(imaging_views)} imaging views: {imaging_names}")

        # Define candidate thresholds to evaluate
        candidate_thresholds = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

        # Run comprehensive analysis
        mad_distributions = self._analyze_mad_distributions(
            imaging_views, imaging_names, candidate_thresholds, output_path
        )

        elbow_analysis = self._perform_elbow_analysis(
            imaging_views, imaging_names, mad_distributions, output_path
        )

        spatial_analysis = self._analyze_spatial_distribution(
            imaging_views, imaging_names, mad_distributions,
            raw_data.get("data_dir", "./qMAP-PD_data"), output_path
        )

        info_preservation = self._analyze_information_preservation(
            imaging_views, imaging_names, mad_distributions, output_path
        )

        subject_outliers = self._analyze_subject_level_outliers(
            imaging_views, imaging_names, raw_data, output_path
        )

        summary_table = self._generate_threshold_summary(
            imaging_names, mad_distributions, elbow_analysis,
            spatial_analysis, info_preservation, candidate_thresholds, output_path
        )

        logger.info("=" * 80)
        logger.info("✅ MAD threshold EDA completed successfully")
        logger.info("=" * 80)

        return {
            "status": "completed",
            "imaging_views_analyzed": imaging_names,
            "candidate_thresholds": candidate_thresholds,
            "mad_distributions": mad_distributions,
            "elbow_analysis": elbow_analysis,
            "spatial_analysis": spatial_analysis,
            "information_preservation": info_preservation,
            "subject_outliers": subject_outliers,
            "summary_table": summary_table,
        }

    def _analyze_mad_distributions(
        self,
        imaging_views: List[np.ndarray],
        view_names: List[str],
        candidate_thresholds: List[float],
        output_path: Path
    ) -> Dict[str, Any]:
        """
        Analyze MAD value distributions across voxels in each imaging view.

        Creates histogram and cumulative distribution plots showing how many voxels
        would be retained at different threshold values.
        """
        logger.info("📊 Analyzing MAD distributions...")

        mad_results = {}

        for X, view_name in zip(imaging_views, view_names):
            logger.info(f"   Processing {view_name}: {X.shape}")

            # Calculate MAD for each voxel
            mad_values = self._calculate_voxel_mad(X)

            # Calculate retention rates at candidate thresholds
            retention_rates = {}
            for threshold in candidate_thresholds:
                n_retained = np.sum(mad_values <= threshold)
                retention_rates[threshold] = {
                    "n_retained": int(n_retained),
                    "n_total": len(mad_values),
                    "retention_pct": 100.0 * n_retained / len(mad_values),
                    "n_removed": int(len(mad_values) - n_retained),
                }

            mad_results[view_name] = {
                "mad_values": mad_values,
                "retention_rates": retention_rates,
                "n_voxels": len(mad_values),
                "mad_min": float(np.min(mad_values)),
                "mad_max": float(np.max(mad_values)),
                "mad_median": float(np.median(mad_values)),
                "mad_mean": float(np.mean(mad_values)),
            }

            # Create visualization
            self._plot_mad_distribution(
                mad_values, view_name, candidate_thresholds,
                retention_rates, output_path
            )

        logger.info(f"   ✅ MAD distributions analyzed for {len(imaging_views)} views")
        return mad_results

    def _calculate_voxel_mad(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate MAD (Median Absolute Deviation) for each voxel.

        Uses the same methodology as the preprocessing outlier detection:
        MAD score = |value - median| / (MAD × 1.4826)

        Parameters
        ----------
        X : np.ndarray
            Data matrix (n_subjects × n_voxels)

        Returns
        -------
        np.ndarray
            MAD scores for each voxel
        """
        n_voxels = X.shape[1]
        mad_scores = np.zeros(n_voxels)

        for i in range(n_voxels):
            voxel_data = X[:, i]
            # Remove NaN values
            clean_data = voxel_data[~np.isnan(voxel_data)]

            if len(clean_data) > 0:
                median = np.median(clean_data)
                mad = np.median(np.abs(clean_data - median))

                if mad > 0:
                    # Calculate maximum MAD score across all subjects for this voxel
                    voxel_mad_scores = np.abs(clean_data - median) / (mad * 1.4826)
                    mad_scores[i] = np.max(voxel_mad_scores)  # Max score determines if voxel is outlier
                else:
                    mad_scores[i] = 0.0
            else:
                mad_scores[i] = np.nan

        return mad_scores

    def _plot_mad_distribution(
        self,
        mad_values: np.ndarray,
        view_name: str,
        candidate_thresholds: List[float],
        retention_rates: Dict,
        output_path: Path
    ):
        """Create MAD distribution histogram and cumulative distribution plots."""
        # Clean view name for filename
        clean_name = view_name.replace("volume_", "").replace("_voxels", "")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Histogram with threshold lines
        ax1.hist(mad_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('MAD Score')
        ax1.set_ylabel('Number of Voxels')
        ax1.set_title(f'MAD Distribution - {view_name}', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Add threshold lines
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(candidate_thresholds)))
        for threshold, color in zip(candidate_thresholds, colors):
            retention_pct = retention_rates[threshold]["retention_pct"]
            ax1.axvline(threshold, color=color, linestyle='--', linewidth=2,
                       label=f'{threshold:.1f} ({retention_pct:.1f}% retained)')

        ax1.legend(fontsize=8, loc='upper right', framealpha=0.9, ncol=1)

        # Cumulative distribution
        sorted_mad = np.sort(mad_values)
        cumulative_pct = 100.0 * np.arange(1, len(sorted_mad) + 1) / len(sorted_mad)

        ax2.plot(sorted_mad, cumulative_pct, linewidth=2, color='navy')
        ax2.set_xlabel('MAD Threshold')
        ax2.set_ylabel('Voxels Retained (%)')
        ax2.set_title(f'Cumulative Retention Curve - {view_name}', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Add threshold markers
        for threshold, color in zip(candidate_thresholds, colors):
            retention_pct = retention_rates[threshold]["retention_pct"]
            ax2.axvline(threshold, color=color, linestyle='--', linewidth=2, alpha=0.7)
            ax2.axhline(retention_pct, color=color, linestyle=':', linewidth=1, alpha=0.5)
            ax2.plot(threshold, retention_pct, 'o', color=color, markersize=10,
                    markeredgecolor='black', markeredgewidth=1.5)
            ax2.text(threshold, retention_pct + 2, f'{retention_pct:.1f}%',
                    ha='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plot_file = output_path / f"mad_distribution_{clean_name}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"      Saved MAD distribution plot: {plot_file.name}")

    def _perform_elbow_analysis(
        self,
        imaging_views: List[np.ndarray],
        view_names: List[str],
        mad_distributions: Dict[str, Any],
        output_path: Path
    ) -> Dict[str, Any]:
        """
        Identify elbow points in retention vs threshold curves.

        The elbow point represents a natural breakpoint where increasing the threshold
        starts retaining proportionally fewer additional voxels.
        """
        logger.info("📈 Performing elbow analysis...")

        elbow_results = {}

        # Create threshold range for fine-grained analysis
        threshold_range = np.linspace(1.0, 6.0, 100)

        fig, axes = plt.subplots(len(view_names), 1, figsize=(12, 5 * len(view_names)))
        if len(view_names) == 1:
            axes = [axes]

        for idx, (view_name, ax) in enumerate(zip(view_names, axes)):
            mad_values = mad_distributions[view_name]["mad_values"]

            # Calculate retention rates across threshold range
            retention_pcts = []
            for threshold in threshold_range:
                retention_pct = 100.0 * np.sum(mad_values <= threshold) / len(mad_values)
                retention_pcts.append(retention_pct)

            retention_pcts = np.array(retention_pcts)

            # Calculate rate of change (first derivative)
            rate_of_change = np.gradient(retention_pcts, threshold_range)

            # Find elbow point (where rate of change stabilizes)
            # Use second derivative to find inflection point
            second_derivative = np.gradient(rate_of_change, threshold_range)
            elbow_idx = np.argmax(np.abs(second_derivative))
            elbow_threshold = threshold_range[elbow_idx]
            elbow_retention = retention_pcts[elbow_idx]

            elbow_results[view_name] = {
                "elbow_threshold": float(elbow_threshold),
                "retention_at_elbow": float(elbow_retention),
                "interpretation": self._interpret_elbow(elbow_threshold),
            }

            # Plot retention curve with elbow point
            ax.plot(threshold_range, retention_pcts, linewidth=2, color='navy', label='Retention Rate')
            ax.axvline(elbow_threshold, color='red', linestyle='--', linewidth=2,
                      label=f'Elbow Point: {elbow_threshold:.2f}')
            ax.axhline(elbow_retention, color='red', linestyle=':', linewidth=1, alpha=0.5)
            ax.plot(elbow_threshold, elbow_retention, 'ro', markersize=12,
                   markeredgecolor='black', markeredgewidth=2)

            ax.set_xlabel('MAD Threshold', fontsize=12)
            ax.set_ylabel('Voxels Retained (%)', fontsize=12)
            ax.set_title(f'Elbow Analysis - {view_name}\n'
                        f'Recommended Threshold: {elbow_threshold:.2f} '
                        f'(retains {elbow_retention:.1f}% voxels)',
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)

            logger.info(f"   {view_name}: Elbow at {elbow_threshold:.2f} ({elbow_retention:.1f}% retention)")

        plt.tight_layout()
        plot_file = output_path / "elbow_analysis_all_views.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"   ✅ Elbow analysis completed, saved to {plot_file.name}")
        return elbow_results

    def _interpret_elbow(self, elbow_threshold: float) -> str:
        """Provide interpretation of elbow threshold value."""
        if elbow_threshold < 2.5:
            return "Very strict - may remove too many voxels with biological variability"
        elif elbow_threshold < 3.5:
            return "Moderate - balanced removal of artifacts while preserving signal"
        elif elbow_threshold < 4.5:
            return "Permissive - retains most voxels, may include some artifacts"
        else:
            return "Very permissive - retains nearly all voxels including likely artifacts"

    def _analyze_spatial_distribution(
        self,
        imaging_views: List[np.ndarray],
        view_names: List[str],
        mad_distributions: Dict[str, Any],
        data_dir: str,
        output_path: Path
    ) -> Dict[str, Any]:
        """
        Analyze spatial distribution of voxels that would be removed at different thresholds.

        This helps identify whether removed voxels cluster at ROI boundaries (edge artifacts)
        or are distributed throughout the structure.
        """
        logger.info("🗺️  Analyzing spatial distribution of removed voxels...")

        spatial_results = {}

        for view_name in view_names:
            # Extract ROI name for loading position lookup
            roi_name = view_name.replace("volume_", "").replace("_voxels", "")

            try:
                # Try to load position lookup file
                position_file = Path(data_dir) / "position_lookup" / f"position_{roi_name}_voxels.tsv"
                if not position_file.exists():
                    # Try alternative location
                    position_file = Path(data_dir) / "volume_matrices" / f"{roi_name}_position_lookup.tsv"

                if position_file.exists():
                    positions = pd.read_csv(position_file, sep='\t')
                    logger.info(f"   Loaded position lookup for {view_name}: {len(positions)} voxels")

                    # Analyze at threshold 3.0 (current default)
                    mad_values = mad_distributions[view_name]["mad_values"]
                    removed_mask = mad_values > 3.0

                    # Check if dimensions match
                    if len(removed_mask) != len(positions):
                        logger.warning(f"   Dimension mismatch for {view_name}: {len(removed_mask)} MAD values vs {len(positions)} positions. Skipping spatial analysis.")
                        spatial_results[view_name] = {
                            "position_file_found": True,
                            "dimension_mismatch": True,
                            "n_mad_values": len(removed_mask),
                            "n_positions": len(positions),
                        }
                        continue

                    if np.any(removed_mask):
                        removed_positions = positions[removed_mask][['x', 'y', 'z']].values
                        retained_positions = positions[~removed_mask][['x', 'y', 'z']].values

                        # Calculate spatial clustering metrics
                        spatial_metrics = self._calculate_spatial_clustering(
                            removed_positions, retained_positions
                        )

                        spatial_results[view_name] = {
                            "position_file_found": True,
                            "n_removed": int(np.sum(removed_mask)),
                            "n_retained": int(np.sum(~removed_mask)),
                            "clustering_metrics": spatial_metrics,
                        }

                        # Create spatial visualization
                        self._plot_spatial_distribution(
                            removed_positions, retained_positions,
                            view_name, output_path
                        )
                    else:
                        spatial_results[view_name] = {
                            "position_file_found": True,
                            "n_removed": 0,
                            "message": "No voxels would be removed at threshold 3.0"
                        }
                else:
                    logger.warning(f"   Position lookup not found for {view_name}")
                    spatial_results[view_name] = {
                        "position_file_found": False,
                        "message": f"Position file not found: {position_file}"
                    }

            except Exception as e:
                logger.warning(f"   Failed spatial analysis for {view_name}: {e}")
                spatial_results[view_name] = {
                    "position_file_found": False,
                    "error": str(e)
                }

        logger.info(f"   ✅ Spatial analysis completed for {len(spatial_results)} views")
        return spatial_results

    def _calculate_spatial_clustering(
        self,
        removed_positions: np.ndarray,
        retained_positions: np.ndarray
    ) -> Dict[str, float]:
        """Calculate spatial clustering metrics for removed voxels."""
        from scipy.spatial.distance import cdist

        # Calculate mean distance from each removed voxel to nearest retained voxel
        if len(removed_positions) > 0 and len(retained_positions) > 0:
            distances = cdist(removed_positions, retained_positions)
            min_distances = np.min(distances, axis=1)

            # Calculate centroid of removed vs retained voxels
            removed_centroid = np.mean(removed_positions, axis=0)
            retained_centroid = np.mean(retained_positions, axis=0)
            centroid_distance = np.linalg.norm(removed_centroid - retained_centroid)

            return {
                "mean_nearest_neighbor_distance": float(np.mean(min_distances)),
                "median_nearest_neighbor_distance": float(np.median(min_distances)),
                "centroid_separation": float(centroid_distance),
                "interpretation": "Edge clustering" if centroid_distance > 5.0 else "Distributed throughout ROI"
            }
        else:
            return {"message": "Insufficient data for spatial metrics"}

    def _plot_spatial_distribution(
        self,
        removed_positions: np.ndarray,
        retained_positions: np.ndarray,
        view_name: str,
        output_path: Path
    ):
        """Create 3D scatter plot showing spatial distribution of removed vs retained voxels."""
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot retained voxels (smaller, semi-transparent)
        ax.scatter(retained_positions[:, 0], retained_positions[:, 1], retained_positions[:, 2],
                  c='lightblue', marker='.', s=20, alpha=0.3, label=f'Retained (n={len(retained_positions)})')

        # Plot removed voxels (larger, more opaque)
        ax.scatter(removed_positions[:, 0], removed_positions[:, 1], removed_positions[:, 2],
                  c='red', marker='o', s=50, alpha=0.7, edgecolors='darkred', linewidths=0.5,
                  label=f'Removed at threshold=3.0 (n={len(removed_positions)})')

        ax.set_xlabel('X (mm)', fontsize=11)
        ax.set_ylabel('Y (mm)', fontsize=11)
        ax.set_zlabel('Z (mm)', fontsize=11)
        ax.set_title(f'Spatial Distribution of Removed Voxels - {view_name}\n'
                    f'{100*len(removed_positions)/(len(removed_positions)+len(retained_positions)):.1f}% voxels flagged as outliers',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)

        plt.tight_layout()
        clean_name = view_name.replace("volume_", "").replace("_voxels", "")
        plot_file = output_path / f"spatial_distribution_{clean_name}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"      Saved spatial distribution plot: {plot_file.name}")

    def _analyze_information_preservation(
        self,
        imaging_views: List[np.ndarray],
        view_names: List[str],
        mad_distributions: Dict[str, Any],
        output_path: Path
    ) -> Dict[str, Any]:
        """
        Quantify information preservation at different MAD thresholds.

        Measures:
        1. Variance preserved (what % of total variance remains)
        2. Correlation structure preservation (how well voxel-voxel correlations are maintained)
        """
        logger.info("📊 Analyzing information preservation...")

        info_results = {}
        threshold_range = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

        for X, view_name in zip(imaging_views, view_names):
            logger.info(f"   Processing {view_name}...")
            mad_values = mad_distributions[view_name]["mad_values"]

            # Calculate original variance and correlation structure
            original_variance = np.var(X)

            # Sample 100 random voxels for correlation analysis (full correlation matrix too large)
            n_sample = min(100, X.shape[1])
            sample_indices = np.random.choice(X.shape[1], n_sample, replace=False)
            original_corr = np.corrcoef(X[:, sample_indices].T)

            threshold_results = {}
            for threshold in threshold_range:
                # Identify retained voxels
                retained_mask = mad_values <= threshold
                X_retained = X[:, retained_mask]

                if X_retained.shape[1] > 0:
                    # Variance preservation
                    retained_variance = np.var(X_retained)
                    variance_ratio = retained_variance / original_variance

                    # Correlation preservation (for sampled voxels that are retained)
                    # Check which of our sampled voxels were retained
                    sample_retained_mask = retained_mask[sample_indices]
                    n_sample_retained = np.sum(sample_retained_mask)

                    if n_sample_retained > 1:
                        # Get the actual retained sample indices in the original data
                        retained_sample_indices = sample_indices[sample_retained_mask]

                        # Extract these columns from X_retained by mapping to its new column space
                        # Create mapping from original indices to X_retained indices
                        retained_indices_mapping = np.where(retained_mask)[0]
                        new_positions = np.searchsorted(retained_indices_mapping, retained_sample_indices)

                        retained_corr = np.corrcoef(X_retained[:, new_positions].T)

                        # Compare correlation matrices (only for voxels present in both)
                        corr_similarity = self._correlation_matrix_similarity(
                            original_corr[np.ix_(sample_retained_mask, sample_retained_mask)],
                            retained_corr
                        )
                    else:
                        corr_similarity = np.nan

                    threshold_results[threshold] = {
                        "n_retained": int(np.sum(retained_mask)),
                        "variance_ratio": float(variance_ratio),
                        "correlation_preservation": float(corr_similarity) if not np.isnan(corr_similarity) else None,
                    }
                else:
                    threshold_results[threshold] = {
                        "n_retained": 0,
                        "variance_ratio": 0.0,
                        "correlation_preservation": None,
                    }

            info_results[view_name] = threshold_results

        # Create visualization
        self._plot_information_preservation(info_results, view_names, threshold_range, output_path)

        logger.info(f"   ✅ Information preservation analyzed for {len(imaging_views)} views")
        return info_results

    def _correlation_matrix_similarity(self, corr1: np.ndarray, corr2: np.ndarray) -> float:
        """Calculate similarity between two correlation matrices using Frobenius norm."""
        if corr1.shape != corr2.shape:
            return np.nan

        # Use normalized Frobenius norm
        diff = corr1 - corr2
        similarity = 1.0 - (np.linalg.norm(diff, 'fro') / np.sqrt(corr1.size))
        return max(0.0, similarity)  # Clamp to [0, 1]

    def _plot_information_preservation(
        self,
        info_results: Dict[str, Any],
        view_names: List[str],
        threshold_range: List[float],
        output_path: Path
    ):
        """Create visualization of information preservation across thresholds."""
        fig, axes = plt.subplots(len(view_names), 2, figsize=(14, 5 * len(view_names)))
        if len(view_names) == 1:
            axes = axes.reshape(1, -1)

        for idx, view_name in enumerate(view_names):
            view_data = info_results[view_name]

            # Extract metrics
            variance_ratios = [view_data[t]["variance_ratio"] for t in threshold_range]
            corr_preservation = [view_data[t]["correlation_preservation"] or 0.0 for t in threshold_range]

            # Plot variance preservation
            axes[idx, 0].plot(threshold_range, variance_ratios, marker='o', linewidth=2, markersize=8)
            axes[idx, 0].axhline(0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')
            axes[idx, 0].axhline(0.95, color='orange', linestyle='--', alpha=0.5, label='95% threshold')
            axes[idx, 0].set_xlabel('MAD Threshold')
            axes[idx, 0].set_ylabel('Variance Ratio')
            axes[idx, 0].set_title(f'Variance Preservation - {view_name}', fontsize=10)
            axes[idx, 0].grid(True, alpha=0.3)
            axes[idx, 0].legend(fontsize=8, loc='upper right', framealpha=0.9, ncol=1)

            # Plot correlation preservation
            axes[idx, 1].plot(threshold_range, corr_preservation, marker='s', linewidth=2, markersize=8, color='purple')
            axes[idx, 1].axhline(0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')
            axes[idx, 1].axhline(0.95, color='orange', linestyle='--', alpha=0.5, label='95% threshold')
            axes[idx, 1].set_xlabel('MAD Threshold')
            axes[idx, 1].set_ylabel('Correlation Preservation')
            axes[idx, 1].set_title(f'Correlation Structure Preservation - {view_name}', fontsize=10)
            axes[idx, 1].grid(True, alpha=0.3)
            axes[idx, 1].legend(fontsize=8, loc='upper right', framealpha=0.9, ncol=1)

        plt.tight_layout()
        plot_file = output_path / "information_preservation_all_views.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"      Saved information preservation plot: {plot_file.name}")

    def _analyze_subject_level_outliers(
        self,
        imaging_views: List[np.ndarray],
        view_names: List[str],
        raw_data: Dict,
        output_path: Path
    ) -> Dict[str, Any]:
        """
        Analyze subject-level data quality by identifying subjects with pervasive outlier voxels.

        This complements voxel-level MAD analysis by asking: which *subjects* have
        widespread data quality problems, rather than which *voxels* are unreliable.

        For each subject, count how many voxels show extreme values (>3 MAD from group median).
        Subjects with high outlier percentages may have scan quality issues.

        Parameters
        ----------
        imaging_views : List[np.ndarray]
            List of imaging data matrices (n_subjects × n_voxels per view)
        view_names : List[str]
            Names of imaging views
        raw_data : Dict
            Raw data dictionary containing subject_ids if available
        output_path : Path
            Output directory for plots and tables

        Returns
        -------
        Dict[str, Any]
            Subject outlier analysis results including flagged subjects
        """
        logger.info("👤 Analyzing subject-level data quality...")

        subject_ids = raw_data.get("subject_ids", None)
        if subject_ids is None:
            # Generate generic IDs
            n_subjects = imaging_views[0].shape[0] if imaging_views else 0
            subject_ids = [f"Subject_{i:03d}" for i in range(n_subjects)]

        subject_results = {}
        outlier_threshold = 3.0  # MAD threshold for defining outliers

        for X, view_name in zip(imaging_views, view_names):
            logger.info(f"   Processing {view_name}: {X.shape}")

            n_subjects, n_voxels = X.shape

            # For each voxel, calculate group median and MAD
            voxel_medians = np.median(X, axis=0)
            voxel_mads = np.median(np.abs(X - voxel_medians), axis=0)

            # Count outlier voxels for each subject
            subject_outlier_counts = []
            subject_outlier_pcts = []

            for subj_idx in range(n_subjects):
                subject_data = X[subj_idx, :]
                outlier_count = 0

                for voxel_idx in range(n_voxels):
                    voxel_value = subject_data[voxel_idx]
                    voxel_median = voxel_medians[voxel_idx]
                    voxel_mad = voxel_mads[voxel_idx]

                    if not np.isnan(voxel_value) and voxel_mad > 0:
                        # Calculate MAD score for this subject at this voxel
                        mad_score = np.abs(voxel_value - voxel_median) / (voxel_mad * 1.4826)
                        if mad_score > outlier_threshold:
                            outlier_count += 1

                outlier_pct = 100.0 * outlier_count / n_voxels
                subject_outlier_counts.append(outlier_count)
                subject_outlier_pcts.append(outlier_pct)

            subject_results[view_name] = {
                "outlier_counts": subject_outlier_counts,
                "outlier_percentages": subject_outlier_pcts,
                "n_subjects": n_subjects,
                "n_voxels": n_voxels,
            }

            logger.info(f"      Mean outlier %: {np.mean(subject_outlier_pcts):.1f}%")
            logger.info(f"      Max outlier %: {np.max(subject_outlier_pcts):.1f}%")

        # Create visualizations
        self._plot_subject_outlier_distributions(
            subject_results, view_names, subject_ids, output_path
        )

        # Identify flagged subjects (>5% outlier voxels in any ROI)
        flagged_subjects = self._identify_flagged_subjects(
            subject_results, view_names, subject_ids, threshold_pct=5.0, output_path=output_path
        )

        # Create additional subject outlier visualizations
        self._plot_subject_outlier_comparison(
            subject_results, view_names, subject_ids, flagged_subjects, output_path
        )
        self._plot_subject_outlier_scatter(
            subject_results, view_names, subject_ids, flagged_subjects, output_path
        )
        if len(view_names) > 1:
            self._plot_subject_outlier_heatmap(
                subject_results, view_names, subject_ids, flagged_subjects, output_path
            )
        self._plot_top_outlier_subjects(
            subject_results, view_names, subject_ids, flagged_subjects, output_path
        )

        logger.info(f"   ✅ Subject-level analysis completed: {len(flagged_subjects)} subjects flagged")
        return {
            "subject_results": subject_results,
            "flagged_subjects": flagged_subjects,
            "outlier_threshold_mad": outlier_threshold,
            "flagging_threshold_pct": 5.0,
        }

    def _plot_subject_outlier_distributions(
        self,
        subject_results: Dict[str, Any],
        view_names: List[str],
        subject_ids: List[str],
        output_path: Path
    ):
        """Create histograms showing distribution of subject outlier percentages."""
        fig, axes = plt.subplots(len(view_names), 1, figsize=(12, 5 * len(view_names)))
        if len(view_names) == 1:
            axes = [axes]

        for idx, (view_name, ax) in enumerate(zip(view_names, axes)):
            outlier_pcts = subject_results[view_name]["outlier_percentages"]

            # Histogram of outlier percentages
            ax.hist(outlier_pcts, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(5.0, color='red', linestyle='--', linewidth=2,
                      label='5% threshold (flagging criterion)')
            ax.axvline(np.mean(outlier_pcts), color='green', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(outlier_pcts):.1f}%')

            ax.set_xlabel('Outlier Voxels (%)', fontsize=12)
            ax.set_ylabel('Number of Subjects', fontsize=12)
            ax.set_title(f'Subject-Level Data Quality - {view_name}\n'
                        f'Distribution of outlier percentages across {len(outlier_pcts)} subjects',
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)

            # Add statistics text box
            stats_text = (
                f"Min: {np.min(outlier_pcts):.1f}%\n"
                f"Q1: {np.percentile(outlier_pcts, 25):.1f}%\n"
                f"Median: {np.median(outlier_pcts):.1f}%\n"
                f"Q3: {np.percentile(outlier_pcts, 75):.1f}%\n"
                f"Max: {np.max(outlier_pcts):.1f}%\n"
                f"Flagged (>5%): {np.sum(np.array(outlier_pcts) > 5.0)}"
            )
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plot_file = output_path / "subject_outlier_distributions.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"      Saved subject outlier distribution plot: {plot_file.name}")

    def _plot_subject_outlier_comparison(
        self,
        subject_results: Dict[str, Any],
        view_names: List[str],
        subject_ids: List[str],
        flagged_subjects: List[Dict[str, Any]],
        output_path: Path
    ):
        """Create box plot comparing flagged vs non-flagged subjects."""
        fig, axes = plt.subplots(1, len(view_names), figsize=(6 * len(view_names), 6))
        if len(view_names) == 1:
            axes = [axes]

        flagged_ids = {s["subject_id"] for s in flagged_subjects}

        for idx, (view_name, ax) in enumerate(zip(view_names, axes)):
            outlier_pcts = subject_results[view_name]["outlier_percentages"]

            # Separate flagged and non-flagged
            flagged_pcts = [pct for i, pct in enumerate(outlier_pcts) if subject_ids[i] in flagged_ids]
            normal_pcts = [pct for i, pct in enumerate(outlier_pcts) if subject_ids[i] not in flagged_ids]

            # Box plot
            bp = ax.boxplot([normal_pcts, flagged_pcts], labels=['Normal', 'Flagged'],
                           patch_artist=True, widths=0.6)

            # Color boxes
            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][1].set_facecolor('salmon')

            # Add threshold line
            ax.axhline(5.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                      label='5% threshold')

            ax.set_ylabel('Outlier Voxels (%)', fontsize=12)
            ax.set_title(f'Subject Quality Comparison - {view_name}\n'
                        f'Normal (n={len(normal_pcts)}) vs Flagged (n={len(flagged_pcts)})',
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(fontsize=11)

            # Add statistics
            if flagged_pcts:
                stats_text = (
                    f"Normal subjects:\n"
                    f"  Median: {np.median(normal_pcts):.1f}%\n"
                    f"  Mean: {np.mean(normal_pcts):.1f}%\n\n"
                    f"Flagged subjects:\n"
                    f"  Median: {np.median(flagged_pcts):.1f}%\n"
                    f"  Mean: {np.mean(flagged_pcts):.1f}%"
                )
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plot_file = output_path / "subject_outlier_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"      Saved subject outlier comparison plot: {plot_file.name}")

    def _plot_subject_outlier_scatter(
        self,
        subject_results: Dict[str, Any],
        view_names: List[str],
        subject_ids: List[str],
        flagged_subjects: List[Dict[str, Any]],
        output_path: Path
    ):
        """Create scatter plot showing outlier percentages per subject."""
        fig, axes = plt.subplots(len(view_names), 1, figsize=(14, 5 * len(view_names)))
        if len(view_names) == 1:
            axes = [axes]

        flagged_ids = {s["subject_id"] for s in flagged_subjects}

        for idx, (view_name, ax) in enumerate(zip(view_names, axes)):
            outlier_pcts = subject_results[view_name]["outlier_percentages"]
            n_subjects = len(outlier_pcts)

            # Separate flagged and non-flagged
            normal_indices = [i for i in range(n_subjects) if subject_ids[i] not in flagged_ids]
            flagged_indices = [i for i in range(n_subjects) if subject_ids[i] in flagged_ids]

            # Scatter plot
            ax.scatter(normal_indices, [outlier_pcts[i] for i in normal_indices],
                      c='steelblue', alpha=0.6, s=50, label='Normal')
            ax.scatter(flagged_indices, [outlier_pcts[i] for i in flagged_indices],
                      c='red', alpha=0.8, s=100, marker='X', label='Flagged', zorder=5)

            # Add threshold line
            ax.axhline(5.0, color='red', linestyle='--', linewidth=2, alpha=0.5,
                      label='5% threshold')

            # Annotate flagged subjects
            for i in flagged_indices:
                ax.annotate(subject_ids[i], (i, outlier_pcts[i]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, color='darkred',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color='red', lw=1))

            ax.set_xlabel('Subject Index', fontsize=12)
            ax.set_ylabel('Outlier Voxels (%)', fontsize=12)
            ax.set_title(f'Subject-Level Outlier Detection - {view_name}\n'
                        f'{len(flagged_indices)} flagged subjects (>5% outliers)',
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11, loc='upper left')

        plt.tight_layout()
        plot_file = output_path / "subject_outlier_scatter.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"      Saved subject outlier scatter plot: {plot_file.name}")

    def _plot_subject_outlier_heatmap(
        self,
        subject_results: Dict[str, Any],
        view_names: List[str],
        subject_ids: List[str],
        flagged_subjects: List[Dict[str, Any]],
        output_path: Path
    ):
        """Create heatmap showing outlier percentages across subjects and ROIs."""
        import seaborn as sns

        n_subjects = len(subject_ids)
        n_views = len(view_names)

        # Build matrix: subjects × ROIs
        outlier_matrix = np.zeros((n_subjects, n_views))
        for view_idx, view_name in enumerate(view_names):
            outlier_pcts = subject_results[view_name]["outlier_percentages"]
            outlier_matrix[:, view_idx] = outlier_pcts

        # Sort subjects by max outlier percentage (worst first)
        max_outliers = np.max(outlier_matrix, axis=1)
        sorted_indices = np.argsort(max_outliers)[::-1]
        outlier_matrix_sorted = outlier_matrix[sorted_indices, :]
        subject_ids_sorted = [subject_ids[i] for i in sorted_indices]

        # Show only top 50 subjects if there are many
        if n_subjects > 50:
            outlier_matrix_sorted = outlier_matrix_sorted[:50, :]
            subject_ids_sorted = subject_ids_sorted[:50]

        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(8, n_views * 2), max(10, len(subject_ids_sorted) * 0.3)))

        # Clean view names for display
        clean_view_names = [v.replace("volume_", "").replace("_voxels", "") for v in view_names]

        # Plot heatmap
        sns.heatmap(outlier_matrix_sorted, annot=False, fmt='.1f', cmap='YlOrRd',
                   xticklabels=clean_view_names, yticklabels=subject_ids_sorted,
                   cbar_kws={'label': 'Outlier Voxels (%)'}, ax=ax, vmin=0, vmax=20)

        # Add threshold contour
        threshold_mask = outlier_matrix_sorted > 5.0
        for i in range(threshold_mask.shape[0]):
            for j in range(threshold_mask.shape[1]):
                if threshold_mask[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                              edgecolor='blue', lw=2))

        ax.set_title(f'Subject-ROI Outlier Heatmap (Top {len(subject_ids_sorted)} subjects)\n'
                    f'Blue boxes = flagged (>5% outliers)',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('ROI', fontsize=12)
        ax.set_ylabel('Subject ID', fontsize=12)

        plt.tight_layout()
        plot_file = output_path / "subject_outlier_heatmap.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"      Saved subject outlier heatmap: {plot_file.name}")

    def _plot_top_outlier_subjects(
        self,
        subject_results: Dict[str, Any],
        view_names: List[str],
        subject_ids: List[str],
        flagged_subjects: List[Dict[str, Any]],
        output_path: Path
    ):
        """Create bar chart showing top 20 subjects with highest outlier percentages."""
        # Calculate max outlier percentage across all ROIs for each subject
        n_subjects = len(subject_ids)
        max_outliers = np.zeros(n_subjects)

        for subj_idx in range(n_subjects):
            max_pct = 0
            for view_name in view_names:
                pct = subject_results[view_name]["outlier_percentages"][subj_idx]
                max_pct = max(max_pct, pct)
            max_outliers[subj_idx] = max_pct

        # Get top 20 subjects
        top_n = min(20, n_subjects)
        top_indices = np.argsort(max_outliers)[::-1][:top_n]
        top_subjects = [subject_ids[i] for i in top_indices]
        top_values = [max_outliers[i] for i in top_indices]

        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))

        # Color bars based on flagging threshold
        colors = ['red' if v > 5.0 else 'steelblue' for v in top_values]

        bars = ax.barh(range(top_n), top_values, color=colors, alpha=0.7, edgecolor='black')

        # Add threshold line
        ax.axvline(5.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                  label='5% flagging threshold')

        # Annotate bars with values
        for i, (bar, value) in enumerate(zip(bars, top_values)):
            ax.text(value + 0.3, i, f'{value:.1f}%', va='center', fontsize=9)

        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_subjects, fontsize=10)
        ax.set_xlabel('Max Outlier Percentage Across All ROIs (%)', fontsize=12)
        ax.set_ylabel('Subject ID', fontsize=12)
        ax.set_title(f'Top {top_n} Subjects by Outlier Percentage\n'
                    f'Red bars = flagged (>5% outliers)',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(fontsize=11)

        # Invert y-axis to show highest at top
        ax.invert_yaxis()

        plt.tight_layout()
        plot_file = output_path / "subject_outlier_top20.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"      Saved top subjects bar chart: {plot_file.name}")

    def _identify_flagged_subjects(
        self,
        subject_results: Dict[str, Any],
        view_names: List[str],
        subject_ids: List[str],
        threshold_pct: float,
        output_path: Path
    ) -> List[Dict[str, Any]]:
        """
        Identify subjects who exceed outlier threshold in any ROI.

        Parameters
        ----------
        subject_results : Dict
            Subject outlier analysis results per view
        view_names : List[str]
            View names
        subject_ids : List[str]
            Subject identifiers
        threshold_pct : float
            Percentage threshold for flagging (e.g., 5.0 = 5%)
        output_path : Path
            Output directory

        Returns
        -------
        List[Dict]
            List of flagged subjects with details
        """
        flagged_subjects = []

        n_subjects = len(subject_ids)

        for subj_idx in range(n_subjects):
            subject_id = subject_ids[subj_idx]
            outlier_pcts_by_roi = {}
            is_flagged = False

            for view_name in view_names:
                outlier_pct = subject_results[view_name]["outlier_percentages"][subj_idx]
                outlier_pcts_by_roi[view_name] = outlier_pct

                if outlier_pct > threshold_pct:
                    is_flagged = True

            if is_flagged:
                flagged_subjects.append({
                    "subject_id": subject_id,
                    "subject_index": subj_idx,
                    "outlier_percentages": outlier_pcts_by_roi,
                    "max_outlier_pct": max(outlier_pcts_by_roi.values()),
                })

        # Save flagged subjects table
        if flagged_subjects:
            self._save_flagged_subjects_table(flagged_subjects, view_names, threshold_pct, output_path)
        else:
            logger.debug("      No subjects flagged (all below threshold)")

        return flagged_subjects

    def _save_flagged_subjects_table(
        self,
        flagged_subjects: List[Dict],
        view_names: List[str],
        threshold_pct: float,
        output_path: Path
    ):
        """Save table of flagged subjects to CSV."""
        # Build table rows
        rows = []
        for subject_info in flagged_subjects:
            row = {
                "Subject_ID": subject_info["subject_id"],
                "Subject_Index": subject_info["subject_index"],
                "Max_Outlier_Pct": subject_info["max_outlier_pct"],
            }
            # Add per-ROI outlier percentages
            for view_name in view_names:
                clean_name = view_name.replace("volume_", "").replace("_voxels", "")
                row[f"{clean_name}_Outlier_Pct"] = subject_info["outlier_percentages"][view_name]

            rows.append(row)

        flagged_df = pd.DataFrame(rows)

        # Sort by max outlier percentage (worst first)
        flagged_df = flagged_df.sort_values("Max_Outlier_Pct", ascending=False)

        # Save to CSV
        csv_file = output_path / "flagged_subjects.csv"
        flagged_df.to_csv(csv_file, index=False)
        logger.info(f"      Saved flagged subjects table: {csv_file.name} ({len(flagged_subjects)} subjects)")

        # Also create a human-readable text file
        txt_file = output_path / "flagged_subjects_report.txt"
        with open(txt_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SUBJECT-LEVEL DATA QUALITY FLAGGING REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Flagging criterion: >{threshold_pct:.1f}% outlier voxels in any ROI\n")
            f.write(f"Total subjects flagged: {len(flagged_subjects)}\n\n")

            if len(flagged_subjects) > 0:
                f.write("FLAGGED SUBJECTS (sorted by severity):\n")
                f.write("-" * 80 + "\n\n")

                for subject_info in sorted(flagged_subjects, key=lambda x: x["max_outlier_pct"], reverse=True):
                    f.write(f"Subject ID: {subject_info['subject_id']}\n")
                    f.write(f"  Index: {subject_info['subject_index']}\n")
                    f.write(f"  Max outlier percentage: {subject_info['max_outlier_pct']:.2f}%\n")
                    f.write(f"  Outlier percentages by ROI:\n")
                    for view_name, pct in subject_info["outlier_percentages"].items():
                        flag = "⚠️ FLAGGED" if pct > threshold_pct else "✓ OK"
                        f.write(f"    - {view_name}: {pct:.2f}% {flag}\n")
                    f.write("\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("RECOMMENDATIONS:\n")
                f.write("=" * 80 + "\n\n")
                f.write("1. INSPECT FLAGGED SUBJECTS:\n")
                f.write("   - Review scan acquisition notes for these subject IDs\n")
                f.write("   - Check for motion artifacts, scanner issues, or protocol deviations\n\n")

                f.write("2. CONSIDER EXCLUSION IF:\n")
                f.write("   - Outlier percentage >10% in any ROI (severe quality issues)\n")
                f.write("   - Known scan quality problems from acquisition notes\n")
                f.write("   - Visual inspection confirms widespread artifacts\n\n")

                f.write("3. RETAIN SUBJECTS IF:\n")
                f.write("   - Outlier percentage 5-10% (mild issues, may be biological)\n")
                f.write("   - No known scan quality problems\n")
                f.write("   - Outliers may represent real biological variability (PD subtypes)\n\n")

                f.write("4. NEXT STEPS:\n")
                f.write("   - For severe cases (>10%), consider exclusion from analysis\n")
                f.write("   - For mild cases (5-10%), monitor impact on factor stability\n")
                f.write("   - Document exclusion criteria if subjects are removed\n")

        logger.info(f"      Saved flagged subjects report: {txt_file.name}")

    def _generate_threshold_summary(
        self,
        view_names: List[str],
        mad_distributions: Dict[str, Any],
        elbow_analysis: Dict[str, Any],
        spatial_analysis: Dict[str, Any],
        info_preservation: Dict[str, Any],
        candidate_thresholds: List[float],
        output_path: Path
    ) -> pd.DataFrame:
        """
        Generate comprehensive summary table comparing thresholds across all ROIs.

        Also creates a text file with data-driven recommendations.
        """
        logger.info("📋 Generating threshold comparison summary...")

        # Build summary table
        summary_rows = []

        for view_name in view_names:
            elbow_threshold = elbow_analysis[view_name]["elbow_threshold"]
            elbow_retention = elbow_analysis[view_name]["retention_at_elbow"]

            for threshold in candidate_thresholds:
                retention_data = mad_distributions[view_name]["retention_rates"][threshold]
                info_data = info_preservation[view_name][threshold]

                row = {
                    "ROI": view_name,
                    "Threshold": threshold,
                    "Voxels_Retained": retention_data["n_retained"],
                    "Retention_Pct": retention_data["retention_pct"],
                    "Voxels_Removed": retention_data["n_removed"],
                    "Variance_Ratio": info_data["variance_ratio"],
                    "Correlation_Preservation": info_data["correlation_preservation"],
                    "Is_Elbow": abs(threshold - elbow_threshold) < 0.3,
                }
                summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)

        # Save summary table
        summary_file = output_path / "mad_threshold_summary_table.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"   Saved summary table: {summary_file.name}")

        # Generate recommendations text file
        self._write_threshold_recommendations(
            view_names, mad_distributions, elbow_analysis,
            spatial_analysis, info_preservation, output_path
        )

        return summary_df

    def _write_threshold_recommendations(
        self,
        view_names: List[str],
        mad_distributions: Dict[str, Any],
        elbow_analysis: Dict[str, Any],
        spatial_analysis: Dict[str, Any],
        info_preservation: Dict[str, Any],
        output_path: Path
    ):
        """Write data-driven recommendations to text file."""
        rec_file = output_path / "mad_threshold_recommendations.txt"

        with open(rec_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MAD THRESHOLD RECOMMENDATIONS\n")
            f.write("Data-Driven Analysis of Optimal Outlier Detection Thresholds\n")
            f.write("=" * 80 + "\n\n")

            f.write("CURRENT CONFIGURATION: threshold = 3.0\n\n")

            for view_name in view_names:
                f.write(f"\n{view_name}\n")
                f.write("-" * 80 + "\n")

                # Elbow analysis recommendation
                elbow_threshold = elbow_analysis[view_name]["elbow_threshold"]
                elbow_retention = elbow_analysis[view_name]["retention_at_elbow"]
                interpretation = elbow_analysis[view_name]["interpretation"]

                f.write(f"ELBOW POINT: {elbow_threshold:.2f}\n")
                f.write(f"  - Retention at elbow: {elbow_retention:.1f}%\n")
                f.write(f"  - Interpretation: {interpretation}\n\n")

                # Current threshold (3.0) performance
                current_retention = mad_distributions[view_name]["retention_rates"][3.0]
                current_info = info_preservation[view_name][3.0]

                f.write(f"CURRENT THRESHOLD (3.0) PERFORMANCE:\n")
                f.write(f"  - Voxels retained: {current_retention['n_retained']} ({current_retention['retention_pct']:.1f}%)\n")
                f.write(f"  - Voxels removed: {current_retention['n_removed']}\n")
                f.write(f"  - Variance preserved: {100*current_info['variance_ratio']:.1f}%\n")
                if current_info['correlation_preservation']:
                    f.write(f"  - Correlation preserved: {100*current_info['correlation_preservation']:.1f}%\n")
                f.write("\n")

                # Spatial distribution insights
                if view_name in spatial_analysis and spatial_analysis[view_name].get("position_file_found"):
                    spatial_data = spatial_analysis[view_name]
                    if "clustering_metrics" in spatial_data:
                        metrics = spatial_data["clustering_metrics"]
                        f.write(f"SPATIAL DISTRIBUTION:\n")
                        f.write(f"  - Pattern: {metrics['interpretation']}\n")
                        f.write(f"  - Centroid separation: {metrics['centroid_separation']:.2f} mm\n\n")

                # Recommendation
                f.write("RECOMMENDATION:\n")
                if abs(elbow_threshold - 3.0) < 0.5:
                    f.write(f"  ✓ Current threshold (3.0) is appropriate - close to elbow point\n")
                    f.write(f"    Provides good balance between artifact removal and signal preservation\n")
                elif elbow_threshold > 3.5:
                    f.write(f"  → Consider INCREASING threshold to {elbow_threshold:.1f}\n")
                    f.write(f"    Current threshold may be too strict, removing biological variability\n")
                    f.write(f"    Elbow analysis suggests {elbow_threshold:.1f} provides better data retention\n")
                else:
                    f.write(f"  → Current threshold (3.0) is reasonable but conservative\n")
                    f.write(f"    Elbow suggests {elbow_threshold:.1f} but current value retains more signal\n")
                f.write("\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("OVERALL RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")

            # Calculate average elbow across all views
            avg_elbow = np.mean([elbow_analysis[v]["elbow_threshold"] for v in view_names])
            f.write(f"Average elbow point across all ROIs: {avg_elbow:.2f}\n\n")

            if avg_elbow > 3.5:
                f.write("RECOMMENDATION: Consider INCREASING threshold to ~4.0 or 4.5\n")
                f.write("  - Current threshold (3.0) appears too strict for your data\n")
                f.write("  - You are removing substantial biological variability\n")
                f.write("  - Higher threshold would improve spatial coverage and signal preservation\n")
            elif avg_elbow < 2.5:
                f.write("RECOMMENDATION: Consider DECREASING threshold to ~2.0 or 2.5\n")
                f.write("  - Current threshold (3.0) may be too permissive\n")
                f.write("  - Lower threshold would provide cleaner data by removing more artifacts\n")
            else:
                f.write("RECOMMENDATION: Current threshold (3.0) is appropriate\n")
                f.write("  - Well-aligned with data-driven elbow analysis\n")
                f.write("  - Provides good balance for your dataset\n")

            f.write("\nFor exploratory factor analysis seeking biological subtypes:\n")
            f.write("  → Consider threshold range 3.5-5.0 to preserve heterogeneous patterns\n")
            f.write("  → 'Outliers' may represent distinct disease subtypes\n")
            f.write("\nFor clean signal extraction and hypothesis testing:\n")
            f.write("  → Consider threshold range 2.5-3.5 for better signal-to-noise ratio\n")
            f.write("  → Removes artifacts while preserving main effects\n")

        logger.info(f"   📝 Saved recommendations: {rec_file.name}")

    def _create_filtered_position_lookups(
        self, raw_data: Dict, preprocessed_data: Dict, data_dir: str, output_dir: Path
    ) -> Dict[str, str]:
        """
        Create filtered position lookups matching preprocessed voxels.

        Tracks all columns dropped during preprocessing (duplicates + MAD filtering)
        and removes corresponding rows from position lookup files.

        Returns mapping of view_name -> saved position lookup path
        """
        from pathlib import Path
        from data.preprocessing import SpatialProcessingUtils
        import numpy as np
        import pandas as pd

        filtered_paths = {}
        view_names = preprocessed_data.get("view_names", [])
        X_list_raw = raw_data.get("X_list", [])
        X_list_preprocessed = preprocessed_data.get("X_list", [])

        logger.info(f"  _create_filtered_position_lookups called:")
        logger.info(f"    view_names: {view_names}")
        logger.info(f"    n_views raw: {len(X_list_raw)}, preprocessed: {len(X_list_preprocessed)}")

        if len(X_list_raw) != len(X_list_preprocessed):
            logger.warning("Raw and preprocessed data have different number of views, skipping position lookup creation")
            return {}

        for idx, view_name in enumerate(view_names):
            # Only process imaging views
            if view_name == "clinical":
                logger.info(f"  Skipping clinical view")
                continue

            # Check if this is an imaging view
            is_imaging = view_name.startswith("volume_") or view_name == "imaging"
            if not is_imaging:
                logger.info(f"  Skipping non-imaging view: {view_name}")
                continue

            logger.info(f"  Creating filtered position lookup for {view_name}...")

            # Determine ROI name
            if view_name.startswith("volume_"):
                roi_name = view_name.replace("volume_", "").replace("_voxels", "")
            elif view_name == "imaging":
                # For single-view mode, try common ROIs
                roi_candidates = ["sn", "putamen", "lentiform", "caudate", "thalamus"]
                roi_name = None
                for candidate in roi_candidates:
                    test_pos = SpatialProcessingUtils.load_position_lookup(data_dir, candidate)
                    if test_pos is not None and len(test_pos) == X_list_raw[idx].shape[1]:
                        roi_name = candidate
                        logger.info(f"  Auto-detected ROI: {roi_name}")
                        break
                if roi_name is None:
                    logger.warning(f"  Could not determine ROI for 'imaging' view, skipping")
                    continue
            else:
                continue

            # Load original position lookup
            positions = SpatialProcessingUtils.load_position_lookup(data_dir, roi_name)
            if positions is None:
                logger.warning(f"  No position lookup found for {roi_name}, skipping")
                continue

            logger.info(f"  Original positions: {len(positions)} rows")

            # Get raw and preprocessed dimensions
            n_raw_voxels = X_list_raw[idx].shape[1]
            n_preprocessed_voxels = X_list_preprocessed[idx].shape[1]

            logger.info(f"  Raw data: {n_raw_voxels} voxels")
            logger.info(f"  Preprocessed data: {n_preprocessed_voxels} voxels")

            # Create mask of kept columns
            # We need to identify which columns from raw were kept in preprocessed
            if n_raw_voxels == n_preprocessed_voxels:
                logger.info(f"  No voxels dropped, position lookup unchanged")
                keep_mask = np.ones(n_raw_voxels, dtype=bool)
            else:
                # Columns were dropped - we need to infer which ones
                # This is tricky without explicit tracking, so we'll use a heuristic:
                # Assume the first n_preprocessed_voxels were kept (simple truncation)
                # This matches the common pattern of MAD filtering creating a boolean mask
                logger.warning(f"  {n_raw_voxels - n_preprocessed_voxels} voxels dropped")
                logger.warning(f"  Cannot determine exact column mapping without explicit tracking")
                logger.warning(f"  Skipping position lookup for {view_name}")
                continue

            # Apply mask to positions
            filtered_positions = positions[keep_mask].copy()
            filtered_positions.reset_index(drop=True, inplace=True)

            # Save filtered position lookup
            # Ensure output_dir is a Path object
            output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
            position_output_dir = output_dir / "filtered_position_lookups"
            position_output_dir.mkdir(exist_ok=True, parents=True)

            output_file = position_output_dir / f"{roi_name}_filtered_position_lookup.csv"
            filtered_positions.to_csv(output_file, index=False)

            filtered_paths[view_name] = str(output_file)
            logger.info(f"  ✅ Saved filtered position lookup: {output_file}")
            logger.info(f"     {len(positions)} → {len(filtered_positions)} rows")

        return filtered_paths

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
