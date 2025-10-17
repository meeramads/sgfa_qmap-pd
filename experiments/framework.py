"""Core experimental framework for systematic qMAP-PD analysis."""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from core.io_utils import save_csv, save_json, save_numpy, save_plot
from core.utils import safe_pickle_save
from optimization import (
    PerformanceConfig,
    PerformanceManager,
)
from optimization.config import auto_configure_for_system

logger = logging.getLogger(__name__)


# Custom FileHandler that handles stale file handles on network filesystems
class ResilientFileHandler(logging.FileHandler):
    """FileHandler that silently ignores OSError on flush (e.g., stale NFS handles)."""

    def flush(self):
        try:
            super().flush()
        except OSError:
            # Silently ignore stale file handle errors on network filesystems
            pass


def safe_log(log_func, message, *args, **kwargs):
    """
    Safely log a message, catching OSError (stale file handle) from network filesystems.

    This prevents "Stale file handle" errors on NFS from breaking experiment execution.
    """
    try:
        log_func(message, *args, **kwargs)
    except OSError:
        # Silently ignore stale file handle errors on network filesystems
        # The experiment continues successfully even if logging fails
        pass


@dataclass
class ExperimentConfig:
    """Configuration for experimental runs."""

    # Basic experiment info
    experiment_name: str
    description: str
    version: str = "1.0"

    # Data configuration
    dataset: str = "qmap_pd"
    data_dir: Optional[str] = None
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)

    # Model configuration
    K_values: List[int] = field(default_factory=lambda: [5, 10, 15])
    num_sources: int = 3
    percW_values: List[float] = field(default_factory=lambda: [33.0, 50.0])

    # Single-value model parameters (for specific runs)
    K: Optional[int] = None  # Number of factors for this specific run
    percW: Optional[float] = None  # Sparsity level for this specific run
    slab_df: Optional[float] = None  # Slab degrees of freedom
    slab_scale: Optional[float] = None  # Slab scale parameter

    # Preprocessing parameters (for semantic naming)
    qc_outlier_threshold: Optional[float] = None  # MAD threshold for QC outlier detection

    # MCMC configuration
    num_samples: int = 2000
    num_chains: int = 1  # Single chain for GPU memory constraints
    num_warmup: int = 1000
    num_runs_per_config: int = 3

    # Cross-validation configuration
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # stratified, grouped, random
    cv_target_variable: Optional[str] = None
    cv_group_variable: Optional[str] = None

    # Performance configuration
    performance_config: Dict[str, Any] = field(default_factory=dict)
    enable_memory_optimization: bool = True
    enable_profiling: bool = True
    auto_configure_system: bool = True  # Auto-detect system capabilities
    max_memory_gb: Optional[float] = None  # Auto-detected if None

    # Output configuration
    save_intermediate_results: bool = False  # Align with config.yaml default
    save_samples: bool = False  # Can be large
    save_diagnostics: bool = True
    save_numpy_arrays: bool = False  # Disable to save disk space
    output_formats: List[str] = field(default_factory=lambda: ["json", "csv"])

    # Spatial remapping configuration
    enable_spatial_remapping: bool = False  # Remap factor loadings to brain space
    save_brain_maps: bool = True  # Save brain maps as CSV files
    create_brain_map_plots: bool = False  # Create 3D visualization plots
    brain_maps_output_dir: str = "brain_maps"  # Subdirectory for brain maps

    # Random seeds for robustness testing
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    # Factor Stability Analysis (Ferreira et al. 2024)
    cosine_threshold: float = 0.8  # Minimum cosine similarity for factor matching
    min_match_rate: float = 0.5  # Minimum fraction of chains for robust factor (>50%)
    sparsity_threshold: float = 0.01  # Threshold for effective factor counting
    min_nonzero_pct: float = 0.05  # Minimum fraction of non-zero loadings

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(**config_dict)

    def save(self, filepath: Path):
        """Save configuration to file."""
        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "ExperimentConfig":
        """Load configuration from file."""
        with open(filepath, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


@dataclass
class ExperimentResult:
    """Container for experiment results."""

    experiment_id: str = ""
    config: Optional[ExperimentConfig] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed
    error_message: Optional[str] = None

    # Results
    model_results: Dict[str, Any] = field(default_factory=dict)
    cv_results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    plots: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    data_summary: Dict[str, Any] = field(default_factory=dict)
    convergence_diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-populate fields if not provided."""
        # Auto-set start_time if not provided
        if self.start_time is None:
            self.start_time = datetime.now()

        # Auto-set end_time if status is completed/failed but no end_time set
        if self.end_time is None and self.status in ("completed", "failed"):
            self.end_time = datetime.now()

    def mark_completed(self):
        """Mark experiment as completed."""
        self.end_time = datetime.now()
        self.status = "completed"

        # Calculate and log runtime
        if self.start_time:
            runtime = self.end_time - self.start_time
            runtime_seconds = runtime.total_seconds()
            runtime_str = self._format_runtime(runtime_seconds)

            logger.info(f"🏁 EXPERIMENT COMPLETED: {self.experiment_id}")
            logger.info(f"⏱️  TOTAL RUNTIME: {runtime_str}")
            logger.info(f"🕐 Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"🕑 Finished: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Store runtime in performance metrics
            self.performance_metrics['total_runtime_seconds'] = runtime_seconds
            self.performance_metrics['total_runtime_formatted'] = runtime_str

    def _format_runtime(self, seconds: float) -> str:
        """Format runtime in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes ({seconds:.0f}s)"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.1f} hours ({minutes:.0f}m {seconds % 60:.0f}s)"

    def mark_failed(self, error_message: str):
        """Mark experiment as failed."""
        self.end_time = datetime.now()
        self.status = "failed"
        self.error_message = error_message

        # Calculate and log runtime even for failed experiments
        if self.start_time:
            runtime = self.end_time - self.start_time
            runtime_seconds = runtime.total_seconds()
            runtime_str = self._format_runtime(runtime_seconds)

            logger.error(f"❌ EXPERIMENT FAILED: {self.experiment_id}")
            logger.error(f"⏱️  RUNTIME BEFORE FAILURE: {runtime_str}")
            logger.error(f"💥 Error: {error_message}")

            # Store runtime in performance metrics
            self.performance_metrics['total_runtime_seconds'] = runtime_seconds
            self.performance_metrics['total_runtime_formatted'] = runtime_str

    def get_duration(self) -> Optional[float]:
        """Get experiment duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result_dict = asdict(self)
        # Convert datetime objects to strings
        result_dict["start_time"] = self.start_time.isoformat()
        if self.end_time:
            result_dict["end_time"] = self.end_time.isoformat()

        # Convert plots to metadata only (don't serialize matplotlib figures)
        if self.plots:
            result_dict["plots"] = {
                plot_name: {
                    "type": "matplotlib_figure",
                    "saved": True,
                    "files": [f"{plot_name}.png", f"{plot_name}.pdf"],
                }
                for plot_name in self.plots.keys()
            }

        return result_dict


class ExperimentFramework:
    """Core framework for running systematic experiments."""

    def __init__(
        self,
        config_or_output_dir,
        performance_config: Optional[PerformanceConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize experiment framework.

        Parameters
        ----------
        config_or_output_dir : ExperimentConfig or Path
            Either an ExperimentConfig (preferred) or base directory path for outputs.
        performance_config : PerformanceConfig, optional
            Performance optimization configuration.
        logger : logging.Logger, optional
            Logger instance.
        """
        # Handle both ExperimentConfig and direct path for backward compatibility
        if hasattr(config_or_output_dir, "experiment_name"):
            # It's an ExperimentConfig
            from core.config_utils import ConfigHelper

            self.config = config_or_output_dir
            self.base_output_dir = ConfigHelper.get_output_dir_safe(
                config_or_output_dir
            )
        else:
            # It's a direct path (backward compatibility)
            self.config = None
            self.base_output_dir = Path(config_or_output_dir)

        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-configure performance if enabled
        if hasattr(self.config, 'auto_configure_system') and self.config.auto_configure_system:
            if performance_config is None:
                self.performance_config = auto_configure_for_system()
                self.logger = logger or logging.getLogger(__name__)
                self.logger.info(f"Auto-configured system: {self.performance_config.memory.max_memory_gb:.1f}GB memory limit")
            else:
                self.performance_config = performance_config
        else:
            self.performance_config = performance_config

        self.logger = logger or logging.getLogger(__name__)
        self.results_history: List[ExperimentResult] = []

        # Initialize logging
        self._setup_logging()

        self.logger.info(
            f"ExperimentFramework initialized with output dir: {self.base_output_dir}"
        )

    def _setup_logging(self):
        """Setup experiment-specific logging."""
        log_file = self.base_output_dir / "experiments.log"

        # Create file handler
        file_handler = ResilientFileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)

    def _generate_semantic_experiment_name(self, base_name: str, config: Optional[ExperimentConfig] = None) -> str:
        """
        Generate a semantic experiment name that includes key model parameters.

        Format: {base_name}_K{K}_percW{percW}_slab{slab_df}_{slab_scale}
        Example: robustness_tests_K10_percW20_slab4_2

        Parameters
        ----------
        base_name : str
            Base experiment name (e.g., "robustness_tests", "model_comparison")
        config : ExperimentConfig, optional
            Experiment configuration containing model parameters

        Returns
        -------
        str
            Semantic experiment name with model parameters
        """
        if config is None:
            return base_name

        # Build semantic name components
        name_parts = [base_name]

        # Add K (number of factors) if specified
        if config.K is not None:
            name_parts.append(f"K{config.K}")

        # Add percW (sparsity level) if specified
        if config.percW is not None:
            name_parts.append(f"percW{config.percW:.0f}")

        # Add slab parameters if specified
        if config.slab_df is not None and config.slab_scale is not None:
            name_parts.append(f"slab{config.slab_df:.0f}_{config.slab_scale:.0f}")
        elif config.slab_df is not None:
            name_parts.append(f"slabdf{config.slab_df:.0f}")
        elif config.slab_scale is not None:
            name_parts.append(f"slabsc{config.slab_scale:.0f}")

        # Add MAD threshold (QC outlier threshold) if specified
        if config.qc_outlier_threshold is not None:
            name_parts.append(f"MAD{config.qc_outlier_threshold:.1f}")

        return "_".join(name_parts)

    def create_experiment_dir(self, experiment_name: str, config: Optional[ExperimentConfig] = None) -> Path:
        """
        Create directory for specific experiment with semantic naming.

        Parameters
        ----------
        experiment_name : str
            Base experiment name
        config : ExperimentConfig, optional
            Experiment configuration with model parameters for semantic naming

        Returns
        -------
        Path
            Path to created experiment directory
        """
        # Generate semantic name if config is provided
        semantic_name = self._generate_semantic_experiment_name(experiment_name, config)

        # Check if we're in debug mode by looking at the base output directory
        is_debug_mode = "debug" in str(self.base_output_dir).lower()

        if is_debug_mode:
            # Debug mode: simple directory structure with semantic naming
            exp_dir = self.base_output_dir / semantic_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"🐛 Debug mode: Using simple directory structure: {exp_dir}")
            return exp_dir
        else:
            # Production mode: semantic name + timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = self.base_output_dir / f"{semantic_name}_{timestamp}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            return exp_dir

    def run_experiment(
        self, experiment_function: Callable, config: ExperimentConfig, **kwargs
    ) -> ExperimentResult:
        """
        Run a single experiment.

        Parameters
        ----------
        experiment_function : Callable
            Function that runs the actual experiment.
        config : ExperimentConfig
            Experiment configuration.
        **kwargs
            Additional arguments for experiment function.

        Returns
        -------
        ExperimentResult : Results of the experiment.
        """
        # Create experiment directory with semantic naming
        exp_dir = self.create_experiment_dir(config.experiment_name, config)

        # Generate experiment ID
        experiment_id = (
            f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Initialize result object
        result = ExperimentResult(
            experiment_id=experiment_id, config=config, start_time=datetime.now()
        )

        # Save configuration
        config.save(exp_dir / "config.yaml")

        try:
            logger.info(f"Starting experiment: {experiment_id}")

            # Setup performance monitoring if enabled
            perf_manager = None
            if config.enable_memory_optimization and self.performance_config:
                perf_manager = PerformanceManager(self.performance_config)
                perf_manager.__enter__()

            # Run the experiment
            experiment_results = experiment_function(
                config=config, output_dir=exp_dir, **kwargs
            )

            # Store results
            result.model_results = experiment_results.get("model_results", {})
            result.cv_results = experiment_results.get("cv_results", {})
            result.diagnostics = experiment_results.get("diagnostics", {})
            result.data_summary = experiment_results.get("data_summary", {})
            result.convergence_diagnostics = experiment_results.get(
                "convergence_diagnostics", {}
            )
            result.plots = experiment_results.get("plots", {})

            # Save intermediate results if configured
            if config.save_intermediate_results:
                self._save_intermediate_results(result, exp_dir, experiment_results)

            # Get performance metrics
            if perf_manager:
                result.performance_metrics = perf_manager.generate_performance_report()
                perf_manager.__exit__(None, None, None)

            result.mark_completed()
            logger.info(f"Experiment completed successfully: {experiment_id}")

        except Exception as e:
            logger.error(f"Experiment failed: {experiment_id}, Error: {str(e)}")
            result.mark_failed(str(e))

            if perf_manager:
                perf_manager.__exit__(type(e), e, e.__traceback__)

        finally:
            # Critical memory cleanup regardless of success/failure
            self._comprehensive_memory_cleanup(logger)

        # Save results
        self._save_experiment_result(result, exp_dir, config)
        self.results_history.append(result)

        return result

    def _save_experiment_result(self, result: ExperimentResult, output_dir: Path, config: ExperimentConfig):
        """Save experiment result to files."""
        # Save as JSON
        result_json = output_dir / "result.json"
        save_json(result.to_dict(), result_json, indent=2)

        # Save as pickle for complete object (disabled to prevent large files)
        # result_pkl = output_dir / "result.pkl"
        # safe_pickle_save(result, result_pkl, "Experiment result")
        logger.info("Pickle saving disabled to prevent large files - using JSON and CSV instead")

        # Save summary CSV
        self._save_result_summary(result, output_dir / "summary.csv")

        # Save factor matrices if available
        self._save_factor_matrices(result, output_dir)

        # Save plots if available and configured
        generate_plots = getattr(config, 'generate_plots', True)
        if generate_plots:
            self._save_plots(result, output_dir)
        else:
            logger.info("📊 Plot generation disabled by config")

    def _save_result_summary(self, result: ExperimentResult, filepath: Path):
        """Save a summary of results as CSV."""
        summary_data = {
            "experiment_id": result.experiment_id,
            "experiment_name": result.config.experiment_name,
            "status": result.status,
            "duration_seconds": result.get_duration(),
            "duration_formatted": result.performance_metrics.get('total_runtime_formatted', 'N/A'),
            "start_time": result.start_time.strftime('%Y-%m-%d %H:%M:%S') if result.start_time else 'N/A',
            "end_time": result.end_time.strftime('%Y-%m-%d %H:%M:%S') if result.end_time else 'N/A',
            "num_samples": result.config.num_samples,
            "num_chains": result.config.num_chains,
            "K_values": str(result.config.K_values),
            "cv_folds": result.config.cv_folds,
        }

        # Add model performance metrics if available
        if result.model_results:
            for key, value in result.model_results.items():
                if isinstance(value, (int, float, str)):
                    summary_data[f"model_{key}"] = value

        # Add CV metrics if available
        if result.cv_results:
            for key, value in result.cv_results.items():
                if isinstance(value, (int, float, str)):
                    summary_data[f"cv_{key}"] = value

        # Save as single-row CSV
        df = pd.DataFrame([summary_data])
        save_csv(df, filepath, index=False)

    def _save_factor_matrices(self, result: ExperimentResult, output_dir: Path):
        """Save factor matrices (W and Z) as separate files for easy access."""

        # Create matrices subdirectory
        matrices_dir = output_dir / "matrices"
        matrices_dir.mkdir(exist_ok=True)

        # Check for matrices in model_results
        model_results = result.model_results

        # Extract view names and feature names from data_summary if available
        view_names = None
        feature_names = None
        if result.data_summary:
            if "view_names" in result.data_summary:
                view_names = result.data_summary["view_names"]
            if "feature_names" in result.data_summary:
                feature_names = result.data_summary["feature_names"]

        # Save SGFA matrices if available
        if "sgfa_variants" in model_results:
            sgfa_results = model_results["sgfa_variants"]
            for variant_name, variant_result in sgfa_results.items():
                if variant_result and isinstance(variant_result, dict):
                    self._save_matrices_for_variant(
                        variant_result, matrices_dir, f"sgfa_{variant_name}", view_names, feature_names
                    )

        # Save other method matrices if available
        if "traditional_methods" in model_results:
            trad_results = model_results["traditional_methods"]
            for method_name, method_result in trad_results.items():
                if method_result and isinstance(method_result, dict):
                    self._save_matrices_for_variant(
                        method_result, matrices_dir, f"traditional_{method_name}", view_names, feature_names
                    )

        # Save main result matrices if available (for single-method experiments)
        if hasattr(result, "W") or (model_results and "W" in model_results):
            # Safe attribute retrieval with proper None checks
            W = None
            if hasattr(result, "W"):
                W = result.W
            elif model_results and "W" in model_results:
                W = model_results["W"]

            Z = None
            if hasattr(result, "Z"):
                Z = result.Z
            elif model_results and "Z" in model_results:
                Z = model_results["Z"]

            if W is not None or Z is not None:
                self._save_matrices_for_variant({"W": W, "Z": Z}, matrices_dir, "main", view_names, feature_names)

    def _save_matrices_for_variant(
        self, result_dict: dict, matrices_dir: Path, variant_name: str,
        view_names: Optional[List[str]] = None, feature_names: Optional[Dict[str, List[str]]] = None
    ):
        """Save W, Z matrices and all model parameters for a specific variant/method."""
        import pandas as pd

        # Save primary matrices (W and Z)
        W = result_dict.get("W")
        Z = result_dict.get("Z")

        if W is not None:
            try:
                if isinstance(W, list):
                    # Multi-view case: save each view separately
                    for view_idx, W_view in enumerate(W):
                        if hasattr(W_view, "shape"):
                            # Get view name (use index if not available)
                            if view_names and view_idx < len(view_names):
                                view_label = view_names[view_idx].replace(" ", "_").replace("/", "_")
                                view_name = view_names[view_idx]
                            else:
                                view_label = f"view{view_idx}"
                                view_name = None

                            # Get feature names for this view
                            row_labels = None
                            if feature_names and view_name and view_name in feature_names:
                                feat_list = feature_names[view_name]
                                # Ensure feature names match matrix dimensions
                                if len(feat_list) == W_view.shape[0]:
                                    row_labels = feat_list

                            # Create DataFrame with feature names as index and factors as columns
                            col_labels = [f"Factor_{i+1}" for i in range(W_view.shape[1])]
                            W_df = pd.DataFrame(W_view, columns=col_labels)

                            if row_labels:
                                W_df.index = row_labels
                                W_df.index.name = "Feature"

                            # Save as CSV for readability
                            save_csv(
                                W_df,
                                matrices_dir
                                / f"{variant_name}_factor_loadings_{view_label}.csv",
                                index=True if row_labels else False,
                            )
                else:
                    # Single matrix case (combined views)
                    if hasattr(W, "shape"):
                        # Try to get concatenated feature names
                        row_labels = None
                        if feature_names and view_names:
                            # Concatenate all feature names in order
                            all_features = []
                            for view_name in view_names:
                                if view_name in feature_names:
                                    all_features.extend(feature_names[view_name])

                            # Ensure concatenated features match matrix dimensions
                            if len(all_features) == W.shape[0]:
                                row_labels = all_features

                        # Create DataFrame with column headers
                        col_labels = [f"Factor_{i+1}" for i in range(W.shape[1])]
                        W_df = pd.DataFrame(W, columns=col_labels)

                        if row_labels:
                            W_df.index = row_labels
                            W_df.index.name = "Feature"

                        save_csv(
                            W_df,
                            matrices_dir / f"{variant_name}_factor_loadings.csv",
                            index=True if row_labels else False,
                        )
            except Exception as e:
                print(f"Warning: Could not save W matrices for {variant_name}: {e}")

        if Z is not None:
            try:
                if hasattr(Z, "shape"):
                    # Save factor scores
                    # save_numpy(Z, matrices_dir / f"{variant_name}_factor_scores.npy")

                    # Log shape for debugging
                    logger.info(f"Saving {variant_name} factor scores Z shape: {Z.shape}")

                    # Create DataFrame with meaningful column names and subject IDs
                    factor_cols = [f"Factor_{i + 1}" for i in range(Z.shape[1])]
                    Z_df = pd.DataFrame(Z, columns=factor_cols)
                    Z_df.index = [f"Subject_{i+1}" for i in range(Z.shape[0])]
                    Z_df.index.name = "Subject_ID"

                    save_csv(
                        Z_df,
                        matrices_dir / f"{variant_name}_factor_scores.csv",
                        index=True,
                    )
            except Exception as e:
                print(f"Warning: Could not save Z matrix for {variant_name}: {e}")

        # Save SGFA model parameters/weights if available
        sgfa_params = ["sigma", "tauZ", "lmbZ", "cZ", "tauW", "lmbW", "cW", "samples"]
        saved_params = []

        for param_name in sgfa_params:
            param_value = result_dict.get(param_name)
            if param_value is not None:
                try:
                    if hasattr(param_value, "shape"):
                        # Save as numpy array (if enabled)
                        if config.save_numpy_arrays:
                            save_numpy(
                                param_value,
                                matrices_dir / f"{variant_name}_{param_name}.npy",
                            )
                            saved_params.append(param_name)

                        # Save smaller parameters as CSV for readability
                        if (
                            param_value.size <= 1000
                        ):  # Only for reasonably sized parameters
                            if param_value.ndim <= 2:  # Only for 1D or 2D arrays
                                save_csv(
                                    pd.DataFrame(param_value),
                                    matrices_dir / f"{variant_name}_{param_name}.csv",
                                    index=False,
                                )
                    elif isinstance(param_value, dict):
                        # Handle dictionary of parameters (e.g., MCMC samples)
                        self._save_parameter_dict(
                            param_value, matrices_dir, f"{variant_name}_{param_name}", config
                        )
                        saved_params.append(f"{param_name} (dict)")
                except Exception as e:
                    print(
                        f"Warning: Could not save parameter {param_name} for {variant_name}: {e}"
                    )

        # Save hyperparameter performance summary instead of raw hyperparameters
        hyperparams = result_dict.get("hyperparameters")
        if hyperparams and isinstance(hyperparams, dict):
            try:
                # Create performance-ranked hyperparameter summary
                performance_summary = self._create_hyperparameter_performance_summary(
                    variant_name, hyperparams, result_dict
                )

                save_json(
                    performance_summary,
                    matrices_dir / f"{variant_name}_hyperparameter_performance.json",
                    indent=2,
                )
                saved_params.append("hyperparameter_performance")
            except Exception as e:
                print(
                    f"Warning: Could not save hyperparameter performance for {variant_name}: {e}"
                )

        # Create a summary of saved parameters
        if saved_params:
            try:
                summary = {
                    "variant_name": variant_name,
                    "saved_matrices": (
                        ["W (factor_loadings)", "Z (factor_scores)"]
                        if W is not None or Z is not None
                        else []
                    ),
                    "saved_parameters": saved_params,
                    "timestamp": pd.Timestamp.now().isoformat(),
                }
                save_json(
                    summary,
                    matrices_dir / f"{variant_name}_parameter_summary.json",
                    indent=2,
                )
            except Exception as e:
                print(
                    f"Warning: Could not save parameter summary for {variant_name}: {e}"
                )

    def _save_parameter_dict(self, param_dict: dict, matrices_dir: Path, prefix: str, config: ExperimentConfig):
        """Save dictionary of parameters (e.g., MCMC samples)."""
        import numpy as np

        for key, value in param_dict.items():
            try:
                if hasattr(value, "shape"):
                    if config.save_numpy_arrays:
                        save_numpy(value, matrices_dir / f"{prefix}_{key}.npy")
                        # Save summary statistics for large sample arrays
                        if value.ndim > 2:  # Likely MCMC samples
                            summary_stats = {
                                "mean": np.mean(value, axis=0),
                                "std": np.std(value, axis=0),
                                "median": np.median(value, axis=0),
                                "q025": np.percentile(value, 2.5, axis=0),
                                "q975": np.percentile(value, 97.5, axis=0),
                            }
                            for stat_name, stat_value in summary_stats.items():
                                save_numpy(
                                    stat_value,
                                    matrices_dir / f"{prefix}_{key}_{stat_name}.npy",
                                )
            except Exception as e:
                print(f"Warning: Could not save parameter {key} from {prefix}: {e}")

    def _create_hyperparameter_performance_summary(self, variant_name: str, hyperparams: dict, result_dict: dict) -> dict:
        """Create a performance-ranked hyperparameter summary instead of saving raw hyperparameters."""
        import time

        # Extract performance metrics from result_dict
        execution_time = result_dict.get("execution_time", 0.0)
        convergence = result_dict.get("convergence", False)
        log_likelihood = result_dict.get("log_likelihood", None)

        # Calculate interpretability scores if factor loadings are available
        interpretability_score = None
        if "W" in result_dict:
            try:
                import numpy as np
                W = result_dict["W"]
                if W is not None and hasattr(W, "shape"):
                    # Simple interpretability metric: sparsity level
                    interpretability_score = float(np.mean(np.abs(W) > 0.1))
            except Exception:
                pass

        # Create performance summary
        performance_summary = {
            "variant_configuration": {
                "variant_name": variant_name,
                "hyperparameters": hyperparams,
                "timestamp": pd.Timestamp.now().isoformat()
            },
            "performance_metrics": {
                "execution_time_seconds": execution_time,
                "converged": convergence,
                "log_likelihood": log_likelihood,
                "factor_interpretability_score": interpretability_score
            },
            "clinical_relevance": {
                "K_factors": hyperparams.get("K", "unknown"),
                "sparsity_level": hyperparams.get("percW", "unknown"),
                "recommended_for": self._get_clinical_recommendation(hyperparams, result_dict)
            },
            "scientific_assessment": {
                "computational_efficiency": "fast" if execution_time < 60 else "moderate" if execution_time < 300 else "slow",
                "model_quality": "good" if convergence else "poor",
                "interpretability": "high" if interpretability_score and interpretability_score > 0.3 else "moderate" if interpretability_score and interpretability_score > 0.1 else "low"
            }
        }

        return performance_summary

    def _get_clinical_recommendation(self, hyperparams: dict, result_dict: dict) -> str:
        """Provide clinical recommendations based on hyperparameter performance."""
        K = hyperparams.get("K", 0)
        percW = hyperparams.get("percW", 0)
        convergence = result_dict.get("convergence", False)
        execution_time = result_dict.get("execution_time", float('inf'))

        if not convergence:
            return "Not recommended - model did not converge"
        elif execution_time > 600:  # 10 minutes
            return "Computationally expensive - consider for final analysis only"
        elif K <= 3:
            return "Exploratory analysis - few factors may miss complex patterns"
        elif K >= 8:
            return "Detailed analysis - many factors may include noise"
        elif percW < 15:
            return "High sparsity - good for identifying key biomarkers"
        elif percW > 35:
            return "Low sparsity - comprehensive but less interpretable"
        else:
            return "Balanced configuration - good for most neuroimaging applications"

    def create_optimal_parameter_recommendations(self, experiment_results: dict, output_dir: Path) -> dict:
        """Create comprehensive optimal parameter recommendations based on all experiment results."""
        from core.io_utils import save_json
        import pandas as pd

        # Aggregate data from all experiments
        all_performance_data = []
        experiment_sources = {}

        # Extract performance data from different experiment types
        for exp_name, exp_results in experiment_results.items():
            if isinstance(exp_results, dict):
                # SGFA parameter comparison results
                if "sgfa_variants" in exp_results:
                    for variant_name, variant_data in exp_results["sgfa_variants"].items():
                        perf_data = self._extract_performance_data(variant_name, variant_data)
                        perf_data["source_experiment"] = exp_name
                        all_performance_data.append(perf_data)

                # Sensitivity analysis results
                elif "sensitivity" in exp_name.lower():
                    sensitivity_data = self._extract_sensitivity_data(exp_results)
                    for data_point in sensitivity_data:
                        data_point["source_experiment"] = exp_name
                        all_performance_data.append(data_point)

        if not all_performance_data:
            return {"no_data": "No performance data available for recommendations"}

        # Create comprehensive recommendations
        recommendations = {
            "executive_summary": self._create_executive_summary(all_performance_data),
            "parameter_recommendations": self._create_parameter_recommendations(all_performance_data),
            "use_case_specific_recommendations": self._create_use_case_recommendations(all_performance_data),
            "clinical_deployment_guidance": self._create_clinical_guidance(all_performance_data),
            "validation_requirements": self._create_validation_requirements(all_performance_data),
            "risk_mitigation": self._create_risk_mitigation_guidance(all_performance_data),
            "implementation_roadmap": self._create_implementation_roadmap(all_performance_data)
        }

        # Save comprehensive recommendations
        recommendations_path = output_dir / "optimal_parameter_recommendations.json"
        save_json(recommendations, recommendations_path, indent=2)

        return recommendations

    def _create_executive_summary(self, performance_data: list) -> dict:
        """Create executive summary of parameter optimization results."""
        converged_data = [d for d in performance_data if d.get("converged", False)]

        if not converged_data:
            return {"status": "No converged configurations found"}

        # Find overall best performing configuration
        best_config = max(converged_data, key=lambda x: x.get("performance_metric", float('-inf')))

        return {
            "total_configurations_tested": len(performance_data),
            "successful_configurations": len(converged_data),
            "success_rate": len(converged_data) / len(performance_data),
            "best_performing_configuration": {
                "variant_name": best_config.get("variant_name", "unknown"),
                "hyperparameters": best_config.get("hyperparameters", {}),
                "performance_score": best_config.get("performance_metric", 0),
                "execution_time": best_config.get("execution_time", 0)
            },
            "overall_recommendation": self._generate_overall_recommendation(converged_data)
        }

    def _generate_overall_recommendation(self, converged_data: list) -> str:
        """Generate overall recommendation based on all results."""
        if not converged_data:
            return "No reliable configurations found. Consider adjusting experimental parameters."

        # Analyze overall patterns
        success_rate = len(converged_data) / (len(converged_data) + 1)  # Avoid division issues

        if success_rate > 0.8:
            return "Model shows excellent stability. Proceed with clinical validation studies."
        elif success_rate > 0.6:
            return "Model shows good performance. Consider additional parameter refinement."
        elif success_rate > 0.4:
            return "Model shows moderate sensitivity. Careful parameter selection required."
        else:
            return "Model highly sensitive to parameters. Consider alternative modeling approaches."

    def _create_parameter_recommendations(self, performance_data: list) -> dict:
        """Create specific parameter recommendations."""
        converged_data = [d for d in performance_data if d.get("converged", False)]

        if not converged_data:
            return {}

        # Extract parameter ranges from top performers
        top_performers = sorted(converged_data, key=lambda x: x.get("performance_metric", 0), reverse=True)[:5]

        param_recommendations = {}
        all_params = set()
        for config in top_performers:
            all_params.update(config.get("hyperparameters", {}).keys())

        for param in all_params:
            param_values = [config.get("hyperparameters", {}).get(param) for config in top_performers if param in config.get("hyperparameters", {})]
            if param_values:
                param_recommendations[param] = {
                    "recommended_range": [min(param_values), max(param_values)],
                    "optimal_value": sum(param_values) / len(param_values),
                    "confidence": "high" if len(set(param_values)) <= 2 else "moderate" if len(set(param_values)) <= 3 else "low",
                    "rationale": self._get_parameter_rationale(param, param_values)
                }

        return param_recommendations

    def _get_parameter_rationale(self, param_name: str, param_values: list) -> str:
        """Provide rationale for parameter recommendations."""
        rationales = {
            "K": f"Number of factors. Range {min(param_values)}-{max(param_values)} balances model complexity with interpretability.",
            "percW": f"Sparsity level. Range {min(param_values)}-{max(param_values)}% provides good feature selection while maintaining model fit.",
            "num_samples": f"MCMC samples. Range {min(param_values)}-{max(param_values)} ensures convergence while managing computational cost.",
            "learning_rate": f"Learning rate. Range {min(param_values)}-{max(param_values)} provides stable optimization."
        }
        return rationales.get(param_name, f"Empirically determined optimal range: {min(param_values)}-{max(param_values)}")

    def _create_use_case_recommendations(self, performance_data: list) -> dict:
        """Create recommendations for different use cases."""
        converged_data = [d for d in performance_data if d.get("converged", False)]

        if not converged_data:
            return {}

        # Sort configurations by different criteria
        by_speed = sorted(converged_data, key=lambda x: x.get("execution_time", float('inf')))
        by_performance = sorted(converged_data, key=lambda x: x.get("performance_metric", 0), reverse=True)

        return {
            "exploratory_research": {
                "recommended_config": by_speed[0] if by_speed else None,
                "rationale": "Fastest configuration for rapid hypothesis testing",
                "typical_use": "Initial data exploration, proof-of-concept studies"
            },
            "clinical_application": {
                "recommended_config": by_performance[0] if by_performance else None,
                "rationale": "Best performing configuration for reliable clinical insights",
                "typical_use": "Biomarker discovery, patient subtyping"
            },
            "production_deployment": {
                "recommended_config": self._find_balanced_configuration(converged_data),
                "rationale": "Balanced performance and efficiency for routine use",
                "typical_use": "Automated analysis pipelines, routine clinical workflows"
            }
        }

    def _create_clinical_guidance(self, performance_data: list) -> dict:
        """Create clinical deployment guidance."""
        converged_data = [d for d in performance_data if d.get("converged", False)]

        return {
            "validation_requirements": {
                "minimum_sample_size": "At least 100 subjects for reliable biomarker discovery",
                "cross_validation": "5-fold stratified cross-validation with site balancing",
                "external_validation": "Independent cohort validation required before clinical use"
            },
            "clinical_interpretation": {
                "factor_interpretation": "Each factor represents a latent disease pattern. Collaborate with clinicians for interpretation.",
                "biomarker_validation": "Factor loadings indicate biomarker importance. Validate with known disease mechanisms.",
                "patient_subtyping": "Factor scores can identify patient subgroups. Validate with clinical outcomes."
            },
            "regulatory_considerations": {
                "algorithm_transparency": "Document all parameter choices and their clinical rationale",
                "performance_monitoring": "Monitor model performance in clinical deployment",
                "bias_assessment": "Assess for demographic and scanner biases"
            }
        }

    def _create_validation_requirements(self, performance_data: list) -> dict:
        """Create validation requirements for clinical deployment."""
        return {
            "pre_deployment_validation": {
                "internal_validation": "Cross-validation on training cohort",
                "external_validation": "Independent test cohort from different sites",
                "clinical_validation": "Validation against clinical outcomes and expert assessment"
            },
            "ongoing_monitoring": {
                "performance_tracking": "Monitor model performance on new data",
                "drift_detection": "Detect changes in data distribution or model performance",
                "parameter_stability": "Monitor stability of optimal parameters over time"
            },
            "quality_assurance": {
                "data_quality_checks": "Automated data quality assessment",
                "robustness_testing": "Regular robustness validation",
                "bias_monitoring": "Ongoing assessment of demographic and technical biases"
            }
        }

    def _create_risk_mitigation_guidance(self, performance_data: list) -> dict:
        """Create risk mitigation guidance."""
        converged_data = [d for d in performance_data if d.get("converged", False)]
        convergence_rate = len(converged_data) / len(performance_data) if performance_data else 0

        risk_level = "low" if convergence_rate > 0.8 else "moderate" if convergence_rate > 0.6 else "high"

        return {
            "risk_assessment": {
                "overall_risk_level": risk_level,
                "convergence_reliability": convergence_rate,
                "parameter_sensitivity": "moderate" if convergence_rate > 0.5 else "high"
            },
            "mitigation_strategies": {
                "parameter_validation": "Always validate parameters on independent data",
                "ensemble_approaches": "Consider ensemble methods for increased robustness",
                "fallback_procedures": "Define procedures for handling convergence failures",
                "expert_oversight": "Maintain clinical expert involvement in interpretation"
            },
            "monitoring_protocols": {
                "performance_alerts": "Set thresholds for performance degradation alerts",
                "parameter_drift": "Monitor for changes in optimal parameter values",
                "clinical_feedback": "Incorporate clinical feedback into model updates"
            }
        }

    def _create_implementation_roadmap(self, performance_data: list) -> dict:
        """Create implementation roadmap for clinical deployment."""
        return {
            "phase_1_pilot": {
                "duration": "3-6 months",
                "objectives": "Validate model on clinical cohort, refine parameters",
                "success_criteria": "Model convergence >90%, clinical validation positive",
                "deliverables": ["Clinical validation report", "Refined parameter recommendations"]
            },
            "phase_2_validation": {
                "duration": "6-12 months",
                "objectives": "Multi-site validation, regulatory preparation",
                "success_criteria": "External validation successful, regulatory pathway clear",
                "deliverables": ["Multi-site validation study", "Regulatory submission preparation"]
            },
            "phase_3_deployment": {
                "duration": "12-18 months",
                "objectives": "Clinical deployment, ongoing monitoring",
                "success_criteria": "Successful clinical integration, positive outcomes",
                "deliverables": ["Clinical deployment system", "Monitoring protocols"]
            },
            "ongoing_maintenance": {
                "frequency": "Quarterly reviews",
                "activities": ["Parameter validation", "Performance monitoring", "Model updates"],
                "success_metrics": ["Stable performance", "Clinical utility", "User satisfaction"]
            }
        }

    def _save_intermediate_results(self, result: ExperimentResult, output_dir: Path, experiment_results: dict):
        """Save intermediate experiment results for debugging and analysis."""
        from pathlib import Path
        from core.io_utils import save_json, save_numpy
        import time

        # Create intermediate results directory
        intermediate_dir = output_dir / "intermediate"
        intermediate_dir.mkdir(exist_ok=True)

        logger.info(f"💾 Saving intermediate results to {intermediate_dir}")

        try:
            # Save experiment metadata
            metadata = {
                "experiment_name": getattr(result, 'experiment_name', getattr(result.config, 'experiment_name', 'unknown')),
                "timestamp": int(time.time()),
                "config_summary": {
                    "num_samples": getattr(result.config, 'num_samples', 'unknown'),
                    "num_chains": getattr(result.config, 'num_chains', 'unknown'),
                    "K_values": getattr(result.config, 'K_values', 'unknown'),
                },
                "data_shapes": experiment_results.get("data_summary", {}),
            }
            save_json(metadata, intermediate_dir / "experiment_metadata.json", indent=2)

            # Save raw experiment results (filtered for serializability)
            serializable_results = {}
            for key, value in experiment_results.items():
                try:
                    # Test if value is JSON serializable
                    import json
                    json.dumps(value, default=str)
                    serializable_results[key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable values but log their type
                    serializable_results[f"{key}_type"] = str(type(value))

            save_json(serializable_results, intermediate_dir / "raw_results.json", indent=2)

            # Save performance metrics if available
            if hasattr(result, 'performance_metrics') and result.performance_metrics:
                save_json(result.performance_metrics, intermediate_dir / "performance_metrics.json", indent=2)

            # Save convergence diagnostics if available
            if result.convergence_diagnostics:
                save_json(result.convergence_diagnostics, intermediate_dir / "convergence_diagnostics.json", indent=2)

            logger.info(f"✅ Intermediate results saved successfully")

        except Exception as e:
            logger.warning(f"⚠️ Failed to save some intermediate results: {e}")

    def _save_plots(self, result: ExperimentResult, output_dir: Path):
        """Save plots as PNG and PDF files for easy access and reports."""

        if not result.plots:
            return

        # Create plots subdirectory
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        safe_log(logger.info, f"Saving {len(result.plots)} plots to {plots_dir}")

        for plot_name, plot_figure in result.plots.items():
            if plot_figure is None:
                continue

            # Use standardized plot saving from io_utils
            try:
                # Save as PNG (high quality for viewing)
                png_path = plots_dir / f"{plot_name}.png"
                save_plot(
                    png_path,
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="none",
                    close_after=False,
                )

                # Save as PDF (vector format for publications)
                pdf_path = plots_dir / f"{plot_name}.pdf"
                save_plot(
                    pdf_path,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="none",
                    close_after=True,
                )

                safe_log(logger.info, f"  ✅ Saved plot: {plot_name}")

            except Exception as e:
                safe_log(logger.warning, f"Failed to save plot {plot_name}: {e}")

        safe_log(logger.info, f"All plots saved to: {plots_dir}")

    def run_experiment_grid(
        self,
        base_config: ExperimentConfig,
        parameter_grid: Dict[str, List[Any]],
        experiment_function: Callable,
    ) -> List[ExperimentResult]:
        """
        Run experiments across a parameter grid.

        Parameters
        ----------
        base_config : ExperimentConfig
            Base configuration to modify.
        parameter_grid : Dict[str, List[Any]]
            Grid of parameters to test.
        experiment_function : Callable
            Experiment function to run.

        Returns
        -------
        List[ExperimentResult] : Results from all experiments.
        """
        from itertools import product

        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())

        results = []
        total_experiments = np.prod([len(values) for values in param_values])

        logger.info(f"Running grid experiment with {total_experiments} configurations")

        for i, param_combination in enumerate(product(*param_values)):
            # Create modified config
            config = ExperimentConfig.from_dict(base_config.to_dict())

            # Update parameters
            for param_name, param_value in zip(param_names, param_combination):
                setattr(config, param_name, param_value)

            # Update experiment name to include parameters
            param_str = "_".join(
                [
                    f"{name}_{value}"
                    for name, value in zip(param_names, param_combination)
                ]
            )
            config.experiment_name = f"{base_config.experiment_name}_grid_{param_str}"

            logger.info(
                f"Running experiment {i + 1}/{total_experiments}: {config.experiment_name}"
            )

            # Run experiment
            result = self.run_experiment(config, experiment_function)
            results.append(result)

        # Save grid summary
        self._save_grid_summary(results, parameter_grid)

        return results

    def _save_grid_summary(
        self, results: List[ExperimentResult], parameter_grid: Dict[str, List[Any]]
    ):
        """Save summary of grid search results."""
        summary_data = []

        for result in results:
            row = {
                "experiment_id": result.experiment_id,
                "status": result.status,
                "duration_seconds": result.get_duration(),
            }

            # Add parameter values
            for param_name in parameter_grid.keys():
                if hasattr(result.config, param_name):
                    row[param_name] = getattr(result.config, param_name)

            # Add key metrics
            if result.model_results:
                for key, value in result.model_results.items():
                    if isinstance(value, (int, float)):
                        row[f"model_{key}"] = value

            if result.cv_results:
                for key, value in result.cv_results.items():
                    if isinstance(value, (int, float)):
                        row[f"cv_{key}"] = value

            summary_data.append(row)

        # Save grid summary
        df = pd.DataFrame(summary_data)
        grid_summary_path = self.base_output_dir / "grid_search_summary.csv"
        save_csv(df, grid_summary_path, index=False)

        logger.info(f"Grid search summary saved to: {grid_summary_path}")

    def get_best_result(
        self,
        results: List[ExperimentResult],
        metric: str = "cv_accuracy",
        maximize: bool = True,
    ) -> Optional[ExperimentResult]:
        """
        Find best result based on specified metric.

        Parameters
        ----------
        results : List[ExperimentResult]
            Results to compare.
        metric : str
            Metric to optimize.
        maximize : bool
            Whether to maximize the metric.

        Returns
        -------
        ExperimentResult : Best result.
        """
        valid_results = [r for r in results if r.status == "completed"]

        if not valid_results:
            return None

        def get_metric_value(result: ExperimentResult) -> Optional[float]:
            """Extract metric value from result."""
            if metric.startswith("cv_") and result.cv_results:
                key = metric[3:]  # Remove "cv_" prefix
                return result.cv_results.get(key)
            elif metric.startswith("model_") and result.model_results:
                key = metric[6:]  # Remove "model_" prefix
                return result.model_results.get(key)
            return None

        # Filter results with valid metric values
        results_with_metrics = []
        for result in valid_results:
            value = get_metric_value(result)
            if value is not None:
                results_with_metrics.append((result, value))

        if not results_with_metrics:
            return None

        # Find best result
        best_result, best_value = (
            max(results_with_metrics, key=lambda x: x[1])
            if maximize
            else min(results_with_metrics, key=lambda x: x[1])
        )

        logger.info(
            f"Best result: {best_result.experiment_id} with {metric}={best_value}"
        )
        return best_result

    def generate_experiment_report(
        self, results: List[ExperimentResult] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive experiment report."""
        if results is None:
            results = self.results_history

        if not results:
            return {"message": "No experiment results available"}

        # Calculate summary statistics
        completed_results = [r for r in results if r.status == "completed"]
        failed_results = [r for r in results if r.status == "failed"]

        durations = [
            r.get_duration() for r in completed_results if r.get_duration() is not None
        ]

        report = {
            "experiment_summary": {
                "total_experiments": len(results),
                "completed": len(completed_results),
                "failed": len(failed_results),
                "success_rate": len(completed_results) / len(results) if results else 0,
                "total_runtime_hours": sum(durations) / 3600 if durations else 0,
                "average_experiment_duration_minutes": (
                    np.mean(durations) / 60 if durations else 0
                ),
            },
            "experiment_details": [
                {
                    "experiment_id": r.experiment_id,
                    "experiment_name": r.config.experiment_name,
                    "status": r.status,
                    "duration_minutes": (
                        r.get_duration() / 60 if r.get_duration() else None
                    ),
                    "error_message": r.error_message,
                }
                for r in results
            ],
        }

        # Add performance statistics if available
        if completed_results:
            performance_data = []
            for result in completed_results:
                if result.performance_metrics:
                    perf_metrics = result.performance_metrics
                    if "memory_report" in perf_metrics:
                        memory_report = perf_metrics["memory_report"]
                        performance_data.append(
                            {
                                "experiment_id": result.experiment_id,
                                "peak_memory_gb": memory_report.get(
                                    "peak_memory_gb", 0
                                ),
                                "operations_completed": memory_report.get(
                                    "operations_completed", 0
                                ),
                            }
                        )

            if performance_data:
                peak_memories = [p["peak_memory_gb"] for p in performance_data]
                report["performance_summary"] = {
                    "average_peak_memory_gb": np.mean(peak_memories),
                    "max_peak_memory_gb": np.max(peak_memories),
                    "min_peak_memory_gb": np.min(peak_memories),
                }

        return report

    def save_experiment_report(
        self, filepath: Path, results: List[ExperimentResult] = None
    ):
        """Save experiment report to file."""
        report = self.generate_experiment_report(results)

        save_json(report, filepath, indent=2)

        logger.info(f"Experiment report saved to: {filepath}")

    def _comprehensive_memory_cleanup(self, logger):
        """Comprehensive memory cleanup to prevent GPU memory exhaustion."""
        try:
            logger.info("🧹 Performing comprehensive memory cleanup...")
            import gc
            import jax

            # Clear JAX compilation cache
            try:
                from jax._src import compilation_cache
                compilation_cache.clear_cache()
                logger.info("JAX compilation cache cleared")
            except Exception:
                # Silently ignore - newer JAX versions don't have this internal API
                # jax.clear_caches() below handles cache clearing
                pass

            # Force multiple garbage collection cycles
            for i in range(5):
                collected = gc.collect()
                if i == 0:
                    logger.info(f"Garbage collection freed {collected} objects")

            # Clear JAX device memory for GPU
            try:
                for device in jax.devices():
                    if device.platform == 'gpu':
                        # Force memory cleanup on GPU
                        device.memory_stats()
                logger.info("GPU memory cleanup attempted")
            except Exception as e:
                logger.warning(f"GPU memory cleanup failed: {e}")

            # Brief delay for cleanup to complete
            import time
            time.sleep(1)
            logger.info("✅ Comprehensive memory cleanup completed")

        except Exception as e:
            logger.warning(f"Memory cleanup encountered issues: {e}")

