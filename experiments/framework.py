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
    percW_values: List[float] = field(default_factory=lambda: [25.0, 33.0, 50.0])

    # MCMC configuration
    num_samples: int = 2000
    num_chains: int = 4
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
    save_intermediate_results: bool = True
    save_samples: bool = False  # Can be large
    save_diagnostics: bool = True
    output_formats: List[str] = field(default_factory=lambda: ["json", "csv"])

    # Reproducibility
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

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

    experiment_id: str
    config: ExperimentConfig
    start_time: datetime
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

    def mark_completed(self):
        """Mark experiment as completed."""
        self.end_time = datetime.now()
        self.status = "completed"

    def mark_failed(self, error_message: str):
        """Mark experiment as failed."""
        self.end_time = datetime.now()
        self.status = "failed"
        self.error_message = error_message

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
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)

    def create_experiment_dir(self, experiment_name: str) -> Path:
        """Create directory for specific experiment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.base_output_dir / f"{experiment_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def run_experiment(
        self, config: ExperimentConfig, experiment_function: Callable, **kwargs
    ) -> ExperimentResult:
        """
        Run a single experiment.

        Parameters
        ----------
        config : ExperimentConfig
            Experiment configuration.
        experiment_function : Callable
            Function that runs the actual experiment.
        **kwargs
            Additional arguments for experiment function.

        Returns
        -------
        ExperimentResult : Results of the experiment.
        """
        # Create experiment directory
        exp_dir = self.create_experiment_dir(config.experiment_name)

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

        # Save as pickle for complete object
        result_pkl = output_dir / "result.pkl"
        safe_pickle_save(result, result_pkl, "Experiment result")

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

        # Save SGFA matrices if available
        if "sgfa_variants" in model_results:
            sgfa_results = model_results["sgfa_variants"]
            for variant_name, variant_result in sgfa_results.items():
                if variant_result and isinstance(variant_result, dict):
                    self._save_matrices_for_variant(
                        variant_result, matrices_dir, f"sgfa_{variant_name}"
                    )

        # Save other method matrices if available
        if "traditional_methods" in model_results:
            trad_results = model_results["traditional_methods"]
            for method_name, method_result in trad_results.items():
                if method_result and isinstance(method_result, dict):
                    self._save_matrices_for_variant(
                        method_result, matrices_dir, f"traditional_{method_name}"
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
                self._save_matrices_for_variant({"W": W, "Z": Z}, matrices_dir, "main")

    def _save_matrices_for_variant(
        self, result_dict: dict, matrices_dir: Path, variant_name: str
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
                            # Save as numpy
                            save_numpy(
                                W_view,
                                matrices_dir
                                / f"{variant_name}_factor_loadings_view{view_idx}.npy",
                            )
                            # Save as CSV for readability
                            save_csv(
                                pd.DataFrame(W_view),
                                matrices_dir
                                / f"{variant_name}_factor_loadings_view{view_idx}.csv",
                                index=False,
                            )
                else:
                    # Single matrix case
                    if hasattr(W, "shape"):
                        save_numpy(
                            W, matrices_dir / f"{variant_name}_factor_loadings.npy"
                        )
                        save_csv(
                            pd.DataFrame(W),
                            matrices_dir / f"{variant_name}_factor_loadings.csv",
                            index=False,
                        )
            except Exception as e:
                print(f"Warning: Could not save W matrices for {variant_name}: {e}")

        if Z is not None:
            try:
                if hasattr(Z, "shape"):
                    # Save factor scores
                    save_numpy(Z, matrices_dir / f"{variant_name}_factor_scores.npy")
                    # Create DataFrame with meaningful column names
                    factor_cols = [f"Factor_{i + 1}" for i in range(Z.shape[1])]
                    save_csv(
                        pd.DataFrame(Z, columns=factor_cols),
                        matrices_dir / f"{variant_name}_factor_scores.csv",
                        index=False,
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
                        # Save as numpy array
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
                            param_value, matrices_dir, f"{variant_name}_{param_name}"
                        )
                        saved_params.append(f"{param_name} (dict)")
                except Exception as e:
                    print(
                        f"Warning: Could not save parameter {param_name} for {variant_name}: {e}"
                    )

        # Save hyperparameters if available
        hyperparams = result_dict.get("hyperparameters")
        if hyperparams and isinstance(hyperparams, dict):
            try:
                # Save as JSON for easy reading
                pass

                save_json(
                    hyperparams,
                    matrices_dir / f"{variant_name}_hyperparameters.json",
                    indent=2,
                )
                saved_params.append("hyperparameters")
            except Exception as e:
                print(
                    f"Warning: Could not save hyperparameters for {variant_name}: {e}"
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

    def _save_parameter_dict(self, param_dict: dict, matrices_dir: Path, prefix: str):
        """Save dictionary of parameters (e.g., MCMC samples)."""
        import numpy as np

        for key, value in param_dict.items():
            try:
                if hasattr(value, "shape"):
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

        logger.info(f"Saving {len(result.plots)} plots to {plots_dir}")

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

                logger.info(f"  ✅ Saved plot: {plot_name}")

            except Exception as e:
                logger.warning(f"Failed to save plot {plot_name}: {e}")

        logger.info(f"All plots saved to: {plots_dir}")

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
            except Exception as e:
                logger.warning(f"Could not clear JAX cache: {e}")

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

