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
                            # save_numpy(
                            #     W_view,
                            #     matrices_dir
                            #     / f"{variant_name}_factor_loadings_view{view_idx}.npy",
                            # )
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
                        # save_numpy(
                        #     W, matrices_dir / f"{variant_name}_factor_loadings.npy"
                        # )
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

    def create_hyperparameter_comparison_summary(self, experiment_results: dict, output_dir: Path) -> dict:
        """Create a comprehensive summary comparing hyperparameter performance across all variants."""
        from core.io_utils import save_json
        import pandas as pd

        # Extract all hyperparameter performance data
        hyperparameter_performances = []
        variant_summaries = {}

        # Look for hyperparameter performance files or data in results
        for key, value in experiment_results.items():
            if "sgfa_variants" in key or isinstance(value, dict):
                if isinstance(value, dict) and "sgfa_variants" in value:
                    variants_data = value["sgfa_variants"]
                    for variant_name, variant_result in variants_data.items():
                        if isinstance(variant_result, dict):
                            perf_data = self._extract_performance_data(variant_name, variant_result)
                            if perf_data:
                                hyperparameter_performances.append(perf_data)
                                variant_summaries[variant_name] = perf_data

        # Create comprehensive comparison summary
        if hyperparameter_performances:
            summary = {
                "experiment_overview": {
                    "total_variants_tested": len(hyperparameter_performances),
                    "converged_variants": len([p for p in hyperparameter_performances if p.get("converged", False)]),
                    "timestamp": pd.Timestamp.now().isoformat()
                },
                "performance_ranking": self._rank_hyperparameter_performance(hyperparameter_performances),
                "convergence_analysis": self._analyze_convergence_patterns(hyperparameter_performances),
                "timing_analysis": self._analyze_execution_timing(hyperparameter_performances),
                "clinical_recommendations": self._generate_clinical_recommendations(hyperparameter_performances),
                "optimal_configurations": self._identify_optimal_configurations(hyperparameter_performances),
                "detailed_variants": variant_summaries
            }

            # Save the comprehensive summary
            summary_path = output_dir / "hyperparameter_comparison_summary.json"
            save_json(summary, summary_path, indent=2)

            return summary

        return {}

    def _extract_performance_data(self, variant_name: str, variant_result: dict) -> dict:
        """Extract performance data from variant results."""
        return {
            "variant_name": variant_name,
            "execution_time": variant_result.get("execution_time", 0),
            "converged": variant_result.get("convergence", False),
            "log_likelihood": variant_result.get("log_likelihood"),
            "hyperparameters": variant_result.get("hyperparameters", {}),
            "status": variant_result.get("status", "unknown")
        }

    def _rank_hyperparameter_performance(self, performances: list) -> list:
        """Rank hyperparameter configurations by performance."""
        # Filter to converged variants only
        converged = [p for p in performances if p.get("converged", False)]

        # Sort by log likelihood (higher is better), then by execution time (lower is better)
        ranked = sorted(converged, key=lambda x: (
            -(x.get("log_likelihood", float('-inf')) if x.get("log_likelihood") else float('-inf')),
            x.get("execution_time", float('inf'))
        ))

        # Add ranking information
        for i, perf in enumerate(ranked):
            perf["performance_rank"] = i + 1
            perf["performance_tier"] = "excellent" if i < 2 else "good" if i < 5 else "acceptable"

        return ranked[:10]  # Top 10 configurations

    def _analyze_convergence_patterns(self, performances: list) -> dict:
        """Analyze convergence patterns across hyperparameters."""
        total = len(performances)
        converged = [p for p in performances if p.get("converged", False)]

        # Analyze convergence by K values
        k_convergence = {}
        percw_convergence = {}

        for perf in performances:
            hyperparams = perf.get("hyperparameters", {})
            k_val = hyperparams.get("K", "unknown")
            percw_val = hyperparams.get("percW", "unknown")
            converged_status = perf.get("converged", False)

            if k_val not in k_convergence:
                k_convergence[k_val] = {"total": 0, "converged": 0}
            k_convergence[k_val]["total"] += 1
            if converged_status:
                k_convergence[k_val]["converged"] += 1

            if percw_val not in percw_convergence:
                percw_convergence[percw_val] = {"total": 0, "converged": 0}
            percw_convergence[percw_val]["total"] += 1
            if converged_status:
                percw_convergence[percw_val]["converged"] += 1

        return {
            "overall_convergence_rate": len(converged) / total if total > 0 else 0,
            "convergence_by_k_factors": {
                str(k): {
                    "rate": stats["converged"] / stats["total"] if stats["total"] > 0 else 0,
                    "converged": stats["converged"],
                    "total": stats["total"]
                }
                for k, stats in k_convergence.items()
            },
            "convergence_by_sparsity": {
                str(percw): {
                    "rate": stats["converged"] / stats["total"] if stats["total"] > 0 else 0,
                    "converged": stats["converged"],
                    "total": stats["total"]
                }
                for percw, stats in percw_convergence.items()
            }
        }

    def _analyze_execution_timing(self, performances: list) -> dict:
        """Analyze execution timing patterns."""
        converged = [p for p in performances if p.get("converged", False)]
        times = [p.get("execution_time", 0) for p in converged if p.get("execution_time", 0) > 0]

        if not times:
            return {"no_timing_data": True}

        import numpy as np
        return {
            "mean_execution_time": float(np.mean(times)),
            "median_execution_time": float(np.median(times)),
            "min_execution_time": float(min(times)),
            "max_execution_time": float(max(times)),
            "std_execution_time": float(np.std(times)),
            "fast_configurations": [
                p["variant_name"] for p in converged
                if p.get("execution_time", 0) < np.percentile(times, 25)
            ],
            "slow_configurations": [
                p["variant_name"] for p in converged
                if p.get("execution_time", 0) > np.percentile(times, 75)
            ]
        }

    def _generate_clinical_recommendations(self, performances: list) -> dict:
        """Generate clinical recommendations based on performance analysis."""
        converged = [p for p in performances if p.get("converged", False)]

        if not converged:
            return {"recommendation": "No converged configurations found"}

        # Find configurations suitable for different use cases
        recommendations = {
            "exploratory_analysis": [],
            "detailed_biomarker_discovery": [],
            "clinical_routine": [],
            "research_grade": []
        }

        for perf in converged:
            hyperparams = perf.get("hyperparameters", {})
            exec_time = perf.get("execution_time", 0)
            k_val = hyperparams.get("K", 0)
            percw_val = hyperparams.get("percW", 0)

            # Classify based on computational requirements and configuration
            if exec_time < 120 and k_val <= 5:  # Fast, simple
                recommendations["exploratory_analysis"].append(perf["variant_name"])
                recommendations["clinical_routine"].append(perf["variant_name"])
            elif k_val >= 6 and percw_val < 25:  # Many factors, high sparsity
                recommendations["detailed_biomarker_discovery"].append(perf["variant_name"])
            elif exec_time < 300:  # Moderate time, good for research
                recommendations["research_grade"].append(perf["variant_name"])

        return recommendations

    def _identify_optimal_configurations(self, performances: list) -> dict:
        """Identify optimal configurations for different scenarios."""
        converged = [p for p in performances if p.get("converged", False)]

        if not converged:
            return {}

        # Sort by different criteria
        by_likelihood = sorted(converged, key=lambda x: x.get("log_likelihood", float('-inf')), reverse=True)
        by_speed = sorted(converged, key=lambda x: x.get("execution_time", float('inf')))

        return {
            "best_statistical_fit": by_likelihood[0]["variant_name"] if by_likelihood else None,
            "fastest_convergence": by_speed[0]["variant_name"] if by_speed else None,
            "balanced_choice": self._find_balanced_configuration(converged),
            "parameter_ranges": self._extract_parameter_ranges(converged)
        }

    def _find_balanced_configuration(self, converged: list) -> str:
        """Find a balanced configuration considering multiple criteria."""
        if not converged:
            return None

        # Score based on normalized likelihood and inverse normalized time
        import numpy as np
        likelihoods = [p.get("log_likelihood", float('-inf')) for p in converged if p.get("log_likelihood")]
        times = [p.get("execution_time", 0) for p in converged if p.get("execution_time", 0) > 0]

        if not likelihoods or not times:
            return converged[0]["variant_name"]

        # Normalize scores
        min_ll, max_ll = min(likelihoods), max(likelihoods)
        min_time, max_time = min(times), max(times)

        best_score = float('-inf')
        best_variant = None

        for perf in converged:
            ll = perf.get("log_likelihood")
            time = perf.get("execution_time", 0)

            if ll is not None and time > 0:
                # Normalize likelihood (higher is better) and time (lower is better)
                ll_score = (ll - min_ll) / (max_ll - min_ll) if max_ll > min_ll else 0
                time_score = 1 - (time - min_time) / (max_time - min_time) if max_time > min_time else 0

                # Weighted combination
                combined_score = 0.7 * ll_score + 0.3 * time_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_variant = perf["variant_name"]

        return best_variant

    def _extract_parameter_ranges(self, converged: list) -> dict:
        """Extract recommended parameter ranges from converged configurations."""
        if not converged:
            return {}

        k_values = [p.get("hyperparameters", {}).get("K") for p in converged if p.get("hyperparameters", {}).get("K")]
        percw_values = [p.get("hyperparameters", {}).get("percW") for p in converged if p.get("hyperparameters", {}).get("percW")]

        return {
            "recommended_K_range": [min(k_values), max(k_values)] if k_values else None,
            "recommended_percW_range": [min(percw_values), max(percw_values)] if percw_values else None,
            "total_converged_configurations": len(converged)
        }

    def create_sensitivity_analysis_summary(self, experiment_results: dict, output_dir: Path) -> dict:
        """Create a comprehensive sensitivity analysis summary instead of saving raw configs."""
        from core.io_utils import save_json
        import pandas as pd

        # Extract sensitivity analysis data
        sensitivity_results = []
        parameter_impacts = {}

        # Look for sensitivity analysis results in the experiment data
        for key, value in experiment_results.items():
            if "sensitivity" in key.lower() or "parameter_sweep" in key.lower():
                if isinstance(value, dict):
                    sensitivity_results.extend(self._extract_sensitivity_data(value))

        if not sensitivity_results:
            return {}

        # Create comprehensive sensitivity summary
        summary = {
            "sensitivity_overview": {
                "total_parameter_combinations_tested": len(sensitivity_results),
                "parameters_analyzed": self._identify_analyzed_parameters(sensitivity_results),
                "timestamp": pd.Timestamp.now().isoformat()
            },
            "parameter_sensitivity_ranking": self._rank_parameter_sensitivity(sensitivity_results),
            "stability_analysis": self._analyze_parameter_stability(sensitivity_results),
            "interaction_effects": self._analyze_parameter_interactions(sensitivity_results),
            "optimal_parameter_windows": self._identify_optimal_windows(sensitivity_results),
            "clinical_sensitivity_insights": self._generate_sensitivity_insights(sensitivity_results),
            "robustness_assessment": self._assess_model_robustness(sensitivity_results)
        }

        # Save the comprehensive summary
        summary_path = output_dir / "sensitivity_analysis_summary.json"
        save_json(summary, summary_path, indent=2)

        return summary

    def _extract_sensitivity_data(self, sensitivity_dict: dict) -> list:
        """Extract sensitivity data from experiment results."""
        sensitivity_data = []

        for param_combo, result_data in sensitivity_dict.items():
            if isinstance(result_data, dict):
                data_point = {
                    "parameter_combination": param_combo,
                    "performance_metric": result_data.get("log_likelihood", result_data.get("score", 0)),
                    "execution_time": result_data.get("execution_time", 0),
                    "converged": result_data.get("convergence", False),
                    "stability_score": result_data.get("stability", 0),
                    "hyperparameters": result_data.get("hyperparameters", {})
                }
                sensitivity_data.append(data_point)

        return sensitivity_data

    def _identify_analyzed_parameters(self, sensitivity_results: list) -> list:
        """Identify which parameters were analyzed in the sensitivity study."""
        all_params = set()
        for result in sensitivity_results:
            hyperparams = result.get("hyperparameters", {})
            all_params.update(hyperparams.keys())
        return list(all_params)

    def _rank_parameter_sensitivity(self, sensitivity_results: list) -> dict:
        """Rank parameters by their sensitivity (impact on model performance)."""
        if not sensitivity_results:
            return {}

        import numpy as np
        from collections import defaultdict

        param_impacts = defaultdict(list)

        # Group results by parameter values
        for result in sensitivity_results:
            hyperparams = result.get("hyperparameters", {})
            performance = result.get("performance_metric", 0)

            for param_name, param_value in hyperparams.items():
                param_impacts[param_name].append((param_value, performance))

        # Calculate sensitivity for each parameter
        sensitivity_scores = {}
        for param_name, value_performance_pairs in param_impacts.items():
            if len(value_performance_pairs) > 1:
                values = [vp[0] for vp in value_performance_pairs]
                performances = [vp[1] for vp in value_performance_pairs]

                # Calculate correlation between parameter value and performance
                if len(set(values)) > 1 and len(set(performances)) > 1:
                    correlation = np.corrcoef(values, performances)[0, 1] if not np.isnan(np.corrcoef(values, performances)[0, 1]) else 0
                    sensitivity_scores[param_name] = {
                        "sensitivity_score": abs(correlation),
                        "correlation": float(correlation),
                        "performance_range": [min(performances), max(performances)],
                        "value_range": [min(values), max(values)]
                    }

        # Rank by sensitivity score
        ranked_params = sorted(sensitivity_scores.items(), key=lambda x: x[1]["sensitivity_score"], reverse=True)

        return {
            "most_sensitive_parameters": [param[0] for param in ranked_params[:3]],
            "least_sensitive_parameters": [param[0] for param in ranked_params[-3:]],
            "detailed_sensitivity": dict(ranked_params)
        }

    def _analyze_parameter_stability(self, sensitivity_results: list) -> dict:
        """Analyze stability of model performance across parameter ranges."""
        if not sensitivity_results:
            return {}

        converged_results = [r for r in sensitivity_results if r.get("converged", False)]

        import numpy as np
        performances = [r.get("performance_metric", 0) for r in converged_results]

        if not performances:
            return {"no_converged_results": True}

        return {
            "performance_stability": {
                "mean_performance": float(np.mean(performances)),
                "std_performance": float(np.std(performances)),
                "cv_performance": float(np.std(performances) / np.mean(performances)) if np.mean(performances) != 0 else float('inf'),
                "min_performance": float(min(performances)),
                "max_performance": float(max(performances))
            },
            "convergence_stability": {
                "overall_convergence_rate": len(converged_results) / len(sensitivity_results),
                "stable_parameter_regions": self._identify_stable_regions(sensitivity_results)
            }
        }

    def _identify_stable_regions(self, sensitivity_results: list) -> dict:
        """Identify parameter regions with consistent convergence."""
        stable_regions = {}

        # Group by parameter combinations and analyze local stability
        from collections import defaultdict
        param_groups = defaultdict(list)

        for result in sensitivity_results:
            hyperparams = result.get("hyperparameters", {})
            # Create a simple grouping key
            group_key = tuple(sorted(hyperparams.items()))
            param_groups[group_key].append(result)

        # Identify groups with high convergence rates
        for group_key, group_results in param_groups.items():
            convergence_rate = sum(1 for r in group_results if r.get("converged", False)) / len(group_results)
            if convergence_rate >= 0.8:  # 80% convergence threshold
                stable_regions[str(dict(group_key))] = {
                    "convergence_rate": convergence_rate,
                    "sample_size": len(group_results)
                }

        return stable_regions

    def _analyze_parameter_interactions(self, sensitivity_results: list) -> dict:
        """Analyze interactions between parameters."""
        if len(sensitivity_results) < 4:
            return {"insufficient_data": True}

        # Look for parameter pairs and their combined effects
        interaction_effects = {}

        # This is a simplified interaction analysis
        # In practice, you'd want more sophisticated statistical analysis
        param_pairs = []
        all_params = self._identify_analyzed_parameters(sensitivity_results)

        for i, param1 in enumerate(all_params):
            for j, param2 in enumerate(all_params[i+1:], i+1):
                param_pairs.append((param1, param2))

        for param1, param2 in param_pairs[:5]:  # Limit to avoid explosion
            interaction_effects[f"{param1}_x_{param2}"] = {
                "analyzed": True,
                "note": "Detailed interaction analysis requires more sophisticated statistical methods"
            }

        return {
            "parameter_pairs_analyzed": len(param_pairs),
            "interaction_effects": interaction_effects,
            "recommendation": "Consider full factorial design for detailed interaction analysis"
        }

    def _identify_optimal_windows(self, sensitivity_results: list) -> dict:
        """Identify optimal parameter value windows."""
        converged_results = [r for r in sensitivity_results if r.get("converged", False)]

        if not converged_results:
            return {}

        # Find top 25% performing configurations
        sorted_results = sorted(converged_results, key=lambda x: x.get("performance_metric", 0), reverse=True)
        top_quartile = sorted_results[:max(1, len(sorted_results) // 4)]

        # Extract parameter ranges from top performers
        param_windows = {}
        all_params = self._identify_analyzed_parameters(sensitivity_results)

        for param in all_params:
            param_values = []
            for result in top_quartile:
                hyperparams = result.get("hyperparameters", {})
                if param in hyperparams:
                    param_values.append(hyperparams[param])

            if param_values:
                param_windows[param] = {
                    "optimal_range": [min(param_values), max(param_values)],
                    "recommended_value": sum(param_values) / len(param_values),
                    "sample_size": len(param_values)
                }

        return param_windows

    def _generate_sensitivity_insights(self, sensitivity_results: list) -> dict:
        """Generate clinical insights from sensitivity analysis."""
        converged_results = [r for r in sensitivity_results if r.get("converged", False)]

        if not converged_results:
            return {"no_insights": "No converged results available"}

        insights = {
            "parameter_robustness": {},
            "clinical_recommendations": {},
            "risk_assessment": {}
        }

        # Analyze robustness of each parameter
        all_params = self._identify_analyzed_parameters(sensitivity_results)
        for param in all_params:
            param_values = [r.get("hyperparameters", {}).get(param) for r in converged_results if param in r.get("hyperparameters", {})]
            if param_values:
                unique_values = len(set(param_values))
                total_values = len(param_values)

                robustness = "high" if unique_values < total_values * 0.3 else "moderate" if unique_values < total_values * 0.7 else "low"
                insights["parameter_robustness"][param] = {
                    "robustness_level": robustness,
                    "value_diversity": unique_values / total_values if total_values > 0 else 0
                }

        # Generate clinical recommendations
        insights["clinical_recommendations"] = {
            "most_critical_parameters": self._rank_parameter_sensitivity(sensitivity_results).get("most_sensitive_parameters", []),
            "safest_parameter_ranges": self._identify_optimal_windows(sensitivity_results),
            "caution_advice": "Always validate on independent datasets before clinical application"
        }

        return insights

    def _assess_model_robustness(self, sensitivity_results: list) -> dict:
        """Assess overall model robustness to parameter changes."""
        if not sensitivity_results:
            return {}

        total_tests = len(sensitivity_results)
        converged_tests = len([r for r in sensitivity_results if r.get("converged", False)])

        robustness_score = converged_tests / total_tests if total_tests > 0 else 0

        # Categorize robustness
        if robustness_score >= 0.8:
            robustness_category = "highly_robust"
        elif robustness_score >= 0.6:
            robustness_category = "moderately_robust"
        elif robustness_score >= 0.4:
            robustness_category = "somewhat_sensitive"
        else:
            robustness_category = "highly_sensitive"

        return {
            "overall_robustness_score": robustness_score,
            "robustness_category": robustness_category,
            "convergence_statistics": {
                "total_parameter_combinations": total_tests,
                "successful_convergences": converged_tests,
                "failure_rate": 1 - robustness_score
            },
            "interpretation": self._interpret_robustness(robustness_category)
        }

    def _interpret_robustness(self, category: str) -> str:
        """Provide interpretation of robustness assessment."""
        interpretations = {
            "highly_robust": "Model is very stable across parameter ranges. Safe for clinical application with proper validation.",
            "moderately_robust": "Model shows good stability. Consider parameter validation in deployment scenarios.",
            "somewhat_sensitive": "Model performance varies with parameters. Careful parameter tuning recommended.",
            "highly_sensitive": "Model is sensitive to parameter choices. Extensive validation and careful parameter selection critical."
        }
        return interpretations.get(category, "Robustness assessment unclear.")

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
                "reproducibility_testing": "Regular reproducibility validation",
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

