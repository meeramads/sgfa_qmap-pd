"""Core experimental framework for systematic qMAP-PD analysis."""

import logging
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
import yaml

from performance import PerformanceProfiler, MemoryOptimizer, PerformanceManager, PerformanceConfig
from core.utils import safe_pickle_save, safe_pickle_load

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
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: Path):
        """Save configuration to file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'ExperimentConfig':
        """Load configuration from file."""
        with open(filepath, 'r') as f:
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
        result_dict['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result_dict['end_time'] = self.end_time.isoformat()
        return result_dict


class ExperimentFramework:
    """Core framework for running systematic experiments."""
    
    def __init__(self, 
                 base_output_dir: Path,
                 performance_config: Optional[PerformanceConfig] = None):
        """
        Initialize experiment framework.
        
        Parameters
        ----------
        base_output_dir : Path
            Base directory for all experiment outputs.
        performance_config : PerformanceConfig, optional
            Performance optimization configuration.
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.performance_config = performance_config
        self.results_history: List[ExperimentResult] = []
        
        # Initialize logging
        self._setup_logging()
        
        logger.info(f"ExperimentFramework initialized with output dir: {base_output_dir}")
    
    def _setup_logging(self):
        """Setup experiment-specific logging."""
        log_file = self.base_output_dir / "experiments.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    
    def run_experiment(self, 
                      config: ExperimentConfig,
                      experiment_function: Callable,
                      **kwargs) -> ExperimentResult:
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
        experiment_id = f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize result object
        result = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            start_time=datetime.now()
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
                config=config,
                output_dir=exp_dir,
                **kwargs
            )
            
            # Store results
            result.model_results = experiment_results.get('model_results', {})
            result.cv_results = experiment_results.get('cv_results', {})
            result.diagnostics = experiment_results.get('diagnostics', {})
            result.data_summary = experiment_results.get('data_summary', {})
            result.convergence_diagnostics = experiment_results.get('convergence_diagnostics', {})
            
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
        
        # Save results
        self._save_experiment_result(result, exp_dir)
        self.results_history.append(result)
        
        return result
    
    def _save_experiment_result(self, result: ExperimentResult, output_dir: Path):
        """Save experiment result to files."""
        # Save as JSON
        result_json = output_dir / "result.json"
        with open(result_json, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        # Save as pickle for complete object
        result_pkl = output_dir / "result.pkl"
        safe_pickle_save(result, result_pkl, "Experiment result")
        
        # Save summary CSV
        self._save_result_summary(result, output_dir / "summary.csv")
    
    def _save_result_summary(self, result: ExperimentResult, filepath: Path):
        """Save a summary of results as CSV."""
        summary_data = {
            'experiment_id': result.experiment_id,
            'experiment_name': result.config.experiment_name,
            'status': result.status,
            'duration_seconds': result.get_duration(),
            'num_samples': result.config.num_samples,
            'num_chains': result.config.num_chains,
            'K_values': str(result.config.K_values),
            'cv_folds': result.config.cv_folds
        }
        
        # Add model performance metrics if available
        if result.model_results:
            for key, value in result.model_results.items():
                if isinstance(value, (int, float, str)):
                    summary_data[f'model_{key}'] = value
        
        # Add CV metrics if available
        if result.cv_results:
            for key, value in result.cv_results.items():
                if isinstance(value, (int, float, str)):
                    summary_data[f'cv_{key}'] = value
        
        # Save as single-row CSV
        df = pd.DataFrame([summary_data])
        df.to_csv(filepath, index=False)
    
    def run_experiment_grid(self,
                           base_config: ExperimentConfig,
                           parameter_grid: Dict[str, List[Any]],
                           experiment_function: Callable) -> List[ExperimentResult]:
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
            param_str = "_".join([f"{name}_{value}" for name, value in zip(param_names, param_combination)])
            config.experiment_name = f"{base_config.experiment_name}_grid_{param_str}"
            
            logger.info(f"Running experiment {i+1}/{total_experiments}: {config.experiment_name}")
            
            # Run experiment
            result = self.run_experiment(config, experiment_function)
            results.append(result)
        
        # Save grid summary
        self._save_grid_summary(results, parameter_grid)
        
        return results
    
    def _save_grid_summary(self, results: List[ExperimentResult], parameter_grid: Dict[str, List[Any]]):
        """Save summary of grid search results."""
        summary_data = []
        
        for result in results:
            row = {
                'experiment_id': result.experiment_id,
                'status': result.status,
                'duration_seconds': result.get_duration()
            }
            
            # Add parameter values
            for param_name in parameter_grid.keys():
                if hasattr(result.config, param_name):
                    row[param_name] = getattr(result.config, param_name)
            
            # Add key metrics
            if result.model_results:
                for key, value in result.model_results.items():
                    if isinstance(value, (int, float)):
                        row[f'model_{key}'] = value
            
            if result.cv_results:
                for key, value in result.cv_results.items():
                    if isinstance(value, (int, float)):
                        row[f'cv_{key}'] = value
            
            summary_data.append(row)
        
        # Save grid summary
        df = pd.DataFrame(summary_data)
        grid_summary_path = self.base_output_dir / "grid_search_summary.csv"
        df.to_csv(grid_summary_path, index=False)
        
        logger.info(f"Grid search summary saved to: {grid_summary_path}")
    
    def get_best_result(self, 
                       results: List[ExperimentResult],
                       metric: str = "cv_accuracy",
                       maximize: bool = True) -> Optional[ExperimentResult]:
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
        best_result, best_value = max(results_with_metrics, key=lambda x: x[1]) if maximize else min(results_with_metrics, key=lambda x: x[1])
        
        logger.info(f"Best result: {best_result.experiment_id} with {metric}={best_value}")
        return best_result
    
    def generate_experiment_report(self, results: List[ExperimentResult] = None) -> Dict[str, Any]:
        """Generate comprehensive experiment report."""
        if results is None:
            results = self.results_history
        
        if not results:
            return {"message": "No experiment results available"}
        
        # Calculate summary statistics
        completed_results = [r for r in results if r.status == "completed"]
        failed_results = [r for r in results if r.status == "failed"]
        
        durations = [r.get_duration() for r in completed_results if r.get_duration() is not None]
        
        report = {
            "experiment_summary": {
                "total_experiments": len(results),
                "completed": len(completed_results),
                "failed": len(failed_results),
                "success_rate": len(completed_results) / len(results) if results else 0,
                "total_runtime_hours": sum(durations) / 3600 if durations else 0,
                "average_experiment_duration_minutes": np.mean(durations) / 60 if durations else 0
            },
            "experiment_details": [
                {
                    "experiment_id": r.experiment_id,
                    "experiment_name": r.config.experiment_name,
                    "status": r.status,
                    "duration_minutes": r.get_duration() / 60 if r.get_duration() else None,
                    "error_message": r.error_message
                }
                for r in results
            ]
        }
        
        # Add performance statistics if available
        if completed_results:
            performance_data = []
            for result in completed_results:
                if result.performance_metrics:
                    perf_metrics = result.performance_metrics
                    if 'memory_report' in perf_metrics:
                        memory_report = perf_metrics['memory_report']
                        performance_data.append({
                            'experiment_id': result.experiment_id,
                            'peak_memory_gb': memory_report.get('peak_memory_gb', 0),
                            'operations_completed': memory_report.get('operations_completed', 0)
                        })
            
            if performance_data:
                peak_memories = [p['peak_memory_gb'] for p in performance_data]
                report["performance_summary"] = {
                    "average_peak_memory_gb": np.mean(peak_memories),
                    "max_peak_memory_gb": np.max(peak_memories),
                    "min_peak_memory_gb": np.min(peak_memories)
                }
        
        return report
    
    def save_experiment_report(self, filepath: Path, results: List[ExperimentResult] = None):
        """Save experiment report to file."""
        report = self.generate_experiment_report(results)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Experiment report saved to: {filepath}")


class ExperimentRunner:
    """High-level interface for running common experiment types."""
    
    def __init__(self, framework: ExperimentFramework):
        """Initialize with experiment framework."""
        self.framework = framework
    
    def run_hyperparameter_search(self,
                                 base_config: ExperimentConfig,
                                 search_space: Dict[str, List[Any]]) -> List[ExperimentResult]:
        """Run hyperparameter search experiment."""
        from .sensitivity_analysis import SensitivityAnalysisExperiments
        
        sensitivity_exp = SensitivityAnalysisExperiments(self.framework)
        return sensitivity_exp.hyperparameter_sensitivity_analysis(base_config, search_space)
    
    def run_method_comparison(self,
                            base_config: ExperimentConfig,
                            methods: List[str]) -> List[ExperimentResult]:
        """Run method comparison experiment."""
        from .method_comparison import MethodComparisonExperiments
        
        comparison_exp = MethodComparisonExperiments(self.framework)
        return comparison_exp.compare_methods(base_config, methods)
    
    def run_reproducibility_test(self,
                                config: ExperimentConfig,
                                num_repetitions: int = 10) -> List[ExperimentResult]:
        """Run reproducibility test."""
        from .reproducibility import ReproducibilityExperiments
        
        repro_exp = ReproducibilityExperiments(self.framework)
        return repro_exp.test_reproducibility(config, num_repetitions)
    
    def run_clinical_validation(self,
                              config: ExperimentConfig,
                              clinical_outcomes: List[str]) -> ExperimentResult:
        """Run clinical validation experiment."""
        from .clinical_validation import ClinicalValidationExperiments
        
        clinical_exp = ClinicalValidationExperiments(self.framework)
        return clinical_exp.validate_clinical_associations(config, clinical_outcomes)