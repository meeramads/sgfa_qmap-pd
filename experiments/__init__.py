"""Comprehensive experimental framework for qMAP-PD SGFA analysis.

All experiment classes now include automatic performance optimization through the
@performance_optimized_experiment decorator, providing:
- Memory optimization and monitoring
- Data streaming for large datasets
- MCMC memory optimization
- Adaptive batch sizing
- System auto-configuration
"""

from .clinical_validation import ClinicalValidationExperiments
from .data_validation import DataValidationExperiments
from .framework import ExperimentConfig, ExperimentFramework
from .runner import ExperimentRunner
from .model_comparison import ModelArchitectureComparison
from .performance_benchmarks import PerformanceBenchmarkExperiments
from .reproducibility import ReproducibilityExperiments
from .sensitivity_analysis import SensitivityAnalysisExperiments
from .sgfa_parameter_comparison import SGFAParameterComparison

__all__ = [
    "ExperimentFramework",
    "ExperimentConfig",
    "ExperimentRunner",
    "DataValidationExperiments",
    "SGFAParameterComparison",
    "ModelArchitectureComparison",
    "SensitivityAnalysisExperiments",
    "ReproducibilityExperiments",
    "PerformanceBenchmarkExperiments",
    "ClinicalValidationExperiments",
]
