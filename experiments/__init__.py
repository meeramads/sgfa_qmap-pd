"""Comprehensive experimental framework for qMAP-PD SGFA analysis."""

from .framework import ExperimentFramework, ExperimentConfig, ExperimentRunner
from .data_validation import DataValidationExperiments
from .sgfa_parameter_comparison import SGFAParameterComparison
from .model_comparison import ModelArchitectureComparison
from .sensitivity_analysis import SensitivityAnalysisExperiments
from .reproducibility import ReproducibilityExperiments
from .performance_benchmarks import PerformanceBenchmarkExperiments
from .clinical_validation import ClinicalValidationExperiments

__all__ = [
    'ExperimentFramework', 'ExperimentConfig', 'ExperimentRunner',
    'DataValidationExperiments', 'SGFAParameterComparison', 'ModelArchitectureComparison',
    'SensitivityAnalysisExperiments', 'ReproducibilityExperiments', 
    'PerformanceBenchmarkExperiments', 'ClinicalValidationExperiments'
]