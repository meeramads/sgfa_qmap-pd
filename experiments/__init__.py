"""Comprehensive experimental framework for qMAP-PD SGFA analysis."""

from .framework import ExperimentFramework, ExperimentConfig, ExperimentRunner
from .data_validation import DataValidationExperiments
from .method_comparison import MethodComparisonExperiments  
from .sensitivity_analysis import SensitivityAnalysisExperiments
from .reproducibility import ReproducibilityExperiments
from .performance_benchmarks import PerformanceBenchmarkExperiments
from .clinical_validation import ClinicalValidationExperiments

__all__ = [
    'ExperimentFramework', 'ExperimentConfig', 'ExperimentRunner',
    'DataValidationExperiments', 'MethodComparisonExperiments',
    'SensitivityAnalysisExperiments', 'ReproducibilityExperiments', 
    'PerformanceBenchmarkExperiments', 'ClinicalValidationExperiments'
]