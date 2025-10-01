"""Clinical validation modules for SGFA qMAP-PD analysis.

This package provides modular clinical validation capabilities:
- data_processing: SGFA execution and data preparation
- subtype_analysis: PD subtype discovery and validation
- progression_analysis: Disease progression modeling
- biomarker_analysis: Biomarker discovery and validation
- external_validation: External cohort validation
- metrics: Clinical performance metrics
- visualization: Clinical plotting utilities
- main_validation: Orchestration and integration

The original ClinicalValidationExperiments class is maintained for backward compatibility
but delegates to these focused modules internally.
"""

from .data_processing import ClinicalDataProcessor
from .subtype_analysis import PDSubtypeAnalyzer
from .progression_analysis import DiseaseProgressionAnalyzer
from .biomarker_analysis import BiomarkerAnalyzer
from .external_validation import ExternalValidator
from .metrics import ClinicalMetrics
from .visualization import ClinicalVisualizer
from .main_validation import ClinicalValidationOrchestrator

# Maintain backward compatibility by exposing the integrated class
from .main_validation import ClinicalValidationExperiments

__all__ = [
    "ClinicalDataProcessor",
    "PDSubtypeAnalyzer",
    "DiseaseProgressionAnalyzer",
    "BiomarkerAnalyzer",
    "ExternalValidator",
    "ClinicalMetrics",
    "ClinicalVisualizer",
    "ClinicalValidationOrchestrator",
    "ClinicalValidationExperiments",  # Backward compatibility
]