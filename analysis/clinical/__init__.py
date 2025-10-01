"""Clinical validation modules for SGFA qMAP-PD analysis.

This package provides modular clinical validation capabilities that can be
used across different experiments:

- metrics: Clinical metrics calculation and interpretability analysis
- data_processing: SGFA execution and data preparation for clinical contexts
- classification: Clinical classification and prediction utilities
- subtype_analysis: PD subtype discovery and validation
- progression_analysis: Disease progression modeling and analysis
- biomarker_analysis: Biomarker discovery and validation
- external_validation: External cohort validation and model transferability

These modules are extracted from the original clinical_validation experiment
to enable reuse across model_comparison and sgfa_parameter_comparison experiments.
"""

from .metrics import ClinicalMetrics
from .data_processing import ClinicalDataProcessor
from .classification import ClinicalClassifier
from .subtype_analysis import PDSubtypeAnalyzer
from .progression_analysis import DiseaseProgressionAnalyzer
from .biomarker_analysis import BiomarkerAnalyzer
from .external_validation import ExternalValidator

__all__ = [
    "ClinicalMetrics",
    "ClinicalDataProcessor",
    "ClinicalClassifier",
    "PDSubtypeAnalyzer",
    "DiseaseProgressionAnalyzer",
    "BiomarkerAnalyzer",
    "ExternalValidator",
]

# Version info
__version__ = "1.0.0"
