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
- integration: High-level workflows coordinating multiple clinical modules

These modules are extracted from the original clinical_validation experiment
to enable reuse across model_comparison and sgfa_configuration_comparison experiments.
"""

from .biomarker_analysis import BiomarkerAnalyzer
from .classification import ClinicalClassifier
from .data_processing import ClinicalDataProcessor
from .external_validation import ExternalValidator
from .metrics import ClinicalMetrics
from .progression_analysis import DiseaseProgressionAnalyzer
from .subtype_analysis import PDSubtypeAnalyzer

# High-level integration workflows
from .integration import (
    ClinicalValidationComponents,
    create_clinical_metrics_calculator,
    create_clinical_processor,
    create_clinical_validation_suite,
    run_comprehensive_clinical_validation,
    run_targeted_clinical_validation,
)

__all__ = [
    # Individual modules
    "ClinicalMetrics",
    "ClinicalDataProcessor",
    "ClinicalClassifier",
    "PDSubtypeAnalyzer",
    "DiseaseProgressionAnalyzer",
    "BiomarkerAnalyzer",
    "ExternalValidator",
    # Integration workflows
    "ClinicalValidationComponents",
    "create_clinical_validation_suite",
    "create_clinical_processor",
    "create_clinical_metrics_calculator",
    "run_comprehensive_clinical_validation",
    "run_targeted_clinical_validation",
]

# Version info
__version__ = "1.0.0"
