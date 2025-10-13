"""
High-level experiment runner - separate from framework base classes.

This module provides high-level interfaces for running common experiment types
without creating circular dependencies with the framework module.
"""

import logging
from typing import Any, Dict, List

from experiments.framework import ExperimentConfig, ExperimentFramework, ExperimentResult

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """High-level interface for running common experiment types."""

    def __init__(self, framework: ExperimentFramework):
        """Initialize with experiment framework."""
        self.framework = framework

    def run_hyperparameter_search(
        self, base_config: ExperimentConfig, search_space: Dict[str, List[Any]]
    ) -> List[ExperimentResult]:
        """Run hyperparameter search experiment."""
        from experiments.sensitivity_analysis import SensitivityAnalysisExperiments

        sensitivity_exp = SensitivityAnalysisExperiments(self.framework)
        return sensitivity_exp.hyperparameter_sensitivity_analysis(
            base_config, search_space
        )

    def run_method_comparison(
        self, base_config: ExperimentConfig, methods: List[str]
    ) -> List[ExperimentResult]:
        """Run method comparison experiment."""
        from experiments.model_comparison import ModelArchitectureComparison

        comparison_exp = ModelArchitectureComparison(self.framework)
        return comparison_exp.compare_methods(base_config, methods)

    def run_reproducibility_test(
        self, config: ExperimentConfig, num_repetitions: int = 10
    ) -> List[ExperimentResult]:
        """Run reproducibility test."""
        from experiments.robustness_testing import ReproducibilityExperiments

        repro_exp = ReproducibilityExperiments(self.framework)
        return repro_exp.test_reproducibility(config, num_repetitions)

    def run_clinical_validation(
        self, config: ExperimentConfig, clinical_outcomes: List[str]
    ) -> ExperimentResult:
        """Run clinical validation experiment."""
        from experiments.clinical_validation import ClinicalValidationExperiments

        clinical_exp = ClinicalValidationExperiments(self.framework)
        return clinical_exp.validate_clinical_associations(config, clinical_outcomes)