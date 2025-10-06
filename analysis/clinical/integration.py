"""
Clinical validation integration factory.

Provides high-level workflows for clinical validation experiments by
coordinating the clinical analysis modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .biomarker_analysis import BiomarkerAnalyzer
from .classification import ClinicalClassifier
from .data_processing import ClinicalDataProcessor
from .external_validation import ExternalValidator
from .metrics import ClinicalMetrics
from .progression_analysis import DiseaseProgressionAnalyzer
from .subtype_analysis import PDSubtypeAnalyzer


logger = logging.getLogger(__name__)


@dataclass
class ClinicalValidationComponents:
    """Container for clinical validation components."""

    metrics: ClinicalMetrics
    processor: ClinicalDataProcessor
    classifier: ClinicalClassifier
    subtype_analyzer: PDSubtypeAnalyzer
    progression_analyzer: DiseaseProgressionAnalyzer
    biomarker_analyzer: BiomarkerAnalyzer
    external_validator: ExternalValidator
    logger: logging.Logger


def create_clinical_validation_suite(
    logger: Optional[logging.Logger] = None,
) -> ClinicalValidationComponents:
    """
    Create a complete suite of clinical validation components.

    Args:
        logger: Optional logger instance

    Returns:
        ClinicalValidationComponents with all modules initialized
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Initializing clinical validation suite...")

    components = ClinicalValidationComponents(
        metrics=ClinicalMetrics(logger=logger),
        processor=ClinicalDataProcessor(logger=logger),
        classifier=ClinicalClassifier(logger=logger),
        subtype_analyzer=PDSubtypeAnalyzer(logger=logger),
        progression_analyzer=DiseaseProgressionAnalyzer(logger=logger),
        biomarker_analyzer=BiomarkerAnalyzer(logger=logger),
        external_validator=ExternalValidator(logger=logger),
        logger=logger,
    )

    logger.info("✅ Clinical validation suite initialized")
    return components


def run_comprehensive_clinical_validation(
    X_list: List[np.ndarray],
    clinical_data: Dict[str, Any],
    factor_results: Dict[str, Any],
    validation_config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive clinical validation workflow.

    Coordinates multiple clinical modules to perform:
    - Clinical metrics calculation
    - PD subtype analysis
    - Disease progression modeling
    - Biomarker discovery
    - Clinical classification

    Args:
        X_list: List of data matrices
        clinical_data: Clinical data dictionary with diagnosis, symptoms, etc.
        factor_results: SGFA factor analysis results
        validation_config: Optional configuration for validation
        logger: Optional logger instance

    Returns:
        Comprehensive validation results dictionary
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if validation_config is None:
        validation_config = {}

    logger.info("🏥 Running comprehensive clinical validation...")

    # Initialize components
    suite = create_clinical_validation_suite(logger=logger)

    results = {
        "validation_type": "comprehensive_clinical",
        "status": "running",
    }

    try:
        # 1. Calculate clinical metrics
        logger.info("📊 Calculating clinical metrics...")
        clinical_metrics = suite.metrics.calculate_clinical_interpretability(
            factor_results, clinical_data
        )
        results["clinical_metrics"] = clinical_metrics

        # 2. Subtype analysis
        logger.info("🔬 Analyzing PD subtypes...")
        subtype_results = suite.subtype_analyzer.discover_subtypes(
            factor_results.get("Z", np.array([])),
            clinical_data,
            n_subtypes=validation_config.get("n_subtypes", 3),
        )
        results["subtype_analysis"] = subtype_results

        # 3. Clinical classification
        if "diagnosis" in clinical_data:
            logger.info("🎯 Running clinical classification...")
            classification_results = suite.classifier.evaluate_clinical_prediction(
                Z=factor_results.get("Z", np.array([])),
                clinical_data=clinical_data,
                prediction_target=validation_config.get("target", "diagnosis"),
            )
            results["classification"] = classification_results

        # 4. Disease progression
        if "updrs" in clinical_data or "disease_duration" in clinical_data:
            logger.info("📈 Analyzing disease progression...")
            progression_results = suite.progression_analyzer.model_disease_progression(
                Z=factor_results.get("Z", np.array([])), clinical_data=clinical_data
            )
            results["progression"] = progression_results

        # 5. Biomarker discovery
        logger.info("💊 Discovering biomarkers...")
        biomarker_results = suite.biomarker_analyzer.discover_biomarkers(
            factor_results, clinical_data, X_list
        )
        results["biomarkers"] = biomarker_results

        results["status"] = "completed"
        logger.info("✅ Comprehensive clinical validation completed")

    except Exception as e:
        from core.error_handling import log_and_return_error

        error_info = log_and_return_error(
            e, logger, "Comprehensive clinical validation"
        )
        results.update(error_info)

    return results


def run_targeted_clinical_validation(
    validation_type: str,
    X_list: List[np.ndarray],
    clinical_data: Dict[str, Any],
    factor_results: Dict[str, Any],
    validation_config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Run targeted clinical validation for specific analysis type.

    Args:
        validation_type: Type of validation ("subtype", "classification",
                        "progression", "biomarker", "external")
        X_list: List of data matrices
        clinical_data: Clinical data dictionary
        factor_results: SGFA factor analysis results
        validation_config: Optional configuration
        logger: Optional logger instance

    Returns:
        Validation results for specified type
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if validation_config is None:
        validation_config = {}

    logger.info(f"🏥 Running {validation_type} validation...")

    suite = create_clinical_validation_suite(logger=logger)

    if validation_type == "subtype":
        return suite.subtype_analyzer.discover_subtypes(
            factor_results.get("Z", np.array([])),
            clinical_data,
            n_subtypes=validation_config.get("n_subtypes", 3),
        )

    elif validation_type == "classification":
        return suite.classifier.evaluate_clinical_prediction(
            Z=factor_results.get("Z", np.array([])),
            clinical_data=clinical_data,
            prediction_target=validation_config.get("target", "diagnosis"),
        )

    elif validation_type == "progression":
        return suite.progression_analyzer.model_disease_progression(
            Z=factor_results.get("Z", np.array([])), clinical_data=clinical_data
        )

    elif validation_type == "biomarker":
        return suite.biomarker_analyzer.discover_biomarkers(
            factor_results, clinical_data, X_list
        )

    elif validation_type == "external":
        external_data = validation_config.get("external_data")
        if external_data is None:
            raise ValueError("External validation requires 'external_data' in config")

        return suite.external_validator.validate_on_external_cohort(
            factor_results, clinical_data, external_data
        )

    else:
        raise ValueError(
            f"Unknown validation type: {validation_type}. "
            f"Choose from: subtype, classification, progression, biomarker, external"
        )


def create_clinical_processor(
    logger: Optional[logging.Logger] = None,
) -> ClinicalDataProcessor:
    """
    Create a standalone ClinicalDataProcessor instance.

    Convenience function for experiments that only need data processing.

    Args:
        logger: Optional logger instance

    Returns:
        ClinicalDataProcessor instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    return ClinicalDataProcessor(logger=logger)


def create_clinical_metrics_calculator(
    logger: Optional[logging.Logger] = None,
) -> ClinicalMetrics:
    """
    Create a standalone ClinicalMetrics instance.

    Convenience function for experiments that only need metrics calculation.

    Args:
        logger: Optional logger instance

    Returns:
        ClinicalMetrics instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    return ClinicalMetrics(logger=logger)
