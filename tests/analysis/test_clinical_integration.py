"""Tests for analysis.clinical.integration module."""

from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

from analysis.clinical.integration import (
    ClinicalValidationComponents,
    create_clinical_validation_suite,
    create_clinical_processor,
    create_clinical_metrics_calculator,
    run_comprehensive_clinical_validation,
    run_targeted_clinical_validation,
)


class TestClinicalValidationComponents:
    """Test ClinicalValidationComponents dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        components = ClinicalValidationComponents(
            metrics=Mock(),
            processor=Mock(),
            classifier=Mock(),
            subtype_analyzer=Mock(),
            progression_analyzer=Mock(),
            biomarker_analyzer=Mock(),
            external_validator=Mock(),
        )

        assert components.metrics is not None
        assert components.processor is not None
        assert components.classifier is not None
        assert components.subtype_analyzer is not None
        assert components.progression_analyzer is not None
        assert components.biomarker_analyzer is not None
        assert components.external_validator is not None

    def test_minimal_initialization(self):
        """Test initialization with minimal components."""
        components = ClinicalValidationComponents(
            metrics=Mock(),
            processor=Mock(),
        )

        assert components.metrics is not None
        assert components.processor is not None
        assert components.classifier is None
        assert components.subtype_analyzer is None


@patch('analysis.clinical.integration.ClinicalMetrics')
class TestCreateClinicalMetricsCalculator:
    """Test create_clinical_metrics_calculator function."""

    def test_basic_creation(self, mock_metrics_class):
        """Test basic metrics calculator creation."""
        mock_metrics = Mock()
        mock_metrics_class.return_value = mock_metrics

        calculator = create_clinical_metrics_calculator()

        mock_metrics_class.assert_called_once()
        assert calculator == mock_metrics

    def test_with_logger(self, mock_metrics_class):
        """Test creation with logger."""
        mock_logger = Mock()
        calculator = create_clinical_metrics_calculator(logger=mock_logger)

        mock_metrics_class.assert_called_once()


@patch('analysis.clinical.integration.ClinicalDataProcessor')
class TestCreateClinicalProcessor:
    """Test create_clinical_processor function."""

    def test_basic_creation(self, mock_processor_class):
        """Test basic processor creation."""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        processor = create_clinical_processor()

        mock_processor_class.assert_called_once()
        assert processor == mock_processor

    def test_with_logger(self, mock_processor_class):
        """Test creation with logger."""
        mock_logger = Mock()
        processor = create_clinical_processor(logger=mock_logger)

        mock_processor_class.assert_called_once()


@patch('analysis.clinical.integration.ExternalValidator')
@patch('analysis.clinical.integration.BiomarkerAnalyzer')
@patch('analysis.clinical.integration.DiseaseProgressionAnalyzer')
@patch('analysis.clinical.integration.PDSubtypeAnalyzer')
@patch('analysis.clinical.integration.ClinicalClassifier')
@patch('analysis.clinical.integration.ClinicalDataProcessor')
@patch('analysis.clinical.integration.ClinicalMetrics')
class TestCreateClinicalValidationSuite:
    """Test create_clinical_validation_suite function."""

    def test_full_suite_creation(
        self,
        mock_metrics_class,
        mock_processor_class,
        mock_classifier_class,
        mock_subtype_class,
        mock_progression_class,
        mock_biomarker_class,
        mock_external_class,
    ):
        """Test creation of full validation suite."""
        suite = create_clinical_validation_suite()

        # All components should be instantiated
        mock_metrics_class.assert_called_once()
        mock_processor_class.assert_called_once()
        mock_classifier_class.assert_called_once()
        mock_subtype_class.assert_called_once()
        mock_progression_class.assert_called_once()
        mock_biomarker_class.assert_called_once()
        mock_external_class.assert_called_once()

        # Check suite structure
        assert isinstance(suite, ClinicalValidationComponents)
        assert suite.metrics is not None
        assert suite.processor is not None
        assert suite.classifier is not None

    def test_with_logger(
        self,
        mock_metrics_class,
        mock_processor_class,
        mock_classifier_class,
        mock_subtype_class,
        mock_progression_class,
        mock_biomarker_class,
        mock_external_class,
    ):
        """Test suite creation with logger."""
        mock_logger = Mock()
        suite = create_clinical_validation_suite(logger=mock_logger)

        # All components instantiated
        assert mock_metrics_class.called
        assert mock_processor_class.called


@patch('analysis.clinical.integration.create_clinical_validation_suite')
class TestRunComprehensiveClinicalValidation:
    """Test run_comprehensive_clinical_validation function."""

    def test_successful_validation(self, mock_create_suite):
        """Test successful comprehensive validation."""
        # Setup mock suite
        mock_suite = Mock()
        mock_suite.metrics = Mock()
        mock_suite.metrics.calculate_clinical_interpretability.return_value = {
            "interpretability_score": 0.85
        }
        mock_suite.subtype_analyzer = Mock()
        mock_suite.subtype_analyzer.discover_subtypes.return_value = {
            "n_subtypes": 3,
            "silhouette_score": 0.6,
        }
        mock_suite.progression_analyzer = Mock()
        mock_suite.progression_analyzer.analyze_disease_progression.return_value = {
            "progression_detected": True
        }
        mock_suite.biomarker_analyzer = Mock()
        mock_suite.biomarker_analyzer.discover_biomarkers.return_value = {
            "n_biomarkers": 5
        }
        mock_create_suite.return_value = mock_suite

        # Input data
        X_list = [np.array([[1, 2], [3, 4]])]
        clinical_data = {"diagnosis": [0, 1]}
        factor_results = {"W": np.array([[0.5, 0.3]]), "Z": np.array([[1.0], [2.0]])}

        result = run_comprehensive_clinical_validation(
            X_list, clinical_data, factor_results
        )

        # Check result structure
        assert result["validation_type"] == "comprehensive_clinical"
        assert result["status"] == "completed"
        assert "clinical_metrics" in result
        assert "subtype_analysis" in result
        assert "progression_analysis" in result
        assert "biomarker_analysis" in result

        # Check that all analyses were called
        mock_suite.metrics.calculate_clinical_interpretability.assert_called_once()
        mock_suite.subtype_analyzer.discover_subtypes.assert_called_once()
        mock_suite.progression_analyzer.analyze_disease_progression.assert_called_once()
        mock_suite.biomarker_analyzer.discover_biomarkers.assert_called_once()

    def test_validation_with_config(self, mock_create_suite):
        """Test validation with custom configuration."""
        mock_suite = Mock()
        mock_suite.metrics = Mock()
        mock_suite.metrics.calculate_clinical_interpretability.return_value = {}
        mock_suite.subtype_analyzer = Mock()
        mock_suite.subtype_analyzer.discover_subtypes.return_value = {}
        mock_suite.progression_analyzer = Mock()
        mock_suite.progression_analyzer.analyze_disease_progression.return_value = {}
        mock_suite.biomarker_analyzer = Mock()
        mock_suite.biomarker_analyzer.discover_biomarkers.return_value = {}
        mock_create_suite.return_value = mock_suite

        X_list = [np.array([[1, 2]])]
        clinical_data = {"diagnosis": [0]}
        factor_results = {"W": np.array([[0.5]]), "Z": np.array([[1.0]])}
        validation_config = {
            "n_subtypes_range": [2, 5],
            "biomarker_threshold": 0.7,
        }

        result = run_comprehensive_clinical_validation(
            X_list, clinical_data, factor_results, validation_config=validation_config
        )

        assert result["status"] == "completed"

    def test_validation_error_handling(self, mock_create_suite):
        """Test error handling in comprehensive validation."""
        # Setup suite that raises error
        mock_suite = Mock()
        mock_suite.metrics = Mock()
        mock_suite.metrics.calculate_clinical_interpretability.side_effect = ValueError(
            "Metrics calculation failed"
        )
        mock_create_suite.return_value = mock_suite

        X_list = [np.array([[1, 2]])]
        clinical_data = {"diagnosis": [0]}
        factor_results = {"W": np.array([[0.5]]), "Z": np.array([[1.0]])}

        result = run_comprehensive_clinical_validation(
            X_list, clinical_data, factor_results
        )

        # Should return error result
        assert result["status"] == "failed"
        assert "error" in result
        assert "Metrics calculation failed" in result["error"]

    def test_validation_with_logger(self, mock_create_suite):
        """Test validation with logger."""
        mock_suite = Mock()
        mock_suite.metrics = Mock()
        mock_suite.metrics.calculate_clinical_interpretability.return_value = {}
        mock_suite.subtype_analyzer = Mock()
        mock_suite.subtype_analyzer.discover_subtypes.return_value = {}
        mock_suite.progression_analyzer = Mock()
        mock_suite.progression_analyzer.analyze_disease_progression.return_value = {}
        mock_suite.biomarker_analyzer = Mock()
        mock_suite.biomarker_analyzer.discover_biomarkers.return_value = {}
        mock_create_suite.return_value = mock_suite

        mock_logger = Mock()

        X_list = [np.array([[1, 2]])]
        clinical_data = {"diagnosis": [0]}
        factor_results = {"W": np.array([[0.5]]), "Z": np.array([[1.0]])}

        result = run_comprehensive_clinical_validation(
            X_list, clinical_data, factor_results, logger=mock_logger
        )

        # Logger should have been used
        assert mock_logger.info.called or mock_logger.debug.called


@patch('analysis.clinical.integration.create_clinical_validation_suite')
class TestRunTargetedClinicalValidation:
    """Test run_targeted_clinical_validation function."""

    def test_metrics_only_validation(self, mock_create_suite):
        """Test targeted validation with metrics only."""
        mock_suite = Mock()
        mock_suite.metrics = Mock()
        mock_suite.metrics.calculate_clinical_interpretability.return_value = {
            "interpretability_score": 0.9
        }
        mock_create_suite.return_value = mock_suite

        X_list = [np.array([[1, 2]])]
        clinical_data = {"diagnosis": [0]}
        factor_results = {"W": np.array([[0.5]]), "Z": np.array([[1.0]])}

        result = run_targeted_clinical_validation(
            X_list,
            clinical_data,
            factor_results,
            analyses=["metrics"],
        )

        assert result["status"] == "completed"
        assert "clinical_metrics" in result
        assert "subtype_analysis" not in result
        assert "progression_analysis" not in result

        # Only metrics should have been called
        mock_suite.metrics.calculate_clinical_interpretability.assert_called_once()

    def test_multiple_targeted_analyses(self, mock_create_suite):
        """Test targeted validation with multiple specific analyses."""
        mock_suite = Mock()
        mock_suite.metrics = Mock()
        mock_suite.metrics.calculate_clinical_interpretability.return_value = {}
        mock_suite.subtype_analyzer = Mock()
        mock_suite.subtype_analyzer.discover_subtypes.return_value = {}
        mock_create_suite.return_value = mock_suite

        X_list = [np.array([[1, 2]])]
        clinical_data = {"diagnosis": [0]}
        factor_results = {"W": np.array([[0.5]]), "Z": np.array([[1.0]])}

        result = run_targeted_clinical_validation(
            X_list,
            clinical_data,
            factor_results,
            analyses=["metrics", "subtypes"],
        )

        assert result["status"] == "completed"
        assert "clinical_metrics" in result
        assert "subtype_analysis" in result
        assert "progression_analysis" not in result
        assert "biomarker_analysis" not in result

        # Only specified analyses called
        mock_suite.metrics.calculate_clinical_interpretability.assert_called_once()
        mock_suite.subtype_analyzer.discover_subtypes.assert_called_once()

    def test_all_analyses_targeted(self, mock_create_suite):
        """Test targeted validation requesting all analyses."""
        mock_suite = Mock()
        mock_suite.metrics = Mock()
        mock_suite.metrics.calculate_clinical_interpretability.return_value = {}
        mock_suite.subtype_analyzer = Mock()
        mock_suite.subtype_analyzer.discover_subtypes.return_value = {}
        mock_suite.progression_analyzer = Mock()
        mock_suite.progression_analyzer.analyze_disease_progression.return_value = {}
        mock_suite.biomarker_analyzer = Mock()
        mock_suite.biomarker_analyzer.discover_biomarkers.return_value = {}
        mock_create_suite.return_value = mock_suite

        X_list = [np.array([[1, 2]])]
        clinical_data = {"diagnosis": [0]}
        factor_results = {"W": np.array([[0.5]]), "Z": np.array([[1.0]])}

        result = run_targeted_clinical_validation(
            X_list,
            clinical_data,
            factor_results,
            analyses=["metrics", "subtypes", "progression", "biomarkers"],
        )

        assert result["status"] == "completed"
        assert "clinical_metrics" in result
        assert "subtype_analysis" in result
        assert "progression_analysis" in result
        assert "biomarker_analysis" in result

    def test_unknown_analysis_type(self, mock_create_suite):
        """Test targeted validation with unknown analysis type."""
        mock_suite = Mock()
        mock_suite.metrics = Mock()
        mock_suite.metrics.calculate_clinical_interpretability.return_value = {}
        mock_create_suite.return_value = mock_suite

        X_list = [np.array([[1, 2]])]
        clinical_data = {"diagnosis": [0]}
        factor_results = {"W": np.array([[0.5]]), "Z": np.array([[1.0]])}

        result = run_targeted_clinical_validation(
            X_list,
            clinical_data,
            factor_results,
            analyses=["metrics", "unknown_analysis"],
        )

        # Should still complete (skip unknown analysis)
        assert result["status"] == "completed"
        assert "clinical_metrics" in result

    def test_targeted_error_handling(self, mock_create_suite):
        """Test error handling in targeted validation."""
        mock_suite = Mock()
        mock_suite.subtype_analyzer = Mock()
        mock_suite.subtype_analyzer.discover_subtypes.side_effect = RuntimeError(
            "Subtype analysis failed"
        )
        mock_create_suite.return_value = mock_suite

        X_list = [np.array([[1, 2]])]
        clinical_data = {"diagnosis": [0]}
        factor_results = {"W": np.array([[0.5]]), "Z": np.array([[1.0]])}

        result = run_targeted_clinical_validation(
            X_list,
            clinical_data,
            factor_results,
            analyses=["subtypes"],
        )

        assert result["status"] == "failed"
        assert "error" in result


class TestIntegration:
    """Integration tests for clinical validation workflows."""

    @patch('analysis.clinical.integration.ExternalValidator')
    @patch('analysis.clinical.integration.BiomarkerAnalyzer')
    @patch('analysis.clinical.integration.DiseaseProgressionAnalyzer')
    @patch('analysis.clinical.integration.PDSubtypeAnalyzer')
    @patch('analysis.clinical.integration.ClinicalClassifier')
    @patch('analysis.clinical.integration.ClinicalDataProcessor')
    @patch('analysis.clinical.integration.ClinicalMetrics')
    def test_complete_validation_workflow(
        self,
        mock_metrics_class,
        mock_processor_class,
        mock_classifier_class,
        mock_subtype_class,
        mock_progression_class,
        mock_biomarker_class,
        mock_external_class,
    ):
        """Test complete validation workflow from suite creation to execution."""
        # Setup all mock components
        mock_metrics = Mock()
        mock_metrics.calculate_clinical_interpretability.return_value = {
            "interpretability_score": 0.85,
            "clinical_coherence": 0.78,
        }
        mock_metrics_class.return_value = mock_metrics

        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier

        mock_subtype = Mock()
        mock_subtype.discover_subtypes.return_value = {
            "n_subtypes": 3,
            "silhouette_score": 0.65,
            "subtype_labels": [0, 1, 2, 0, 1],
        }
        mock_subtype_class.return_value = mock_subtype

        mock_progression = Mock()
        mock_progression.analyze_disease_progression.return_value = {
            "progression_detected": True,
            "progression_rate": 0.15,
        }
        mock_progression_class.return_value = mock_progression

        mock_biomarker = Mock()
        mock_biomarker.discover_biomarkers.return_value = {
            "n_biomarkers": 5,
            "top_biomarkers": ["factor_1", "factor_3"],
        }
        mock_biomarker_class.return_value = mock_biomarker

        mock_external = Mock()
        mock_external_class.return_value = mock_external

        # Create suite
        suite = create_clinical_validation_suite()

        # Verify suite components
        assert suite.metrics is not None
        assert suite.processor is not None
        assert suite.classifier is not None
        assert suite.subtype_analyzer is not None

        # Prepare data
        X_list = [
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            np.array([[10, 11], [12, 13], [14, 15]]),
        ]
        clinical_data = {
            "diagnosis": [0, 1, 1],
            "age": [55, 62, 58],
            "updrs": [20, 35, 28],
        }
        factor_results = {
            "W": np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]]),
            "Z": np.array([[1.0, 0.5], [2.0, 1.0], [1.5, 0.8]]),
        }

        # Run comprehensive validation
        result = run_comprehensive_clinical_validation(
            X_list, clinical_data, factor_results
        )

        # Verify results
        assert result["status"] == "completed"
        assert result["validation_type"] == "comprehensive_clinical"

        # Check all analyses were executed
        assert "clinical_metrics" in result
        assert result["clinical_metrics"]["interpretability_score"] == 0.85

        assert "subtype_analysis" in result
        assert result["subtype_analysis"]["n_subtypes"] == 3

        assert "progression_analysis" in result
        assert result["progression_analysis"]["progression_detected"] is True

        assert "biomarker_analysis" in result
        assert result["biomarker_analysis"]["n_biomarkers"] == 5

    @patch('analysis.clinical.integration.create_clinical_validation_suite')
    def test_error_recovery_in_workflow(self, mock_create_suite):
        """Test that errors in one analysis don't prevent others from running."""
        # Setup suite with one failing component
        mock_suite = Mock()

        mock_suite.metrics = Mock()
        mock_suite.metrics.calculate_clinical_interpretability.return_value = {
            "score": 0.8
        }

        # This one fails
        mock_suite.subtype_analyzer = Mock()
        mock_suite.subtype_analyzer.discover_subtypes.side_effect = ValueError(
            "Subtype analysis failed"
        )

        mock_suite.progression_analyzer = Mock()
        mock_suite.progression_analyzer.analyze_disease_progression.return_value = {
            "progression": True
        }

        mock_suite.biomarker_analyzer = Mock()
        mock_suite.biomarker_analyzer.discover_biomarkers.return_value = {
            "biomarkers": 3
        }

        mock_create_suite.return_value = mock_suite

        X_list = [np.array([[1, 2]])]
        clinical_data = {"diagnosis": [0]}
        factor_results = {"W": np.array([[0.5]]), "Z": np.array([[1.0]])}

        result = run_comprehensive_clinical_validation(
            X_list, clinical_data, factor_results
        )

        # Should fail overall due to error
        assert result["status"] == "failed"
        assert "error" in result


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch('analysis.clinical.integration.create_clinical_validation_suite')
    def test_empty_data(self, mock_create_suite):
        """Test validation with empty data."""
        mock_suite = Mock()
        mock_suite.metrics = Mock()
        mock_suite.metrics.calculate_clinical_interpretability.side_effect = ValueError(
            "Empty data"
        )
        mock_create_suite.return_value = mock_suite

        result = run_comprehensive_clinical_validation([], {}, {})

        assert result["status"] == "failed"
        assert "error" in result

    @patch('analysis.clinical.integration.create_clinical_validation_suite')
    def test_mismatched_data_shapes(self, mock_create_suite):
        """Test validation with mismatched data shapes."""
        mock_suite = Mock()
        mock_suite.metrics = Mock()
        mock_suite.metrics.calculate_clinical_interpretability.side_effect = ValueError(
            "Shape mismatch"
        )
        mock_create_suite.return_value = mock_suite

        X_list = [np.array([[1, 2]])]
        clinical_data = {"diagnosis": [0, 1, 2]}  # Mismatch: 3 vs 1 sample
        factor_results = {"W": np.array([[0.5]]), "Z": np.array([[1.0]])}

        result = run_comprehensive_clinical_validation(
            X_list, clinical_data, factor_results
        )

        assert result["status"] == "failed"

    @patch('analysis.clinical.integration.create_clinical_validation_suite')
    def test_none_inputs(self, mock_create_suite):
        """Test validation with None inputs."""
        mock_suite = Mock()
        mock_suite.metrics = Mock()
        mock_suite.metrics.calculate_clinical_interpretability.side_effect = TypeError(
            "NoneType"
        )
        mock_create_suite.return_value = mock_suite

        result = run_comprehensive_clinical_validation(None, None, None)

        assert result["status"] == "failed"
        assert "error" in result
