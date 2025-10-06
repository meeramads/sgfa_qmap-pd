"""Tests for analysis.clinical.metrics module."""

from unittest.mock import Mock

import numpy as np
import pytest

from analysis.clinical.metrics import ClinicalMetrics


class TestClinicalMetricsInit:
    """Test ClinicalMetrics initialization."""

    def test_default_initialization(self):
        """Test initialization with defaults."""
        metrics = ClinicalMetrics()

        assert metrics.metrics_list == [
            "accuracy", "precision", "recall", "f1_score", "roc_auc"
        ]
        assert metrics.logger is not None

    def test_custom_metrics_list(self):
        """Test initialization with custom metrics list."""
        custom_metrics = ["accuracy", "precision"]
        metrics = ClinicalMetrics(metrics_list=custom_metrics)

        assert metrics.metrics_list == custom_metrics

    def test_custom_logger(self):
        """Test initialization with custom logger."""
        mock_logger = Mock()
        metrics = ClinicalMetrics(logger=mock_logger)

        assert metrics.logger == mock_logger


class TestCalculateDetailedMetrics:
    """Test calculate_detailed_metrics method."""

    def test_binary_classification_metrics(self):
        """Test metrics calculation for binary classification."""
        metrics_calc = ClinicalMetrics()

        # Perfect predictions
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9]
        ])

        result = metrics_calc.calculate_detailed_metrics(
            y_true, y_pred, y_pred_proba
        )

        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1_score"] == 1.0
        assert "roc_auc" in result
        assert "confusion_matrix" in result

    def test_binary_classification_imperfect(self):
        """Test metrics with imperfect predictions."""
        metrics_calc = ClinicalMetrics()

        y_true = np.array([0, 0, 1, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0])  # 2 errors
        y_pred_proba = np.array([
            [0.8, 0.2],
            [0.4, 0.6],  # FP
            [0.3, 0.7],
            [0.1, 0.9],
            [0.6, 0.4],  # FN
            [0.9, 0.1]
        ])

        result = metrics_calc.calculate_detailed_metrics(
            y_true, y_pred, y_pred_proba
        )

        # 4 correct out of 6
        assert result["accuracy"] == pytest.approx(4/6, rel=1e-5)
        assert 0 <= result["precision"] <= 1
        assert 0 <= result["recall"] <= 1
        assert 0 <= result["f1_score"] <= 1

    def test_multiclass_classification(self):
        """Test metrics for multiclass classification."""
        metrics_calc = ClinicalMetrics()

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        y_pred_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])

        result = metrics_calc.calculate_detailed_metrics(
            y_true, y_pred, y_pred_proba
        )

        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1_score"] == 1.0

    def test_requested_metrics_subset(self):
        """Test calculating only requested metrics."""
        metrics_calc = ClinicalMetrics()

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9]
        ])

        result = metrics_calc.calculate_detailed_metrics(
            y_true, y_pred, y_pred_proba,
            requested_metrics=["accuracy", "precision"]
        )

        assert "accuracy" in result
        assert "precision" in result
        # These should not be calculated
        assert "recall" not in result or result["recall"] is None
        assert "f1_score" not in result or result["f1_score"] is None

    def test_confusion_matrix_binary(self):
        """Test confusion matrix calculation for binary classification."""
        metrics_calc = ClinicalMetrics()

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_pred_proba = np.array([
            [0.9, 0.1],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.6, 0.4]
        ])

        result = metrics_calc.calculate_detailed_metrics(
            y_true, y_pred, y_pred_proba
        )

        cm = np.array(result["confusion_matrix"])
        # TN=1, FP=1, FN=1, TP=1
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 1  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN
        assert cm[1, 1] == 1  # TP

    def test_roc_auc_multiclass(self):
        """Test ROC AUC for multiclass."""
        metrics_calc = ClinicalMetrics()

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        y_pred_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])

        result = metrics_calc.calculate_detailed_metrics(
            y_true, y_pred, y_pred_proba
        )

        assert "roc_auc" in result
        # Perfect classification should give ROC AUC = 1.0
        assert result["roc_auc"] == pytest.approx(1.0, rel=1e-5)

    def test_zero_division_handling(self):
        """Test handling of zero division in metrics."""
        metrics_calc = ClinicalMetrics()

        # Edge case: all same class predicted
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])  # All predict class 0
        y_pred_proba = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.7, 0.3]
        ])

        result = metrics_calc.calculate_detailed_metrics(
            y_true, y_pred, y_pred_proba
        )

        # Should not raise error, should use zero_division=0
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result


class TestAnalyzeClinicalInterpretability:
    """Test analyze_clinical_interpretability method."""

    def test_basic_interpretability_analysis(self):
        """Test basic interpretability analysis."""
        metrics_calc = ClinicalMetrics()

        # Simple factor scores and clinical data
        Z = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [1.0, 0.2]
        ])

        clinical_data = {
            "diagnosis": np.array([0, 0, 1, 1]),
            "age": np.array([50, 55, 60, 65]),
            "updrs": np.array([10, 15, 25, 30])
        }

        result = metrics_calc.analyze_clinical_interpretability(
            Z, clinical_data
        )

        assert "factor_clinical_correlations" in result
        assert "significant_correlations" in result
        assert "interpretability_score" in result

    def test_interpretability_with_dataframe(self):
        """Test interpretability analysis with pandas DataFrame."""
        import pandas as pd

        metrics_calc = ClinicalMetrics()

        Z = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [1.0, 0.2]
        ])

        clinical_df = pd.DataFrame({
            "diagnosis": [0, 0, 1, 1],
            "age": [50, 55, 60, 65],
            "updrs": [10, 15, 25, 30]
        })

        result = metrics_calc.analyze_clinical_interpretability(
            Z, clinical_df
        )

        assert "factor_clinical_correlations" in result
        assert isinstance(result["factor_clinical_correlations"], dict)

    def test_interpretability_score_calculation(self):
        """Test interpretability score is in valid range."""
        metrics_calc = ClinicalMetrics()

        Z = np.array([
            [1.0, 0.0, 0.5],
            [0.5, 0.5, 0.3],
            [0.0, 1.0, 0.8],
            [1.0, 0.2, 0.1]
        ])

        clinical_data = {
            "var1": np.array([10, 15, 20, 25]),
            "var2": np.array([1.0, 1.5, 2.0, 2.5]),
        }

        result = metrics_calc.analyze_clinical_interpretability(
            Z, clinical_data
        )

        # Interpretability score should be between 0 and 1
        assert 0 <= result["interpretability_score"] <= 1

    def test_empty_clinical_data(self):
        """Test with empty clinical data."""
        metrics_calc = ClinicalMetrics()

        Z = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
        ])

        clinical_data = {}

        result = metrics_calc.analyze_clinical_interpretability(
            Z, clinical_data
        )

        # Should handle gracefully
        assert "interpretability_score" in result
        # Score should be 0 or low when no clinical data
        assert result["interpretability_score"] >= 0

    def test_single_factor(self):
        """Test interpretability with single factor."""
        metrics_calc = ClinicalMetrics()

        Z = np.array([
            [1.0],
            [0.5],
            [0.0],
            [0.8]
        ])

        clinical_data = {
            "diagnosis": np.array([0, 0, 1, 1]),
        }

        result = metrics_calc.analyze_clinical_interpretability(
            Z, clinical_data
        )

        assert "factor_clinical_correlations" in result
        assert "interpretability_score" in result


class TestIntegration:
    """Integration tests for ClinicalMetrics."""

    def test_full_workflow(self):
        """Test complete workflow with classification and interpretability."""
        metrics_calc = ClinicalMetrics()

        # Simulate a classification task
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 1])  # One error
        y_pred_proba = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.4, 0.6],
            [0.3, 0.7]
        ])

        # Calculate metrics
        class_metrics = metrics_calc.calculate_detailed_metrics(
            y_true, y_pred, y_pred_proba
        )

        # Verify all expected metrics present
        assert "accuracy" in class_metrics
        assert "precision" in class_metrics
        assert "recall" in class_metrics
        assert "f1_score" in class_metrics
        assert "roc_auc" in class_metrics
        assert "confusion_matrix" in class_metrics

        # Simulate factor scores
        Z = np.random.randn(6, 3)

        clinical_data = {
            "diagnosis": y_true,
            "severity": np.array([10, 12, 25, 30, 15, 28])
        }

        # Calculate interpretability
        interp_results = metrics_calc.analyze_clinical_interpretability(
            Z, clinical_data
        )

        assert "interpretability_score" in interp_results
        assert "factor_clinical_correlations" in interp_results


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_class(self):
        """Test with only one class."""
        metrics_calc = ClinicalMetrics()

        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        y_pred_proba = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0]
        ])

        result = metrics_calc.calculate_detailed_metrics(
            y_true, y_pred, y_pred_proba
        )

        # Should handle gracefully
        assert result["accuracy"] == 1.0

    def test_mismatched_shapes(self):
        """Test error handling for mismatched shapes."""
        metrics_calc = ClinicalMetrics()

        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])  # Wrong shape
        y_pred_proba = np.array([[0.8, 0.2], [0.3, 0.7]])

        # Should raise error or handle gracefully
        with pytest.raises((ValueError, IndexError)):
            metrics_calc.calculate_detailed_metrics(
                y_true, y_pred, y_pred_proba
            )

    def test_nan_handling_in_interpretability(self):
        """Test handling of NaN values in clinical data."""
        metrics_calc = ClinicalMetrics()

        Z = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ])

        clinical_data = {
            "var1": np.array([1.0, np.nan, 3.0]),
            "var2": np.array([10, 20, 30])
        }

        result = metrics_calc.analyze_clinical_interpretability(
            Z, clinical_data
        )

        # Should handle NaNs gracefully
        assert "interpretability_score" in result
        assert not np.isnan(result["interpretability_score"])
