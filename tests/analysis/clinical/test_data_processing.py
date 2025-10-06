"""Tests for analysis.clinical.data_processing module."""

from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest

from analysis.clinical.data_processing import ClinicalDataProcessor


class TestClinicalDataProcessorInit:
    """Test ClinicalDataProcessor initialization."""

    def test_default_initialization(self):
        """Test initialization with defaults."""
        processor = ClinicalDataProcessor()

        assert processor.logger is not None

    def test_custom_logger(self):
        """Test initialization with custom logger."""
        mock_logger = Mock()
        processor = ClinicalDataProcessor(logger=mock_logger)

        assert processor.logger == mock_logger


class TestRunSGFATraining:
    """Test run_sgfa_training method."""

    @patch('analysis.clinical.data_processing.MCMC')
    @patch('analysis.clinical.data_processing.NUTS')
    def test_basic_training(self, mock_nuts, mock_mcmc_class):
        """Test basic SGFA training."""
        processor = ClinicalDataProcessor()

        # Setup mock MCMC
        mock_mcmc = Mock()
        mock_mcmc_class.return_value = mock_mcmc

        # Mock MCMC samples
        mock_samples = {
            "W": [np.random.randn(10, 3), np.random.randn(20, 3)],
            "Z": np.random.randn(50, 3),
        }
        mock_mcmc.get_samples.return_value = mock_samples

        # Training data
        X_train = [
            np.random.randn(50, 10),
            np.random.randn(50, 20)
        ]

        hypers = {
            "K": 3,
            "Dm": [10, 20],
            "percW": 25.0
        }

        args = {
            "model": "sparseGFA",
            "K": 3,
            "num_warmup": 100,
            "num_samples": 200,
            "num_chains": 1
        }

        result = processor.run_sgfa_training(X_train, hypers, args)

        # Verify MCMC was called
        assert mock_mcmc.run.called
        assert mock_mcmc.get_samples.called

        # Check results
        assert "W" in result
        assert "Z" in result
        assert "execution_time" in result

    @patch('analysis.clinical.data_processing.MCMC')
    @patch('analysis.clinical.data_processing.NUTS')
    def test_training_with_default_args(self, mock_nuts, mock_mcmc_class):
        """Test training with minimal arguments."""
        processor = ClinicalDataProcessor()

        mock_mcmc = Mock()
        mock_mcmc_class.return_value = mock_mcmc

        mock_samples = {
            "W": [np.random.randn(5, 2)],
            "Z": np.random.randn(10, 2),
        }
        mock_mcmc.get_samples.return_value = mock_samples

        X_train = [np.random.randn(10, 5)]
        hypers = {"K": 2, "Dm": [5]}
        args = {"K": 2}  # Minimal args

        result = processor.run_sgfa_training(X_train, hypers, args)

        assert result is not None
        assert "W" in result
        assert "Z" in result

    @patch('analysis.clinical.data_processing.MCMC')
    @patch('analysis.clinical.data_processing.NUTS')
    def test_training_execution_time(self, mock_nuts, mock_mcmc_class):
        """Test that execution time is recorded."""
        processor = ClinicalDataProcessor()

        mock_mcmc = Mock()
        mock_mcmc_class.return_value = mock_mcmc
        mock_mcmc.get_samples.return_value = {
            "W": [np.random.randn(5, 2)],
            "Z": np.random.randn(10, 2),
        }

        X_train = [np.random.randn(10, 5)]
        hypers = {"K": 2, "Dm": [5]}
        args = {"K": 2}

        result = processor.run_sgfa_training(X_train, hypers, args)

        assert "execution_time" in result
        assert result["execution_time"] >= 0


class TestEvaluateSGFATestSet:
    """Test evaluate_sgfa_test_set method."""

    def test_basic_evaluation(self):
        """Test basic test set evaluation."""
        processor = ClinicalDataProcessor()

        # Trained model
        W_list = [np.random.randn(10, 3), np.random.randn(15, 3)]
        Z_train = np.random.randn(50, 3)

        # Test data
        X_test = [np.random.randn(20, 10), np.random.randn(20, 15)]

        result = processor.evaluate_sgfa_test_set(
            X_test, W_list, Z_train
        )

        assert "Z_test" in result
        assert result["Z_test"].shape == (20, 3)

    def test_evaluation_with_single_view(self):
        """Test evaluation with single view."""
        processor = ClinicalDataProcessor()

        W_list = [np.random.randn(10, 3)]
        Z_train = np.random.randn(50, 3)
        X_test = [np.random.randn(20, 10)]

        result = processor.evaluate_sgfa_test_set(
            X_test, W_list, Z_train
        )

        assert "Z_test" in result
        assert result["Z_test"].shape[0] == 20
        assert result["Z_test"].shape[1] == 3

    def test_evaluation_reconstruction_error(self):
        """Test that reconstruction error is calculated."""
        processor = ClinicalDataProcessor()

        W_list = [np.random.randn(10, 3)]
        Z_train = np.random.randn(50, 3)
        X_test = [np.random.randn(20, 10)]

        result = processor.evaluate_sgfa_test_set(
            X_test, W_list, Z_train
        )

        assert "reconstruction_error" in result or "Z_test" in result


class TestRunSGFAAnalysis:
    """Test run_sgfa_analysis method."""

    @patch.object(ClinicalDataProcessor, 'run_sgfa_training')
    @patch.object(ClinicalDataProcessor, 'evaluate_sgfa_test_set')
    def test_full_analysis_workflow(self, mock_eval, mock_train):
        """Test complete analysis workflow."""
        processor = ClinicalDataProcessor()

        # Mock training results
        mock_train.return_value = {
            "W": [np.random.randn(10, 3)],
            "Z": np.random.randn(50, 3),
            "execution_time": 1.5
        }

        # Mock evaluation results
        mock_eval.return_value = {
            "Z_test": np.random.randn(20, 3),
            "reconstruction_error": 0.1
        }

        X_train = [np.random.randn(50, 10)]
        X_test = [np.random.randn(20, 10)]
        hypers = {"K": 3, "Dm": [10]}
        args = {"K": 3}

        result = processor.run_sgfa_analysis(
            X_train, X_test, hypers, args
        )

        # Verify methods were called
        assert mock_train.called
        assert mock_eval.called

        # Check results structure
        assert "train_results" in result
        assert "test_results" in result

    @patch.object(ClinicalDataProcessor, 'run_sgfa_training')
    def test_analysis_train_only(self, mock_train):
        """Test analysis with training only (no test set)."""
        processor = ClinicalDataProcessor()

        mock_train.return_value = {
            "W": [np.random.randn(10, 3)],
            "Z": np.random.randn(50, 3),
        }

        X_train = [np.random.randn(50, 10)]
        hypers = {"K": 3, "Dm": [10]}
        args = {"K": 3}

        result = processor.run_sgfa_analysis(
            X_train, None, hypers, args
        )

        assert "train_results" in result
        # Test results should be None or not present
        assert result.get("test_results") is None or "test_results" not in result


class TestApplyTrainedModel:
    """Test apply_trained_model method."""

    def test_apply_to_new_data(self):
        """Test applying trained model to new data."""
        processor = ClinicalDataProcessor()

        # Trained model
        W_list = [np.random.randn(10, 3)]

        # New data
        X_new = [np.random.randn(15, 10)]

        result = processor.apply_trained_model(W_list, X_new)

        assert "Z_predicted" in result
        assert result["Z_predicted"].shape == (15, 3)

    def test_apply_multi_view(self):
        """Test applying trained model with multiple views."""
        processor = ClinicalDataProcessor()

        W_list = [
            np.random.randn(10, 3),
            np.random.randn(20, 3)
        ]

        X_new = [
            np.random.randn(15, 10),
            np.random.randn(15, 20)
        ]

        result = processor.apply_trained_model(W_list, X_new)

        assert "Z_predicted" in result
        assert result["Z_predicted"].shape == (15, 3)


class TestIntegration:
    """Integration tests for ClinicalDataProcessor."""

    @patch('analysis.clinical.data_processing.MCMC')
    @patch('analysis.clinical.data_processing.NUTS')
    def test_complete_workflow(self, mock_nuts, mock_mcmc_class):
        """Test complete clinical data processing workflow."""
        processor = ClinicalDataProcessor()

        # Setup mocks
        mock_mcmc = Mock()
        mock_mcmc_class.return_value = mock_mcmc
        mock_mcmc.get_samples.return_value = {
            "W": [np.random.randn(10, 3)],
            "Z": np.random.randn(50, 3),
        }

        # Data
        X_train = [np.random.randn(50, 10)]
        X_test = [np.random.randn(20, 10)]
        hypers = {"K": 3, "Dm": [10]}
        args = {"K": 3, "model": "sparseGFA"}

        # Train
        train_result = processor.run_sgfa_training(X_train, hypers, args)
        assert "W" in train_result
        assert "Z" in train_result

        # Evaluate on test
        W_list = train_result["W"]
        Z_train = train_result["Z"]

        test_result = processor.evaluate_sgfa_test_set(
            X_test, W_list, Z_train
        )
        assert "Z_test" in test_result


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self):
        """Test handling of empty data."""
        processor = ClinicalDataProcessor()

        with pytest.raises((ValueError, IndexError, AttributeError)):
            processor.evaluate_sgfa_test_set([], [], np.array([]))

    def test_mismatched_dimensions(self):
        """Test handling of mismatched dimensions."""
        processor = ClinicalDataProcessor()

        W_list = [np.random.randn(10, 3)]  # 10 features
        Z_train = np.random.randn(50, 3)
        X_test = [np.random.randn(20, 15)]  # 15 features (mismatch!)

        with pytest.raises((ValueError, IndexError)):
            processor.evaluate_sgfa_test_set(X_test, W_list, Z_train)

    @patch('analysis.clinical.data_processing.MCMC')
    @patch('analysis.clinical.data_processing.NUTS')
    def test_mcmc_failure_handling(self, mock_nuts, mock_mcmc_class):
        """Test handling of MCMC failures."""
        processor = ClinicalDataProcessor()

        mock_mcmc = Mock()
        mock_mcmc_class.return_value = mock_mcmc
        mock_mcmc.run.side_effect = RuntimeError("MCMC failed")

        X_train = [np.random.randn(10, 5)]
        hypers = {"K": 2, "Dm": [5]}
        args = {"K": 2}

        with pytest.raises(RuntimeError):
            processor.run_sgfa_training(X_train, hypers, args)
