"""Tests for ModelRunner."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from analysis.model_runner import ModelRunner, ModelRunnerConfig


@pytest.mark.unit
@pytest.mark.model
class TestModelRunner:
    """Test ModelRunner functionality."""

    def test_model_runner_init(self, sample_config):
        """Test ModelRunner initialization."""
        runner = ModelRunner(ModelRunnerConfig.from_object(sample_config))

        assert runner.config == sample_config

    def test_run_standard_analysis_single_run(
        self, sample_config, sample_synthetic_data, sample_hyperparameters
    ):
        """Test standard analysis with single run."""
        sample_config.num_runs = 1
        runner = ModelRunner(ModelRunnerConfig.from_object(sample_config))

        mock_run_results = {
            "Z": np.random.normal(0, 1, (100, 50, 3)),
            "W": np.random.normal(0, 1, (100, 45, 3)),
            "log_likelihood": np.random.normal(-1000, 100, 100),
        }

        with patch.object(runner, "_run_single_mcmc", return_value=mock_run_results):
            results = runner.run_standard_analysis(
                sample_synthetic_data["X_list"],
                sample_hyperparameters,
                sample_synthetic_data,
            )

            assert "run_1" in results
            assert results["run_1"] == mock_run_results

    def test_run_standard_analysis_multiple_runs(
        self, sample_config, sample_synthetic_data, sample_hyperparameters
    ):
        """Test standard analysis with multiple runs."""
        sample_config.num_runs = 3
        runner = ModelRunner(ModelRunnerConfig.from_object(sample_config))

        mock_results = [
            {"Z": np.random.normal(0, 1, (100, 50, 3)), "run_id": 1},
            {"Z": np.random.normal(0, 1, (100, 50, 3)), "run_id": 2},
            {"Z": np.random.normal(0, 1, (100, 50, 3)), "run_id": 3},
        ]

        with patch.object(runner, "_run_single_mcmc", side_effect=mock_results):
            results = runner.run_standard_analysis(
                sample_synthetic_data["X_list"],
                sample_hyperparameters,
                sample_synthetic_data,
            )

            assert len(results) == 3
            assert "run_1" in results
            assert "run_2" in results
            assert "run_3" in results

    @patch("analysis.model_runner.logger")
    def test_run_standard_analysis_with_failure(
        self, mock_logger, sample_config, sample_synthetic_data, sample_hyperparameters
    ):
        """Test standard analysis when one run fails."""
        sample_config.num_runs = 2
        runner = ModelRunner(ModelRunnerConfig.from_object(sample_config))

        # First run succeeds, second fails
        mock_results = [
            {"Z": np.random.normal(0, 1, (100, 50, 3))},
            Exception("MCMC failed"),
        ]

        def side_effect(*args, **kwargs):
            result = mock_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        with patch.object(runner, "_run_single_mcmc", side_effect=side_effect):
            results = runner.run_standard_analysis(
                sample_synthetic_data["X_list"],
                sample_hyperparameters,
                sample_synthetic_data,
            )

            # Should have one successful run
            assert "run_1" in results
            assert len(results) == 1

            # Should have logged the error
            assert mock_logger.error.called

    def test_run_single_mcmc_mock(
        self,
        sample_config,
        sample_synthetic_data,
        sample_hyperparameters,
        mock_mcmc_results,
    ):
        """Test single MCMC run with mocked inference."""
        sample_config.num_samples = 500
        sample_config.num_chains = 2
        sample_config.num_warmup = 100

        runner = ModelRunner(ModelRunnerConfig.from_object(sample_config))

        # Mock the MCMC and inference components
        mock_mcmc = Mock()
        mock_mcmc.get_samples.return_value = mock_mcmc_results

        with patch("analysis.model_runner.MCMC", return_value=mock_mcmc), patch(
            "analysis.model_runner.NUTS"
        ), patch("run_analysis.run_inference") as mock_inference:

            mock_inference.return_value = mock_mcmc_results

            results = runner._run_single_mcmc(
                1,
                sample_synthetic_data["X_list"],
                sample_hyperparameters,
                sample_synthetic_data,
            )

            # Should return the mocked results
            assert "Z" in results
            assert "W" in results
            assert "log_likelihood" in results

    def test_extract_robust_parameters_success(self, sample_config, mock_mcmc_results):
        """Test robust parameter extraction from MCMC results."""
        runner = ModelRunner(ModelRunnerConfig.from_object(sample_config))

        extracted = runner._extract_robust_parameters(mock_mcmc_results)

        assert extracted["extraction_successful"] is True
        assert "Z_mean" in extracted
        assert "W_mean" in extracted
        assert "sigma_mean" in extracted

        # Check shapes are reasonable
        assert extracted["Z_mean"].shape == (50, 3)  # subjects x factors
        assert extracted["W_mean"].shape == (45, 3)  # features x factors
        assert extracted["sigma_mean"].shape == (3,)  # views

    def test_extract_robust_parameters_missing_data(self, sample_config):
        """Test parameter extraction with missing data."""
        runner = ModelRunner(ModelRunnerConfig.from_object(sample_config))

        incomplete_results = {
            "Z": np.random.normal(0, 1, (100, 50, 3))
        }  # Missing W, sigma

        extracted = runner._extract_robust_parameters(incomplete_results)

        assert extracted["extraction_successful"] is False
        assert "missing_parameters" in extracted["reason"]

    def test_extract_robust_parameters_invalid_shapes(self, sample_config):
        """Test parameter extraction with invalid shapes."""
        runner = ModelRunner(ModelRunnerConfig.from_object(sample_config))

        invalid_results = {
            "Z": np.random.normal(
                0, 1, (10, 5)
            ),  # Wrong shape (missing factor dimension)
            "W": np.random.normal(0, 1, (100, 45, 3)),
            "sigma": np.random.gamma(2, 1, (100, 3)),
        }

        extracted = runner._extract_robust_parameters(invalid_results)

        assert extracted["extraction_successful"] is False
        assert "shape_error" in extracted["reason"]

    @patch("analysis.model_runner.logger")
    def test_extract_robust_parameters_exception(self, mock_logger, sample_config):
        """Test parameter extraction with exception."""
        runner = ModelRunner(ModelRunnerConfig.from_object(sample_config))

        # Create results that will cause an exception during processing
        bad_results = {"Z": "not_an_array"}

        extracted = runner._extract_robust_parameters(bad_results)

        assert extracted["extraction_successful"] is False
        assert "extraction_error" in extracted["reason"]
        mock_logger.error.assert_called()


@pytest.mark.integration
@pytest.mark.slow
class TestModelRunnerIntegration:
    """Integration tests for ModelRunner (slower, more complete tests)."""

    def test_full_analysis_pipeline(
        self, sample_config, sample_synthetic_data, sample_hyperparameters
    ):
        """Test complete analysis pipeline."""
        # Use minimal settings for faster test
        sample_config.num_runs = 1
        sample_config.num_samples = 100
        sample_config.num_chains = 1

        runner = ModelRunner(ModelRunnerConfig.from_object(sample_config))

        # Mock heavy computation components
        with patch("run_analysis.run_inference") as mock_inference:
            mock_inference.return_value = {
                "Z": np.random.normal(0, 1, (100, 50, 3)),
                "W": np.random.normal(0, 1, (100, 45, 3)),
                "sigma": np.random.gamma(2, 1, (100, 3)),
                "log_likelihood": np.random.normal(-1000, 50, 100),
            }

            results = runner.run_standard_analysis(
                sample_synthetic_data["X_list"],
                sample_hyperparameters,
                sample_synthetic_data,
            )

            assert len(results) == 1
            assert "run_1" in results
            run_result = results["run_1"]

            # Check that extracted parameters are present
            assert "Z" in run_result
            assert "W" in run_result
            assert "sigma" in run_result
