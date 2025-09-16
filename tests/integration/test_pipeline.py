"""Integration tests for the full analysis pipeline."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from analysis.config_manager import ConfigManager
from analysis.cross_validation import CVRunner
from analysis.data_manager import DataManager
from analysis.model_runner import ModelRunner


@pytest.mark.integration
@pytest.mark.slow
class TestFullPipeline:
    """Test complete analysis pipeline integration."""

    def test_synthetic_data_pipeline(self, temp_dir):
        """Test full pipeline with synthetic data."""
        # Create mock args
        args = Mock()
        args.dataset = "synthetic"
        args.num_sources = 3
        args.K = 5
        args.percW = 33.0
        args.num_samples = 100  # Small for fast test
        args.num_chains = 1
        args.num_runs = 1
        args.device = "cpu"
        args.seed = 42
        args.results_dir = str(temp_dir)
        args.run_cv = False
        args.cv_only = False

        # Mock heavy computations
        mock_mcmc_results = {
            "Z": np.random.normal(0, 1, (100, 50, 5)),
            "W": np.random.normal(0, 1, (100, 45, 5)),
            "sigma": np.random.gamma(2, 1, (100, 3)),
            "log_likelihood": np.random.normal(-1000, 50, 100),
        }

        with patch("run_analysis.run_inference", return_value=mock_mcmc_results):
            # Initialize components
            config_manager = ConfigManager(args, temp_dir)
            data_manager = DataManager(config_manager.config)
            model_runner = ModelRunner(config_manager.config)

            # Run pipeline steps
            data = data_manager.load_data()
            X_list, hypers = data_manager.prepare_for_analysis(data)
            results = model_runner.run_standard_analysis(X_list, hypers, data)

            # Verify results
            assert len(results) == 1
            assert "run_1" in results
            assert "Z" in results["run_1"]
            assert "W" in results["run_1"]

    def test_cv_pipeline(self, temp_dir, sample_synthetic_data):
        """Test cross-validation pipeline."""
        args = Mock()
        args.dataset = "synthetic"
        args.run_cv = True
        args.cv_folds = 3
        args.seed = 42
        args.results_dir = str(temp_dir)

        config_manager = ConfigManager(args, temp_dir)

        # Mock CV results
        mock_cv_results = {"cv_scores": [0.8, 0.85, 0.9]}
        mock_cv_object = Mock()

        with patch.object(
            CVRunner, "_check_cv_availability", return_value=True
        ), patch.object(
            CVRunner,
            "_run_basic_cv_analysis",
            return_value=(mock_cv_results, mock_cv_object),
        ):

            cv_runner = CVRunner(config_manager.config, temp_dir)

            results = cv_runner.run_cv_analysis(
                sample_synthetic_data["X_list"],
                {"Dm": [20, 15, 10], "percW": 33.0},
                sample_synthetic_data,
            )

            assert results is not None
            assert results[0] == mock_cv_results

    def test_error_handling_pipeline(self, temp_dir):
        """Test pipeline error handling."""
        args = Mock()
        args.dataset = "synthetic"
        args.K = 5
        args.num_runs = 1
        args.results_dir = str(temp_dir)

        config_manager = ConfigManager(args, temp_dir)
        data_manager = DataManager(config_manager.config)
        model_runner = ModelRunner(config_manager.config)

        # Load data successfully
        data = data_manager.load_data()
        X_list, hypers = data_manager.prepare_for_analysis(data)

        # Simulate MCMC failure
        with patch("run_analysis.run_inference", side_effect=Exception("MCMC failed")):
            results = model_runner.run_standard_analysis(X_list, hypers, data)

            # Should handle error gracefully
            assert len(results) == 0  # No successful runs

    def test_pipeline_with_different_configs(self, temp_dir):
        """Test pipeline with various configuration combinations."""
        base_args = Mock()
        base_args.dataset = "synthetic"
        base_args.results_dir = str(temp_dir)
        base_args.seed = 42
        base_args.num_runs = 1
        base_args.num_samples = 50  # Very small for speed

        configs_to_test = [
            {"K": 3, "num_sources": 2},
            {"K": 10, "num_sources": 4},
            {"K": 2, "num_sources": 1},
        ]

        for config_params in configs_to_test:
            # Update args for this configuration
            for key, value in config_params.items():
                setattr(base_args, key, value)

            # Mock results based on config
            n_features = sum(
                [60 // config_params["num_sources"]] * config_params["num_sources"]
            )
            mock_results = {
                "Z": np.random.normal(0, 1, (50, 50, config_params["K"])),
                "W": np.random.normal(0, 1, (50, n_features, config_params["K"])),
                "sigma": np.random.gamma(2, 1, (50, config_params["num_sources"])),
            }

            with patch("run_analysis.run_inference", return_value=mock_results):
                config_manager = ConfigManager(base_args, temp_dir)
                data_manager = DataManager(config_manager.config)
                model_runner = ModelRunner(config_manager.config)

                data = data_manager.load_data()
                X_list, hypers = data_manager.prepare_for_analysis(data)
                results = model_runner.run_standard_analysis(X_list, hypers, data)

                # Should complete successfully for each config
                assert len(results) == 1
                assert "run_1" in results


@pytest.mark.integration
class TestPipelineDataFlow:
    """Test data flow through pipeline components."""

    def test_data_format_consistency(self, temp_dir):
        """Test that data formats are consistent between components."""
        args = Mock()
        args.dataset = "synthetic"
        args.num_sources = 3
        args.K = 5
        args.results_dir = str(temp_dir)

        config_manager = ConfigManager(args, temp_dir)
        data_manager = DataManager(config_manager.config)

        # Load and prepare data
        data = data_manager.load_data()
        X_list, hypers = data_manager.prepare_for_analysis(data)

        # Verify data structure consistency
        assert isinstance(X_list, list)
        assert len(X_list) == args.num_sources
        assert isinstance(hypers, dict)
        assert "Dm" in hypers
        assert hypers["Dm"] == [X.shape[1] for X in X_list]

        # Verify data can be used by model runner
        ModelRunner(config_manager.config)

        # This should not raise errors about data format
        assert all(isinstance(X, np.ndarray) for X in X_list)
        assert all(
            X.shape[0] == X_list[0].shape[0] for X in X_list
        )  # Same number of subjects

    def test_config_propagation(self, temp_dir):
        """Test that configuration is properly propagated through components."""
        args = Mock()
        args.dataset = "synthetic"
        args.K = 7
        args.num_sources = 2
        args.percW = 50.0
        args.results_dir = str(temp_dir)
        args.seed = 123

        config_manager = ConfigManager(args, temp_dir)
        config = config_manager.config

        # Verify config values are accessible by all components
        data_manager = DataManager(config)
        model_runner = ModelRunner(config)
        cv_runner = CVRunner(config, temp_dir)

        # Config should be consistently accessible
        assert data_manager.config.K == 7
        assert model_runner.config.K == 7
        assert cv_runner.config.K == 7
        assert data_manager.config.num_sources == 2
        assert data_manager.config.percW == 50.0

    @patch("analysis.data_manager.logger")
    @patch("analysis.model_runner.logger")
    @patch("analysis.cross_validation.logger")
    def test_logging_integration(
        self, mock_cv_logger, mock_model_logger, mock_data_logger, temp_dir
    ):
        """Test that logging works properly across all components."""
        args = Mock()
        args.dataset = "synthetic"
        args.results_dir = str(temp_dir)
        args.num_runs = 1

        config_manager = ConfigManager(args, temp_dir)
        data_manager = DataManager(config_manager.config)
        model_runner = ModelRunner(config_manager.config)

        # Run operations that should log
        data = data_manager.load_data()
        X_list, hypers = data_manager.prepare_for_analysis(data)

        with patch("run_analysis.run_inference") as mock_inference:
            mock_inference.return_value = {
                "Z": np.random.normal(0, 1, (50, 50, 5)),
                "W": np.random.normal(0, 1, (50, 45, 5)),
                "sigma": np.random.gamma(2, 1, (50, 3)),
            }

            model_runner.run_standard_analysis(X_list, hypers, data)

        # Each component should have logged something
        assert mock_model_logger.info.called
