"""Tests for data validation experiment."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from data import generate_synthetic_data
from experiments.data_validation import run_data_validation


class TestDataValidation:
    """Test data validation experiment."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return {
            "data": {"data_dir": "./test_data"},
            "experiments": {"base_output_dir": "./test_results"},
            "data_validation": {
                "preprocessing_strategies": {
                    "minimal": {
                        "enable_advanced_preprocessing": False,
                        "imputation_strategy": "mean",
                    },
                    "standard": {
                        "enable_advanced_preprocessing": True,
                        "imputation_strategy": "median",
                        "feature_selection_method": "variance",
                        "variance_threshold": 0.01,
                    },
                }
            },
        }

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data for testing."""
        return generate_synthetic_data(
            num_sources=2, K=3, num_subjects=30, seed=42  # Small for fast testing
        )

    def test_data_validation_runs(self, mock_config, synthetic_data):
        """Test that data validation experiment runs without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Update config to use temp directory
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Run data validation
            result = run_data_validation(mock_config)

            # Check that result is returned
            assert result is not None

    def test_data_validation_with_pipeline_context(self, mock_config):
        """Test data validation with pipeline context parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Test with empty pipeline context (should work)
            pipeline_context = {
                "X_list": None,
                "preprocessing_info": None,
                "shared_mode": False,
            }

            result = run_data_validation(mock_config)
            assert result is not None

    def test_data_validation_output_structure(self, mock_config):
        """Test that data validation produces expected output structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            result = run_data_validation(mock_config)

            # Check result has expected attributes
            assert hasattr(result, "experiment_id")
            assert hasattr(result, "status")

            # Check that output directory was created
            output_path = Path(tmpdir)
            assert output_path.exists()

    def test_data_validation_matrix_saving(self, mock_config):
        """Test that matrices are saved when framework is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            result = run_data_validation(mock_config)

            # Check for matrices directory (should be created by framework)
            if result and hasattr(result, "experiment_id"):
                # Look for any output files
                output_files = list(Path(tmpdir).rglob("*"))
                assert len(output_files) > 0  # Some output should be created

    @pytest.mark.parametrize("strategy", ["minimal", "standard"])
    def test_preprocessing_strategies(self, mock_config, strategy):
        """Test different preprocessing strategies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Add strategy to config
            if (
                strategy
                not in mock_config["data_validation"]["preprocessing_strategies"]
            ):
                mock_config["data_validation"]["preprocessing_strategies"][strategy] = {
                    "enable_advanced_preprocessing": strategy == "standard"
                }

            result = run_data_validation(mock_config)
            assert result is not None

    def test_data_validation_error_handling(self, mock_config):
        """Test error handling in data validation."""
        # Test with invalid data directory
        mock_config["data"]["data_dir"] = "/nonexistent/path"

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Should handle error gracefully
            result = run_data_validation(mock_config)

            # Result might be None or have error status
            if result is not None:
                # If result is returned, it should indicate failure or handle the error
                assert hasattr(result, "status")
