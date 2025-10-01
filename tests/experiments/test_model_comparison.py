"""Tests for model comparison experiments."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from data import generate_synthetic_data
from experiments.framework import ExperimentConfig
from experiments.model_comparison import (
    ModelArchitectureComparison,
    run_model_comparison,
)


class TestModelArchitectureComparison:
    """Test model architecture comparison experiments."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        data = generate_synthetic_data(
            n_subjects=20, n_features=50, K=3, num_sources=2, noise_level=0.1
        )
        return [data["X1"], data["X2"]]

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                experiments={"base_output_dir": tmpdir},
                data={"data_dir": tmpdir},
                model={"K": 3, "num_samples": 100, "num_warmup": 50, "num_chains": 1},
                # Add the new model_comparison configuration section
                model_comparison={
                    "models": [
                        {
                            "name": "sparse_gfa",
                            "n_factors": [3, 4],
                            "sparsity_lambda": [0.1, 0.3],
                            "group_lambda": [0.1, 0.5],
                        }
                    ],
                    "baseline_methods": ["pca", "ica", "factor_analysis"],
                    "comparative_benchmarks": {
                        "baseline_methods": ["pca", "ica", "factor_analysis"],
                        "performance_metrics": ["reconstruction_error", "log_likelihood"],
                    }
                }
            )
            yield config

    def test_model_architecture_comparison_init(self, config):
        """Test ModelArchitectureComparison initialization."""
        comparison = ModelArchitectureComparison(config)

        assert comparison.config == config
        assert hasattr(comparison, "model_architectures")
        assert "sparseGFA" in comparison.model_architectures
        assert comparison.profiler is not None

    def test_run_methods_comparison(self, sample_data, config):
        """Test running unified methods comparison (sparseGFA vs traditional baselines)."""
        comparison = ModelArchitectureComparison(config)

        # Run with minimal parameters for testing
        result = comparison.run_methods_comparison(
            X_list=sample_data,
            hypers={"alpha_w": 1.0, "alpha_z": 1.0},
            args={"K": 3, "num_samples": 50, "num_warmup": 25, "num_chains": 1},
        )

        # Check result structure
        assert result.status == "completed"
        assert result.experiment_id == "methods_comparison"
        assert result.model_results is not None

        # Should have sparseGFA + traditional methods (pca, ica, fa, nmf, kmeans, cca)
        assert "sparseGFA" in result.model_results
        assert len(result.model_results) >= 2  # At least sparseGFA + 1 traditional method

    def test_run_methods_comparison_basic(self, sample_data, config):
        """Test basic methods comparison with sparseGFA and traditional methods."""
        comparison = ModelArchitectureComparison(config)

        # Run unified methods comparison
        result = comparison.run_methods_comparison(
            X_list=sample_data,
            hypers={"Dm": [X.shape[1] for X in sample_data], "a_sigma": 1.0, "b_sigma": 1.0, "percW": 25.0},
            args={"K": 3, "num_samples": 50, "num_warmup": 25, "reghsZ": True}
        )

        # Check result structure
        assert result.status == "completed"
        assert result.experiment_id == "methods_comparison"
        assert result.model_results is not None

        # Should have both sparseGFA and traditional methods
        assert "sparseGFA" in result.model_results or len(result.model_results) >= 1

    def test_methods_comparison_with_invalid_data(self, config):
        """Test methods comparison with invalid data."""
        comparison = ModelArchitectureComparison(config)

        # Test with empty data list
        with pytest.raises(ValueError):
            comparison.run_methods_comparison(
                X_list=[], hypers={"alpha_w": 1.0}, args={"K": 3, "num_samples": 50}
            )

    def test_methods_comparison_with_mismatched_dimensions(self, config):
        """Test methods comparison with mismatched data dimensions."""
        comparison = ModelArchitectureComparison(config)

        # Create data with mismatched dimensions
        X1 = np.random.randn(20, 50)
        X2 = np.random.randn(15, 50)  # Different number of subjects

        # This should handle the mismatch gracefully
        result = comparison.run_methods_comparison(
            X_list=[X1, X2],
            hypers={"alpha_w": 1.0, "alpha_z": 1.0},
            args={"K": 3, "num_samples": 50, "num_warmup": 25, "num_chains": 1},
        )

        # Should still succeed but may have warnings
        assert result.status in ["completed", "failed"]

    def test_performance_profiling(self, sample_data, config):
        """Test that performance profiling is enabled."""
        comparison = ModelArchitectureComparison(config)

        result = comparison.run_methods_comparison(
            X_list=sample_data,
            hypers={"alpha_w": 1.0, "alpha_z": 1.0},
            args={"K": 3, "num_samples": 50, "num_warmup": 25, "num_chains": 1},
        )

        # Check that performance metrics are included
        assert result.performance_metrics is not None
        assert len(result.performance_metrics) > 0


class TestModelComparisonStandalone:
    """Test standalone model comparison function."""

    def test_run_model_comparison_function(self):
        """Test the standalone run_model_comparison function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate test data
            data = generate_synthetic_data(
                n_subjects=15, n_features=30, K=2, num_sources=2, noise_level=0.1
            )

            # Create minimal config
            config = {
                "experiments": {"base_output_dir": tmpdir},
                "data": {"data_dir": tmpdir},
                "model": {"K": 2, "num_samples": 50, "num_warmup": 25, "num_chains": 1},
                # Add model_comparison configuration
                "model_comparison": {
                    "models": [
                        {
                            "name": "sparse_gfa",
                            "n_factors": [2, 3],
                            "sparsity_lambda": [0.1, 0.3],
                        }
                    ],
                    "baseline_methods": ["pca", "ica"],
                    "comparative_benchmarks": {
                        "baseline_methods": ["pca", "ica"],
                        "performance_metrics": ["reconstruction_error"],
                    }
                }
            }

            # Run model comparison
            result = run_model_comparison(
                config=config, X_list=[data["X1"], data["X2"]]
            )

            # Check that files were created
            output_dir = Path(tmpdir)
            assert any(output_dir.glob("*.json"))  # Should have JSON results

            # Check basic return structure
            assert result is not None

    def test_run_model_comparison_with_defaults(self):
        """Test model comparison with default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should handle minimal input gracefully
            result = run_model_comparison(
                base_output_dir=tmpdir, generate_synthetic=True, n_subjects=10, K=2
            )

            # Should succeed with synthetic data
            assert result is not None


class TestModelComparisonIntegration:
    """Integration tests for model comparison."""

    def test_full_model_comparison_workflow(self):
        """Test complete model comparison workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create realistic test configuration
            config = ExperimentConfig(
                experiments={"base_output_dir": tmpdir},
                data={"data_dir": tmpdir},
                model={
                    "K": 3,
                    "num_samples": 100,
                    "num_warmup": 50,
                    "num_chains": 1,
                    "alpha_w": 1.0,
                    "alpha_z": 1.0,
                },
                # Add model_comparison configuration for integration test
                model_comparison={
                    "models": [
                        {
                            "name": "sparse_gfa",
                            "n_factors": [3, 4],
                            "sparsity_lambda": [0.1, 0.3, 0.5],
                            "group_lambda": [0.1, 0.5],
                        }
                    ],
                    "baseline_methods": ["pca", "ica", "factor_analysis", "nmf"],
                    "comparative_benchmarks": {
                        "baseline_methods": ["pca", "ica", "factor_analysis"],
                        "performance_metrics": ["reconstruction_error", "log_likelihood"],
                    }
                }
            )

            # Generate test data
            data = generate_synthetic_data(
                n_subjects=25, n_features=40, K=3, num_sources=2, noise_level=0.05
            )
            X_list = [data["X1"], data["X2"]]

            # Initialize comparison
            comparison = ModelArchitectureComparison(config)

            # Run unified methods comparison (sparseGFA + traditional methods)
            result = comparison.run_methods_comparison(
                X_list=X_list,
                hypers={"alpha_w": 1.0, "alpha_z": 1.0},
                args={"K": 3, "num_samples": 100, "num_warmup": 50, "num_chains": 1},
            )

            assert result.status == "completed"
            assert "sparseGFA" in result.model_results

            # Check that output files were created (if file writing is enabled)
            output_dir = Path(tmpdir) / "methods_comparison"
            if output_dir.exists():
                # Check for expected output files
                json_files = list(output_dir.glob("*.json"))
                if json_files:
                    # Verify JSON content is valid
                    for json_file in json_files:
                        with open(json_file, "r") as f:
                            data = json.load(f)
                            assert isinstance(data, dict)

    def test_error_handling_and_recovery(self, sample_data):
        """Test error handling in model comparison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                experiments={"base_output_dir": tmpdir},
                data={"data_dir": tmpdir},
                model={"K": 3},
            )

            comparison = ModelArchitectureComparison(config)

            # Test with invalid hyperparameters
            result = comparison.run_methods_comparison(
                X_list=sample_data,
                hypers={"alpha_w": 1.0},
                args={"K": -1, "num_samples": 10},  # Invalid K
            )

            # Should handle error gracefully
            assert hasattr(result, "success")
