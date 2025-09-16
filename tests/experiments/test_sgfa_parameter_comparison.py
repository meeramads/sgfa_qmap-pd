"""Tests for SGFA parameter comparison experiments."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from data import generate_synthetic_data
from experiments.framework import ExperimentConfig
from experiments.sgfa_parameter_comparison import (
    SGFAParameterComparison,
    run_method_comparison,
)


class TestSGFAParameterComparison:
    """Test SGFA parameter comparison experiments."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        data = generate_synthetic_data(
            n_subjects=25, n_features=60, K=4, num_sources=3, noise_level=0.1
        )
        return [data["X1"], data["X2"], data["X3"]]

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                experiments={"base_output_dir": tmpdir},
                data={"data_dir": tmpdir},
                model={
                    "K": 4,
                    "num_samples": 100,
                    "num_warmup": 50,
                    "num_chains": 1,
                    "alpha_w": 1.0,
                    "alpha_z": 1.0,
                },
            )
            yield config

    def test_sgfa_parameter_comparison_init(self, config):
        """Test SGFAParameterComparison initialization."""
        comparison = SGFAParameterComparison(config)

        assert comparison.config == config
        assert hasattr(comparison, "sgfa_variants")
        assert "standard" in comparison.sgfa_variants
        assert "sparse_only" in comparison.sgfa_variants
        assert "group_only" in comparison.sgfa_variants
        assert comparison.profiler is not None

    def test_run_sgfa_variant_comparison(self, sample_data, config):
        """Test running SGFA variant comparison."""
        comparison = SGFAParameterComparison(config)

        # Run with minimal parameters for testing
        result = comparison.run_sgfa_variant_comparison(
            X_list=sample_data,
            hypers={"K": 4, "alpha_w": 1.0, "alpha_z": 1.0},
            args={"num_samples": 50, "num_warmup": 25, "num_chains": 1},
        )

        # Check result structure
        assert result.success is True
        assert result.experiment_name == "sgfa_variant_comparison"
        assert result.data is not None
        assert "variant_results" in result.data
        assert "performance_comparison" in result.data

        # Check that variants were tested
        variant_results = result.data["variant_results"]
        assert len(variant_results) >= 1
        assert any("standard" in str(variant_results))

    def test_run_traditional_method_comparison(self, sample_data, config):
        """Test running traditional method comparison."""
        comparison = SGFAParameterComparison(config)

        # Mock SGFA results for comparison
        sgfa_results = {
            "log_likelihood": -800.0,
            "W_mean": np.random.randn(60, 4),
            "Z_mean": np.random.randn(25, 4),
            "reconstruction_error": 0.15,
        }

        # Run traditional method comparison
        result = comparison.run_traditional_method_comparison(
            X_list=sample_data, sgfa_results=sgfa_results
        )

        # Check result structure
        assert result.success is True
        assert result.experiment_name == "traditional_method_comparison"
        assert result.data is not None
        assert "method_results" in result.data

    def test_run_multiview_capability_assessment(self, sample_data, config):
        """Test running multiview capability assessment."""
        comparison = SGFAParameterComparison(config)

        # Run multiview assessment
        result = comparison.run_multiview_capability_assessment(X_list=sample_data)

        # Check result structure
        assert result.success is True
        assert result.experiment_name == "multiview_capability_assessment"
        assert result.data is not None
        assert "multiview_results" in result.data

        # Should test different numbers of views
        multiview_results = result.data["multiview_results"]
        assert len(multiview_results) >= 1

    def test_run_scalability_comparison(self, sample_data, config):
        """Test running scalability comparison."""
        comparison = SGFAParameterComparison(config)

        # Define test sample and feature sizes
        sample_sizes = [10, 20]
        feature_sizes = [30, 50]

        # Run scalability comparison
        result = comparison.run_scalability_comparison(
            X_list=sample_data, sample_sizes=sample_sizes, feature_sizes=feature_sizes
        )

        # Check result structure
        assert result.success is True
        assert result.experiment_name == "scalability_comparison"
        assert result.data is not None
        assert "scalability_results" in result.data

        # Should have results for different sizes
        scalability_results = result.data["scalability_results"]
        assert len(scalability_results) >= 1

    def test_sgfa_variant_comparison_with_invalid_data(self, config):
        """Test SGFA variant comparison with invalid data."""
        comparison = SGFAParameterComparison(config)

        # Test with empty data list
        with pytest.raises(ValueError):
            comparison.run_sgfa_variant_comparison(
                X_list=[], hypers={"K": 4}, args={"num_samples": 50}
            )

    def test_sgfa_variant_comparison_with_invalid_hyperparameters(
        self, sample_data, config
    ):
        """Test SGFA variant comparison with invalid hyperparameters."""
        comparison = SGFAParameterComparison(config)

        # Test with invalid K value
        with pytest.raises(ValueError):
            comparison.run_sgfa_variant_comparison(
                X_list=sample_data,
                hypers={"K": 0},  # Invalid K
                args={"num_samples": 50},
            )

    def test_scalability_comparison_parameter_validation(self, sample_data, config):
        """Test scalability comparison parameter validation."""
        comparison = SGFAParameterComparison(config)

        # Test with invalid sample sizes
        with pytest.raises(ValueError):
            comparison.run_scalability_comparison(
                X_list=sample_data,
                sample_sizes=[],  # Empty list
                feature_sizes=[30, 50],
            )

        # Test with invalid feature sizes
        with pytest.raises(ValueError):
            comparison.run_scalability_comparison(
                X_list=sample_data,
                sample_sizes=[10, 20],
                feature_sizes=[],  # Empty list
            )

    def test_performance_profiling(self, sample_data, config):
        """Test that performance profiling is enabled."""
        comparison = SGFAParameterComparison(config)

        result = comparison.run_sgfa_variant_comparison(
            X_list=sample_data,
            hypers={"K": 4, "alpha_w": 1.0, "alpha_z": 1.0},
            args={"num_samples": 50, "num_warmup": 25, "num_chains": 1},
        )

        # Check that performance metrics are included
        assert result.performance_metrics is not None
        assert (
            "timing" in result.performance_metrics
            or "memory" in result.performance_metrics
        )


class TestSGFAParameterComparisonStandalone:
    """Test standalone SGFA parameter comparison function."""

    def test_run_method_comparison_function(self):
        """Test the standalone run_method_comparison function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate test data
            data = generate_synthetic_data(
                n_subjects=20, n_features=40, K=3, num_sources=2, noise_level=0.1
            )

            # Create minimal config
            config = {
                "experiments": {"base_output_dir": tmpdir},
                "data": {"data_dir": tmpdir},
                "model": {"K": 3, "num_samples": 50, "num_warmup": 25, "num_chains": 1},
            }

            # Run method comparison
            result = run_method_comparison(config)

            # Check that function completes
            assert result is not None

    def test_run_method_comparison_with_custom_data(self):
        """Test method comparison with custom data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate custom test data
            np.random.randn(15, 25)
            np.random.randn(15, 30)

            config = {
                "experiments": {"base_output_dir": tmpdir},
                "data": {"data_dir": tmpdir},
                "model": {"K": 2, "num_samples": 50, "num_warmup": 25},
            }

            # Should handle custom data
            result = run_method_comparison(config)
            assert result is not None


class TestSGFAParameterComparisonIntegration:
    """Integration tests for SGFA parameter comparison."""

    def test_full_sgfa_parameter_comparison_workflow(self):
        """Test complete SGFA parameter comparison workflow."""
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
            )

            # Generate test data
            data = generate_synthetic_data(
                n_subjects=20, n_features=35, K=3, num_sources=2, noise_level=0.05
            )
            X_list = [data["X1"], data["X2"]]

            # Initialize comparison
            comparison = SGFAParameterComparison(config)

            # Run SGFA variant comparison
            variant_result = comparison.run_sgfa_variant_comparison(
                X_list=X_list,
                hypers={"K": 3, "alpha_w": 1.0, "alpha_z": 1.0},
                args={"num_samples": 100, "num_warmup": 50, "num_chains": 1},
            )

            assert variant_result.success is True

            # Run multiview capability assessment
            multiview_result = comparison.run_multiview_capability_assessment(
                X_list=X_list
            )

            assert multiview_result.success is True

            # Run scalability comparison with small sizes
            scalability_result = comparison.run_scalability_comparison(
                X_list=X_list, sample_sizes=[10, 15], feature_sizes=[20, 25]
            )

            assert scalability_result.success is True

            # Check that output files were created
            output_dir = Path(tmpdir)
            json_files = list(output_dir.glob("**/*.json"))
            assert len(json_files) >= 1

    def test_parameter_comparison_with_different_k_values(self):
        """Test parameter comparison across different K values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                experiments={"base_output_dir": tmpdir},
                data={"data_dir": tmpdir},
                model={"num_samples": 50, "num_warmup": 25, "num_chains": 1},
            )

            # Generate test data
            data = generate_synthetic_data(
                n_subjects=15, n_features=25, K=3, num_sources=2
            )
            X_list = [data["X1"], data["X2"]]

            comparison = SGFAParameterComparison(config)

            # Test different K values
            for K in [2, 3, 4]:
                result = comparison.run_sgfa_variant_comparison(
                    X_list=X_list,
                    hypers={"K": K, "alpha_w": 1.0, "alpha_z": 1.0},
                    args={"num_samples": 50, "num_warmup": 25, "num_chains": 1},
                )

                assert result.success is True
                assert result.data["variant_results"] is not None

    def test_error_handling_and_recovery(self):
        """Test error handling in SGFA parameter comparison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                experiments={"base_output_dir": tmpdir},
                data={"data_dir": tmpdir},
                model={"K": 3},
            )

            # Generate problematic data
            X_problematic = [np.array([[np.inf, 1, 2], [3, 4, 5]])]  # Contains infinity

            comparison = SGFAParameterComparison(config)

            # Should handle problematic data gracefully
            result = comparison.run_sgfa_variant_comparison(
                X_list=X_problematic,
                hypers={"K": 2, "alpha_w": 1.0, "alpha_z": 1.0},
                args={"num_samples": 10, "num_warmup": 5},
            )

            # Error should be handled by @experiment_handler decorator
            assert hasattr(result, "success")
