"""Tests for SGFA hyperparameter tuning experiments."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from data import generate_synthetic_data
from experiments.framework import ExperimentConfig
from experiments.sgfa_hyperparameter_tuning import (
    SGFAHyperparameterTuning,
    run_sgfa_hyperparameter_tuning,  # Main entry point for SGFA hyperparameter optimization
)


class TestSGFAHyperparameterTuning:
    """Test SGFA hyperparameter tuning experiments."""

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
                # Add the new sgfa_hyperparameter_tuning configuration section
                sgfa_hyperparameter_tuning={
                    "parameter_ranges": {
                        "n_factors": [3, 4, 5],
                        "sparsity_lambda": [0.1, 0.3, 0.5],
                        "group_lambda": [0.1, 0.5],
                    },
                    "scalability_analysis": {
                        "sample_size_ranges": [20, 30],
                        "feature_size_ranges": [40, 60],
                    }
                }
            )
            yield config

    def test_sgfa_hyperparameter_tuning_init(self, config):
        """Test SGFAHyperparameterTuning initialization."""
        comparison = SGFAHyperparameterTuning(config)

        assert comparison.config == config
        assert hasattr(comparison, "sgfa_variants")
        assert "standard" in comparison.sgfa_variants
        assert "sparse_only" in comparison.sgfa_variants
        assert "group_only" in comparison.sgfa_variants
        assert comparison.profiler is not None

    def test_run_sgfa_variant_comparison(self, sample_data, config):
        """Test running SGFA variant comparison."""
        comparison = SGFAHyperparameterTuning(config)

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

    def test_run_comprehensive_sgfa_scalability_analysis(self, sample_data, config):
        """Test running comprehensive SGFA scalability analysis."""
        comparison = SGFAHyperparameterTuning(config)

        # Prepare minimal args for scalability testing
        hypers = {
            "percW": 25.0,
            "Dm": [X.shape[1] for X in sample_data],
            "a_sigma": 1.0,
            "b_sigma": 1.0,
        }
        args = {
            "K": 3,
            "num_samples": 50,  # Reduced for testing
            "num_warmup": 25,   # Reduced for testing
            "num_chains": 1,
        }

        # Run scalability analysis (this will use the config parameter ranges)
        result = comparison.run_comprehensive_sgfa_scalability_analysis(
            X_list=sample_data, hypers=hypers, args=args
        )

        # Check result structure
        assert result.status == "completed"
        assert result.model_results is not None
        # Verify different scalability tests were performed
        assert "sample_scalability" in result.model_results or "feature_scalability" in result.model_results

    def test_run_multiview_capability_assessment(self, sample_data, config):
        """Test running multiview capability assessment."""
        comparison = SGFAHyperparameterTuning(config)

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
        comparison = SGFAHyperparameterTuning(config)

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
        comparison = SGFAHyperparameterTuning(config)

        # Test with empty data list
        with pytest.raises(ValueError):
            comparison.run_sgfa_variant_comparison(
                X_list=[], hypers={"K": 4}, args={"num_samples": 50}
            )

    def test_sgfa_variant_comparison_with_invalid_hyperparameters(
        self, sample_data, config
    ):
        """Test SGFA variant comparison with invalid hyperparameters."""
        comparison = SGFAHyperparameterTuning(config)

        # Test with invalid K value
        with pytest.raises(ValueError):
            comparison.run_sgfa_variant_comparison(
                X_list=sample_data,
                hypers={"K": 0},  # Invalid K
                args={"num_samples": 50},
            )

    def test_scalability_comparison_parameter_validation(self, sample_data, config):
        """Test scalability comparison parameter validation."""
        comparison = SGFAHyperparameterTuning(config)

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
        comparison = SGFAHyperparameterTuning(config)

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


class TestSGFAHyperparameterTuningStandalone:
    """Test standalone SGFA parameter comparison function."""

    def test_run_sgfa_hyperparameter_tuning_function(self):
        """Test the standalone run_sgfa_hyperparameter_tuning function."""
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

            # Add sgfa_hyperparameter_tuning config section
            config["sgfa_hyperparameter_tuning"] = {
                "parameter_ranges": {
                    "n_factors": [3, 4],
                    "sparsity_lambda": [0.1, 0.3],
                },
                "scalability_analysis": {
                    "sample_size_ranges": [20],
                    "feature_size_ranges": [40],
                }
            }

            # Run SGFA parameter comparison
            result = run_sgfa_hyperparameter_tuning(config)

            # Check that function completes
            assert result is not None

    def test_run_sgfa_hyperparameter_tuning_with_custom_data(self):
        """Test SGFA parameter comparison with custom data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate custom test data
            np.random.randn(15, 25)
            np.random.randn(15, 30)

            config = {
                "experiments": {"base_output_dir": tmpdir},
                "data": {"data_dir": tmpdir},
                "model": {"K": 2, "num_samples": 50, "num_warmup": 25},
                "sgfa_hyperparameter_tuning": {
                    "parameter_ranges": {
                        "n_factors": [2, 3],
                        "sparsity_lambda": [0.2, 0.4],
                    },
                    "scalability_analysis": {
                        "sample_size_ranges": [15],
                        "feature_size_ranges": [25, 30],
                    }
                }
            }

            # Should handle custom data
            result = run_sgfa_hyperparameter_tuning(config)
            assert result is not None


class TestSGFAHyperparameterTuningIntegration:
    """Integration tests for SGFA parameter comparison."""

    def test_full_sgfa_hyperparameter_tuning_workflow(self):
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
            comparison = SGFAHyperparameterTuning(config)

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

            comparison = SGFAHyperparameterTuning(config)

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

            comparison = SGFAHyperparameterTuning(config)

            # Should handle problematic data gracefully
            result = comparison.run_sgfa_variant_comparison(
                X_list=X_problematic,
                hypers={"K": 2, "alpha_w": 1.0, "alpha_z": 1.0},
                args={"num_samples": 10, "num_warmup": 5},
            )

            # Error should be handled by @experiment_handler decorator
            assert hasattr(result, "success")
