"""Tests for model comparison experiments."""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

from experiments.model_comparison import ModelArchitectureComparison, run_model_comparison
from experiments.framework import ExperimentConfig
from data import generate_synthetic_data


class TestModelArchitectureComparison:
    """Test model architecture comparison experiments."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        data = generate_synthetic_data(
            n_subjects=20,
            n_features=50,
            K=3,
            num_sources=2,
            noise_level=0.1
        )
        return [data['X1'], data['X2']]

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                experiments={'base_output_dir': tmpdir},
                data={'data_dir': tmpdir},
                model={
                    'K': 3,
                    'num_samples': 100,
                    'num_warmup': 50,
                    'num_chains': 1
                }
            )
            yield config

    def test_model_architecture_comparison_init(self, config):
        """Test ModelArchitectureComparison initialization."""
        comparison = ModelArchitectureComparison(config)

        assert comparison.config == config
        assert hasattr(comparison, 'model_architectures')
        assert 'sparseGFA' in comparison.model_architectures
        assert comparison.profiler is not None

    def test_run_model_architecture_comparison(self, sample_data, config):
        """Test running model architecture comparison."""
        comparison = ModelArchitectureComparison(config)

        # Run with minimal parameters for testing
        result = comparison.run_model_architecture_comparison(
            X_list=sample_data,
            base_hypers={'K': 3, 'alpha_w': 1.0, 'alpha_z': 1.0},
            args={'num_samples': 50, 'num_warmup': 25, 'num_chains': 1}
        )

        # Check result structure
        assert result.success is True
        assert result.experiment_name == "model_architecture_comparison"
        assert result.data is not None
        assert 'comparison_results' in result.data
        assert 'model_rankings' in result.data

        # Check that at least one model was tested
        assert len(result.data['comparison_results']) >= 1

    def test_run_traditional_method_comparison(self, sample_data, config):
        """Test running traditional method comparison."""
        comparison = ModelArchitectureComparison(config)

        # First run SGFA to get reference results
        sgfa_results = {
            'log_likelihood': -1000.0,
            'W_mean': np.random.randn(50, 3),
            'Z_mean': np.random.randn(20, 3)
        }

        # Run traditional method comparison
        result = comparison.run_traditional_method_comparison(
            X_list=sample_data,
            sgfa_results=sgfa_results
        )

        # Check result structure
        assert result.success is True
        assert result.experiment_name == "traditional_method_comparison"
        assert result.data is not None
        assert 'method_results' in result.data
        assert 'comparison_metrics' in result.data

    def test_model_architecture_comparison_with_invalid_data(self, config):
        """Test model comparison with invalid data."""
        comparison = ModelArchitectureComparison(config)

        # Test with empty data list
        with pytest.raises(ValueError):
            comparison.run_model_architecture_comparison(
                X_list=[],
                base_hypers={'K': 3},
                args={'num_samples': 50}
            )

    def test_model_architecture_comparison_with_mismatched_dimensions(self, config):
        """Test model comparison with mismatched data dimensions."""
        comparison = ModelArchitectureComparison(config)

        # Create data with mismatched dimensions
        X1 = np.random.randn(20, 50)
        X2 = np.random.randn(15, 50)  # Different number of subjects

        # This should handle the mismatch gracefully
        result = comparison.run_model_architecture_comparison(
            X_list=[X1, X2],
            base_hypers={'K': 3, 'alpha_w': 1.0, 'alpha_z': 1.0},
            args={'num_samples': 50, 'num_warmup': 25, 'num_chains': 1}
        )

        # Should still succeed but may have warnings
        assert result.success is True

    def test_performance_profiling(self, sample_data, config):
        """Test that performance profiling is enabled."""
        comparison = ModelArchitectureComparison(config)

        result = comparison.run_model_architecture_comparison(
            X_list=sample_data,
            base_hypers={'K': 3, 'alpha_w': 1.0, 'alpha_z': 1.0},
            args={'num_samples': 50, 'num_warmup': 25, 'num_chains': 1}
        )

        # Check that performance metrics are included
        assert result.performance_metrics is not None
        assert 'timing' in result.performance_metrics or 'memory' in result.performance_metrics


class TestModelComparisonStandalone:
    """Test standalone model comparison function."""

    def test_run_model_comparison_function(self):
        """Test the standalone run_model_comparison function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate test data
            data = generate_synthetic_data(
                n_subjects=15,
                n_features=30,
                K=2,
                num_sources=2,
                noise_level=0.1
            )

            # Create minimal config
            config = {
                'experiments': {'base_output_dir': tmpdir},
                'data': {'data_dir': tmpdir},
                'model': {
                    'K': 2,
                    'num_samples': 50,
                    'num_warmup': 25,
                    'num_chains': 1
                }
            }

            # Run model comparison
            result = run_model_comparison(
                config=config,
                X_list=[data['X1'], data['X2']]
            )

            # Check that files were created
            output_dir = Path(tmpdir)
            assert any(output_dir.glob('*.json'))  # Should have JSON results

            # Check basic return structure
            assert result is not None

    def test_run_model_comparison_with_defaults(self):
        """Test model comparison with default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should handle minimal input gracefully
            result = run_model_comparison(
                base_output_dir=tmpdir,
                generate_synthetic=True,
                n_subjects=10,
                K=2
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
                experiments={'base_output_dir': tmpdir},
                data={'data_dir': tmpdir},
                model={
                    'K': 3,
                    'num_samples': 100,
                    'num_warmup': 50,
                    'num_chains': 1,
                    'alpha_w': 1.0,
                    'alpha_z': 1.0
                }
            )

            # Generate test data
            data = generate_synthetic_data(
                n_subjects=25,
                n_features=40,
                K=3,
                num_sources=2,
                noise_level=0.05
            )
            X_list = [data['X1'], data['X2']]

            # Initialize comparison
            comparison = ModelArchitectureComparison(config)

            # Run model architecture comparison
            arch_result = comparison.run_model_architecture_comparison(
                X_list=X_list,
                base_hypers={'K': 3, 'alpha_w': 1.0, 'alpha_z': 1.0},
                args={'num_samples': 100, 'num_warmup': 50, 'num_chains': 1}
            )

            assert arch_result.success is True

            # Extract SGFA results for traditional comparison
            sgfa_results = arch_result.data['comparison_results'].get('sparseGFA', {})

            # Run traditional method comparison
            trad_result = comparison.run_traditional_method_comparison(
                X_list=X_list,
                sgfa_results=sgfa_results
            )

            assert trad_result.success is True

            # Check that output files were created
            output_dir = Path(tmpdir) / "model_architecture_comparison"
            assert output_dir.exists()

            # Check for expected output files
            json_files = list(output_dir.glob('*.json'))
            assert len(json_files) >= 1

            # Verify JSON content is valid
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    assert isinstance(data, dict)

    def test_error_handling_and_recovery(self, sample_data):
        """Test error handling in model comparison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                experiments={'base_output_dir': tmpdir},
                data={'data_dir': tmpdir},
                model={'K': 3}
            )

            comparison = ModelArchitectureComparison(config)

            # Test with invalid hyperparameters
            result = comparison.run_model_architecture_comparison(
                X_list=sample_data,
                base_hypers={'K': -1},  # Invalid K
                args={'num_samples': 10}
            )

            # Should handle error gracefully
            assert hasattr(result, 'success')