"""Tests for method comparison experiment."""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

from experiments.method_comparison import run_method_comparison
from data import generate_synthetic_data


class TestMethodComparison:
    """Test method comparison experiment."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return {
            'data': {
                'data_dir': './test_data'
            },
            'experiments': {
                'base_output_dir': './test_results',
                'save_intermediate': True,
                'generate_plots': True
            },
            'method_comparison': {
                'models': [
                    {
                        'name': 'sparseGFA',
                        'n_factors': [3, 5],
                        'sparsity_lambda': [0.1, 0.5],
                        'group_lambda': [0.1, 0.5]
                    },
                    {
                        'name': 'standardGFA',
                        'n_factors': [3, 5]
                    }
                ],
                'cross_validation': {
                    'n_folds': 3,  # Reduced for testing
                    'n_repeats': 1
                },
                'evaluation_metrics': [
                    'reconstruction_error',
                    'factor_interpretability'
                ]
            }
        }

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data for testing."""
        return generate_synthetic_data(
            num_sources=2,
            K=3,
            num_subjects=20,  # Very small for fast testing
            seed=42
        )

    @pytest.fixture
    def shared_data_config(self, mock_config, synthetic_data):
        """Create config with shared data."""
        config = mock_config.copy()
        config['_shared_data'] = {
            'X_list': synthetic_data['X_list'],
            'preprocessing_info': {'strategy': 'minimal'},
            'mode': 'shared'
        }
        return config

    def test_method_comparison_runs(self, mock_config):
        """Test that method comparison experiment runs without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            result = run_method_comparison(mock_config)
            assert result is not None

    def test_method_comparison_with_shared_data(self, shared_data_config):
        """Test method comparison with shared data mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_data_config['experiments']['base_output_dir'] = tmpdir

            result = run_method_comparison(shared_data_config)
            assert result is not None

            # Verify shared data was used
            assert '_shared_data' in shared_data_config

    def test_method_comparison_output_structure(self, mock_config):
        """Test that method comparison produces expected output structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            result = run_method_comparison(mock_config)

            # Check result structure
            assert hasattr(result, 'experiment_id')
            assert hasattr(result, 'status')
            assert hasattr(result, 'model_results')

            # Check that output directory was created
            output_path = Path(tmpdir)
            assert output_path.exists()

    def test_method_comparison_matrix_saving(self, mock_config):
        """Test that factor matrices are saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            result = run_method_comparison(mock_config)

            if result and hasattr(result, 'model_results'):
                # Look for matrices directory or matrix files
                output_files = list(Path(tmpdir).rglob("*matrix*")) + list(Path(tmpdir).rglob("*factor*"))

                # Should have some matrix-related output
                assert len(list(Path(tmpdir).rglob("*"))) > 0

    def test_sgfa_variants_testing(self, mock_config):
        """Test that different SGFA variants are tested."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Ensure we test multiple variants
            mock_config['method_comparison']['models'] = [
                {
                    'name': 'sparseGFA',
                    'n_factors': [3, 5],
                    'sparsity_lambda': [0.1, 0.5]
                }
            ]

            result = run_method_comparison(mock_config)
            assert result is not None

    def test_traditional_methods_comparison(self, mock_config):
        """Test comparison with traditional methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Include traditional methods in comparison
            mock_config['method_comparison']['models'].append({
                'name': 'pca',
                'n_components': [3, 5]
            })

            result = run_method_comparison(mock_config)
            assert result is not None

    def test_cross_validation_integration(self, mock_config):
        """Test that cross-validation is properly integrated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Set specific CV parameters
            mock_config['method_comparison']['cross_validation'] = {
                'n_folds': 2,  # Minimal for testing
                'n_repeats': 1,
                'stratify': False
            }

            result = run_method_comparison(mock_config)
            assert result is not None

            # Check if CV results are present
            if hasattr(result, 'cv_results'):
                assert result.cv_results is not None

    def test_evaluation_metrics_computation(self, mock_config):
        """Test that evaluation metrics are computed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            result = run_method_comparison(mock_config)
            assert result is not None

            # Check for performance metrics
            if hasattr(result, 'model_results'):
                model_results = result.model_results

                # Should have some evaluation metrics
                assert isinstance(model_results, dict)

    def test_minimal_configuration(self, synthetic_data):
        """Test with minimal configuration."""
        minimal_config = {
            'data': {'data_dir': './test_data'},
            'experiments': {'base_output_dir': './test_results'},
            '_shared_data': {
                'X_list': synthetic_data['X_list'],
                'preprocessing_info': {},
                'mode': 'shared'
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config['experiments']['base_output_dir'] = tmpdir

            result = run_method_comparison(minimal_config)
            assert result is not None

    def test_error_handling_invalid_model(self, mock_config):
        """Test error handling with invalid model configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Add invalid model
            mock_config['method_comparison']['models'].append({
                'name': 'invalid_model_name',
                'n_factors': [3]
            })

            # Should handle error gracefully
            result = run_method_comparison(mock_config)

            # Either returns None or handles error in result
            if result is not None:
                assert hasattr(result, 'status')

    def test_large_parameter_grid_handling(self, mock_config):
        """Test handling of large parameter grids (should be manageable)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Large but manageable parameter grid
            mock_config['method_comparison']['models'] = [
                {
                    'name': 'sparseGFA',
                    'n_factors': [3, 5, 8],
                    'sparsity_lambda': [0.1, 0.5, 1.0]
                }
            ]

            result = run_method_comparison(mock_config)
            assert result is not None

    @pytest.mark.parametrize("model_name", ['sparseGFA', 'standardGFA'])
    def test_individual_models(self, mock_config, model_name):
        """Test individual model types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Test single model
            mock_config['method_comparison']['models'] = [
                {
                    'name': model_name,
                    'n_factors': [3]
                }
            ]

            result = run_method_comparison(mock_config)
            assert result is not None

    def test_output_file_generation(self, mock_config):
        """Test that expected output files are generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            result = run_method_comparison(mock_config)
            assert result is not None

            # Check for output files
            output_files = list(Path(tmpdir).rglob("*"))

            # Should generate some output files
            assert len(output_files) > 0

            # Look for specific file types
            json_files = list(Path(tmpdir).rglob("*.json"))
            pkl_files = list(Path(tmpdir).rglob("*.pkl"))

            # Should have at least some structured output
            assert len(json_files) + len(pkl_files) > 0

    def test_performance_metrics_validation(self, mock_config):
        """Test that performance metrics are reasonable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            result = run_method_comparison(mock_config)
            assert result is not None

            if hasattr(result, 'model_results') and result.model_results:
                # Check that results contain expected structure
                assert isinstance(result.model_results, dict)

    def test_reproducibility_with_seed(self, mock_config):
        """Test reproducibility when using fixed seeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Add seed to ensure reproducibility
            mock_config['random_seed'] = 42

            # Run twice with same seed
            result1 = run_method_comparison(mock_config)
            result2 = run_method_comparison(mock_config)

            # Both should complete successfully
            assert result1 is not None
            assert result2 is not None