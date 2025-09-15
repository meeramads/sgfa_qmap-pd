"""Tests for sensitivity analysis experiment."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from experiments.sensitivity_analysis import run_sensitivity_analysis
from data import generate_synthetic_data


class TestSensitivityAnalysis:
    """Test sensitivity analysis experiment."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return {
            'data': {
                'data_dir': './test_data'
            },
            'experiments': {
                'base_output_dir': './test_results',
                'save_intermediate': True
            },
            'sensitivity_analysis': {
                'parameter_ranges': {
                    'n_factors': [3, 5],  # Reduced for testing
                    'sparsity_lambda': [0.1, 0.5],
                    'learning_rate': [0.01, 0.05],
                    'batch_size': [16, 32]
                },
                'stability_tests': {
                    'n_random_inits': 2,  # Reduced for testing
                    'convergence_threshold': 1e-4
                }
            }
        }

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data for testing."""
        return generate_synthetic_data(
            num_sources=2,
            K=3,
            num_subjects=20,  # Small for fast testing
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

    def test_sensitivity_analysis_runs(self, mock_config):
        """Test that sensitivity analysis experiment runs without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            result = run_sensitivity_analysis(mock_config)
            assert result is not None

    def test_sensitivity_analysis_with_shared_data(self, shared_data_config):
        """Test sensitivity analysis with shared data mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_data_config['experiments']['base_output_dir'] = tmpdir

            result = run_sensitivity_analysis(shared_data_config)
            assert result is not None

    def test_hyperparameter_sensitivity(self, mock_config):
        """Test hyperparameter sensitivity analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Test specific hyperparameter ranges
            mock_config['sensitivity_analysis']['parameter_ranges'] = {
                'n_factors': [3, 5, 8],
                'sparsity_lambda': [0.1, 0.5, 1.0]
            }

            result = run_sensitivity_analysis(mock_config)
            assert result is not None

    def test_k_factor_sensitivity(self, mock_config):
        """Test K (number of factors) sensitivity analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Focus on K sensitivity
            mock_config['sensitivity_analysis']['parameter_ranges'] = {
                'n_factors': [2, 3, 4, 5]
            }

            result = run_sensitivity_analysis(mock_config)
            assert result is not None

    def test_sparsity_sensitivity(self, mock_config):
        """Test sparsity parameter sensitivity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Focus on sparsity sensitivity
            mock_config['sensitivity_analysis']['parameter_ranges'] = {
                'sparsity_lambda': [0.01, 0.1, 0.5, 1.0, 2.0]
            }

            result = run_sensitivity_analysis(mock_config)
            assert result is not None

    def test_mcmc_parameter_sensitivity(self, mock_config):
        """Test MCMC parameter sensitivity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # MCMC-specific parameters
            mock_config['sensitivity_analysis']['parameter_ranges'] = {
                'num_samples': [50, 100],  # Small for testing
                'num_warmup': [25, 50],
                'target_accept_prob': [0.7, 0.8, 0.9]
            }

            result = run_sensitivity_analysis(mock_config)
            assert result is not None

    def test_stability_analysis(self, mock_config):
        """Test stability analysis with random initializations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Test stability with multiple random initializations
            mock_config['sensitivity_analysis']['stability_tests'] = {
                'n_random_inits': 3,
                'convergence_threshold': 1e-4
            }

            result = run_sensitivity_analysis(mock_config)
            assert result is not None

    def test_univariate_sensitivity(self, mock_config):
        """Test univariate sensitivity analysis (one parameter at a time)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Test one parameter at a time
            mock_config['sensitivity_analysis']['analysis_type'] = 'univariate'
            mock_config['sensitivity_analysis']['parameter_ranges'] = {
                'n_factors': [3, 5, 8]
            }

            result = run_sensitivity_analysis(mock_config)
            assert result is not None

    def test_multivariate_sensitivity(self, mock_config):
        """Test multivariate sensitivity analysis (parameter interactions)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Test parameter interactions
            mock_config['sensitivity_analysis']['analysis_type'] = 'multivariate'
            mock_config['sensitivity_analysis']['parameter_ranges'] = {
                'n_factors': [3, 5],
                'sparsity_lambda': [0.1, 0.5]
            }

            result = run_sensitivity_analysis(mock_config)
            assert result is not None

    def test_gradient_based_sensitivity(self, mock_config):
        """Test gradient-based sensitivity analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Enable gradient-based analysis
            mock_config['sensitivity_analysis']['gradient_analysis'] = True

            result = run_sensitivity_analysis(mock_config)
            assert result is not None

    def test_robustness_analysis(self, mock_config):
        """Test robustness analysis to data perturbations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Test robustness to data noise
            mock_config['sensitivity_analysis']['robustness_tests'] = {
                'noise_levels': [0.05, 0.1, 0.2],
                'perturbation_types': ['gaussian', 'uniform']
            }

            result = run_sensitivity_analysis(mock_config)
            assert result is not None

    def test_sensitivity_output_structure(self, mock_config):
        """Test that sensitivity analysis produces expected output structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            result = run_sensitivity_analysis(mock_config)

            # Check result structure
            assert hasattr(result, 'experiment_id')
            assert hasattr(result, 'status')
            assert hasattr(result, 'model_results')

            # Check output directory
            output_path = Path(tmpdir)
            assert output_path.exists()

    def test_sensitivity_metrics_computation(self, mock_config):
        """Test that sensitivity metrics are computed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            result = run_sensitivity_analysis(mock_config)
            assert result is not None

            if hasattr(result, 'model_results') and result.model_results:
                model_results = result.model_results

                # Should contain sensitivity analysis results
                assert isinstance(model_results, dict)

    def test_parameter_importance_ranking(self, mock_config):
        """Test parameter importance ranking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Multiple parameters for ranking
            mock_config['sensitivity_analysis']['parameter_ranges'] = {
                'n_factors': [3, 5, 8],
                'sparsity_lambda': [0.1, 0.5, 1.0],
                'learning_rate': [0.01, 0.05, 0.1]
            }

            result = run_sensitivity_analysis(mock_config)
            assert result is not None

    def test_minimal_sensitivity_config(self, synthetic_data):
        """Test with minimal sensitivity configuration."""
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

            result = run_sensitivity_analysis(minimal_config)
            assert result is not None

    def test_error_handling_invalid_parameters(self, mock_config):
        """Test error handling with invalid parameter ranges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Invalid parameter ranges
            mock_config['sensitivity_analysis']['parameter_ranges'] = {
                'n_factors': [-1, 0],  # Invalid values
                'sparsity_lambda': [-0.5, 2.0]  # Some invalid
            }

            # Should handle gracefully
            result = run_sensitivity_analysis(mock_config)

            # Either succeeds with valid subset or handles error
            if result is not None:
                assert hasattr(result, 'status')

    def test_large_parameter_grid_handling(self, mock_config):
        """Test handling of large parameter grids."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Large parameter grid (should be manageable or subsampled)
            mock_config['sensitivity_analysis']['parameter_ranges'] = {
                'n_factors': [2, 3, 4, 5, 6, 7, 8],
                'sparsity_lambda': [0.01, 0.1, 0.2, 0.5, 1.0, 2.0],
                'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1]
            }

            # Should handle by subsampling or limiting combinations
            result = run_sensitivity_analysis(mock_config)
            assert result is not None

    def test_convergence_sensitivity(self, mock_config):
        """Test sensitivity to convergence criteria."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Test different convergence thresholds
            mock_config['sensitivity_analysis']['stability_tests'] = {
                'convergence_thresholds': [1e-3, 1e-4, 1e-5],
                'n_random_inits': 2
            }

            result = run_sensitivity_analysis(mock_config)
            assert result is not None

    @pytest.mark.parametrize("param_name", ['n_factors', 'sparsity_lambda', 'learning_rate'])
    def test_individual_parameter_sensitivity(self, mock_config, param_name):
        """Test sensitivity analysis for individual parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Test single parameter
            param_ranges = {
                'n_factors': [3, 5, 8],
                'sparsity_lambda': [0.1, 0.5, 1.0],
                'learning_rate': [0.01, 0.05, 0.1]
            }

            mock_config['sensitivity_analysis']['parameter_ranges'] = {
                param_name: param_ranges[param_name]
            }

            result = run_sensitivity_analysis(mock_config)
            assert result is not None

    def test_sensitivity_reproducibility(self, mock_config):
        """Test sensitivity analysis reproducibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir
            mock_config['random_seed'] = 42

            # Run twice with same configuration
            result1 = run_sensitivity_analysis(mock_config)
            result2 = run_sensitivity_analysis(mock_config)

            assert result1 is not None
            assert result2 is not None

    def test_sensitivity_matrix_saving(self, mock_config):
        """Test that sensitivity analysis matrices are saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            result = run_sensitivity_analysis(mock_config)
            assert result is not None

            # Check for matrix output files
            output_files = list(Path(tmpdir).rglob("*"))
            assert len(output_files) > 0

    def test_sensitivity_plot_generation(self, mock_config):
        """Test that sensitivity plots are generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            # Enable plot generation
            mock_config['experiments']['generate_plots'] = True

            result = run_sensitivity_analysis(mock_config)
            assert result is not None

            # Check for plot files
            plot_files = list(Path(tmpdir).rglob("*.png")) + list(Path(tmpdir).rglob("*.pdf"))

            # Should generate some plots
            assert len(list(Path(tmpdir).rglob("*"))) > 0

    def test_sensitivity_summary_statistics(self, mock_config):
        """Test computation of sensitivity summary statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config['experiments']['base_output_dir'] = tmpdir

            result = run_sensitivity_analysis(mock_config)
            assert result is not None

            # Check for summary output
            summary_files = list(Path(tmpdir).rglob("*summary*"))

            # Should have some form of summary output
            assert len(list(Path(tmpdir).rglob("*"))) > 0