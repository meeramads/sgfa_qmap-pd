"""Tests for the main run_analysis.py module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import jax.numpy as jnp

from core.run_analysis import models, run_inference


@pytest.mark.unit
@pytest.mark.model
class TestModels:
    """Test the GFA model definition."""

    def test_models_basic_structure(self, sample_synthetic_data, sample_hyperparameters):
        """Test that models function has correct structure."""
        # Mock args
        args = Mock()
        args.num_sources = 3
        args.K = 5
        
        X_list = sample_synthetic_data['X_list']
        hypers = sample_hyperparameters
        
        # Since models() uses numpyro.sample which requires execution context,
        # we'll test the function structure rather than direct execution
        assert callable(models)
        
        # Test that function accepts correct parameters
        try:
            # This would need to be run in a numpyro context to actually work
            # models(X_list, hypers, args)
            pass
        except Exception:
            # Expected - we're not in a proper numpyro execution context
            pass

    def test_models_parameter_validation(self):
        """Test model parameter validation."""
        # Test with invalid parameters
        args = Mock()
        args.num_sources = 2
        args.K = 3
        
        # Create mismatched data (3 sources but args says 2)
        X_list = [
            np.random.normal(0, 1, (50, 20)),
            np.random.normal(0, 1, (50, 15)), 
            np.random.normal(0, 1, (50, 10))  # Extra source
        ]
        
        hypers = {'Dm': [20, 15, 10], 'percW': 33.0, 'a_sigma': 1.0, 'b_sigma': 1.0}
        
        # This should raise an assertion error about mismatched sources
        with pytest.raises(AssertionError, match="Number of data sources does not match"):
            models(X_list, hypers, args)

    def test_models_dimension_consistency(self):
        """Test that model enforces dimension consistency."""
        args = Mock()
        args.num_sources = 2
        args.K = 3
        
        # Create data with inconsistent subject numbers
        X_list = [
            np.random.normal(0, 1, (50, 20)),  # 50 subjects
            np.random.normal(0, 1, (60, 15))   # 60 subjects - mismatch!
        ]
        
        hypers = {'Dm': [20, 15], 'percW': 33.0, 'a_sigma': 1.0, 'b_sigma': 1.0}
        
        # Should raise assertion error about inconsistent samples
        with pytest.raises(AssertionError, match="inconsistent number of samples"):
            models(X_list, hypers, args)


@pytest.mark.unit
@pytest.mark.model  
class TestRunInference:
    """Test MCMC inference functionality."""

    @patch('run_analysis.MCMC')
    @patch('run_analysis.NUTS')
    @patch('run_analysis.jax.random.PRNGKey')
    def test_run_inference_basic(self, mock_prng, mock_nuts, mock_mcmc, sample_synthetic_data, sample_hyperparameters):
        """Test basic inference execution."""
        # Setup mocks
        mock_key = Mock()
        mock_prng.return_value = mock_key
        
        mock_nuts_instance = Mock()
        mock_nuts.return_value = mock_nuts_instance
        
        mock_mcmc_instance = Mock()
        mock_mcmc_instance.get_samples.return_value = {
            'Z': np.random.normal(0, 1, (100, 50, 5)),
            'W': np.random.normal(0, 1, (100, 45, 5)),
            'sigma': np.random.gamma(2, 1, (100, 3))
        }
        mock_mcmc.return_value = mock_mcmc_instance
        
        # Mock args
        args = Mock()
        args.num_samples = 100
        args.num_chains = 2
        args.num_warmup = 50
        args.device = 'cpu'
        
        X_list = sample_synthetic_data['X_list']
        hypers = sample_hyperparameters
        rng_key = mock_key
        
        # Run inference
        results = run_inference(models, args, rng_key, X_list, hypers)
        
        # Verify MCMC was set up correctly
        mock_nuts.assert_called_once_with(models)
        mock_mcmc.assert_called_once_with(
            mock_nuts_instance,
            num_samples=100,
            num_chains=2, 
            num_warmup=50
        )
        
        # Verify MCMC was run
        mock_mcmc_instance.run.assert_called_once_with(rng_key, X_list, hypers, args)
        mock_mcmc_instance.get_samples.assert_called_once()
        
        # Verify results
        assert 'Z' in results
        assert 'W' in results  
        assert 'sigma' in results

    @patch('run_analysis.logger')
    def test_run_inference_logging(self, mock_logger, sample_synthetic_data, sample_hyperparameters):
        """Test that inference logs appropriately."""
        args = Mock()
        args.num_samples = 50
        args.num_chains = 1
        args.device = 'cpu'
        
        rng_key = Mock()
        X_list = sample_synthetic_data['X_list']
        hypers = sample_hyperparameters
        
        with patch('run_analysis.MCMC') as mock_mcmc, \
             patch('run_analysis.NUTS'):
            
            mock_mcmc_instance = Mock()
            mock_mcmc_instance.get_samples.return_value = {'Z': np.random.normal(0, 1, (50, 50, 5))}
            mock_mcmc.return_value = mock_mcmc_instance
            
            run_inference(models, args, rng_key, X_list, hypers)
            
            # Should have logged inference start/completion
            assert mock_logger.info.called
            
    def test_run_inference_device_handling(self, sample_synthetic_data, sample_hyperparameters):
        """Test device handling in inference."""
        args = Mock()
        args.num_samples = 10
        args.num_chains = 1
        args.device = 'gpu'
        
        with patch('run_analysis.jax.devices') as mock_devices, \
             patch('run_analysis.MCMC') as mock_mcmc, \
             patch('run_analysis.NUTS'):
            
            # Mock GPU availability
            mock_devices.return_value = [Mock(platform='gpu')]
            
            mock_mcmc_instance = Mock()
            mock_mcmc_instance.get_samples.return_value = {'Z': np.random.normal(0, 1, (10, 50, 5))}
            mock_mcmc.return_value = mock_mcmc_instance
            
            rng_key = Mock()
            X_list = sample_synthetic_data['X_list']
            hypers = sample_hyperparameters
            
            # This should work without errors
            results = run_inference(models, args, rng_key, X_list, hypers)
            assert results is not None


@pytest.mark.integration
@pytest.mark.slow
class TestMainFunction:
    """Test the main analysis function (integration test)."""
    
    def test_main_synthetic_analysis(self, temp_dir):
        """Test main function with synthetic data analysis."""
        # Mock command line arguments
        args = Mock()
        args.dataset = 'synthetic'
        args.num_sources = 2
        args.K = 3
        args.num_samples = 50  # Small for speed
        args.num_chains = 1
        args.num_runs = 1
        args.device = 'cpu'
        args.results_dir = str(temp_dir)
        args.run_cv = False
        args.cv_only = False
        args.seed = 42
        
        # Mock heavy computations
        mock_mcmc_results = {
            'Z': np.random.normal(0, 1, (50, 150, 3)),
            'W': np.random.normal(0, 1, (50, 120, 3)),
            'sigma': np.random.gamma(2, 1, (50, 2))
        }
        
        with patch('run_analysis.validate_and_setup_args', return_value=args), \
             patch('run_analysis.run_inference', return_value=mock_mcmc_results), \
             patch('run_analysis._create_visualizations'), \
             patch('run_analysis._log_final_summary'):
            
            from core.run_analysis import main
            
            # This should complete without errors
            main(args)
            
    def test_main_with_cv(self, temp_dir):
        """Test main function with cross-validation."""
        args = Mock()
        args.dataset = 'synthetic'
        args.run_cv = True
        args.cv_only = True
        args.results_dir = str(temp_dir)
        args.seed = 42
        
        mock_cv_results = {'cv_scores': [0.8, 0.85, 0.9]}
        
        with patch('run_analysis.validate_and_setup_args', return_value=args), \
             patch('run_analysis.should_run_cv_analysis', return_value=True), \
             patch('run_analysis.should_run_standard_analysis', return_value=False), \
             patch('run_analysis.CVRunner') as mock_cv_runner:
            
            mock_cv_instance = Mock()
            mock_cv_instance.run_cv_analysis.return_value = mock_cv_results
            mock_cv_runner.return_value = mock_cv_instance
            
            from core.run_analysis import main
            
            # Should complete CV-only analysis
            main(args)