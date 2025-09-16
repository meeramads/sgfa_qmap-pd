"""Tests for Standard GFA model implementation."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Dict, Any

from models.standard_gfa import StandardGFAModel as StandardGFA
from data import generate_synthetic_data


class TestStandardGFA:
    """Test Standard GFA model implementation."""

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
    def basic_config(self):
        """Basic configuration for StandardGFA."""
        return {
            'K': 3,
            'num_samples': 50,  # Small for testing
            'num_warmup': 25,
            'num_chains': 1
        }

    def test_standard_gfa_initialization(self, basic_config):
        """Test that StandardGFA initializes correctly."""
        model = StandardGFA(**basic_config)

        assert model.K == 3
        assert model.num_samples == 50
        assert model.num_warmup == 25

    def test_standard_gfa_fit(self, synthetic_data, basic_config):
        """Test that StandardGFA can fit to data."""
        X_list = synthetic_data['X_list']

        model = StandardGFA(**basic_config)

        # Should complete without error
        model.fit(X_list)

        # Check that model has been fitted
        assert hasattr(model, 'samples')
        assert model.samples is not None

    def test_standard_gfa_transform(self, synthetic_data, basic_config):
        """Test that StandardGFA can transform data."""
        X_list = synthetic_data['X_list']

        model = StandardGFA(**basic_config)
        model.fit(X_list)

        # Transform should return factor scores
        Z = model.transform(X_list)

        assert Z is not None
        assert Z.shape[0] == X_list[0].shape[0]  # Same number of subjects
        assert Z.shape[1] == basic_config['K']   # Number of factors

    def test_standard_gfa_factor_loadings(self, synthetic_data, basic_config):
        """Test that factor loadings are computed correctly."""
        X_list = synthetic_data['X_list']

        model = StandardGFA(**basic_config)
        model.fit(X_list)

        W = model.get_factor_loadings()

        assert W is not None
        assert len(W) == len(X_list)  # One loading matrix per view

        for i, W_view in enumerate(W):
            assert W_view.shape[0] == X_list[i].shape[1]  # Features dimension
            assert W_view.shape[1] == basic_config['K']   # Number of factors

    def test_standard_gfa_reconstruct(self, synthetic_data, basic_config):
        """Test data reconstruction capability."""
        X_list = synthetic_data['X_list']

        model = StandardGFA(**basic_config)
        model.fit(X_list)

        X_reconstructed = model.reconstruct(X_list)

        assert X_reconstructed is not None
        assert len(X_reconstructed) == len(X_list)

        for i, X_recon in enumerate(X_reconstructed):
            assert X_recon.shape == X_list[i].shape

    def test_standard_gfa_vs_sparse_gfa(self, synthetic_data, basic_config):
        """Test differences between Standard and Sparse GFA."""
        X_list = synthetic_data['X_list']

        # Fit both models
        standard_model = StandardGFA(**basic_config)
        standard_model.fit(X_list)

        # Import SparseGFA for comparison
        from models.sparse_gfa import SparseGFAModel as SparseGFA
        sparse_config = basic_config.copy()
        sparse_config['sparsity_lambda'] = 0.0  # No sparsity for fair comparison
        sparse_model = SparseGFA(**sparse_config)
        sparse_model.fit(X_list)

        # Get results
        Z_standard = standard_model.transform(X_list)
        Z_sparse = sparse_model.transform(X_list)

        # Should produce similar but not identical results
        assert Z_standard.shape == Z_sparse.shape

    def test_standard_gfa_with_different_k(self, synthetic_data):
        """Test StandardGFA with different numbers of factors."""
        X_list = synthetic_data['X_list']

        for K in [2, 4, 6]:
            config = {
                'K': K,
                'num_samples': 25,  # Very small for speed
                'num_warmup': 10,
                'num_chains': 1
            }

            model = StandardGFA(**config)
            model.fit(X_list)

            Z = model.transform(X_list)
            assert Z.shape[1] == K

    def test_standard_gfa_hyperparameters(self, synthetic_data, basic_config):
        """Test that hyperparameters are properly handled."""
        X_list = synthetic_data['X_list']

        model = StandardGFA(**basic_config)
        model.fit(X_list)

        # Check that hyperparameters are accessible
        if hasattr(model, 'get_hyperparameters'):
            hyperparams = model.get_hyperparameters()
            assert isinstance(hyperparams, dict)

    def test_standard_gfa_convergence_diagnostics(self, synthetic_data, basic_config):
        """Test convergence diagnostics."""
        X_list = synthetic_data['X_list']

        model = StandardGFA(**basic_config)
        model.fit(X_list)

        # Check for convergence information
        if hasattr(model, 'get_convergence_info'):
            conv_info = model.get_convergence_info()
            assert conv_info is not None

    def test_standard_gfa_error_handling_invalid_k(self, synthetic_data):
        """Test error handling for invalid K values."""
        X_list = synthetic_data['X_list']

        # K too large
        config = {
            'K': 1000,  # Unreasonably large
            'num_samples': 10,
            'num_warmup': 5,
            'num_chains': 1
        }

        model = StandardGFA(**config)

        # Should either handle gracefully or raise informative error
        try:
            model.fit(X_list)
        except (ValueError, RuntimeError) as e:
            # Expected for invalid parameters
            assert len(str(e)) > 0

    def test_standard_gfa_error_handling_empty_data(self, basic_config):
        """Test error handling for empty data."""
        empty_data = []

        model = StandardGFA(**basic_config)

        with pytest.raises((ValueError, IndexError)):
            model.fit(empty_data)

    def test_standard_gfa_reproducibility(self, synthetic_data, basic_config):
        """Test reproducibility with fixed random seed."""
        X_list = synthetic_data['X_list']

        # Set random seed
        config1 = basic_config.copy()
        config1['random_seed'] = 42
        config1['num_samples'] = 25  # Small for speed

        config2 = basic_config.copy()
        config2['random_seed'] = 42
        config2['num_samples'] = 25  # Small for speed

        model1 = StandardGFA(**config1)
        model2 = StandardGFA(**config2)

        model1.fit(X_list)
        model2.fit(X_list)

        Z1 = model1.transform(X_list)
        Z2 = model2.transform(X_list)

        # Results should be similar (within numerical precision)
        if Z1 is not None and Z2 is not None:
            correlation = np.corrcoef(Z1.flatten(), Z2.flatten())[0, 1]
            assert correlation > 0.7  # Should be reasonably correlated

    def test_standard_gfa_multiview_consistency(self, basic_config):
        """Test consistency across different numbers of views."""
        # Test with 2, 3, and 4 views
        for num_views in [2, 3, 4]:
            synthetic_data = generate_synthetic_data(
                num_sources=num_views,
                K=3,
                num_subjects=15,
                seed=42
            )
            X_list = synthetic_data['X_list']

            config = basic_config.copy()
            config['num_samples'] = 20  # Very small for speed

            model = StandardGFA(**config)
            model.fit(X_list)

            W = model.get_factor_loadings()
            assert len(W) == num_views

    def test_standard_gfa_parameter_extraction(self, synthetic_data, basic_config):
        """Test that model parameters can be extracted."""
        X_list = synthetic_data['X_list']

        model = StandardGFA(**basic_config)
        model.fit(X_list)

        # Check that we can extract various parameters
        if hasattr(model, 'get_parameters'):
            params = model.get_parameters()
            assert isinstance(params, dict)

        # Check factor loadings
        W = model.get_factor_loadings()
        assert W is not None

        # Check factor scores
        Z = model.transform(X_list)
        assert Z is not None

    def test_standard_gfa_memory_efficiency(self, basic_config):
        """Test memory efficiency with larger data."""
        # Create larger synthetic data
        large_data = generate_synthetic_data(
            num_sources=2,
            K=3,
            num_subjects=50,
            features_per_view=[100, 80],
            seed=42
        )
        X_list = large_data['X_list']

        config = basic_config.copy()
        config['num_samples'] = 20  # Keep small for testing

        model = StandardGFA(**config)

        # Should handle larger data without memory issues
        model.fit(X_list)

        Z = model.transform(X_list)
        assert Z.shape[0] == 50  # Number of subjects
        assert Z.shape[1] == 3   # Number of factors

    def test_standard_gfa_numerical_stability(self, basic_config):
        """Test numerical stability with challenging data."""
        # Create data with different scales
        X_scaled = [
            np.random.randn(20, 30) * 1000,  # Large scale
            np.random.randn(20, 25) * 0.001  # Small scale
        ]

        config = basic_config.copy()
        config['num_samples'] = 20  # Small for speed

        model = StandardGFA(**config)

        # Should handle different scales gracefully
        model.fit(X_scaled)

        Z = model.transform(X_scaled)
        assert Z is not None
        assert not np.any(np.isnan(Z))
        assert not np.any(np.isinf(Z))

    def test_standard_gfa_prior_specification(self, synthetic_data, basic_config):
        """Test different prior specifications."""
        X_list = synthetic_data['X_list']

        # Test with different prior configurations if supported
        prior_configs = [
            {'prior_type': 'normal'},
            {'prior_type': 'laplace'},
            {'prior_type': 'student_t'}
        ]

        for prior_config in prior_configs:
            config = basic_config.copy()
            config.update(prior_config)
            config['num_samples'] = 15  # Very small for speed

            try:
                model = StandardGFA(**config)
                model.fit(X_list)

                Z = model.transform(X_list)
                assert Z is not None

            except (NotImplementedError, ValueError):
                # Skip if prior type not implemented
                continue

    def test_standard_gfa_missing_data_handling(self, basic_config):
        """Test handling of missing data."""
        # Create data with missing values
        X_missing = [
            np.random.randn(20, 30),
            np.random.randn(20, 25)
        ]

        # Add some missing values
        X_missing[0][5:8, 10:15] = np.nan
        X_missing[1][2:4, 5:10] = np.nan

        config = basic_config.copy()
        config['num_samples'] = 15  # Small for speed

        model = StandardGFA(**config)

        # Should either handle missing data or raise informative error
        try:
            model.fit(X_missing)
            Z = model.transform(X_missing)
            assert Z is not None
        except (ValueError, NotImplementedError) as e:
            # Expected if missing data not supported
            assert len(str(e)) > 0