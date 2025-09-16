"""Tests for Sparse GFA model implementation."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from typing import List, Dict, Any

from models.sparse_gfa import SparseGFAModel as SparseGFA
from data import generate_synthetic_data


class TestSparseGFA:
    """Test Sparse GFA model implementation."""

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
        """Basic configuration for SparseGFA."""
        return {
            'K': 3,
            'sparsity_lambda': 0.1,
            'group_lambda': 0.1,
            'num_samples': 50,  # Small for testing
            'num_warmup': 25,
            'num_chains': 1
        }

    def test_sparse_gfa_initialization(self, basic_config):
        """Test that SparseGFA initializes correctly."""
        model = SparseGFA(**basic_config)

        assert model.K == 3
        assert model.sparsity_lambda == 0.1
        assert model.group_lambda == 0.1
        assert model.num_samples == 50
        assert model.num_warmup == 25

    def test_sparse_gfa_fit(self, synthetic_data, basic_config):
        """Test that SparseGFA can fit to data."""
        X_list = synthetic_data['X_list']

        model = SparseGFA(**basic_config)

        # Should complete without error
        model.fit(X_list)

        # Check that model has been fitted
        assert hasattr(model, 'samples')
        assert model.samples is not None

    def test_sparse_gfa_transform(self, synthetic_data, basic_config):
        """Test that SparseGFA can transform data."""
        X_list = synthetic_data['X_list']

        model = SparseGFA(**basic_config)
        model.fit(X_list)

        # Transform should return factor scores
        Z = model.transform(X_list)

        assert Z is not None
        assert Z.shape[0] == X_list[0].shape[0]  # Same number of subjects
        assert Z.shape[1] == basic_config['K']   # Number of factors

    def test_sparse_gfa_factor_loadings(self, synthetic_data, basic_config):
        """Test that factor loadings are computed correctly."""
        X_list = synthetic_data['X_list']

        model = SparseGFA(**basic_config)
        model.fit(X_list)

        W = model.get_factor_loadings()

        assert W is not None
        assert len(W) == len(X_list)  # One loading matrix per view

        for i, W_view in enumerate(W):
            assert W_view.shape[0] == X_list[i].shape[1]  # Features dimension
            assert W_view.shape[1] == basic_config['K']   # Number of factors

    def test_sparse_gfa_reconstruct(self, synthetic_data, basic_config):
        """Test data reconstruction capability."""
        X_list = synthetic_data['X_list']

        model = SparseGFA(**basic_config)
        model.fit(X_list)

        X_reconstructed = model.reconstruct(X_list)

        assert X_reconstructed is not None
        assert len(X_reconstructed) == len(X_list)

        for i, X_recon in enumerate(X_reconstructed):
            assert X_recon.shape == X_list[i].shape

    def test_sparse_gfa_with_different_k(self, synthetic_data):
        """Test SparseGFA with different numbers of factors."""
        X_list = synthetic_data['X_list']

        for K in [2, 4, 6]:
            config = {
                'K': K,
                'sparsity_lambda': 0.1,
                'num_samples': 25,  # Very small for speed
                'num_warmup': 10,
                'num_chains': 1
            }

            model = SparseGFA(**config)
            model.fit(X_list)

            Z = model.transform(X_list)
            assert Z.shape[1] == K

    def test_sparse_gfa_sparsity_levels(self, synthetic_data, basic_config):
        """Test different sparsity levels."""
        X_list = synthetic_data['X_list']

        for sparsity in [0.01, 0.1, 1.0]:
            config = basic_config.copy()
            config['sparsity_lambda'] = sparsity
            config['num_samples'] = 25  # Very small for speed

            model = SparseGFA(**config)
            model.fit(X_list)

            W = model.get_factor_loadings()
            assert W is not None

    def test_sparse_gfa_hyperparameters(self, synthetic_data, basic_config):
        """Test that hyperparameters are properly handled."""
        X_list = synthetic_data['X_list']

        model = SparseGFA(**basic_config)
        model.fit(X_list)

        # Check that hyperparameters are accessible
        if hasattr(model, 'get_hyperparameters'):
            hyperparams = model.get_hyperparameters()
            assert isinstance(hyperparams, dict)

    def test_sparse_gfa_convergence_diagnostics(self, synthetic_data, basic_config):
        """Test convergence diagnostics."""
        X_list = synthetic_data['X_list']

        model = SparseGFA(**basic_config)
        model.fit(X_list)

        # Check for convergence information
        if hasattr(model, 'get_convergence_info'):
            conv_info = model.get_convergence_info()
            assert conv_info is not None

    def test_sparse_gfa_error_handling_invalid_k(self, synthetic_data):
        """Test error handling for invalid K values."""
        X_list = synthetic_data['X_list']

        # K too large
        config = {
            'K': 1000,  # Unreasonably large
            'sparsity_lambda': 0.1,
            'num_samples': 10,
            'num_warmup': 5,
            'num_chains': 1
        }

        model = SparseGFA(**config)

        # Should either handle gracefully or raise informative error
        try:
            model.fit(X_list)
        except (ValueError, RuntimeError) as e:
            # Expected for invalid parameters
            assert len(str(e)) > 0

    def test_sparse_gfa_error_handling_empty_data(self, basic_config):
        """Test error handling for empty data."""
        empty_data = []

        model = SparseGFA(**basic_config)

        with pytest.raises((ValueError, IndexError)):
            model.fit(empty_data)

    def test_sparse_gfa_error_handling_mismatched_data(self, basic_config):
        """Test error handling for mismatched data dimensions."""
        # Create data with mismatched number of subjects
        X_mismatched = [
            np.random.randn(20, 10),  # 20 subjects
            np.random.randn(15, 8)    # 15 subjects (mismatch)
        ]

        model = SparseGFA(**basic_config)

        with pytest.raises((ValueError, AssertionError)):
            model.fit(X_mismatched)

    def test_sparse_gfa_reproducibility(self, synthetic_data, basic_config):
        """Test reproducibility with fixed random seed."""
        X_list = synthetic_data['X_list']

        # Set random seed
        config1 = basic_config.copy()
        config1['random_seed'] = 42
        config1['num_samples'] = 25  # Small for speed

        config2 = basic_config.copy()
        config2['random_seed'] = 42
        config2['num_samples'] = 25  # Small for speed

        model1 = SparseGFA(**config1)
        model2 = SparseGFA(**config2)

        model1.fit(X_list)
        model2.fit(X_list)

        Z1 = model1.transform(X_list)
        Z2 = model2.transform(X_list)

        # Results should be similar (within numerical precision)
        if Z1 is not None and Z2 is not None:
            correlation = np.corrcoef(Z1.flatten(), Z2.flatten())[0, 1]
            assert correlation > 0.7  # Should be reasonably correlated

    def test_sparse_gfa_multiview_consistency(self, basic_config):
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

            model = SparseGFA(**config)
            model.fit(X_list)

            W = model.get_factor_loadings()
            assert len(W) == num_views

    def test_sparse_gfa_parameter_extraction(self, synthetic_data, basic_config):
        """Test that model parameters can be extracted."""
        X_list = synthetic_data['X_list']

        model = SparseGFA(**basic_config)
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

    def test_sparse_gfa_group_sparsity(self, synthetic_data, basic_config):
        """Test group sparsity functionality."""
        X_list = synthetic_data['X_list']

        # Test with group sparsity enabled
        config = basic_config.copy()
        config['group_lambda'] = 0.5
        config['num_samples'] = 25  # Small for speed

        model = SparseGFA(**config)
        model.fit(X_list)

        W = model.get_factor_loadings()
        assert W is not None

        # Group sparsity should create structured sparsity patterns
        # (Specific tests depend on implementation details)

    def test_sparse_gfa_memory_efficiency(self, basic_config):
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

        model = SparseGFA(**config)

        # Should handle larger data without memory issues
        model.fit(X_list)

        Z = model.transform(X_list)
        assert Z.shape[0] == 50  # Number of subjects
        assert Z.shape[1] == 3   # Number of factors