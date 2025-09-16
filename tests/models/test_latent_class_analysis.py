"""Tests for Latent Class Analysis model."""

from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest

from models.latent_class_analysis import LatentClassAnalysisModel


class TestLatentClassAnalysisModel:
    """Test LCA model implementation."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration object."""
        config = Mock()
        config.K = 3  # 3 latent classes
        config.num_sources = 2
        return config

    @pytest.fixture
    def hypers(self):
        """Sample hyperparameters."""
        return {
            "Dm": [10, 15],  # Two sources with 10 and 15 features
            "a_sigma": 1.0,
            "b_sigma": 1.0,
        }

    @pytest.fixture
    def sample_data(self):
        """Sample multi-view data."""
        np.random.seed(42)
        X1 = np.random.randn(20, 10)  # 20 subjects, 10 features
        X2 = np.random.randn(20, 15)  # 20 subjects, 15 features
        return [jnp.array(X1), jnp.array(X2)]

    def test_model_initialization(self, mock_config, hypers):
        """Test LCA model initialization."""
        model = LatentClassAnalysisModel(mock_config, hypers)

        assert model.K == 3
        assert model.num_sources == 2
        assert model.config == mock_config
        assert model.hypers == hypers

    def test_get_model_name(self, mock_config, hypers):
        """Test model name method."""
        model = LatentClassAnalysisModel(mock_config, hypers)
        assert model.get_model_name() == "LatentClassAnalysis"

    def test_get_memory_warning(self, mock_config, hypers):
        """Test memory warning method."""
        model = LatentClassAnalysisModel(mock_config, hypers)
        warning = model.get_memory_warning()

        assert "WARNING" in warning
        assert "computationally intensive" in warning
        assert "memory" in warning

    def test_estimate_memory_requirements(self, mock_config, hypers):
        """Test memory estimation method."""
        model = LatentClassAnalysisModel(mock_config, hypers)

        N, D, K = 100, 25, 3
        memory_est = model.estimate_memory_requirements(N, D, K)

        # Check that all expected keys are present
        expected_keys = [
            "total_gb",
            "parameters_gb",
            "gradients_gb",
            "samples_gb",
            "recommended_min_gpu_gb",
        ]
        for key in expected_keys:
            assert key in memory_est
            assert isinstance(memory_est[key], float)
            assert memory_est[key] > 0

        # Check that total memory is sum of components
        components = (
            memory_est["parameters_gb"]
            + memory_est["gradients_gb"]
            + memory_est["samples_gb"]
        )
        assert abs(memory_est["total_gb"] - components) < 0.01

    def test_model_call_structure(self, mock_config, hypers, sample_data):
        """Test that model call method has correct structure (without running inference)."""
        model = LatentClassAnalysisModel(mock_config, hypers)

        # This test just checks that the method exists and accepts the right arguments
        # We don't actually run inference as it would be computationally expensive
        assert hasattr(model, "__call__")
        assert callable(model)

        # Check that calling the model doesn't immediately fail
        # (actual inference would require numpyro context)
        try:
            # This will fail due to missing numpyro context, but should not fail due
            # to method signature
            model(sample_data)
        except Exception as e:
            # Expected to fail without proper numpyro context
            # Just ensure it's not a signature error
            assert (
                "plate" in str(e) or "sample" in str(e) or "numpyro" in str(e).lower()
            )

    def test_computational_warning_in_docstring(self, mock_config, hypers):
        """Test that computational warning is included in model docstring."""
        model = LatentClassAnalysisModel(mock_config, hypers)
        docstring = model.__class__.__doc__

        assert "computationally intensive" in docstring
        assert "memory" in docstring
        assert "GPU" in docstring
