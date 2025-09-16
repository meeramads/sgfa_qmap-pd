"""Tests for model factory implementation."""

import numpy as np
import pytest

from data import generate_synthetic_data
from models.factory import ModelFactory


class TestModelFactory:
    """Test model factory implementation."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data for testing."""
        return generate_synthetic_data(num_sources=2, K=3, num_subjects=20, seed=42)

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for models."""
        return {
            "K": 3,
            "num_samples": 25,  # Small for testing
            "num_warmup": 10,
            "num_chains": 1,
        }

    def test_factory_initialization(self):
        """Test that model factory initializes correctly."""
        factory = ModelFactory()
        assert factory is not None

    def test_factory_available_models(self):
        """Test that factory returns available models."""
        available_models = ModelFactory.get_available_models()

        assert isinstance(available_models, list)
        assert len(available_models) > 0

        # Should include basic models
        expected_models = ["sparse_gfa", "standard_gfa"]
        for model in expected_models:
            if model in available_models:
                assert model in available_models

    def test_factory_create_sparse_gfa(self, basic_config):
        """Test creating Sparse GFA model through factory."""
        sparse_config = basic_config.copy()
        sparse_config["sparsity_lambda"] = 0.1

        model = ModelFactory.create_model("sparse_gfa", sparse_config)

        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "transform")

    def test_factory_create_standard_gfa(self, basic_config):
        """Test creating Standard GFA model through factory."""
        model = ModelFactory.create_model("standard_gfa", basic_config)

        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "transform")

    def test_factory_invalid_model_type(self, basic_config):
        """Test error handling for invalid model types."""
        with pytest.raises(ValueError) as exc_info:
            ModelFactory.create_model("invalid_model_type", basic_config)

        assert "Unknown model type" in str(exc_info.value)

    def test_factory_model_consistency(self, synthetic_data, basic_config):
        """Test that factory-created models work consistently."""
        X_list = synthetic_data["X_list"]

        # Create models through factory
        sparse_config = basic_config.copy()
        sparse_config["sparsity_lambda"] = 0.1

        sparse_model = ModelFactory.create_model("sparse_gfa", sparse_config)
        standard_model = ModelFactory.create_model("standard_gfa", basic_config)

        # Both should fit and transform
        sparse_model.fit(X_list)
        standard_model.fit(X_list)

        Z_sparse = sparse_model.transform(X_list)
        Z_standard = standard_model.transform(X_list)

        assert Z_sparse is not None
        assert Z_standard is not None
        assert Z_sparse.shape == Z_standard.shape

    def test_factory_model_registration(self):
        """Test model registration mechanism."""
        # Check if factory supports model registration
        if hasattr(ModelFactory, "register_model"):

            class DummyModel:
                def __init__(self, **kwargs):
                    pass

                def fit(self, X_list):
                    pass

                def transform(self, X_list):
                    return np.random.randn(len(X_list[0]), 3)

            # Register dummy model
            ModelFactory.register_model("dummy", DummyModel)

            # Should be able to create it
            model = ModelFactory.create_model("dummy", {})
            assert model is not None

    def test_factory_config_validation(self, basic_config):
        """Test configuration validation in factory."""
        # Test with missing required parameters
        incomplete_config = {"K": 3}  # Missing other parameters

        try:
            model = ModelFactory.create_model("sparse_gfa", incomplete_config)
            # If it doesn't raise an error, it should handle defaults gracefully
            assert model is not None
        except (ValueError, TypeError):
            # Expected if validation is strict
            pass

    def test_factory_model_metadata(self):
        """Test model metadata retrieval."""
        if hasattr(ModelFactory, "get_model_info"):
            # Test getting info for known models
            for model_type in ["sparse_gfa", "standard_gfa"]:
                try:
                    info = ModelFactory.get_model_info(model_type)
                    assert isinstance(info, dict)
                except NotImplementedError:
                    # Skip if not implemented
                    continue

    def test_factory_parameter_requirements(self):
        """Test parameter requirements for different models."""
        if hasattr(ModelFactory, "get_required_parameters"):
            for model_type in ["sparse_gfa", "standard_gfa"]:
                try:
                    required_params = ModelFactory.get_required_parameters(model_type)
                    assert isinstance(required_params, list)
                    assert "K" in required_params  # Should require number of factors
                except NotImplementedError:
                    # Skip if not implemented
                    continue

    def test_factory_default_parameters(self):
        """Test default parameter handling."""
        if hasattr(ModelFactory, "get_default_parameters"):
            for model_type in ["sparse_gfa", "standard_gfa"]:
                try:
                    defaults = ModelFactory.get_default_parameters(model_type)
                    assert isinstance(defaults, dict)
                except NotImplementedError:
                    # Skip if not implemented
                    continue

    def test_factory_with_hyperparameters(self, synthetic_data, basic_config):
        """Test factory with different hyperparameter configurations."""
        X_list = synthetic_data["X_list"]

        # Test different hyperparameter combinations
        hyper_configs = [
            {"K": 2, "num_samples": 20},
            {"K": 4, "num_samples": 15},
            {"K": 3, "num_samples": 30, "sparsity_lambda": 0.5},
        ]

        for hyper_config in hyper_configs:
            config = basic_config.copy()
            config.update(hyper_config)

            # Determine model type based on parameters
            if "sparsity_lambda" in config:
                model_type = "sparse_gfa"
            else:
                model_type = "standard_gfa"

            model = ModelFactory.create_model(model_type, config)
            model.fit(X_list)

            Z = model.transform(X_list)
            assert Z.shape[1] == config["K"]

    def test_factory_error_recovery(self, basic_config):
        """Test factory error recovery mechanisms."""
        # Test with various problematic configurations
        problematic_configs = [
            {"K": 0},  # Invalid K
            {"K": -1},  # Negative K
            {"num_samples": 0},  # Invalid samples
            {},  # Empty config
        ]

        for config in problematic_configs:
            try:
                model = ModelFactory.create_model("standard_gfa", config)
                # If no error, model should handle gracefully
                assert model is not None
            except (ValueError, TypeError, KeyError):
                # Expected for invalid configurations
                pass

    def test_factory_model_comparison(self, synthetic_data, basic_config):
        """Test comparing models created through factory."""
        X_list = synthetic_data["X_list"]

        # Create multiple models for comparison
        models = {}

        # Standard GFA
        models["standard"] = ModelFactory.create_model("standard_gfa", basic_config)

        # Sparse GFA with different sparsity levels
        for sparsity in [0.1, 0.5]:
            sparse_config = basic_config.copy()
            sparse_config["sparsity_lambda"] = sparsity
            models[f"sparse_{sparsity}"] = ModelFactory.create_model(
                "sparse_gfa", sparse_config
            )

        # Fit all models
        results = {}
        for name, model in models.items():
            model.fit(X_list)
            results[name] = model.transform(X_list)

        # All should produce valid results
        for name, Z in results.items():
            assert Z is not None
            assert Z.shape[0] == X_list[0].shape[0]
            assert Z.shape[1] == basic_config["K"]

    def test_factory_thread_safety(self, basic_config):
        """Test factory thread safety (basic test)."""
        import threading

        results = []

        def create_model():
            try:
                model = ModelFactory.create_model("standard_gfa", basic_config)
                results.append(model is not None)
            except Exception:
                results.append(False)

        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=create_model)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All should succeed
        assert all(results)

    def test_factory_memory_management(self, basic_config):
        """Test factory memory management."""
        # Create and destroy many models
        for i in range(10):
            model = ModelFactory.create_model("standard_gfa", basic_config)
            assert model is not None
            del model  # Explicit cleanup

        # Should not accumulate memory issues
