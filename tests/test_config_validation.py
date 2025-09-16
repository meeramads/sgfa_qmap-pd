"""Tests for configuration validation system."""

import tempfile
from pathlib import Path

import pytest

from core.config_schema import (
    ConfigurationValidator,
    ConfigValidationError,
    DataConfig,
    ExperimentsConfig,
    ModelConfig,
)
from core.config_utils import get_default_configuration, validate_configuration


class TestConfigValidation:
    """Test configuration validation system."""

    def test_valid_configuration(self):
        """Test validation of valid configuration."""
        config = {
            "data": {"data_dir": "."},  # Current directory exists
            "experiments": {"base_output_dir": "./results"},
            "model": {
                "model_type": "sparse_gfa",
                "K": 5,
                "num_samples": 1000,
                "sparsity_lambda": 0.1,
            },
        }

        # Should not raise any errors
        validated_config = validate_configuration(config)
        assert validated_config is not None

    def test_invalid_model_type(self):
        """Test validation with invalid model type."""
        config = {
            "data": {"data_dir": "."},
            "experiments": {"base_output_dir": "./results"},
            "model": {"model_type": "invalid_model", "K": 5},
        }

        with pytest.raises(ConfigValidationError):
            validate_configuration(config)

    def test_invalid_k_value(self):
        """Test validation with invalid K value."""
        config = {
            "data": {"data_dir": "."},
            "experiments": {"base_output_dir": "./results"},
            "model": {"model_type": "sparse_gfa", "K": 0},  # Invalid
        }

        with pytest.raises(ConfigValidationError):
            validate_configuration(config)

    def test_missing_sparsity_lambda(self):
        """Test that sparse_gfa without sparsity_lambda gets fixed."""
        config = {
            "data": {"data_dir": "."},
            "experiments": {"base_output_dir": "./results"},
            "model": {
                "model_type": "sparse_gfa",
                "K": 5,
                # Missing sparsity_lambda
            },
        }

        # Should automatically add sparsity_lambda
        validated_config = validate_configuration(config)
        assert validated_config["model"]["sparsity_lambda"] == 0.1

    def test_merge_with_defaults(self):
        """Test merging partial configuration with defaults."""
        partial_config = {"model": {"K": 3}}

        merged_config = ConfigurationValidator.merge_with_defaults(partial_config)

        # Should have defaults for other sections
        assert "data" in merged_config
        assert "experiments" in merged_config
        assert merged_config["model"]["K"] == 3  # User value preserved
        assert "model_type" in merged_config["model"]  # Default added

    def test_data_config_validation(self):
        """Test DataConfig validation."""
        # Valid config
        valid_config = DataConfig(data_dir=".")
        errors = valid_config.validate()
        assert len(errors) == 0

        # Invalid config - missing directory
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = str(Path(tmpdir) / "nonexistent")
            invalid_config = DataConfig(data_dir=invalid_path)
            errors = invalid_config.validate()
            assert len(errors) > 0
            assert "does not exist" in errors[0]

    def test_experiments_config_validation(self):
        """Test ExperimentsConfig validation."""
        # Valid config
        valid_config = ExperimentsConfig(base_output_dir="./results")
        errors = valid_config.validate()
        assert len(errors) == 0

        # Invalid config - too many parallel jobs
        invalid_config = ExperimentsConfig(
            base_output_dir="./results", max_parallel_jobs=100  # Too many
        )
        errors = invalid_config.validate()
        assert len(errors) > 0
        assert "should not exceed" in errors[0]

    def test_model_config_validation(self):
        """Test ModelConfig validation."""
        # Valid sparse GFA config
        valid_config = ModelConfig(model_type="sparse_gfa", K=5, sparsity_lambda=0.1)
        errors = valid_config.validate()
        assert len(errors) == 0

        # Invalid config - negative sparsity
        invalid_config = ModelConfig(
            model_type="sparse_gfa", K=5, sparsity_lambda=-0.1  # Invalid
        )
        errors = invalid_config.validate()
        assert len(errors) > 0
        assert "must be >= 0" in errors[0]

    def test_mcmc_parameter_validation(self):
        """Test MCMC parameter validation."""
        # Invalid - warmup >= samples
        invalid_config = ModelConfig(
            model_type="standard_gfa",
            K=3,
            num_samples=100,
            num_warmup=100,  # Should be < num_samples
        )
        errors = invalid_config.validate()
        assert len(errors) > 0
        assert "must be < num_samples" in errors[0]

    def test_memory_estimation(self):
        """Test memory requirement estimation."""
        model_config = {"K": 10, "num_samples": 2000, "num_chains": 4}

        estimated_memory = ConfigurationValidator._estimate_memory_requirements(
            model_config
        )
        assert estimated_memory > 0
        assert isinstance(estimated_memory, float)

    def test_cross_section_validation(self):
        """Test validation across configuration sections."""
        # Config with high memory requirements
        config = {
            "data": {"data_dir": "."},
            "experiments": {"base_output_dir": "./results"},
            "model": {
                "model_type": "sparse_gfa",
                "K": 20,
                "num_samples": 10000,
                "num_chains": 4,
                "sparsity_lambda": 0.1,
            },
            "system": {"memory_limit_gb": 1.0},  # Low limit
        }

        errors = ConfigurationValidator.validate_configuration(config)
        # Should have memory warning/error
        memory_errors = [e for e in errors if "memory" in e.lower()]
        assert len(memory_errors) > 0

    def test_get_default_configuration(self):
        """Test getting default configuration."""
        default_config = get_default_configuration()

        # Should have all required sections
        required_sections = ["data", "experiments", "model", "preprocessing"]
        for section in required_sections:
            assert section in default_config

        # Should be valid
        validated_config = validate_configuration(default_config)
        assert validated_config is not None

    def test_configuration_fixes(self):
        """Test automatic configuration fixes."""
        # Config with fixable issues
        config = {
            "data": {"data_dir": "./data"},  # Relative path
            "experiments": {"base_output_dir": "./results"},  # Relative path
            "model": {
                "model_type": "sparse_gfa",
                "K": 5,
                # Missing sparsity_lambda - should be auto-added
            },
        }

        fixed_config = ConfigurationValidator._apply_fixes(config)

        # Should have absolute paths
        assert Path(fixed_config["data"]["data_dir"]).is_absolute()
        assert Path(fixed_config["experiments"]["base_output_dir"]).is_absolute()

        # Should have sparsity_lambda
        assert "sparsity_lambda" in fixed_config["model"]

    def test_validation_error_messages(self):
        """Test that validation error messages are informative."""
        config = {
            "data": {"data_dir": ""},  # Empty path
            "experiments": {"base_output_dir": "./results"},
            "model": {
                "model_type": "invalid_type",
                "K": -1,  # Invalid
                "num_samples": 0,  # Invalid
            },
        }

        try:
            validate_configuration(config)
            assert False, "Should have raised ConfigValidationError"
        except ConfigValidationError as e:
            error_message = str(e)
            # Should contain specific error information
            assert "data_dir is required" in error_message or "data:" in error_message
            assert "model_type" in error_message
            assert "K" in error_message

    def test_preprocessing_config_validation(self):
        """Test preprocessing configuration validation."""
        config = {
            "data": {"data_dir": "."},
            "experiments": {"base_output_dir": "./results"},
            "preprocessing": {
                "strategy": "invalid_strategy",  # Invalid
                "variance_threshold": 1.5,  # Invalid (> 1)
            },
        }

        errors = ConfigurationValidator.validate_configuration(config)
        preprocessing_errors = [e for e in errors if "preprocessing:" in e]
        assert len(preprocessing_errors) > 0

    def test_empty_configuration(self):
        """Test handling of empty configuration."""
        empty_config = {}

        # Should be fixable with defaults
        try:
            validated_config = validate_configuration(empty_config)
            assert validated_config is not None
            # Should have all required sections
            assert "data" in validated_config
            assert "experiments" in validated_config
        except ConfigValidationError:
            # If not fixable, should get descriptive error
            pass

    def test_partial_configuration_sections(self):
        """Test handling of partial configuration sections."""
        partial_config = {
            "data": {"data_dir": "."},
            "model": {"K": 3},  # Incomplete model config
        }

        # Should merge with defaults and validate
        validated_config = validate_configuration(partial_config)

        assert validated_config["model"]["K"] == 3  # User value preserved
        assert "model_type" in validated_config["model"]  # Default added
        assert "experiments" in validated_config  # Default section added

    def test_configuration_warnings(self):
        """Test configuration warning system."""
        from core.config_utils import check_configuration_warnings

        # Config that should generate warnings
        config = {
            "model": {
                "K": 25,  # Large number of factors
                "num_samples": 15000,  # Large number of samples
            },
            "system": {
                "use_gpu": True  # GPU requested (may not be available in test env)
            },
        }

        warnings = check_configuration_warnings(config)

        # Should have warnings about performance
        assert any("factors" in w for w in warnings)
        assert any("samples" in w for w in warnings)

    def test_schema_dataclass_creation(self):
        """Test creating schema dataclasses from dictionaries."""
        config_dict = {
            "data": {"data_dir": "."},
            "experiments": {"base_output_dir": "./results", "max_parallel_jobs": 2},
            "model": {"model_type": "sparse_gfa", "K": 5, "sparsity_lambda": 0.2},
        }

        config_objects = ConfigurationValidator.create_from_dict(config_dict)

        # Should create appropriate dataclass objects
        assert isinstance(config_objects["data"], DataConfig)
        assert isinstance(config_objects["experiments"], ExperimentsConfig)
        assert isinstance(config_objects["model"], ModelConfig)

        # Should preserve values
        assert config_objects["data"].data_dir == "."
        assert config_objects["experiments"].max_parallel_jobs == 2
        assert config_objects["model"].sparsity_lambda == 0.2
