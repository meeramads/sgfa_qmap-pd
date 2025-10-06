"""Tests for core.validation_utils module."""

import numpy as np
import pytest

from core.validation_utils import (
    EXPERIMENT_CONFIG_SCHEMA,
    ParameterValidator,
    ResultValidator,
    validate_config_structure,
    validate_data_types,
    validate_parameters,
)


class TestValidateParametersDecorator:
    """Test validate_parameters decorator."""

    def test_simple_validation_passes(self):
        """Test simple validation that passes."""
        @validate_parameters(x=lambda val: val > 0)
        def func(x):
            return x * 2

        result = func(5)
        assert result == 10

    def test_simple_validation_fails(self):
        """Test simple validation that fails."""
        @validate_parameters(x=lambda val: val > 0)
        def func(x):
            return x * 2

        with pytest.raises(ValueError) as exc_info:
            func(-1)

        assert "Validation failed" in str(exc_info.value)

    def test_validation_with_custom_error_message(self):
        """Test validation with custom error message."""
        @validate_parameters(
            x=(lambda val: val > 0, "x must be positive")
        )
        def func(x):
            return x

        with pytest.raises(ValueError) as exc_info:
            func(-1)

        assert "x must be positive" in str(exc_info.value)

    def test_multiple_parameters(self):
        """Test validation of multiple parameters."""
        @validate_parameters(
            x=lambda val: val > 0,
            y=lambda val: val < 100
        )
        def func(x, y):
            return x + y

        # Should pass
        result = func(5, 50)
        assert result == 55

        # Should fail on x
        with pytest.raises(ValueError):
            func(-1, 50)

        # Should fail on y
        with pytest.raises(ValueError):
            func(5, 150)

    def test_validation_with_kwargs(self):
        """Test validation with keyword arguments."""
        @validate_parameters(x=lambda val: val > 0)
        def func(x, y=10):
            return x + y

        result = func(x=5, y=20)
        assert result == 25

        with pytest.raises(ValueError):
            func(x=-1, y=20)

    def test_validation_skips_missing_parameters(self):
        """Test that validation is skipped for missing optional parameters."""
        @validate_parameters(x=lambda val: val > 0)
        def func(x=None):
            return x

        # Should not validate None default
        result = func()
        assert result is None


class TestValidateDataTypesDecorator:
    """Test validate_data_types decorator."""

    def test_simple_type_validation(self):
        """Test simple type validation."""
        @validate_data_types(x=int, y=str)
        def func(x, y):
            return f"{y}: {x}"

        result = func(5, "value")
        assert result == "value: 5"

    def test_type_validation_fails(self):
        """Test type validation failure."""
        @validate_data_types(x=int)
        def func(x):
            return x

        with pytest.raises(TypeError) as exc_info:
            func("not an int")

        assert "must be of type" in str(exc_info.value)

    def test_numpy_array_type_validation(self):
        """Test numpy array type validation."""
        @validate_data_types(data=np.ndarray)
        def func(data):
            return data.shape

        arr = np.array([1, 2, 3])
        result = func(arr)
        assert result == (3,)

        with pytest.raises(TypeError):
            func([1, 2, 3])  # List, not ndarray

    def test_tuple_of_types(self):
        """Test validation with tuple of allowed types."""
        @validate_data_types(x=(int, float))
        def func(x):
            return x * 2

        assert func(5) == 10
        assert func(5.5) == 11.0

        with pytest.raises(TypeError):
            func("string")

    def test_optional_parameter(self):
        """Test validation with optional parameters (allow None)."""
        @validate_data_types(config=(dict, type(None)))
        def func(config=None):
            return config

        assert func({"key": "value"}) == {"key": "value"}
        assert func(None) is None
        assert func() is None


class TestParameterValidator:
    """Test ParameterValidator class."""

    def test_validate_positive_success(self):
        """Test validate_positive with valid value."""
        # Should not raise
        ParameterValidator.validate_positive(5, "test_param")
        ParameterValidator.validate_positive(0.01, "test_param")

    def test_validate_positive_failure(self):
        """Test validate_positive with invalid value."""
        with pytest.raises(ValueError) as exc_info:
            ParameterValidator.validate_positive(-1, "test_param")

        assert "test_param must be positive" in str(exc_info.value)

    def test_validate_positive_zero_fails(self):
        """Test that zero fails positive validation."""
        with pytest.raises(ValueError):
            ParameterValidator.validate_positive(0, "test_param")

    def test_validate_range_within_bounds(self):
        """Test validate_range with value within bounds."""
        # Should not raise
        ParameterValidator.validate_range(5, min_val=0, max_val=10, name="value")

    def test_validate_range_below_minimum(self):
        """Test validate_range with value below minimum."""
        with pytest.raises(ValueError) as exc_info:
            ParameterValidator.validate_range(-1, min_val=0, name="value")

        assert "must be >= 0" in str(exc_info.value)

    def test_validate_range_above_maximum(self):
        """Test validate_range with value above maximum."""
        with pytest.raises(ValueError) as exc_info:
            ParameterValidator.validate_range(15, max_val=10, name="value")

        assert "must be <= 10" in str(exc_info.value)

    def test_validate_range_only_min(self):
        """Test validate_range with only minimum specified."""
        ParameterValidator.validate_range(100, min_val=0, name="value")

        with pytest.raises(ValueError):
            ParameterValidator.validate_range(-1, min_val=0, name="value")

    def test_validate_range_only_max(self):
        """Test validate_range with only maximum specified."""
        ParameterValidator.validate_range(5, max_val=10, name="value")

        with pytest.raises(ValueError):
            ParameterValidator.validate_range(15, max_val=10, name="value")

    def test_validate_array_shape_exact_match(self):
        """Test validate_array_shape with exact shape match."""
        arr = np.zeros((10, 5))

        # Should not raise
        ParameterValidator.validate_array_shape(arr, expected_shape=(10, 5), name="array")

    def test_validate_array_shape_mismatch(self):
        """Test validate_array_shape with shape mismatch."""
        arr = np.zeros((10, 5))

        with pytest.raises(ValueError) as exc_info:
            ParameterValidator.validate_array_shape(arr, expected_shape=(5, 10), name="array")

        assert "shape must be (5, 10)" in str(exc_info.value)

    def test_validate_array_min_dims(self):
        """Test validate_array_shape with minimum dimensions."""
        arr = np.zeros((10, 5, 3))

        ParameterValidator.validate_array_shape(arr, min_dims=2, name="array")

        with pytest.raises(ValueError) as exc_info:
            ParameterValidator.validate_array_shape(arr, min_dims=4, name="array")

        assert "at least 4 dimensions" in str(exc_info.value)

    def test_validate_array_max_dims(self):
        """Test validate_array_shape with maximum dimensions."""
        arr = np.zeros((10, 5))

        ParameterValidator.validate_array_shape(arr, max_dims=3, name="array")

        with pytest.raises(ValueError) as exc_info:
            ParameterValidator.validate_array_shape(arr, max_dims=1, name="array")

        assert "at most 1 dimensions" in str(exc_info.value)

    def test_validate_list_type_success(self):
        """Test validate_list_type with correct types."""
        lst = [1, 2, 3, 4]

        # Should not raise
        ParameterValidator.validate_list_type(lst, int, name="numbers")

    def test_validate_list_type_failure(self):
        """Test validate_list_type with wrong types."""
        lst = [1, 2, "three", 4]

        with pytest.raises(TypeError) as exc_info:
            ParameterValidator.validate_list_type(lst, int, name="numbers")

        assert "numbers[2]" in str(exc_info.value)
        assert "must be of type int" in str(exc_info.value)

    def test_validate_list_type_not_a_list(self):
        """Test validate_list_type with non-list input."""
        with pytest.raises(TypeError) as exc_info:
            ParameterValidator.validate_list_type("not a list", str, name="items")

        assert "must be a list" in str(exc_info.value)

    def test_validate_dict_keys_success(self):
        """Test validate_dict_keys with all required keys."""
        dct = {"key1": 1, "key2": 2}

        # Should not raise
        ParameterValidator.validate_dict_keys(dct, required_keys=["key1", "key2"], name="config")

    def test_validate_dict_keys_missing_required(self):
        """Test validate_dict_keys with missing required keys."""
        dct = {"key1": 1}

        with pytest.raises(ValueError) as exc_info:
            ParameterValidator.validate_dict_keys(
                dct, required_keys=["key1", "key2"], name="config"
            )

        assert "missing required keys" in str(exc_info.value)
        assert "key2" in str(exc_info.value)

    def test_validate_dict_keys_unexpected_keys(self):
        """Test validate_dict_keys with unexpected keys."""
        dct = {"key1": 1, "key2": 2, "unexpected": 3}

        with pytest.raises(ValueError) as exc_info:
            ParameterValidator.validate_dict_keys(
                dct,
                required_keys=["key1"],
                optional_keys=["key2"],
                name="config"
            )

        assert "unexpected keys" in str(exc_info.value)
        assert "unexpected" in str(exc_info.value)

    def test_validate_dict_keys_not_a_dict(self):
        """Test validate_dict_keys with non-dict input."""
        with pytest.raises(TypeError) as exc_info:
            ParameterValidator.validate_dict_keys(
                "not a dict", required_keys=[], name="config"
            )

        assert "must be a dictionary" in str(exc_info.value)


class TestResultValidator:
    """Test ResultValidator class."""

    def test_validate_experiment_result_success(self):
        """Test validate_experiment_result with valid result."""
        result = {
            "success": True,
            "experiment_name": "test_exp",
            "data": [1, 2, 3]
        }

        # Should not raise
        ResultValidator.validate_experiment_result(result)

    def test_validate_experiment_result_missing_fields(self):
        """Test validate_experiment_result with missing required fields."""
        result = {"data": [1, 2, 3]}

        with pytest.raises(ValueError) as exc_info:
            ResultValidator.validate_experiment_result(result)

        assert "missing required fields" in str(exc_info.value)

    def test_validate_experiment_result_custom_fields(self):
        """Test validate_experiment_result with custom required fields."""
        result = {"custom_field": "value"}

        with pytest.raises(ValueError):
            ResultValidator.validate_experiment_result(result, required_fields=["custom_field", "other"])

    def test_validate_experiment_result_invalid_success_type(self):
        """Test validate_experiment_result with invalid success field type."""
        result = {"success": "true", "experiment_name": "test"}

        with pytest.raises(TypeError) as exc_info:
            ResultValidator.validate_experiment_result(result)

        assert "'success' field must be boolean" in str(exc_info.value)

    def test_validate_experiment_result_not_dict(self):
        """Test validate_experiment_result with non-dict input."""
        with pytest.raises(TypeError) as exc_info:
            ResultValidator.validate_experiment_result("not a dict")

        assert "must be a dictionary" in str(exc_info.value)

    def test_validate_mcmc_samples_success(self):
        """Test validate_mcmc_samples with valid samples."""
        samples = {
            "W": np.random.randn(100, 10),
            "Z": np.random.randn(100, 5),
            "tau": np.random.randn(100)
        }

        # Should not raise
        ResultValidator.validate_mcmc_samples(samples)

    def test_validate_mcmc_samples_not_dict(self):
        """Test validate_mcmc_samples with non-dict input."""
        with pytest.raises(TypeError) as exc_info:
            ResultValidator.validate_mcmc_samples([1, 2, 3])

        assert "must be a dictionary" in str(exc_info.value)

    def test_validate_mcmc_samples_empty(self):
        """Test validate_mcmc_samples with empty dict."""
        with pytest.raises(ValueError) as exc_info:
            ResultValidator.validate_mcmc_samples({})

        assert "empty" in str(exc_info.value)

    def test_validate_mcmc_samples_non_array_values(self):
        """Test validate_mcmc_samples with non-array values."""
        samples = {"W": [1, 2, 3]}  # List instead of array

        with pytest.raises(TypeError) as exc_info:
            ResultValidator.validate_mcmc_samples(samples)

        assert "must be numpy array" in str(exc_info.value)

    def test_validate_data_matrices_success(self):
        """Test validate_data_matrices with valid data."""
        X_list = [
            np.random.randn(50, 10),
            np.random.randn(50, 15),
            np.random.randn(50, 8)
        ]

        # Should not raise
        ResultValidator.validate_data_matrices(X_list)

    def test_validate_data_matrices_not_list(self):
        """Test validate_data_matrices with non-list input."""
        with pytest.raises(TypeError) as exc_info:
            ResultValidator.validate_data_matrices(np.random.randn(50, 10))

        assert "must be a list" in str(exc_info.value)

    def test_validate_data_matrices_empty_list(self):
        """Test validate_data_matrices with empty list."""
        with pytest.raises(ValueError) as exc_info:
            ResultValidator.validate_data_matrices([])

        assert "cannot be empty" in str(exc_info.value)

    def test_validate_data_matrices_not_array(self):
        """Test validate_data_matrices with non-array element."""
        X_list = [np.random.randn(50, 10), [1, 2, 3]]

        with pytest.raises(TypeError) as exc_info:
            ResultValidator.validate_data_matrices(X_list)

        assert "must be numpy array" in str(exc_info.value)

    def test_validate_data_matrices_wrong_dimensions(self):
        """Test validate_data_matrices with wrong dimensions."""
        X_list = [np.random.randn(50, 10, 5)]  # 3D instead of 2D

        with pytest.raises(ValueError) as exc_info:
            ResultValidator.validate_data_matrices(X_list)

        assert "must be 2D array" in str(exc_info.value)

    def test_validate_data_matrices_too_few_samples(self):
        """Test validate_data_matrices with too few samples."""
        X_list = [np.random.randn(5, 10)]  # Only 5 samples

        with pytest.raises(ValueError) as exc_info:
            ResultValidator.validate_data_matrices(X_list, min_samples=10)

        assert "at least 10 samples" in str(exc_info.value)

    def test_validate_data_matrices_with_nan(self):
        """Test validate_data_matrices with NaN values."""
        X = np.random.randn(50, 10)
        X[0, 0] = np.nan
        X_list = [X]

        with pytest.raises(ValueError) as exc_info:
            ResultValidator.validate_data_matrices(X_list)

        assert "NaN values" in str(exc_info.value)

    def test_validate_data_matrices_with_inf(self):
        """Test validate_data_matrices with infinite values."""
        X = np.random.randn(50, 10)
        X[0, 0] = np.inf
        X_list = [X]

        with pytest.raises(ValueError) as exc_info:
            ResultValidator.validate_data_matrices(X_list)

        assert "infinite values" in str(exc_info.value)


class TestValidateConfigStructure:
    """Test validate_config_structure function."""

    def test_valid_config(self):
        """Test with valid configuration."""
        config = {
            "model": {
                "K": 5,
                "num_samples": 1000
            },
            "data": {
                "data_dir": "/path/to/data"
            }
        }
        schema = {
            "model": {
                "K": int,
                "num_samples": int
            },
            "data": {
                "data_dir": str
            }
        }

        warnings = validate_config_structure(config, schema)
        assert len(warnings) == 0

    def test_missing_key(self):
        """Test with missing configuration key."""
        config = {"model": {}}
        schema = {"model": {"K": int}}

        warnings = validate_config_structure(config, schema)

        assert len(warnings) > 0
        assert any("Missing config key" in w for w in warnings)

    def test_wrong_type(self):
        """Test with wrong type."""
        config = {"model": {"K": "five"}}  # String instead of int
        schema = {"model": {"K": int}}

        warnings = validate_config_structure(config, schema)

        assert len(warnings) > 0
        assert any("should be int" in w for w in warnings)

    def test_custom_validator(self):
        """Test with custom validator function."""
        config = {"model": {"K": 100}}
        schema = {"model": {"K": lambda x: 1 <= x <= 50}}

        warnings = validate_config_structure(config, schema)

        assert len(warnings) > 0
        assert any("failed validation" in w for w in warnings)

    def test_nested_structure(self):
        """Test with nested configuration structure."""
        config = {
            "section1": {
                "subsection": {
                    "value": 10
                }
            }
        }
        schema = {
            "section1": {
                "subsection": {
                    "value": int
                }
            }
        }

        warnings = validate_config_structure(config, schema)
        assert len(warnings) == 0


class TestExperimentConfigSchema:
    """Test predefined EXPERIMENT_CONFIG_SCHEMA."""

    def test_schema_exists(self):
        """Test that schema is defined."""
        assert EXPERIMENT_CONFIG_SCHEMA is not None
        assert isinstance(EXPERIMENT_CONFIG_SCHEMA, dict)

    def test_schema_has_expected_sections(self):
        """Test that schema has expected sections."""
        assert "model" in EXPERIMENT_CONFIG_SCHEMA
        assert "data" in EXPERIMENT_CONFIG_SCHEMA
        assert "experiments" in EXPERIMENT_CONFIG_SCHEMA

    def test_valid_config_against_schema(self):
        """Test valid configuration against schema."""
        config = {
            "model": {
                "K": 5,
                "num_samples": 1000,
                "num_warmup": 500
            },
            "data": {
                "data_dir": "/path/to/data"
            },
            "experiments": {
                "base_output_dir": "/path/to/output"
            }
        }

        warnings = validate_config_structure(config, EXPERIMENT_CONFIG_SCHEMA)
        assert len(warnings) == 0

    def test_invalid_K_value(self):
        """Test with invalid K value (out of range)."""
        config = {
            "model": {
                "K": 100,  # Too large
                "num_samples": 1000,
                "num_warmup": 500
            },
            "data": {
                "data_dir": "/path"
            },
            "experiments": {
                "base_output_dir": "/output"
            }
        }

        warnings = validate_config_structure(config, EXPERIMENT_CONFIG_SCHEMA)
        assert any("K" in w and "failed validation" in w for w in warnings)
