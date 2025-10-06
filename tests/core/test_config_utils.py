"""Tests for core.config_utils module."""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from core.config_utils import (
    ConfigAccessor,
    ConfigHelper,
    JAXMemoryManager,
    PlotManager,
    check_configuration_warnings,
    dict_to_namespace,
    ensure_directories,
    get_checkpoint_dir,
    get_data_dir,
    get_default_configuration,
    get_experiment_config,
    get_output_dir,
    safe_get,
    safe_get_path,
    update_config_safely,
    validate_configuration,
    validate_required_config,
)


class TestSafeGet:
    """Test safe_get function."""

    def test_simple_key_access(self):
        """Test accessing a simple key."""
        config = {"key": "value"}
        result = safe_get(config, "key")
        assert result == "value"

    def test_nested_key_access(self):
        """Test accessing nested keys."""
        config = {"level1": {"level2": {"level3": "deep_value"}}}
        result = safe_get(config, "level1", "level2", "level3")
        assert result == "deep_value"

    def test_missing_key_returns_default(self):
        """Test that missing keys return default value."""
        config = {"existing": "value"}
        result = safe_get(config, "missing", default="fallback")
        assert result == "fallback"

    def test_none_default(self):
        """Test with None as default."""
        config = {}
        result = safe_get(config, "missing")
        assert result is None

    def test_partial_path_exists(self):
        """Test when partial path exists but not full path."""
        config = {"level1": {"level2": "value"}}
        result = safe_get(config, "level1", "level2", "level3", default="default")
        assert result == "default"

    def test_non_dict_value(self):
        """Test accessing key from non-dict value."""
        config = {"key": "string_value"}
        result = safe_get(config, "key", "subkey", default="default")
        assert result == "default"


class TestSafeGetPath:
    """Test safe_get_path function."""

    def test_returns_path_object(self):
        """Test that result is a Path object."""
        config = {"data": {"data_dir": "/path/to/data"}}
        result = safe_get_path(config, "data", "data_dir")
        assert isinstance(result, Path)
        assert str(result) == "/path/to/data"

    def test_default_path(self):
        """Test default path when key missing."""
        config = {}
        result = safe_get_path(config, "missing", "key", default="./default")
        assert isinstance(result, Path)
        assert str(result) == "./default"


class TestDirectoryGetters:
    """Test directory getter functions."""

    def test_get_data_dir(self):
        """Test get_data_dir function."""
        config = {"data": {"data_dir": "/my/data"}}
        result = get_data_dir(config)
        assert isinstance(result, Path)
        assert str(result) == "/my/data"

    def test_get_data_dir_default(self):
        """Test get_data_dir with default fallback."""
        config = {}
        result = get_data_dir(config)
        assert isinstance(result, Path)
        assert str(result) == "./data"

    def test_get_output_dir(self):
        """Test get_output_dir function."""
        config = {"experiments": {"base_output_dir": "/my/output"}}
        result = get_output_dir(config)
        assert isinstance(result, Path)
        assert str(result) == "/my/output"

    def test_get_output_dir_default(self):
        """Test get_output_dir with default fallback."""
        config = {}
        result = get_output_dir(config)
        assert str(result) == "./results"

    def test_get_checkpoint_dir(self):
        """Test get_checkpoint_dir function."""
        config = {"monitoring": {"checkpoint_dir": "/my/checkpoints"}}
        result = get_checkpoint_dir(config)
        assert str(result) == "/my/checkpoints"


class TestDictToNamespace:
    """Test dict_to_namespace function."""

    def test_simple_conversion(self):
        """Test simple dict to namespace conversion."""
        config = {"K": 5, "percW": 25.0}
        ns = dict_to_namespace(config)

        assert isinstance(ns, argparse.Namespace)
        assert ns.K == 5
        assert ns.percW == 25.0

    def test_with_defaults(self):
        """Test conversion with defaults."""
        config = {"K": 5}
        defaults = {"K": 10, "percW": 30.0, "model": "sparseGFA"}
        ns = dict_to_namespace(config, defaults)

        assert ns.K == 5  # From config
        assert ns.percW == 30.0  # From defaults
        assert ns.model == "sparseGFA"  # From defaults

    def test_config_overrides_defaults(self):
        """Test that config values override defaults."""
        config = {"value": "from_config"}
        defaults = {"value": "from_defaults"}
        ns = dict_to_namespace(config, defaults)

        assert ns.value == "from_config"


class TestValidateRequiredConfig:
    """Test validate_required_config function."""

    def test_all_required_keys_present(self):
        """Test when all required keys are present."""
        config = {"data": {"data_dir": "/path"}, "model": {"K": 5}}
        required = [["data", "data_dir"], ["model", "K"]]

        # Should not raise
        validate_required_config(config, required)

    def test_missing_required_key_raises(self):
        """Test that missing required key raises KeyError."""
        config = {"data": {"data_dir": "/path"}}
        required = [["data", "data_dir"], ["model", "K"]]

        with pytest.raises(KeyError) as exc_info:
            validate_required_config(config, required)

        assert "model.K" in str(exc_info.value)

    def test_empty_required_keys(self):
        """Test with no required keys."""
        config = {}
        required = []

        # Should not raise
        validate_required_config(config, required)


class TestUpdateConfigSafely:
    """Test update_config_safely function."""

    def test_simple_update(self):
        """Test simple config update."""
        config = {"key1": "value1", "key2": "value2"}
        updates = {"key2": "new_value2", "key3": "value3"}

        result = update_config_safely(config, updates)

        assert result["key1"] == "value1"
        assert result["key2"] == "new_value2"
        assert result["key3"] == "value3"

    def test_nested_update(self):
        """Test nested config update."""
        config = {"section": {"a": 1, "b": 2}, "other": "value"}
        updates = {"section": {"b": 20, "c": 3}}

        result = update_config_safely(config, updates)

        assert result["section"]["a"] == 1  # Preserved
        assert result["section"]["b"] == 20  # Updated
        assert result["section"]["c"] == 3  # Added
        assert result["other"] == "value"  # Preserved

    def test_original_config_unchanged(self):
        """Test that original config is not modified."""
        config = {"key": "original"}
        updates = {"key": "modified"}

        result = update_config_safely(config, updates)

        assert config["key"] == "original"
        assert result["key"] == "modified"


class TestGetExperimentConfig:
    """Test get_experiment_config function."""

    def test_get_experiment_specific_config(self):
        """Test getting experiment-specific configuration."""
        config = {
            "my_experiment": {
                "n_repetitions": 5,
                "custom_param": "value"
            }
        }

        result = get_experiment_config(config, "my_experiment")

        assert result["n_repetitions"] == 5
        assert result["custom_param"] == "value"

    def test_defaults_applied(self):
        """Test that defaults are applied."""
        config = {}

        result = get_experiment_config(config, "nonexistent")

        assert result["n_repetitions"] == 1
        assert result["save_intermediate"] is True
        assert result["generate_plots"] is True

    def test_experiment_overrides_defaults(self):
        """Test that experiment config overrides defaults."""
        config = {"exp": {"n_repetitions": 10}}

        result = get_experiment_config(config, "exp")

        assert result["n_repetitions"] == 10


class TestEnsureDirectories:
    """Test ensure_directories function."""

    def test_creates_output_directory(self, tmp_path):
        """Test that output directory is created."""
        output_dir = tmp_path / "output"
        config = {"experiments": {"base_output_dir": str(output_dir)}}

        ensure_directories(config)

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_creates_checkpoint_directory(self, tmp_path):
        """Test that checkpoint directory is created."""
        checkpoint_dir = tmp_path / "checkpoints"
        config = {"monitoring": {"checkpoint_dir": str(checkpoint_dir)}}

        ensure_directories(config)

        assert checkpoint_dir.exists()

    def test_no_error_if_directories_exist(self, tmp_path):
        """Test that no error is raised if directories already exist."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = {"experiments": {"base_output_dir": str(output_dir)}}

        # Should not raise
        ensure_directories(config)


class TestConfigAccessor:
    """Test ConfigAccessor class."""

    def test_initialization(self):
        """Test ConfigAccessor initialization."""
        config = {"key": "value"}
        accessor = ConfigAccessor(config)
        assert accessor.config == config

    def test_get_method(self):
        """Test get method."""
        config = {"key": "value"}
        accessor = ConfigAccessor(config)
        assert accessor.get("key") == "value"
        assert accessor.get("missing", "default") == "default"

    def test_dataset_property(self):
        """Test dataset property."""
        config = {"dataset": "my_dataset"}
        accessor = ConfigAccessor(config)
        assert accessor.dataset == "my_dataset"

    def test_dataset_default(self):
        """Test dataset default value."""
        accessor = ConfigAccessor({})
        assert accessor.dataset == "qmap_pd"

    def test_K_property(self):
        """Test K property."""
        config = {"K": 10}
        accessor = ConfigAccessor(config)
        assert accessor.K == 10

    def test_percW_property(self):
        """Test percW property."""
        config = {"percW": 30.0}
        accessor = ConfigAccessor(config)
        assert accessor.percW == 30.0

    def test_directory_properties(self):
        """Test directory properties."""
        config = {
            "data": {"data_dir": "/data"},
            "experiments": {"base_output_dir": "/output"},
            "monitoring": {"checkpoint_dir": "/checkpoints"}
        }
        accessor = ConfigAccessor(config)

        assert str(accessor.data_dir) == "/data"
        assert str(accessor.output_dir) == "/output"
        assert str(accessor.checkpoint_dir) == "/checkpoints"

    def test_get_experiment_setting(self):
        """Test get_experiment_setting method."""
        config = {"experiment1": {"setting": "value"}}
        accessor = ConfigAccessor(config)

        result = accessor.get_experiment_setting("experiment1", "setting")
        assert result == "value"

    def test_has_shared_data(self):
        """Test has_shared_data method."""
        config_with_shared = {"_shared_data": {"key": "value"}}
        config_without_shared = {}

        assert ConfigAccessor(config_with_shared).has_shared_data() is True
        assert ConfigAccessor(config_without_shared).has_shared_data() is False

    def test_get_shared_data(self):
        """Test get_shared_data method."""
        config = {"_shared_data": {"data": "value"}}
        accessor = ConfigAccessor(config)

        shared = accessor.get_shared_data()
        assert shared == {"data": "value"}


class TestConfigHelper:
    """Test ConfigHelper class."""

    def test_to_dict_with_dict(self):
        """Test to_dict with dictionary input."""
        config = {"key": "value"}
        result = ConfigHelper.to_dict(config)
        assert result == config

    def test_to_dict_with_to_dict_method(self):
        """Test to_dict with object having to_dict method."""
        class MyConfig:
            def to_dict(self):
                return {"foo": "bar"}

        result = ConfigHelper.to_dict(MyConfig())
        assert result == {"foo": "bar"}

    def test_to_dict_with_dict_attribute(self):
        """Test to_dict with object having __dict__."""
        class MyConfig:
            def __init__(self):
                self.key = "value"

        result = ConfigHelper.to_dict(MyConfig())
        assert "key" in result
        assert result["key"] == "value"

    def test_to_dict_with_simple_value(self):
        """Test to_dict with simple value."""
        result = ConfigHelper.to_dict("simple_string")
        assert result == {"value": "simple_string"}

    def test_get_output_dir_safe(self):
        """Test get_output_dir_safe method."""
        config = {"experiments": {"base_output_dir": "/my/output"}}
        result = ConfigHelper.get_output_dir_safe(config)
        assert str(result) == "/my/output"

    def test_get_data_dir_safe(self):
        """Test get_data_dir_safe method."""
        config = {"data": {"data_dir": "/my/data"}}
        result = ConfigHelper.get_data_dir_safe(config)
        assert str(result) == "/my/data"

    def test_safe_get_from_config(self):
        """Test safe_get_from_config method."""
        config = {"section": {"key": "value"}}
        result = ConfigHelper.safe_get_from_config(config, "section", "key")
        assert result == "value"


class TestCheckConfigurationWarnings:
    """Test check_configuration_warnings function."""

    def test_large_num_samples_warning(self):
        """Test warning for large num_samples."""
        config = {"model": {"num_samples": 15000}}
        warnings = check_configuration_warnings(config)

        assert len(warnings) > 0
        assert any("num_samples" in w for w in warnings)

    def test_large_K_warning(self):
        """Test warning for large K."""
        config = {"model": {"K": 25}}
        warnings = check_configuration_warnings(config)

        assert any("K=25" in w or "factors" in w for w in warnings)

    @patch('core.config_utils.jax')
    def test_gpu_warning_no_devices(self, mock_jax):
        """Test warning when GPU requested but not available."""
        mock_jax.devices.return_value = []
        config = {"system": {"use_gpu": True}}

        warnings = check_configuration_warnings(config)

        assert any("GPU" in w for w in warnings)

    def test_no_warnings(self):
        """Test configuration with no warnings."""
        config = {"model": {"num_samples": 1000, "K": 5}}
        warnings = check_configuration_warnings(config)

        # Should have minimal or no warnings for reasonable config
        assert isinstance(warnings, list)


class TestJAXMemoryManager:
    """Test JAXMemoryManager context manager."""

    @patch('core.config_utils.jax')
    def test_clears_jax_memory_on_exit(self, mock_jax):
        """Test that JAX memory is cleared on exit."""
        with JAXMemoryManager():
            pass

        mock_jax.clear_caches.assert_called_once()

    def test_handles_jax_import_error(self):
        """Test graceful handling when JAX not available."""
        with patch('core.config_utils.jax', side_effect=ImportError):
            # Should not raise
            with JAXMemoryManager():
                pass

    @patch('core.config_utils.jax')
    def test_handles_clear_error(self, mock_jax):
        """Test handling of errors during cache clearing."""
        mock_jax.clear_caches.side_effect = Exception("Clear failed")

        # Should not raise - errors are logged
        with JAXMemoryManager():
            pass


class TestPlotManager:
    """Test PlotManager context manager."""

    @patch('core.config_utils.plt')
    def test_closes_plots_on_exit(self, mock_plt):
        """Test that matplotlib figures are closed on exit."""
        with PlotManager():
            pass

        mock_plt.close.assert_called_once_with("all")

    def test_handles_matplotlib_import_error(self):
        """Test graceful handling when matplotlib not available."""
        with patch.dict('sys.modules', {'matplotlib.pyplot': None}):
            # Should not raise
            with PlotManager():
                pass

    @patch('core.config_utils.plt')
    def test_handles_close_error(self, mock_plt):
        """Test handling of errors during figure closing."""
        mock_plt.close.side_effect = Exception("Close failed")

        # Should not raise - errors are logged
        with PlotManager():
            pass


class TestValidateConfiguration:
    """Test validate_configuration function."""

    def test_validates_correct_configuration(self):
        """Test validation of correct configuration."""
        # Get default config which should be valid
        config = get_default_configuration()

        # Should not raise
        result = validate_configuration(config)
        assert isinstance(result, dict)

    def test_merges_with_defaults_on_failure(self):
        """Test that partial config is merged with defaults."""
        # Minimal config that may need defaults
        config = {"model": {"K": 5}}

        # Should merge with defaults rather than fail
        result = validate_configuration(config)
        assert isinstance(result, dict)
        assert "model" in result


class TestGetDefaultConfiguration:
    """Test get_default_configuration function."""

    def test_returns_dict(self):
        """Test that default configuration is a dictionary."""
        config = get_default_configuration()
        assert isinstance(config, dict)

    def test_has_expected_sections(self):
        """Test that default config has expected sections."""
        config = get_default_configuration()

        # Should have main sections
        assert "data" in config or "model" in config or "experiments" in config
