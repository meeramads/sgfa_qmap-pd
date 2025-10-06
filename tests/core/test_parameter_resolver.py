"""Tests for core.parameter_resolver module."""

import argparse
from unittest.mock import Mock

import pytest

from core.parameter_resolver import ParameterResolver


class TestParameterResolverInit:
    """Test ParameterResolver initialization."""

    def test_init_with_single_source(self):
        """Test initialization with single source."""
        source = {"K": 5}
        resolver = ParameterResolver(source)

        assert len(resolver.sources) == 1
        assert resolver.sources[0] == source

    def test_init_with_multiple_sources(self):
        """Test initialization with multiple sources."""
        source1 = {"K": 5}
        source2 = {"percW": 25.0}
        resolver = ParameterResolver(source1, source2)

        assert len(resolver.sources) == 2

    def test_filters_none_sources(self):
        """Test that None sources are filtered out."""
        source = {"K": 5}
        resolver = ParameterResolver(source, None, None)

        assert len(resolver.sources) == 1
        assert resolver.sources[0] == source

    def test_init_with_no_sources(self):
        """Test initialization with no sources."""
        resolver = ParameterResolver()

        assert len(resolver.sources) == 0

    def test_init_with_all_none(self):
        """Test initialization with all None sources."""
        resolver = ParameterResolver(None, None, None)

        assert len(resolver.sources) == 0


class TestParameterResolverGet:
    """Test ParameterResolver.get() method."""

    def test_get_from_dict_source(self):
        """Test getting parameter from dictionary source."""
        source = {"K": 5, "percW": 25.0}
        resolver = ParameterResolver(source)

        assert resolver.get("K") == 5
        assert resolver.get("percW") == 25.0

    def test_get_from_object_source(self):
        """Test getting parameter from object with attributes."""
        args = argparse.Namespace(K=10, model="sparseGFA")
        resolver = ParameterResolver(args)

        assert resolver.get("K") == 10
        assert resolver.get("model") == "sparseGFA"

    def test_get_returns_default_when_not_found(self):
        """Test that default is returned when parameter not found."""
        resolver = ParameterResolver({})

        result = resolver.get("missing", default="fallback")
        assert result == "fallback"

    def test_get_none_default(self):
        """Test get with None as default."""
        resolver = ParameterResolver({})

        result = resolver.get("missing")
        assert result is None

    def test_get_from_first_source_with_value(self):
        """Test that value from first source is returned."""
        source1 = {"K": 5}
        source2 = {"K": 10}
        resolver = ParameterResolver(source1, source2)

        assert resolver.get("K") == 5  # From source1, not source2

    def test_get_falls_back_to_later_source(self):
        """Test fallback to later source when key missing in earlier."""
        source1 = {"K": 5}
        source2 = {"percW": 25.0}
        resolver = ParameterResolver(source1, source2)

        assert resolver.get("percW") == 25.0  # From source2

    def test_get_skips_none_values(self):
        """Test that None values are skipped in favor of later sources."""
        source1 = {"K": None}
        source2 = {"K": 10}
        resolver = ParameterResolver(source1, source2)

        assert resolver.get("K") == 10  # Skips None in source1

    def test_get_with_mixed_source_types(self):
        """Test get with both dict and object sources."""
        args = argparse.Namespace(K=5)
        hypers = {"percW": 25.0}
        config = {"model": "sparseGFA"}
        resolver = ParameterResolver(args, hypers, config)

        assert resolver.get("K") == 5  # From args
        assert resolver.get("percW") == 25.0  # From hypers
        assert resolver.get("model") == "sparseGFA"  # From config


class TestParameterResolverGetRequired:
    """Test ParameterResolver.get_required() method."""

    def test_get_required_returns_value(self):
        """Test get_required returns value when parameter exists."""
        source = {"K": 5}
        resolver = ParameterResolver(source)

        result = resolver.get_required("K")
        assert result == 5

    def test_get_required_raises_when_missing(self):
        """Test get_required raises KeyError when parameter missing."""
        resolver = ParameterResolver({})

        with pytest.raises(KeyError) as exc_info:
            resolver.get_required("missing_param")

        assert "missing_param" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_get_required_raises_when_all_none(self):
        """Test get_required raises when all sources have None."""
        source1 = {"K": None}
        source2 = {"K": None}
        resolver = ParameterResolver(source1, source2)

        with pytest.raises(KeyError):
            resolver.get_required("K")

    def test_get_required_error_message_includes_source_types(self):
        """Test that error message includes source types."""
        args = argparse.Namespace()
        hypers = {}
        resolver = ParameterResolver(args, hypers)

        with pytest.raises(KeyError) as exc_info:
            resolver.get_required("missing")

        error_msg = str(exc_info.value)
        assert "Namespace" in error_msg or "dict" in error_msg


class TestParameterResolverGetAll:
    """Test ParameterResolver.get_all() method."""

    def test_get_all_returns_dict(self):
        """Test get_all returns dictionary."""
        source = {"K": 5, "percW": 25.0}
        resolver = ParameterResolver(source)

        result = resolver.get_all(["K", "percW"])

        assert isinstance(result, dict)
        assert result == {"K": 5, "percW": 25.0}

    def test_get_all_with_defaults(self):
        """Test get_all with defaults for missing keys."""
        source = {"K": 5}
        resolver = ParameterResolver(source)
        defaults = {"K": 10, "percW": 30.0}

        result = resolver.get_all(["K", "percW"], defaults=defaults)

        assert result["K"] == 5  # From source
        assert result["percW"] == 30.0  # From defaults

    def test_get_all_missing_keys_without_defaults(self):
        """Test get_all with missing keys and no defaults."""
        source = {"K": 5}
        resolver = ParameterResolver(source)

        result = resolver.get_all(["K", "missing"])

        assert result["K"] == 5
        assert result["missing"] is None

    def test_get_all_empty_keys(self):
        """Test get_all with empty keys list."""
        source = {"K": 5}
        resolver = ParameterResolver(source)

        result = resolver.get_all([])

        assert result == {}

    def test_get_all_multiple_sources(self):
        """Test get_all with multiple sources."""
        source1 = {"K": 5}
        source2 = {"percW": 25.0}
        source3 = {"model": "sparseGFA"}
        resolver = ParameterResolver(source1, source2, source3)

        result = resolver.get_all(["K", "percW", "model"])

        assert result == {"K": 5, "percW": 25.0, "model": "sparseGFA"}


class TestParameterResolverHas:
    """Test ParameterResolver.has() method."""

    def test_has_returns_true_when_present(self):
        """Test has returns True when parameter exists."""
        source = {"K": 5}
        resolver = ParameterResolver(source)

        assert resolver.has("K") is True

    def test_has_returns_false_when_missing(self):
        """Test has returns False when parameter missing."""
        resolver = ParameterResolver({})

        assert resolver.has("missing") is False

    def test_has_returns_false_when_all_none(self):
        """Test has returns False when all values are None."""
        source1 = {"K": None}
        source2 = {"K": None}
        resolver = ParameterResolver(source1, source2)

        assert resolver.has("K") is False

    def test_has_returns_true_from_any_source(self):
        """Test has returns True if parameter exists in any source."""
        source1 = {}
        source2 = {"K": 5}
        resolver = ParameterResolver(source1, source2)

        assert resolver.has("K") is True

    def test_has_with_zero_value(self):
        """Test has with zero value (should return True)."""
        source = {"K": 0}
        resolver = ParameterResolver(source)

        # 0 is not None, so has should return True
        assert resolver.has("K") is True

    def test_has_with_false_value(self):
        """Test has with False value (should return True)."""
        source = {"flag": False}
        resolver = ParameterResolver(source)

        # False is not None, so has should return True
        assert resolver.has("flag") is True


class TestParameterResolverRepr:
    """Test ParameterResolver.__repr__() method."""

    def test_repr_with_single_source(self):
        """Test repr with single source."""
        source = {"K": 5}
        resolver = ParameterResolver(source)

        repr_str = repr(resolver)

        assert "ParameterResolver" in repr_str
        assert "sources" in repr_str
        assert "dict" in repr_str

    def test_repr_with_multiple_sources(self):
        """Test repr with multiple sources."""
        args = argparse.Namespace()
        hypers = {}
        resolver = ParameterResolver(args, hypers)

        repr_str = repr(resolver)

        assert "ParameterResolver" in repr_str
        assert "Namespace" in repr_str or "dict" in repr_str

    def test_repr_with_no_sources(self):
        """Test repr with no sources."""
        resolver = ParameterResolver()

        repr_str = repr(resolver)

        assert "ParameterResolver" in repr_str
        assert "sources=[]" in repr_str


class TestParameterResolverIntegration:
    """Integration tests for ParameterResolver."""

    def test_typical_usage_pattern(self):
        """Test typical usage pattern: args -> hypers -> config -> default."""
        # Simulate command-line args
        args = argparse.Namespace(K=7)

        # Hyperparameters
        hypers = {"percW": 30.0, "tau_param": 0.5}

        # Configuration
        config = {"model": "sparseGFA", "K": 10, "num_samples": 1000}

        # Create resolver
        resolver = ParameterResolver(args, hypers, config)

        # K should come from args (first source)
        assert resolver.get("K") == 7

        # percW should come from hypers
        assert resolver.get("percW") == 30.0

        # model should come from config
        assert resolver.get("model") == "sparseGFA"

        # Missing param should use default
        assert resolver.get("missing", default="default_value") == "default_value"

    def test_with_none_values_in_chain(self):
        """Test parameter resolution with None values in chain."""
        args = argparse.Namespace(K=None, percW=25.0)
        hypers = {"K": 5}
        config = {"K": 10}

        resolver = ParameterResolver(args, hypers, config)

        # K is None in args, should fall back to hypers
        assert resolver.get("K") == 5

        # percW exists in args
        assert resolver.get("percW") == 25.0

    def test_batch_parameter_retrieval(self):
        """Test retrieving multiple parameters at once."""
        source = {"K": 5, "percW": 25.0, "model": "sparseGFA"}
        resolver = ParameterResolver(source)

        defaults = {
            "K": 10,
            "percW": 30.0,
            "num_samples": 1000,
            "num_warmup": 500
        }

        params = resolver.get_all(
            ["K", "percW", "model", "num_samples", "num_warmup"],
            defaults=defaults
        )

        assert params["K"] == 5  # From source
        assert params["percW"] == 25.0  # From source
        assert params["model"] == "sparseGFA"  # From source
        assert params["num_samples"] == 1000  # From defaults
        assert params["num_warmup"] == 500  # From defaults

    def test_parameter_validation_pattern(self):
        """Test using has() for parameter validation."""
        source = {"K": 5}
        resolver = ParameterResolver(source)

        # Check required parameters
        if resolver.has("K"):
            K = resolver.get("K")
            assert K == 5
        else:
            pytest.fail("K should exist")

        # Check optional parameters
        if resolver.has("optional_param"):
            pytest.fail("optional_param should not exist")
        else:
            # Use default
            optional = resolver.get("optional_param", default="default")
            assert optional == "default"

    def test_with_custom_config_objects(self):
        """Test with custom configuration objects."""
        class CustomConfig:
            def __init__(self):
                self.K = 5
                self.model = "sparseGFA"

        class DictConfig:
            def __init__(self):
                self.data = {"percW": 25.0}

        custom_config = CustomConfig()
        dict_config = DictConfig()

        resolver = ParameterResolver(custom_config, dict_config.data)

        assert resolver.get("K") == 5
        assert resolver.get("model") == "sparseGFA"
        assert resolver.get("percW") == 25.0

    def test_priority_order(self):
        """Test that source priority order is maintained."""
        high_priority = {"value": "high"}
        medium_priority = {"value": "medium", "other": "from_medium"}
        low_priority = {"value": "low", "another": "from_low"}

        resolver = ParameterResolver(high_priority, medium_priority, low_priority)

        # Should get from highest priority source with the value
        assert resolver.get("value") == "high"
        assert resolver.get("other") == "from_medium"
        assert resolver.get("another") == "from_low"

    def test_empty_resolver(self):
        """Test resolver with no sources."""
        resolver = ParameterResolver()

        assert resolver.get("anything") is None
        assert resolver.get("anything", default="default") == "default"
        assert resolver.has("anything") is False
        assert resolver.get_all(["a", "b"]) == {"a": None, "b": None}

        with pytest.raises(KeyError):
            resolver.get_required("required")
