"""Tests for core.results_cache module."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from core.results_cache import SGFAResultsCache


class MockArgs:
    """Mock args object for testing."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestSGFAResultsCacheInit:
    """Test SGFAResultsCache initialization."""

    def test_init_memory_only(self):
        """Test initialization with memory-only cache."""
        cache = SGFAResultsCache(cache_dir=None)

        assert cache.cache_dir is None
        assert cache.memory_cache == {}

    def test_init_with_cache_dir(self, tmp_path):
        """Test initialization with cache directory."""
        cache_dir = tmp_path / "cache"

        cache = SGFAResultsCache(cache_dir=cache_dir)

        assert cache.cache_dir == cache_dir
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_init_creates_cache_directory(self, tmp_path):
        """Test that initialization creates cache directory."""
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()

        cache = SGFAResultsCache(cache_dir=cache_dir)

        assert cache_dir.exists()


class TestComputeHash:
    """Test _compute_hash method."""

    def test_compute_hash_consistency(self):
        """Test that same inputs produce same hash."""
        cache = SGFAResultsCache()

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5, "percW": 25.0}
        args = MockArgs(num_warmup=100, num_samples=200)

        hash1 = cache._compute_hash(X_list, hypers, args)
        hash2 = cache._compute_hash(X_list, hypers, args)

        assert hash1 == hash2

    def test_compute_hash_different_data(self):
        """Test that different data produces different hash."""
        cache = SGFAResultsCache()

        X_list1 = [np.random.randn(10, 5)]
        X_list2 = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args = MockArgs(num_warmup=100)

        hash1 = cache._compute_hash(X_list1, hypers, args)
        hash2 = cache._compute_hash(X_list2, hypers, args)

        assert hash1 != hash2

    def test_compute_hash_different_hypers(self):
        """Test that different hyperparameters produce different hash."""
        cache = SGFAResultsCache()

        X_list = [np.random.randn(10, 5)]
        hypers1 = {"K": 5}
        hypers2 = {"K": 10}
        args = MockArgs(num_warmup=100)

        hash1 = cache._compute_hash(X_list, hypers1, args)
        hash2 = cache._compute_hash(X_list, hypers2, args)

        assert hash1 != hash2

    def test_compute_hash_different_args(self):
        """Test that different args produce different hash."""
        cache = SGFAResultsCache()

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args1 = MockArgs(num_warmup=100, num_samples=200)
        args2 = MockArgs(num_warmup=100, num_samples=300)

        hash1 = cache._compute_hash(X_list, hypers, args1)
        hash2 = cache._compute_hash(X_list, hypers, args2)

        assert hash1 != hash2

    def test_compute_hash_with_dict_args(self):
        """Test hash computation with dict args instead of object."""
        cache = SGFAResultsCache()

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args_dict = {"num_warmup": 100, "num_samples": 200}

        hash_result = cache._compute_hash(X_list, hypers, args_dict)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 16  # SHA256 truncated to 16 chars

    def test_compute_hash_ignores_none_args(self):
        """Test that None values in args are ignored."""
        cache = SGFAResultsCache()

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args1 = MockArgs(num_warmup=100, num_samples=None)
        args2 = MockArgs(num_warmup=100)

        hash1 = cache._compute_hash(X_list, hypers, args1)
        hash2 = cache._compute_hash(X_list, hypers, args2)

        # Should be the same since None is ignored
        assert hash1 == hash2

    def test_compute_hash_hyper_order_independent(self):
        """Test that hyperparameter order doesn't affect hash."""
        cache = SGFAResultsCache()

        X_list = [np.random.randn(10, 5)]
        hypers1 = {"K": 5, "percW": 25.0}
        hypers2 = {"percW": 25.0, "K": 5}
        args = MockArgs(num_warmup=100)

        hash1 = cache._compute_hash(X_list, hypers1, args)
        hash2 = cache._compute_hash(X_list, hypers2, args)

        assert hash1 == hash2

    def test_compute_hash_multiple_views(self):
        """Test hash computation with multiple data views."""
        cache = SGFAResultsCache()

        X_list = [
            np.random.randn(10, 5),
            np.random.randn(10, 8),
            np.random.randn(10, 3)
        ]
        hypers = {"K": 5}
        args = MockArgs(num_warmup=100)

        hash_result = cache._compute_hash(X_list, hypers, args)

        assert isinstance(hash_result, str)


class TestGetMethod:
    """Test get method."""

    def test_get_from_empty_cache_returns_none(self):
        """Test getting from empty cache returns None."""
        cache = SGFAResultsCache()

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args = MockArgs(num_warmup=100)

        result = cache.get(X_list, hypers, args)

        assert result is None

    def test_get_from_memory_cache(self):
        """Test getting result from memory cache."""
        cache = SGFAResultsCache()

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args = MockArgs(num_warmup=100)

        # Put result in cache
        expected_result = {"Z": np.random.randn(10, 5), "success": True}
        cache.put(X_list, hypers, args, expected_result)

        # Get it back
        result = cache.get(X_list, hypers, args)

        assert result == expected_result

    def test_get_from_disk_cache(self, tmp_path):
        """Test getting result from disk cache."""
        cache_dir = tmp_path / "cache"
        cache = SGFAResultsCache(cache_dir=cache_dir)

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args = MockArgs(num_warmup=100)

        # Put result in cache
        expected_result = {"Z": np.random.randn(10, 5), "success": True}
        cache.put(X_list, hypers, args, expected_result)

        # Create new cache instance (clears memory cache)
        new_cache = SGFAResultsCache(cache_dir=cache_dir)

        # Should still get result from disk
        result = new_cache.get(X_list, hypers, args)

        # Compare Z arrays separately
        np.testing.assert_array_equal(result["Z"], expected_result["Z"])
        assert result["success"] == expected_result["success"]

    def test_get_different_data_returns_none(self):
        """Test that different data returns None."""
        cache = SGFAResultsCache()

        X_list1 = [np.random.randn(10, 5)]
        X_list2 = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args = MockArgs(num_warmup=100)

        # Put result for X_list1
        cache.put(X_list1, hypers, args, {"result": "data1"})

        # Try to get with X_list2
        result = cache.get(X_list2, hypers, args)

        assert result is None

    def test_get_populates_memory_from_disk(self, tmp_path):
        """Test that disk cache populates memory cache on get."""
        cache_dir = tmp_path / "cache"
        cache = SGFAResultsCache(cache_dir=cache_dir)

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args = MockArgs(num_warmup=100)

        # Put in cache
        expected_result = {"success": True}
        cache.put(X_list, hypers, args, expected_result)

        # Create new instance (empty memory cache)
        new_cache = SGFAResultsCache(cache_dir=cache_dir)
        assert len(new_cache.memory_cache) == 0

        # Get from disk
        new_cache.get(X_list, hypers, args)

        # Should now be in memory cache
        assert len(new_cache.memory_cache) == 1


class TestPutMethod:
    """Test put method."""

    def test_put_stores_in_memory(self):
        """Test that put stores result in memory cache."""
        cache = SGFAResultsCache()

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args = MockArgs(num_warmup=100)
        result = {"success": True}

        cache.put(X_list, hypers, args, result)

        assert len(cache.memory_cache) == 1

    def test_put_stores_on_disk(self, tmp_path):
        """Test that put stores result on disk."""
        cache_dir = tmp_path / "cache"
        cache = SGFAResultsCache(cache_dir=cache_dir)

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args = MockArgs(num_warmup=100)
        result = {"success": True}

        cache.put(X_list, hypers, args, result)

        # Check that file was created
        cache_files = list(cache_dir.glob("*.pkl"))
        assert len(cache_files) == 1

    def test_put_overwrites_existing(self):
        """Test that put overwrites existing cache entry."""
        cache = SGFAResultsCache()

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args = MockArgs(num_warmup=100)

        # Put first result
        cache.put(X_list, hypers, args, {"result": "first"})

        # Put second result (same key)
        cache.put(X_list, hypers, args, {"result": "second"})

        # Should get the second result
        result = cache.get(X_list, hypers, args)
        assert result == {"result": "second"}

    def test_put_handles_complex_results(self, tmp_path):
        """Test that put handles complex result structures."""
        cache_dir = tmp_path / "cache"
        cache = SGFAResultsCache(cache_dir=cache_dir)

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args = MockArgs(num_warmup=100)

        # Complex result with nested structures
        result = {
            "Z": np.random.randn(10, 5),
            "W": [np.random.randn(5, 5), np.random.randn(8, 5)],
            "metadata": {
                "success": True,
                "iterations": 1000
            }
        }

        cache.put(X_list, hypers, args, result)
        retrieved = cache.get(X_list, hypers, args)

        np.testing.assert_array_equal(retrieved["Z"], result["Z"])
        assert retrieved["metadata"] == result["metadata"]


class TestClearMethod:
    """Test clear method."""

    def test_clear_memory_cache(self):
        """Test clearing memory cache."""
        cache = SGFAResultsCache()

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args = MockArgs(num_warmup=100)

        cache.put(X_list, hypers, args, {"result": "data"})
        assert len(cache.memory_cache) > 0

        cache.clear()

        assert len(cache.memory_cache) == 0

    def test_clear_disk_cache(self, tmp_path):
        """Test clearing disk cache."""
        cache_dir = tmp_path / "cache"
        cache = SGFAResultsCache(cache_dir=cache_dir)

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args = MockArgs(num_warmup=100)

        cache.put(X_list, hypers, args, {"result": "data"})

        # Verify file exists
        assert len(list(cache_dir.glob("*.pkl"))) > 0

        cache.clear()

        # Verify file deleted
        assert len(list(cache_dir.glob("*.pkl"))) == 0

    def test_clear_both_caches(self, tmp_path):
        """Test clearing both memory and disk caches."""
        cache_dir = tmp_path / "cache"
        cache = SGFAResultsCache(cache_dir=cache_dir)

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args = MockArgs(num_warmup=100)

        cache.put(X_list, hypers, args, {"result": "data"})

        cache.clear()

        assert len(cache.memory_cache) == 0
        assert len(list(cache_dir.glob("*.pkl"))) == 0


class TestGetStatsMethod:
    """Test get_stats method."""

    def test_get_stats_empty_cache(self):
        """Test get_stats on empty cache."""
        cache = SGFAResultsCache()

        stats = cache.get_stats()

        assert stats["memory_entries"] == 0
        assert stats["disk_entries"] == 0

    def test_get_stats_memory_only(self):
        """Test get_stats with memory-only cache."""
        cache = SGFAResultsCache()

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args = MockArgs(num_warmup=100)

        cache.put(X_list, hypers, args, {"result": "data"})

        stats = cache.get_stats()

        assert stats["memory_entries"] == 1
        assert stats["disk_entries"] == 0

    def test_get_stats_with_disk_cache(self, tmp_path):
        """Test get_stats with disk cache."""
        cache_dir = tmp_path / "cache"
        cache = SGFAResultsCache(cache_dir=cache_dir)

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args = MockArgs(num_warmup=100)

        cache.put(X_list, hypers, args, {"result": "data"})

        stats = cache.get_stats()

        assert stats["memory_entries"] == 1
        assert stats["disk_entries"] == 1

    def test_get_stats_multiple_entries(self, tmp_path):
        """Test get_stats with multiple cache entries."""
        cache_dir = tmp_path / "cache"
        cache = SGFAResultsCache(cache_dir=cache_dir)

        # Add multiple entries with different parameters
        for i in range(3):
            X_list = [np.random.randn(10, 5)]
            hypers = {"K": 5 + i}
            args = MockArgs(num_warmup=100)
            cache.put(X_list, hypers, args, {"result": f"data{i}"})

        stats = cache.get_stats()

        assert stats["memory_entries"] == 3
        assert stats["disk_entries"] == 3


class TestSGFAResultsCacheIntegration:
    """Integration tests for SGFAResultsCache."""

    def test_full_cache_workflow(self, tmp_path):
        """Test complete cache workflow."""
        cache_dir = tmp_path / "cache"
        cache = SGFAResultsCache(cache_dir=cache_dir)

        X_list = [np.random.randn(20, 10), np.random.randn(20, 15)]
        hypers = {"K": 5, "percW": 25.0}
        args = MockArgs(num_warmup=500, num_samples=1000, num_chains=1)

        # First access - cache miss
        result1 = cache.get(X_list, hypers, args)
        assert result1 is None

        # Store result
        sgfa_result = {
            "Z": np.random.randn(20, 5),
            "W": [np.random.randn(10, 5), np.random.randn(15, 5)],
            "success": True,
            "log_likelihood": -123.45
        }
        cache.put(X_list, hypers, args, sgfa_result)

        # Second access - cache hit
        result2 = cache.get(X_list, hypers, args)
        assert result2 is not None
        assert result2["success"] is True
        np.testing.assert_array_equal(result2["Z"], sgfa_result["Z"])

        # Check stats
        stats = cache.get_stats()
        assert stats["memory_entries"] == 1
        assert stats["disk_entries"] == 1

    def test_cache_persistence_across_instances(self, tmp_path):
        """Test that cache persists across different instances."""
        cache_dir = tmp_path / "cache"

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args = MockArgs(num_warmup=100)
        result_data = {"success": True, "value": 42}

        # Create first cache instance and store
        cache1 = SGFAResultsCache(cache_dir=cache_dir)
        cache1.put(X_list, hypers, args, result_data)

        # Create second cache instance
        cache2 = SGFAResultsCache(cache_dir=cache_dir)

        # Should retrieve from disk
        retrieved = cache2.get(X_list, hypers, args)
        assert retrieved == result_data

    def test_memory_only_vs_disk_cache(self, tmp_path):
        """Test difference between memory-only and disk cache."""
        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args = MockArgs(num_warmup=100)
        result = {"success": True}

        # Memory-only cache
        cache_mem = SGFAResultsCache(cache_dir=None)
        cache_mem.put(X_list, hypers, args, result)

        # Disk cache
        cache_disk = SGFAResultsCache(cache_dir=tmp_path / "cache")
        cache_disk.put(X_list, hypers, args, result)

        # Get stats
        stats_mem = cache_mem.get_stats()
        stats_disk = cache_disk.get_stats()

        assert stats_mem["disk_entries"] == 0
        assert stats_disk["disk_entries"] == 1

    def test_cache_with_different_random_seeds(self):
        """Test that different random seeds produce different cache keys."""
        cache = SGFAResultsCache()

        X_list = [np.random.randn(10, 5)]
        hypers = {"K": 5}
        args1 = MockArgs(num_warmup=100, seed=42)
        args2 = MockArgs(num_warmup=100, seed=43)

        cache.put(X_list, hypers, args1, {"seed": 42})
        cache.put(X_list, hypers, args2, {"seed": 43})

        # Should be two separate cache entries
        assert cache.get(X_list, hypers, args1)["seed"] == 42
        assert cache.get(X_list, hypers, args2)["seed"] == 43

    def test_cache_handles_large_results(self, tmp_path):
        """Test cache with large result objects."""
        cache_dir = tmp_path / "cache"
        cache = SGFAResultsCache(cache_dir=cache_dir)

        X_list = [np.random.randn(100, 50)]
        hypers = {"K": 10}
        args = MockArgs(num_warmup=100)

        # Large result
        large_result = {
            "samples": {
                f"param_{i}": np.random.randn(1000, 100) for i in range(5)
            }
        }

        cache.put(X_list, hypers, args, large_result)
        retrieved = cache.get(X_list, hypers, args)

        assert "samples" in retrieved
        assert len(retrieved["samples"]) == 5
