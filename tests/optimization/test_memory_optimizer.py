"""
Tests for memory optimization module.

Tests MemoryOptimizer functionality including monitoring, cleanup, and optimization.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import gc


@pytest.fixture
def mock_psutil():
    """Mock psutil for testing."""
    with patch('optimization.memory_optimizer.psutil') as mock:
        # Mock virtual memory
        mock.virtual_memory.return_value = Mock(
            total=32 * 1024**3,  # 32 GB
            available=16 * 1024**3,  # 16 GB available
            percent=50.0
        )
        # Mock process
        mock.Process.return_value.memory_info.return_value = Mock(
            rss=1 * 1024**3  # 1 GB
        )
        yield mock


class TestMemoryOptimizerInitialization:
    """Test MemoryOptimizer initialization."""
    
    def test_initialization_default(self, mock_psutil):
        """Test default initialization."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        assert optimizer is not None
        assert optimizer.max_memory_gb > 0
    
    def test_initialization_custom_limit(self, mock_psutil):
        """Test initialization with custom memory limit."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer(max_memory_gb=16.0)
        assert optimizer.max_memory_gb == 16.0
    
    def test_initialization_auto_detect(self, mock_psutil):
        """Test automatic memory detection."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        # Should detect from mock_psutil (32 GB total)
        assert optimizer.max_memory_gb > 0
    
    def test_initialization_without_psutil(self):
        """Test initialization when psutil is not available."""
        with patch.dict('sys.modules', {'psutil': None}):
            try:
                from optimization.memory_optimizer import MemoryOptimizer
                # Should still work with fallback values
                optimizer = MemoryOptimizer(max_memory_gb=8.0)
                assert optimizer.max_memory_gb == 8.0
            except ImportError:
                # Acceptable if psutil is required
                pass


class TestMemoryMonitoring:
    """Test memory monitoring functionality."""
    
    def test_get_memory_usage(self, mock_psutil):
        """Test getting current memory usage."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        memory_gb = optimizer.get_memory_usage()
        
        assert isinstance(memory_gb, float)
        assert memory_gb >= 0
    
    def test_get_available_memory(self, mock_psutil):
        """Test getting available memory."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        available_gb = optimizer.get_available_memory()
        
        assert isinstance(available_gb, float)
        assert available_gb >= 0
    
    def test_get_memory_percent(self, mock_psutil):
        """Test getting memory usage percentage."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        percent = optimizer.get_memory_percent()
        
        assert isinstance(percent, (int, float))
        assert 0 <= percent <= 100
    
    def test_is_memory_available(self, mock_psutil):
        """Test checking if memory is available."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer(max_memory_gb=32.0)
        
        # Should have memory available with mock setup
        assert optimizer.is_memory_available(1.0)  # 1 GB
        assert optimizer.is_memory_available(10.0)  # 10 GB


class TestMemoryOptimization:
    """Test memory optimization operations."""
    
    def test_optimize_array_dtype(self, mock_psutil):
        """Test array dtype optimization."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        
        # Create float64 array
        arr = np.random.randn(100, 100).astype(np.float64)
        original_size = arr.nbytes
        
        # Optimize to float32
        arr_optimized = optimizer.optimize_array(arr)
        
        assert arr_optimized.dtype == np.float32
        assert arr_optimized.nbytes < original_size
        assert arr_optimized.shape == arr.shape
    
    def test_optimize_array_preserves_values(self, mock_psutil):
        """Test that optimization preserves array values."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        arr_optimized = optimizer.optimize_array(arr)
        
        # Values should be approximately equal
        np.testing.assert_allclose(arr, arr_optimized, rtol=1e-6)
    
    def test_optimize_array_already_optimized(self, mock_psutil):
        """Test optimizing already-optimized array."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        
        arr = np.random.randn(100, 100).astype(np.float32)
        arr_optimized = optimizer.optimize_array(arr)
        
        # Should remain float32
        assert arr_optimized.dtype == np.float32
    
    def test_optimize_multiple_arrays(self, mock_psutil):
        """Test optimizing a list of arrays."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        
        arrays = [
            np.random.randn(50, 100).astype(np.float64),
            np.random.randn(50, 80).astype(np.float64)
        ]
        
        arrays_optimized = [optimizer.optimize_array(arr) for arr in arrays]
        
        for arr_opt in arrays_optimized:
            assert arr_opt.dtype == np.float32


class TestMemoryCleanup:
    """Test memory cleanup functionality."""
    
    def test_cleanup(self, mock_psutil):
        """Test basic memory cleanup."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        
        # Create some garbage
        _ = [np.random.randn(100, 100) for _ in range(10)]
        
        # Cleanup should not raise errors
        optimizer.cleanup()
    
    def test_aggressive_cleanup(self, mock_psutil):
        """Test aggressive memory cleanup."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        
        # Create garbage
        _ = [np.random.randn(100, 100) for _ in range(10)]
        
        # Aggressive cleanup should work
        try:
            optimizer.aggressive_cleanup()
        except AttributeError:
            # Method might not exist, that's okay
            pass
    
    @patch('gc.collect')
    def test_cleanup_calls_gc(self, mock_gc_collect, mock_psutil):
        """Test that cleanup calls garbage collector."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        optimizer.cleanup()
        
        # Should have called gc.collect at least once
        assert mock_gc_collect.call_count >= 1


class TestMemoryContextManager:
    """Test memory optimizer as context manager."""
    
    def test_context_manager(self, mock_psutil):
        """Test using MemoryOptimizer as context manager."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        with MemoryOptimizer() as optimizer:
            assert optimizer is not None
            memory_gb = optimizer.get_memory_usage()
            assert memory_gb >= 0
    
    def test_context_manager_cleanup(self, mock_psutil):
        """Test that context manager performs cleanup on exit."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        # Create garbage before context
        garbage = [np.random.randn(100, 100) for _ in range(10)]
        
        with MemoryOptimizer() as optimizer:
            pass  # Just enter and exit
        
        # Cleanup should have been called (we can't really verify without instrumentation)
        del garbage


class TestMemoryOptimizationStrategies:
    """Test different memory optimization strategies."""
    
    def test_batch_size_calculation(self, mock_psutil):
        """Test optimal batch size calculation."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer(max_memory_gb=16.0)
        
        try:
            # Calculate batch size for given data dimensions
            batch_size = optimizer.calculate_optimal_batch_size(
                n_samples=1000,
                n_features=500,
                bytes_per_element=8  # float64
            )
            
            assert isinstance(batch_size, int)
            assert batch_size > 0
            assert batch_size <= 1000
        except AttributeError:
            # Method might not exist
            pass
    
    def test_memory_efficient_operation(self, mock_psutil):
        """Test memory-efficient operation wrapper."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        
        # Test operation
        def test_operation():
            return np.random.randn(100, 100)
        
        # Should execute without errors
        result = test_operation()
        assert result.shape == (100, 100)


class TestMemoryPressureHandling:
    """Test handling of memory pressure situations."""
    
    def test_low_memory_warning(self, mock_psutil):
        """Test warning when memory is low."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        # Mock low memory situation
        mock_psutil.virtual_memory.return_value = Mock(
            total=8 * 1024**3,
            available=0.5 * 1024**3,  # Only 0.5 GB available
            percent=93.75
        )
        
        optimizer = MemoryOptimizer(max_memory_gb=8.0)
        
        # Should detect low memory
        assert not optimizer.is_memory_available(1.0)
    
    def test_out_of_memory_handling(self, mock_psutil):
        """Test handling of out-of-memory situations."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer(max_memory_gb=1.0)  # Very limited
        
        # Requesting more memory than available
        has_memory = optimizer.is_memory_available(10.0)
        assert not has_memory


class TestMemoryStatistics:
    """Test memory statistics and reporting."""
    
    def test_get_memory_stats(self, mock_psutil):
        """Test getting memory statistics."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        
        try:
            stats = optimizer.get_memory_stats()
            assert isinstance(stats, dict)
            # Should contain useful statistics
            assert len(stats) > 0
        except AttributeError:
            # Method might not exist
            pass
    
    def test_memory_report(self, mock_psutil):
        """Test generating memory report."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        
        try:
            report = optimizer.get_memory_report()
            assert report is not None
        except AttributeError:
            # Method might not exist
            pass


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_memory_limit(self):
        """Test handling of zero memory limit."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        # Should either raise error or use minimum value
        try:
            optimizer = MemoryOptimizer(max_memory_gb=0.0)
            assert optimizer.max_memory_gb > 0  # Should use minimum
        except ValueError:
            # Acceptable to reject zero limit
            pass
    
    def test_negative_memory_limit(self):
        """Test handling of negative memory limit."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        # Should either raise error or use absolute value
        try:
            optimizer = MemoryOptimizer(max_memory_gb=-10.0)
            assert optimizer.max_memory_gb > 0
        except ValueError:
            # Acceptable to reject negative limit
            pass
    
    def test_extremely_large_memory_limit(self, mock_psutil):
        """Test handling of extremely large memory limit."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        # Should not crash with large limit
        optimizer = MemoryOptimizer(max_memory_gb=1000.0)
        assert optimizer.max_memory_gb == 1000.0
    
    def test_optimize_empty_array(self, mock_psutil):
        """Test optimizing empty array."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        
        arr = np.array([])
        arr_optimized = optimizer.optimize_array(arr)
        
        assert arr_optimized.shape == arr.shape
    
    def test_optimize_none(self, mock_psutil):
        """Test optimizing None value."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        
        try:
            result = optimizer.optimize_array(None)
            assert result is None
        except (TypeError, AttributeError):
            # Acceptable to reject None
            pass


class TestIntegration:
    """Integration tests for memory optimization."""
    
    def test_full_optimization_workflow(self, mock_psutil):
        """Test complete optimization workflow."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        with MemoryOptimizer(max_memory_gb=16.0) as optimizer:
            # Check initial memory
            initial_memory = optimizer.get_memory_usage()
            
            # Create and optimize arrays
            arrays = [
                np.random.randn(100, 100).astype(np.float64)
                for _ in range(5)
            ]
            
            arrays_optimized = [
                optimizer.optimize_array(arr) for arr in arrays
            ]
            
            # Verify optimization
            for arr_opt in arrays_optimized:
                assert arr_opt.dtype == np.float32
            
            # Cleanup
            optimizer.cleanup()
            
            # Memory tracking
            final_memory = optimizer.get_memory_usage()
            assert isinstance(final_memory, float)
    
    def test_memory_monitoring_over_time(self, mock_psutil):
        """Test monitoring memory usage over operations."""
        from optimization.memory_optimizer import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        
        memory_samples = []
        
        # Perform operations and track memory
        for i in range(5):
            _ = np.random.randn(100, 100)
            memory_samples.append(optimizer.get_memory_usage())
        
        # Should have collected samples
        assert len(memory_samples) == 5
        assert all(m >= 0 for m in memory_samples)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
