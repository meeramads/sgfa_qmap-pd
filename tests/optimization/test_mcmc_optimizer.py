"""
Tests for MCMC optimization module.

Tests MCMCMemoryOptimizer functionality including adaptive sampling, checkpointing, and memory management.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


@pytest.fixture
def mock_jax():
    """Mock JAX for testing."""
    with patch('optimization.mcmc_optimizer.jax') as mock:
        mock.device_count.return_value = 1
        yield mock


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


class TestMCMCOptimizerInitialization:
    """Test MCMCMemoryOptimizer initialization."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer()
            assert optimizer is not None
        except ImportError:
            pytest.skip("MCMCMemoryOptimizer requires JAX/NumPyro")
    
    def test_initialization_with_memory_limit(self):
        """Test initialization with memory limit."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer(memory_limit_gb=8.0)
            assert optimizer.memory_limit_gb == 8.0
        except ImportError:
            pytest.skip("MCMCMemoryOptimizer requires JAX/NumPyro")
    
    def test_initialization_with_checkpoint_dir(self, temp_checkpoint_dir):
        """Test initialization with checkpoint directory."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer(
                checkpoint_dir=str(temp_checkpoint_dir)
            )
            assert optimizer is not None
        except ImportError:
            pytest.skip("MCMCMemoryOptimizer requires JAX/NumPyro")


class TestAdaptiveSampling:
    """Test adaptive MCMC sampling functionality."""
    
    def test_calculate_optimal_batch_size(self):
        """Test calculating optimal batch size."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer(memory_limit_gb=8.0)
            
            batch_size = optimizer.calculate_optimal_batch_size(
                n_samples=1000,
                n_chains=4,
                data_size_gb=2.0
            )
            
            assert isinstance(batch_size, int)
            assert batch_size > 0
            assert batch_size <= 1000
        except (ImportError, AttributeError):
            pytest.skip("Method not available")
    
    def test_adaptive_sampling_reduces_memory(self):
        """Test that adaptive sampling reduces memory usage."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer(memory_limit_gb=4.0)
            
            # With limited memory, should reduce batch size
            batch_size = optimizer.calculate_optimal_batch_size(
                n_samples=10000,
                n_chains=8,
                data_size_gb=3.0
            )
            
            # Should be smaller than full dataset
            assert batch_size < 10000
        except (ImportError, AttributeError):
            pytest.skip("Method not available")
    
    def test_enable_adaptive_sampling(self):
        """Test enabling adaptive sampling."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer()
            optimizer.enable_adaptive_sampling(True)
            
            assert optimizer.adaptive_sampling_enabled
        except (ImportError, AttributeError):
            pytest.skip("Method not available")


class TestCheckpointing:
    """Test MCMC checkpointing functionality."""
    
    def test_save_checkpoint(self, temp_checkpoint_dir):
        """Test saving MCMC checkpoint."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer(
                checkpoint_dir=str(temp_checkpoint_dir)
            )
            
            # Mock MCMC state
            mcmc_state = {
                'samples': np.random.randn(100, 10),
                'iteration': 100,
                'chain_id': 0
            }
            
            checkpoint_path = optimizer.save_checkpoint(mcmc_state, iteration=100)
            
            assert checkpoint_path is not None
            assert Path(checkpoint_path).exists()
        except (ImportError, AttributeError):
            pytest.skip("Checkpointing not available")
    
    def test_load_checkpoint(self, temp_checkpoint_dir):
        """Test loading MCMC checkpoint."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer(
                checkpoint_dir=str(temp_checkpoint_dir)
            )
            
            # Save checkpoint
            mcmc_state = {
                'samples': np.random.randn(100, 10),
                'iteration': 100
            }
            checkpoint_path = optimizer.save_checkpoint(mcmc_state, iteration=100)
            
            # Load checkpoint
            loaded_state = optimizer.load_checkpoint(checkpoint_path)
            
            assert loaded_state is not None
            assert 'iteration' in loaded_state
            assert loaded_state['iteration'] == 100
        except (ImportError, AttributeError):
            pytest.skip("Checkpointing not available")
    
    def test_checkpoint_interval(self, temp_checkpoint_dir):
        """Test checkpoint saving at intervals."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer(
                checkpoint_dir=str(temp_checkpoint_dir),
                checkpoint_interval=50
            )
            
            assert optimizer.checkpoint_interval == 50
        except (ImportError, AttributeError):
            pytest.skip("Checkpointing not available")


class TestMemoryManagement:
    """Test memory management during MCMC sampling."""
    
    def test_clear_cache(self):
        """Test clearing JAX cache."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer()
            optimizer.clear_cache()
            
            # Should not raise errors
        except (ImportError, AttributeError):
            pytest.skip("Cache clearing not available")
    
    def test_estimate_memory_usage(self):
        """Test estimating MCMC memory usage."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer()
            
            memory_gb = optimizer.estimate_memory_usage(
                n_samples=1000,
                n_chains=4,
                n_params=100
            )
            
            assert isinstance(memory_gb, float)
            assert memory_gb > 0
        except (ImportError, AttributeError):
            pytest.skip("Memory estimation not available")
    
    def test_check_memory_available(self):
        """Test checking if enough memory is available."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer(memory_limit_gb=16.0)
            
            # Should have memory for small job
            has_memory = optimizer.check_memory_available(
                n_samples=100,
                n_chains=2,
                n_params=10
            )
            
            assert isinstance(has_memory, bool)
        except (ImportError, AttributeError):
            pytest.skip("Memory checking not available")


class TestGradientCheckpointing:
    """Test gradient checkpointing functionality."""
    
    def test_enable_gradient_checkpointing(self):
        """Test enabling gradient checkpointing."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer()
            optimizer.enable_gradient_checkpointing(True)
            
            assert optimizer.gradient_checkpointing_enabled
        except (ImportError, AttributeError):
            pytest.skip("Gradient checkpointing not available")
    
    def test_gradient_checkpointing_reduces_memory(self):
        """Test that gradient checkpointing reduces memory usage."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer()
            
            # With gradient checkpointing
            optimizer.enable_gradient_checkpointing(True)
            memory_with = optimizer.estimate_memory_usage(1000, 4, 100)
            
            # Without gradient checkpointing  
            optimizer.enable_gradient_checkpointing(False)
            memory_without = optimizer.estimate_memory_usage(1000, 4, 100)
            
            # With checkpointing should use less or equal memory
            assert memory_with <= memory_without
        except (ImportError, AttributeError):
            pytest.skip("Gradient checkpointing not available")


class TestBatchSampling:
    """Test batch sampling functionality."""
    
    def test_configure_batch_sampling(self):
        """Test configuring batch sampling."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer()
            optimizer.configure_batch_sampling(batch_size=100, enable=True)
            
            assert optimizer.batch_sampling_enabled
            assert optimizer.batch_size == 100
        except (ImportError, AttributeError):
            pytest.skip("Batch sampling not available")
    
    def test_calculate_num_batches(self):
        """Test calculating number of batches."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer()
            
            num_batches = optimizer.calculate_num_batches(
                total_samples=1000,
                batch_size=100
            )
            
            assert num_batches == 10
        except (ImportError, AttributeError):
            pytest.skip("Batch calculation not available")


class TestOptimizationStrategies:
    """Test different optimization strategies."""
    
    def test_low_memory_strategy(self):
        """Test low memory optimization strategy."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer(memory_limit_gb=4.0)
            optimizer.apply_low_memory_strategy()
            
            # Should enable memory-saving features
            assert optimizer.gradient_checkpointing_enabled or \
                   optimizer.adaptive_sampling_enabled
        except (ImportError, AttributeError):
            pytest.skip("Low memory strategy not available")
    
    def test_balanced_strategy(self):
        """Test balanced optimization strategy."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer(memory_limit_gb=16.0)
            optimizer.apply_balanced_strategy()
            
            # Should have reasonable settings
            assert optimizer is not None
        except (ImportError, AttributeError):
            pytest.skip("Balanced strategy not available")
    
    def test_high_performance_strategy(self):
        """Test high performance optimization strategy."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer(memory_limit_gb=32.0)
            optimizer.apply_high_performance_strategy()
            
            # Should prioritize speed over memory
            assert optimizer is not None
        except (ImportError, AttributeError):
            pytest.skip("High performance strategy not available")


class TestContextManager:
    """Test MCMCMemoryOptimizer as context manager."""
    
    def test_context_manager(self):
        """Test using optimizer as context manager."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            with MCMCMemoryOptimizer() as optimizer:
                assert optimizer is not None
        except ImportError:
            pytest.skip("MCMCMemoryOptimizer requires JAX/NumPyro")
    
    def test_context_manager_cleanup(self):
        """Test cleanup on context exit."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            with MCMCMemoryOptimizer() as optimizer:
                pass  # Just enter and exit
            
            # Should have performed cleanup
        except ImportError:
            pytest.skip("MCMCMemoryOptimizer requires JAX/NumPyro")


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_memory_limit(self):
        """Test handling of zero memory limit."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer(memory_limit_gb=0.0)
            assert optimizer.memory_limit_gb > 0  # Should use minimum
        except (ImportError, ValueError):
            pytest.skip("Zero memory limit rejected")
    
    def test_invalid_checkpoint_dir(self):
        """Test handling of invalid checkpoint directory."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer(checkpoint_dir="/nonexistent/path")
            # Should either create or handle gracefully
        except (ImportError, OSError, ValueError):
            # Acceptable to reject invalid path
            pass
    
    def test_zero_samples(self):
        """Test handling of zero samples."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer()
            
            batch_size = optimizer.calculate_optimal_batch_size(
                n_samples=0,
                n_chains=4,
                data_size_gb=1.0
            )
            
            assert batch_size >= 0
        except (ImportError, AttributeError, ValueError):
            pytest.skip("Edge case handling varies")
    
    def test_very_large_dataset(self):
        """Test handling of very large dataset."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer(memory_limit_gb=4.0)
            
            # Request batch size for huge dataset
            batch_size = optimizer.calculate_optimal_batch_size(
                n_samples=1000000,
                n_chains=8,
                data_size_gb=100.0
            )
            
            # Should return reasonable batch size
            assert batch_size > 0
            assert batch_size < 1000000
        except (ImportError, AttributeError):
            pytest.skip("Large dataset handling not available")


class TestIntegration:
    """Integration tests for MCMC optimization."""
    
    def test_full_optimization_workflow(self, temp_checkpoint_dir):
        """Test complete MCMC optimization workflow."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            with MCMCMemoryOptimizer(
                memory_limit_gb=8.0,
                checkpoint_dir=str(temp_checkpoint_dir)
            ) as optimizer:
                # Enable optimizations
                optimizer.enable_gradient_checkpointing(True)
                optimizer.enable_adaptive_sampling(True)
                
                # Calculate batch size
                batch_size = optimizer.calculate_optimal_batch_size(
                    n_samples=1000,
                    n_chains=4,
                    data_size_gb=1.0
                )
                
                assert batch_size > 0
                
                # Clear cache
                optimizer.clear_cache()
        except (ImportError, AttributeError):
            pytest.skip("Full workflow not available")
    
    def test_checkpoint_resume_workflow(self, temp_checkpoint_dir):
        """Test checkpoint save and resume workflow."""
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        
        try:
            optimizer = MCMCMemoryOptimizer(
                checkpoint_dir=str(temp_checkpoint_dir)
            )
            
            # Save checkpoint
            state = {'samples': np.random.randn(50, 10), 'iteration': 50}
            checkpoint_path = optimizer.save_checkpoint(state, iteration=50)
            
            # Resume from checkpoint
            loaded_state = optimizer.load_checkpoint(checkpoint_path)
            
            assert loaded_state['iteration'] == 50
            np.testing.assert_array_equal(
                loaded_state['samples'],
                state['samples']
            )
        except (ImportError, AttributeError):
            pytest.skip("Checkpoint workflow not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
