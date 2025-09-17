#!/usr/bin/env python
"""
Simple test script to verify optimization integration works.
Tests the core optimization features without full experiment complexity.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_optimization_imports():
    """Test that all optimization modules can be imported."""
    logger.info("🔍 Testing optimization module imports...")

    try:
        from optimization.config import PerformanceConfig, auto_configure_for_system
        from optimization.integration import PerformanceManager, performance_optimized
        from optimization.memory_optimizer import MemoryOptimizer
        from optimization.data_streaming import ChunkedDataLoader
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        from optimization.experiment_mixins import PerformanceOptimizedMixin, performance_optimized_experiment

        logger.info("✅ All optimization imports successful")
        return True

    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_auto_configuration():
    """Test system auto-configuration."""
    logger.info("🔧 Testing auto-configuration...")

    try:
        from optimization.config import auto_configure_for_system

        config = auto_configure_for_system()
        logger.info(f"✅ Auto-config successful: {config.memory.max_memory_gb:.1f}GB memory limit")

        # Test config methods
        config.save("test_config.yaml")
        loaded_config = config.load("test_config.yaml")
        logger.info(f"✅ Config save/load successful")

        # Cleanup
        Path("test_config.yaml").unlink()

        return True

    except Exception as e:
        logger.error(f"❌ Auto-configuration failed: {e}")
        return False


def test_memory_optimization():
    """Test memory optimization features."""
    logger.info("🧠 Testing memory optimization...")

    try:
        from optimization.memory_optimizer import MemoryOptimizer, memory_efficient

        # Test basic memory optimizer
        optimizer = MemoryOptimizer(max_memory_gb=2.0, enable_profiling=True)

        # Create test data
        test_array = np.random.normal(0, 1, (100, 200)).astype(np.float64)
        logger.info(f"   Original array dtype: {test_array.dtype}")

        # Test array optimization
        optimized_array = optimizer.optimize_array_memory(test_array)
        logger.info(f"   Optimized array dtype: {optimized_array.dtype}")

        # Test decorator
        @memory_efficient(max_memory_gb=2.0)
        def test_function():
            return np.mean(np.random.normal(0, 1, (500, 500)))

        result = test_function()
        logger.info(f"✅ Memory optimization successful, result: {result:.4f}")

        return True

    except Exception as e:
        logger.error(f"❌ Memory optimization failed: {e}")
        return False


def test_data_streaming():
    """Test data streaming functionality."""
    logger.info("📊 Testing data streaming...")

    try:
        from optimization.data_streaming import ChunkedDataLoader
        from data.synthetic import generate_synthetic_data

        # Generate test data
        synthetic_data = generate_synthetic_data(num_sources=2, K=3)
        logger.info(f"   Generated data: {len(synthetic_data['X_list'])} views")

        # Test chunked loading
        loader = ChunkedDataLoader(memory_limit_gb=1.0)

        chunk_count = 0
        for chunk_id, chunk_data in enumerate(
            loader.load_multiview_data_chunked(synthetic_data, chunk_subjects=25)
        ):
            chunk_count += 1
            chunk_size = chunk_data["X_list"][0].shape[0]
            logger.info(f"   Processed chunk {chunk_id}: {chunk_size} subjects")

            if chunk_count >= 2:  # Limit for test
                break

        logger.info(f"✅ Data streaming successful, processed {chunk_count} chunks")
        return True

    except Exception as e:
        logger.error(f"❌ Data streaming failed: {e}")
        return False


def test_mcmc_optimization():
    """Test MCMC optimization features."""
    logger.info("⚡ Testing MCMC optimization...")

    try:
        from optimization.mcmc_optimizer import MCMCMemoryOptimizer
        from data.synthetic import generate_synthetic_data

        # Generate test data
        data = generate_synthetic_data(num_sources=2, K=2)
        X_list = data["X_list"]

        # Test MCMC optimizer
        optimizer = MCMCMemoryOptimizer(
            memory_limit_gb=2.0,
            enable_checkpointing=True,
            enable_batch_sampling=True
        )

        base_config = {"num_samples": 100, "num_chains": 1, "K": 2}
        optimized_config = optimizer.optimize_mcmc_config(X_list, base_config)

        logger.info(f"   Original samples: {base_config['num_samples']}")
        logger.info(f"   Optimized samples: {optimized_config['num_samples']}")
        logger.info(f"   Estimated memory: {optimized_config['estimated_memory_gb']:.2f}GB")

        logger.info("✅ MCMC optimization successful")
        return True

    except Exception as e:
        logger.error(f"❌ MCMC optimization failed: {e}")
        return False


def test_performance_manager():
    """Test performance manager integration."""
    logger.info("🚀 Testing performance manager...")

    try:
        from optimization.integration import PerformanceManager, performance_optimized
        from optimization.config import auto_configure_for_system

        config = auto_configure_for_system()

        # Test context manager
        with PerformanceManager(config) as manager:
            logger.info(f"   Memory status: {manager.get_memory_status()}")

            # Test function profiling
            def test_computation():
                return np.mean(np.random.normal(0, 1, (200, 200)))

            result = manager.profile_function(test_computation, "test_computation")
            logger.info(f"   Computation result: {result:.4f}")

            metrics = manager.profiler.get_current_metrics()
            if metrics:
                logger.info(f"   Execution time: {metrics.execution_time:.3f}s")

        # Test decorator
        @performance_optimized(config=config, enable_profiling=True)
        def optimized_function():
            return np.linalg.svd(np.random.normal(0, 1, (100, 100)), full_matrices=False)

        U, s, Vt = optimized_function()
        logger.info(f"✅ Performance manager successful, top singular value: {s[0]:.4f}")

        return True

    except Exception as e:
        logger.error(f"❌ Performance manager failed: {e}")
        return False


def test_experiment_mixins():
    """Test experiment optimization mixins."""
    logger.info("🎯 Testing experiment mixins...")

    try:
        from optimization.experiment_mixins import PerformanceOptimizedMixin, performance_optimized_experiment
        from experiments.framework import ExperimentFramework
        from optimization.config import auto_configure_for_system

        # Test mixin usage
        @performance_optimized_experiment()
        class TestExperiment(PerformanceOptimizedMixin):
            def __init__(self):
                # Initialize with auto-config
                config = auto_configure_for_system()
                self.config = config
                self.logger = logger

            def test_method(self):
                # Test memory optimization context
                with self.memory_optimized_context():
                    # Create test data
                    X_list = [np.random.normal(0, 1, (50, 100)) for _ in range(2)]

                    # Test memory optimization
                    X_opt, saved_memory = self.optimize_arrays_for_memory(X_list)
                    logger.info(f"   Memory saved: {saved_memory:.2f}GB")

                    # Test batch size calculation
                    batch_size = self.calculate_adaptive_batch_size(X_opt)
                    logger.info(f"   Adaptive batch size: {batch_size}")

                    return X_opt

        # Test the experiment
        experiment = TestExperiment()
        result = experiment.test_method()

        logger.info(f"✅ Experiment mixins successful, processed {len(result)} arrays")
        return True

    except Exception as e:
        logger.error(f"❌ Experiment mixins failed: {e}")
        return False


def main():
    """Run all optimization tests."""
    logger.info("🧪 Starting Optimization Integration Tests")
    logger.info("=" * 60)

    tests = [
        ("Import Test", test_optimization_imports),
        ("Auto Configuration", test_auto_configuration),
        ("Memory Optimization", test_memory_optimization),
        ("Data Streaming", test_data_streaming),
        ("MCMC Optimization", test_mcmc_optimization),
        ("Performance Manager", test_performance_manager),
        ("Experiment Mixins", test_experiment_mixins),
    ]

    results = {}
    start_time = time.time()

    for test_name, test_func in tests:
        logger.info(f"\n{'─' * 60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'─' * 60}")

        try:
            success = test_func()
            results[test_name] = "✅ PASS" if success else "❌ FAIL"
        except Exception as e:
            logger.error(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = "💥 CRASH"

    duration = time.time() - start_time

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total duration: {duration:.2f}s")

    passed = len([r for r in results.values() if "PASS" in r])
    failed = len([r for r in results.values() if "FAIL" in r])
    crashed = len([r for r in results.values() if "CRASH" in r])

    logger.info(f"Passed: {passed}/{len(tests)}")
    logger.info(f"Failed: {failed}/{len(tests)}")
    logger.info(f"Crashed: {crashed}/{len(tests)}")

    for test_name, result in results.items():
        logger.info(f"  {result} {test_name}")

    if passed == len(tests):
        logger.info("\n🎉 All optimization features working correctly!")
        return 0
    else:
        logger.info(f"\n⚠️  {failed + crashed} tests had issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())