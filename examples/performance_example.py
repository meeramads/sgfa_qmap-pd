"""Example usage of performance optimization features."""

import logging
from pathlib import Path

import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Basic performance optimization example."""
    print("=" * 60)
    print("BASIC PERFORMANCE OPTIMIZATION EXAMPLE")
    print("=" * 60)

    from optimization import PerformanceManager, auto_configure_for_system

    # Auto-configure for current system
    config = auto_configure_for_system()
    print(
        f"Auto-configured for system: {config.memory.max_memory_gb:.1f}GB memory limit"
    )

    # Use performance manager
    with PerformanceManager(config) as manager:
        print(f"Memory status: {manager.get_memory_status()}")

        # Profile a function
        def example_computation():
            # Simulate some computation
            data = np.random.normal(0, 1, (1000, 500))
            result = np.dot(data, data.T)
            return np.mean(result)

        result = manager.profile_function(example_computation, "matrix_computation")
        print(f"Computation result: {result:.4f}")

        # Get performance metrics
        metrics = manager.profiler.get_current_metrics()
        if metrics:
            print(f"Execution time: {metrics.execution_time:.3f}s")
            print(f"Peak memory: {metrics.peak_memory_gb:.2f}GB")


def example_data_chunking():
    """Data chunking and streaming example."""
    print("\n" + "=" * 60)
    print("DATA CHUNKING EXAMPLE")
    print("=" * 60)

    from data.synthetic import generate_synthetic_data
    from optimization.data_streaming import ChunkedDataLoader

    # Generate synthetic data
    synthetic_data = generate_synthetic_data(num_sources=3, K=5)
    print(
        f"Generated data: {synthetic_data['X_list'][0].shape[0]} subjects, "
        f"{len(synthetic_data['X_list'])} views"
    )

    # Create chunked data loader
    loader = ChunkedDataLoader(memory_limit_gb=2.0)

    # Process data in chunks
    total_processed = 0
    for chunk_id, chunk_data in enumerate(
        loader.load_multiview_data_chunked(synthetic_data, chunk_subjects=50)
    ):
        chunk_size = chunk_data["X_list"][0].shape[0]
        total_processed += chunk_size
        print(
            f"Processed chunk {chunk_id}: {chunk_size} subjects "
            f"(total: {total_processed})"
        )

        if chunk_id >= 2:  # Limit for demo
            break

    print(f"Total subjects processed: {total_processed}")


def example_memory_optimization():
    """Memory optimization example."""
    print("\n" + "=" * 60)
    print("MEMORY OPTIMIZATION EXAMPLE")
    print("=" * 60)

    from optimization.memory_optimizer import MemoryOptimizer, memory_efficient

    # Create memory optimizer
    optimizer = MemoryOptimizer(max_memory_gb=4.0, enable_profiling=True)

    # Start monitoring
    optimizer.start_monitoring()

    @memory_efficient(max_memory_gb=4.0, enable_profiling=True)
    def memory_intensive_function():
        """Example memory-intensive function."""
        # Simulate memory usage
        arrays = []
        for i in range(5):
            arr = np.random.normal(0, 1, (1000, 1000))
            # Optimize array memory
            optimized_arr = optimizer.optimize_array_memory(arr)
            arrays.append(optimized_arr)

        return np.mean([np.mean(arr) for arr in arrays])

    # Run function
    result = memory_intensive_function()
    print(f"Memory-optimized computation result: {result:.6f}")

    # Stop monitoring and get report
    optimizer.stop_monitoring()
    report = optimizer.get_memory_report()

    print(f"Peak memory usage: {report['peak_memory_gb']:.2f}GB")
    print(f"Operations completed: {report['operations_completed']}")
    print(f"Memory pressure: {report['memory_pressure']}")


def example_mcmc_optimization():
    """MCMC optimization example."""
    print("\n" + "=" * 60)
    print("MCMC OPTIMIZATION EXAMPLE")
    print("=" * 60)

    from data.synthetic import generate_synthetic_data
    from optimization.mcmc_optimizer import MCMCMemoryOptimizer

    # Generate test data
    data = generate_synthetic_data(num_sources=2, K=3)
    X_list = data["X_list"]

    # Create MCMC optimizer
    optimizer = MCMCMemoryOptimizer(
        memory_limit_gb=4.0, enable_checkpointing=True, enable_batch_sampling=True
    )

    # Optimize MCMC configuration
    base_config = {"num_samples": 1000, "num_chains": 2, "K": 3}

    optimized_config = optimizer.optimize_mcmc_config(X_list, base_config)

    print(f"Original samples: {base_config['num_samples']}")
    print(f"Optimized samples: {optimized_config['num_samples']}")
    print(f"Estimated memory: {optimized_config['estimated_memory_gb']:.2f}GB")
    print(f"Thinning interval: {optimized_config['thinning_interval']}")
    print(f"Subsample ratio: {optimized_config['subsample_ratio']:.2f}")


def example_configuration_management():
    """Configuration management example."""
    print("\n" + "=" * 60)
    print("CONFIGURATION MANAGEMENT EXAMPLE")
    print("=" * 60)

    from optimization.config import PerformanceConfig, create_memory_efficient_config

    # Create different configuration presets
    configs = {
        "memory_efficient": create_memory_efficient_config(),
        "balanced": PerformanceConfig().create_preset("balanced"),
        "fast": PerformanceConfig().create_preset("fast"),
    }

    print("Configuration Comparison:")
    print("-" * 60)
    print(f"{'Setting':<30} {'Memory Eff':<12} {'Balanced':<12} {'Fast':<12}")
    print("-" * 60)

    for config_name, config in configs.items():
        if config_name == "memory_efficient":
            print(
                f"{'Max Memory (GB)':<30} {config.memory.max_memory_gb:<12.1f}", end=""
            )
        elif config_name == "balanced":
            print(f" {config.memory.max_memory_gb:<12.1f}", end="")
        elif config_name == "fast":
            print(f" {config.memory.max_memory_gb:<12.1f}")

    for config_name, config in configs.items():
        if config_name == "memory_efficient":
            print(
                f"{'Enable Compression':<30} {str(config.data.enable_compression):<12}",
                end="",
            )
        elif config_name == "balanced":
            print(f" {str(config.data.enable_compression):<12}", end="")
        elif config_name == "fast":
            print(f" {str(config.data.enable_compression)}")

    for config_name, config in configs.items():
        if config_name == "memory_efficient":
            print(
                f"{'Enable Profiling':<30} {str(config.profiling.enable_profiling):<12}",
                end="",
            )
        elif config_name == "balanced":
            print(f" {str(config.profiling.enable_profiling):<12}", end="")
        elif config_name == "fast":
            print(f" {str(config.profiling.enable_profiling)}")

    # Save configuration example
    config_file = Path("example_performance_config.yaml")
    configs["balanced"].save(config_file)
    print(f"\nConfiguration saved to: {config_file}")

    # Load configuration example
    loaded_config = PerformanceConfig.load(config_file)
    print(
        f"Loaded configuration with { loaded_config.memory.max_memory_gb:.1f}GB memory limit"
    )

    # Clean up
    config_file.unlink()


def example_benchmarking():
    """Performance benchmarking example."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARKING EXAMPLE")
    print("=" * 60)

    from optimization.profiler import ComparisonBenchmark, benchmark_function

    # Define functions to benchmark
    def numpy_computation():
        data = np.random.normal(0, 1, (500, 500))
        return np.linalg.svd(data, full_matrices=False)

    def matrix_multiply():
        A = np.random.normal(0, 1, (1000, 500))
        B = np.random.normal(0, 1, (500, 300))
        return np.dot(A, B)

    def element_operations():
        data = np.random.normal(0, 1, (2000, 1000))
        return np.exp(np.sin(data)).sum()

    # Create comparison benchmark
    benchmark = ComparisonBenchmark()

    # Add benchmarks
    benchmark.add_benchmark("SVD Decomposition", numpy_computation, iterations=3)
    benchmark.add_benchmark("Matrix Multiplication", matrix_multiply, iterations=5)
    benchmark.add_benchmark("Element Operations", element_operations, iterations=3)

    # Print comparison
    benchmark.print_comparison()

    # Individual function benchmark
    print("\nDetailed benchmark for matrix multiplication:")
    detailed_metrics = benchmark_function(
        matrix_multiply, iterations=10, warmup_iterations=2
    )
    print(f"Average time: {detailed_metrics.execution_time:.4f}s")
    print(f"Std deviation: {detailed_metrics.additional_metrics['std_time']:.4f}s")
    print(f"Min time: {detailed_metrics.additional_metrics['min_time']:.4f}s")
    print(f"Max time: {detailed_metrics.additional_metrics['max_time']:.4f}s")


def example_integration():
    """Integration with existing codebase example."""
    print("\n" + "=" * 60)
    print("INTEGRATION EXAMPLE")
    print("=" * 60)

    from optimization.config import auto_configure_for_system
    from optimization.integration import PerformanceManager, performance_optimized

    # Auto-configure performance settings
    config = auto_configure_for_system()
    print(f"System optimized configuration: {config.memory.max_memory_gb:.1f}GB limit")

    # Use decorator for automatic optimization
    @performance_optimized(config=config, enable_profiling=True)
    def analysis_function(n_samples=1000, n_features=100):
        """Example analysis function."""
        # Simulate data loading
        X = np.random.normal(0, 1, (n_samples, n_features))

        # Simulate preprocessing
        X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        # Simulate model fitting
        U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)

        return {
            "components": Vt[:10],  # Top 10 components
            "explained_variance": s[:10] ** 2,
        }

    # Run optimized analysis
    result = analysis_function(n_samples=2000, n_features=500)
    print(f"Analysis completed. Top eigenvalue: {result['explained_variance'][0]:.2f}")

    # Manual performance management
    with PerformanceManager(config) as manager:
        print(f"Memory status: {manager.get_memory_status()}")

        # Example of data loading optimization
        from data.synthetic import generate_synthetic_data

        data = generate_synthetic_data(num_sources=2, K=4)

        # Process in chunks
        chunk_count = 0
        for chunk in manager.optimize_data_loading(data, chunk_subjects=75):
            chunk_count += 1
            if chunk_count >= 2:  # Limit for demo
                break

        print(f"Processed {chunk_count} data chunks")


if __name__ == "__main__":
    print("Performance Optimization Examples")
    print("=" * 60)

    # Run all examples
    try:
        example_basic_usage()
        example_data_chunking()
        example_memory_optimization()
        example_mcmc_optimization()
        example_configuration_management()
        example_benchmarking()
        example_integration()

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise
