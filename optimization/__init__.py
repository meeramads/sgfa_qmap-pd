"""Performance optimization and memory management utilities."""

from .config import PerformanceConfig, auto_configure_for_system
from .data_streaming import ChunkedDataLoader, DataStreamer
from .integration import PerformanceManager, performance_optimized
from .mcmc_optimizer import GradientCheckpointer, MCMCMemoryOptimizer
from .memory_optimizer import MemoryOptimizer, adaptive_batch_size, memory_profile
from .profiler import PerformanceProfiler, benchmark_function
from .experiment_mixins import (
    PerformanceOptimizedMixin,
    StreamingDataMixin,
    performance_optimized_experiment,
    create_optimized_experiment_config,
)

__all__ = [
    "MemoryOptimizer",
    "memory_profile",
    "adaptive_batch_size",
    "DataStreamer",
    "ChunkedDataLoader",
    "MCMCMemoryOptimizer",
    "GradientCheckpointer",
    "PerformanceProfiler",
    "benchmark_function",
    "PerformanceManager",
    "performance_optimized",
    "PerformanceConfig",
    "auto_configure_for_system",
    # Experiment optimization mixins
    "PerformanceOptimizedMixin",
    "StreamingDataMixin",
    "performance_optimized_experiment",
    "create_optimized_experiment_config",
]
