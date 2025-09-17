"""Reusable performance optimization mixins for experiments."""

import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import auto_configure_for_system
from .data_streaming import ChunkedDataLoader, DataStreamer, memory_efficient_data_context
from .mcmc_optimizer import MCMCMemoryOptimizer
from .memory_optimizer import MemoryOptimizer, adaptive_batch_size, memory_limit_context

logger = logging.getLogger(__name__)


class PerformanceOptimizedMixin:
    """Mixin that adds performance optimization capabilities to experiments."""

    def _initialize_performance_optimization(
        self,
        config=None,
        auto_configure: bool = True,
        enable_memory_optimizer: bool = True,
        enable_data_streaming: bool = True,
        enable_mcmc_optimizer: bool = True
    ):
        """Initialize all performance optimization components."""

        # Auto-configure system if enabled
        if auto_configure:
            self.performance_config = auto_configure_for_system()
            self.logger.info(
                f"Auto-configured system: {self.performance_config.memory.max_memory_gb:.1f}GB memory limit"
            )
            max_memory_gb = self.performance_config.memory.max_memory_gb
        else:
            max_memory_gb = getattr(config, 'max_memory_gb', 8.0)

        # Initialize memory optimizer
        if enable_memory_optimizer:
            self.memory_optimizer = MemoryOptimizer(
                max_memory_gb=max_memory_gb,
                warning_threshold=0.80,
                critical_threshold=0.90,
                enable_profiling=True
            )
            self.logger.info(f"Memory optimizer initialized with {max_memory_gb:.1f}GB limit")

        # Initialize data streamer
        if enable_data_streaming:
            self.data_streamer = DataStreamer(
                memory_limit_gb=max_memory_gb,
                preload_next=True,
                cache_chunks=False  # Disable for experiments to avoid memory buildup
            )

            self.chunked_loader = ChunkedDataLoader(
                memory_limit_gb=max_memory_gb,
                enable_compression=True,
                parallel_loading=False
            )
            self.logger.info("Data streaming components initialized")

        # Initialize MCMC optimizer
        if enable_mcmc_optimizer:
            self.mcmc_optimizer = MCMCMemoryOptimizer(
                memory_limit_gb=max_memory_gb,
                enable_checkpointing=True,
                enable_batch_sampling=True
            )
            self.logger.info("MCMC memory optimizer initialized")

    @contextmanager
    def memory_optimized_context(self):
        """Context manager for memory-optimized experiment execution."""
        if not hasattr(self, 'memory_optimizer'):
            yield
            return

        with memory_limit_context(limit_gb=self.memory_optimizer.max_memory_gb) as mem_ctx:
            self.memory_optimizer.start_monitoring(interval=1.0)
            try:
                yield mem_ctx
            finally:
                self.memory_optimizer.stop_monitoring()
                # Generate memory report
                report = self.memory_optimizer.get_memory_report()
                self.logger.info(
                    f"Memory context completed: peak={report['peak_memory_gb']:.2f}GB, "
                    f"pressure={report['memory_pressure']}"
                )

    def optimize_arrays_for_memory(self, X_list: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """Optimize array memory usage and return memory saved."""
        if not hasattr(self, 'memory_optimizer'):
            return X_list, 0.0

        X_optimized = []
        total_memory_saved = 0.0

        for i, X in enumerate(X_list):
            X_opt = self.memory_optimizer.optimize_array_memory(X)
            memory_saved = X.nbytes - X_opt.nbytes
            total_memory_saved += memory_saved
            X_optimized.append(X_opt)

            if memory_saved > 0:
                self.logger.debug(f"View {i}: Saved {memory_saved / 1024**2:.1f}MB")

        if total_memory_saved > 0:
            self.logger.info(f"Memory optimization saved {total_memory_saved / 1024**2:.1f}MB total")

        return X_optimized, total_memory_saved

    def calculate_adaptive_batch_size(
        self,
        X_list: List[np.ndarray],
        min_batch: int = 5,
        max_batch: Optional[int] = None,
        operation_type: str = "cv"
    ) -> int:
        """Calculate adaptive batch size for different operation types."""
        if not hasattr(self, 'memory_optimizer'):
            return min(50, X_list[0].shape[0])  # Default fallback

        n_subjects = X_list[0].shape[0]
        element_size = sum(X.dtype.itemsize * X.shape[1] for X in X_list)

        # Adjust min/max based on operation type
        if operation_type == "cv":
            min_batch = max(min_batch, 5)  # Minimum for meaningful CV
            max_batch = max_batch or min(50, n_subjects)
        elif operation_type == "mcmc":
            min_batch = max(min_batch, 10)  # Minimum for MCMC
            max_batch = max_batch or n_subjects
        elif operation_type == "traditional":
            min_batch = max(min_batch, 50)  # Minimum for decomposition
            max_batch = max_batch or n_subjects

        batch_size = adaptive_batch_size(
            total_size=n_subjects,
            memory_limit_gb=self.memory_optimizer.max_memory_gb,
            element_size_bytes=element_size,
            min_batch=min_batch,
            max_batch=max_batch
        )

        self.logger.info(f"Adaptive batch size for {operation_type}: {batch_size} subjects")
        return batch_size

    def memory_efficient_operation(self, func: Callable, *args, **kwargs) -> Any:
        """Execute operation with memory monitoring and optimization."""
        if not hasattr(self, 'memory_optimizer'):
            return func(*args, **kwargs)

        return self.memory_optimizer.memory_efficient_operation(func, *args, **kwargs)

    def optimize_mcmc_config(self, X_list: List[np.ndarray], base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize MCMC configuration for current data and system."""
        if not hasattr(self, 'mcmc_optimizer'):
            return {}

        mcmc_config = self.mcmc_optimizer.optimize_mcmc_config(X_list, base_config)

        if mcmc_config:
            self.logger.info(f"MCMC optimized: {list(mcmc_config.keys())}")

        return mcmc_config

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            "auto_configured": hasattr(self, 'performance_config'),
            "components_initialized": {
                "memory_optimizer": hasattr(self, 'memory_optimizer'),
                "data_streamer": hasattr(self, 'data_streamer'),
                "mcmc_optimizer": hasattr(self, 'mcmc_optimizer'),
            }
        }

        if hasattr(self, 'memory_optimizer'):
            summary["memory_report"] = self.memory_optimizer.get_memory_report()

        if hasattr(self, 'performance_config'):
            summary["system_config"] = {
                "max_memory_gb": self.performance_config.memory.max_memory_gb,
                "warning_threshold": self.performance_config.memory.warning_threshold,
                "critical_threshold": self.performance_config.memory.critical_threshold,
            }

        return summary


class StreamingDataMixin:
    """Mixin for memory-efficient data streaming in experiments."""

    def stream_cv_folds(
        self,
        X_list: List[np.ndarray],
        cv_splits: List[Tuple[np.ndarray, np.ndarray]]
    ):
        """Stream CV folds in memory-efficient manner."""
        if not hasattr(self, 'chunked_loader'):
            # Fallback to regular processing
            for train_idx, test_idx in cv_splits:
                yield [X[train_idx] for X in X_list], [X[test_idx] for X in X_list]
            return

        with memory_efficient_data_context(self.chunked_loader.memory_limit_gb) as loader:
            for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
                # Memory-efficient data splitting
                X_train = [X[train_idx] for X in X_list]
                X_test = [X[test_idx] for X in X_list]

                # Optimize arrays
                if hasattr(self, 'memory_optimizer'):
                    X_train = [self.memory_optimizer.optimize_array_memory(X) for X in X_train]
                    X_test = [self.memory_optimizer.optimize_array_memory(X) for X in X_test]

                yield X_train, X_test


def performance_optimized_experiment(
    auto_configure: bool = True,
    enable_memory_optimizer: bool = True,
    enable_data_streaming: bool = True,
    enable_mcmc_optimizer: bool = True
):
    """
    Decorator to add performance optimization to experiment methods.

    Usage:
    ------
    @performance_optimized_experiment()
    def my_experiment_method(self, X_list, ...):
        # Method automatically gets performance optimization
        with self.memory_optimized_context():
            X_optimized, _ = self.optimize_arrays_for_memory(X_list)
            batch_size = self.calculate_adaptive_batch_size(X_optimized)
            # ... rest of experiment
    """
    def decorator(cls):
        # Add the mixin to the class
        if not issubclass(cls, PerformanceOptimizedMixin):
            class EnhancedClass(PerformanceOptimizedMixin, StreamingDataMixin, cls):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    # Initialize performance optimization
                    self._initialize_performance_optimization(
                        config=getattr(self, 'config', None),
                        auto_configure=auto_configure,
                        enable_memory_optimizer=enable_memory_optimizer,
                        enable_data_streaming=enable_data_streaming,
                        enable_mcmc_optimizer=enable_mcmc_optimizer
                    )

            EnhancedClass.__name__ = cls.__name__
            EnhancedClass.__qualname__ = cls.__qualname__
            return EnhancedClass

        return cls

    return decorator


# Convenience functions for common patterns
def create_optimized_experiment_config(**kwargs) -> Dict[str, Any]:
    """Create experiment config with performance optimization defaults."""
    defaults = {
        'auto_configure_system': True,
        'enable_memory_optimization': True,
        'enable_profiling': True,
        'max_memory_gb': None,  # Auto-detected
    }
    defaults.update(kwargs)
    return defaults