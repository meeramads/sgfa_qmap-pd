"""Memory optimization utilities for SGFA analysis."""

import functools
import gc
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Union

import jax
import numpy as np
import psutil

from core.io_utils import save_json

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Comprehensive memory optimization and monitoring."""

    def __init__(
        self,
        max_memory_gb: float = None,
        warning_threshold: float = 0.85,
        critical_threshold: float = 0.95,
        gc_frequency: int = 100,
        enable_profiling: bool = False,
    ):
        """
        Initialize memory optimizer.

        Parameters
        ----------
        max_memory_gb : float, optional
            Maximum memory limit in GB. If None, uses 80% of available memory.
        warning_threshold : float
            Memory usage fraction to trigger warnings.
        critical_threshold : float
            Memory usage fraction to trigger aggressive cleanup.
        gc_frequency : int
            Frequency of garbage collection cycles.
        enable_profiling : bool
            Enable detailed memory profiling.
        """
        self.max_memory_gb = max_memory_gb or self._get_safe_memory_limit()
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.gc_frequency = gc_frequency
        self.enable_profiling = enable_profiling

        self._operation_count = 0
        self._memory_history = []
        self._peak_memory = 0.0
        self._monitoring_active = False
        self._monitor_thread = None

        logger.info(
            f"MemoryOptimizer initialized: max_memory={self.max_memory_gb:.2f}GB"
        )

    def _get_safe_memory_limit(self) -> float:
        """Get safe memory limit (80% of available)."""
        available_gb = psutil.virtual_memory().available / (1024**3)
        return available_gb * 0.8

    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()

        current_gb = memory_info.rss / (1024**3)
        system_memory = psutil.virtual_memory()

        return {
            "current_gb": current_gb,
            "peak_gb": self._peak_memory,
            "system_available_gb": system_memory.available / (1024**3),
            "system_used_percent": system_memory.percent,
            "limit_gb": self.max_memory_gb,
            "usage_fraction": current_gb / self.max_memory_gb,
        }

    def check_memory_pressure(self) -> str:
        """
        Check current memory pressure level.

        Returns
        -------
        str : 'safe', 'warning', 'critical'
        """
        usage = self.get_current_usage()
        fraction = usage["usage_fraction"]

        if fraction >= self.critical_threshold:
            return "critical"
        elif fraction >= self.warning_threshold:
            return "warning"
        else:
            return "safe"

    def aggressive_cleanup(self):
        """Perform aggressive memory cleanup."""
        logger.warning("Performing aggressive memory cleanup")

        # Force garbage collection multiple times
        for _ in range(3):
            collected = gc.collect()
            logger.debug(f"GC collected {collected} objects")

        # Clear JAX compilation cache
        try:
            jax.clear_backends()
            logger.debug("Cleared JAX backends")
        except Exception as e:
            logger.debug(f"Could not clear JAX backends: {e}")

        # Clear NumPy memory pool if available
        try:
            np.memmap._mmap_counter = {}
            logger.debug("Cleared NumPy memmap cache")
        except Exception:
            pass

    def optimize_array_memory(
        self, array: np.ndarray, target_dtype: str = None
    ) -> np.ndarray:
        """
        Optimize array memory usage.

        Parameters
        ----------
        array : np.ndarray
            Input array to optimize.
        target_dtype : str, optional
            Target data type for optimization.

        Returns
        -------
        np.ndarray : Optimized array.
        """
        if target_dtype is None:
            # Auto-select optimal dtype
            if array.dtype == np.float64:
                if np.allclose(array, array.astype(np.float32), rtol=1e-6):
                    target_dtype = np.float32
                    logger.debug(
                        f"Optimized array from float64 to float32, saved {array.nbytes * 0.5 / 1024**2:.1f}MB"
                    )
            elif array.dtype in [np.int64, np.int32]:
                max_val = np.max(np.abs(array))
                if max_val < 32767:
                    target_dtype = np.int16
                elif max_val < 2147483647 and array.dtype == np.int64:
                    target_dtype = np.int32

        if target_dtype and target_dtype != array.dtype:
            return array.astype(target_dtype)

        return array

    def memory_efficient_operation(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with memory monitoring and cleanup.

        Parameters
        ----------
        func : Callable
            Function to execute.
        *args, **kwargs
            Arguments for the function.

        Returns
        -------
        Any : Function result.
        """
        self._operation_count += 1

        # Check memory before operation
        pressure_before = self.check_memory_pressure()
        if pressure_before == "critical":
            self.aggressive_cleanup()

        try:
            with memory_profile(enabled=self.enable_profiling) as profiler:
                result = func(*args, **kwargs)

            # Log memory usage if profiling enabled
            if self.enable_profiling and profiler.peak_memory_gb > 0:
                logger.info(f"Operation used peak {profiler.peak_memory_gb:.2f}GB")

        except MemoryError as e:
            logger.error(f"Memory error during operation: {e}")
            self.aggressive_cleanup()
            raise

        # Periodic cleanup
        if self._operation_count % self.gc_frequency == 0:
            collected = gc.collect()
            logger.debug(f"Periodic GC collected {collected} objects")

        return result

    def start_monitoring(self, interval: float = 1.0):
        """Start background memory monitoring."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_memory, args=(interval,), daemon=True
        )
        self._monitor_thread.start()
        logger.info("Started memory monitoring")

    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Stopped memory monitoring")

    def _monitor_memory(self, interval: float):
        """Background memory monitoring loop."""
        while self._monitoring_active:
            try:
                usage = self.get_current_usage()
                current_gb = usage["current_gb"]

                # Update peak memory
                self._peak_memory = max(self._peak_memory, current_gb)

                # Store history (keep last 1000 points)
                self._memory_history.append(
                    {
                        "timestamp": time.time(),
                        "memory_gb": current_gb,
                        "usage_fraction": usage["usage_fraction"],
                    }
                )
                if len(self._memory_history) > 1000:
                    self._memory_history.pop(0)

                # Check for memory pressure
                pressure = self.check_memory_pressure()
                if pressure == "critical":
                    logger.warning(
                        f"CRITICAL: Memory usage at {usage['usage_fraction']:.1%}"
                    )
                    self.aggressive_cleanup()
                elif pressure == "warning":
                    logger.warning(
                        f"WARNING: Memory usage at {usage['usage_fraction']:.1%}"
                    )

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(interval)

    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory report."""
        usage = self.get_current_usage()

        report = {
            "current_usage": usage,
            "peak_memory_gb": self._peak_memory,
            "operations_completed": self._operation_count,
            "memory_pressure": self.check_memory_pressure(),
            "gc_stats": {"collections": gc.get_stats(), "count": gc.get_count()},
        }

        # Add history statistics if available
        if self._memory_history:
            history_memory = [h["memory_gb"] for h in self._memory_history]
            report["memory_history"] = {
                "mean_gb": np.mean(history_memory),
                "std_gb": np.std(history_memory),
                "min_gb": np.min(history_memory),
                "max_gb": np.max(history_memory),
                "points": len(self._memory_history),
            }

        return report

    def save_memory_profile(self, filepath: Union[str, Path]):
        """Save memory profile to file."""
        report = self.get_memory_report()

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Recursively convert numpy types
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_numpy(obj)

        save_json(deep_convert(report), filepath)

        logger.info(f"Memory profile saved to {filepath}")


class MemoryProfiler:
    """Context manager for memory profiling."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.start_memory = 0.0
        self.peak_memory_gb = 0.0
        self.memory_delta_gb = 0.0

    def __enter__(self):
        if self.enabled:
            self.start_memory = psutil.Process().memory_info().rss / (1024**3)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            end_memory = psutil.Process().memory_info().rss / (1024**3)
            self.memory_delta_gb = end_memory - self.start_memory
            self.peak_memory_gb = max(self.start_memory, end_memory)


def memory_profile(enabled: bool = True):
    """Create memory profiler context manager."""
    return MemoryProfiler(enabled=enabled)


def adaptive_batch_size(
    total_size: int,
    memory_limit_gb: float,
    element_size_bytes: int,
    min_batch: int = 1,
    max_batch: int = None,
) -> int:
    """
    Calculate adaptive batch size based on available memory.

    Parameters
    ----------
    total_size : int
        Total number of elements to process.
    memory_limit_gb : float
        Available memory limit in GB.
    element_size_bytes : int
        Size of each element in bytes.
    min_batch : int
        Minimum batch size.
    max_batch : int, optional
        Maximum batch size.

    Returns
    -------
    int : Optimal batch size.
    """
    # Reserve 20% of memory for other operations
    available_bytes = memory_limit_gb * (1024**3) * 0.8

    # Calculate max elements that fit in memory
    max_elements = int(available_bytes / element_size_bytes)

    # Choose batch size
    batch_size = min(max_elements, total_size)

    if max_batch:
        batch_size = min(batch_size, max_batch)

    batch_size = max(batch_size, min_batch)

    logger.debug(
        f"Adaptive batch size: {batch_size} (total={total_size}, "
        f"memory_limit={memory_limit_gb:.2f}GB)"
    )

    return batch_size


def memory_efficient(
    max_memory_gb: float = None,
    cleanup_frequency: int = 10,
    enable_profiling: bool = False,
):
    """
    Decorator for memory-efficient function execution.

    Parameters
    ----------
    max_memory_gb : float, optional
        Memory limit in GB.
    cleanup_frequency : int
        Frequency of cleanup operations.
    enable_profiling : bool
        Enable memory profiling.
    """

    def decorator(func: Callable) -> Callable:
        optimizer = MemoryOptimizer(
            max_memory_gb=max_memory_gb,
            gc_frequency=cleanup_frequency,
            enable_profiling=enable_profiling,
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return optimizer.memory_efficient_operation(func, *args, **kwargs)

        # Attach optimizer to function for external access
        wrapper._memory_optimizer = optimizer
        return wrapper

    return decorator


@contextmanager
def memory_limit_context(limit_gb: float):
    """Context manager that enforces memory limits."""
    optimizer = MemoryOptimizer(max_memory_gb=limit_gb, enable_profiling=True)
    optimizer.start_monitoring(interval=0.5)  # More frequent monitoring

    try:
        yield optimizer
    finally:
        optimizer.stop_monitoring()

        # Final report
        report = optimizer.get_memory_report()
        logger.info(
            f"Memory context completed: peak={report['peak_memory_gb']:.2f}GB, "
            f"pressure={report['memory_pressure']}"
        )


# Singleton instance for global use
_global_optimizer = None


def get_global_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = MemoryOptimizer()
    return _global_optimizer


def configure_global_optimizer(**kwargs):
    """Configure global memory optimizer."""
    global _global_optimizer
    _global_optimizer = MemoryOptimizer(**kwargs)
    return _global_optimizer
