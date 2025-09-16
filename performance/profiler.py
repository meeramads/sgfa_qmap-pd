"""Performance profiling and benchmarking utilities."""

import time
import logging
import functools
import threading
from contextlib import contextmanager
from typing import Dict, Any, Callable, List, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path
import json
import psutil
import gc
from dataclasses import dataclass, asdict
from core.io_utils import save_json, save_csv

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    function_name: str
    execution_time: float
    peak_memory_gb: float
    memory_delta_gb: float
    cpu_percent: float
    iterations: int = 1
    throughput: Optional[float] = None
    additional_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_summary_string(self) -> str:
        """Generate summary string."""
        summary = (f"{self.function_name}: "
                  f"{self.execution_time:.3f}s, "
                  f"{self.peak_memory_gb:.2f}GB peak, "
                  f"{self.cpu_percent:.1f}% CPU")
        
        if self.throughput:
            summary += f", {self.throughput:.1f} items/s"
        
        return summary


class PerformanceProfiler:
    """Comprehensive performance profiler."""
    
    def __init__(self, 
                 enable_memory_tracking: bool = True,
                 enable_cpu_tracking: bool = True,
                 sampling_interval: float = 0.1,
                 auto_gc: bool = True):
        """
        Initialize performance profiler.
        
        Parameters
        ----------
        enable_memory_tracking : bool
            Enable memory usage tracking.
        enable_cpu_tracking : bool
            Enable CPU usage tracking.
        sampling_interval : float
            Sampling interval in seconds.
        auto_gc : bool
            Automatically trigger garbage collection before profiling.
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_cpu_tracking = enable_cpu_tracking
        self.sampling_interval = sampling_interval
        self.auto_gc = auto_gc
        
        self.metrics_history: List[PerformanceMetrics] = []
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        self._current_metrics = None
        
        # Initialize monitoring data
        self._start_time = 0.0
        self._start_memory = 0.0
        self._peak_memory = 0.0
        self._cpu_samples = []

    @contextmanager
    def profile(self, 
                function_name: str,
                expected_items: Optional[int] = None,
                **additional_metrics):
        """
        Context manager for profiling code blocks.
        
        Parameters
        ----------
        function_name : str
            Name of the function/operation being profiled.
        expected_items : int, optional
            Number of items processed for throughput calculation.
        **additional_metrics
            Additional metrics to track.
        """
        if self.auto_gc:
            gc.collect()
        
        # Start monitoring
        self._start_monitoring()
        self._start_time = time.perf_counter()
        
        if self.enable_memory_tracking:
            self._start_memory = psutil.Process().memory_info().rss / (1024**3)
            self._peak_memory = self._start_memory
        
        try:
            yield self
        finally:
            # Stop monitoring and calculate metrics
            end_time = time.perf_counter()
            execution_time = end_time - self._start_time
            
            self._stop_monitoring()
            
            # Calculate final metrics
            memory_delta = 0.0
            peak_memory = 0.0
            cpu_percent = 0.0
            
            if self.enable_memory_tracking:
                end_memory = psutil.Process().memory_info().rss / (1024**3)
                memory_delta = end_memory - self._start_memory
                peak_memory = self._peak_memory
            
            if self.enable_cpu_tracking and self._cpu_samples:
                cpu_percent = np.mean(self._cpu_samples)
            
            # Calculate throughput
            throughput = None
            if expected_items and execution_time > 0:
                throughput = expected_items / execution_time
            
            # Create metrics object
            metrics = PerformanceMetrics(
                function_name=function_name,
                execution_time=execution_time,
                peak_memory_gb=peak_memory,
                memory_delta_gb=memory_delta,
                cpu_percent=cpu_percent,
                throughput=throughput,
                additional_metrics=additional_metrics
            )
            
            self.metrics_history.append(metrics)
            self._current_metrics = metrics
            
            # Log summary
            logger.info(f"Performance: {metrics.to_summary_string()}")

    def _start_monitoring(self):
        """Start background monitoring."""
        if not (self.enable_memory_tracking or self.enable_cpu_tracking):
            return
        
        self._monitoring_active = True
        self._cpu_samples = []
        
        self._monitoring_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self._monitoring_thread.start()

    def _stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1.0)

    def _monitor_resources(self):
        """Background resource monitoring loop."""
        process = psutil.Process()
        
        while self._monitoring_active:
            try:
                # Memory monitoring
                if self.enable_memory_tracking:
                    current_memory = process.memory_info().rss / (1024**3)
                    self._peak_memory = max(self._peak_memory, current_memory)
                
                # CPU monitoring
                if self.enable_cpu_tracking:
                    cpu_percent = process.cpu_percent()
                    if cpu_percent > 0:  # Filter out initial zero readings
                        self._cpu_samples.append(cpu_percent)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.debug(f"Error in resource monitoring: {e}")
                break

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get metrics from the most recent profiling session."""
        return self._current_metrics

    def get_metrics_history(self) -> List[PerformanceMetrics]:
        """Get all historical metrics."""
        return self.metrics_history.copy()

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {"message": "No profiling data available"}
        
        # Aggregate statistics
        execution_times = [m.execution_time for m in self.metrics_history]
        memory_usage = [m.peak_memory_gb for m in self.metrics_history]
        cpu_usage = [m.cpu_percent for m in self.metrics_history if m.cpu_percent > 0]
        
        report = {
            "summary": {
                "total_sessions": len(self.metrics_history),
                "total_execution_time": sum(execution_times),
                "avg_execution_time": np.mean(execution_times),
                "std_execution_time": np.std(execution_times),
                "min_execution_time": min(execution_times),
                "max_execution_time": max(execution_times)
            },
            "memory": {
                "avg_peak_memory_gb": np.mean(memory_usage),
                "max_peak_memory_gb": max(memory_usage),
                "min_peak_memory_gb": min(memory_usage),
                "std_peak_memory_gb": np.std(memory_usage)
            } if memory_usage else {},
            "cpu": {
                "avg_cpu_percent": np.mean(cpu_usage),
                "max_cpu_percent": max(cpu_usage),
                "min_cpu_percent": min(cpu_usage)
            } if cpu_usage else {},
            "details": [m.to_dict() for m in self.metrics_history]
        }
        
        return report

    def save_report(self, filepath: Union[str, Path], format: str = 'json'):
        """
        Save performance report to file.
        
        Parameters
        ----------
        filepath : Union[str, Path]
            Output file path.
        format : str
            Output format ('json', 'csv').
        """
        filepath = Path(filepath)
        report = self.generate_report()
        
        if format == 'json':
            save_json(report, filepath)
        elif format == 'csv':
            df = pd.DataFrame([m.to_dict() for m in self.metrics_history])
            save_csv(df, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Performance report saved to {filepath}")

    def clear_history(self):
        """Clear performance metrics history."""
        self.metrics_history.clear()
        self._current_metrics = None
        logger.debug("Performance metrics history cleared")

    def compare_functions(self, function_names: List[str] = None) -> pd.DataFrame:
        """
        Compare performance across different functions.
        
        Parameters
        ----------
        function_names : List[str], optional
            Specific functions to compare. If None, compares all.
            
        Returns
        -------
        pd.DataFrame : Comparison table.
        """
        if not self.metrics_history:
            return pd.DataFrame()
        
        # Filter metrics by function names if specified
        if function_names:
            filtered_metrics = [m for m in self.metrics_history if m.function_name in function_names]
        else:
            filtered_metrics = self.metrics_history
        
        # Group by function name
        function_groups = {}
        for metric in filtered_metrics:
            if metric.function_name not in function_groups:
                function_groups[metric.function_name] = []
            function_groups[metric.function_name].append(metric)
        
        # Calculate statistics for each function
        comparison_data = []
        for func_name, metrics in function_groups.items():
            exec_times = [m.execution_time for m in metrics]
            memory_usage = [m.peak_memory_gb for m in metrics]
            
            comparison_data.append({
                'function': func_name,
                'count': len(metrics),
                'avg_time': np.mean(exec_times),
                'std_time': np.std(exec_times),
                'min_time': min(exec_times),
                'max_time': max(exec_times),
                'avg_memory_gb': np.mean(memory_usage),
                'max_memory_gb': max(memory_usage),
                'total_time': sum(exec_times)
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('avg_time', ascending=False) if not df.empty else df


def benchmark_function(func: Callable,
                      iterations: int = 10,
                      warmup_iterations: int = 2,
                      enable_profiling: bool = True,
                      **kwargs) -> PerformanceMetrics:
    """
    Benchmark a function with multiple iterations.
    
    Parameters
    ----------
    func : Callable
        Function to benchmark.
    iterations : int
        Number of benchmark iterations.
    warmup_iterations : int
        Number of warmup iterations (not measured).
    enable_profiling : bool
        Enable detailed profiling.
    **kwargs
        Arguments to pass to the function.
        
    Returns
    -------
    PerformanceMetrics : Benchmark results.
    """
    profiler = PerformanceProfiler() if enable_profiling else None
    
    # Warmup iterations
    for _ in range(warmup_iterations):
        try:
            func(**kwargs)
        except Exception as e:
            logger.warning(f"Warmup iteration failed: {e}")
    
    # Benchmark iterations
    execution_times = []
    memory_usage = []
    
    for i in range(iterations):
        if profiler:
            with profiler.profile(f"{func.__name__}_iter_{i}"):
                try:
                    result = func(**kwargs)
                except Exception as e:
                    logger.error(f"Benchmark iteration {i} failed: {e}")
                    continue
            
            current_metrics = profiler.get_current_metrics()
            if current_metrics:
                execution_times.append(current_metrics.execution_time)
                memory_usage.append(current_metrics.peak_memory_gb)
        else:
            # Simple timing without detailed profiling
            start_time = time.perf_counter()
            try:
                result = func(**kwargs)
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)
            except Exception as e:
                logger.error(f"Benchmark iteration {i} failed: {e}")
                continue
    
    # Calculate aggregate metrics
    if not execution_times:
        raise RuntimeError("All benchmark iterations failed")
    
    avg_time = np.mean(execution_times)
    avg_memory = np.mean(memory_usage) if memory_usage else 0.0
    
    metrics = PerformanceMetrics(
        function_name=func.__name__,
        execution_time=avg_time,
        peak_memory_gb=avg_memory,
        memory_delta_gb=0.0,  # Not meaningful for averaged results
        cpu_percent=0.0,      # Not tracked in simple benchmark
        iterations=iterations,
        additional_metrics={
            'std_time': np.std(execution_times),
            'min_time': min(execution_times),
            'max_time': max(execution_times),
            'all_times': execution_times
        }
    )
    
    logger.info(f"Benchmark completed: {metrics.to_summary_string()} "
               f"(std: {metrics.additional_metrics['std_time']:.3f}s)")
    
    return metrics


def profile_memory_usage(func: Callable = None, 
                        interval: float = 0.1,
                        duration: float = None):
    """
    Decorator or context manager for detailed memory profiling.
    
    Parameters
    ----------
    func : Callable, optional
        Function to profile (when used as decorator).
    interval : float
        Memory sampling interval.
    duration : float, optional
        Maximum profiling duration.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            profiler = PerformanceProfiler(sampling_interval=interval)
            
            with profiler.profile(fn.__name__):
                result = fn(*args, **kwargs)
            
            return result
        return wrapper
    
    if func is None:
        # Used as decorator with parameters
        return decorator
    else:
        # Used as simple decorator
        return decorator(func)


class ComparisonBenchmark:
    """Benchmark multiple functions or configurations for comparison."""
    
    def __init__(self):
        self.results: List[PerformanceMetrics] = []
    
    def add_benchmark(self, 
                     name: str,
                     func: Callable,
                     iterations: int = 5,
                     **kwargs) -> PerformanceMetrics:
        """
        Add a function to benchmark.
        
        Parameters
        ----------
        name : str
            Benchmark name.
        func : Callable
            Function to benchmark.
        iterations : int
            Number of iterations.
        **kwargs
            Function arguments.
            
        Returns
        -------
        PerformanceMetrics : Benchmark results.
        """
        logger.info(f"Benchmarking {name}...")
        
        metrics = benchmark_function(
            func, 
            iterations=iterations,
            **kwargs
        )
        metrics.function_name = name
        
        self.results.append(metrics)
        return metrics
    
    def generate_comparison(self) -> pd.DataFrame:
        """Generate comparison table."""
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        for metrics in self.results:
            comparison_data.append({
                'name': metrics.function_name,
                'avg_time': metrics.execution_time,
                'peak_memory_gb': metrics.peak_memory_gb,
                'iterations': metrics.iterations,
                'throughput': metrics.throughput or 0.0,
                'std_time': metrics.additional_metrics.get('std_time', 0.0),
                'min_time': metrics.additional_metrics.get('min_time', 0.0),
                'max_time': metrics.additional_metrics.get('max_time', 0.0)
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('avg_time')
    
    def print_comparison(self):
        """Print formatted comparison table."""
        df = self.generate_comparison()
        if df.empty:
            print("No benchmark results available")
            return
        
        print("\n" + "="*80)
        print("BENCHMARK COMPARISON")
        print("="*80)
        
        for _, row in df.iterrows():
            print(f"{row['name']:30} | "
                  f"Time: {row['avg_time']:8.3f}s ± {row['std_time']:6.3f} | "
                  f"Memory: {row['peak_memory_gb']:6.2f}GB | "
                  f"Iters: {row['iterations']:3d}")
        
        print("="*80)
    
    def clear(self):
        """Clear benchmark results."""
        self.results.clear()


# Global profiler instance
_global_profiler = PerformanceProfiler()

def get_global_profiler() -> PerformanceProfiler:
    """Get global profiler instance."""
    return _global_profiler

def reset_global_profiler():
    """Reset global profiler."""
    global _global_profiler
    _global_profiler = PerformanceProfiler()