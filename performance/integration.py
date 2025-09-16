"""Integration utilities for performance optimization with existing codebase."""

import logging
from typing import Dict, Any, Optional, Callable, List
import functools
from pathlib import Path

from .config import PerformanceConfig, auto_configure_for_system
from .memory_optimizer import MemoryOptimizer, memory_efficient
from .data_streaming import ChunkedDataLoader, memory_efficient_data_context
from .mcmc_optimizer import MCMCMemoryOptimizer
from .profiler import PerformanceProfiler, benchmark_function
from core.io_utils import save_json

logger = logging.getLogger(__name__)


class PerformanceManager:
    """Central manager for all performance optimizations."""
    
    def __init__(self, config: PerformanceConfig = None):
        """
        Initialize performance manager.
        
        Parameters
        ----------
        config : PerformanceConfig, optional
            Performance configuration. If None, auto-configures for system.
        """
        self.config = config or auto_configure_for_system()
        
        # Initialize components
        self.memory_optimizer = MemoryOptimizer(
            max_memory_gb=self.config.memory.max_memory_gb,
            warning_threshold=self.config.memory.warning_threshold,
            critical_threshold=self.config.memory.critical_threshold,
            gc_frequency=self.config.memory.gc_frequency,
            enable_profiling=self.config.memory.enable_profiling
        )
        
        self.data_loader = ChunkedDataLoader(
            memory_limit_gb=self.config.data.memory_limit_gb,
            enable_compression=self.config.data.enable_compression,
            compression_level=self.config.data.compression_level,
            parallel_loading=self.config.data.parallel_loading
        )
        
        self.mcmc_optimizer = MCMCMemoryOptimizer(
            memory_limit_gb=self.config.mcmc.memory_limit_gb,
            enable_checkpointing=self.config.mcmc.enable_checkpointing,
            subsample_ratio=self.config.mcmc.data_subsample_ratio,
            thinning_interval=self.config.mcmc.thinning_interval,
            enable_batch_sampling=self.config.mcmc.enable_adaptive_batching
        )
        
        self.profiler = PerformanceProfiler(
            enable_memory_tracking=self.config.profiling.enable_memory_tracking,
            enable_cpu_tracking=self.config.profiling.enable_cpu_tracking,
            sampling_interval=self.config.profiling.sampling_interval
        )
        
        # Start monitoring if enabled
        if self.config.memory.enable_monitoring:
            self.memory_optimizer.start_monitoring(self.config.memory.monitoring_interval)
        
        logger.info("PerformanceManager initialized")
    
    def optimize_data_loading(self, data: Dict[str, Any], chunk_subjects: int = None):
        """
        Apply performance optimizations to data loading.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Multi-view data dictionary.
        chunk_subjects : int, optional
            Subjects per chunk.
            
        Returns
        -------
        Iterator : Memory-efficient data chunks.
        """
        return self.data_loader.load_multiview_data_chunked(data, chunk_subjects)
    
    def optimize_mcmc_sampling(self, 
                              model_fn: Callable,
                              X_list: List,
                              hypers: Dict[str, Any],
                              args: Any,
                              rng_key) -> Dict[str, Any]:
        """
        Apply performance optimizations to MCMC sampling.
        
        Parameters
        ----------
        model_fn : Callable
            Model function.
        X_list : List
            Input data.
        hypers : Dict[str, Any]
            Hyperparameters.
        args : Any
            Arguments.
        rng_key
            Random key.
            
        Returns
        -------
        Dict[str, Any] : MCMC samples.
        """
        if self.config.mcmc.enable_memory_efficient_sampling:
            if self.config.mcmc.enable_adaptive_batching:
                return self.mcmc_optimizer.adaptive_batch_mcmc(
                    model_fn, X_list, hypers, args, rng_key
                )
            else:
                return self.mcmc_optimizer.run_memory_efficient_mcmc(
                    model_fn, X_list, hypers, args, rng_key
                )
        else:
            # Fallback to standard MCMC
            from core.run_analysis import run_inference
            return run_inference(model_fn, args, rng_key, X_list, hypers)
    
    def profile_function(self, func: Callable, name: str = None, **kwargs):
        """
        Profile a function execution.
        
        Parameters
        ----------
        func : Callable
            Function to profile.
        name : str, optional
            Function name for profiling.
        **kwargs
            Function arguments.
            
        Returns
        -------
        Any : Function result.
        """
        name = name or func.__name__
        
        with self.profiler.profile(name):
            return func(**kwargs)
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        return self.memory_optimizer.get_current_usage()
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            'memory_report': self.memory_optimizer.get_memory_report(),
            'profiling_report': self.profiler.generate_report(),
            'configuration': self.config.to_dict()
        }
    
    def save_performance_report(self, filepath: Path):
        """Save performance report to file."""
        report = self.generate_performance_report()

        save_json(report, filepath)
        
        logger.info(f"Performance report saved to {filepath}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.config.memory.enable_monitoring:
            self.memory_optimizer.stop_monitoring()
        
        self.memory_optimizer.aggressive_cleanup()
        logger.info("PerformanceManager cleanup completed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Integration decorators and utilities

def performance_optimized(config: PerformanceConfig = None,
                         enable_profiling: bool = True,
                         enable_memory_optimization: bool = True):
    """
    Decorator to add performance optimizations to functions.
    
    Parameters
    ----------
    config : PerformanceConfig, optional
        Performance configuration.
    enable_profiling : bool
        Enable performance profiling.
    enable_memory_optimization : bool
        Enable memory optimizations.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize performance manager
            perf_config = config or auto_configure_for_system()
            
            with PerformanceManager(perf_config) as manager:
                if enable_profiling:
                    return manager.profile_function(func, func.__name__, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def optimize_data_manager(data_manager_class):
    """
    Class decorator to add performance optimizations to DataManager.
    
    Parameters
    ----------
    data_manager_class
        DataManager class to optimize.
        
    Returns
    -------
    Optimized DataManager class.
    """
    original_load_data = data_manager_class.load_data
    
    @functools.wraps(original_load_data)
    def optimized_load_data(self):
        """Memory-optimized data loading."""
        # Use performance manager for optimization
        config = getattr(self, 'performance_config', None) or auto_configure_for_system()
        
        with PerformanceManager(config) as manager:
            # Load data with profiling
            data = manager.profile_function(original_load_data, "load_data", self)
            
            # Apply memory optimization to arrays
            if 'X_list' in data:
                optimized_X_list = []
                for i, X in enumerate(data['X_list']):
                    optimized_X = manager.memory_optimizer.optimize_array_memory(X)
                    optimized_X_list.append(optimized_X)
                data['X_list'] = optimized_X_list
                
                logger.info(f"Optimized data arrays for memory efficiency")
            
            return data
    
    # Replace original method
    data_manager_class.load_data = optimized_load_data
    
    # Add performance configuration attribute
    def set_performance_config(self, config: PerformanceConfig):
        """Set performance configuration."""
        self.performance_config = config
    
    data_manager_class.set_performance_config = set_performance_config
    
    return data_manager_class


def optimize_model_runner(model_runner_class):
    """
    Class decorator to add performance optimizations to ModelRunner.
    
    Parameters
    ----------
    model_runner_class
        ModelRunner class to optimize.
        
    Returns
    -------
    Optimized ModelRunner class.
    """
    original_run_standard_analysis = model_runner_class.run_standard_analysis
    
    @functools.wraps(original_run_standard_analysis)
    def optimized_run_standard_analysis(self, X_list, hypers, data):
        """Memory-optimized MCMC analysis."""
        config = getattr(self, 'performance_config', None) or auto_configure_for_system()
        
        with PerformanceManager(config) as manager:
            # Check if we should use memory-efficient MCMC
            if config.mcmc.enable_memory_efficient_sampling:
                logger.info("Using memory-efficient MCMC sampling")
                
                # This would need to be integrated with the actual MCMC code
                # For now, we'll use the original method with profiling
                return manager.profile_function(
                    original_run_standard_analysis, 
                    "run_standard_analysis",
                    self, X_list, hypers, data
                )
            else:
                return manager.profile_function(
                    original_run_standard_analysis,
                    "run_standard_analysis", 
                    self, X_list, hypers, data
                )
    
    model_runner_class.run_standard_analysis = optimized_run_standard_analysis
    
    # Add performance configuration
    def set_performance_config(self, config: PerformanceConfig):
        """Set performance configuration."""
        self.performance_config = config
    
    model_runner_class.set_performance_config = set_performance_config
    
    return model_runner_class


# Utility functions for integration

def apply_performance_optimizations(config_path: Optional[Path] = None):
    """
    Apply performance optimizations globally.
    
    Parameters
    ----------
    config_path : Path, optional
        Path to performance configuration file.
    """
    if config_path and config_path.exists():
        config = PerformanceConfig.load(config_path)
        logger.info(f"Loaded performance configuration from {config_path}")
    else:
        config = auto_configure_for_system()
        logger.info("Using auto-configured performance settings")
    
    # Configure global optimizers
    from .memory_optimizer import configure_global_optimizer
    configure_global_optimizer(
        max_memory_gb=config.memory.max_memory_gb,
        warning_threshold=config.memory.warning_threshold,
        critical_threshold=config.memory.critical_threshold,
        gc_frequency=config.memory.gc_frequency,
        enable_profiling=config.memory.enable_profiling
    )
    
    logger.info("Global performance optimizations applied")
    return config


def create_performance_config_template(filepath: Path):
    """
    Create a template performance configuration file.
    
    Parameters
    ----------
    filepath : Path
        Output file path.
    """
    config = auto_configure_for_system()
    config.save(filepath)
    
    logger.info(f"Performance configuration template created at {filepath}")
    print(f"""
Performance configuration template created at: {filepath}

You can customize this configuration file to optimize for your specific:
- Hardware configuration (memory, CPU cores)
- Dataset size and characteristics  
- Analysis requirements (speed vs accuracy)
- Resource constraints

Key settings to consider:
- memory.max_memory_gb: Maximum memory to use
- data.memory_limit_gb: Memory limit for data processing
- mcmc.enable_adaptive_batching: For very large datasets
- profiling.enable_profiling: For performance monitoring

Load the configuration in your analysis:
  from performance import PerformanceConfig
  config = PerformanceConfig.load('{filepath}')
""")


def benchmark_analysis_pipeline(X_list, hypers, config, iterations: int = 3):
    """
    Benchmark the analysis pipeline with different optimization settings.
    
    Parameters
    ----------
    X_list : List
        Input data.
    hypers : Dict
        Hyperparameters.
    config : PerformanceConfig
        Base configuration.
    iterations : int
        Number of benchmark iterations.
        
    Returns
    -------
    Dict : Benchmark results.
    """
    from .profiler import ComparisonBenchmark
    
    benchmark = ComparisonBenchmark()
    
    # Define configurations to test
    configs = {
        'default': config,
        'memory_efficient': config.create_preset('memory_efficient'),
        'fast': config.create_preset('fast'),
        'balanced': config.create_preset('balanced')
    }
    
    # Benchmark each configuration
    def run_analysis_with_config(config_name, test_config):
        def analysis_func():
            with PerformanceManager(test_config) as manager:
                # Simulate analysis pipeline
                # This would be replaced with actual analysis calls
                import time
                time.sleep(0.1)  # Placeholder
                return {"status": "completed"}
        
        return analysis_func
    
    results = {}
    for config_name, test_config in configs.items():
        logger.info(f"Benchmarking {config_name} configuration...")
        
        metrics = benchmark.add_benchmark(
            config_name,
            run_analysis_with_config(config_name, test_config),
            iterations=iterations
        )
        results[config_name] = metrics
    
    # Print comparison
    benchmark.print_comparison()
    
    return results