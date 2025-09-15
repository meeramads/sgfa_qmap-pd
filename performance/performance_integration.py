#!/usr/bin/env python3
"""
Comprehensive performance optimization integration for remote workstation pipeline.
Replaces basic resource management with sophisticated performance optimization tools.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
import time
import psutil

logger = logging.getLogger(__name__)


def get_optimal_performance_configuration(config: Dict, data_characteristics: Dict = None) -> Tuple[Any, Dict]:
    """
    Automatically determine optimal performance configuration based on system and data characteristics.

    Args:
        config: Configuration dictionary
        data_characteristics: Optional data characteristics (subjects, features, views, etc.)

    Returns:
        Tuple of (performance_config, configuration_info)
    """
    try:
        logger.info("🚀 === OPTIMAL PERFORMANCE CONFIGURATION ===")

        # Get system information
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        cpu_count = psutil.cpu_count()

        logger.info(f"System characteristics:")
        logger.info(f"   Total memory: {total_memory_gb:.1f} GB")
        logger.info(f"   Available memory: {available_memory_gb:.1f} GB")
        logger.info(f"   CPU cores: {cpu_count}")

        # Analyze data characteristics if provided
        memory_intensive = False
        large_dataset = False
        mcmc_heavy = False

        if data_characteristics:
            n_subjects = data_characteristics.get('n_subjects', 0)
            total_features = data_characteristics.get('total_features', 0)
            n_views = data_characteristics.get('n_views', 0)

            # Estimate memory requirements
            estimated_data_size_gb = (n_subjects * total_features * 8) / (1024**3)  # float64
            memory_intensive = estimated_data_size_gb > (available_memory_gb * 0.3)
            large_dataset = n_subjects > 200 or total_features > 50000
            mcmc_heavy = config.get('training', {}).get('mcmc_config', {}).get('num_samples', 0) > 1000

            logger.info(f"Data characteristics:")
            logger.info(f"   Subjects: {n_subjects}")
            logger.info(f"   Total features: {total_features}")
            logger.info(f"   Views: {n_views}")
            logger.info(f"   Estimated data size: {estimated_data_size_gb:.2f} GB")
            logger.info(f"   Memory intensive: {memory_intensive}")
            logger.info(f"   Large dataset: {large_dataset}")
            logger.info(f"   MCMC heavy: {mcmc_heavy}")

        # Determine optimal strategy
        if memory_intensive and large_dataset:
            strategy = "aggressive_memory_optimization"
            reason = f"Large dataset ({data_characteristics.get('n_subjects', 0)} subjects, {data_characteristics.get('total_features', 0)} features) requires aggressive memory optimization"
        elif memory_intensive:
            strategy = "memory_efficient"
            reason = f"Memory-intensive workload requires efficient memory management"
        elif mcmc_heavy:
            strategy = "mcmc_optimized"
            reason = f"Heavy MCMC sampling requires MCMC-specific optimizations"
        elif available_memory_gb < 8:
            strategy = "low_memory"
            reason = f"Limited memory ({available_memory_gb:.1f} GB) requires conservative optimization"
        else:
            strategy = "balanced"
            reason = f"Standard optimization for balanced performance"

        # Create performance configuration based on strategy
        from performance.config import PerformanceConfig, MemoryConfig, DataConfig, MCMCConfig, ProfilingConfig

        if strategy == "aggressive_memory_optimization":
            perf_config = PerformanceConfig(
                memory=MemoryConfig(
                    max_memory_gb=available_memory_gb * 0.85,
                    warning_threshold=0.75,
                    critical_threshold=0.90,
                    enable_aggressive_cleanup=True,
                    gc_frequency=50,
                    enable_dtype_optimization=True,
                    target_dtype="float32",
                    enable_monitoring=True,
                    monitoring_interval=0.5
                ),
                data=DataConfig(
                    enable_chunking=True,
                    chunk_size=min(50, data_characteristics.get('n_subjects', 100) // 4) if data_characteristics else 50,
                    memory_limit_gb=available_memory_gb * 0.4,
                    enable_compression=True,
                    compression_level=6,
                    parallel_loading=min(cpu_count, 4)
                ),
                mcmc=MCMCConfig(
                    memory_limit_gb=available_memory_gb * 0.6,
                    enable_checkpointing=True,
                    checkpoint_frequency=100,
                    data_subsample_ratio=0.8,
                    thinning_interval=2,
                    enable_adaptive_batching=True
                ),
                profiling=ProfilingConfig(
                    enable_memory_tracking=True,
                    enable_cpu_tracking=True,
                    sampling_interval=1.0
                )
            )
        elif strategy == "memory_efficient":
            perf_config = PerformanceConfig(
                memory=MemoryConfig(
                    max_memory_gb=available_memory_gb * 0.8,
                    warning_threshold=0.80,
                    critical_threshold=0.90,
                    enable_aggressive_cleanup=True,
                    gc_frequency=75,
                    enable_dtype_optimization=True,
                    target_dtype="float32",
                    enable_monitoring=True
                ),
                data=DataConfig(
                    enable_chunking=True,
                    memory_limit_gb=available_memory_gb * 0.5,
                    enable_compression=True,
                    compression_level=3,
                    parallel_loading=min(cpu_count, 3)
                ),
                mcmc=MCMCConfig(
                    memory_limit_gb=available_memory_gb * 0.5,
                    enable_checkpointing=True,
                    data_subsample_ratio=0.9
                )
            )
        elif strategy == "mcmc_optimized":
            perf_config = PerformanceConfig(
                memory=MemoryConfig(
                    max_memory_gb=available_memory_gb * 0.8,
                    enable_aggressive_cleanup=True,
                    gc_frequency=100
                ),
                mcmc=MCMCConfig(
                    memory_limit_gb=available_memory_gb * 0.7,
                    enable_checkpointing=True,
                    checkpoint_frequency=50,
                    data_subsample_ratio=0.95,
                    thinning_interval=1,
                    enable_adaptive_batching=True
                ),
                profiling=ProfilingConfig(
                    enable_memory_tracking=True,
                    enable_cpu_tracking=True
                )
            )
        elif strategy == "low_memory":
            perf_config = PerformanceConfig(
                memory=MemoryConfig(
                    max_memory_gb=available_memory_gb * 0.7,
                    warning_threshold=0.70,
                    critical_threshold=0.85,
                    enable_aggressive_cleanup=True,
                    gc_frequency=25,
                    enable_dtype_optimization=True,
                    target_dtype="float32",
                    enable_monitoring=True
                ),
                data=DataConfig(
                    enable_chunking=True,
                    chunk_size=20,
                    memory_limit_gb=available_memory_gb * 0.3,
                    enable_compression=True,
                    compression_level=9,
                    parallel_loading=1
                ),
                mcmc=MCMCConfig(
                    memory_limit_gb=available_memory_gb * 0.4,
                    enable_checkpointing=True,
                    checkpoint_frequency=25,
                    data_subsample_ratio=0.7,
                    thinning_interval=3
                )
            )
        else:  # balanced
            perf_config = PerformanceConfig(
                memory=MemoryConfig(
                    max_memory_gb=available_memory_gb * 0.8,
                    enable_monitoring=True
                ),
                data=DataConfig(
                    enable_chunking=True,
                    memory_limit_gb=available_memory_gb * 0.5,
                    parallel_loading=min(cpu_count, 2)
                ),
                mcmc=MCMCConfig(
                    memory_limit_gb=available_memory_gb * 0.6,
                    enable_checkpointing=True
                )
            )

        configuration_info = {
            'strategy': strategy,
            'reason': reason,
            'system_characteristics': {
                'total_memory_gb': total_memory_gb,
                'available_memory_gb': available_memory_gb,
                'cpu_count': cpu_count
            },
            'data_characteristics': data_characteristics or {},
            'optimization_features': {
                'memory_monitoring': perf_config.memory.enable_monitoring,
                'data_chunking': perf_config.data.enable_chunking,
                'memory_compression': perf_config.data.enable_compression,
                'mcmc_checkpointing': perf_config.mcmc.enable_checkpointing,
                'dtype_optimization': perf_config.memory.enable_dtype_optimization
            }
        }

        logger.info(f"✅ Optimal performance strategy selected: {strategy}")
        logger.info(f"   Reason: {reason}")
        logger.info(f"   Memory limit: {perf_config.memory.max_memory_gb:.1f} GB")
        logger.info(f"   Data chunking: {perf_config.data.enable_chunking}")
        logger.info(f"   MCMC checkpointing: {perf_config.mcmc.enable_checkpointing}")

        return perf_config, configuration_info

    except Exception as e:
        logger.error(f"Performance configuration failed: {e}")
        # Fallback to default configuration
        from performance.config import auto_configure_for_system
        fallback_config = auto_configure_for_system()
        fallback_info = {
            'strategy': 'fallback_auto',
            'reason': f'Configuration failed: {e}',
            'error': str(e)
        }
        return fallback_config, fallback_info


def apply_performance_optimization_to_pipeline(config: Dict, X_list: List[np.ndarray] = None,
                                              data_dir: str = None, output_dir: str = None) -> Tuple[Any, Dict]:
    """
    Apply comprehensive performance optimization to the remote workstation pipeline.

    Args:
        config: Configuration dictionary
        X_list: Data list for analysis
        data_dir: Data directory path
        output_dir: Output directory for performance logs

    Returns:
        Tuple of (performance_manager, optimization_info)
    """
    try:
        logger.info("🚀 === PERFORMANCE OPTIMIZATION INTEGRATION ===")

        # Analyze data characteristics
        data_characteristics = {
            'n_subjects': X_list[0].shape[0] if X_list else 0,
            'total_features': sum(X.shape[1] for X in X_list) if X_list else 0,
            'n_views': len(X_list) if X_list else 0,
            'view_shapes': [X.shape for X in X_list] if X_list else []
        }

        # Get optimal performance configuration
        perf_config, config_info = get_optimal_performance_configuration(config, data_characteristics)

        # Initialize performance manager
        try:
            from performance import PerformanceManager
            performance_manager = PerformanceManager(perf_config)
            logger.info("✅ PerformanceManager initialized successfully")
            framework_available = True
        except ImportError as e:
            logger.warning(f"❌ Performance framework not available: {e}")
            return _fallback_performance_optimization(config, X_list, data_dir)

        # Apply data optimization if data is available
        optimized_data_info = {}
        if X_list:
            try:
                logger.info("Applying data loading optimizations...")

                # Check if chunking is beneficial
                if data_characteristics['n_subjects'] > 100 or data_characteristics['total_features'] > 10000:
                    logger.info(f"   Large dataset detected - enabling chunked processing")
                    # Note: Actual chunking would be applied during analysis, not here
                    optimized_data_info['chunking_enabled'] = True
                    optimized_data_info['recommended_chunk_size'] = perf_config.data.chunk_size
                else:
                    logger.info(f"   Standard dataset - chunking not required")
                    optimized_data_info['chunking_enabled'] = False

                # Apply memory optimization
                if perf_config.memory.enable_dtype_optimization:
                    logger.info(f"   Applying dtype optimization to {perf_config.memory.target_dtype}")
                    # Convert data types if memory optimization is enabled
                    if perf_config.memory.target_dtype == "float32":
                        for i, X in enumerate(X_list):
                            if X.dtype == np.float64:
                                X_list[i] = X.astype(np.float32)
                                logger.info(f"   View {i}: {X.dtype} → float32 (memory savings: ~50%)")

                optimized_data_info['dtype_optimization_applied'] = perf_config.memory.enable_dtype_optimization
                optimized_data_info['target_dtype'] = perf_config.memory.target_dtype

            except Exception as data_e:
                logger.warning(f"Data optimization failed: {data_e}")
                optimized_data_info['optimization_failed'] = str(data_e)

        # Setup monitoring if enabled
        monitoring_info = {}
        if perf_config.memory.enable_monitoring:
            try:
                logger.info("Starting performance monitoring...")
                performance_manager.memory_optimizer.start_monitoring()
                monitoring_info['memory_monitoring'] = True
                monitoring_info['monitoring_interval'] = perf_config.memory.monitoring_interval
            except Exception as monitor_e:
                logger.warning(f"Performance monitoring setup failed: {monitor_e}")
                monitoring_info['monitoring_failed'] = str(monitor_e)

        # Setup profiling if output directory provided
        profiling_info = {}
        if output_dir and perf_config.profiling.enable_memory_tracking:
            try:
                profiling_dir = Path(output_dir) / "performance_profiles"
                profiling_dir.mkdir(exist_ok=True)
                logger.info(f"Performance profiling enabled: {profiling_dir}")
                profiling_info['profiling_enabled'] = True
                profiling_info['profiling_dir'] = str(profiling_dir)
            except Exception as prof_e:
                logger.warning(f"Performance profiling setup failed: {prof_e}")
                profiling_info['profiling_failed'] = str(prof_e)

        # Compile optimization information
        optimization_info = {
            'performance_optimization_enabled': True,
            'framework_available': framework_available,
            'configuration_info': config_info,
            'data_optimization': optimized_data_info,
            'monitoring_info': monitoring_info,
            'profiling_info': profiling_info,
            'performance_features': {
                'memory_optimization': True,
                'data_chunking': perf_config.data.enable_chunking,
                'mcmc_optimization': perf_config.mcmc.enable_checkpointing,
                'performance_monitoring': perf_config.memory.enable_monitoring,
                'compression': perf_config.data.enable_compression
            }
        }

        logger.info("✅ Performance optimization integration completed")
        logger.info(f"   Strategy: {config_info['strategy']}")
        logger.info(f"   Memory limit: {perf_config.memory.max_memory_gb:.1f} GB")
        logger.info(f"   Features enabled: {list(optimization_info['performance_features'].keys())}")

        # Wrap performance manager with additional methods
        wrapped_manager = _wrap_performance_manager(performance_manager, optimization_info)
        return wrapped_manager, optimization_info

    except Exception as e:
        logger.error(f"❌ Performance optimization integration failed: {e}")
        return _fallback_performance_optimization(config, X_list, data_dir)


def _fallback_performance_optimization(config: Dict, X_list: List[np.ndarray],
                                     data_dir: str) -> Tuple[None, Dict]:
    """Fallback performance optimization when comprehensive framework is unavailable."""
    try:
        logger.info("🔧 Running fallback performance optimization...")

        # Basic memory optimization
        import gc
        gc.collect()

        # Basic system information
        import psutil
        memory_info = psutil.virtual_memory()

        # Basic dtype optimization
        optimizations_applied = []
        if X_list:
            for i, X in enumerate(X_list):
                if X.dtype == np.float64:
                    X_list[i] = X.astype(np.float32)
                    optimizations_applied.append(f"View {i}: float64 → float32")

        fallback_info = {
            'performance_optimization_enabled': False,
            'framework_available': False,
            'fallback_reason': 'comprehensive_performance_framework_unavailable',
            'basic_optimizations': {
                'garbage_collection': True,
                'dtype_optimization': len(optimizations_applied) > 0,
                'optimizations_applied': optimizations_applied
            },
            'system_info': {
                'total_memory_gb': memory_info.total / (1024**3),
                'available_memory_gb': memory_info.available / (1024**3),
                'memory_percent': memory_info.percent
            }
        }

        logger.info(f"✅ Fallback performance optimization completed")
        logger.info(f"   Basic optimizations applied: {len(optimizations_applied)} dtype conversions")
        logger.info(f"   Available memory: {fallback_info['system_info']['available_memory_gb']:.1f} GB")

        return None, fallback_info

    except Exception as e:
        logger.error(f"❌ Fallback performance optimization failed: {e}")
        return None, {'performance_optimization_enabled': False, 'status': 'failed', 'error': str(e)}


def optimize_mcmc_execution(performance_manager: Any, model_fn: Callable,
                           X_list: List[np.ndarray], hypers: Dict, args: Any, rng_key) -> Tuple[Any, Dict]:
    """
    Apply performance optimizations specifically to MCMC execution.

    Args:
        performance_manager: PerformanceManager instance
        model_fn: Model function for MCMC
        X_list: Input data
        hypers: Hyperparameters
        args: Arguments for model
        rng_key: Random key

    Returns:
        Tuple of (mcmc_results, optimization_info)
    """
    if performance_manager is None:
        logger.info("No performance manager available - running standard MCMC")
        return None, {'mcmc_optimization': False, 'reason': 'no_performance_manager'}

    try:
        logger.info("🔥 Applying MCMC performance optimizations...")

        # Apply MCMC-specific optimizations
        start_time = time.time()

        # Use performance manager's MCMC optimization
        mcmc_results = performance_manager.optimize_mcmc_sampling(
            model_fn=model_fn,
            X_list=X_list,
            hypers=hypers,
            args=args,
            rng_key=rng_key
        )

        optimization_time = time.time() - start_time

        optimization_info = {
            'mcmc_optimization': True,
            'optimization_time_seconds': optimization_time,
            'memory_checkpointing': performance_manager.config.mcmc.enable_checkpointing,
            'adaptive_batching': performance_manager.config.mcmc.enable_adaptive_batching,
            'data_subsampling': performance_manager.config.mcmc.data_subsample_ratio < 1.0
        }

        logger.info(f"✅ MCMC optimization completed in {optimization_time:.2f}s")

        return mcmc_results, optimization_info

    except Exception as e:
        logger.error(f"MCMC optimization failed: {e}")
        return None, {'mcmc_optimization': False, 'error': str(e)}


def integrate_performance_with_pipeline(config: Dict, data_dir: str,
                                       X_list: List[np.ndarray] = None,
                                       output_dir: str = None) -> Tuple[Any, Dict]:
    """
    Main integration function for performance optimization in the pipeline.

    Args:
        config: Configuration dictionary
        data_dir: Data directory path
        X_list: Optional preprocessed data (can be provided later)
        output_dir: Output directory for performance logs

    Returns:
        Tuple of (performance_manager, comprehensive_performance_info)
    """
    try:
        logger.info("🚀 === PERFORMANCE OPTIMIZATION PIPELINE INTEGRATION ===")

        # Apply comprehensive performance optimization
        performance_manager, optimization_info = apply_performance_optimization_to_pipeline(
            config, X_list, data_dir, output_dir
        )

        # Create integration summary
        integration_summary = {
            'performance_integration_enabled': True,
            'framework_available': optimization_info.get('framework_available', False),
            'optimization_strategy': optimization_info.get('configuration_info', {}).get('strategy', 'unknown'),
            'features_enabled': list(optimization_info.get('performance_features', {}).keys()),
            'system_optimized_for': optimization_info.get('configuration_info', {}).get('system_characteristics', {})
        }

        # Add memory optimization details
        if 'data_optimization' in optimization_info:
            data_opt = optimization_info['data_optimization']
            integration_summary['memory_optimization'] = {
                'dtype_optimization': data_opt.get('dtype_optimization_applied', False),
                'chunking_enabled': data_opt.get('chunking_enabled', False),
                'target_dtype': data_opt.get('target_dtype', 'unknown')
            }

        # Combine all information
        comprehensive_performance_info = {
            'performance_integration': integration_summary,
            'optimization_details': optimization_info
        }

        logger.info("✅ Performance optimization pipeline integration completed")
        if integration_summary['framework_available']:
            logger.info(f"   Strategy: {integration_summary['optimization_strategy']}")
            logger.info(f"   Features: {', '.join(integration_summary['features_enabled'])}")
        else:
            logger.info("   Using fallback optimization (framework unavailable)")

        return performance_manager, comprehensive_performance_info

    except Exception as e:
        logger.error(f"❌ Performance optimization pipeline integration failed: {e}")

        # Fallback integration summary
        fallback_summary = {
            'performance_integration_enabled': False,
            'framework_available': False,
            'error': str(e),
            'fallback_optimization': True
        }

        return None, {'status': 'failed', 'error': str(e), 'fallback_info': fallback_summary}


class PerformanceManagerWrapper:
    """Wrapper for PerformanceManager to add additional methods."""

    def __init__(self, manager, optimization_info):
        self.manager = manager
        self.optimization_info = optimization_info

    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped manager."""
        if self.manager:
            return getattr(self.manager, name)
        else:
            raise AttributeError(f"No performance manager available for attribute '{name}'")

    def optimize_data_arrays(self, X_list):
        """
        Optimize data arrays for memory and performance.

        Args:
            X_list: List of data arrays to optimize

        Returns:
            Optimized data arrays
        """
        try:
            if not X_list:
                return []

            optimized_arrays = []

            for i, X in enumerate(X_list):
                # Apply dtype optimization
                if self.optimization_info.get("performance_features", {}).get("dtype_optimization", False):
                    if self.manager and hasattr(self.manager, "memory_optimizer"):
                        X_opt = self.manager.memory_optimizer.optimize_array_memory(X)
                    else:
                        # Fallback dtype optimization
                        if X.dtype == "float64":
                            X_opt = X.astype("float32")
                            logger.debug(f"Optimized array {i} dtype: float64 -> float32")
                        else:
                            X_opt = X
                else:
                    X_opt = X

                optimized_arrays.append(X_opt)

            total_memory_before = sum(X.nbytes for X in X_list) / (1024 * 1024)  # MB
            total_memory_after = sum(X.nbytes for X in optimized_arrays) / (1024 * 1024)  # MB
            logger.info(f"Data optimization: {total_memory_before:.1f}MB -> {total_memory_after:.1f}MB")

            return optimized_arrays

        except Exception as e:
            logger.warning(f"Data array optimization failed: {e}")
            return X_list  # Return original arrays if optimization fails


def _wrap_performance_manager(manager, optimization_info):
    """Wrap performance manager with additional functionality."""
    return PerformanceManagerWrapper(manager, optimization_info)

