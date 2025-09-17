"""Configuration management for performance optimization."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import psutil
import yaml

from core.io_utils import save_json

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Memory optimization configuration."""

    # Memory limits
    max_memory_gb: float = field(
        default_factory=lambda: psutil.virtual_memory().available / (1024**3) * 0.8
    )
    warning_threshold: float = 0.85
    critical_threshold: float = 0.95

    # Optimization strategies
    enable_aggressive_cleanup: bool = True
    gc_frequency: int = 100
    enable_dtype_optimization: bool = True
    target_dtype: str = "float32"

    # Monitoring
    enable_monitoring: bool = True
    monitoring_interval: float = 1.0
    enable_profiling: bool = False

    def __post_init__(self):
        # Validate thresholds
        if not 0.0 < self.warning_threshold < 1.0:
            raise ValueError("warning_threshold must be between 0 and 1")
        if not 0.0 < self.critical_threshold < 1.0:
            raise ValueError("critical_threshold must be between 0 and 1")
        if self.critical_threshold <= self.warning_threshold:
            raise ValueError(
                "critical_threshold must be greater than warning_threshold"
            )


@dataclass
class DataConfig:
    """Data processing configuration."""

    # Chunking parameters
    enable_chunking: bool = True
    chunk_size: Optional[int] = None  # Auto-calculated if None
    memory_limit_gb: float = 4.0

    # Streaming parameters
    enable_streaming: bool = True
    preload_next_chunk: bool = True
    cache_chunks: bool = False

    # Compression
    enable_compression: bool = True
    compression_level: int = 6

    # I/O optimization
    parallel_loading: bool = False
    io_buffer_size: int = 64 * 1024  # 64KB default

    # Data subsampling
    enable_subsampling: bool = False
    subsample_ratio: float = 1.0
    subsample_seed: int = 42


@dataclass
class MCMCConfig:
    """MCMC optimization configuration."""

    # Memory constraints
    memory_limit_gb: float = 8.0
    enable_memory_efficient_sampling: bool = True

    # Sampling optimization
    enable_adaptive_batching: bool = True
    batch_size: Optional[int] = None
    min_batch_size: int = 50

    # Gradient checkpointing
    enable_checkpointing: bool = True
    checkpoint_frequency: int = 5
    checkpoint_policy: str = "dots_with_no_batch_dims_saveable"  # JAX checkpoint policy

    # GPU memory management
    clear_caches_between_runs: bool = True
    force_device_sync: bool = True

    # Sample thinning
    enable_thinning: bool = False
    thinning_interval: int = 1

    # Chain optimization
    use_diagonal_mass: bool = True  # Less memory than full mass matrix
    max_tree_depth: int = 8

    # Convergence and efficiency
    target_accept_prob: float = 0.8
    adapt_step_size: bool = True

    # Data subsampling for very large datasets
    enable_data_subsampling: bool = False
    data_subsample_ratio: float = 1.0


@dataclass
class ProfilingConfig:
    """Performance profiling configuration."""

    # Profiling settings
    enable_profiling: bool = True
    enable_memory_tracking: bool = True
    enable_cpu_tracking: bool = True
    sampling_interval: float = 0.1

    # Benchmarking
    benchmark_iterations: int = 5
    warmup_iterations: int = 2

    # Reporting
    auto_save_reports: bool = False
    report_format: str = "json"  # json, csv
    report_directory: Optional[str] = None

    # Comparison benchmarks
    enable_comparisons: bool = False
    comparison_functions: list = field(default_factory=list)


@dataclass
class PerformanceConfig:
    """Master performance configuration."""

    # Sub-configurations
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mcmc: MCMCConfig = field(default_factory=MCMCConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)

    # Global settings
    device: str = "cpu"  # cpu, gpu, tpu
    num_threads: Optional[int] = None  # Auto-detect if None
    enable_jit: bool = True

    # Experimental features
    experimental_features: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        # Set default num_threads
        if self.num_threads is None:
            self.num_threads = min(8, psutil.cpu_count())

        # Validate device
        if self.device not in ["cpu", "gpu", "tpu"]:
            raise ValueError(f"Invalid device: {self.device}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PerformanceConfig":
        """Create configuration from dictionary."""
        # Extract sub-configurations
        memory_config = MemoryConfig(**config_dict.get("memory", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        mcmc_config = MCMCConfig(**config_dict.get("mcmc", {}))
        profiling_config = ProfilingConfig(**config_dict.get("profiling", {}))

        # Extract global settings
        global_settings = {
            k: v
            for k, v in config_dict.items()
            if k not in ["memory", "data", "mcmc", "profiling"]
        }

        return cls(
            memory=memory_config,
            data=data_config,
            mcmc=mcmc_config,
            profiling=profiling_config,
            **global_settings,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "memory": {
                "max_memory_gb": self.memory.max_memory_gb,
                "warning_threshold": self.memory.warning_threshold,
                "critical_threshold": self.memory.critical_threshold,
                "enable_aggressive_cleanup": self.memory.enable_aggressive_cleanup,
                "gc_frequency": self.memory.gc_frequency,
                "enable_dtype_optimization": self.memory.enable_dtype_optimization,
                "target_dtype": self.memory.target_dtype,
                "enable_monitoring": self.memory.enable_monitoring,
                "monitoring_interval": self.memory.monitoring_interval,
                "enable_profiling": self.memory.enable_profiling,
            },
            "data": {
                "enable_chunking": self.data.enable_chunking,
                "chunk_size": self.data.chunk_size,
                "memory_limit_gb": self.data.memory_limit_gb,
                "enable_streaming": self.data.enable_streaming,
                "preload_next_chunk": self.data.preload_next_chunk,
                "cache_chunks": self.data.cache_chunks,
                "enable_compression": self.data.enable_compression,
                "compression_level": self.data.compression_level,
                "parallel_loading": self.data.parallel_loading,
                "io_buffer_size": self.data.io_buffer_size,
                "enable_subsampling": self.data.enable_subsampling,
                "subsample_ratio": self.data.subsample_ratio,
                "subsample_seed": self.data.subsample_seed,
            },
            "mcmc": {
                "memory_limit_gb": self.mcmc.memory_limit_gb,
                "enable_memory_efficient_sampling": self.mcmc.enable_memory_efficient_sampling,
                "enable_adaptive_batching": self.mcmc.enable_adaptive_batching,
                "batch_size": self.mcmc.batch_size,
                "min_batch_size": self.mcmc.min_batch_size,
                "enable_checkpointing": self.mcmc.enable_checkpointing,
                "checkpoint_frequency": self.mcmc.checkpoint_frequency,
                "enable_thinning": self.mcmc.enable_thinning,
                "thinning_interval": self.mcmc.thinning_interval,
                "use_diagonal_mass": self.mcmc.use_diagonal_mass,
                "max_tree_depth": self.mcmc.max_tree_depth,
                "target_accept_prob": self.mcmc.target_accept_prob,
                "adapt_step_size": self.mcmc.adapt_step_size,
                "enable_data_subsampling": self.mcmc.enable_data_subsampling,
                "data_subsample_ratio": self.mcmc.data_subsample_ratio,
            },
            "profiling": {
                "enable_profiling": self.profiling.enable_profiling,
                "enable_memory_tracking": self.profiling.enable_memory_tracking,
                "enable_cpu_tracking": self.profiling.enable_cpu_tracking,
                "sampling_interval": self.profiling.sampling_interval,
                "benchmark_iterations": self.profiling.benchmark_iterations,
                "warmup_iterations": self.profiling.warmup_iterations,
                "auto_save_reports": self.profiling.auto_save_reports,
                "report_format": self.profiling.report_format,
                "report_directory": self.profiling.report_directory,
                "enable_comparisons": self.profiling.enable_comparisons,
                "comparison_functions": self.profiling.comparison_functions,
            },
            "device": self.device,
            "num_threads": self.num_threads,
            "enable_jit": self.enable_jit,
            "experimental_features": self.experimental_features,
        }

    def save(self, filepath: Union[str, Path], format: str = "yaml"):
        """
        Save configuration to file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Output file path.
        format : str
            File format (yaml, json).
        """
        filepath = Path(filepath)
        config_dict = self.to_dict()

        if format.lower() == "yaml":
            with open(filepath, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            save_json(config_dict, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Performance configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PerformanceConfig":
        """
        Load configuration from file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Input file path.

        Returns
        -------
        PerformanceConfig : Loaded configuration.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        if filepath.suffix.lower() in [".yml", ".yaml"]:
            with open(filepath, "r") as f:
                config_dict = yaml.safe_load(f)
        elif filepath.suffix.lower() == ".json":
            with open(filepath, "r") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        logger.info(f"Performance configuration loaded from {filepath}")
        return cls.from_dict(config_dict)

    def optimize_for_system(self):
        """Automatically optimize configuration for current system."""
        # Get system information
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()

        # Optimize memory settings
        available_gb = memory.available / (1024**3)
        self.memory.max_memory_gb = min(available_gb * 0.8, self.memory.max_memory_gb)

        # Optimize data processing
        if available_gb < 4.0:
            # Low memory system
            self.data.memory_limit_gb = min(2.0, available_gb * 0.4)
            self.data.enable_compression = True
            self.data.cache_chunks = False
        elif available_gb > 16.0:
            # High memory system
            self.data.memory_limit_gb = min(8.0, available_gb * 0.3)
            self.data.cache_chunks = True
            self.data.parallel_loading = True

        # Optimize MCMC settings
        if available_gb < 8.0:
            self.mcmc.memory_limit_gb = min(4.0, available_gb * 0.5)
            self.mcmc.enable_adaptive_batching = True
            self.mcmc.use_diagonal_mass = True

        # Optimize threading
        self.num_threads = min(cpu_count, 8)  # Cap at 8 to avoid overhead

        # Enable optimizations for limited memory
        if available_gb < 8.0:
            self.memory.enable_aggressive_cleanup = True
            self.memory.enable_dtype_optimization = True
            self.mcmc.enable_checkpointing = True

        logger.info(
            f"Configuration optimized for system: "
            f"{available_gb:.1f}GB RAM, {cpu_count} CPUs"
        )

    def create_preset(self, preset_name: str) -> "PerformanceConfig":
        """
        Create configuration preset.

        Parameters
        ----------
        preset_name : str
            Preset name: 'fast', 'balanced', 'memory_efficient', 'high_precision'.

        Returns
        -------
        PerformanceConfig : Configured preset.
        """
        config = PerformanceConfig()

        if preset_name == "fast":
            # Optimize for speed
            config.memory.enable_monitoring = False
            config.memory.enable_profiling = False
            config.data.enable_compression = False
            config.data.cache_chunks = True
            config.mcmc.enable_checkpointing = False
            config.mcmc.use_diagonal_mass = True
            config.profiling.enable_profiling = False

        elif preset_name == "balanced":
            # Balanced performance and memory usage
            config.memory.enable_monitoring = True
            config.data.enable_compression = True
            config.data.compression_level = 3
            config.mcmc.enable_checkpointing = True
            config.profiling.enable_profiling = True

        elif preset_name == "memory_efficient":
            # Optimize for low memory usage
            config.memory.max_memory_gb = min(4.0, config.memory.max_memory_gb)
            config.memory.warning_threshold = 0.75
            config.memory.critical_threshold = 0.90
            config.memory.enable_aggressive_cleanup = True
            config.memory.gc_frequency = 50
            config.data.memory_limit_gb = 2.0
            config.data.enable_compression = True
            config.data.compression_level = 9
            config.data.cache_chunks = False
            config.mcmc.memory_limit_gb = 3.0
            config.mcmc.enable_adaptive_batching = True
            config.mcmc.enable_checkpointing = True
            config.mcmc.enable_thinning = True
            config.mcmc.thinning_interval = 2

        elif preset_name == "high_precision":
            # Optimize for accuracy over speed/memory
            config.memory.target_dtype = "float64"
            config.data.enable_compression = False
            config.mcmc.use_diagonal_mass = False
            config.mcmc.max_tree_depth = 12
            config.mcmc.enable_thinning = False
            config.profiling.enable_profiling = True
            config.profiling.enable_comparisons = True

        else:
            raise ValueError(f"Unknown preset: {preset_name}")

        logger.info(f"Created '{preset_name}' configuration preset")
        return config


# Convenience functions for common configurations


def create_memory_efficient_config(max_memory_gb: float = None) -> PerformanceConfig:
    """Create memory-efficient configuration."""
    config = PerformanceConfig()
    return config.create_preset("memory_efficient")


def create_fast_config() -> PerformanceConfig:
    """Create speed-optimized configuration."""
    config = PerformanceConfig()
    return config.create_preset("fast")


def create_balanced_config() -> PerformanceConfig:
    """Create balanced configuration."""
    config = PerformanceConfig()
    return config.create_preset("balanced")


def auto_configure_for_system() -> PerformanceConfig:
    """Automatically configure for current system."""
    config = PerformanceConfig()
    config.optimize_for_system()
    return config


# Default configuration
DEFAULT_CONFIG = PerformanceConfig()
