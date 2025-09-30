"""Configuration schema validation for SGFA qMAP-PD analysis."""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors."""


class LogLevel(Enum):
    """Valid logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelType(Enum):
    """Valid model types."""

    SPARSE_GFA = "sparse_gfa"
    STANDARD_GFA = "standard_gfa"


class PreprocessingStrategy(Enum):
    """Valid preprocessing strategies."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    ADVANCED = "advanced"
    CLINICAL_FOCUSED = "clinical_focused"


@dataclass
class DataConfig:
    """Data configuration schema."""

    data_dir: str
    clinical_file: Optional[str] = None
    volume_dir: Optional[str] = None
    imaging_as_single_view: bool = True

    def validate(self) -> List[str]:
        """Validate data configuration."""
        errors = []

        # Check data directory
        if not self.data_dir:
            errors.append("data_dir is required")
        elif not Path(self.data_dir).exists():
            errors.append(f"data_dir does not exist: {self.data_dir}")

        return errors


@dataclass
class ExperimentsConfig:
    """Experiments configuration schema."""

    base_output_dir: str
    save_intermediate: bool = True
    generate_plots: bool = True
    enable_spatial_analysis: bool = True
    save_pickle_results: bool = False
    max_parallel_jobs: int = 1

    def validate(self) -> List[str]:
        """Validate experiments configuration."""
        errors = []

        # Check output directory
        if not self.base_output_dir:
            errors.append("base_output_dir is required")

        # Check parallel jobs
        if self.max_parallel_jobs < 1:
            errors.append("max_parallel_jobs must be >= 1")
        elif self.max_parallel_jobs > 16:
            errors.append("max_parallel_jobs should not exceed 16")

        return errors


@dataclass
class ModelConfig:
    """Model configuration schema."""

    model_type: str
    K: int
    num_samples: int = 1000
    num_warmup: int = 500
    num_chains: int = 2
    sparsity_lambda: Optional[float] = None
    group_lambda: Optional[float] = None
    random_seed: Optional[int] = None

    def validate(self) -> List[str]:
        """Validate model configuration."""
        errors = []

        # Check model type
        valid_types = [t.value for t in ModelType]
        if self.model_type not in valid_types:
            errors.append(
                f"model_type must be one of {valid_types}, got {self.model_type}"
            )

        # Check K (number of factors)
        if self.K < 1:
            errors.append("K (number of factors) must be >= 1")
        elif self.K > 50:
            errors.append("K (number of factors) should not exceed 50")

        # Check MCMC parameters
        if self.num_samples < 10:
            errors.append("num_samples must be >= 10")
        elif self.num_samples > 50000:
            errors.append("num_samples should not exceed 50000 for performance")

        if self.num_warmup < 10:
            errors.append("num_warmup must be >= 10")
        elif self.num_warmup >= self.num_samples:
            errors.append("num_warmup must be < num_samples")

        if self.num_chains < 1:
            errors.append("num_chains must be >= 1")
        elif self.num_chains > 8:
            errors.append("num_chains should not exceed 8")

        # Check sparsity parameters
        if self.sparsity_lambda is not None:
            if self.sparsity_lambda < 0:
                errors.append("sparsity_lambda must be >= 0")
            elif self.sparsity_lambda > 10:
                errors.append("sparsity_lambda should not exceed 10")

        if self.group_lambda is not None:
            if self.group_lambda < 0:
                errors.append("group_lambda must be >= 0")
            elif self.group_lambda > 10:
                errors.append("group_lambda should not exceed 10")

        # Check random seed
        if self.random_seed is not None:
            if not isinstance(self.random_seed, int):
                errors.append("random_seed must be an integer")
            elif self.random_seed < 0:
                errors.append("random_seed must be >= 0")

        return errors


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration schema."""

    strategy: str = "standard"
    enable_advanced_preprocessing: bool = True
    enable_spatial_processing: bool = True
    imputation_strategy: str = "median"
    feature_selection_method: str = "variance"
    variance_threshold: float = 0.01
    missing_threshold: float = 0.5

    def validate(self) -> List[str]:
        """Validate preprocessing configuration."""
        errors = []

        # Check strategy
        valid_strategies = [s.value for s in PreprocessingStrategy]
        if self.strategy not in valid_strategies:
            errors.append(
                f"strategy must be one of {valid_strategies}, got {self.strategy}"
            )

        # Check imputation strategy
        valid_imputation = ["mean", "median", "mode", "drop"]
        if self.imputation_strategy not in valid_imputation:
            errors.append(f"imputation_strategy must be one of {valid_imputation}")

        # Check feature selection
        valid_selection = ["variance", "correlation", "mutual_info", "statistical", "combined", "none"]
        if self.feature_selection_method not in valid_selection:
            errors.append(f"feature_selection_method must be one of {valid_selection}")

        # Check thresholds
        if not 0 <= self.variance_threshold <= 1:
            errors.append("variance_threshold must be between 0 and 1")

        if not 0 <= self.missing_threshold <= 1:
            errors.append("missing_threshold must be between 0 and 1")

        return errors


@dataclass
class CrossValidationConfig:
    """Cross-validation configuration schema."""

    n_folds: int = 5
    n_repeats: int = 1
    stratified: bool = True
    group_aware: bool = False
    random_seed: Optional[int] = None

    def validate(self) -> List[str]:
        """Validate cross-validation configuration."""
        errors = []

        if self.n_folds < 2:
            errors.append("n_folds must be >= 2")
        elif self.n_folds > 20:
            errors.append("n_folds should not exceed 20")

        if self.n_repeats < 1:
            errors.append("n_repeats must be >= 1")
        elif self.n_repeats > 10:
            errors.append("n_repeats should not exceed 10")

        if self.random_seed is not None and self.random_seed < 0:
            errors.append("random_seed must be >= 0")

        return errors


@dataclass
class MonitoringConfig:
    """Monitoring configuration schema."""

    checkpoint_dir: str = "./results/checkpoints"
    log_level: str = "INFO"
    save_checkpoints: bool = True
    checkpoint_interval: int = 100

    def validate(self) -> List[str]:
        """Validate monitoring configuration."""
        errors = []

        # Check log level
        valid_levels = [level.value for level in LogLevel]
        if self.log_level not in valid_levels:
            errors.append(f"log_level must be one of {valid_levels}")

        # Check checkpoint interval
        if self.checkpoint_interval < 1:
            errors.append("checkpoint_interval must be >= 1")

        return errors


@dataclass
class SystemConfig:
    """System configuration schema."""

    use_gpu: bool = True
    memory_limit_gb: Optional[float] = None
    n_cpu_cores: Optional[int] = None

    def validate(self) -> List[str]:
        """Validate system configuration."""
        errors = []

        if self.memory_limit_gb is not None:
            if self.memory_limit_gb <= 0:
                errors.append("memory_limit_gb must be > 0")
            elif self.memory_limit_gb > 1000:
                errors.append("memory_limit_gb seems unreasonably large")

        if self.n_cpu_cores is not None:
            if self.n_cpu_cores < 1:
                errors.append("n_cpu_cores must be >= 1")
            elif self.n_cpu_cores > 128:
                errors.append("n_cpu_cores seems unreasonably large")

        return errors


class ConfigurationValidator:
    """Main configuration validator."""

    @staticmethod
    def create_from_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create configuration objects from dictionary."""
        config_objects = {}

        # Data configuration
        if "data" in config_dict:
            config_objects["data"] = DataConfig(**config_dict["data"])

        # Experiments configuration
        if "experiments" in config_dict:
            config_objects["experiments"] = ExperimentsConfig(
                **config_dict["experiments"]
            )

        # Model configuration
        if "model" in config_dict:
            config_objects["model"] = ModelConfig(**config_dict["model"])

        # Preprocessing configuration
        if "preprocessing" in config_dict:
            config_objects["preprocessing"] = PreprocessingConfig(
                **config_dict["preprocessing"]
            )
        else:
            config_objects["preprocessing"] = PreprocessingConfig()  # Use defaults

        # Cross-validation configuration
        if "cross_validation" in config_dict:
            config_objects["cross_validation"] = CrossValidationConfig(
                **config_dict["cross_validation"]
            )
        else:
            config_objects["cross_validation"] = CrossValidationConfig()  # Use defaults

        # Monitoring configuration
        if "monitoring" in config_dict:
            config_objects["monitoring"] = MonitoringConfig(**config_dict["monitoring"])
        else:
            config_objects["monitoring"] = MonitoringConfig()  # Use defaults

        # System configuration
        if "system" in config_dict:
            config_objects["system"] = SystemConfig(**config_dict["system"])
        else:
            config_objects["system"] = SystemConfig()  # Use defaults

        return config_objects

    @staticmethod
    def validate_configuration(config_dict: Dict[str, Any]) -> List[str]:
        """Validate complete configuration."""
        all_errors = []

        try:
            config_objects = ConfigurationValidator.create_from_dict(config_dict)

            # Validate each configuration section
            for section_name, config_obj in config_objects.items():
                if hasattr(config_obj, "validate"):
                    section_errors = config_obj.validate()
                    if section_errors:
                        for error in section_errors:
                            all_errors.append(f"{section_name}: {error}")

        except (TypeError, ValueError) as e:
            all_errors.append(f"Configuration structure error: {str(e)}")

        # Cross-section validation
        cross_errors = ConfigurationValidator._validate_cross_sections(config_dict)
        all_errors.extend(cross_errors)

        return all_errors

    @staticmethod
    def _validate_cross_sections(config_dict: Dict[str, Any]) -> List[str]:
        """Validate relationships between configuration sections."""
        errors = []

        # Check if sparse GFA has sparsity parameters
        model_config = config_dict.get("model")
        if model_config:
            if model_config.get("model_type") == "sparse_gfa":
                if model_config.get("sparsity_lambda") is None:
                    errors.append("sparse_gfa model requires sparsity_lambda parameter")

        # Check if advanced preprocessing is enabled for complex experiments
        exp_config = config_dict.get("experiments")
        prep_config = config_dict.get("preprocessing")
        if exp_config and prep_config:

            if exp_config.get("generate_plots", True) and not prep_config.get(
                "enable_advanced_preprocessing", True
            ):
                logger.warning(
                    "Consider enabling advanced preprocessing for better plot quality"
                )

        # Check memory requirements vs system limits
        if "model" in config_dict and "system" in config_dict:
            model_config = config_dict["model"]
            system_config = config_dict["system"]

            # Estimate memory requirements
            estimated_memory = ConfigurationValidator._estimate_memory_requirements(
                model_config
            )
            system_memory = system_config.get("memory_limit_gb")

            if system_memory and estimated_memory > system_memory:
                errors.append(
                    f"Estimated memory requirement ({ estimated_memory:.1f}GB) exceeds system limit ({system_memory}GB)"
                )

        return errors

    @staticmethod
    def _estimate_memory_requirements(model_config: Dict[str, Any]) -> float:
        """Estimate memory requirements for model configuration."""
        # Rough estimation based on model parameters
        K = model_config.get("K", 5)
        num_samples = model_config.get("num_samples", 1000)
        num_chains = model_config.get("num_chains", 2)

        # Estimate in GB (very rough approximation)
        estimated_gb = (K * num_samples * num_chains * 8) / (
            1024**3
        )  # 8 bytes per float64
        estimated_gb *= 10  # Factor for temporary arrays and computation overhead

        return max(estimated_gb, 0.5)  # Minimum 0.5GB

    @staticmethod
    def validate_and_fix_configuration(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration and apply fixes where possible."""
        fixed_config = config_dict.copy()

        # Apply fixes
        fixed_config = ConfigurationValidator._apply_fixes(fixed_config)

        # Validate after fixes
        errors = ConfigurationValidator.validate_configuration(fixed_config)

        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            raise ConfigValidationError(error_message)

        logger.info("Configuration validation passed")
        return fixed_config

    @staticmethod
    def _apply_fixes(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply automatic fixes to configuration."""
        fixed_config = config_dict.copy()

        # Ensure required sections exist
        if "data" not in fixed_config:
            fixed_config["data"] = {"data_dir": "./data"}

        if "experiments" not in fixed_config:
            fixed_config["experiments"] = {"base_output_dir": "./results"}

        # Fix model configuration
        if "model" in fixed_config:
            model_config = fixed_config["model"]

            # Ensure sparse GFA has sparsity parameters
            if model_config.get("model_type") == "sparse_gfa":
                if "sparsity_lambda" not in model_config:
                    model_config["sparsity_lambda"] = 0.1
                    logger.info("Added default sparsity_lambda=0.1 for sparse_gfa")

        # Fix paths to be absolute
        if "data" in fixed_config:
            data_dir = fixed_config["data"]["data_dir"]
            if not Path(data_dir).is_absolute():
                fixed_config["data"]["data_dir"] = str(Path(data_dir).resolve())

        if "experiments" in fixed_config:
            output_dir = fixed_config["experiments"]["base_output_dir"]
            if not Path(output_dir).is_absolute():
                fixed_config["experiments"]["base_output_dir"] = str(
                    Path(output_dir).resolve()
                )

        return fixed_config

    @staticmethod
    def get_default_configuration() -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "data": {"data_dir": "./data", "imaging_as_single_view": True},
            "experiments": {
                "base_output_dir": "./results",
                "save_intermediate": True,
                "generate_plots": True,
                "max_parallel_jobs": 1,
            },
            "model": {
                "model_type": "sparse_gfa",
                "K": 5,
                "num_samples": 1000,
                "num_warmup": 500,
                "num_chains": 2,
                "sparsity_lambda": 0.1,
            },
            "preprocessing": {
                "strategy": "standard",
                "enable_advanced_preprocessing": True,
                "imputation_strategy": "median",
            },
            "cross_validation": {"n_folds": 5, "n_repeats": 1, "stratified": True},
            "monitoring": {
                "checkpoint_dir": "./results/checkpoints",
                "log_level": "INFO",
                "save_checkpoints": True,
            },
            "system": {"use_gpu": True},
        }

    @staticmethod
    def merge_with_defaults(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user configuration with defaults."""
        default_config = ConfigurationValidator.get_default_configuration()

        def deep_merge(base: Dict, update: Dict) -> Dict:
            """Deep merge two dictionaries."""
            merged = base.copy()
            for key, value in update.items():
                if (
                    key in merged
                    and isinstance(merged[key], dict)
                    and isinstance(value, dict)
                ):
                    merged[key] = deep_merge(merged[key], value)
                else:
                    merged[key] = value
            return merged

        return deep_merge(default_config, config_dict)
