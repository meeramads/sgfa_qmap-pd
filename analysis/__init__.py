"""Analysis package for SGFA pipeline."""

from .config_manager import ConfigManager
from .data_manager import DataManager
from .model_runner import ModelRunner
from .component_factory import (
    AnalysisComponents,
    AnalysisMode,
    create_analysis_components,
    integrate_analysis_with_pipeline,
    run_sgfa_with_components,
    prepare_experiment_data,
    quick_sgfa_run,
)

__all__ = [
    "ConfigManager",
    "DataManager",
    "ModelRunner",
    "AnalysisComponents",
    "AnalysisMode",
    "create_analysis_components",
    "integrate_analysis_with_pipeline",
    "run_sgfa_with_components",
    "prepare_experiment_data",
    "quick_sgfa_run",
]
