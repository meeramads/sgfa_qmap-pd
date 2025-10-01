"""
Visualization package for Sparse GFA Analysis.

This package provides modular visualization capabilities:
- FactorVisualizer: Factor analysis plots
- PreprocessingVisualizer: Data preprocessing summaries
- CrossValidationVisualizer: CV results and subtype analysis
- BrainVisualizer: Neuroimaging-specific visualizations
- ComparisonVisualizer: Experiment comparison and benchmarking plots
- ReportGenerator: Comprehensive analysis reports
- VisualizationManager: Orchestrates all visualization tasks
"""

from .brain_plots import BrainVisualizer
from .comparison_plots import ComparisonVisualizer
from .cv_plots import CrossValidationVisualizer
from .factor_plots import FactorVisualizer
from .manager import VisualizationManager
from .preprocessing_plots import PreprocessingVisualizer
from .report_generator import ReportGenerator

__version__ = "2.1.0"
__all__ = [
    "VisualizationManager",
    "FactorVisualizer",
    "PreprocessingVisualizer",
    "CrossValidationVisualizer",
    "BrainVisualizer",
    "ComparisonVisualizer",
    "ReportGenerator",
]
