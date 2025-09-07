"""
Visualization package for Sparse GFA Analysis.

This package provides modular visualization capabilities:
- FactorVisualizer: Factor analysis plots
- PreprocessingVisualizer: Data preprocessing summaries  
- CrossValidationVisualizer: CV results and subtype analysis
- BrainVisualizer: Neuroimaging-specific visualizations
- ReportGenerator: Comprehensive analysis reports
- VisualizationManager: Orchestrates all visualization tasks
"""

from .manager import VisualizationManager
from .factor_plots import FactorVisualizer
from .preprocessing_plots import PreprocessingVisualizer
from .cv_plots import CrossValidationVisualizer
from .brain_plots import BrainVisualizer
from .report_generator import ReportGenerator

__version__ = "2.0.0"
__all__ = [
    'VisualizationManager',
    'FactorVisualizer', 
    'PreprocessingVisualizer',
    'CrossValidationVisualizer',
    'BrainVisualizer',
    'ReportGenerator'
]