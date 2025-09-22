"""
Core functionality for SGFA qMAP-PD analysis.

This package contains the essential components that form the foundation
of the entire analysis pipeline:

- get_data.py: Data acquisition and initial loading functions
- run_analysis.py: Main analysis pipeline orchestration and SGFA execution
- utils.py: Core utility functions used throughout the framework
- visualization.py: Essential plotting and visualization functions

These modules are actively used by the experimental framework, remote
workstation scripts, and all analysis components.
"""

# Core legacy modules - avoid circular imports
from . import get_data, utils

# Conditionally import to avoid circular dependencies
def _get_visualization():
    from . import visualization
    return visualization

def _get_run_analysis():
    from . import run_analysis
    return run_analysis

# Lazy loading to avoid circular imports
__all__ = ["get_data", "utils"]
