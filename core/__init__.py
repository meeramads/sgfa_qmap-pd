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

# Core legacy modules
from . import get_data
from . import run_analysis  
from . import utils
from . import visualization

__all__ = ['get_data', 'run_analysis', 'utils', 'visualization']