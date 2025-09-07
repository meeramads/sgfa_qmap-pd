"""Models package - Factory for creating GFA model instances."""

from .factory import ModelFactory
from .base import BaseGFAModel
from .sparse_gfa import SparseGFAModel
from .standard_gfa import StandardGFAModel

# Expose the factory method as a standalone function for convenience
def create_model(model_type: str, config, hypers, **kwargs):
    """
    Create a model instance using the ModelFactory.
    
    Parameters:
    -----------
    model_type : str
        Type of model ('sparseGFA', 'GFA', 'neuroGFA')
    config : Configuration object
    hypers : Dict
        Hyperparameters
    **kwargs : Additional model-specific arguments
    
    Returns:
    --------
    BaseGFAModel instance
    """
    return ModelFactory.create_model(model_type, config, hypers, **kwargs)

__all__ = [
    'ModelFactory',
    'create_model', 
    'BaseGFAModel',
    'SparseGFAModel', 
    'StandardGFAModel'
]