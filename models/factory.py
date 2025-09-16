# models/factory.py
"""Factory for creating model instances."""

from typing import Dict, List, Optional
from .base import BaseGFAModel
from .sparse_gfa import SparseGFAModel
from .standard_gfa import StandardGFAModel
from .latent_class_analysis import LatentClassAnalysisModel
from .variants.neuroimaging_gfa import NeuroimagingGFAModel


class ModelFactory:
    """Factory for creating GFA model instances."""
    
    _models = {
        'sparseGFA': SparseGFAModel,
        'GFA': StandardGFAModel,
        'neuroGFA': NeuroimagingGFAModel,
        'LCA': LatentClassAnalysisModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, config, hypers: Dict,
                    **kwargs) -> BaseGFAModel:
        """
        Create a model instance.
        
        Parameters:
        -----------
        model_type : str
            Type of model ('sparseGFA', 'GFA', 'neuroGFA', 'LCA')
        config : Configuration object
        hypers : Dict
            Hyperparameters
        **kwargs : Additional model-specific arguments
        
        Returns:
        --------
        BaseGFAModel instance
        """
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(cls._models.keys())}")
        
        model_class = cls._models[model_type]
        
        # Special handling for neuroimaging model
        if model_type == 'neuroGFA':
            return model_class(config, hypers, spatial_info=kwargs.get('spatial_info'))

        # Special handling for LCA model (memory warning)
        if model_type == 'LCA':
            lca_model = model_class(config, hypers)
            # Log memory warning when creating LCA model
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(lca_model.get_memory_warning())
            return lca_model

        return model_class(config, hypers)
    
    @classmethod
    def register_model(cls, name: str, model_class):
        """Register a new model type."""
        if not issubclass(model_class, BaseGFAModel):
            raise TypeError("Model must inherit from BaseGFAModel")
        cls._models[name] = model_class
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List available model types."""
        return list(cls._models.keys())