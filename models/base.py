# models/base.py
"""Base model interface for all GFA variants."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


class BaseGFAModel(ABC):
    """Abstract base class for GFA models."""

    def __init__(self, config, hypers: Dict):
        self.config = config
        self.hypers = hypers

    @abstractmethod
    def __call__(self, X_list: List[jnp.ndarray], *args, **kwargs):
        """Model definition for NumPyro."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return model name for logging."""
        pass
