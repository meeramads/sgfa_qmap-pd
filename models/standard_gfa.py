# models/standard_gfa.py
"""Standard GFA model with ARD prior.

⚠️  FUTURE WORK: Standard GFA is implemented but not actively used in this project
due to memory constraints with high-dimensional neuroimaging data. It is computationally
more intensive than Sparse GFA. Available for future use when larger memory resources
become available.
"""

from typing import Dict, List

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import lax

from .base import BaseGFAModel


class StandardGFAModel(BaseGFAModel):
    """Standard GFA with Automatic Relevance Determination (ARD) prior.

    ⚠️  Note: Not currently used due to memory constraints. Implementation
    available for future work with larger computational resources.
    """

    def __init__(self, config, hypers: Dict):
        super().__init__(config, hypers)

        # Validate required config attributes
        required_attrs = ['K', 'num_sources']
        for attr in required_attrs:
            if not hasattr(config, attr):
                raise AttributeError(f"Config missing required attribute: '{attr}'")

        self.K = config.K
        self.num_sources = config.num_sources

    def __call__(self, X_list: List[jnp.ndarray]):
        """
        Standard GFA model implementation with ARD prior.
        """
        N = X_list[0].shape[0]
        M = self.num_sources
        # Use static Python ints for dimensions to avoid JAX concretization errors
        Dm_static = self.hypers["Dm"]  # Keep as Python list
        Dm = jnp.array(Dm_static)  # JAX array for computation
        D = sum(Dm_static)  # Static Python int
        K = self.K

        # Sample noise parameters
        sigma = numpyro.sample(
            "sigma",
            dist.Gamma(self.hypers["a_sigma"], self.hypers["b_sigma"]),
            sample_shape=(1, M),
        )

        # Sample latent factors (standard normal)
        Z = numpyro.sample("Z", dist.Normal(0, 1), sample_shape=(N, K))

        # Sample loadings with ARD prior
        W = numpyro.sample("W", dist.Normal(0, 1), sample_shape=(D, K))

        # ARD precision parameters
        alpha = numpyro.sample("alpha", dist.Gamma(1e-3, 1e-3), sample_shape=(M, K))

        # Apply ARD to loadings for each source
        d = 0
        for m in range(M):
            width = Dm_static[m]  # Use static dimension

            # Extract and scale chunk
            W_chunk = lax.dynamic_slice(W, (d, 0), (width, K))
            W_chunk = W_chunk * (1 / jnp.sqrt(alpha[m, :]))
            W = lax.dynamic_update_slice(W, W_chunk, (d, 0))

            # Generate observations
            W_chunk = lax.dynamic_slice(W, (d, 0), (width, K))
            X_m = jnp.asarray(X_list[m])

            numpyro.sample(
                f"X{m + 1}",
                dist.Normal(jnp.dot(Z, W_chunk.T), 1 / jnp.sqrt(sigma[0, m])),
                obs=X_m,
            )

            d += width

    def get_model_name(self) -> str:
        return f"GFA_K{self.K}"
