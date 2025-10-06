# models/sparse_gfa.py
"""Sparse GFA model with regularized horseshoe priors.

✅ PRIMARY MODEL: This is the main model actively used in the project.
Sparse GFA provides memory-efficient group factor analysis with automatic
relevance determination and sparsity-inducing priors, making it suitable
for high-dimensional neuroimaging data with limited computational resources.
"""

from typing import Dict, List

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import lax

from .base import BaseGFAModel


class SparseGFAModel(BaseGFAModel):
    """Sparse GFA with regularized horseshoe priors.

    This is the primary model used in the project for multi-view factor analysis
    of neuroimaging and clinical data.
    """

    def __init__(self, config, hypers: Dict):
        super().__init__(config, hypers)

        # Validate required config attributes
        required_attrs = ['K', 'num_sources', 'reghsZ']
        for attr in required_attrs:
            if not hasattr(config, attr):
                raise AttributeError(f"Config missing required attribute: '{attr}'")

        self.K = config.K
        self.num_sources = config.num_sources
        self.reghsZ = config.reghsZ

    def __call__(self, X_list: List[jnp.ndarray]):
        """
        Sparse GFA model implementation.

        Parameters:
        -----------
        X_list : List of data matrices for each view
        """
        N = X_list[0].shape[0]  # Number of subjects
        M = self.num_sources  # Number of data sources
        # Use static Python ints for dimensions to avoid JAX concretization errors
        Dm_static = self.hypers["Dm"]  # Keep as Python list
        Dm = jnp.array(Dm_static)  # JAX array for computation
        D = sum(Dm_static)  # Static Python int
        K = self.K  # Number of factors
        percW = self.hypers["percW"]

        # Sample noise parameters (sigma)
        sigma = numpyro.sample(
            "sigma",
            dist.Gamma(self.hypers["a_sigma"], self.hypers["b_sigma"]),
            sample_shape=(1, M),
        )

        # Sample latent factors Z with horseshoe prior
        Z = self._sample_latent_factors(N, K)

        # Sample loadings W with horseshoe prior
        W = self._sample_loadings(D, K, Dm, Dm_static, percW, sigma, N)

        # Generate observations for each data source
        self._generate_observations(X_list, Z, W, Dm, Dm_static, sigma)

    def _sample_latent_factors(self, N: int, K: int) -> jnp.ndarray:
        """Sample latent factors Z with optional regularized horseshoe."""
        Z = numpyro.sample("Z", dist.Normal(0, 1), sample_shape=(N, K))

        # Horseshoe prior on Z
        tauZ = numpyro.sample(
            "tauZ", dist.TruncatedCauchy(scale=1), sample_shape=(1, K)
        )
        lmbZ = numpyro.sample(
            "lmbZ", dist.TruncatedCauchy(scale=1), sample_shape=(N, K)
        )

        if self.reghsZ:
            # Regularized horseshoe
            cZ_tmp = numpyro.sample(
                "cZ",
                dist.InverseGamma(
                    0.5 * self.hypers["slab_df"], 0.5 * self.hypers["slab_df"]
                ),
                sample_shape=(1, K),
            )
            cZ = self.hypers["slab_scale"] * jnp.sqrt(cZ_tmp)

            # Apply regularization
            lmbZ_sqr = jnp.square(lmbZ)
            for k in range(K):
                lmbZ_tilde = jnp.sqrt(
                    lmbZ_sqr[:, k]
                    * cZ[0, k] ** 2
                    / (cZ[0, k] ** 2 + tauZ[0, k] ** 2 * lmbZ_sqr[:, k])
                )
                Z = Z.at[:, k].set(Z[:, k] * lmbZ_tilde * tauZ[0, k])
        else:
            Z = Z * lmbZ * tauZ

        return Z

    def _sample_loadings(
        self, D: int, K: int, Dm: jnp.ndarray, Dm_static: list, percW: float, sigma: jnp.ndarray, N: int
    ) -> jnp.ndarray:
        """Sample loading matrix W with sparsity-inducing priors."""
        W = numpyro.sample("W", dist.Normal(0, 1), sample_shape=(D, K))

        # Horseshoe prior on W
        lmbW = numpyro.sample(
            "lmbW", dist.TruncatedCauchy(scale=1), sample_shape=(D, K)
        )

        # Slab parameters
        cW_tmp = numpyro.sample(
            "cW",
            dist.InverseGamma(
                0.5 * self.hypers["slab_df"], 0.5 * self.hypers["slab_df"]
            ),
            sample_shape=(self.num_sources, K),
        )
        cW = self.hypers["slab_scale"] * jnp.sqrt(cW_tmp)

        # Calculate expected sparsity per source using static dimensions
        pW_static = [max(1, min(int((percW / 100.0) * dim), dim - 1)) for dim in Dm_static]

        # Apply sparsity to each source
        d = 0
        for m in range(self.num_sources):
            pW_m = pW_static[m]
            Dm_m = Dm_static[m]
            scaleW = pW_m / ((Dm_m - pW_m) * jnp.sqrt(N))
            tauW = numpyro.sample(
                f"tauW{m + 1}",
                dist.TruncatedCauchy(scale=scaleW * 1 / jnp.sqrt(sigma[0, m])),
            )

            # Extract chunk for this source using static width
            width = Dm_static[m]
            lmbW_chunk = lax.dynamic_slice(lmbW, (d, 0), (width, K))

            # Apply regularized horseshoe
            lmbW_sqr = jnp.square(lmbW_chunk)
            lmbW_tilde = jnp.sqrt(
                cW[m, :] ** 2 * lmbW_sqr / (cW[m, :] ** 2 + tauW**2 * lmbW_sqr)
            )

            W_chunk = lax.dynamic_slice(W, (d, 0), (width, K))
            W_chunk = W_chunk * lmbW_tilde * tauW
            W = lax.dynamic_update_slice(W, W_chunk, (d, 0))

            d += width

        return W

    def _generate_observations(
        self,
        X_list: List[jnp.ndarray],
        Z: jnp.ndarray,
        W: jnp.ndarray,
        Dm: jnp.ndarray,
        Dm_static: list,
        sigma: jnp.ndarray,
    ):
        """Generate observations for each data source."""
        d = 0
        for m in range(self.num_sources):
            X_m = jnp.asarray(X_list[m])
            width = Dm_static[m]

            # Extract loadings for this source using static width
            W_chunk = lax.dynamic_slice(W, (d, 0), (width, Z.shape[1]))

            # Sample observations
            numpyro.sample(
                f"X{m + 1}",
                dist.Normal(jnp.dot(Z, W_chunk.T), 1 / jnp.sqrt(sigma[0, m])),
                obs=X_m,
            )

            d += width

    def get_model_name(self) -> str:
        return f"SparseGFA_K{self.K}_percW{self.hypers['percW']}"
