# models/sparse_gfa.py
"""Sparse GFA model with regularized horseshoe priors."""

import jax
import jax.numpy as jnp
from jax import lax
import numpyro
import numpyro.distributions as dist
import numpy as np
from typing import List, Dict
from .base import BaseGFAModel


class SparseGFAModel(BaseGFAModel):
    """Sparse GFA with regularized horseshoe priors."""
    
    def __init__(self, config, hypers: Dict):
        super().__init__(config, hypers)
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
        M = self.num_sources     # Number of data sources
        Dm = jnp.array(self.hypers['Dm'])  # Dimensions per source
        D = int(Dm.sum())        # Total dimensions
        K = self.K               # Number of factors
        percW = self.hypers['percW']
        
        # Sample noise parameters (sigma)
        sigma = numpyro.sample(
            "sigma", 
            dist.Gamma(self.hypers['a_sigma'], self.hypers['b_sigma']),
            sample_shape=(1, M)
        )
        
        # Sample latent factors Z with horseshoe prior
        Z = self._sample_latent_factors(N, K)
        
        # Sample loadings W with horseshoe prior
        W = self._sample_loadings(D, K, Dm, percW, sigma, N)
        
        # Generate observations for each data source
        self._generate_observations(X_list, Z, W, Dm, sigma)
        
    def _sample_latent_factors(self, N: int, K: int) -> jnp.ndarray:
        """Sample latent factors Z with optional regularized horseshoe."""
        Z = numpyro.sample("Z", dist.Normal(0, 1), sample_shape=(N, K))
        
        # Horseshoe prior on Z
        tauZ = numpyro.sample("tauZ", dist.TruncatedCauchy(scale=1), sample_shape=(1, K))
        lmbZ = numpyro.sample("lmbZ", dist.TruncatedCauchy(scale=1), sample_shape=(N, K))
        
        if self.reghsZ:
            # Regularized horseshoe
            cZ_tmp = numpyro.sample(
                "cZ", 
                dist.InverseGamma(0.5 * self.hypers['slab_df'], 0.5 * self.hypers['slab_df']),
                sample_shape=(1, K)
            )
            cZ = self.hypers['slab_scale'] * jnp.sqrt(cZ_tmp)
            
            # Apply regularization
            lmbZ_sqr = jnp.square(lmbZ)
            for k in range(K):
                lmbZ_tilde = jnp.sqrt(
                    lmbZ_sqr[:, k] * cZ[0, k]**2 / 
                    (cZ[0, k]**2 + tauZ[0, k]**2 * lmbZ_sqr[:, k])
                )
                Z = Z.at[:, k].set(Z[:, k] * lmbZ_tilde * tauZ[0, k])
        else:
            Z = Z * lmbZ * tauZ
            
        return Z
    
    def _sample_loadings(self, D: int, K: int, Dm: jnp.ndarray, 
                        percW: float, sigma: jnp.ndarray, N: int) -> jnp.ndarray:
        """Sample loading matrix W with sparsity-inducing priors."""
        W = numpyro.sample("W", dist.Normal(0, 1), sample_shape=(D, K))
        
        # Horseshoe prior on W
        lmbW = numpyro.sample("lmbW", dist.TruncatedCauchy(scale=1), sample_shape=(D, K))
        
        # Slab parameters
        cW_tmp = numpyro.sample(
            "cW",
            dist.InverseGamma(0.5 * self.hypers['slab_df'], 0.5 * self.hypers['slab_df']),
            sample_shape=(self.num_sources, K)
        )
        cW = self.hypers['slab_scale'] * jnp.sqrt(cW_tmp)
        
        # Calculate expected sparsity per source
        pW = jnp.round((percW / 100.0) * Dm).astype(int)
        pW = jnp.clip(pW, 1, Dm - 1)
        
        # Apply sparsity to each source
        d = 0
        for m in range(self.num_sources):
            scaleW = pW[m] / ((Dm[m] - pW[m]) * jnp.sqrt(N))
            tauW = numpyro.sample(
                f'tauW{m+1}',
                dist.TruncatedCauchy(scale=scaleW * 1/jnp.sqrt(sigma[0, m]))
            )
            
            # Extract chunk for this source
            width = int(Dm[m])
            lmbW_chunk = lax.dynamic_slice(lmbW, (d, 0), (width, K))
            
            # Apply regularized horseshoe
            lmbW_sqr = jnp.square(lmbW_chunk)
            lmbW_tilde = jnp.sqrt(
                cW[m, :] ** 2 * lmbW_sqr / 
                (cW[m, :] ** 2 + tauW ** 2 * lmbW_sqr)
            )
            
            W_chunk = lax.dynamic_slice(W, (d, 0), (width, K))
            W_chunk = W_chunk * lmbW_tilde * tauW
            W = lax.dynamic_update_slice(W, W_chunk, (d, 0))
            
            d += width
            
        return W
    
    def _generate_observations(self, X_list: List[jnp.ndarray], Z: jnp.ndarray,
                              W: jnp.ndarray, Dm: jnp.ndarray, sigma: jnp.ndarray):
        """Generate observations for each data source."""
        d = 0
        for m in range(self.num_sources):
            X_m = jnp.asarray(X_list[m])
            width = int(Dm[m])
            
            # Extract loadings for this source
            W_chunk = lax.dynamic_slice(W, (d, 0), (width, Z.shape[1]))
            
            # Sample observations
            numpyro.sample(
                f'X{m+1}',
                dist.Normal(jnp.dot(Z, W_chunk.T), 1/jnp.sqrt(sigma[0, m])),
                obs=X_m
            )
            
            d += width
    
    def get_model_name(self) -> str:
        return f"SparseGFA_K{self.K}_percW{self.hypers['percW']}"