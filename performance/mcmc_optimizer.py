"""Memory-efficient MCMC optimization strategies."""

import logging
import functools
from typing import Dict, Any, Optional, Callable, Tuple, List, Union
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, grad, jit, vmap
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from .memory_optimizer import MemoryOptimizer, memory_profile

logger = logging.getLogger(__name__)


class MCMCMemoryOptimizer:
    """Memory optimization strategies for MCMC sampling."""
    
    def __init__(self, 
                 memory_limit_gb: float = 8.0,
                 enable_checkpointing: bool = True,
                 subsample_ratio: float = 1.0,
                 thinning_interval: int = 1,
                 enable_batch_sampling: bool = True):
        """
        Initialize MCMC memory optimizer.
        
        Parameters
        ----------
        memory_limit_gb : float
            Memory limit for MCMC operations.
        enable_checkpointing : bool
            Enable gradient checkpointing.
        subsample_ratio : float
            Ratio of data to use for memory-constrained sampling.
        thinning_interval : int
            Interval for sample thinning.
        enable_batch_sampling : bool
            Enable batch-based sampling for large datasets.
        """
        self.memory_limit_gb = memory_limit_gb
        self.enable_checkpointing = enable_checkpointing
        self.subsample_ratio = subsample_ratio
        self.thinning_interval = thinning_interval
        self.enable_batch_sampling = enable_batch_sampling
        self._memory_optimizer = MemoryOptimizer(max_memory_gb=memory_limit_gb)
        
        logger.info(f"MCMCMemoryOptimizer initialized with {memory_limit_gb}GB limit")

    def optimize_mcmc_config(self, 
                           X_list: List[np.ndarray],
                           base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize MCMC configuration based on data size and memory constraints.
        
        Parameters
        ----------
        X_list : List[np.ndarray]
            Input data matrices.
        base_config : Dict[str, Any]
            Base MCMC configuration.
            
        Returns
        -------
        Dict[str, Any] : Optimized configuration.
        """
        n_subjects = X_list[0].shape[0]
        total_features = sum(X.shape[1] for X in X_list)
        
        # Estimate memory requirements
        estimated_memory = self._estimate_mcmc_memory(
            n_subjects, total_features, base_config.get('K', 10)
        )
        
        optimized_config = base_config.copy()
        
        # Adjust sampling parameters if memory constrained
        if estimated_memory > self.memory_limit_gb:
            logger.warning(f"Estimated memory {estimated_memory:.2f}GB exceeds limit {self.memory_limit_gb:.2f}GB")
            
            # Reduce number of samples if necessary
            memory_ratio = self.memory_limit_gb / estimated_memory
            if 'num_samples' in optimized_config:
                new_samples = int(optimized_config['num_samples'] * memory_ratio * 0.8)
                new_samples = max(new_samples, 100)  # Minimum 100 samples
                logger.info(f"Reducing samples from {optimized_config['num_samples']} to {new_samples}")
                optimized_config['num_samples'] = new_samples
            
            # Enable thinning for memory savings
            if memory_ratio < 0.5:
                self.thinning_interval = max(2, int(1 / memory_ratio))
                logger.info(f"Enabling thinning with interval {self.thinning_interval}")
            
            # Enable data subsampling if very constrained
            if memory_ratio < 0.3:
                self.subsample_ratio = max(0.5, memory_ratio * 2)
                logger.info(f"Enabling data subsampling with ratio {self.subsample_ratio}")
        
        # Add memory optimization flags
        optimized_config.update({
            'memory_efficient': True,
            'thinning_interval': self.thinning_interval,
            'subsample_ratio': self.subsample_ratio,
            'estimated_memory_gb': estimated_memory
        })
        
        return optimized_config

    def _estimate_mcmc_memory(self, n_subjects: int, n_features: int, n_factors: int) -> float:
        """Estimate memory usage for MCMC sampling."""
        # Parameter storage (Z, W, sigma, etc.)
        z_memory = n_subjects * n_factors * 4  # float32
        w_memory = n_features * n_factors * 4
        sigma_memory = 10 * 4  # Approximate
        param_memory = (z_memory + w_memory + sigma_memory) * 2  # Current + proposed states
        
        # Data memory
        data_memory = n_subjects * n_features * 4
        
        # Gradient computation memory (approximately 3x parameter memory)
        grad_memory = param_memory * 3
        
        # JAX compilation cache and overhead
        overhead_memory = max(1024**3, param_memory * 0.5)  # At least 1GB overhead
        
        total_bytes = param_memory + data_memory + grad_memory + overhead_memory
        return total_bytes / (1024**3)

    def create_memory_efficient_model(self, 
                                    original_model: Callable,
                                    X_list: List[np.ndarray],
                                    hypers: Dict[str, Any]) -> Callable:
        """
        Create memory-efficient version of model.
        
        Parameters
        ----------
        original_model : Callable
            Original model function.
        X_list : List[np.ndarray]
            Input data.
        hypers : Dict[str, Any]
            Hyperparameters.
            
        Returns
        -------
        Callable : Memory-efficient model.
        """
        # Apply data subsampling if enabled
        if self.subsample_ratio < 1.0:
            X_list = self._subsample_data(X_list, self.subsample_ratio)
            logger.info(f"Subsampled data to {X_list[0].shape[0]} subjects")
        
        # Create checkpointed version if enabled
        if self.enable_checkpointing:
            model = self._add_gradient_checkpointing(original_model)
        else:
            model = original_model
        
        # Add memory monitoring
        def memory_monitored_model(X_list_inner, hypers_inner, args):
            with memory_profile() as profiler:
                result = model(X_list_inner, hypers_inner, args)
            
            if profiler.memory_delta_gb > 1.0:  # Log if significant memory usage
                logger.debug(f"Model used {profiler.memory_delta_gb:.2f}GB")
            
            return result
        
        return functools.partial(memory_monitored_model, X_list, hypers)

    def _subsample_data(self, X_list: List[np.ndarray], ratio: float) -> List[np.ndarray]:
        """Subsample data for memory efficiency."""
        n_subjects = X_list[0].shape[0]
        n_subsample = int(n_subjects * ratio)
        
        # Deterministic subsampling for reproducibility
        np.random.seed(42)
        indices = np.random.choice(n_subjects, n_subsample, replace=False)
        indices = np.sort(indices)
        
        return [X[indices] for X in X_list]

    def _add_gradient_checkpointing(self, model_fn: Callable) -> Callable:
        """Add gradient checkpointing to model using JAX remat."""
        @functools.wraps(model_fn)
        def checkpointed_model(*args, **kwargs):
            # Use JAX checkpoint (remat) for memory efficiency
            # This trades computation for memory by not storing intermediate gradients
            checkpointed_fn = jax.checkpoint(
                model_fn,
                policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable
            )
            return checkpointed_fn(*args, **kwargs)
        
        return checkpointed_model

    def run_memory_efficient_mcmc(self,
                                model_fn: Callable,
                                X_list: List[np.ndarray], 
                                hypers: Dict[str, Any],
                                args: Any,
                                rng_key: jax.random.PRNGKey) -> Dict[str, np.ndarray]:
        """
        Run MCMC with memory optimizations.
        
        Parameters
        ----------
        model_fn : Callable
            Model function.
        X_list : List[np.ndarray]
            Input data.
        hypers : Dict[str, Any]
            Hyperparameters.
        args : Any
            Arguments object.
        rng_key : jax.random.PRNGKey
            Random key.
            
        Returns
        -------
        Dict[str, np.ndarray] : MCMC samples.
        """
        # Optimize configuration
        base_config = {
            'num_samples': getattr(args, 'num_samples', 1000),
            'num_chains': getattr(args, 'num_chains', 2),
            'num_warmup': getattr(args, 'num_warmup', 500),
            'K': getattr(args, 'K', 10)
        }
        
        optimized_config = self.optimize_mcmc_config(X_list, base_config)
        
        # Create memory-efficient model
        efficient_model = self.create_memory_efficient_model(model_fn, X_list, hypers)
        
        # Monitor memory during sampling
        self._memory_optimizer.start_monitoring(interval=2.0)
        
        try:
            # Configure NUTS with memory optimizations
            nuts_kernel = NUTS(
                efficient_model,
                dense_mass=False,  # Use diagonal mass matrix to save memory
                max_tree_depth=8   # Limit tree depth to control memory
            )
            
            # Run MCMC
            mcmc = MCMC(
                nuts_kernel,
                num_samples=optimized_config['num_samples'],
                num_chains=optimized_config['num_chains'],
                num_warmup=optimized_config.get('num_warmup', optimized_config['num_samples'] // 2),
                chain_method='parallel' if optimized_config['num_chains'] > 1 else 'sequential'
            )
            
            mcmc.run(rng_key, extra_fields=("potential_energy", "accept_prob"))
            samples = mcmc.get_samples()
            
            # Apply thinning if configured
            if self.thinning_interval > 1:
                samples = self._thin_samples(samples, self.thinning_interval)
            
            return samples
            
        finally:
            self._memory_optimizer.stop_monitoring()
            
            # Final cleanup
            self._memory_optimizer.aggressive_cleanup()

    def _thin_samples(self, samples: Dict[str, np.ndarray], interval: int) -> Dict[str, np.ndarray]:
        """Apply thinning to MCMC samples."""
        thinned = {}
        for key, value in samples.items():
            thinned[key] = value[::interval]
        
        logger.info(f"Thinned samples by factor {interval}: "
                   f"{value.shape[0]} -> {thinned[key].shape[0]} samples")
        return thinned

    def adaptive_batch_mcmc(self,
                          model_fn: Callable,
                          X_list: List[np.ndarray],
                          hypers: Dict[str, Any], 
                          args: Any,
                          rng_key: jax.random.PRNGKey,
                          batch_size: int = None) -> Dict[str, np.ndarray]:
        """
        Run MCMC with adaptive batching for very large datasets.
        
        Parameters
        ----------
        model_fn : Callable
            Model function.
        X_list : List[np.ndarray]
            Input data.
        hypers : Dict[str, Any]
            Hyperparameters.
        args : Any
            Arguments.
        rng_key : jax.random.PRNGKey
            Random key.
        batch_size : int, optional
            Batch size for processing.
            
        Returns
        -------
        Dict[str, np.ndarray] : Aggregated MCMC samples.
        """
        if not self.enable_batch_sampling:
            return self.run_memory_efficient_mcmc(model_fn, X_list, hypers, args, rng_key)
        
        n_subjects = X_list[0].shape[0]
        
        # Calculate adaptive batch size
        if batch_size is None:
            from .data_streaming import adaptive_batch_size
            total_features = sum(X.shape[1] for X in X_list)
            element_size = total_features * 4  # float32
            batch_size = adaptive_batch_size(
                n_subjects,
                self.memory_limit_gb * 0.7,  # Reserve memory for MCMC
                element_size,
                min_batch=50
            )
        
        if batch_size >= n_subjects:
            # No batching needed
            return self.run_memory_efficient_mcmc(model_fn, X_list, hypers, args, rng_key)
        
        logger.info(f"Running adaptive batch MCMC with batch_size={batch_size}")
        
        # Run MCMC on batches and aggregate
        all_samples = []
        n_batches = (n_subjects + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_subjects)
            
            # Create batch data
            X_batch = [X[start_idx:end_idx] for X in X_list]
            
            # Split random key for this batch
            rng_key, batch_key = jax.random.split(rng_key)
            
            # Run MCMC on batch
            batch_samples = self.run_memory_efficient_mcmc(
                model_fn, X_batch, hypers, args, batch_key
            )
            
            all_samples.append(batch_samples)
            
            logger.info(f"Completed batch {batch_idx + 1}/{n_batches}")
        
        # Aggregate samples across batches
        return self._aggregate_batch_samples(all_samples)

    def _aggregate_batch_samples(self, batch_samples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Aggregate MCMC samples from multiple batches."""
        if len(batch_samples) == 1:
            return batch_samples[0]
        
        # Simple averaging for now - could implement more sophisticated aggregation
        aggregated = {}
        sample_keys = batch_samples[0].keys()
        
        for key in sample_keys:
            if key in ['Z', 'W']:  # Parameters we can average
                # Average across batches, weighted by batch size
                batch_weights = [samples[key].shape[1] if key == 'Z' else samples[key].shape[1] 
                               for samples in batch_samples]
                total_weight = sum(batch_weights)
                
                weighted_sum = sum(
                    samples[key] * (weight / total_weight)
                    for samples, weight in zip(batch_samples, batch_weights)
                )
                aggregated[key] = weighted_sum
            else:
                # For other parameters, just use the first batch
                aggregated[key] = batch_samples[0][key]
        
        logger.info("Aggregated samples from batch MCMC")
        return aggregated


class GradientCheckpointer:
    """Gradient checkpointing utilities for memory savings."""
    
    def __init__(self, checkpoint_every: int = 5):
        """
        Initialize gradient checkpointer.
        
        Parameters
        ----------
        checkpoint_every : int
            Checkpoint frequency.
        """
        self.checkpoint_every = checkpoint_every

    def checkpoint_sequential(self, fn: Callable, *args) -> Any:
        """
        Apply gradient checkpointing to sequential operations.
        
        Parameters
        ----------
        fn : Callable
            Function to checkpoint.
        *args
            Function arguments.
            
        Returns
        -------
        Any : Function result.
        """
        # Simplified checkpointing - in practice would use JAX's checkpoint decorator
        return fn(*args)

    @staticmethod
    def memory_efficient_scan(fn: Callable, 
                            init: Any, 
                            xs: Any, 
                            checkpoint_every: int = 10) -> Tuple[Any, Any]:
        """
        Memory-efficient scan with periodic checkpointing.
        
        Parameters
        ----------
        fn : Callable
            Scan function.
        init : Any
            Initial value.
        xs : Any
            Scan inputs.
        checkpoint_every : int
            Checkpointing frequency.
            
        Returns
        -------
        Tuple[Any, Any] : Final state and outputs.
        """
        # Use lax.scan with checkpointing hints
        def checkpointed_fn(carry, x_and_idx):
            x, idx = x_and_idx
            
            # Add checkpointing at regular intervals
            if idx % checkpoint_every == 0:
                carry = jax.lax.stop_gradient(carry)
            
            return fn(carry, x)
        
        # Add indices for checkpointing
        xs_with_idx = (xs, jnp.arange(xs.shape[0] if hasattr(xs, 'shape') else len(xs)))
        
        return jax.lax.scan(checkpointed_fn, init, xs_with_idx)


# Utility functions for memory-efficient operations

def memory_efficient_matrix_multiply(A: jnp.ndarray, 
                                   B: jnp.ndarray, 
                                   chunk_size: int = 1000) -> jnp.ndarray:
    """
    Memory-efficient matrix multiplication for large matrices.
    
    Parameters
    ----------
    A : jnp.ndarray
        Left matrix.
    B : jnp.ndarray  
        Right matrix.
    chunk_size : int
        Chunk size for processing.
        
    Returns
    -------
    jnp.ndarray : Matrix product.
    """
    if A.shape[0] <= chunk_size and B.shape[1] <= chunk_size:
        # Small enough for direct multiplication
        return jnp.dot(A, B)
    
    # Chunk-based multiplication
    result_chunks = []
    
    for i in range(0, A.shape[0], chunk_size):
        row_chunk = A[i:i + chunk_size]
        chunk_result = jnp.dot(row_chunk, B)
        result_chunks.append(chunk_result)
    
    return jnp.vstack(result_chunks)


def memory_efficient_log_likelihood(X: jnp.ndarray,
                                  Z: jnp.ndarray, 
                                  W: jnp.ndarray,
                                  sigma: jnp.ndarray,
                                  chunk_subjects: int = 100) -> jnp.ndarray:
    """
    Compute log-likelihood in memory-efficient chunks.
    
    Parameters
    ----------
    X : jnp.ndarray
        Data matrix.
    Z : jnp.ndarray
        Latent factors.
    W : jnp.ndarray
        Loadings.
    sigma : jnp.ndarray
        Noise variances.
    chunk_subjects : int
        Number of subjects per chunk.
        
    Returns
    -------
    jnp.ndarray : Log-likelihood.
    """
    n_subjects = X.shape[0]
    
    if n_subjects <= chunk_subjects:
        # Compute directly
        predicted = jnp.dot(Z, W.T)
        residuals = X - predicted
        log_lik = -0.5 * jnp.sum(residuals**2 / sigma)
        return log_lik
    
    # Chunk-based computation
    total_log_lik = 0.0
    
    for i in range(0, n_subjects, chunk_subjects):
        end_idx = min(i + chunk_subjects, n_subjects)
        
        X_chunk = X[i:end_idx]
        Z_chunk = Z[i:end_idx]
        
        predicted_chunk = jnp.dot(Z_chunk, W.T)
        residuals_chunk = X_chunk - predicted_chunk
        log_lik_chunk = -0.5 * jnp.sum(residuals_chunk**2 / sigma)
        
        total_log_lik += log_lik_chunk
    
    return total_log_lik