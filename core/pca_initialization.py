"""PCA-based initialization for MCMC sampling in GFA models.

Provides smart initialization of latent factors Z and loadings W using PCA,
which can significantly improve MCMC convergence for sparse_gfa_fixed model.
"""

import logging
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def compute_pca_initialization(
    X_list: List[np.ndarray],
    K: int,
    variance_explained: float = 0.95,
) -> Dict[str, np.ndarray]:
    """Compute PCA-based initialization for GFA model parameters.

    Uses PCA to initialize:
    - Z (latent factors): PCA scores from concatenated data
    - W (loadings): PCA loadings for each view

    Parameters
    ----------
    X_list : List[np.ndarray]
        List of data matrices, each (n_samples, n_features_m)
    K : int
        Number of latent factors
    variance_explained : float
        Target variance to explain (default: 0.95)

    Returns
    -------
    Dict with keys:
        'Z': np.ndarray, shape (n_samples, K) - Initial latent factors
        'W': np.ndarray, shape (sum(n_features), K) - Initial loadings (concatenated)
        'W_list': List[np.ndarray] - Initial loadings per view
        'variance_explained': float - Actual variance explained
        'n_components': int - Number of components used
    """
    logger.info("🔧 Computing PCA initialization...")
    n_samples = X_list[0].shape[0]
    n_views = len(X_list)
    logger.info(f"  Input: {n_views} views, N={n_samples} samples, K={K} factors")
    logger.info(f"  View shapes: {[X.shape for X in X_list]}")

    # Concatenate all views
    X_concat = np.concatenate(X_list, axis=1)
    logger.info(f"  Concatenated data shape: {X_concat.shape}")

    logger.info(f"Computing PCA initialization for K={K} factors")
    logger.info(f"  Data shape: {X_concat.shape} ({n_samples} samples, {X_concat.shape[1]} total features)")
    logger.info(f"  Number of views: {n_views}")

    # Fit PCA on concatenated data
    # Use min(K, n_samples-1, n_features) components
    n_components = min(K, n_samples - 1, X_concat.shape[1])

    pca = PCA(n_components=n_components)
    Z_pca = pca.fit_transform(X_concat)  # Shape: (n_samples, n_components)

    # Get loadings (components are stored as rows in sklearn)
    W_pca = pca.components_.T  # Shape: (n_features, n_components)

    # If K > n_components, pad with zeros
    if K > n_components:
        logger.info(f"  Requested K={K} but only {n_components} components available, padding with zeros")
        Z_init = np.zeros((n_samples, K))
        Z_init[:, :n_components] = Z_pca

        W_init = np.zeros((X_concat.shape[1], K))
        W_init[:, :n_components] = W_pca
    else:
        Z_init = Z_pca[:, :K]
        W_init = W_pca[:, :K]

    # Split W back into views
    W_list = []
    start_idx = 0
    for X in X_list:
        n_features = X.shape[1]
        W_view = W_init[start_idx:start_idx + n_features, :]
        W_list.append(W_view)
        start_idx += n_features

    # Compute variance explained
    var_explained = np.sum(pca.explained_variance_ratio_[:min(K, n_components)])

    logger.info(f"  PCA initialization computed:")
    logger.info(f"    Z shape: {Z_init.shape}")
    logger.info(f"    W shape: {W_init.shape}")
    logger.info(f"    Variance explained: {var_explained:.2%}")
    logger.info(f"    Per-component variance: {pca.explained_variance_ratio_[:min(K, n_components)]}")

    return {
        'Z': Z_init,
        'W': W_init,
        'W_list': W_list,
        'variance_explained': var_explained,
        'n_components': n_components,
        'pca_model': pca,
    }


def create_numpyro_init_params(
    X_list: List[np.ndarray],
    K: int,
    model_type: str = "sparse_gfa_fixed",
) -> Dict[str, jnp.ndarray]:
    """Create initialization parameters for NumPyro MCMC.

    Parameters
    ----------
    X_list : List[np.ndarray]
        List of data matrices
    K : int
        Number of latent factors
    model_type : str
        Model type ('sparse_gfa' or 'sparse_gfa_fixed')

    Returns
    -------
    Dict mapping parameter names to JAX arrays for initialization
    """
    logger.info("🔧 Creating NumPyro initialization parameters...")
    logger.info(f"  Model type: {model_type}")
    logger.info(f"  K={K}, n_views={len(X_list)}")

    # Compute PCA initialization
    pca_init = compute_pca_initialization(X_list, K)
    logger.info("  ✓ PCA initialization computed")

    # Convert to JAX arrays
    init_params = {}

    if model_type == "sparse_gfa_fixed":
        logger.info("  Using sparse_gfa_fixed parameterization (non-centered)")
        # For sparse_gfa_fixed (non-centered parameterization)
        # Initialize from PCA to place all chains in same mode (Erosheva & Curtis 2017)

        N = X_list[0].shape[0]
        D_total = pca_init['W'].shape[0]

        # Calculate expected τ₀ for proper initialization
        # For Z: D₀ ≈ K (expected effective dimensionality)
        tau0_Z = (K / (N - K)) * (1.0 / jnp.sqrt(N))

        # CRITICAL: After tau prior fix, model samples tau directly (not tau_tilde)
        # Initialize tauZ near tau0_Z (data-dependent prior scale)
        init_params['tauZ'] = jnp.ones((1, K)) * tau0_Z

        # Initialize Z_raw from PCA scores
        # Since Z = Z_raw * lmbZ_tilde * tauZ, need to rescale
        # Use lmbZ_tilde ≈ 0.5 (conservative) and tauZ ≈ tau0_Z
        # So Z_raw ≈ Z_pca / (tau0_Z * 0.5)
        init_params['Z_raw'] = jnp.array(pca_init['Z'] / (tau0_Z * 0.5))

        # Initialize W_raw from PCA loadings
        # Similar rescaling: W_raw ≈ W_pca / (tau0_W * 0.5)
        # Use conservative tau0_W ≈ 0.05 (typical for imaging data)
        tau0_W_approx = 0.05
        init_params['W_raw'] = jnp.array(pca_init['W'] / (tau0_W_approx * 0.5))

        # Initialize tauW per view near tau0_W (data-dependent prior scale)
        # Model now samples tauW{m+1} directly (not tauW_tilde_{m+1})
        for m in range(len(X_list)):
            init_params[f'tauW{m+1}'] = tau0_W_approx  # Scalar per view

        # Initialize local scales conservatively
        logger.info(f"  Initializing lmbZ: shape ({N}, {K})")
        init_params['lmbZ'] = jnp.ones((N, K)) * 0.5
        logger.info(f"  Initializing lmbW: shape ({D_total}, {K})")
        init_params['lmbW'] = jnp.ones((D_total, K)) * 0.5

        # Initialize slab parameters near expected values
        # c2_tilde ~ IG(2,2), so E[c2_tilde] = 2/(2-1) = 2
        # With slab_scale=2, E[c2] = 4 * 2 = 8, so c2_tilde ≈ 2
        logger.info(f"  Initializing cZ_tilde: shape (1, {K})")
        init_params['cZ_tilde'] = jnp.ones((1, K)) * 2.0
        logger.info(f"  Initializing cW_tilde: shape ({len(X_list)}, {K})")
        init_params['cW_tilde'] = jnp.ones((len(X_list), K)) * 2.0

    else:
        logger.info("  Using sparse_gfa parameterization (centered)")
        # For sparse_gfa (centered parameterization)
        # Can directly initialize Z and W from PCA
        # Note: The model samples Z and W from Normal(0,1) then transforms them
        # So we need to provide the untransformed values

        N = X_list[0].shape[0]
        D_total = pca_init['W'].shape[0]

        # The model samples Z ~ Normal(0,1) then transforms as: Z * lmbZ * tauZ
        # To initialize at PCA values, work backwards:
        # If Z_final = Z_raw * lmbZ * tauZ = Z_pca
        # Then Z_raw = Z_pca / (lmbZ * tauZ)
        # Use lmbZ ≈ 1 and tauZ ≈ 1 for simplicity
        init_params['Z'] = jnp.array(pca_init['Z'])
        init_params['W'] = jnp.array(pca_init['W'])

        # Initialize horseshoe scales conservatively
        init_params['tauZ'] = jnp.ones((1, K))
        init_params['lmbZ'] = jnp.ones((N, K))
        init_params['lmbW'] = jnp.ones((D_total, K))

        # Initialize per-view tauW parameters (tauW1, tauW2, ...)
        for m in range(len(X_list)):
            init_params[f'tauW{m+1}'] = jnp.array(1.0)  # Scalar per view

        # Initialize slab parameters: cZ and cW are sampled from InverseGamma
        # E[IG(a,b)] = b/(a-1) for a > 1
        # With a = b = slab_df/2 (typically slab_df=4, so a=b=2)
        # E[cZ] = 2/(2-1) = 2
        init_params['cZ'] = jnp.ones((1, K)) * 2.0
        init_params['cW'] = jnp.ones((len(X_list), K)) * 2.0

    # Initialize noise parameters (sigma)
    logger.info(f"  Initializing sigma: shape (1, {len(X_list)})")
    init_params['sigma'] = jnp.ones((1, len(X_list)))

    logger.info(f"✓ Created NumPyro init params for {model_type}")
    logger.info(f"  Total parameters initialized: {len(init_params)}")
    logger.info(f"  Parameter names: {list(init_params.keys())}")
    for key, val in init_params.items():
        logger.info(f"    {key}: shape={val.shape}, dtype={val.dtype}")

    return init_params


def should_use_pca_initialization(config: Dict) -> bool:
    """Determine if PCA initialization should be used based on config.

    Parameters
    ----------
    config : Dict
        Configuration dictionary

    Returns
    -------
    bool : True if PCA initialization should be used
    """
    # Check if using sparse_gfa_fixed model
    model_type = config.get("model", {}).get("model_type", "sparseGFA")

    # Check if explicitly enabled in config
    use_pca_init = config.get("model", {}).get("use_pca_initialization", None)

    # Default behavior: use PCA init for sparse_gfa_fixed, don't use for others
    if use_pca_init is None:
        use_pca_init = (model_type == "sparse_gfa_fixed")

    logger.info(f"PCA initialization: {'ENABLED' if use_pca_init else 'DISABLED'} (model_type={model_type})")

    return use_pca_init
