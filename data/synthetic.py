"""Synthetic data generation module."""

import numpy as np
import pandas as pd
import logging
from scipy.stats import invgamma
from typing import Dict, Any

logger = logging.getLogger(__name__)


def generate_synthetic_data(num_sources: int = 3, K: int = 3, percW: float = 33.0) -> Dict[str, Any]:
    """
    Generate synthetic multi-view data for testing.
    
    Parameters
    ----------
    num_sources : int, default=3
        Number of data sources/views
    K : int, default=3
        Number of latent components
    percW : float, default=33.0
        Percentage of non-zero loadings per component
        
    Returns
    -------
    Dict containing synthetic data in multi-view format compatible with analysis pipeline
    """
    logger.info("Generating synthetic data")
    
    # Fixed parameters for reproducible synthetic data
    N = 150  # number of samples
    Dm = np.array([60, 40, 20])  # features per source (adjust if num_sources != 3)
    
    # Adjust feature dimensions if different number of sources
    if num_sources != 3:
        # Scale feature dimensions proportionally
        total_features = sum(Dm)
        features_per_source = total_features // num_sources
        remainder = total_features % num_sources
        Dm = np.array([features_per_source] * num_sources)
        Dm[:remainder] += 1  # Distribute remainder
    
    D = sum(Dm)
    
    # Generate latent factors Z with regularized horseshoe prior
    Z = np.random.normal(0, 1, (N, K))
    
    # Lambda Z (local shrinkage for latent factors)
    lmbZ0 = 0.001
    lmbZ = 200 * np.ones((N, K))
    lmbZ[50:, 0] = lmbZ0
    lmbZ[0:50, 1] = lmbZ0
    lmbZ[100:150, 1] = lmbZ0
    
    # Tau Z (global shrinkage for latent factors)
    tauZ = 0.01
    for k in range(K):
        Z[:, k] = Z[:, k] * lmbZ[:, k] * tauZ
    
    # Noise variances per source
    sigma = np.array([3, 6, 4])
    if num_sources != 3:
        sigma = np.random.uniform(2, 7, num_sources)
    
    # Generate loadings W with regularized horseshoe prior
    # Lambda W (local shrinkage for loadings)
    pW = np.round((percW/100) * Dm)  # Number of non-zero loadings per component
    lmbW = np.zeros((D, K)) + 0.01
    lmbW0 = 100
    
    d = 0
    for m in range(num_sources):
        for k in range(K):
            # Select random features to be non-zero
            selected_features = np.random.choice(
                np.arange(Dm[m]), int(pW[m]), replace=False
            ) + d
            lmbW[selected_features, k] = lmbW0
        d += Dm[m]
    
    # Tau W (global shrinkage for loadings)
    tauW = np.zeros((1, num_sources))
    for m in range(num_sources):
        scaleW = pW[m] / ((Dm[m] - pW[m]) * np.sqrt(N))
        tauW[0, m] = scaleW * 1/np.sqrt(sigma[m])
    
    # Slab and spike prior
    hypers = {'slab_df': 4.0, 'slab_scale': 2.0}
    cW = np.reshape(
        invgamma.rvs(0.5 * hypers['slab_df'],
                    scale=0.5 * hypers['slab_df'], 
                    size=num_sources*K),
        (num_sources, K)
    )
    cW = hypers['slab_scale'] * np.sqrt(cW)
    
    # Generate loadings
    W = np.random.normal(0, 1, (D, K))
    d = 0
    for m in range(num_sources):
        lmbW_sqr = np.square(lmbW[d:d+Dm[m], :])
        lmbW[d:d+Dm[m], :] = np.sqrt(
            cW[m, :] ** 2 * lmbW_sqr / 
            (cW[m, :] ** 2 + tauW[0, m] ** 2 * lmbW_sqr)
        )
        W[d:d+Dm[m], :] = W[d:d+Dm[m], :] * lmbW[d:d+Dm[m], :] * tauW[0, m]
        d += Dm[m]
    
    # Generate observed data X
    X = np.dot(Z, W.T) + np.random.normal(0, 1, (N, D)) / np.sqrt(np.repeat(sigma, Dm))
    
    # Split into multi-view format
    X_list = []
    d = 0
    for m in range(num_sources):
        X_list.append(X[:, d:d+Dm[m]])
        d += Dm[m]
    
    # Create subject IDs and feature names
    subject_ids = [f"subj_{i:03d}" for i in range(N)]
    feature_names = {}
    view_names = []
    
    for m in range(num_sources):
        view_name = f"view_{m+1}"
        view_names.append(view_name)
        feature_names[view_name] = [f"{view_name}_feat_{j:03d}" for j in range(Dm[m])]
    
    # Create clinical DataFrame (empty for synthetic data)
    clinical = pd.DataFrame(index=subject_ids)
    
    # Create scalers (identity for synthetic data)
    scalers = {}
    for view_name in view_names:
        scalers[view_name] = {
            "mu": np.zeros((1, Dm[view_names.index(view_name)])),
            "sd": np.ones((1, Dm[view_names.index(view_name)]))
        }
    
    return {
        "X_list": X_list,
        "view_names": view_names,
        "feature_names": feature_names,
        "subject_ids": subject_ids,
        "clinical": clinical,
        "scalers": scalers,
        "meta": {
            "dataset": "synthetic",
            "Dm": Dm.tolist(),
            "N": N,
            "K_true": K,
            "preprocessing_enabled": False,
        },
        # Include ground truth for evaluation
        "ground_truth": {
            "Z": Z,
            "W": W,
            "sigma": sigma,
            "lmbZ": lmbZ,
            "lmbW": lmbW,
            "tauZ": tauZ,
            "tauW": tauW,
            "cW": cW,
            "K_true": K,
            "Dm": Dm
        }
    }