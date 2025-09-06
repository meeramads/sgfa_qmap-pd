"""
High-level data access interface.
Combines loading and preprocessing based on dataset type.
"""

from typing import Dict, Any 
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from scipy.stats import invgamma
import logging

# Import the updated loader
from loader_qmap_pd import load_qmap_pd as qmap_pd

logging.basicConfig(level=logging.INFO)

def synthetic_data(hypers, args):
    """Generate synthetic data for testing"""
    logging.info("Generating synthetic data")

    M = args.num_sources #number of data sources
    N = 150 #number of samples/examples 
    Dm = np.array([60, 40, 20]) #number of features/variables in each data source
    D = sum(Dm)
    K_true = 3 #number of latent components to generate the data

    # Implement regularized horseshoe prior over Z
    Z = np.reshape(np.random.normal(0, 1, N * K_true), (N,K_true))
    
    #lambda Z
    lmbZ0 = 0.001
    lmbZ = 200 * np.ones((N,K_true))
    lmbZ[50:,0] = lmbZ0
    lmbZ[0:50,1] = lmbZ0; lmbZ[100:150,1] = lmbZ0
    
    #tau Z
    tauZ = 0.01
    for k in range(K_true):
        Z[:,k] = Z[:,k] * lmbZ[:,k] * tauZ
    
    #sigmas
    sigma = np.array([3, 6, 4])
    logging.debug(f"Z shape: {Z.shape}, sigma: {sigma}")
    
    # Implement regularized horseshoe prior over W
    #lambda W
    percW = 33 * np.ones((1,K_true))
    pW = np.round((percW/100) * Dm)
    lmbW = np.zeros((D,K_true)) * 0.01
    lmbW0 = 100; d = 0
    for m in range(M): 
        for k in range(K_true):
            lmbW[np.random.choice(
                    np.arange(Dm[m]), int(pW[0,m]), replace=False) + d, k] = lmbW0
        d += Dm[m]
    #tau W        
    tauW = np.zeros((1,M))
    for m in range(M):
        scaleW = pW[0,m] / ((Dm[m] - pW[0,m]) * np.sqrt(N))
        tauW[0,m] =  scaleW * 1/np.sqrt(sigma[m])   
    #c W   
    cW = np.reshape(invgamma.rvs(0.5 * hypers['slab_df'],
        scale=0.5 * hypers['slab_df'], size=M*K_true),(M,K_true))
    cW = hypers['slab_scale'] * np.sqrt(cW)    
    W = np.random.normal(0, 1, (D,K_true))
    X = np.zeros((N,D)); d = 0
    for m in range(M): 
        lmbW_sqr = np.reshape(np.square(lmbW[d:d+Dm[m],:]), (Dm[m],K_true))
        lmbW[d:d+Dm[m],:] = np.sqrt(cW[m,:] ** 2 * lmbW_sqr / 
                (cW[m,:] ** 2 + tauW[0,m] ** 2 * lmbW_sqr))
        W[d:d+Dm[m],:] = W[d:d+Dm[m],:] * lmbW[d:d+Dm[m],:] * tauW[0,m]
        
        # Generate X^(m)
        X[:,d:d+Dm[m]] = np.dot(Z,W[d:d+Dm[m],:].T) + \
            np.reshape(np.random.normal(0, 1/np.sqrt(sigma[m]), N*Dm[m]),(N,Dm[m])) 
        d += Dm[m]

    #Save parameters
    data = {'X': X, 'Z': Z, 'tauZ': tauZ, 'lmbZ': lmbZ, 'sigma': sigma,
        'W': W, 'tauW': tauW, 'lmbW': lmbW, 'cW': cW, 'K_true': K_true, 'Dm': Dm}       

    return data

def get_data(dataset: str, data_dir: str = None, **kwargs):
    """
    High-level interface for getting data with optional preprocessing.
    
    Parameters
    ----------
    dataset : str
        Dataset name ('qmap_pd', 'synthetic')
    data_dir : str  
        Data directory path
    **kwargs : dict
        Additional arguments for data loading and preprocessing
    
    Returns
    -------
    dict : Data ready for modeling (preprocessed if requested)
    """
    
    ds = dataset.lower()
    
    if ds in {"qmap_pd", "qmap-pd", "qmap"}:
        # Load qMAP-PD data with all preprocessing options
        if data_dir is None:
            raise ValueError("data_dir must be provided for qMAP-PD dataset")
        
        return qmap_pd(data_dir, **kwargs)
        
    elif ds in {"synthetic", "toy"}:
        # Generate synthetic data
        syn = synthetic_data(
            hypers={"slab_df": 1.0, "slab_scale": 1.0}, 
            args=type("Args", (), {"num_sources": kwargs.get("num_sources", 3)})()
        )
        
        # Convert to format compatible with multiview processing
        return {
            "X_list": [syn["X"]],
            "view_names": ["synthetic"],
            "feature_names": {"synthetic": [f"f{i}" for i in range(syn["X"].shape[1])]},
            "subject_ids": [f"s{i}" for i in range(syn["X"].shape[0])],
            "clinical": pd.DataFrame(index=[f"s{i}" for i in range(syn["X"].shape[0])]),
            "scalers": {"synthetic": {"mu": np.zeros((1, syn["X"].shape[1])), "sd": np.ones((1, syn["X"].shape[1]))}},
            "meta": {
                "dataset": "synthetic",
                "Dm": syn["Dm"],
                "N": syn["X"].shape[0],
                "preprocessing_enabled": False,
            },
            # Include ground truth for synthetic data
            "ground_truth": {
                "Z": syn["Z"],
                "W": syn["W"], 
                "sigma": syn["sigma"],
                "K_true": syn["K_true"]
            }
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: 'qmap_pd', 'synthetic'")