"""Test configuration and fixtures."""

import shutil
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config():
    """Create a sample configuration object."""
    config = Mock()
    config.dataset = "synthetic"
    config.num_sources = 3
    config.K = 5
    config.percW = 33.0
    config.num_samples = 1000
    config.num_chains = 2
    config.cv_folds = 5
    config.seed = 42
    config.device = "cpu"
    config.neuroimaging_cv = False
    config.nested_cv = False
    config.cv_only = False
    config.run_cv = False
    return config


@pytest.fixture
def sample_synthetic_data():
    """Generate small synthetic dataset for testing."""
    np.random.seed(42)
    N = 50  # Small for fast tests

    # Create three data views
    X1 = np.random.normal(0, 1, (N, 20))  # View 1: 20 features
    X2 = np.random.normal(0, 1, (N, 15))  # View 2: 15 features
    X3 = np.random.normal(0, 1, (N, 10))  # View 3: 10 features

    X_list = [X1, X2, X3]

    # Create subject IDs
    subject_ids = [f"subj_{i:03d}" for i in range(N)]

    # Create feature names
    feature_names = {
        "view_1": [f"view_1_feat_{j:03d}" for j in range(20)],
        "view_2": [f"view_2_feat_{j:03d}" for j in range(15)],
        "view_3": [f"view_3_feat_{j:03d}" for j in range(10)],
    }

    # Create clinical dataframe
    clinical = pd.DataFrame(
        {
            "age": np.random.normal(65, 10, N),
            "gender": np.random.choice(["M", "F"], N),
            "diagnosis": np.random.choice(["PD", "HC"], N),
        },
        index=subject_ids,
    )

    # Create scalers
    scalers = {}
    view_names = ["view_1", "view_2", "view_3"]
    for i, view_name in enumerate(view_names):
        scalers[view_name] = {
            "mu": np.zeros((1, X_list[i].shape[1])),
            "sd": np.ones((1, X_list[i].shape[1])),
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
            "Dm": [20, 15, 10],
            "N": N,
            "K_true": 3,
            "preprocessing_enabled": False,
        },
        "ground_truth": {
            "Z": np.random.normal(0, 1, (N, 3)),
            "W": np.random.normal(0, 1, (45, 3)),  # 20+15+10 = 45
            "sigma": np.array([1.0, 1.5, 2.0]),
            "K_true": 3,
            "Dm": np.array([20, 15, 10]),
        },
    }


@pytest.fixture
def sample_hyperparameters():
    """Create sample hyperparameters."""
    return {
        "a_sigma": 1.0,
        "b_sigma": 1.0,
        "nu_local": 1.0,
        "nu_global": 1.0,
        "slab_scale": 2.0,
        "slab_df": 4.0,
        "percW": 33.0,
        "Dm": [20, 15, 10],
    }


@pytest.fixture
def mock_mcmc_results():
    """Create mock MCMC results for testing."""
    return {
        "Z": np.random.normal(0, 1, (1000, 50, 3)),  # samples x subjects x factors
        "W": np.random.normal(0, 1, (1000, 45, 3)),  # samples x features x factors
        "sigma": np.random.gamma(2, 1, (1000, 3)),  # samples x views
        "log_likelihood": -1000 * np.random.exponential(1, 1000),
    }


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    import logging

    logging.basicConfig(level=logging.WARNING)  # Reduce noise in tests


@pytest.fixture
def mock_jax():
    """Mock JAX operations for testing without GPU/TPU."""
    import jax.numpy as jnp

    # Mock random key
    mock_key = np.random.RandomState(42)

    # Mock basic jax operations
    return Mock(
        random=Mock(PRNGKey=lambda x: mock_key, split=lambda key, n: [mock_key] * n),
        numpy=jnp,
    )
