"""
High-level data access interface.
Combines loading and preprocessing based on dataset type.
"""

import logging
from typing import Any, Dict

from data.qmap_pd import load_qmap_pd as qmap_pd
from data.synthetic import generate_synthetic_data

logging.basicConfig(level=logging.INFO)


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
        # Generate synthetic data using the new module
        return generate_synthetic_data(
            num_sources=kwargs.get("num_sources", 3),
            K=kwargs.get("K", 3),
            percW=kwargs.get("percW", 33.0),
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. Supported: 'qmap_pd', 'synthetic'"
        )
