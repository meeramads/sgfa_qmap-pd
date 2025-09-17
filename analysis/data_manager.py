# analysis/data_manager.py
"""Module for data loading and preprocessing."""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data loading and preprocessing."""

    def __init__(self, args, hyperparameters_dir=None):
        self.args = args
        self.hyperparameters_dir = hyperparameters_dir
        self.preprocessor = None

    def load_data(self) -> Dict:
        """Load data based on configuration."""
        if self.args.dataset == "synthetic":
            return self._load_synthetic_data()
        elif self.args.dataset == "qmap_pd":
            return self._load_qmap_data()
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")

    def _load_qmap_data(self) -> Dict:
        """Load qMAP-PD dataset with preprocessing."""
        from data.qmap_pd import load_qmap_pd

        data = load_qmap_pd(
            data_dir=self.config.data_dir,
            clinical_rel=self.config.clinical_rel,
            volumes_rel=self.config.volumes_rel,
            imaging_as_single_view=not self.config.roi_views,
            enable_advanced_preprocessing=self.config.enable_preprocessing,
            enable_spatial_processing=self.config.enable_spatial_processing,
            **self.config.preprocessing_params,
        )

        # Log preprocessing results if available
        if "preprocessing" in data:
            self._log_preprocessing_results(data["preprocessing"])

        return data

    def _load_synthetic_data(self) -> Dict:
        """Generate synthetic data for testing."""
        from data.synthetic import generate_synthetic_data

        return generate_synthetic_data(
            self.args.num_sources, self.args.K, getattr(self.args, 'percW', 25.0)
        )

    def prepare_for_analysis(self, data: Dict) -> Tuple[List[np.ndarray], Dict]:
        """Prepare data for analysis."""
        X_list = data["X_list"]

        # Setup hyperparameters
        hypers = {
            "a_sigma": 1,
            "b_sigma": 1,
            "nu_local": 1,
            "nu_global": 1,
            "slab_scale": 2,
            "slab_df": 4,
            "percW": getattr(self.args, 'percW', 25.0),
            "Dm": [X.shape[1] for X in X_list],
        }

        return X_list, hypers

    def _log_preprocessing_results(self, preprocessing: Dict):
        """Log preprocessing results."""
        if "feature_reduction" in preprocessing:
            logger.info("=== Preprocessing Applied ===")
            for view, stats in preprocessing["feature_reduction"].items():
                logger.info(
                    f"{view}: {stats['original']} -> {stats['processed']} features "
                    f"({stats['reduction_ratio']:.2%} retained)"
                )
