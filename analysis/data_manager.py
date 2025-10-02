# analysis/data_manager.py
"""Module for data loading and preprocessing."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DataManagerConfig:
    """Configuration for DataManager.

    Attributes:
        dataset: Dataset name ('qmap_pd' or 'synthetic')
        data_dir: Directory containing data files
        clinical_rel: Relative path to clinical data
        volumes_rel: Relative path to volumes data
        roi_views: Whether to split ROIs into separate views
        enable_preprocessing: Enable advanced preprocessing
        enable_spatial_processing: Enable spatial preprocessing
        preprocessing_params: Additional preprocessing parameters
        num_sources: Number of sources (for synthetic data)
        K: Number of factors (for synthetic data)
        percW: Sparsity percentage (for synthetic data)
    """
    dataset: str = "qmap_pd"
    data_dir: Optional[str] = None
    clinical_rel: Optional[str] = None
    volumes_rel: Optional[str] = None
    roi_views: bool = True
    enable_preprocessing: bool = True
    enable_spatial_processing: bool = False
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)

    # Synthetic data params
    num_sources: int = 4
    K: int = 10
    percW: float = 25.0

    @classmethod
    def from_dict(cls, d: Dict) -> 'DataManagerConfig':
        """Create config from dictionary, extracting only valid fields."""
        valid_fields = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_fields})

    @classmethod
    def from_object(cls, obj: Any) -> 'DataManagerConfig':
        """Create config from object with attributes."""
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)

        # Extract attributes from object
        valid_fields = cls.__dataclass_fields__.keys()
        kwargs = {}
        for field_name in valid_fields:
            if hasattr(obj, field_name):
                kwargs[field_name] = getattr(obj, field_name)
        return cls(**kwargs)


class DataManager:
    """Manages data loading and preprocessing."""

    def __init__(self, config: DataManagerConfig):
        """Initialize DataManager with proper config.

        Args:
            config: DataManagerConfig instance
        """
        if not isinstance(config, DataManagerConfig):
            # Auto-convert if needed (temporary for migration)
            config = DataManagerConfig.from_object(config)

        self.config = config
        self.preprocessor = None

    def load_data(self) -> Dict:
        """Load data based on configuration."""
        logger.info(f"📊 Loading dataset: {self.config.dataset}")

        if self.config.dataset == "synthetic":
            logger.info("🎲 Generating synthetic data for testing")
            return self._load_synthetic_data()
        elif self.config.dataset == "qmap_pd":
            logger.info(f"🧠 Loading qMAP-PD data from: {self.config.data_dir}")
            return self._load_qmap_data()
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")

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
            self.config.num_sources, self.config.K, self.config.percW
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
            "percW": self.config.percW,
            "Dm": [X.shape[1] for X in X_list],
        }

        return X_list, hypers

    def load_and_prepare_data(self, hypers: Dict) -> Tuple[List[np.ndarray], Dict, Dict]:
        """Load data and prepare it for analysis (backward compatibility method)."""
        # Load data
        data = self.load_data()

        # Prepare for analysis
        X_list, prepared_hypers = self.prepare_for_analysis(data)

        # Merge provided hypers with prepared hypers
        final_hypers = {**prepared_hypers, **hypers}

        return X_list, data, final_hypers

    def estimate_memory_requirements(self, X_list: List[np.ndarray]) -> None:
        """Estimate memory requirements for the analysis (backward compatibility method)."""
        total_size = sum(X.nbytes for X in X_list)
        total_size_gb = total_size / (1024**3)

        logger.info(f"Estimated memory requirement: {total_size_gb:.2f} GB")

        if total_size_gb > 8:
            logger.warning("Large dataset detected. Consider using memory optimization.")
        elif total_size_gb > 16:
            logger.error("Very large dataset. Memory optimization strongly recommended.")

    def _log_preprocessing_results(self, preprocessing: Dict):
        """Log preprocessing results."""
        if "feature_reduction" in preprocessing:
            logger.info("=== Preprocessing Applied ===")
            for view, stats in preprocessing["feature_reduction"].items():
                logger.info(
                    f"{view}: {stats['original']} -> {stats['processed']} features "
                    f"({stats['reduction_ratio']:.2%} retained)"
                )
