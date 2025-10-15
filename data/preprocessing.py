"""
Complete Preprocessing Module for Sparse Bayesian Group Factor Analysis.
Includes neuroimaging-aware preprocessing for qMRI data.

This module can be used in two ways:

1. As a Python module (imported by other scripts):
   from data.preprocessing import NeuroImagingPreprocessor, preprocess_neuroimaging_data

2. As a standalone CLI tool (run directly):
   python -m data.preprocessing --data_dir qMAP-PD_data --inspect_only
   python -m data.preprocessing --data_dir qMAP-PD_data --compare_methods
"""

# For standalone CLI functionality
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    mutual_info_classif,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from core.io_utils import save_plot

logging.basicConfig(level=logging.INFO)

# == COMPATIBILITY CHECKS ==


def check_preprocessing_compatibility():
    """Check if all required dependencies are available"""
    missing = []

    try:
        pass
    except ImportError:
        missing.append("scikit-learn (preprocessing)")

    try:
        pass
    except ImportError:
        missing.append("scikit-learn (KNN imputer)")

    try:
        pass
    except ImportError:
        missing.append("scikit-learn (iterative imputer)")

    # Check for neuroimaging-specific dependencies
    try:
        pass
    except ImportError:
        missing.append("scipy (spatial functions)")

    if missing:
        logging.warning(f"Missing optional dependencies: {missing}")
        logging.warning("Some preprocessing methods may not be available")

    return len(missing) == 0


# == CONFIGURATION CLASSES ==


class PreprocessingConfig:
    """Base preprocessing configuration with validation"""

    VALID_IMPUTATION = ["median", "mean", "knn", "iterative"]
    VALID_SELECTION = ["variance", "statistical", "mutual_info", "combined", "none"]

    def __init__(self, **kwargs):
        # Extract preprocessing params
        self.enable_preprocessing = kwargs.get("enable_preprocessing", False)
        self.imputation_strategy = kwargs.get("imputation_strategy", "median")
        self.feature_selection_method = kwargs.get(
            "feature_selection_method", "variance"
        )
        self.n_top_features = kwargs.get("n_top_features", None)
        self.missing_threshold = kwargs.get("missing_threshold", 0.1)
        self.variance_threshold = kwargs.get("variance_threshold", 0.0)
        self.target_variable = kwargs.get("target_variable", None)
        self.cross_validate_sources = kwargs.get("cross_validate_sources", False)
        self.optimize_preprocessing = kwargs.get("optimize_preprocessing", False)

        # PCA dimensionality reduction params
        self.enable_pca = kwargs.get("enable_pca", False)
        self.pca_n_components = kwargs.get("pca_n_components", None)
        self.pca_variance_threshold = kwargs.get("pca_variance_threshold", 0.95)
        self.pca_whiten = kwargs.get("pca_whiten", False)

        if self.enable_preprocessing:
            self.validate()

    def validate(self):
        """Validate preprocessing parameters"""
        if self.imputation_strategy not in self.VALID_IMPUTATION:
            raise ValueError(
                f"Invalid imputation_strategy. Must be one of: {self.VALID_IMPUTATION}"
            )

        if self.feature_selection_method not in self.VALID_SELECTION:
            raise ValueError(
                f"Invalid feature_selection. Must be one of: {self.VALID_SELECTION}"
            )

        if self.n_top_features is not None and self.n_top_features <= 0:
            raise ValueError(
                f"Invalid n_top_features={ self.n_top_features}. Must be positive integer or None."
            )

        if not (0.0 <= self.missing_threshold <= 1.0):
            raise ValueError(f"Invalid missing_threshold. Must be between 0.0 and 1.0.")

        if self.variance_threshold < 0:
            raise ValueError(f"Invalid variance_threshold. Must be non-negative.")

        # Validate PCA parameters
        if self.enable_pca:
            if self.pca_n_components is not None and self.pca_n_components <= 0:
                raise ValueError(f"Invalid pca_n_components. Must be positive integer or None.")

            if not (0.0 < self.pca_variance_threshold <= 1.0):
                raise ValueError(f"Invalid pca_variance_threshold. Must be between 0.0 and 1.0.")

    def to_dict(self):
        """Convert to dict for passing to functions"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("VALID_")}


class NeuroImagingConfig(PreprocessingConfig):
    """Extended config for neuroimaging-specific preprocessing"""

    def __init__(self, **kwargs):
        # Neuroimaging-specific parameters
        self.enable_spatial_processing = kwargs.get("enable_spatial_processing", True)
        self.spatial_smoothing_fwhm = kwargs.get(
            "spatial_smoothing_fwhm", None
        )  # in mm
        self.harmonize_scanners = kwargs.get("harmonize_scanners", False)
        self.scanner_info_col = kwargs.get("scanner_info_col", None)
        self.spatial_imputation = kwargs.get("spatial_imputation", True)
        self.roi_based_selection = kwargs.get("roi_based_selection", True)
        self.qc_outlier_threshold = kwargs.get("qc_outlier_threshold", 3.0)  # std devs
        self.enable_combat_harmonization = kwargs.get(
            "enable_combat_harmonization", False
        )
        self.spatial_neighbor_radius = kwargs.get("spatial_neighbor_radius", 5.0)  # mm
        self.min_voxel_distance = kwargs.get(
            "min_voxel_distance", 3.0
        )  # mm for feature selection

        # Call parent constructor
        super().__init__(**kwargs)

    def validate(self):
        """Extended validation for neuroimaging parameters"""
        super().validate()

        if self.spatial_smoothing_fwhm is not None and self.spatial_smoothing_fwhm <= 0:
            raise ValueError("spatial_smoothing_fwhm must be positive or None")

        if self.qc_outlier_threshold <= 0:
            raise ValueError("qc_outlier_threshold must be positive")

        if self.spatial_neighbor_radius <= 0:
            raise ValueError("spatial_neighbor_radius must be positive")

        if self.min_voxel_distance < 0:
            raise ValueError("min_voxel_distance must be non-negative")


# == SPATIAL PROCESSING UTILITIES ==


class SpatialProcessingUtils:
    """Utilities for spatial processing of neuroimaging data"""

    @staticmethod
    def load_position_lookup(data_dir: str, roi_name: str) -> Optional[pd.DataFrame]:
        """Load spatial position information for ROI voxels"""
        try:
            position_file = (
                Path(data_dir) / "position_lookup" / f"position_{roi_name}_voxels.tsv"
            )
            if position_file.exists():
                positions = pd.read_csv(
                    position_file, sep="\t", header=None, names=["x", "y", "z"]
                )
                logging.info(f"Loaded {len(positions)} voxel positions for {roi_name}")
                return positions
            else:
                logging.warning(f"Position lookup file not found: {position_file}")
                return None
        except Exception as e:
            logging.warning(f"Could not load position lookup for {roi_name}: {e}")
            return None

    @staticmethod
    def find_spatial_neighbors(
        voxel_idx: int, positions: pd.DataFrame, radius_mm: float = 5.0
    ) -> np.ndarray:
        """Find spatial neighbors of a voxel within given radius"""
        if positions is None or voxel_idx >= len(positions):
            return np.array([])

        target_pos = positions.iloc[voxel_idx][["x", "y", "z"]].values
        all_positions = positions[["x", "y", "z"]].values

        # Calculate Euclidean distances
        distances = np.sqrt(np.sum((all_positions - target_pos) ** 2, axis=1))

        # Find neighbors within radius (excluding self)
        neighbors = np.where((distances <= radius_mm) & (distances > 0))[0]

        return neighbors

    @staticmethod
    def detect_outlier_voxels(X: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Detect voxels with outlier intensities that may indicate artifacts"""
        outlier_mask = np.zeros(X.shape[1], dtype=bool)
        outlier_scores = []  # Track max MAD scores for reporting

        for i in range(X.shape[1]):
            voxel_data = X[:, i]
            if not np.all(np.isnan(voxel_data)):
                # Remove NaN for statistics
                clean_data = voxel_data[~np.isnan(voxel_data)]
                if len(clean_data) > 0:
                    median = np.median(clean_data)
                    mad = np.median(np.abs(clean_data - median))

                    # Check for extreme outliers using MAD (more robust than std)
                    if mad > 0:  # Avoid division by zero
                        mad_scores = np.abs(clean_data - median) / (mad * 1.4826)
                        max_mad_score = np.max(mad_scores)
                        outlier_scores.append(max_mad_score)
                        if max_mad_score > threshold:
                            outlier_mask[i] = True

        # Log summary statistics
        if len(outlier_scores) > 0:
            outlier_scores = np.array(outlier_scores)
            logging.info(
                f"  MAD outlier detection: threshold={threshold:.1f}, "
                f"max_score={np.max(outlier_scores):.2f}, "
                f"median_score={np.median(outlier_scores):.2f}, "
                f"outliers={np.sum(outlier_mask)}/{len(outlier_mask)} voxels"
            )

        return outlier_mask

    @staticmethod
    def apply_basic_harmonization(
        X: np.ndarray,
        scanner_info: np.ndarray,
        global_mean: Optional[np.ndarray] = None,
        global_std: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Apply basic scanner harmonization (simplified ComBat-like approach)"""

        X_harmonized = X.copy()
        unique_scanners = np.unique(scanner_info[~pd.isna(scanner_info)])
        harmonization_stats = {}

        # Compute global statistics if not provided
        if global_mean is None:
            global_mean = np.nanmean(X, axis=0)
        if global_std is None:
            global_std = np.nanstd(X, axis=0)
            global_std = np.where(global_std < 1e-6, 1.0, global_std)

        for scanner in unique_scanners:
            scanner_mask = scanner_info == scanner
            n_subjects = np.sum(scanner_mask)

            if n_subjects > 1:  # Need at least 2 subjects
                scanner_data = X[scanner_mask, :]

                # Compute scanner-specific statistics
                scanner_mean = np.nanmean(scanner_data, axis=0)
                scanner_std = np.nanstd(scanner_data, axis=0)
                scanner_std = np.where(scanner_std < 1e-6, 1.0, scanner_std)

                harmonization_stats[scanner] = {
                    "mean": scanner_mean,
                    "std": scanner_std,
                    "n_subjects": n_subjects,
                }

                # Apply harmonization: (x - scanner_mean) / scanner_std * global_std +
                # global_mean
                X_harmonized[scanner_mask, :] = (
                    (scanner_data - scanner_mean) / scanner_std
                ) * global_std + global_mean

                logging.info(f"Harmonized {n_subjects} subjects from scanner {scanner}")

        return X_harmonized, harmonization_stats


# == BASE PREPROCESSOR CLASSES ==


class BasePreprocessor:
    """Base preprocessing utilities - consolidates scattered functions"""

    @staticmethod
    def fit_basic_scaler(X: np.ndarray, eps: float = 1e-8) -> Dict[str, np.ndarray]:
        """
        Basic scaling (moved from loader_qmap_pd.py).
        Returns dict with mu, sd for compatibility.
        """
        mu = np.nanmean(X, axis=0, keepdims=True)
        sd = np.nanstd(X, axis=0, keepdims=True)
        sd = np.where(sd < eps, 1.0, sd)
        return {"mu": mu, "sd": sd}

    @staticmethod
    def apply_basic_scaler(
        X: np.ndarray, scaler_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Apply basic scaling (moved from loader_qmap_pd.py)"""
        Xz = (X - scaler_dict["mu"]) / scaler_dict["sd"]
        return np.where(np.isnan(Xz), 0.0, Xz)


class BasicPreprocessor(BasePreprocessor):
    """Minimal preprocessor for basic scaling only"""

    def __init__(self):
        self.scalers_ = {}
        self.is_basic = True

    def fit_transform(
        self,
        X_list: List[np.ndarray],
        view_names: List[str],
        y: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Basic preprocessing - just scaling"""
        processed_X = []

        for X, view_name in zip(X_list, view_names):
            if X.shape[1] > 0:
                scaler_dict = self.fit_basic_scaler(X)
                X_scaled = self.apply_basic_scaler(X, scaler_dict)
                self.scalers_[view_name] = scaler_dict
            else:
                X_scaled = X
                self.scalers_[view_name] = {
                    "mu": np.zeros((1, 0)),
                    "sd": np.ones((1, 0)),
                }

            processed_X.append(X_scaled)

        return processed_X

    def transform(
        self, X_list: List[np.ndarray], view_names: List[str]
    ) -> List[np.ndarray]:
        """Transform using basic scaling"""
        processed_X = []
        for X, view_name in zip(X_list, view_names):
            if view_name in self.scalers_:
                X_scaled = self.apply_basic_scaler(X, self.scalers_[view_name])
            else:
                X_scaled = X
            processed_X.append(X_scaled)
        return processed_X


class AdvancedPreprocessor(BasePreprocessor):
    """
    Advanced preprocessing following research methodologies.

    Key features:
    - Multiple imputation strategies
    - Variance-based and statistical feature selection
    - Cross-validation for optimal feature/source selection
    - Robust scaling with outlier handling
    """

    def __init__(
        self,
        missing_threshold: float = 0.1,
        variance_threshold: float = 0.0,
        n_top_features: Optional[int] = None,
        imputation_strategy: str = "median",
        feature_selection_method: str = "variance",
        random_state: int = 42,
        enable_pca: bool = False,
        pca_n_components: Optional[int] = None,
        pca_variance_threshold: float = 0.95,
        pca_whiten: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        missing_threshold : float
            Remove features with > this fraction of missing values
        variance_threshold : float
            Remove features with variance below this threshold
        n_top_features : int, optional
            Number of top features to select (if None, uses variance_threshold)
        imputation_strategy : str
            'median', 'mean', 'knn', or 'iterative'
        feature_selection_method : str
            'variance', 'statistical', 'mutual_info', or 'combined'
        enable_pca : bool
            Whether to apply PCA dimensionality reduction
        pca_n_components : int, optional
            Number of PCA components (if None, uses pca_variance_threshold)
        pca_variance_threshold : float
            Cumulative variance to retain if pca_n_components is None (default: 0.95)
        pca_whiten : bool
            Whether to whiten the PCA components (default: False)
        """
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        self.n_top_features = n_top_features
        self.imputation_strategy = imputation_strategy
        self.feature_selection_method = feature_selection_method
        self.random_state = random_state

        # PCA parameters
        self.enable_pca = enable_pca
        self.pca_n_components = pca_n_components
        self.pca_variance_threshold = pca_variance_threshold
        self.pca_whiten = pca_whiten

        # Storage for fitted transformers
        self.imputers_ = {}
        self.scalers_ = {}
        self.feature_selectors_ = {}
        self.selected_features_ = {}
        self.pca_transformers_ = {}
        self.is_basic = False

    def fit_transform_imputation(self, X: np.ndarray, view_name: str) -> np.ndarray:
        """
        Fit and apply imputation strategy to data.
        """
        logging.info(f"Applying {self.imputation_strategy} imputation to {view_name}")

        # Check missing data percentage
        missing_pct = np.isnan(X).mean(axis=0)
        features_to_drop = missing_pct > self.missing_threshold

        if np.any(features_to_drop):
            logging.warning(
                f"Dropping { np.sum(features_to_drop)} features with >{ self.missing_threshold * 100}% missing data"
            )
            X = X[:, ~features_to_drop]

        if self.imputation_strategy == "median":
            imputer = SimpleImputer(strategy="median")
        elif self.imputation_strategy == "mean":
            imputer = SimpleImputer(strategy="mean")
        elif self.imputation_strategy == "knn":
            # KNN imputation - more sophisticated than simple median
            imputer = KNNImputer(n_neighbors=5)
        elif self.imputation_strategy == "iterative":
            try:
                from sklearn.impute import IterativeImputer

                imputer = IterativeImputer(random_state=self.random_state)
            except ImportError:
                logging.warning(
                    "IterativeImputer not available, falling back to median"
                )
                imputer = SimpleImputer(strategy="median")
        else:
            raise ValueError(f"Unknown imputation strategy: {self.imputation_strategy}")

        X_imputed = imputer.fit_transform(X)
        self.imputers_[view_name] = imputer

        # Store feature mask for consistent application during transform
        self.selected_features_[f"{view_name}_missing_mask"] = ~features_to_drop

        return X_imputed

    def fit_transform_feature_selection(
        self, X: np.ndarray, view_name: str, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply feature selection following both papers' methodologies.

        Ferreira et al.: "We reduced the dimensionality to the 500 genes with the highest
        average variance over the data views"

        Bunte et al.: Cross-validation to select most predictive features
        """
        logging.info(
            f"Applying {self.feature_selection_method} feature selection to {view_name}"
        )

        if self.feature_selection_method == "none":
            return X
        elif self.feature_selection_method == "variance":
            return self._variance_based_selection(X, view_name)
        elif self.feature_selection_method == "statistical" and y is not None:
            return self._statistical_selection(X, view_name, y)
        elif self.feature_selection_method == "mutual_info" and y is not None:
            return self._mutual_info_selection(X, view_name, y)
        elif self.feature_selection_method == "combined" and y is not None:
            return self._combined_selection(X, view_name, y)
        else:
            logging.warning(
                f"Unsupervised feature selection for {view_name} - using variance"
            )
            return self._variance_based_selection(X, view_name)

    def _variance_based_selection(self, X: np.ndarray, view_name: str) -> np.ndarray:
        """Variance-based feature selection as in Ferreira et al."""
        if self.n_top_features is not None:
            # Select top-k features by variance
            variances = np.var(X, axis=0)
            top_indices = np.argsort(variances)[::-1][: self.n_top_features]
            X_selected = X[:, top_indices]
            self.selected_features_[f"{view_name}_variance_indices"] = top_indices
        else:
            # Use variance threshold
            selector = VarianceThreshold(threshold=self.variance_threshold)
            X_selected = selector.fit_transform(X)
            self.feature_selectors_[f"{view_name}_variance"] = selector

        logging.info(
            f"Selected { X_selected.shape[1]} features for {view_name} (was { X.shape[1]})"
        )
        return X_selected

    def _statistical_selection(
        self, X: np.ndarray, view_name: str, y: np.ndarray
    ) -> np.ndarray:
        """Statistical feature selection using F-test or correlation."""
        n_features = self.n_top_features or min(500, X.shape[1] // 2)

        if len(np.unique(y)) > 10:  # Regression
            # Use correlation for continuous targets
            correlations = np.abs(
                [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
            )
            correlations = np.nan_to_num(correlations)  # Handle NaN correlations
            top_indices = np.argsort(correlations)[::-1][:n_features]
            X_selected = X[:, top_indices]
            self.selected_features_[f"{view_name}_corr_indices"] = top_indices
        else:  # Classification
            selector = SelectKBest(score_func=f_classif, k=n_features)
            X_selected = selector.fit_transform(X, y)
            self.feature_selectors_[f"{view_name}_ftest"] = selector

        logging.info(
            f"Selected { X_selected.shape[1]} features for {view_name} using statistical selection"
        )
        return X_selected

    def _mutual_info_selection(
        self, X: np.ndarray, view_name: str, y: np.ndarray
    ) -> np.ndarray:
        """Mutual information-based feature selection."""
        n_features = self.n_top_features or min(500, X.shape[1] // 2)

        selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        self.feature_selectors_[f"{view_name}_mutinfo"] = selector

        logging.info(
            f"Selected { X_selected.shape[1]} features for {view_name} using mutual information"
        )
        return X_selected

    def _combined_selection(
        self, X: np.ndarray, view_name: str, y: np.ndarray
    ) -> np.ndarray:
        """Combined variance + statistical selection."""
        # First apply variance threshold
        X_var = self._variance_based_selection(X, f"{view_name}_temp")

        # Then apply statistical selection on reduced set
        if y is not None:
            X_selected = self._statistical_selection(X_var, view_name, y)
        else:
            X_selected = X_var

        return X_selected

    def fit_transform_scaling(
        self, X: np.ndarray, view_name: str, robust: bool = True
    ) -> np.ndarray:
        """
        Apply standardization.
        """
        logging.info(
            f"Applying {'robust' if robust else 'standard'} scaling to {view_name}"
        )

        if robust:
            try:
                from sklearn.preprocessing import RobustScaler

                scaler = RobustScaler()
            except ImportError:
                logging.warning("RobustScaler not available, using StandardScaler")
                scaler = StandardScaler()
        else:
            scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)
        self.scalers_[view_name] = scaler

        return X_scaled

    def fit_transform_pca(
        self, X: np.ndarray, view_name: str
    ) -> np.ndarray:
        """
        Apply PCA dimensionality reduction.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix (n_samples, n_features)
        view_name : str
            Name of the view being processed

        Returns
        -------
        np.ndarray
            PCA-transformed data (n_samples, n_components)
        """
        if not self.enable_pca:
            return X

        n_features_before = X.shape[1]
        logging.info(
            f"Applying PCA dimensionality reduction to {view_name} "
            f"({n_features_before} features)"
        )

        from sklearn.decomposition import PCA

        # Determine number of components
        if self.pca_n_components is not None:
            # Use fixed number of components
            n_components = min(self.pca_n_components, X.shape[0], X.shape[1])
            logging.info(
                f"  Strategy: fixed number of components (n={n_components})"
            )
        else:
            # Use variance threshold to determine components
            # First fit with max components to see variance explained
            n_components = min(X.shape[0], X.shape[1])
            logging.info(
                f"  Strategy: variance threshold ({self.pca_variance_threshold*100:.0f}% variance)"
            )

        # Fit PCA
        pca = PCA(
            n_components=n_components,
            whiten=self.pca_whiten,
            random_state=self.random_state
        )
        X_pca = pca.fit_transform(X)

        # If using variance threshold, select components that meet threshold
        if self.pca_n_components is None:
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components_selected = np.searchsorted(cumsum_variance, self.pca_variance_threshold) + 1
            n_components_selected = min(n_components_selected, X_pca.shape[1])

            if n_components_selected < X_pca.shape[1]:
                actual_variance = cumsum_variance[n_components_selected-1]
                logging.info(
                    f"  Selected {n_components_selected} components "
                    f"(actual variance: {actual_variance:.2%})"
                )
                X_pca = X_pca[:, :n_components_selected]

                # Refit PCA with selected number of components for cleaner transform
                pca = PCA(
                    n_components=n_components_selected,
                    whiten=self.pca_whiten,
                    random_state=self.random_state
                )
                X_pca = pca.fit_transform(X)

        # Store PCA transformer
        self.pca_transformers_[view_name] = pca

        # Log transformation summary
        total_variance = pca.explained_variance_ratio_.sum()
        n_components_final = X_pca.shape[1]
        reduction_pct = ((n_features_before - n_components_final) / n_features_before) * 100

        logging.info(
            f"  PCA complete: {n_features_before} features → {n_components_final} components "
            f"({reduction_pct:.1f}% reduction, {total_variance:.2%} variance retained)"
        )

        # Log top principal components contribution
        if n_components_final >= 5:
            top5_variance = pca.explained_variance_ratio_[:5].sum()
            logging.info(
                f"  Top 5 PCs explain {top5_variance:.2%} of variance"
            )

        return X_pca

    def inverse_transform_pca_loadings(
        self, W_pcs: np.ndarray, view_name: str
    ) -> Optional[np.ndarray]:
        """
        Transform factor loadings from PC space back to original feature space.

        This is critical for brain remapping when PCA is used. Factor loadings
        from SGFA (W_pcs) are in principal component space and need to be
        transformed back to voxel space for spatial visualization.

        Parameters
        ----------
        W_pcs : np.ndarray
            Factor loadings in PC space, shape (n_components, n_factors)
        view_name : str
            Name of the view to transform

        Returns
        -------
        np.ndarray or None
            Factor loadings in original feature space (n_features, n_factors)
            Returns None if PCA was not applied to this view

        Mathematical relationship:
            X_pcs = X_features @ PCA.components_.T
            Therefore: W_features = PCA.components_.T @ W_pcs

        Examples
        --------
        >>> # After running SGFA with PCA preprocessing
        >>> W_pcs = sgfa_results['W'][0]  # Shape: (50, 8) - 50 PCs, 8 factors
        >>> W_voxels = preprocessor.inverse_transform_pca_loadings(W_pcs, 'volume_sn_voxels')
        >>> # Shape: (850, 8) - 850 voxels, 8 factors - ready for brain remapping!
        """
        if view_name not in self.pca_transformers_:
            logging.warning(f"No PCA transformer found for {view_name}. Returning None.")
            return None

        pca = self.pca_transformers_[view_name]
        n_factors = W_pcs.shape[1] if len(W_pcs.shape) > 1 else 1

        logging.info(
            f"Performing inverse PCA transform for brain remapping: {view_name}"
        )
        logging.info(
            f"  Input: W_pcs shape = {W_pcs.shape} "
            f"({W_pcs.shape[0]} principal components × {n_factors} factors)"
        )

        # Transform: W_features = PCA.components_.T @ W_pcs
        # PCA.components_ shape: (n_components, n_features)
        # PCA.components_.T shape: (n_features, n_components)
        # W_pcs shape: (n_components, n_factors)
        # Result shape: (n_features, n_factors)
        W_features = pca.components_.T @ W_pcs

        logging.info(
            f"  Output: W_voxels shape = {W_features.shape} "
            f"({W_features.shape[0]} voxels × {n_factors} factors)"
        )
        logging.info(
            f"  Brain remapping ready: W_voxels can be used with position lookup vectors"
        )

        return W_features

    def has_pca(self, view_name: str) -> bool:
        """Check if PCA was applied to a view."""
        return view_name in self.pca_transformers_

    def get_pca_info(self, view_name: str) -> Optional[Dict]:
        """
        Get PCA transformation information for a view.

        Returns
        -------
        dict or None
            Dictionary with PCA information:
            - n_components: Number of components
            - explained_variance_ratio: Variance explained by each component
            - total_variance: Total variance explained
            - components_shape: Shape of PCA components matrix
        """
        if view_name not in self.pca_transformers_:
            return None

        pca = self.pca_transformers_[view_name]
        return {
            'n_components': pca.n_components_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'total_variance': pca.explained_variance_ratio_.sum(),
            'components_shape': pca.components_.shape,
            'n_features_original': pca.components_.shape[1],
        }

    def fit_transform(
        self,
        X_list: List[np.ndarray],
        view_names: List[str],
        y: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Complete preprocessing pipeline.

        Parameters
        ----------
        X_list : List of data matrices
        view_names : List of view names
        y : Target variable (optional, for supervised feature selection)

        Returns
        -------
        List of preprocessed data matrices
        """
        logging.info("Starting complete preprocessing pipeline")

        X_processed = []

        for X, view_name in zip(X_list, view_names):
            logging.info(f"Processing view: {view_name} (shape: {X.shape})")

            # Step 1: Handle missing data
            X_imputed = self.fit_transform_imputation(X, view_name)

            # Step 2: Feature selection
            X_selected = self.fit_transform_feature_selection(X_imputed, view_name, y)

            # Step 3: Scaling
            X_scaled = self.fit_transform_scaling(X_selected, view_name)

            # Step 4: PCA dimensionality reduction (if enabled)
            X_final = self.fit_transform_pca(X_scaled, view_name)

            X_processed.append(X_final)

            logging.info(f"Final shape for {view_name}: {X_final.shape}")

        return X_processed

    def transform(
        self, X_list: List[np.ndarray], view_names: List[str]
    ) -> List[np.ndarray]:
        """Transform new data using fitted preprocessors."""
        X_processed = []

        for X, view_name in zip(X_list, view_names):
            # Apply stored transformations

            # 1. Missing data handling
            if f"{view_name}_missing_mask" in self.selected_features_:
                mask = self.selected_features_[f"{view_name}_missing_mask"]
                X = X[:, mask]

            if view_name in self.imputers_:
                X_imputed = self.imputers_[view_name].transform(X)
            else:
                X_imputed = X

            # 2. Feature selection
            if f"{view_name}_variance_indices" in self.selected_features_:
                indices = self.selected_features_[f"{view_name}_variance_indices"]
                X_selected = X_imputed[:, indices]
            elif f"{view_name}_variance" in self.feature_selectors_:
                X_selected = self.feature_selectors_[f"{view_name}_variance"].transform(
                    X_imputed
                )
            else:
                X_selected = X_imputed

            # 3. Scaling
            if view_name in self.scalers_:
                X_scaled = self.scalers_[view_name].transform(X_selected)
            else:
                X_scaled = X_selected

            # 4. PCA transformation
            if view_name in self.pca_transformers_:
                X_final = self.pca_transformers_[view_name].transform(X_scaled)
            else:
                X_final = X_scaled

            X_processed.append(X_final)

        return X_processed


# == NEUROIMAGING-AWARE PREPROCESSOR ==


class NeuroImagingPreprocessor(AdvancedPreprocessor):
    """qMRI and neuroimaging-aware preprocessor with spatial processing"""

    def __init__(self, data_dir: str = None, **kwargs):
        # Extract neuroimaging-specific parameters
        self.data_dir = data_dir
        self.enable_spatial_processing = kwargs.pop("enable_spatial_processing", True)
        self.spatial_smoothing_fwhm = kwargs.pop("spatial_smoothing_fwhm", None)
        self.harmonize_scanners = kwargs.pop("harmonize_scanners", False)
        self.scanner_info_col = kwargs.pop("scanner_info_col", None)
        self.spatial_imputation = kwargs.pop("spatial_imputation", True)
        self.roi_based_selection = kwargs.pop("roi_based_selection", True)
        self.qc_outlier_threshold = kwargs.pop("qc_outlier_threshold", 3.0)
        self.enable_combat_harmonization = kwargs.pop(
            "enable_combat_harmonization", False
        )
        self.spatial_neighbor_radius = kwargs.pop("spatial_neighbor_radius", 5.0)
        self.min_voxel_distance = kwargs.pop("min_voxel_distance", 3.0)

        # Initialize parent
        super().__init__(**kwargs)

        # Storage for spatial information
        self.position_lookups_ = {}
        self.outlier_masks_ = {}
        self.harmonization_params_ = {}

    def _is_imaging_view(self, view_name: str) -> bool:
        """Check if view contains imaging data"""
        imaging_keywords = [
            "imaging",
            "volume_",
            "sn",
            "putamen",
            "lentiform",
            "bg-all",
        ]
        return any(keyword in view_name.lower() for keyword in imaging_keywords)

    def _extract_roi_name(self, view_name: str) -> str:
        """Extract ROI name from view name"""
        view_lower = view_name.lower()
        if "sn" in view_lower:
            return "sn"
        elif "putamen" in view_lower:
            return "putamen"
        elif "lentiform" in view_lower:
            return "lentiform"
        elif "bg-all" in view_lower:
            return "bg-all"
        else:
            return view_lower.replace("volume_", "").replace("_voxels", "")

    def _apply_quality_control(self, X: np.ndarray, view_name: str) -> np.ndarray:
        """Apply quality control checks for imaging data"""
        if not self._is_imaging_view(view_name):
            return X

        n_voxels_before = X.shape[1]
        logging.info(
            f"Applying MAD-based quality control to {view_name} "
            f"({n_voxels_before} voxels, threshold={self.qc_outlier_threshold:.1f})"
        )

        # Detect outlier voxels
        outlier_mask = SpatialProcessingUtils.detect_outlier_voxels(
            X, self.qc_outlier_threshold
        )

        if np.any(outlier_mask):
            n_outliers = np.sum(outlier_mask)
            n_retained = n_voxels_before - n_outliers
            retention_pct = (n_retained / n_voxels_before) * 100

            logging.warning(
                f"  Outlier voxels detected: {n_outliers}/{n_voxels_before} "
                f"({(n_outliers/n_voxels_before)*100:.1f}%)"
            )
            logging.info(
                f"  Voxels retained: {n_retained}/{n_voxels_before} ({retention_pct:.1f}%)"
            )

            X_clean = X[:, ~outlier_mask]
            self.outlier_masks_[view_name] = ~outlier_mask  # Store kept voxels
        else:
            logging.info(f"  No outlier voxels detected - all {n_voxels_before} voxels retained")
            X_clean = X
            self.outlier_masks_[view_name] = np.ones(X.shape[1], dtype=bool)

        return X_clean

    def _apply_spatial_imputation(self, X: np.ndarray, view_name: str) -> np.ndarray:
        """Apply spatially-aware imputation for imaging data"""
        if not self._is_imaging_view(view_name) or not self.spatial_imputation:
            return X

        logging.info(f"Applying spatial imputation to {view_name}")

        # Load position information
        if self.data_dir:
            roi_name = self._extract_roi_name(view_name)
            positions = SpatialProcessingUtils.load_position_lookup(
                self.data_dir, roi_name
            )
            self.position_lookups_[view_name] = positions
        else:
            positions = None

        if positions is None:
            logging.warning(
                f"No spatial information available for {view_name}, using standard imputation"
            )
            return X

        X_imputed = X.copy()
        n_spatial_imputations = 0

        # For each subject, impute missing voxels using spatial neighbors
        for subject_idx in range(X.shape[0]):
            missing_voxels = np.where(np.isnan(X[subject_idx, :]))[0]

            for voxel_idx in missing_voxels:
                if voxel_idx < len(positions):
                    # Find spatial neighbors
                    neighbors = SpatialProcessingUtils.find_spatial_neighbors(
                        voxel_idx, positions, radius_mm=self.spatial_neighbor_radius
                    )

                    if len(neighbors) > 0:
                        # Use median of non-missing neighbors
                        neighbor_values = X[subject_idx, neighbors]
                        valid_neighbors = neighbor_values[~np.isnan(neighbor_values)]

                        if len(valid_neighbors) > 0:
                            X_imputed[subject_idx, voxel_idx] = np.median(
                                valid_neighbors
                            )
                            n_spatial_imputations += 1

        if n_spatial_imputations > 0:
            logging.info(
                f"Performed {n_spatial_imputations} spatial imputations for {view_name}"
            )

        return X_imputed

    def _apply_scanner_harmonization(
        self, X: np.ndarray, view_name: str, scanner_info: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply scanner harmonization using ComBat-like approach"""
        if not self._is_imaging_view(view_name) or not self.harmonize_scanners:
            return X

        if scanner_info is None:
            logging.warning(
                f"No scanner info provided for harmonization of {view_name}"
            )
            return X

        logging.info(f"Applying scanner harmonization to {view_name}")

        X_harmonized, harmonization_stats = (
            SpatialProcessingUtils.apply_basic_harmonization(X, scanner_info)
        )

        self.harmonization_params_[view_name] = harmonization_stats
        return X_harmonized

    def _apply_roi_based_feature_selection(
        self, X: np.ndarray, view_name: str
    ) -> np.ndarray:
        """Apply ROI-based feature selection instead of pure variance"""
        if not self._is_imaging_view(view_name) or not self.roi_based_selection:
            return self._variance_based_selection(X, view_name)

        logging.info(f"Applying ROI-based feature selection to {view_name}")

        # Load position information to understand ROI structure
        positions = self.position_lookups_.get(view_name)
        if positions is None and self.data_dir:
            roi_name = self._extract_roi_name(view_name)
            positions = SpatialProcessingUtils.load_position_lookup(
                self.data_dir, roi_name
            )
            self.position_lookups_[view_name] = positions

        if positions is None:
            logging.warning(
                f"No spatial information for ROI-based selection, using variance"
            )
            return self._variance_based_selection(X, view_name)

        # Calculate variance for each voxel
        variances = np.var(X, axis=0)

        if self.n_top_features is not None:
            # Select top features by variance, but ensure spatial distribution
            n_select = min(self.n_top_features, X.shape[1])

            selected_indices = []
            variance_order = np.argsort(variances)[::-1]
            selected_positions = []

            for idx in variance_order:
                if len(selected_indices) >= n_select:
                    break

                if idx < len(positions):
                    pos = positions.iloc[idx][["x", "y", "z"]].values

                    # Check if too close to already selected voxels
                    too_close = False
                    for sel_pos in selected_positions:
                        if (
                            np.sqrt(np.sum((pos - sel_pos) ** 2))
                            < self.min_voxel_distance
                        ):
                            too_close = True
                            break

                    if not too_close:
                        selected_indices.append(idx)
                        selected_positions.append(pos)
                else:
                    # No position info for this voxel, include it
                    selected_indices.append(idx)

            # Fill remaining slots with highest variance if needed
            remaining_slots = n_select - len(selected_indices)
            if remaining_slots > 0:
                for idx in variance_order:
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                        remaining_slots -= 1
                        if remaining_slots == 0:
                            break

            X_selected = X[:, selected_indices]
            self.selected_features_[f"{view_name}_roi_indices"] = np.array(
                selected_indices
            )

        else:
            # Use variance threshold
            threshold = self.variance_threshold
            selected_mask = variances > threshold
            X_selected = X[:, selected_mask]
            self.selected_features_[f"{view_name}_roi_mask"] = selected_mask

        logging.info(
            f"ROI-based selection: {X_selected.shape[1]} features from {X.shape[1]} for {view_name}"
        )
        return X_selected

    def fit_transform_imputation(
        self, X: np.ndarray, view_name: str, scanner_info: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Enhanced imputation with neuroimaging-specific processing"""

        # Step 1: Quality control for imaging data
        X_qc = self._apply_quality_control(X, view_name)

        # Step 2: Spatial imputation for imaging data
        if self._is_imaging_view(view_name):
            X_spatial = self._apply_spatial_imputation(X_qc, view_name)
        else:
            X_spatial = X_qc

        # Step 3: Scanner harmonization
        X_harmonized = self._apply_scanner_harmonization(
            X_spatial, view_name, scanner_info
        )

        # Step 4: Standard imputation for remaining missing values
        X_imputed = super().fit_transform_imputation(X_harmonized, view_name)

        return X_imputed

    def fit_transform_feature_selection(
        self, X: np.ndarray, view_name: str, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Enhanced feature selection with spatial awareness"""

        if self._is_imaging_view(view_name) and self.roi_based_selection:
            return self._apply_roi_based_feature_selection(X, view_name)
        else:
            return super().fit_transform_feature_selection(X, view_name, y)

    def fit_transform(
        self,
        X_list: List[np.ndarray],
        view_names: List[str],
        y: Optional[np.ndarray] = None,
        scanner_info: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Enhanced preprocessing pipeline for neuroimaging data.

        Parameters
        ----------
        X_list : List of data matrices
        view_names : List of view names
        y : Target variable (optional)
        scanner_info : Scanner/site information for harmonization (optional)

        Returns
        -------
        List of preprocessed data matrices
        """
        logging.info("Starting neuroimaging-aware preprocessing pipeline")

        X_processed = []

        for X, view_name in zip(X_list, view_names):
            logging.info(f"Processing view: {view_name} (shape: {X.shape})")

            # Step 1: Enhanced imputation with spatial processing
            X_imputed = self.fit_transform_imputation(X, view_name, scanner_info)

            # Step 2: Enhanced feature selection with spatial awareness
            X_selected = self.fit_transform_feature_selection(X_imputed, view_name, y)

            # Step 3: Scaling
            X_scaled = self.fit_transform_scaling(X_selected, view_name)

            X_processed.append(X_scaled)

            logging.info(f"Final shape for {view_name}: {X_scaled.shape}")

        return X_processed


# == VALIDATION UTILITIES ==


def validate_preprocessing_inputs(
    X_list: List[np.ndarray],
    view_names: List[str],
    config: Union[PreprocessingConfig, NeuroImagingConfig],
) -> None:
    """Validate inputs before preprocessing"""

    if len(X_list) != len(view_names):
        raise ValueError(
            f"Mismatch: {len(X_list)} data arrays but {len(view_names)} view names"
        )

    for i, (X, view_name) in enumerate(zip(X_list, view_names)):
        if not isinstance(X, np.ndarray):
            raise TypeError(
                f"X_list[{i}] ({view_name}) must be numpy array, got {type(X)}"
            )

        if X.ndim != 2:
            raise ValueError(
                f"X_list[{i}] ({view_name}) must be 2D, got shape {X.shape}"
            )

        if X.shape[0] == 0:
            raise ValueError(f"X_list[{i}] ({view_name}) has no samples")

    # Check all arrays have same number of samples
    n_samples = [X.shape[0] for X in X_list]
    if len(set(n_samples)) > 1:
        raise ValueError(
            f"Inconsistent sample counts across views: {dict(zip(view_names, n_samples))}"
        )


# == HIGH-LEVEL PREPROCESSING INTERFACES ==


def create_preprocessor_from_config(
    config: Union[PreprocessingConfig, NeuroImagingConfig], data_dir: str = None
) -> Union[BasicPreprocessor, AdvancedPreprocessor, NeuroImagingPreprocessor]:
    """Create appropriate preprocessor based on configuration"""

    if not config.enable_preprocessing:
        return BasicPreprocessor()
    elif isinstance(config, NeuroImagingConfig) and config.enable_spatial_processing:
        return NeuroImagingPreprocessor(data_dir=data_dir, **config.to_dict())
    else:
        return AdvancedPreprocessor(**config.to_dict())


def preprocess_data_from_config(
    X_list: List[np.ndarray],
    view_names: List[str],
    config: Union[PreprocessingConfig, NeuroImagingConfig],
    data_dir: str = None,
    scanner_info: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
) -> Tuple[
    List[np.ndarray],
    Union[BasicPreprocessor, AdvancedPreprocessor, NeuroImagingPreprocessor],
]:
    """
    Main preprocessing interface using configuration object.
    """
    # Create preprocessor
    preprocessor = create_preprocessor_from_config(config, data_dir)

    # Apply preprocessing
    if isinstance(preprocessor, NeuroImagingPreprocessor):
        X_processed = preprocessor.fit_transform(X_list, view_names, y, scanner_info)
    else:
        X_processed = preprocessor.fit_transform(X_list, view_names, y)

    return X_processed, preprocessor


def preprocess_neuroimaging_data(
    X_list: List[np.ndarray],
    view_names: List[str],
    config: Union[PreprocessingConfig, NeuroImagingConfig],
    data_dir: str = None,
    scanner_info: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    validate_inputs: bool = True,
) -> Tuple[List[np.ndarray], Any, Dict[str, Any]]:
    """
    Complete neuroimaging preprocessing pipeline with metadata.

    Returns
    -------
    X_processed : List of preprocessed arrays
    preprocessor : Fitted preprocessor object
    metadata : Dictionary with preprocessing information including spatial processing
    """

    if validate_inputs:
        validate_preprocessing_inputs(X_list, view_names, config)

    # Store original shapes for metadata
    original_shapes = [X.shape for X in X_list]

    # Apply preprocessing
    X_processed, preprocessor = preprocess_data_from_config(
        X_list, view_names, config, data_dir, scanner_info, y
    )

    # Create metadata
    metadata = {
        "config": config.to_dict(),
        "original_shapes": original_shapes,
        "processed_shapes": [X.shape for X in X_processed],
        "feature_reduction": {},
        "preprocessing_type": type(preprocessor).__name__,
        "spatial_processing_applied": isinstance(
            preprocessor, NeuroImagingPreprocessor
        ),
    }

    # Add spatial processing metadata
    if isinstance(preprocessor, NeuroImagingPreprocessor):
        metadata.update(
            {
                "position_lookups_loaded": list(preprocessor.position_lookups_.keys()),
                "outlier_masks": {
                    k: v.sum() for k, v in preprocessor.outlier_masks_.items()
                },
                "harmonization_applied": bool(preprocessor.harmonization_params_),
            }
        )

    # Calculate feature reduction stats
    for i, view_name in enumerate(view_names):
        orig_features = original_shapes[i][1]
        proc_features = X_processed[i].shape[1]
        metadata["feature_reduction"][view_name] = {
            "original": orig_features,
            "processed": proc_features,
            "reduction_ratio": proc_features / max(1, orig_features),
            "reduction_count": orig_features - proc_features,
        }

    return X_processed, preprocessor, metadata


def summarize_preprocessing_results(metadata: Dict[str, Any]) -> str:
    """Create human-readable summary of preprocessing results"""

    summary = ["=== PREPROCESSING SUMMARY ==="]
    summary.append(f"Type: {metadata['preprocessing_type']}")

    if metadata["config"]["enable_preprocessing"]:
        summary.append(f"Imputation: {metadata['config']['imputation_strategy']}")
        summary.append(
            f"Feature selection: {metadata['config']['feature_selection_method']}"
        )

    if metadata.get("spatial_processing_applied", False):
        summary.append("Spatial processing: ENABLED")
        if metadata.get("position_lookups_loaded"):
            summary.append(
                f"Position data loaded for: { ', '.join( metadata['position_lookups_loaded'])}"
            )
        if metadata.get("harmonization_applied"):
            summary.append("Scanner harmonization: APPLIED")

    summary.append("\nFeature reduction by view:")
    for view_name, stats in metadata["feature_reduction"].items():
        reduction_pct = (1 - stats["reduction_ratio"]) * 100
        summary.append(
            f"  {view_name}: {stats['original']} -> {stats['processed']} "
            f"({reduction_pct:.1f}% reduction)"
        )

    return "\n".join(summary)


# == PARAMETER EXTRACTION FROM ARGS ==


def extract_preprocessing_config_from_args(
    args,
) -> Union[PreprocessingConfig, NeuroImagingConfig]:
    """
    Extract preprocessing config from command line args.
    Returns NeuroImagingConfig if spatial processing is enabled.
    """

    # Check if neuroimaging features are requested
    enable_spatial = getattr(args, "enable_spatial_processing", False)

    config_dict = {
        "enable_preprocessing": getattr(args, "enable_preprocessing", False),
        "imputation_strategy": getattr(args, "imputation_strategy", "median"),
        "feature_selection_method": getattr(args, "feature_selection", "variance"),
        "n_top_features": getattr(args, "n_top_features", None),
        "missing_threshold": getattr(args, "missing_threshold", 0.1),
        "variance_threshold": getattr(args, "variance_threshold", 0.0),
        "target_variable": getattr(args, "target_variable", None),
        "cross_validate_sources": getattr(args, "cross_validate_sources", False),
        "optimize_preprocessing": getattr(args, "optimize_preprocessing", False),
    }

    if enable_spatial:
        # Add neuroimaging-specific parameters
        config_dict.update(
            {
                "enable_spatial_processing": True,
                "spatial_imputation": getattr(args, "spatial_imputation", True),
                "roi_based_selection": getattr(args, "roi_based_selection", True),
                "harmonize_scanners": getattr(args, "harmonize_scanners", False),
                "scanner_info_col": getattr(args, "scanner_info_col", None),
                "qc_outlier_threshold": getattr(args, "qc_outlier_threshold", 3.0),
                "spatial_neighbor_radius": getattr(
                    args, "spatial_neighbor_radius", 5.0
                ),
                "min_voxel_distance": getattr(args, "min_voxel_distance", 3.0),
            }
        )
        return NeuroImagingConfig(**config_dict)
    else:
        return PreprocessingConfig(**config_dict)


def create_preprocessor_from_args(
    args, validate: bool = True
) -> Union[PreprocessingConfig, NeuroImagingConfig]:
    """
    Factory function to create preprocessing config from command line args.
    Includes validation and helpful error messages.
    """
    try:
        config = extract_preprocessing_config_from_args(args)
        if validate:
            config.validate()  # This will raise helpful error messages
        return config
    except Exception as e:
        logging.error(f"Failed to create preprocessing config: {e}")
        logging.error("Please check your preprocessing parameters")
        raise


# == CROSS-VALIDATION AND OPTIMIZATION ==


def cross_validate_source_combinations(
    X_list: List[np.ndarray], view_names: List[str], y: np.ndarray, cv_folds: int = 7
) -> Dict[str, float]:
    """
    Cross-validate different combinations of data sources following Bunte et al.
    """
    from itertools import combinations

    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    results = {}

    # Test all combinations of views
    for r in range(1, len(view_names) + 1):
        for combo in combinations(range(len(view_names)), r):
            combo_names = [view_names[i] for i in combo]
            combo_name = ", ".join(combo_names)

            # Concatenate selected views
            X_combo = np.concatenate([X_list[i] for i in combo], axis=1)

            # Cross-validation
            model = Ridge(alpha=1.0)
            cv = (
                StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                if len(np.unique(y)) < 10
                else cv_folds
            )

            scores = []
            for train_idx, test_idx in (
                cv.split(X_combo, y)
                if hasattr(cv, "split")
                else [(None, None)] * cv_folds
            ):
                if train_idx is None:  # Simple k-fold for regression
                    fold_scores = cross_val_score(
                        model,
                        X_combo,
                        y,
                        cv=cv_folds,
                        scoring="neg_mean_squared_error",
                    )
                    rmse_scores = np.sqrt(-fold_scores)
                    scores.extend(rmse_scores)
                    break
                else:
                    X_train, X_test = X_combo[train_idx], X_combo[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    scores.append(rmse)

            results[combo_name] = {
                "rmse_mean": np.mean(scores),
                "rmse_std": np.std(scores),
                "n_features": X_combo.shape[1],
            }

    return results


def optimize_preprocessing_parameters(
    X_list: List[np.ndarray],
    view_names: List[str],
    y: Optional[np.ndarray] = None,
    param_grid: Optional[Dict] = None,
) -> Dict:
    """
    Optimize preprocessing parameters using cross-validation.

    Tests different combinations of:
    - Imputation strategies
    - Feature selection methods
    - Number of top features
    """
    if param_grid is None:
        param_grid = {
            "imputation_strategy": ["median", "mean", "knn"],
            "feature_selection_method": ["variance", "statistical", "combined"],
            "n_top_features": [100, 200, 500, None],
            "variance_threshold": [0.0, 0.01, 0.1],
        }

    from itertools import product

    best_score = float("inf")
    best_params = {}
    results = []

    # Generate parameter combinations
    keys = list(param_grid.keys())
    for values in product(*[param_grid[key] for key in keys]):
        params = dict(zip(keys, values))

        # Skip incompatible combinations
        if params["n_top_features"] is not None and params["variance_threshold"] > 0:
            continue

        try:
            preprocessor = AdvancedPreprocessor(**params)
            X_processed = preprocessor.fit_transform(X_list, view_names, y)

            # Simple evaluation metric
            if y is not None:
                # Use total explained variance as a proxy for quality
                from sklearn.decomposition import PCA

                X_combined = np.concatenate(X_processed, axis=1)
                pca = PCA(
                    n_components=min(10, X_combined.shape[1], X_combined.shape[0])
                )
                pca.fit(X_combined)
                score = (
                    -pca.explained_variance_ratio_.sum()
                )  # Negative for minimization
            else:
                # Use reconstruction error as quality metric
                score = sum([np.mean(X**2) for X in X_processed])

            results.append(
                {
                    "params": params,
                    "score": score,
                    "n_features_total": sum([X.shape[1] for X in X_processed]),
                }
            )

            if score < best_score:
                best_score = score
                best_params = params

        except Exception as e:
            logging.warning(f"Failed with params {params}: {e}")
            continue

    return {
        "best_params": best_params,
        "best_score": best_score,
        "all_results": results,
    }


# == STANDALONE CLI FUNCTIONALITY ==


class PreprocessingInspector:
    """Class for inspecting preprocessing pipeline behavior."""

    def __init__(self, data_dir: str, **load_kwargs):
        self.data_dir = Path(data_dir)
        self.load_kwargs = load_kwargs
        self.raw_data = None
        self.preprocessed_data = None

    def load_raw_data(self):
        """Load raw data for inspection."""
        logging.info("Loading raw data...")

        # Use dynamic import to avoid circular dependency
        try:
            from data.qmap_pd import load_qmap_pd

            self.raw_data = load_qmap_pd(
                data_dir=str(self.data_dir),
                enable_advanced_preprocessing=False,
                **self.load_kwargs,
            )

        except ImportError as e:
            logging.error(f"Could not import data loader: {e}")
            logging.error(
                "Make sure loader_qmap_pd.py is available in the Python path."
            )
            raise ImportError(
                "Data loader not available. The preprocessing inspector requires "
                "loader_qmap_pd.py to be available for loading raw data."
            )

        logging.info(f"Loaded data: {len(self.raw_data['X_list'])} views")
        for i, (view_name, X) in enumerate(
            zip(self.raw_data["view_names"], self.raw_data["X_list"])
        ):
            logging.info(f"  {view_name}: {X.shape} (N={X.shape[0]}, D={X.shape[1]})")

    def inspect_raw_data(self):
        """Inspect characteristics of raw data."""
        if self.raw_data is None:
            self.load_raw_data()

        logging.info("=== RAW DATA INSPECTION ===")

        for view_name, X in zip(self.raw_data["view_names"], self.raw_data["X_list"]):
            logging.info(f"\n--- {view_name.upper()} ---")

            # Basic statistics
            logging.info(f"Shape: {X.shape}")
            logging.info(f"Data type: {X.dtype}")

            # Missing data
            missing_pct = np.isnan(X).sum() / X.size * 100
            logging.info(f"Missing data: {missing_pct:.2f}%")

            if missing_pct > 0:
                missing_per_feature = np.isnan(X).mean(axis=0)
                high_missing = np.sum(missing_per_feature > 0.1)
                logging.info(f"Features with >10% missing: {high_missing}/{X.shape[1]}")

            # Variance statistics
            if not np.any(np.isnan(X)):
                variances = np.var(X, axis=0)
                zero_var_features = np.sum(variances < 1e-10)
                low_var_features = np.sum(variances < 0.01)
                logging.info(
                    f"Zero variance features: {zero_var_features}/{X.shape[1]}"
                )
                logging.info(
                    f"Low variance features (< 0.01): {low_var_features}/{X.shape[1]}"
                )
                logging.info(
                    f"Variance range: [{ np.min(variances):.6f}, { np.max(variances):.2f}]"
                )

            # Value ranges
            if not np.any(np.isnan(X)):
                logging.info(f"Value range: [{np.min(X):.3f}, {np.max(X):.3f}]")
                logging.info(f"Mean absolute value: {np.mean(np.abs(X)):.3f}")

    def create_inspection_plots(self, output_dir: str = "preprocessing_inspection"):
        """Create plots for data inspection."""
        if self.raw_data is None:
            self.load_raw_data()

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        logging.info(f"Creating inspection plots in {output_dir}")

        for view_idx, (view_name, X) in enumerate(
            zip(self.raw_data["view_names"], self.raw_data["X_list"])
        ):

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"{view_name.title()} - Data Inspection", fontsize=16)

            # 1. Missing data pattern
            if np.any(np.isnan(X)):
                missing_pattern = np.isnan(X).astype(int)
                im1 = axes[0, 0].imshow(missing_pattern.T, aspect="auto", cmap="Reds")
                axes[0, 0].set_title("Missing Data Pattern")
                axes[0, 0].set_xlabel("Subjects")
                axes[0, 0].set_ylabel("Features")
                plt.colorbar(im1, ax=axes[0, 0])
            else:
                axes[0, 0].text(
                    0.5,
                    0.5,
                    "No Missing Data",
                    ha="center",
                    va="center",
                    transform=axes[0, 0].transAxes,
                    fontsize=14,
                )
                axes[0, 0].set_title("Missing Data Pattern")

            # 2. Feature variance distribution
            if not np.any(np.isnan(X)):
                variances = np.var(X, axis=0)
                axes[0, 1].hist(
                    np.log10(variances + 1e-10), bins=50, alpha=0.7, color="skyblue"
                )
                axes[0, 1].set_title("Feature Variance Distribution")
                axes[0, 1].set_xlabel("Log10(Variance)")
                axes[0, 1].set_ylabel("Count")
                axes[0, 1].axvline(
                    np.log10(0.01),
                    color="red",
                    linestyle="--",
                    label="Low variance threshold",
                )
                axes[0, 1].legend()

            # 3. Data distribution sample
            sample_features = min(5, X.shape[1])
            feature_indices = np.random.choice(
                X.shape[1], sample_features, replace=False
            )

            for i, feat_idx in enumerate(feature_indices):
                if not np.any(np.isnan(X[:, feat_idx])):
                    axes[1, 0].hist(
                        X[:, feat_idx], bins=30, alpha=0.6, label=f"Feature {feat_idx}"
                    )
            axes[1, 0].set_title("Sample Feature Distributions")
            axes[1, 0].set_xlabel("Value")
            axes[1, 0].set_ylabel("Count")
            axes[1, 0].legend()

            # 4. Correlation matrix sample
            if X.shape[1] > 1 and not np.any(np.isnan(X)):
                sample_size = min(50, X.shape[1])
                sample_indices = np.random.choice(
                    X.shape[1], sample_size, replace=False
                )
                X_sample = X[:, sample_indices]
                corr_matrix = np.corrcoef(X_sample.T)

                im4 = axes[1, 1].imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
                axes[1, 1].set_title("Feature Correlation Sample")
                plt.colorbar(im4, ax=axes[1, 1])

            plt.tight_layout()
            save_plot(output_dir / f"{view_name}_inspection.png", dpi=150)

        logging.info("Inspection plots saved")

    def test_preprocessing_methods(self, methods_to_test: List[str] = None):
        """Test different preprocessing methods and compare results."""
        if self.raw_data is None:
            self.load_raw_data()

        if methods_to_test is None:
            methods_to_test = ["median", "mean", "knn"]

        logging.info("=== TESTING PREPROCESSING METHODS ===")

        results = {}

        for method in methods_to_test:
            logging.info(f"\nTesting imputation method: {method}")

            try:
                # Test preprocessing
                preprocessor = AdvancedPreprocessor(
                    imputation_strategy=method,
                    feature_selection_method="variance",
                    n_top_features=None,
                    variance_threshold=0.01,
                )

                X_processed = preprocessor.fit_transform(
                    self.raw_data["X_list"], self.raw_data["view_names"]
                )

                # Collect results
                method_results = {
                    "processed_shapes": [X.shape for X in X_processed],
                    "feature_reduction": {},
                }

                for i, view_name in enumerate(self.raw_data["view_names"]):
                    original_features = self.raw_data["X_list"][i].shape[1]
                    processed_features = X_processed[i].shape[1]

                    method_results["feature_reduction"][view_name] = {
                        "original": original_features,
                        "processed": processed_features,
                        "reduction_ratio": processed_features / original_features,
                    }

                results[method] = method_results
                logging.info(f"  SUCCESS: {method} completed")

            except Exception as e:
                logging.error(f"  FAILED: {method} failed: {e}")
                results[method] = {"error": str(e)}

        # Print comparison
        logging.info("\n=== PREPROCESSING METHOD COMPARISON ===")

        for view_name in self.raw_data["view_names"]:
            logging.info(f"\n--- {view_name} ---")
            logging.info("Method\t\tOriginal\tProcessed\tRetention")

            for method, result in results.items():
                if "error" not in result:
                    view_result = result["feature_reduction"][view_name]
                    logging.info(
                        f"{ method:12s}\t{ view_result['original']:8d}\t{ view_result['processed']:8d}\t{ view_result['reduction_ratio']:8.2%}"
                    )
                else:
                    logging.info(f"{method:12s}\tFAILED")

        return results

    def optimize_preprocessing_pipeline(self, target_variable: str = None):
        """Test preprocessing parameter optimization."""
        if self.raw_data is None:
            self.load_raw_data()

        logging.info("=== OPTIMIZING PREPROCESSING PARAMETERS ===")

        # Extract target if specified
        y = None
        if target_variable and "clinical" in self.raw_data:
            clinical_df = self.raw_data["clinical"]
            if target_variable in clinical_df.columns:
                y = clinical_df[target_variable].values
                logging.info(f"Using {target_variable} as target for optimization")
            else:
                logging.warning(
                    f"Target variable {target_variable} not found in clinical data"
                )
                logging.info(f"Available columns: {list(clinical_df.columns)}")

        # Run optimization
        try:
            optimization_results = optimize_preprocessing_parameters(
                self.raw_data["X_list"], self.raw_data["view_names"], y
            )

            logging.info("Optimization completed!")
            logging.info(f"Best parameters: {optimization_results['best_params']}")
            logging.info(f"Best score: {optimization_results['best_score']:.4f}")

            # Show top 5 parameter combinations
            sorted_results = sorted(
                optimization_results["all_results"], key=lambda x: x["score"]
            )[:5]

            logging.info("\nTop 5 parameter combinations:")
            for i, result in enumerate(sorted_results, 1):
                logging.info(f"{i}. Score: {result['score']:.4f}")
                logging.info(f"   Params: {result['params']}")
                logging.info(f"   Features: {result['n_features_total']}")

            return optimization_results

        except Exception as e:
            logging.error(f"Optimization failed: {e}")
            return None


# == SPATIAL REMAPPING UTILITIES ==


def get_selected_voxel_positions(
    preprocessor,
    view_name: str,
    data_dir: str,
    roi_name: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Get spatial positions for selected voxels after feature selection.

    This function maps selected voxel indices back to their spatial coordinates,
    enabling reconstruction of brain maps from factor loadings.

    Parameters
    ----------
    preprocessor : NeuroImagingPreprocessor or AdvancedPreprocessor
        Fitted preprocessor containing selected feature indices
    view_name : str
        Name of the imaging view (e.g., 'volume_sn_voxels')
    data_dir : str
        Path to data directory containing position_lookup files
    roi_name : str, optional
        ROI name (e.g., 'sn', 'putamen'). If None, extracted from view_name

    Returns
    -------
    pd.DataFrame or None
        DataFrame with spatial positions for selected voxels
        Columns typically: ['x', 'y', 'z'] or voxel indices
        Rows correspond to selected voxels in same order as factor loadings

    Examples
    --------
    >>> # After running SGFA with feature selection
    >>> results = run_sgfa(X_list, ...)
    >>> preprocessor = results['preprocessor']
    >>>
    >>> # Get positions for selected voxels
    >>> positions = get_selected_voxel_positions(
    ...     preprocessor,
    ...     view_name='volume_sn_voxels',
    ...     data_dir='qMAP-PD_data'
    ... )
    >>>
    >>> # Map factor loadings to brain space
    >>> W_sn = results['W'][sn_view_idx]  # Shape: (n_selected_voxels, K)
    >>> for k in range(K):
    ...     factor_k = W_sn[:, k]
    ...     brain_map = pd.DataFrame({
    ...         'x': positions['x'],
    ...         'y': positions['y'],
    ...         'z': positions['z'],
    ...         'loading': factor_k
    ...     })
    """
    # Extract ROI name if not provided
    if roi_name is None:
        roi_name = view_name.lower().replace('volume_', '').replace('_voxels', '')

    # Load full position lookup
    positions_all = SpatialProcessingUtils.load_position_lookup(data_dir, roi_name)

    if positions_all is None:
        logging.warning(f"No position lookup found for {roi_name}")
        return None

    # Get selected feature indices
    selected_indices = None

    if hasattr(preprocessor, 'selected_features_'):
        # Check various possible index keys
        possible_keys = [
            f"{view_name}_variance_indices",
            f"{view_name}_roi_indices",
            f"{view_name}_corr_indices",
            f"{view_name}_temp_variance_indices",  # From combined selection
        ]

        for key in possible_keys:
            if key in preprocessor.selected_features_:
                selected_indices = preprocessor.selected_features_[key]
                logging.info(f"Found selected indices under key: {key}")
                break

    if selected_indices is None:
        logging.warning(f"No selected feature indices found for {view_name}")
        logging.info("Returning full position lookup (no feature selection applied)")
        return positions_all

    # Index positions by selected voxels
    positions_selected = positions_all.iloc[selected_indices].reset_index(drop=True)

    logging.info(
        f"Spatial remapping: {len(positions_all)} total voxels → "
        f"{len(positions_selected)} selected voxels"
    )

    return positions_selected


def create_brain_map_from_factors(
    factor_loadings: np.ndarray,
    positions: pd.DataFrame,
    factor_index: Optional[int] = None
) -> pd.DataFrame:
    """
    Create spatial brain map from factor loadings and voxel positions.

    Parameters
    ----------
    factor_loadings : np.ndarray
        Factor loadings for imaging view, shape (n_voxels, n_factors)
        or (n_voxels,) for a single factor
    positions : pd.DataFrame
        Spatial positions for selected voxels, shape (n_voxels, 3+)
        Must have columns for spatial coordinates (e.g., 'x', 'y', 'z')
    factor_index : int, optional
        If factor_loadings is 2D, which factor to map (column index)
        If None and 2D, uses first factor

    Returns
    -------
    pd.DataFrame
        Brain map with columns: ['x', 'y', 'z', 'loading']
        Can be used for visualization or saved for further analysis

    Examples
    --------
    >>> # Single factor
    >>> W_sn = results['W'][0]  # Shape: (687, 3)
    >>> positions = get_selected_voxel_positions(...)
    >>>
    >>> brain_map = create_brain_map_from_factors(
    ...     W_sn, positions, factor_index=0
    ... )
    >>>
    >>> # Visualize or save
    >>> brain_map.to_csv('factor_0_brain_map.csv', index=False)
    """
    # Handle 1D or 2D factor loadings
    if factor_loadings.ndim == 2:
        if factor_index is None:
            factor_index = 0
            logging.info(f"Using factor {factor_index} (default)")
        loadings = factor_loadings[:, factor_index]
    else:
        loadings = factor_loadings

    # Validate dimensions
    if len(loadings) != len(positions):
        raise ValueError(
            f"Dimension mismatch: {len(loadings)} loadings vs "
            f"{len(positions)} positions"
        )

    # Create brain map
    brain_map = positions.copy()
    brain_map['loading'] = loadings

    # Sort by loading magnitude for easier inspection
    brain_map['abs_loading'] = np.abs(loadings)
    brain_map = brain_map.sort_values('abs_loading', ascending=False)
    brain_map = brain_map.drop(columns=['abs_loading'])

    return brain_map


def remap_all_factors_to_brain(
    W_list: List[np.ndarray],
    view_names: List[str],
    preprocessor,
    data_dir: str
) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    Remap all factor loadings for all imaging views to brain space.

    Parameters
    ----------
    W_list : List[np.ndarray]
        List of factor loading matrices, one per view
        Each W has shape (n_features_in_view, n_factors)
    view_names : List[str]
        Names of views corresponding to W_list
    preprocessor : NeuroImagingPreprocessor
        Fitted preprocessor with selected feature indices
    data_dir : str
        Path to data directory containing position lookups

    Returns
    -------
    Dict[str, Dict[int, pd.DataFrame]]
        Nested dictionary: {view_name: {factor_k: brain_map_df}}
        Only includes imaging views (clinical views skipped)

    Examples
    --------
    >>> results = run_sgfa(X_list, ...)
    >>>
    >>> brain_maps = remap_all_factors_to_brain(
    ...     W_list=results['W'],
    ...     view_names=results['view_names'],
    ...     preprocessor=results['preprocessor'],
    ...     data_dir='qMAP-PD_data'
    ... )
    >>>
    >>> # Access specific factor in specific ROI
    >>> sn_factor_0 = brain_maps['volume_sn_voxels'][0]
    >>> sn_factor_0.to_csv('sn_factor0_map.csv')
    >>>
    >>> # Plot all factors for an ROI
    >>> for k, brain_map in brain_maps['volume_sn_voxels'].items():
    ...     plot_brain_map(brain_map, title=f'SN Factor {k}')
    """
    all_brain_maps = {}

    for view_idx, (W, view_name) in enumerate(zip(W_list, view_names)):
        # Skip clinical views
        if 'clinical' in view_name.lower():
            logging.info(f"Skipping clinical view: {view_name}")
            continue

        logging.info(f"Remapping factors for {view_name}...")

        # Get positions for this view
        positions = get_selected_voxel_positions(
            preprocessor, view_name, data_dir
        )

        if positions is None:
            logging.warning(f"Could not get positions for {view_name}, skipping")
            continue

        # Create brain map for each factor
        view_brain_maps = {}
        n_factors = W.shape[1]

        for k in range(n_factors):
            brain_map = create_brain_map_from_factors(
                W, positions, factor_index=k
            )
            view_brain_maps[k] = brain_map

        all_brain_maps[view_name] = view_brain_maps
        logging.info(f"  → Created {n_factors} brain maps for {view_name}")

    return all_brain_maps


# == CLI MAIN FUNCTION ==


def main():
    """Main function for standalone CLI usage."""
    parser = argparse.ArgumentParser(
        description="Preprocessing Module - Standalone Mode"
    )

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="qMAP-PD_data")
    parser.add_argument(
        "--clinical_rel", type=str, default="data_clinical/pd_motor_gfa_data.tsv"
    )
    parser.add_argument("--volumes_rel", type=str, default="volume_matrices")
    parser.add_argument("--id_col", type=str, default="sid")
    parser.add_argument("--roi_views", action="store_true")

    # Inspection modes
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Only inspect raw data without preprocessing",
    )
    parser.add_argument(
        "--create_plots", action="store_true", help="Create inspection plots"
    )
    parser.add_argument(
        "--compare_methods",
        action="store_true",
        help="Compare different preprocessing methods",
    )
    parser.add_argument(
        "--optimize_params",
        action="store_true",
        help="Optimize preprocessing parameters",
    )

    # Preprocessing parameters (for testing specific configurations)
    parser.add_argument(
        "--imputation_strategy",
        type=str,
        default="median",
        choices=["median", "mean", "knn", "iterative"],
    )
    parser.add_argument(
        "--feature_selection",
        type=str,
        default="variance",
        choices=["variance", "statistical", "mutual_info", "combined"],
    )
    parser.add_argument("--n_top_features", type=int, default=None)
    parser.add_argument("--target_variable", type=str, default=None)

    # Neuroimaging-specific parameters
    parser.add_argument(
        "--enable_spatial_processing",
        action="store_true",
        help="Enable spatial processing for neuroimaging data",
    )
    parser.add_argument(
        "--spatial_imputation",
        action="store_true",
        help="Use spatial neighbors for imputation",
    )
    parser.add_argument(
        "--roi_based_selection",
        action="store_true",
        help="Use ROI-based feature selection",
    )
    parser.add_argument(
        "--harmonize_scanners", action="store_true", help="Apply scanner harmonization"
    )

    # Output
    parser.add_argument("--output_dir", type=str, default="preprocessing_inspection")

    args = parser.parse_args()

    # Check dependencies
    check_preprocessing_compatibility()

    # Create inspector
    inspector = PreprocessingInspector(
        data_dir=args.data_dir,
        clinical_rel=args.clinical_rel,
        volumes_rel=args.volumes_rel,
        imaging_as_single_view=not args.roi_views,
        id_col=args.id_col,
    )

    # Run requested analyses
    if args.inspect_only:
        inspector.inspect_raw_data()

        if args.create_plots:
            inspector.create_inspection_plots(args.output_dir)

    elif args.compare_methods:
        inspector.inspect_raw_data()
        inspector.test_preprocessing_methods()

        if args.create_plots:
            inspector.create_inspection_plots(args.output_dir)

    elif args.optimize_params:
        inspector.inspect_raw_data()
        inspector.optimize_preprocessing_pipeline(args.target_variable)

    else:
        # Test specific preprocessing configuration
        logging.info("=== TESTING SPECIFIC PREPROCESSING CONFIGURATION ===")
        inspector.inspect_raw_data()

        # Create config
        config = NeuroImagingConfig(
            enable_preprocessing=True,
            enable_spatial_processing=args.enable_spatial_processing,
            imputation_strategy=args.imputation_strategy,
            feature_selection_method=args.feature_selection,
            n_top_features=args.n_top_features,
            spatial_imputation=args.spatial_imputation,
            roi_based_selection=args.roi_based_selection,
            harmonize_scanners=args.harmonize_scanners,
        )

        # Extract target if specified
        y = None
        if args.target_variable and "clinical" in inspector.raw_data:
            clinical_df = inspector.raw_data["clinical"]
            if args.target_variable in clinical_df.columns:
                y = clinical_df[args.target_variable].values
                logging.info(f"Using {args.target_variable} as target")

        # Process data
        X_processed, preprocessor, metadata = preprocess_neuroimaging_data(
            inspector.raw_data["X_list"],
            inspector.raw_data["view_names"],
            config,
            data_dir=args.data_dir,
            y=y,
        )

        # Show results
        logging.info("\n" + summarize_preprocessing_results(metadata))

        if args.create_plots:
            inspector.create_inspection_plots(args.output_dir)


# == SCRIPT ENTRY POINT ==

if __name__ == "__main__":
    # Runs when the script is executed directly
    main()
else:
    # Runs when the module is imported
    logging.info("Preprocessing module imported successfully")
    check_preprocessing_compatibility()
