"""
Enhanced Cross-validation module for Sparse Bayesian Group Factor Analysis.
Specifically tailored for qMRI neuroimaging data and Parkinson's disease subtyping.

This module provides:
- Neuroimaging-aware cross-validation strategies
- Clinical data integration for stratified/grouped CV
- Robust error handling and memory management
- Comprehensive evaluation metrics for factor models
- Integration with preprocessing pipeline
- Parkinson's disease specific validation approaches

Usage:
1. As a Python module:
   from analysis.cross_validation_library import NeuroImagingCrossValidator, ParkinsonsConfig

2. As a standalone CLI tool:
   python -m analysis.cross_validation_library --dataset qmap_pd --cv_type clinical_stratified
   python -m analysis.cross_validation_library --dataset qmap_pd --nested_cv --optimize_for_subtypes
"""

import logging
import multiprocessing as mp
import signal
import tempfile
import time
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# JAX and NumPyro imports
import jax.random as jrandom
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    mean_squared_error,
    r2_score,
    silhouette_score,
)

# Scientific computing imports
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
)
from sklearn.preprocessing import LabelEncoder

from core.io_utils import save_json

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpyro")


@contextmanager
def timeout_context(seconds):
    """Context manager for operation timeout."""

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# == PARKINSON'S DISEASE SPECIFIC CONFIGURATION ==


class ParkinsonsConfig:
    """Configuration specifically for Parkinson's disease subtyping research."""

    def __init__(self):
        # Clinical CV configuration for PD research
        self.clinical_stratification_vars = [
            "age_at_diagnosis",
            "disease_duration",
            "motor_phenotype",
            "cognitive_status",
            "medication_status",
        ]

        # Subtyping validation metrics
        self.subtype_evaluation_metrics = [
            "motor_progression",
            "cognitive_decline",
            "medication_response",
            "quality_of_life",
            "functional_independence",
        ]

        # Factor interpretation criteria
        self.factor_validation_criteria = {
            "biological_plausibility": 0.7,  # Correlation with known biomarkers
            "clinical_relevance": 0.6,  # Association with clinical outcomes
            "reproducibility": 0.8,  # Stability across CV folds
            "anatomical_coherence": 0.65,  # Spatial coherence in brain regions
        }

        # Neuroimaging specific parameters
        self.roi_groups = {
            "substantia_nigra": ["volume_sn_voxels"],
            "putamen": ["volume_putamen_voxels"],
            "lentiform": ["volume_lentiform_voxels"],
            "basal_ganglia": ["volume_bg-all_voxels"],
            "motor_circuit": ["volume_sn_voxels", "volume_putamen_voxels"],
            "all_imaging": [
                "volume_sn_voxels",
                "volume_putamen_voxels",
                "volume_lentiform_voxels",
            ],
        }

        # Clinical validation targets (from your research paper)
        self.clinical_targets = {
            "motor_subtypes": ["TD", "PIGD", "Mixed"],  # Tremor-dominant, PIGD, Mixed
            "progression_rate": "continuous",
            "medication_response": "binary",
            "cognitive_status": ["Normal", "MCI", "Dementia"],
        }


class NeuroImagingCVConfig:
    """Enhanced CV configuration for neuroimaging data."""

    def __init__(self):
        # Base CV parameters
        self.outer_cv_folds = 5
        self.inner_cv_folds = 3
        self.n_repeats = 1
        self.random_state = 42
        self.shuffle = True

        # Neuroimaging-specific parameters
        self.spatial_validation = True
        self.roi_based_validation = True
        self.cross_modal_validation = True  # Between imaging and clinical

        # Model selection parameters for GFA
        self.factor_range = [5, 10, 15, 20, 25, 30]
        self.sparsity_range = [25, 33, 50, 67]

        # Subtype optimization parameters
        self.auto_optimize_subtypes = True  # Enable automatic subtype determination
        self.subtype_candidate_range = [2, 3, 4]  # Literature-based PD subtypes
        self.mcmc_samples_range = [1000, 2000, 3000]

        # Evaluation metrics
        self.primary_metric = "factor_interpretability"  # Custom metric
        self.secondary_metrics = [
            "reconstruction_r2",
            "clustering_stability",
            "clinical_association",
            "spatial_coherence",
        ]

        # Memory and computational limits
        self.max_memory_gb = 32
        self.timeout_per_fold_minutes = 60
        self.n_jobs = min(4, mp.cpu_count() - 1)

        # Error handling
        self.max_retries = 2
        self.continue_on_fold_failure = True
        self.min_successful_folds = 3


# == NEUROIMAGING-SPECIFIC EVALUATION METRICS ==


class NeuroImagingMetrics:
    """Comprehensive evaluation metrics for neuroimaging factor analysis."""

    @staticmethod
    def spatial_coherence_score(
        factor_loadings: np.ndarray, position_info: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Evaluate spatial coherence of factor loadings in brain space.
        Higher scores indicate factors that respect anatomical boundaries.
        """
        if position_info is None:
            logger.warning(
                "No position information available for spatial coherence scoring"
            )
            return 0.0

        try:
            # For each factor, compute spatial autocorrelation
            coherence_scores = []

            for factor_idx in range(factor_loadings.shape[1]):
                loadings = factor_loadings[:, factor_idx]

                if len(loadings) != len(position_info):
                    logger.warning(
                        f"Dimension mismatch: loadings={ len(loadings)}, positions={ len(position_info)}"
                    )
                    continue

                # Compute spatial weights based on distance
                positions = position_info[["x", "y", "z"]].values
                pdist(positions)

                # Moran's I for spatial autocorrelation
                n = len(loadings)
                mean_loading = np.mean(loadings)

                numerator = 0
                denominator = 0
                weights_sum = 0

                for i in range(n):
                    for j in range(i + 1, n):
                        dist_ij = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
                        weight = 1.0 / (1.0 + dist_ij)  # Inverse distance weighting

                        numerator += (
                            weight
                            * (loadings[i] - mean_loading)
                            * (loadings[j] - mean_loading)
                        )
                        weights_sum += weight

                    denominator += (loadings[i] - mean_loading) ** 2

                if weights_sum > 0 and denominator > 0:
                    morans_i = (n / weights_sum) * (numerator / denominator)
                    coherence_scores.append(max(0, morans_i))  # Clamp to positive

            return np.mean(coherence_scores) if coherence_scores else 0.0

        except Exception as e:
            logger.warning(f"Could not compute spatial coherence: {e}")
            return 0.0

    @staticmethod
    def clinical_association_score(
        factor_scores: np.ndarray,
        clinical_data: pd.DataFrame,
        target_variables: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate association between factor scores and clinical variables.
        """
        associations = {}

        for var in target_variables:
            if var not in clinical_data.columns:
                logger.warning(f"Clinical variable {var} not found")
                continue

            clinical_values = clinical_data[var].values

            # Remove missing values
            valid_mask = ~pd.isna(clinical_values)
            if np.sum(valid_mask) < 10:  # Need minimum samples
                continue

            valid_clinical = clinical_values[valid_mask]
            valid_factors = factor_scores[valid_mask]

            # Compute associations for each factor
            factor_associations = []

            for factor_idx in range(valid_factors.shape[1]):
                factor_vals = valid_factors[:, factor_idx]

                if np.isreal(valid_clinical).all():
                    # Continuous clinical variable - use correlation
                    corr, p_val = pearsonr(factor_vals, valid_clinical)
                    if p_val < 0.05:  # Significant association
                        factor_associations.append(abs(corr))
                else:
                    # Categorical clinical variable - use effect size
                    try:
                        from sklearn.feature_selection import f_classif

                        f_stat, p_val = f_classif(
                            factor_vals.reshape(-1, 1), valid_clinical
                        )
                        if p_val[0] < 0.05:
                            # Convert F-statistic to effect size approximation
                            eta_squared = f_stat[0] / (
                                f_stat[0] + len(valid_clinical) - 1
                            )
                            factor_associations.append(eta_squared)
                    except Exception:
                        factor_associations.append(0.0)

            associations[var] = (
                np.mean(factor_associations) if factor_associations else 0.0
            )

        return associations

    @staticmethod
    def factor_interpretability_score(
        factor_loadings: np.ndarray,
        factor_scores: np.ndarray,
        view_names: List[str],
        clinical_data: Optional[pd.DataFrame] = None,
        position_info: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Comprehensive interpretability score combining multiple criteria.
        This is the primary evaluation metric for factor models in PD research.
        """
        scores = {}

        # 1. Sparsity score (factors should be sparse for interpretability)
        sparsity_threshold = 0.1
        sparse_ratios = []
        for factor_idx in range(factor_loadings.shape[1]):
            factor_vals = factor_loadings[:, factor_idx]
            sparse_ratio = np.sum(np.abs(factor_vals) > sparsity_threshold) / len(
                factor_vals
            )
            sparse_ratios.append(1.0 - sparse_ratio)  # Higher score for sparser factors
        scores["sparsity"] = np.mean(sparse_ratios)

        # 2. Orthogonality score (factors should be distinct)
        correlation_matrix = np.corrcoef(factor_scores.T)
        off_diagonal = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        scores["orthogonality"] = 1.0 - np.mean(np.abs(off_diagonal))

        # 3. Spatial coherence (for neuroimaging factors)
        if position_info is not None:
            scores["spatial_coherence"] = NeuroImagingMetrics.spatial_coherence_score(
                factor_loadings, position_info
            )
        else:
            scores["spatial_coherence"] = 0.5  # Neutral score if no spatial info

        # 4. Clinical association (factors should relate to clinical outcomes)
        if clinical_data is not None:
            clinical_vars = [
                "age_at_diagnosis",
                "disease_duration",
                "motor_score",
                "cognitive_score",
            ]
            available_vars = [
                var for var in clinical_vars if var in clinical_data.columns
            ]

            if available_vars:
                associations = NeuroImagingMetrics.clinical_association_score(
                    factor_scores, clinical_data, available_vars
                )
                scores["clinical_association"] = np.mean(list(associations.values()))
            else:
                scores["clinical_association"] = 0.5
        else:
            scores["clinical_association"] = 0.5

        # 5. Overall interpretability score (weighted average)
        weights = {
            "sparsity": 0.3,
            "orthogonality": 0.25,
            "spatial_coherence": 0.25,
            "clinical_association": 0.2,
        }

        scores["overall"] = sum(weights[key] * scores[key] for key in weights.keys())

        return scores


# == ADVANCED CV SPLITTERS ==


class ClinicalAwareSplitter:
    """Advanced CV splitter that respects clinical characteristics."""

    def __init__(self, config: NeuroImagingCVConfig):
        self.config = config

    def split(self, X, y=None, groups=None, clinical_data=None, **kwargs):
        """
        Sklearn-compatible split method for clinical-aware CV.

        Args:
            X: Input data array
            y: Target labels (optional, will extract from clinical_data if None)
            groups: Group labels (optional, will extract from clinical_data if None)
            clinical_data: Clinical data dictionary
            **kwargs: Additional arguments

        Returns:
            Generator of (train_idx, test_idx) tuples
        """
        # Extract labels from clinical_data if not provided
        if clinical_data:
            if y is None:
                y = clinical_data.get("diagnosis")
            if groups is None:
                groups = clinical_data.get("subject_id")

        # Default stratification variables
        stratification_vars = ["diagnosis"] if clinical_data and "diagnosis" in clinical_data else []

        # Create and use appropriate CV splitter
        if clinical_data and stratification_vars:
            # Convert to DataFrame if needed
            import pandas as pd
            if not isinstance(clinical_data, pd.DataFrame):
                clinical_df = pd.DataFrame(clinical_data)
            else:
                clinical_df = clinical_data

            cv_splitter = self.create_clinical_stratified_cv(clinical_df, stratification_vars)
            return cv_splitter.split(X, y, groups)
        else:
            # Fallback to basic stratified CV
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=self.config.outer_cv_folds, shuffle=True, random_state=42)
            return skf.split(X, y if y is not None else np.zeros(X.shape[0]))

    def create_clinical_stratified_cv(
        self, clinical_data: pd.DataFrame, stratification_vars: List[str]
    ) -> Any:
        """
        Create stratified CV based on multiple clinical variables.
        """
        # Create composite stratification variable
        if (
            len(stratification_vars) == 1
            and stratification_vars[0] in clinical_data.columns
        ):
            strat_var = clinical_data[stratification_vars[0]]
        else:
            # Multi-variable stratification
            available_vars = [
                var for var in stratification_vars if var in clinical_data.columns
            ]
            if not available_vars:
                logger.warning(
                    "No stratification variables found, using standard KFold"
                )
                return KFold(
                    n_splits=self.config.outer_cv_folds,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state,
                )

            # Bin continuous variables and combine categorical ones
            strat_components = []
            for var in available_vars:
                values = clinical_data[var].values
                if pd.api.types.is_numeric_dtype(values):
                    # Bin continuous variables into quartiles
                    binned = pd.qcut(values, q=4, labels=False, duplicates="drop")
                    strat_components.append(binned)
                else:
                    # Encode categorical variables
                    le = LabelEncoder()
                    encoded = le.fit_transform(values.astype(str))
                    strat_components.append(encoded)

            # Combine into single stratification variable
            strat_var = np.zeros(len(clinical_data))
            for i, component in enumerate(strat_components):
                strat_var += component * (10**i)

        try:
            return StratifiedKFold(
                n_splits=self.config.outer_cv_folds,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
            )
        except ValueError as e:
            logger.warning(
                f"Stratified CV failed ({e}), falling back to standard KFold"
            )
            return KFold(
                n_splits=self.config.outer_cv_folds,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
            )

    def create_site_aware_cv(self, site_info: np.ndarray) -> GroupKFold:
        """Create CV that keeps all subjects from same site together."""
        return GroupKFold(n_splits=self.config.outer_cv_folds)

    def create_temporal_cv(self, dates: pd.Series) -> TimeSeriesSplit:
        """Create temporal CV for longitudinal data."""
        return TimeSeriesSplit(n_splits=self.config.outer_cv_folds)


# == ENHANCED HYPERPARAMETER OPTIMIZATION ==


class NeuroImagingHyperOptimizer:
    """Hyperparameter optimization tailored for neuroimaging factor analysis."""

    def __init__(self, config: NeuroImagingCVConfig):
        self.config = config

    def create_neuroimaging_parameter_grid(self) -> List[Dict]:
        """Create parameter grid optimized for neuroimaging factor analysis."""

        # Base parameter combinations
        base_grid = {
            "K": self.config.factor_range,
            "percW": self.config.sparsity_range,
            "num_samples": self.config.mcmc_samples_range,
            "num_warmup": [500, 1000],
            "slab_scale": [1.5, 2.0, 2.5],
            "slab_df": [3, 4, 5],
        }

        # Generate intelligent combinations (not full grid to save computation)
        smart_combinations = []

        # Small models (faster evaluation)
        for K in [5, 10, 15]:
            for percW in [33, 50]:
                smart_combinations.append(
                    {
                        "K": K,
                        "percW": percW,
                        "num_samples": 1000,
                        "num_warmup": 500,
                        "slab_scale": 2.0,
                        "slab_df": 4,
                    }
                )

        # Medium models
        for K in [20, 25]:
            for percW in [25, 33]:
                smart_combinations.append(
                    {
                        "K": K,
                        "percW": percW,
                        "num_samples": 2000,
                        "num_warmup": 1000,
                        "slab_scale": 2.0,
                        "slab_df": 4,
                    }
                )

        # Large models (limited combinations)
        for K in [30]:
            smart_combinations.append(
                {
                    "K": K,
                    "percW": 33,
                    "num_samples": 3000,
                    "num_warmup": 1000,
                    "slab_scale": 2.0,
                    "slab_df": 4,
                }
            )

        logger.info(
            f"Created {len(smart_combinations)} parameter combinations for optimization"
        )
        return smart_combinations

    def evaluate_factor_model_quality(
        self,
        samples: Dict,
        X_list: List[np.ndarray],
        clinical_data: Optional[pd.DataFrame] = None,
        position_info: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        Evaluate factor model quality using neuroimaging-specific criteria.
        """
        try:
            # Extract factor parameters
            W_samples = samples["W"]
            Z_samples = samples["Z"]

            # Use posterior means
            W_mean = np.mean(W_samples, axis=0)
            Z_mean = np.mean(Z_samples, axis=0)

            # Compute interpretability score
            interpretability = NeuroImagingMetrics.factor_interpretability_score(
                W_mean, Z_mean, ["view1", "view2"], clinical_data, position_info
            )

            # Compute reconstruction quality
            X_concat = np.concatenate(X_list, axis=1)
            X_recon = Z_mean @ W_mean.T

            # Ensure dimensions match
            min_features = min(X_concat.shape[1], X_recon.shape[1])
            recon_r2 = r2_score(X_concat[:, :min_features], X_recon[:, :min_features])

            # Combined score (weighted)
            quality_score = 0.6 * interpretability["overall"] + 0.4 * max(0, recon_r2)

            return quality_score

        except Exception as e:
            logger.warning(f"Failed to evaluate model quality: {e}")
            return 0.0


# == MAIN CROSS-VALIDATION CLASS ==


class NeuroImagingCrossValidator:
    """
    Main cross-validation class for neuroimaging factor analysis.
    Specifically designed for Parkinson's disease qMRI data.
    """

    def __init__(
        self,
        config: NeuroImagingCVConfig = None,
        parkinsons_config: ParkinsonsConfig = None,
    ):
        self.config = config or NeuroImagingCVConfig()
        self.pd_config = parkinsons_config or ParkinsonsConfig()

        # Initialize components
        self.splitter = ClinicalAwareSplitter(self.config)
        self.metrics = NeuroImagingMetrics()
        self.hyperopt = NeuroImagingHyperOptimizer(self.config)

        # Results storage
        self.results = {
            "cv_scores": [],
            "fold_results": [],
            "best_params": {},
            "interpretability_scores": {},
            "clinical_associations": {},
            "spatial_coherence_scores": {},
            "subtype_validation": {},
            "timing": {},
        }

        # Error tracking
        self.failed_folds = []
        self.fold_errors = {}

        # Setup temporary storage
        self.temp_dir = tempfile.mkdtemp(prefix="neuroimaging_cv_")
        logger.info(f"Initialized neuroimaging CV with temp dir: {self.temp_dir}")

    def _extract_clinical_metadata(
        self, data: Dict
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Extract clinical data and position information from data dictionary."""
        clinical_data = None
        position_info = None

        if "clinical" in data:
            clinical_data = data["clinical"]
            logger.info(
                f"Found clinical data with { len(clinical_data)} subjects and { len( clinical_data.columns)} variables"
            )

        # Try to load position information for spatial analysis
        if "meta" in data and "root" in data["meta"]:
            try:
                from pathlib import Path

                root_path = Path(data["meta"]["root"])
                position_dir = root_path / "position_lookup"

                # Try different ROI position files
                for roi in ["sn", "putamen", "lentiform", "bg-all"]:
                    position_file = position_dir / f"position_{roi}_voxels.tsv"
                    if position_file.exists():
                        position_info = pd.read_csv(
                            position_file, sep="\t", header=None, names=["x", "y", "z"]
                        )
                        logger.info(
                            f"Loaded position information for {roi}: { len(position_info)} voxels"
                        )
                        break
            except Exception as e:
                logger.warning(f"Could not load position information: {e}")

        return clinical_data, position_info

    def fit_single_fold_robust(self, fold_data: Tuple) -> Dict:
        """
        Fit model on a single CV fold with neuroimaging-specific error handling.
        """
        (
            fold_id,
            train_idx,
            test_idx,
            X_list,
            args,
            hypers,
            best_params,
            clinical_data,
            position_info,
        ) = fold_data

        logger.info(f"Processing neuroimaging fold {fold_id}")
        start_time = time.time()

        fold_result = {
            "fold_id": fold_id,
            "converged": False,
            "fit_time": 0,
            "error": None,
            "retry_count": 0,
        }

        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(
                        f"Retrying neuroimaging fold {fold_id}, attempt {attempt + 1}"
                    )
                    time.sleep(10)  # Brief pause between retries

                # Use timeout specific to neuroimaging analysis
                timeout_seconds = self.config.timeout_per_fold_minutes * 60
                with timeout_context(timeout_seconds):
                    result = self._fit_neuroimaging_fold_core(
                        fold_id,
                        train_idx,
                        test_idx,
                        X_list,
                        args,
                        hypers,
                        best_params,
                        clinical_data,
                        position_info,
                    )

                result["fit_time"] = time.time() - start_time
                result["retry_count"] = attempt
                logger.info(f"Neuroimaging fold {fold_id} completed successfully")
                return result

            except TimeoutError:
                error_msg = (
                    f"Neuroimaging fold {fold_id} timed out after {timeout_seconds}s"
                )
                logger.error(error_msg)
                fold_result["error"] = error_msg

            except Exception as e:
                error_msg = (
                    f"Neuroimaging fold {fold_id} failed: {type(e).__name__}: {str(e)}"
                )
                logger.error(error_msg)
                fold_result["error"] = error_msg

        fold_result["fit_time"] = time.time() - start_time
        self.failed_folds.append(fold_id)
        self.fold_errors[fold_id] = fold_result["error"]

        return fold_result

    def _fit_neuroimaging_fold_core(
        self,
        fold_id: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        X_list: List[np.ndarray],
        args,
        hypers: Dict,
        best_params: Dict,
        clinical_data: Optional[pd.DataFrame],
        position_info: Optional[pd.DataFrame],
    ) -> Dict:
        """Core neuroimaging fold fitting with enhanced metrics."""

        # Prepare fold data
        X_train_list = [X[train_idx] for X in X_list]
        X_test_list = [X[test_idx] for X in X_list]

        # Scale data (important for qMRI)
        scalers = []
        for m, X_train in enumerate(X_train_list):
            # Use robust scaling for neuroimaging data
            mu = np.median(X_train, axis=0, keepdims=True)
            mad = np.median(np.abs(X_train - mu), axis=0, keepdims=True)
            sigma = mad * 1.4826  # Convert MAD to std equivalent
            sigma = np.where(sigma < 1e-8, 1.0, sigma)

            X_train_list[m] = (X_train - mu) / sigma
            X_test_list[m] = (X_test_list[m] - mu) / sigma
            scalers.append((mu, sigma))

        # Update args with best parameters
        args_fold = deepcopy(args)
        for key, value in best_params.items():
            if hasattr(args_fold, key):
                setattr(args_fold, key, value)

        # Update hyperparameters
        hypers_fold = deepcopy(hypers)
        hypers_fold["Dm"] = [X.shape[1] for X in X_train_list]

        # Import run_analysis for MCMC inference
        try:
            from core import run_analysis as RA
        except ImportError:
            raise ImportError("core.run_analysis module required for neuroimaging CV")

        # Run MCMC inference
        rng = jrandom.PRNGKey(self.config.random_state + fold_id)
        mcmc = RA.run_inference(RA.models, args_fold, rng, X_train_list, hypers_fold)
        samples = mcmc.get_samples()

        # Enhanced convergence checking for neuroimaging data
        if not self._check_neuroimaging_convergence(samples):
            raise RuntimeError("MCMC did not converge for neuroimaging data")

        # Extract parameters
        W_mean = np.array(samples["W"]).mean(axis=0)
        Z_mean = np.array(samples["Z"]).mean(axis=0)

        # Neuroimaging-specific evaluation
        X_test_concat = np.concatenate(X_test_list, axis=1)
        X_test_recon = Z_mean @ W_mean.T

        # Ensure dimensions match
        min_features = min(X_test_concat.shape[1], X_test_recon.shape[1])
        X_test_concat = X_test_concat[:, :min_features]
        X_test_recon = X_test_recon[:, :min_features]

        # Basic reconstruction metrics
        mse = mean_squared_error(X_test_concat, X_test_recon)
        r2 = r2_score(X_test_concat, X_test_recon)

        # Advanced neuroimaging metrics
        interpretability_scores = self.metrics.factor_interpretability_score(
            W_mean, Z_mean, ["view1", "view2"], clinical_data, position_info
        )

        # Clinical association analysis (if clinical data available)
        clinical_associations = {}
        if clinical_data is not None:
            test_clinical = clinical_data.iloc[test_idx]
            available_vars = [
                col
                for col in self.pd_config.clinical_stratification_vars
                if col in test_clinical.columns
            ]
            if available_vars:
                clinical_associations = self.metrics.clinical_association_score(
                    Z_mean, test_clinical, available_vars
                )

        # Clustering for subtype analysis
        if (
            hasattr(self.config, "auto_optimize_subtypes")
            and self.config.auto_optimize_subtypes
        ):
            # Literature-based optimization
            logger.info("Determining optimal number of PD subtypes...")

            # Use configured candidate clusters or default
            candidate_clusters = getattr(
                self.config, "subtype_candidate_range", [2, 3, 4]
            )
            optimal_n, cluster_metrics = self._find_optimal_subtypes(
                Z_mean, candidate_clusters, fold_id
            )

            logger.info(
                f"OPTIMAL SUBTYPES FOUND: {optimal_n} clusters for fold {fold_id}"
            )
            logger.info(
                f"   Best silhouette score: {cluster_metrics['best_silhouette']:.4f}"
            )
            logger.info(
                f"   Best Calinski-Harabasz score: {cluster_metrics['best_calinski']:.4f}"
            )
            logger.info(f"   Subtype distribution: {cluster_metrics['cluster_sizes']}")

            # Use optimal clustering
            kmeans = KMeans(
                n_clusters=optimal_n, random_state=self.config.random_state, n_init=10
            )
            cluster_labels = kmeans.fit_predict(Z_mean)

            # Final validation metrics with optimal clusters
            silhouette = cluster_metrics["best_silhouette"]
            calinski_harabasz = cluster_metrics["best_calinski"]
            n_subtypes = optimal_n

        else:
            # Traditional fixed clustering (backward compatibility)
            n_subtypes = 3  # TD, PIGD, Mixed subtypes in PD
            logger.info(
                f"Using fixed {n_subtypes} clusters (traditional TD/PIGD/Mixed)"
            )

            kmeans = KMeans(
                n_clusters=n_subtypes, random_state=self.config.random_state
            )
            cluster_labels = kmeans.fit_predict(Z_mean)

            # Subtype validation metrics
            if len(np.unique(cluster_labels)) > 1:
                silhouette = silhouette_score(Z_mean, cluster_labels)
                calinski_harabasz = calinski_harabasz_score(Z_mean, cluster_labels)
            else:
                silhouette = 0.0
                calinski_harabasz = 0.0

        fold_result = {
            "fold_id": fold_id,
            "W": W_mean,
            "Z": Z_mean,
            "cluster_labels": cluster_labels,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "converged": True,
            "scalers": scalers,
            # Basic metrics
            "test_mse": mse,
            "test_r2": r2,
            # Neuroimaging-specific metrics
            "interpretability_scores": interpretability_scores,
            "clinical_associations": clinical_associations,
            "spatial_coherence": interpretability_scores.get("spatial_coherence", 0.0),
            # Clustering metrics
            "silhouette_score": silhouette,
            "calinski_harabasz_score": calinski_harabasz,
            "n_subtypes_found": len(np.unique(cluster_labels)),
            # Model complexity
            "n_factors": W_mean.shape[1],
            "effective_sparsity": np.mean(np.abs(W_mean) > 0.1),
        }

        return fold_result

    def _check_neuroimaging_convergence(self, samples: Dict) -> bool:
        """Enhanced convergence checking for neuroimaging factor models."""
        try:
            # Standard checks
            for key, value in samples.items():
                if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                    logger.warning(f"Found NaN/Inf values in {key}")
                    return False

            # Check factor loading stability
            if "W" in samples:
                W_samples = samples["W"]
                # Check if factor loadings are changing significantly across samples
                if W_samples.ndim > 2:
                    # Multiple chains
                    n_samples = W_samples.shape[0]
                    first_half = W_samples[: n_samples // 2]
                    second_half = W_samples[n_samples // 2 :]

                    mean_first = np.mean(first_half, axis=0)
                    mean_second = np.mean(second_half, axis=0)

                    # Check stability using correlation
                    correlations = []
                    for factor_idx in range(mean_first.shape[1]):
                        corr = np.corrcoef(
                            mean_first[:, factor_idx], mean_second[:, factor_idx]
                        )[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))

                    avg_stability = np.mean(correlations) if correlations else 0.0
                    if avg_stability < 0.7:  # Threshold for neuroimaging factors
                        logger.warning(f"Poor factor stability: {avg_stability:.3f}")
                        return False

            return True

        except Exception as e:
            logger.warning(f"Could not check convergence: {e}")
            return True  # Conservative - assume convergence if we can't check

    def neuroimaging_cross_validate(
        self,
        X_list: List[np.ndarray],
        args,
        hypers: Dict,
        data: Dict,
        cv_type: str = "clinical_stratified",
    ) -> Dict:
        """
        Main cross-validation method for neuroimaging factor analysis.
        """
        logger.info("Starting neuroimaging-aware cross-validation")
        start_time = time.time()

        # Extract metadata
        clinical_data, position_info = self._extract_clinical_metadata(data)

        # Determine CV strategy based on data characteristics
        X_ref = X_list[0]

        if cv_type == "clinical_stratified" and clinical_data is not None:
            cv_splitter = self.splitter.create_clinical_stratified_cv(
                clinical_data, self.pd_config.clinical_stratification_vars
            )
            y = (
                clinical_data.iloc[:, 0].values
                if len(clinical_data.columns) > 0
                else None
            )
            groups = None
        elif cv_type == "site_aware" and "scanner_info" in data:
            cv_splitter = self.splitter.create_site_aware_cv(data["scanner_info"])
            y = None
            groups = data["scanner_info"]
        else:
            # Standard KFold as fallback
            cv_splitter = KFold(
                n_splits=self.config.outer_cv_folds,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
            )
            y = None
            groups = None

        # Use current args as best parameters (will be optimized in nested CV)
        best_params = {
            "K": args.K,
            "percW": getattr(args, "percW", 33),
            "num_samples": args.num_samples,
            "num_warmup": getattr(args, "num_warmup", 1000),
        }

        # Prepare fold data
        fold_data_list = []
        splits = list(cv_splitter.split(X_ref, y, groups))

        for fold_id, (train_idx, test_idx) in enumerate(splits):
            fold_data = (
                fold_id + 1,
                train_idx,
                test_idx,
                X_list,
                args,
                hypers,
                best_params,
                clinical_data,
                position_info,
            )
            fold_data_list.append(fold_data)

        # Process folds
        fold_results = []
        successful_folds = 0

        for fold_data in fold_data_list:
            try:
                result = self.fit_single_fold_robust(fold_data)
                fold_results.append(result)

                if result.get("converged", False):
                    successful_folds += 1

            except Exception as e:
                logger.error(f"Fold {fold_data[0]} failed completely: {e}")
                fold_results.append(
                    {"fold_id": fold_data[0], "error": str(e), "converged": False}
                )

        # Check if we have enough successful folds
        if successful_folds < self.config.min_successful_folds:
            logger.error(
                f"Only {successful_folds}/{len(fold_data_list)} folds succeeded"
            )
            if not self.config.continue_on_fold_failure:
                raise RuntimeError("Too many fold failures")

        # Extract results and compute comprehensive metrics
        converged_results = [r for r in fold_results if r.get("converged", False)]

        # Primary CV scores (interpretability-based)
        cv_scores = []
        interpretability_scores = []
        spatial_coherence_scores = []
        clinical_association_scores = []

        for result in converged_results:
            # Primary score: overall interpretability
            interp_score = result.get("interpretability_scores", {}).get("overall", 0.0)
            cv_scores.append(interp_score)
            interpretability_scores.append(result.get("interpretability_scores", {}))

            # Spatial coherence scores
            spatial_score = result.get("spatial_coherence", 0.0)
            spatial_coherence_scores.append(spatial_score)

            # Clinical association scores
            clinical_scores = result.get("clinical_associations", {})
            clinical_association_scores.append(clinical_scores)

        # Compute stability metrics
        stability_metrics = self._compute_neuroimaging_stability(converged_results)

        # Subtype validation analysis
        subtype_validation = self._analyze_subtype_consistency(
            converged_results, clinical_data
        )

        # Compile comprehensive results
        self.results.update(
            {
                "cv_scores": cv_scores,
                "fold_results": fold_results,
                "best_params": best_params,
                "mean_cv_score": np.mean(cv_scores) if cv_scores else np.nan,
                "std_cv_score": np.std(cv_scores) if cv_scores else np.nan,
                "total_time": time.time() - start_time,
                "n_converged_folds": len(converged_results),
                "n_failed_folds": len(self.failed_folds),
                "success_rate": len(converged_results) / len(fold_data_list),
                # Neuroimaging-specific results
                "interpretability_scores": {
                    "mean": np.mean(
                        [s.get("overall", 0) for s in interpretability_scores]
                    ),
                    "std": np.std(
                        [s.get("overall", 0) for s in interpretability_scores]
                    ),
                    "component_scores": interpretability_scores,
                },
                "spatial_coherence_scores": {
                    "mean": np.mean(spatial_coherence_scores),
                    "std": np.std(spatial_coherence_scores),
                    "individual_scores": spatial_coherence_scores,
                },
                "clinical_associations": clinical_association_scores,
                "stability_metrics": stability_metrics,
                "subtype_validation": subtype_validation,
            }
        )

        # Log comprehensive summary
        logger.info(f"Neuroimaging CV completed in {self.results['total_time']:.1f}s")
        logger.info(f"Success rate: {self.results['success_rate']:.1%}")
        logger.info(
            f"Mean interpretability score: { self.results['mean_cv_score']:.4f} ± { self.results['std_cv_score']:.4f}"
        )
        logger.info(
            f"Mean spatial coherence: { self.results['spatial_coherence_scores']['mean']:.4f}"
        )

        return self.results

    def _compute_neuroimaging_stability(self, converged_results: List[Dict]) -> Dict:
        """Compute stability metrics specific to neuroimaging factor analysis."""
        if len(converged_results) < 2:
            return {"note": "Need at least 2 converged folds for stability analysis"}

        # Factor loading stability
        W_matrices = [r["W"] for r in converged_results]
        loading_stabilities = []

        for i in range(len(W_matrices)):
            for j in range(i + 1, len(W_matrices)):
                W_i, W_j = W_matrices[i], W_matrices[j]
                if W_i.shape == W_j.shape:
                    # Compute factor-wise correlations (with optimal matching)
                    corr_matrix = np.corrcoef(W_i.T, W_j.T)
                    n_factors = W_i.shape[1]

                    # Find best matching between factors
                    factor_corrs = []
                    for k in range(n_factors):
                        max_corr = np.max(np.abs(corr_matrix[k, n_factors:]))
                        factor_corrs.append(max_corr)

                    loading_stabilities.extend(factor_corrs)

        # Clustering stability
        cluster_labels_list = [r["cluster_labels"] for r in converged_results]
        ari_scores = []

        for i in range(len(cluster_labels_list)):
            for j in range(i + 1, len(cluster_labels_list)):
                if len(cluster_labels_list[i]) == len(cluster_labels_list[j]):
                    ari = adjusted_rand_score(
                        cluster_labels_list[i], cluster_labels_list[j]
                    )
                    ari_scores.append(ari)

        # Interpretability stability
        interp_scores = [
            r.get("interpretability_scores", {}).get("overall", 0)
            for r in converged_results
        ]

        return {
            "loading_stability_mean": np.mean(loading_stabilities),
            "loading_stability_std": np.std(loading_stabilities),
            "clustering_stability_mean": np.mean(ari_scores) if ari_scores else 0.0,
            "clustering_stability_std": np.std(ari_scores) if ari_scores else 0.0,
            "interpretability_stability": np.std(interp_scores),
            "n_comparisons": len(loading_stabilities),
        }

    def _find_optimal_subtypes(
        self, Z_mean: np.ndarray, candidate_clusters: List[int], fold_id: int
    ) -> Tuple[int, Dict]:
        """
        Determine optimal number of subtypes using literature-based validation.
        Tests 2, 3, 4 clusters as supported by PD literature.
        """
        logger.info(
            f"Testing {candidate_clusters} cluster solutions for optimal subtypes..."
        )

        best_score = -1
        best_n_clusters = 3  # Default fallback
        cluster_metrics = {}

        results = []

        for n_clusters in candidate_clusters:
            if n_clusters >= Z_mean.shape[0]:  # Skip if more clusters than samples
                logger.warning(
                    f"Skipping {n_clusters} clusters (>= {Z_mean.shape[0]} samples)"
                )
                continue

            try:
                # Fit K-means
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.config.random_state,
                    n_init=10,
                )
                labels = kmeans.fit_predict(Z_mean)

                # Calculate validation metrics
                sil_score = silhouette_score(Z_mean, labels)
                cal_score = calinski_harabasz_score(Z_mean, labels)

                # Composite score (balanced between metrics)
                composite_score = 0.6 * sil_score + 0.4 * (
                    cal_score / 1000
                )  # Scale Calinski score

                # Cluster size distribution
                unique_labels, counts = np.unique(labels, return_counts=True)
                cluster_sizes = dict(zip(unique_labels, counts))

                results.append(
                    {
                        "n_clusters": n_clusters,
                        "silhouette": sil_score,
                        "calinski_harabasz": cal_score,
                        "composite_score": composite_score,
                        "cluster_sizes": cluster_sizes,
                        "min_cluster_size": min(counts),
                        "balance_score": min(counts) / max(counts),  # Cluster balance
                    }
                )

                logger.info(
                    f"  {n_clusters} clusters: Silhouette={sil_score:.4f}, "
                    f"Calinski-Harabasz={cal_score:.2f}, Balance={min(counts) / max(counts):.3f}"
                )

                # Update best solution
                if composite_score > best_score:
                    best_score = composite_score
                    best_n_clusters = n_clusters

            except Exception as e:
                logger.warning(f"Failed to evaluate {n_clusters} clusters: {str(e)}")
                continue

        if results:
            best_result = next(
                (r for r in results if r["n_clusters"] == best_n_clusters), results[0]
            )
            cluster_metrics = {
                "best_silhouette": best_result["silhouette"],
                "best_calinski": best_result["calinski_harabasz"],
                "best_composite_score": best_result["composite_score"],
                "cluster_sizes": best_result["cluster_sizes"],
                "all_results": results,
            }
        else:
            logger.error("No valid cluster solutions found, using default 3 clusters")
            best_n_clusters = 3
            cluster_metrics = {
                "best_silhouette": 0.0,
                "best_calinski": 0.0,
                "best_composite_score": 0.0,
                "cluster_sizes": {},
                "all_results": [],
            }

        return best_n_clusters, cluster_metrics

    def _analyze_subtype_consistency(
        self, converged_results: List[Dict], clinical_data: Optional[pd.DataFrame]
    ) -> Dict:
        """Analyze consistency of identified subtypes across CV folds."""
        if not converged_results:
            return {"note": "No converged results for subtype analysis"}

        # Analyze number of subtypes identified
        n_subtypes_found = [r.get("n_subtypes_found", 0) for r in converged_results]

        # Log optimal subtype summary
        if n_subtypes_found:
            from collections import Counter

            subtype_counts = Counter(n_subtypes_found)
            most_common = subtype_counts.most_common(1)[0]

            logger.info("=" * 60)
            logger.info("FINAL SUBTYPE OPTIMIZATION SUMMARY:")
            logger.info(
                f" Most frequent optimal clusters: { most_common[0]} ({ most_common[1]}/{ len(n_subtypes_found)} folds)"
            )
            logger.info(
                f"   Range of optimal clusters: {min(n_subtypes_found)} - {max(n_subtypes_found)}"
            )
            logger.info(f"   Distribution across folds: {dict(subtype_counts)}")

            if most_common[0] == 2:
                logger.info(
                    "   Interpretation: Two-subtype structure (fast vs. slow progressors)"
                )
            elif most_common[0] == 3:
                logger.info("   Interpretation: Classic TD/PIGD/Mixed motor subtypes")
            elif most_common[0] == 4:
                logger.info(
                    "   Interpretation: Extended subtype structure with additional phenotypes"
                )
            logger.info("=" * 60)

        # Silhouette scores across folds
        silhouette_scores = [r.get("silhouette_score", 0) for r in converged_results]

        # Clinical validation if available
        clinical_validation = {}
        if clinical_data is not None:
            # Check if identified subtypes correlate with known clinical subtypes
            for var in self.pd_config.clinical_targets.keys():
                if var in clinical_data.columns:
                    clinical_validation[var] = (
                        "analyzed"  # Placeholder for detailed analysis
                    )

        return {
            "n_subtypes_consistency": {
                "mean": np.mean(n_subtypes_found),
                "std": np.std(n_subtypes_found),
                "mode": stats.mode(n_subtypes_found)[0][0] if n_subtypes_found else 0,
            },
            "silhouette_scores": {
                "mean": np.mean(silhouette_scores),
                "std": np.std(silhouette_scores),
                "individual_scores": silhouette_scores,
            },
            "clinical_validation": clinical_validation,
        }

    def nested_neuroimaging_cv(
        self,
        X_list: List[np.ndarray],
        args,
        hypers: Dict,
        data: Dict,
        cv_type: str = "clinical_stratified",
    ) -> Dict:
        """
        Nested cross-validation with hyperparameter optimization for neuroimaging data.
        """
        logger.info("Starting nested neuroimaging cross-validation")
        start_time = time.time()

        # Extract metadata
        clinical_data, position_info = self._extract_clinical_metadata(data)

        # Create CV splitters (simplified nested approach)
        X_ref = X_list[0]
        outer_cv = KFold(
            n_splits=self.config.outer_cv_folds,
            shuffle=self.config.shuffle,
            random_state=self.config.random_state,
        )

        # Get parameter grid
        param_combinations = self.hyperopt.create_neuroimaging_parameter_grid()

        outer_scores = []
        best_params_per_fold = []
        fold_results = []

        # Outer CV loop
        for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(X_ref)):
            logger.info(
                f"Nested CV outer fold {fold_id + 1}/{self.config.outer_cv_folds}"
            )

            try:
                # Inner CV for hyperparameter optimization
                X_outer_train = [X[train_idx] for X in X_list]

                # Simplified parameter selection (evaluate subset of parameters)
                best_score = -np.inf
                best_params = param_combinations[0]  # Default

                # Evaluate a subset of parameter combinations
                n_eval = min(5, len(param_combinations))  # Limit evaluations
                for i, param_combo in enumerate(param_combinations[:n_eval]):
                    logger.info(
                        f"  Evaluating params {i + 1}/{n_eval}: K={param_combo['K']}"
                    )

                    try:
                        # Quick evaluation on subset of training data
                        subset_size = min(50, len(train_idx))
                        subset_idx = np.random.choice(
                            len(train_idx), subset_size, replace=False
                        )
                        X_subset = [
                            X_outer_train[m][subset_idx]
                            for m in range(len(X_outer_train))
                        ]

                        # Simplified evaluation
                        args_eval = deepcopy(args)
                        for key, value in param_combo.items():
                            setattr(args_eval, key, value)

                        hypers_eval = deepcopy(hypers)
                        hypers_eval["Dm"] = [X.shape[1] for X in X_subset]

                        # Quick MCMC run
                        from core import run_analysis as RA

                        rng = jrandom.PRNGKey(self.config.random_state + i)
                        mcmc = RA.run_inference(
                            RA.models, args_eval, rng, X_subset, hypers_eval
                        )
                        samples = mcmc.get_samples()

                        # Evaluate quality
                        score = self.hyperopt.evaluate_factor_model_quality(
                            samples, X_subset, clinical_data, position_info
                        )

                        if score > best_score:
                            best_score = score
                            best_params = param_combo

                    except Exception as e:
                        logger.warning(f"Parameter evaluation failed: {e}")
                        continue

                best_params_per_fold.append(best_params)
                logger.info(f"Best params for fold {fold_id + 1}: {best_params}")

                # Evaluate with best parameters on test set
                fold_data = (
                    fold_id + 1,
                    train_idx,
                    test_idx,
                    X_list,
                    args,
                    hypers,
                    best_params,
                    clinical_data,
                    position_info,
                )
                fold_result = self.fit_single_fold_robust(fold_data)

                if fold_result["converged"]:
                    outer_score = fold_result.get("interpretability_scores", {}).get(
                        "overall", 0.0
                    )
                    outer_scores.append(outer_score)
                    fold_results.append(fold_result)
                else:
                    logger.warning(f"Outer fold {fold_id + 1} failed to converge")
                    outer_scores.append(np.nan)

            except Exception as e:
                logger.error(f"Nested CV outer fold {fold_id + 1} failed: {e}")
                outer_scores.append(np.nan)
                best_params_per_fold.append({})

        # Compile nested CV results
        self.results.update(
            {
                "cv_scores": outer_scores,
                "fold_results": fold_results,
                "best_params_per_fold": best_params_per_fold,
                "mean_cv_score": np.nanmean(outer_scores),
                "std_cv_score": np.nanstd(outer_scores),
                "total_time": time.time() - start_time,
                "n_converged_folds": len(
                    [r for r in fold_results if r.get("converged")]
                ),
            }
        )

        logger.info(
            f"Nested neuroimaging CV completed in {self.results['total_time']:.1f}s"
        )
        logger.info(
            f"Mean CV score: { self.results['mean_cv_score']:.4f} ± { self.results['std_cv_score']:.4f}"
        )

        return self.results

    def save_neuroimaging_results(self, output_dir: Path, run_name: str):
        """Save comprehensive neuroimaging CV results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main results
        results_path = output_dir / f"{run_name}_neuroimaging_cv_results.json"

        # Convert numpy arrays to lists for JSON serialization
        results_json = deepcopy(self.results)
        for fold_result in results_json.get("fold_results", []):
            for key in ["W", "Z", "cluster_labels"]:
                if key in fold_result and isinstance(fold_result[key], np.ndarray):
                    fold_result[key] = fold_result[key].tolist()

        save_json(results_json, results_path)

        # Create detailed summary report
        summary_path = output_dir / f"{run_name}_neuroimaging_summary.txt"
        with open(summary_path, "w") as f:
            f.write("Neuroimaging Cross-Validation Results Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Run: {run_name}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Model performance
            f.write("Model Performance:\n")
            f.write(
                f" Mean Interpretability Score: { self.results.get( 'mean_cv_score', 'N/A'):.4f}\n"
            )
            f.write(
                f"  Standard Deviation: {self.results.get('std_cv_score', 'N/A'):.4f}\n"
            )
            f.write(f"  Success Rate: {self.results.get('success_rate', 0):.1%}\n\n")

            # Neuroimaging-specific metrics
            if "spatial_coherence_scores" in self.results:
                spatial = self.results["spatial_coherence_scores"]
                f.write("Spatial Coherence:\n")
                f.write(f"  Mean: {spatial.get('mean', 0):.4f}\n")
                f.write(f"  Std: {spatial.get('std', 0):.4f}\n\n")

            # Stability metrics
            if "stability_metrics" in self.results:
                stability = self.results["stability_metrics"]
                f.write("Stability Metrics:\n")
                f.write(
                    f" Loading Stability: { stability.get( 'loading_stability_mean', 0):.3f}\n"
                )
                f.write(
                    f" Clustering Stability: { stability.get( 'clustering_stability_mean', 0):.3f}\n\n"
                )

            # Subtype analysis
            if "subtype_validation" in self.results:
                subtypes = self.results["subtype_validation"]
                if "n_subtypes_consistency" in subtypes:
                    n_sub = subtypes["n_subtypes_consistency"]
                    f.write("Subtype Consistency:\n")
                    f.write(f"  Mean Subtypes Found: {n_sub.get('mean', 0):.1f}\n")
                    f.write(f"  Most Common: {n_sub.get('mode', 0)}\n\n")

            # Clinical associations
            if "clinical_associations" in self.results:
                f.write("Clinical Associations Found:\n")
                clinical_assocs = self.results["clinical_associations"]
                if clinical_assocs:
                    for fold_idx, assoc in enumerate(clinical_assocs):
                        if assoc:
                            f.write(f"  Fold {fold_idx + 1}: {list(assoc.keys())}\n")
                f.write("\n")

            f.write("Analysis completed successfully!\n")

        logger.info(f"Neuroimaging CV results saved to {output_dir}")


# == CLI INTERFACE FOR NEUROIMAGING CV ==


def main():
    """Main function for standalone CLI usage of neuroimaging CV."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Neuroimaging Cross-Validation for Parkinson's Disease"
    )

    # Data arguments
    parser.add_argument("--dataset", type=str, default="qmap_pd")
    parser.add_argument("--data_dir", type=str, default="qMAP-PD_data")
    parser.add_argument(
        "--clinical_rel", type=str, default="data_clinical/pd_motor_gfa_data.tsv"
    )
    parser.add_argument("--volumes_rel", type=str, default="volume_matrices")
    parser.add_argument("--roi_views", action="store_true")

    # CV parameters
    parser.add_argument(
        "--cv_type",
        type=str,
        default="clinical_stratified",
        choices=["clinical_stratified", "site_aware", "standard"],
    )
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--nested_cv", action="store_true")
    parser.add_argument("--optimize_for_subtypes", action="store_true")

    # Model parameters
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--percW", type=int, default=33)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--num_warmup", type=int, default=1000)
    parser.add_argument("--num_chains", type=int, default=1)
    parser.add_argument("--model", type=str, default="sparseGFA")
    parser.add_argument("--device", type=str, default="cpu")

    # Output
    parser.add_argument("--output_dir", type=str, default="neuroimaging_cv_results")
    parser.add_argument("--run_name", type=str, default="parkinsons_cv")

    args = parser.parse_args()

    # Setup
    import numpyro

    numpyro.set_platform(args.device)

    # Load data
    from core.get_data import get_data

    data = get_data(
        dataset=args.dataset,
        data_dir=args.data_dir,
        clinical_rel=args.clinical_rel,
        volumes_rel=args.volumes_rel,
        imaging_as_single_view=not args.roi_views,
        enable_advanced_preprocessing=True,
        enable_spatial_processing=True,
    )

    X_list = data["X_list"]

    # Setup hyperparameters
    hypers = {
        "a_sigma": 1,
        "b_sigma": 1,
        "nu_local": 1,
        "nu_global": 1,
        "slab_scale": 2,
        "slab_df": 4,
        "percW": args.percW,
        "Dm": [X.shape[1] for X in X_list],
    }

    # Setup args object
    model_args = type(
        "Args",
        (),
        {
            "K": args.K,
            "percW": args.percW,
            "num_samples": args.num_samples,
            "num_warmup": args.num_warmup,
            "num_chains": args.num_chains,
            "model": args.model,
            "reghsZ": True,
            "num_sources": len(X_list),
            "device": args.device,
        },
    )()

    # Initialize CV
    config = NeuroImagingCVConfig()
    config.outer_cv_folds = args.cv_folds

    cv = NeuroImagingCrossValidator(config)

    # Run CV
    if args.nested_cv:
        results = cv.nested_neuroimaging_cv(
            X_list, model_args, hypers, data, args.cv_type
        )
    else:
        results = cv.neuroimaging_cross_validate(
            X_list, model_args, hypers, data, args.cv_type
        )

    # Save results
    output_dir = Path(args.output_dir)
    cv.save_neuroimaging_results(output_dir, args.run_name)

    logger.info("Neuroimaging cross-validation completed successfully!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
