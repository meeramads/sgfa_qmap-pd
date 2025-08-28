"""
Cross-validation module for Sparse Bayesian Group Factor Analysis.

This module can be used in two ways:

1. As a Python module (imported by other scripts):
   from crossvalidation import SparseBayesianGFACrossValidator, CVConfig

2. As a standalone CLI tool (run directly):
   python crossvalidation.py --dataset qmap_pd --quick_test
   python crossvalidation.py --dataset qmap_pd --cv_folds 5
"""

import json
import os
import csv
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from collections import defaultdict
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, 
    TimeSeriesSplit, RepeatedKFold,
    ParameterGrid, ParameterSampler
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, calinski_harabasz_score
)
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

# JAX and NumPyro imports
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpyro
import numpyro.distributions as dist

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpyro")

# == CORE CV CLASSES (for import) ==

class CVConfig:
    """Configuration class for cross-validation parameters."""
    
    def __init__(self):
        # CV Strategy Configuration
        self.outer_cv_folds = 5
        self.inner_cv_folds = 3
        self.n_repeats = 1
        self.test_size = 0.2
        self.random_state = 42
        self.shuffle = True
        
        # Model Selection Configuration
        self.param_search_method = 'grid'  # 'grid', 'random', 'bayesian'
        self.n_param_samples = 50  # for random/bayesian search
        self.scoring_metric = 'reconstruction_r2'
        self.stability_threshold = 0.7
        
        # Computational Configuration
        self.n_jobs = min(4, mp.cpu_count() - 1)
        self.parallel_backend = 'processes'  # 'threads', 'processes'
        self.timeout_seconds = 1800  # 30 minutes per fold
        
        # Output Configuration
        self.save_fold_models = False
        self.create_visualizations = True
        self.detailed_logging = True


class AdvancedCVSplitter:
    """Advanced cross-validation splitting with multiple strategies."""
    
    def __init__(self, config: CVConfig):
        self.config = config
    
    def create_cv_splitter(self, X, y=None, groups=None, cv_type='standard'):
        """
        Create CV splitter based on data characteristics.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like, optional
            Target variable for stratification
        groups : array-like, optional
            Group labels for grouped CV
        cv_type : str
            Type of CV: 'standard', 'stratified', 'grouped', 'repeated', 'time_series'
        """
        n_samples = len(X)
        
        if cv_type == 'grouped' and groups is not None:
            # GroupKFold for medical data with patient/site groupings
            cv = GroupKFold(n_splits=self.config.outer_cv_folds)
            logger.info(f"Using GroupKFold with {len(np.unique(groups))} unique groups")
            
        elif cv_type == 'stratified' and y is not None:
            # Stratified for classification or ordinal targets
            if len(np.unique(y)) < 10:  # Likely categorical
                cv = StratifiedKFold(
                    n_splits=self.config.outer_cv_folds,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state
                )
                logger.info("Using StratifiedKFold for categorical target")
            else:
                # Bin continuous targets for stratification
                y_binned = pd.qcut(y, q=5, duplicates='drop').codes
                cv = StratifiedKFold(
                    n_splits=self.config.outer_cv_folds,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state
                )
                logger.info("Using StratifiedKFold with binned continuous target")
                
        elif cv_type == 'repeated':
            # Repeated K-fold for more robust estimates
            cv = RepeatedKFold(
                n_splits=self.config.outer_cv_folds,
                n_repeats=self.config.n_repeats,
                random_state=self.config.random_state
            )
            logger.info(f"Using RepeatedKFold with {self.config.n_repeats} repeats")
            
        elif cv_type == 'time_series':
            # Time series split (preserves temporal order)
            cv = TimeSeriesSplit(n_splits=self.config.outer_cv_folds)
            logger.info("Using TimeSeriesSplit for temporal data")
            
        else:
            # Standard K-fold
            cv = KFold(
                n_splits=self.config.outer_cv_folds,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state
            )
            logger.info("Using standard KFold")
        
        return cv
    
    def create_nested_cv_splits(self, X, y=None, groups=None, cv_type='standard'):
        """Create nested CV splits for hyperparameter tuning."""
        outer_cv = self.create_cv_splitter(X, y, groups, cv_type)
        inner_cv = self.create_cv_splitter(X, y, groups, cv_type)
        inner_cv.n_splits = self.config.inner_cv_folds
        
        return outer_cv, inner_cv


class BayesianModelComparison:
    """Bayesian model comparison methods for factor models."""
    
    @staticmethod
    def compute_waic(log_likelihood_samples: np.ndarray) -> Tuple[float, float]:
        """
        Compute Watanabe-Akaike Information Criterion (WAIC).
        
        Parameters
        ----------
        log_likelihood_samples : ndarray, shape (n_samples, n_observations)
            Log-likelihood samples from MCMC
        """
        # Pointwise log predictive density
        lppd = np.log(np.mean(np.exp(log_likelihood_samples), axis=0))
        lppd_sum = np.sum(lppd)
        
        # Effective number of parameters (penalty term)
        p_waic = np.sum(np.var(log_likelihood_samples, axis=0))
        
        # WAIC
        waic = -2 * (lppd_sum - p_waic)
        
        # Standard error (approximation)
        se = np.sqrt(len(lppd) * np.var(-2 * (lppd - np.var(log_likelihood_samples, axis=0))))
        
        return waic, se
    
    @staticmethod
    def compute_loo(log_likelihood_samples: np.ndarray) -> Tuple[float, float]:
        """
        Compute Leave-One-Out Cross-Validation (LOO) using Pareto smoothed importance sampling.
        
        Note: This is a simplified implementation. For production, use ArviZ or similar.
        """
        n_samples, n_obs = log_likelihood_samples.shape
        
        # Pointwise LOO
        loo_pointwise = []
        for i in range(n_obs):
            # Importance weights
            log_weights = log_likelihood_samples[:, i]
            log_weights = log_weights - np.max(log_weights)  # Numerical stability
            weights = np.exp(log_weights)
            weights = weights / np.sum(weights)
            
            # LOO estimate for this observation
            loo_i = np.log(np.sum(weights * np.exp(log_likelihood_samples[:, i])))
            loo_pointwise.append(loo_i)
        
        loo_pointwise = np.array(loo_pointwise)
        elpd_loo = np.sum(loo_pointwise)
        
        # Effective number of parameters
        p_loo = np.sum(np.var(log_likelihood_samples, axis=0))
        
        # LOO
        loo = -2 * elpd_loo
        se = np.sqrt(n_obs * np.var(-2 * loo_pointwise))
        
        return loo, se


class ComprehensiveMetrics:
    """Comprehensive evaluation metrics for sparse GFA models."""
    
    @staticmethod
    def reconstruction_metrics(X_true_list, X_pred_list):
        """Compute reconstruction metrics across multiple views."""
        metrics = {}
        
        for view_idx, (X_true, X_pred) in enumerate(zip(X_true_list, X_pred_list)):
            view_name = f"view_{view_idx + 1}"
            
            # Basic metrics
            mse = mean_squared_error(X_true, X_pred)
            mae = mean_absolute_error(X_true, X_pred)
            r2 = r2_score(X_true, X_pred)
            
            # Normalized metrics
            rmse_normalized = np.sqrt(mse) / (np.max(X_true) - np.min(X_true))
            
            # Correlation-based metrics
            correlations = []
            for feature_idx in range(X_true.shape[1]):
                if np.std(X_true[:, feature_idx]) > 1e-10:  # Avoid constant features
                    corr = np.corrcoef(X_true[:, feature_idx], X_pred[:, feature_idx])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            mean_feature_correlation = np.mean(correlations) if correlations else 0.0
            
            metrics[view_name] = {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'rmse_normalized': rmse_normalized,
                'mean_feature_correlation': mean_feature_correlation,
                'n_features': X_true.shape[1]
            }
        
        # Overall metrics
        all_r2 = [metrics[f"view_{i+1}"]["r2"] for i in range(len(X_true_list))]
        metrics['overall'] = {
            'mean_r2': np.mean(all_r2),
            'std_r2': np.std(all_r2),
            'min_r2': np.min(all_r2),
            'max_r2': np.max(all_r2)
        }
        
        return metrics
    
    @staticmethod
    def stability_metrics(W_matrices, Z_matrices):
        """Compute stability metrics across CV folds."""
        n_folds = len(W_matrices)
        
        if n_folds < 2:
            return {'note': 'Need at least 2 folds for stability analysis'}
        
        # Factor loading stability
        W_correlations = []
        for i in range(n_folds):
            for j in range(i + 1, n_folds):
                W_i, W_j = W_matrices[i], W_matrices[j]
                if W_i.shape == W_j.shape:
                    # Compute correlations between matched factors
                    corr_matrix = np.corrcoef(W_i.T, W_j.T)
                    n_factors = W_i.shape[1]
                    factor_corrs = []
                    
                    # Hungarian matching or simple max correlation matching
                    for k in range(n_factors):
                        max_corr = np.max(np.abs(corr_matrix[k, n_factors:]))
                        factor_corrs.append(max_corr)
                    
                    W_correlations.extend(factor_corrs)
        
        # Subject score stability  
        Z_correlations = []
        for i in range(n_folds):
            for j in range(i + 1, n_folds):
                Z_i, Z_j = Z_matrices[i], Z_matrices[j]
                if Z_i.shape == Z_j.shape:
                    # Compute correlations between matched factors
                    for k in range(Z_i.shape[1]):
                        corr = np.corrcoef(Z_i[:, k], Z_j[:, k])[0, 1]
                        if not np.isnan(corr):
                            Z_correlations.append(np.abs(corr))
        
        return {
            'loading_stability_mean': np.mean(W_correlations),
            'loading_stability_std': np.std(W_correlations),
            'score_stability_mean': np.mean(Z_correlations),
            'score_stability_std': np.std(Z_correlations),
            'n_loading_comparisons': len(W_correlations),
            'n_score_comparisons': len(Z_correlations)
        }
    
    @staticmethod
    def clustering_metrics(cluster_labels_list, true_labels=None):
        """Compute clustering stability and quality metrics."""
        n_folds = len(cluster_labels_list)
        
        if n_folds < 2:
            return {'note': 'Need at least 2 folds for clustering analysis'}
        
        # Clustering stability (ARI between folds)
        ari_scores = []
        for i in range(n_folds):
            for j in range(i + 1, n_folds):
                labels_i, labels_j = cluster_labels_list[i], cluster_labels_list[j]
                if len(labels_i) == len(labels_j):
                    ari = adjusted_rand_score(labels_i, labels_j)
                    ari_scores.append(ari)
        
        metrics = {
            'ari_stability_mean': np.mean(ari_scores),
            'ari_stability_std': np.std(ari_scores),
            'n_ari_comparisons': len(ari_scores)
        }
        
        # If true labels available, compute predictive performance
        if true_labels is not None:
            ari_vs_true = []
            nmi_vs_true = []
            
            for labels in cluster_labels_list:
                if len(labels) == len(true_labels):
                    ari = adjusted_rand_score(true_labels, labels)
                    nmi = normalized_mutual_info_score(true_labels, labels)
                    ari_vs_true.append(ari)
                    nmi_vs_true.append(nmi)
            
            metrics.update({
                'ari_vs_true_mean': np.mean(ari_vs_true),
                'ari_vs_true_std': np.std(ari_vs_true),
                'nmi_vs_true_mean': np.mean(nmi_vs_true),
                'nmi_vs_true_std': np.std(nmi_vs_true)
            })
        
        return metrics


class HyperparameterOptimizer:
    """Hyperparameter optimization for sparse GFA models."""
    
    def __init__(self, config: CVConfig):
        self.config = config
        
    def create_parameter_grid(self, search_space=None):
        """Create parameter grid for hyperparameter search."""
        if search_space is None:
            search_space = {
                'K': [5, 10, 15, 20, 25],
                'percW': [25, 33, 50],
                'num_samples': [800, 1000, 1500],
                'num_warmup': [500, 800],
                'slab_scale': [1.5, 2.0, 2.5],
                'slab_df': [3, 4, 5]
            }
        
        if self.config.param_search_method == 'grid':
            param_combinations = list(ParameterGrid(search_space))
        elif self.config.param_search_method == 'random':
            param_combinations = list(ParameterSampler(
                search_space, 
                n_iter=self.config.n_param_samples,
                random_state=self.config.random_state
            ))
        else:  # Simplified Bayesian optimization placeholder
            param_combinations = list(ParameterGrid(search_space))[:self.config.n_param_samples]
        
        logger.info(f"Created {len(param_combinations)} parameter combinations")
        return param_combinations
    
    def evaluate_parameters(self, params, X_train_list, X_val_list, args, hypers):
        """Evaluate a specific parameter combination."""
        try:
            # Update arguments with current parameters
            args_eval = deepcopy(args)
            for key, value in params.items():
                setattr(args_eval, key, value)
            
            # Update hyperparameters
            hypers_eval = deepcopy(hypers)
            for key in ['slab_scale', 'slab_df']:
                if key in params:
                    hypers_eval[key] = params[key]
            
            # Import here to avoid dependency issues when used as module
            try:
                import run_analysis as RA
            except ImportError:
                raise ImportError("run_analysis module not available for parameter evaluation")
            
            # Run inference
            rng = jrandom.PRNGKey(self.config.random_state)
            mcmc = RA.run_inference(RA.models, args_eval, rng, X_train_list, hypers_eval)
            samples = mcmc.get_samples()
            
            # Compute validation score (simplified)
            W_mean = np.array(samples['W']).mean(axis=0)
            
            # Reconstruct validation data (simplified)
            X_val_concat = np.concatenate(X_val_list, axis=1)
            X_train_concat = np.concatenate(X_train_list, axis=1)
            
            # Simple scoring metric
            score = -np.mean((X_val_concat - X_train_concat @ W_mean @ W_mean.T) ** 2)
            
            return {
                'score': score,
                'params': params,
                'converged': True
            }
            
        except Exception as e:
            logger.warning(f"Parameter evaluation failed: {e}")
            return {
                'score': -np.inf,
                'params': params,
                'converged': False,
                'error': str(e)
            }


class SparseBayesianGFACrossValidator:
    """Main cross-validation class for Sparse Group Factor Analysis."""
    
    def __init__(self, config: CVConfig = None):
        self.config = config or CVConfig()
        self.splitter = AdvancedCVSplitter(self.config)
        self.metrics_calculator = ComprehensiveMetrics()
        self.hyperopt = HyperparameterOptimizer(self.config)
        self.model_comparison = BayesianModelComparison()
        
        # Results storage
        self.results = {
            'cv_scores': [],
            'fold_results': [],
            'best_params': {},
            'stability_metrics': {},
            'model_comparison': {},
            'timing': {}
        }
    
    def fit_single_fold(self, fold_data):
        """Fit model on a single CV fold."""
        fold_id, train_idx, test_idx, X_list, args, hypers, best_params = fold_data
        
        logger.info(f"Processing fold {fold_id}")
        start_time = time.time()
        
        try:
            # Prepare fold data
            X_train_list = [X[train_idx] for X in X_list]
            X_test_list = [X[test_idx] for X in X_list]
            
            # Scale data
            scalers = []
            for m, X_train in enumerate(X_train_list):
                mu = np.mean(X_train, axis=0, keepdims=True)
                sigma = np.std(X_train, axis=0, keepdims=True)
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
            hypers_fold['Dm'] = [X.shape[1] for X in X_train_list]
            
            # Import here to avoid dependency issues
            try:
                import run_analysis as RA
            except ImportError:
                raise ImportError("run_analysis module not available for CV")
            
            # Run inference
            rng = jrandom.PRNGKey(self.config.random_state + fold_id)
            mcmc = RA.run_inference(RA.models, args_fold, rng, X_train_list, hypers_fold)
            samples = mcmc.get_samples()
            
            # Extract parameters
            W_mean = np.array(samples['W']).mean(axis=0)
            Z_mean = np.array(samples['Z']).mean(axis=0)
            sigma_mean = np.array(samples['sigma']).mean(axis=0)
            
            # Evaluate reconstruction on test set
            X_test_concat = np.concatenate(X_test_list, axis=1)
            X_test_recon = Z_mean @ W_mean.T
            
            # Split reconstruction back to views
            X_test_recon_list = []
            d = 0
            for m, dim in enumerate(hypers_fold['Dm']):
                X_test_recon_list.append(X_test_recon[:, d:d+dim])
                d += dim
            
            # Compute metrics
            recon_metrics = self.metrics_calculator.reconstruction_metrics(
                X_test_list, X_test_recon_list
            )
            
            # Simple clustering (k-means on Z_mean)
            from sklearn.cluster import KMeans
            n_clusters = getattr(args_fold, 'n_subtypes', 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_state)
            cluster_labels = kmeans.fit_predict(Z_mean)
            
            fold_result = {
                'fold_id': fold_id,
                'W': W_mean,
                'Z': Z_mean,
                'cluster_labels': cluster_labels,
                'recon_metrics': recon_metrics,
                'n_train': len(train_idx),
                'n_test': len(test_idx),
                'fit_time': time.time() - start_time,
                'converged': True
            }
            
            logger.info(f"Fold {fold_id} completed in {fold_result['fit_time']:.1f}s")
            return fold_result
            
        except Exception as e:
            logger.error(f"Fold {fold_id} failed: {e}")
            return {
                'fold_id': fold_id,
                'error': str(e),
                'converged': False,
                'fit_time': time.time() - start_time
            }
    
    def nested_cross_validate(self, X_list, args, hypers, y=None, groups=None, 
                            cv_type='standard', search_space=None):
        """Perform nested cross-validation with hyperparameter optimization."""
        logger.info("Starting nested cross-validation")
        start_time = time.time()
        
        # Create CV splitters
        X_ref = X_list[0]  # Use first view for CV splitting
        outer_cv, inner_cv = self.splitter.create_nested_cv_splits(
            X_ref, y, groups, cv_type
        )
        
        # Parameter grid
        param_combinations = self.hyperopt.create_parameter_grid(search_space)
        
        outer_scores = []
        fold_results = []
        best_params_per_fold = []
        
        # Outer CV loop
        for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(X_ref, y, groups)):
            logger.info(f"Outer fold {fold_id + 1}/{self.config.outer_cv_folds}")
            
            # Inner CV for hyperparameter optimization
            X_outer_train = [X[train_idx] for X in X_list]
            y_outer_train = y[train_idx] if y is not None else None
            groups_outer_train = groups[train_idx] if groups is not None else None
            
            # Hyperparameter optimization
            best_score = -np.inf
            best_params = {}
            
            for param_combo in param_combinations:
                inner_scores = []
                
                # Inner CV loop
                for inner_train_idx, inner_val_idx in inner_cv.split(
                    X_outer_train[0], y_outer_train, groups_outer_train
                ):
                    X_inner_train = [X[inner_train_idx] for X in X_outer_train]
                    X_inner_val = [X[inner_val_idx] for X in X_outer_train]
                    
                    # Evaluate parameters
                    result = self.hyperopt.evaluate_parameters(
                        param_combo, X_inner_train, X_inner_val, args, hypers
                    )
                    
                    if result['converged']:
                        inner_scores.append(result['score'])
                    else:
                        inner_scores.append(-np.inf)
                
                avg_inner_score = np.mean(inner_scores)
                if avg_inner_score > best_score:
                    best_score = avg_inner_score
                    best_params = param_combo
            
            best_params_per_fold.append(best_params)
            logger.info(f"Best parameters for fold {fold_id + 1}: {best_params}")
            
            # Prepare data for outer evaluation
            fold_data = (fold_id + 1, train_idx, test_idx, X_list, args, hypers, best_params)
            fold_result = self.fit_single_fold(fold_data)
            
            if fold_result['converged']:
                outer_score = fold_result['recon_metrics']['overall']['mean_r2']
                outer_scores.append(outer_score)
                fold_results.append(fold_result)
            else:
                logger.warning(f"Fold {fold_id + 1} failed to converge")
                outer_scores.append(np.nan)
        
        # Compile results
        self.results.update({
            'cv_scores': outer_scores,
            'fold_results': fold_results,
            'best_params_per_fold': best_params_per_fold,
            'mean_cv_score': np.nanmean(outer_scores),
            'std_cv_score': np.nanstd(outer_scores),
            'total_time': time.time() - start_time
        })
        
        # Compute stability metrics
        if len(fold_results) >= 2:
            converged_results = [r for r in fold_results if r['converged']]
            if len(converged_results) >= 2:
                W_matrices = [r['W'] for r in converged_results]
                Z_matrices = [r['Z'] for r in converged_results]
                cluster_labels = [r['cluster_labels'] for r in converged_results]
                
                self.results['stability_metrics'] = self.metrics_calculator.stability_metrics(
                    W_matrices, Z_matrices
                )
                
                self.results['clustering_metrics'] = self.metrics_calculator.clustering_metrics(
                    cluster_labels
                )
        
        logger.info(f"Nested CV completed in {self.results['total_time']:.1f}s")
        logger.info(f"Mean CV score: {self.results['mean_cv_score']:.4f} ± {self.results['std_cv_score']:.4f}")
        
        return self.results
    
    def standard_cross_validate(self, X_list, args, hypers, y=None, groups=None, 
                              cv_type='standard'):
        """Perform standard cross-validation (no hyperparameter optimization)."""
        logger.info("Starting standard cross-validation")
        start_time = time.time()
        
        # Create CV splitter
        X_ref = X_list[0]
        cv_splitter = self.splitter.create_cv_splitter(X_ref, y, groups, cv_type)
        
        # Use current args as best parameters
        best_params = {
            'K': args.K,
            'percW': args.percW,
            'num_samples': args.num_samples,
            'num_warmup': args.num_warmup
        }
        
        # Prepare fold data for parallel processing
        fold_data_list = []
        for fold_id, (train_idx, test_idx) in enumerate(cv_splitter.split(X_ref, y, groups)):
            fold_data = (fold_id + 1, train_idx, test_idx, X_list, args, hypers, best_params)
            fold_data_list.append(fold_data)
        
        # Process folds
        if self.config.n_jobs == 1:
            # Sequential processing
            fold_results = []
            for fold_data in fold_data_list:
                result = self.fit_single_fold(fold_data)
                fold_results.append(result)
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
                future_to_fold = {
                    executor.submit(self.fit_single_fold, fold_data): fold_data[0]
                    for fold_data in fold_data_list
                }
                
                fold_results = []
                for future in as_completed(future_to_fold, timeout=self.config.timeout_seconds):
                    fold_id = future_to_fold[future]
                    try:
                        result = future.result()
                        fold_results.append(result)
                    except Exception as e:
                        logger.error(f"Fold {fold_id} failed: {e}")
                        fold_results.append({
                            'fold_id': fold_id,
                            'error': str(e),
                            'converged': False
                        })
        
        # Sort results by fold_id
        fold_results.sort(key=lambda x: x['fold_id'])
        
        # Compile results
        converged_results = [r for r in fold_results if r.get('converged', False)]
        cv_scores = [r['recon_metrics']['overall']['mean_r2'] for r in converged_results]
        
        self.results.update({
            'cv_scores': cv_scores,
            'fold_results': fold_results,
            'best_params': best_params,
            'mean_cv_score': np.mean(cv_scores) if cv_scores else np.nan,
            'std_cv_score': np.std(cv_scores) if cv_scores else np.nan,
            'total_time': time.time() - start_time,
            'n_converged_folds': len(converged_results)
        })
        
        # Compute additional metrics
        if len(converged_results) >= 2:
            W_matrices = [r['W'] for r in converged_results]
            Z_matrices = [r['Z'] for r in converged_results]
            cluster_labels = [r['cluster_labels'] for r in converged_results]
            
            self.results['stability_metrics'] = self.metrics_calculator.stability_metrics(
                W_matrices, Z_matrices
            )
            
            self.results['clustering_metrics'] = self.metrics_calculator.clustering_metrics(
                cluster_labels
            )
        
        logger.info(f"Standard CV completed in {self.results['total_time']:.1f}s")
        logger.info(f"Mean CV score: {self.results['mean_cv_score']:.4f} ± {self.results['std_cv_score']:.4f}")
        logger.info(f"Converged folds: {self.results['n_converged_folds']}/{len(fold_results)}")
        
        return self.results
    
    def save_results(self, output_dir: Path, run_name: str):
        """Save cross-validation results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Main results JSON
        results_path = output_dir / f"{run_name}_cv_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        results_json = deepcopy(self.results)
        for fold_result in results_json.get('fold_results', []):
            if 'W' in fold_result:
                fold_result['W'] = fold_result['W'].tolist()
            if 'Z' in fold_result:
                fold_result['Z'] = fold_result['Z'].tolist()
            if 'cluster_labels' in fold_result:
                fold_result['cluster_labels'] = fold_result['cluster_labels'].tolist()
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
        
        # Create summary report
        summary_path = output_dir / f"{run_name}_cv_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Cross-Validation Results Summary\n")
            f.write(f"{'='*40}\n\n")
            f.write(f"Run: {run_name}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  CV Folds: {self.config.outer_cv_folds}\n")
            f.write(f"  Scoring Metric: {self.config.scoring_metric}\n")
            f.write(f"  Parallel Jobs: {self.config.n_jobs}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Mean CV Score: {self.results.get('mean_cv_score', 'N/A'):.4f}\n")
            f.write(f"  Std CV Score: {self.results.get('std_cv_score', 'N/A'):.4f}\n")
            f.write(f"  Converged Folds: {self.results.get('n_converged_folds', 'N/A')}\n")
            f.write(f"  Total Time: {self.results.get('total_time', 'N/A'):.1f}s\n\n")
            
            if 'stability_metrics' in self.results:
                stability = self.results['stability_metrics']
                f.write(f"Stability Metrics:\n")
                f.write(f"  Loading Stability: {stability.get('loading_stability_mean', 'N/A'):.3f} ± {stability.get('loading_stability_std', 'N/A'):.3f}\n")
                f.write(f"  Score Stability: {stability.get('score_stability_mean', 'N/A'):.3f} ± {stability.get('score_stability_std', 'N/A'):.3f}\n\n")
            
            if 'clustering_metrics' in self.results:
                clustering = self.results['clustering_metrics']
                f.write(f"Clustering Metrics:\n")
                f.write(f"  ARI Stability: {clustering.get('ari_stability_mean', 'N/A'):.3f} ± {clustering.get('ari_stability_std', 'N/A'):.3f}\n")
        
        logger.info(f"Summary saved to {summary_path}")

# == STANDALONE CLI FUNCTIONALITY ==

class CVInspector:
    """Class for inspecting cross-validation behavior."""
    
    def __init__(self, dataset: str, **load_kwargs):
        self.dataset = dataset
        self.load_kwargs = load_kwargs
        self.data = None
        self.X_list = None
        self.clinical_df = None
        
    def load_data(self):
        """Load data for CV inspection."""
        logger.info(f"Loading {self.dataset} data for CV inspection...")
        
        if self.dataset == 'synthetic':
            logger.error("Synthetic data inspection not implemented in standalone version")
            return False
            
        try:
            # Import here to avoid dependency issues when used as module
            from get_data import get_data
            
            self.data = get_data(
                dataset=self.dataset,
                **self.load_kwargs
            )
            
            self.X_list = [np.array(X, dtype=float) for X in self.data["X_list"]]
            self.clinical_df = self.data.get('clinical')
            
            logger.info(f"Loaded data: N={self.X_list[0].shape[0]}, M={len(self.X_list)} views")
            logger.info(f"View shapes: {[X.shape for X in self.X_list]}")
            
            if self.clinical_df is not None:
                logger.info(f"Clinical variables: {list(self.clinical_df.columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def inspect_cv_strategies(self, cv_types: List[str] = None):
        """Inspect different CV strategies and their behavior."""
        if not self.load_data():
            return
            
        if cv_types is None:
            cv_types = ['standard', 'stratified', 'grouped', 'repeated']
        
        logger.info("=== CROSS-VALIDATION STRATEGY INSPECTION ===")
        
        X_ref = self.X_list[0]  # Reference matrix for CV splitting
        n_samples = X_ref.shape[0]
        
        config = CVConfig()
        config.outer_cv_folds = 5
        
        splitter = AdvancedCVSplitter(config)
        
        for cv_type in cv_types:
            logger.info(f"\n--- {cv_type.upper()} CV ---")
            
            try:
                # Prepare variables for CV type
                y = None
                groups = None
                
                if cv_type == 'stratified':
                    # Create dummy stratification variable
                    y = np.random.choice([0, 1], size=n_samples)
                    logger.info("Using dummy binary stratification variable")
                    
                elif cv_type == 'grouped':
                    # Create dummy groups
                    n_groups = max(5, n_samples // 10)
                    groups = np.random.choice(n_groups, size=n_samples)
                    logger.info(f"Using dummy groups (n_groups={n_groups})")
                
                # Create CV splitter
                cv = splitter.create_cv_splitter(X_ref, y, groups, cv_type)
                
                # Analyze splits
                fold_sizes = []
                fold_info = []
                
                splits = list(cv.split(X_ref, y, groups))
                
                for fold_id, (train_idx, test_idx) in enumerate(splits):
                    train_size = len(train_idx)
                    test_size = len(test_idx)
                    test_pct = test_size / n_samples * 100
                    
                    fold_sizes.append(test_size)
                    fold_info.append({
                        'fold': fold_id + 1,
                        'train_size': train_size,
                        'test_size': test_size,
                        'test_pct': test_pct
                    })
                
                # Summary statistics
                logger.info(f"Number of folds: {len(splits)}")
                logger.info(f"Test set sizes: {fold_sizes}")
                logger.info(f"Test set %: {[f'{info['test_pct']:.1f}%' for info in fold_info]}")
                logger.info(f"Average test size: {np.mean(fold_sizes):.1f} ± {np.std(fold_sizes):.1f}")
                
                # Check for overlap (should be zero)
                all_test_indices = set()
                overlaps = 0
                for _, test_idx in splits:
                    test_set = set(test_idx)
                    overlap_size = len(all_test_indices.intersection(test_set))
                    overlaps += overlap_size
                    all_test_indices.update(test_set)
                
                logger.info(f"Index overlaps between folds: {overlaps} (should be 0)")
                logger.info(f"Total unique test indices: {len(all_test_indices)}/{n_samples}")
                
            except Exception as e:
                logger.error(f"Failed to create {cv_type} CV: {e}")
    
    def debug_single_fold(self, fold_id: int = 0, quick_test: bool = True):
        """Debug a single CV fold with detailed logging."""
        if not self.load_data():
            return
            
        logger.info(f"=== DEBUGGING SINGLE FOLD (fold {fold_id + 1}) ===")
        
        # Setup minimal args for testing
        import argparse
        args = argparse.Namespace()
        args.K = 5
        args.num_samples = 200 if quick_test else 1000
        args.num_warmup = 100 if quick_test else 500
        args.num_chains = 1
        args.model = 'sparseGFA'
        args.percW = 33
        args.reghsZ = True
        args.device = 'cpu'
        args.num_sources = len(self.X_list)
        args.seed = 42
        
        # Setup hyperparameters
        hypers = {
            'a_sigma': 1, 'b_sigma': 1,
            'nu_local': 1, 'nu_global': 1,
            'slab_scale': 2, 'slab_df': 4,
            'percW': args.percW,
            'Dm': [X.shape[1] for X in self.X_list]
        }
        
        logger.info(f"Test configuration: K={args.K}, samples={args.num_samples}, chains={args.num_chains}")
        
        # Create CV split
        config = CVConfig()
        config.outer_cv_folds = 5
        splitter = AdvancedCVSplitter(config)
        
        cv = splitter.create_cv_splitter(self.X_list[0])
        splits = list(cv.split(self.X_list[0]))
        
        if fold_id >= len(splits):
            logger.error(f"Fold {fold_id} not available (only {len(splits)} folds)")
            return
        
        train_idx, test_idx = splits[fold_id]
        logger.info(f"Fold {fold_id + 1}: train={len(train_idx)}, test={len(test_idx)}")
        
        # Prepare fold data
        X_train_list = [X[train_idx] for X in self.X_list]
        X_test_list = [X[test_idx] for X in self.X_list]
        
        logger.info("Training data shapes:")
        for i, X_train in enumerate(X_train_list):
            logger.info(f"  View {i+1}: {X_train.shape}")
        
        # Simple scaling
        scalers = []
        for m, X_train in enumerate(X_train_list):
            mu = np.mean(X_train, axis=0, keepdims=True)
            sigma = np.std(X_train, axis=0, keepdims=True)
            sigma = np.where(sigma < 1e-8, 1.0, sigma)
            
            X_train_list[m] = (X_train - mu) / sigma
            X_test_list[m] = (X_test_list[m] - mu) / sigma
            scalers.append((mu, sigma))
        
        logger.info("Data scaled successfully")
        
        try:
            # Import run_analysis here for single fold test
            import run_analysis as RA
            
            # Run inference
            logger.info("Starting MCMC inference...")
            start_time = time.time()
            
            rng = jrandom.PRNGKey(args.seed)
            mcmc = RA.run_inference(RA.models, args, rng, X_train_list, hypers)
            samples = mcmc.get_samples()
            
            inference_time = time.time() - start_time
            logger.info(f"MCMC completed in {inference_time:.1f}s")
            
            # Basic diagnostics
            logger.info("=== MCMC DIAGNOSTICS ===")
            
            for param_name in ['W', 'Z', 'sigma']:
                if param_name in samples:
                    param_samples = samples[param_name]
                    logger.info(f"{param_name}: shape={param_samples.shape}")
                    
                    # Check for NaNs/Infs
                    n_nans = np.sum(np.isnan(param_samples))
                    n_infs = np.sum(np.isinf(param_samples))
                    if n_nans > 0 or n_infs > 0:
                        logger.warning(f"  {param_name} has {n_nans} NaNs, {n_infs} Infs")
                    
                    # Basic statistics
                    mean_val = np.mean(param_samples)
                    std_val = np.std(param_samples)
                    logger.info(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
            
            # Extract point estimates
            W_mean = np.array(samples['W']).mean(axis=0)
            Z_mean = np.array(samples['Z']).mean(axis=0)
            
            # Evaluate reconstruction on test set
            logger.info("=== EVALUATING RECONSTRUCTION ===")
            
            X_test_concat = np.concatenate(X_test_list, axis=1)
            X_test_recon = Z_mean @ W_mean.T
            
            # Basic metrics
            mse = mean_squared_error(X_test_concat, X_test_recon)
            r2 = r2_score(X_test_concat, X_test_recon)
            
            logger.info(f"Test MSE: {mse:.4f}")
            logger.info(f"Test R²: {r2:.4f}")
            
            # Per-view metrics
            d = 0
            for m, dim in enumerate(hypers['Dm']):
                X_test_view = X_test_list[m]
                X_recon_view = X_test_recon[:, d:d+dim]
                
                view_r2 = r2_score(X_test_view, X_recon_view)
                logger.info(f"  View {m+1} R²: {view_r2:.4f}")
                d += dim
            
            logger.info("Single fold debugging completed successfully!")
            
            return {
                'inference_time': inference_time,
                'test_mse': mse,
                'test_r2': r2,
                'samples': samples
            }
            
        except Exception as e:
            logger.error(f"Single fold debugging failed: {e}")
            logger.exception("Full traceback:")
            return None
    
    def quick_test(self, n_folds: int = 3):
        """Run a quick CV test with minimal parameters."""
        if not self.load_data():
            return
            
        logger.info("=== QUICK CROSS-VALIDATION TEST ===")
        logger.info("Using minimal parameters for speed")
        
        # Setup minimal configuration
        config = CVConfig()
        config.outer_cv_folds = n_folds
        config.n_jobs = 1  # Sequential for debugging
        config.timeout_seconds = 300  # 5 minute timeout
        
        # Setup minimal args
        import argparse
        args = argparse.Namespace()
        args.K = 3
        args.num_samples = 100
        args.num_warmup = 50
        args.num_chains = 1
        args.model = 'sparseGFA'
        args.percW = 33
        args.reghsZ = True
        args.device = 'cpu'
        args.num_sources = len(self.X_list)
        args.seed = 42
        
        # Setup hyperparameters
        hypers = {
            'a_sigma': 1, 'b_sigma': 1,
            'nu_local': 1, 'nu_global': 1,
            'slab_scale': 2, 'slab_df': 4,
            'percW': args.percW,
            'Dm': [X.shape[1] for X in self.X_list]
        }
        
        logger.info(f"Quick test: {n_folds} folds, K={args.K}, {args.num_samples} samples")
        
        try:
            # Initialize CV
            cv = SparseBayesianGFACrossValidator(config)
            
            # Run standard CV
            start_time = time.time()
            results = cv.standard_cross_validate(
                self.X_list, args, hypers, cv_type='standard'
            )
            total_time = time.time() - start_time
            
            logger.info(f"Quick CV completed in {total_time:.1f}s")
            logger.info(f"Mean CV score: {results.get('mean_cv_score', 'N/A'):.4f}")
            logger.info(f"Converged folds: {results.get('n_converged_folds', 'N/A')}/{n_folds}")
            
            return results
            
        except Exception as e:
            logger.error(f"Quick test failed: {e}")
            logger.exception("Full traceback:")
            return None
    
    def compare_cv_strategies(self):
        """Compare different CV strategies on the same data."""
        if not self.load_data():
            return
        
        logger.info("=== COMPARING CV STRATEGIES ===")
        
        strategies = ['standard', 'stratified', 'grouped']
        
        # Create dummy variables for testing
        n_samples = self.X_list[0].shape[0]
        y_dummy = np.random.choice([0, 1], size=n_samples)  # Binary for stratification
        groups_dummy = np.random.choice(5, size=n_samples)   # 5 groups
        
        results = {}
        
        for strategy in strategies:
            logger.info(f"\nTesting {strategy} CV strategy...")
            
            try:
                # Prepare variables
                y = y_dummy if strategy == 'stratified' else None
                groups = groups_dummy if strategy == 'grouped' else None
                
                # Create splitter
                config = CVConfig()
                config.outer_cv_folds = 3  # Small for testing
                splitter = AdvancedCVSplitter(config)
                
                cv = splitter.create_cv_splitter(self.X_list[0], y, groups, strategy)
                splits = list(cv.split(self.X_list[0], y, groups))
                
                # Analyze splits
                fold_sizes = [len(test_idx) for _, test_idx in splits]
                fold_balance = np.std(fold_sizes) / np.mean(fold_sizes)  # CV of fold sizes
                
                results[strategy] = {
                    'n_folds': len(splits),
                    'fold_sizes': fold_sizes,
                    'balance_metric': fold_balance,
                    'success': True
                }
                
                logger.info(f"  ✓ {strategy}: {len(splits)} folds, balance metric: {fold_balance:.4f}")
                
            except Exception as e:
                logger.error(f"  ✗ {strategy}: failed - {e}")
                results[strategy] = {'success': False, 'error': str(e)}
        
        # Summary comparison
        logger.info("\n=== STRATEGY COMPARISON SUMMARY ===")
        logger.info("Strategy\t\tFolds\tBalance\tStatus")
        logger.info("-" * 50)
        
        for strategy, result in results.items():
            if result['success']:
                balance = result['balance_metric']
                status = "✓ Success"
                logger.info(f"{strategy:15s}\t{result['n_folds']}\t{balance:.4f}\t{status}")
            else:
                logger.info(f"{strategy:15s}\t-\t-\t✗ Failed")
        
        return results

# == CLI MAIN FUNCTION ==

def main():
    """Main function for standalone CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-Validation Module - Standalone Mode")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="qmap_pd", choices=["qmap_pd"])
    parser.add_argument("--data_dir", type=str, default="qMAP-PD_data")
    parser.add_argument("--clinical_rel", type=str, default="data_clinical/pd_motor_gfa_data.tsv")
    parser.add_argument("--volumes_rel", type=str, default="volume_matrices")
    parser.add_argument("--id_col", type=str, default="sid")
    parser.add_argument("--roi_views", action="store_true")
    
    # Model arguments (for testing)
    parser.add_argument("--K", type=int, default=15, help="Number of factors")
    parser.add_argument("--num_samples", type=int, default=1000, help="MCMC samples")
    parser.add_argument("--num_warmup", type=int, default=500, help="MCMC warmup")
    parser.add_argument("--num_chains", type=int, default=1, help="MCMC chains")
    parser.add_argument("--percW", type=int, default=33, help="Sparsity percentage")
    parser.add_argument("--model", type=str, default="sparseGFA", choices=["sparseGFA", "GFA"])
    parser.add_argument("--reghsZ", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Inspection modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--inspect_cv_strategy", action="store_true",
                           help="Inspect CV strategies without running models")
    mode_group.add_argument("--debug_single_fold", action="store_true",
                           help="Debug a single CV fold with detailed logging")
    mode_group.add_argument("--quick_test", action="store_true",
                           help="Run quick CV test with minimal parameters")
    mode_group.add_argument("--compare_cv_strategies", action="store_true",
                           help="Compare different CV strategies")
    
    # CV parameters
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--cv_type", type=str, default="standard",
                       choices=["standard", "stratified", "grouped", "repeated"])
    parser.add_argument("--nested_cv", action="store_true")
    parser.add_argument("--fold_id", type=int, default=0, 
                       help="Fold ID for single fold debugging")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--group_col", type=str, default=None)
    
    # Preprocessing (for data loading)
    parser.add_argument("--enable_preprocessing", action="store_true")
    parser.add_argument("--feature_selection", type=str, default='variance')
    parser.add_argument("--n_top_features", type=int, default=None)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="cv_inspection")
    parser.add_argument("--run_name", type=str, default="cv_analysis")
    
    args = parser.parse_args()
    
    # Setup
    numpyro.set_platform(args.device)
    np.random.seed(args.seed)
    
    # Create inspector
    load_kwargs = {
        'data_dir': args.data_dir,
        'clinical_rel': args.clinical_rel,
        'volumes_rel': args.volumes_rel,
        'imaging_as_single_view': not args.roi_views,
        'id_col': args.id_col,
        'enable_advanced_preprocessing': args.enable_preprocessing,
        'feature_selection_method': args.feature_selection,
        'n_top_features': args.n_top_features,
    }
    
    inspector = CVInspector(args.dataset, **load_kwargs)
    
    # Run requested analysis
    if args.inspect_cv_strategy:
        inspector.inspect_cv_strategies()
        
    elif args.debug_single_fold:
        inspector.debug_single_fold(args.fold_id, quick_test=True)
        
    elif args.quick_test:
        inspector.quick_test(args.cv_folds)
        
    elif args.compare_cv_strategies:
        inspector.compare_cv_strategies()
        
    else:
        # Full CV analysis
        logger.info("Running full cross-validation analysis...")
        logger.info("(This may take a while - consider using --quick_test first)")
        
        # Load data
        if not inspector.load_data():
            logger.error("Failed to load data")
            return
        
        # Prepare variables
        y = None
        groups = None
        
        if args.target_col and inspector.clinical_df is not None:
            if args.target_col in inspector.clinical_df.columns:
                y = inspector.clinical_df[args.target_col].values
                logger.info(f"Using {args.target_col} for stratification")
        
        if args.group_col and inspector.clinical_df is not None:
            if args.group_col in inspector.clinical_df.columns:
                groups = inspector.clinical_df[args.group_col].values
                logger.info(f"Using {args.group_col} for grouping")
        
        # Setup hyperparameters
        hypers = {
            'a_sigma': 1, 'b_sigma': 1,
            'nu_local': 1, 'nu_global': 1,
            'slab_scale': 2, 'slab_df': 4,
            'percW': args.percW,
            'Dm': [X.shape[1] for X in inspector.X_list]
        }
        
        # Configure CV
        config = CVConfig()
        config.outer_cv_folds = args.cv_folds
        config.n_jobs = args.n_jobs
        config.random_state = args.seed
        
        # Initialize CV
        cv = SparseBayesianGFACrossValidator(config)
        
        # Run CV
        if args.nested_cv:
            search_space = {
                'K': [args.K//2, args.K, args.K*2] if args.K > 5 else [5, 10, 15],
                'percW': [25, 33, 50],
                'num_samples': [args.num_samples//2, args.num_samples] if args.num_samples > 1000 else [800, 1000]
            }
            results = cv.nested_cross_validate(
                inspector.X_list, args, hypers, y, groups, args.cv_type, search_space
            )
        else:
            results = cv.standard_cross_validate(
                inspector.X_list, args, hypers, y, groups, args.cv_type
            )
        
        # Save results
        output_dir = Path(args.output_dir)
        cv.save_results(output_dir, args.run_name)
        
        # Create visualizations if available
        try:
            from visualization import plot_cv_results
            plot_cv_results(results, str(output_dir), args.run_name)
            logger.info("CV visualizations created")
        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")
        
        logger.info("Cross-validation analysis completed!")
        logger.info(f"Results saved to {output_dir}")

# == SCRIPT ENTRY POINT ==

if __name__ == "__main__":
    # Runs when the script is executed directly
    main()
else:
    # Runs when the module is imported
    logger.info("Cross-validation module imported successfully")
