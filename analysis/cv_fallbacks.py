"""Cross-validation fallback utilities for advanced neuroimaging CV features."""

import logging
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from sklearn.model_selection import StratifiedKFold


logger = logging.getLogger(__name__)


class CVFallbackHandler:
    """Handles fallbacks from advanced neuroimaging CV to basic sklearn CV."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def with_cv_split_fallback(
        self,
        advanced_split_func: Callable,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        clinical_data: Optional[Dict] = None,
        cv_folds: int = 5,
        random_state: int = 42,
        **kwargs
    ) -> List:
        """
        Try advanced CV split first, fallback to sklearn StratifiedKFold only for serious errors.

        Args:
            advanced_split_func: Function that implements advanced CV splitting
            X: Input data for splitting
            y: Target labels (optional, will use clinical_data["diagnosis"] if None)
            groups: Group labels (optional)
            clinical_data: Clinical data dictionary (optional)
            cv_folds: Number of CV folds for fallback
            random_state: Random state for reproducibility
            **kwargs: Additional arguments for advanced_split_func

        Returns:
            List of (train_idx, test_idx) tuples
        """
        try:
            # Try advanced CV splitting
            splits = list(advanced_split_func(
                X=X, y=y, groups=groups, clinical_data=clinical_data, **kwargs
            ))
            self.logger.info(f"✅ Clinical-aware CV split successful: {len(splits)} folds")
            return splits

        except Exception as e:
            self.logger.warning(f"Clinical-aware CV split failed ({e}), falling back to basic sklearn CV")

            # Simplified fallback - only for serious errors (data issues, etc.)
            if y is None and clinical_data:
                y = clinical_data.get("diagnosis", np.zeros(X.shape[0]))
            elif y is None:
                y = np.zeros(X.shape[0])  # Default dummy labels

            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            splits = list(skf.split(X, y))
            self.logger.info(f"✅ Fallback sklearn CV successful: {len(splits)} folds")
            return splits


class MetricsFallbackHandler:
    """Handles fallbacks from advanced neuroimaging metrics to basic metrics."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def with_metrics_fallback(
        self,
        advanced_metrics_func: Callable,
        fallback_metrics: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Try advanced metrics calculation first, fallback to basic metrics.

        Args:
            advanced_metrics_func: Function that implements advanced metrics
            fallback_metrics: Basic metrics to use as fallback
            **kwargs: Arguments for advanced_metrics_func

        Returns:
            Dictionary of calculated metrics
        """
        try:
            # Try advanced metrics calculation
            metrics = advanced_metrics_func(**kwargs)
            self.logger.debug("✅ Advanced metrics calculation successful")
            return metrics

        except (AttributeError, NotImplementedError) as e:
            self.logger.warning(f"Advanced metrics calculation failed ({e}), using basic metrics")
            return fallback_metrics

    def create_basic_fold_metrics(
        self,
        train_result: Dict,
        test_metrics: Dict,
        fold_idx: int,
        **extra_fields
    ) -> Dict[str, Any]:
        """Create basic fallback metrics for fold evaluation."""
        return {
            "mse": test_metrics.get("mse", float('inf')),
            "r2": test_metrics.get("r2", 0.0),
            "convergence": train_result.get("convergence", False),
            "fold_idx": fold_idx,
            **extra_fields
        }

    def create_basic_model_metrics(
        self,
        model_result: Dict,
        model_name: str,
        W_list: Optional[List] = None,
        **extra_fields
    ) -> Dict[str, Any]:
        """Create basic fallback metrics for model comparison."""
        n_factors = 0
        if W_list and len(W_list) > 0 and len(W_list[0]) > 0:
            n_factors = len(W_list[0][0])

        return {
            "mse": model_result.get("mse", float('inf')),
            "r2": model_result.get("r2", 0.0),
            "convergence": model_result.get("convergence", False),
            "n_factors": n_factors,
            "model_name": model_name,
            **extra_fields
        }


class HyperoptFallbackHandler:
    """Handles fallbacks from advanced hyperparameter optimization to basic grid search."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def with_hyperopt_fallback(
        self,
        advanced_hyperopt_func: Callable,
        search_space: Dict,
        objective_function: Callable,
        max_combinations: int = 20,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Try advanced hyperparameter optimization first, fallback to basic grid search.

        Args:
            advanced_hyperopt_func: Function that implements advanced optimization
            search_space: Hyperparameter search space definition
            objective_function: Objective function to optimize
            max_combinations: Maximum parameter combinations to try in fallback
            **kwargs: Arguments for advanced_hyperopt_func

        Returns:
            Dictionary with optimization results
        """
        try:
            # Try advanced hyperparameter optimization
            result = advanced_hyperopt_func(**kwargs)
            self.logger.info("✅ Advanced hyperparameter optimization successful")
            return result

        except (AttributeError, NotImplementedError) as e:
            self.logger.warning(f"Advanced hyperparameter optimization failed ({e}), falling back to basic grid search")
            return self._run_basic_grid_search(search_space, objective_function, max_combinations)

    def _run_basic_grid_search(
        self,
        search_space: Dict,
        objective_function: Callable,
        max_combinations: int
    ) -> Dict[str, Any]:
        """Run basic grid search as fallback."""
        from itertools import product

        self.logger.info("Running basic grid search optimization")

        # Create parameter grids
        param_grids = {}
        for param, config in search_space.items():
            if config['type'] == 'int':
                param_grids[param] = list(range(config['low'], min(config['high'] + 1, config['low'] + 5)))
            elif config['type'] == 'float':
                param_grids[param] = [config['low'], (config['low'] + config['high']) / 2, config['high']]
            elif config['type'] == 'choice':
                param_grids[param] = config['choices'][:3]  # Limit to first 3 choices

        # Generate parameter combinations
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        all_combinations = list(product(*param_values))[:max_combinations]

        best_score = float('-inf')
        best_params = {}
        trials = []

        for i, combination in enumerate(all_combinations):
            params = dict(zip(param_names, combination))
            self.logger.info(f"Testing parameter combination {i+1}/{len(all_combinations)}: {params}")

            try:
                score = objective_function(params)
                trials.append({"params": params, "score": score})

                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                self.logger.warning(f"Parameter combination failed: {e}")
                trials.append({"params": params, "score": float('-inf'), "error": str(e)})

        self.logger.info(f"✅ Basic grid search completed: best_score={best_score}")
        return {
            'best_params': best_params,
            'best_score': best_score,
            'trials': trials,
            'optimization_history': trials,
            'method': 'basic_grid_search'
        }


# Convenience functions for common patterns
def with_cv_fallback(logger: logging.Logger):
    """Decorator factory for CV splitting with fallback."""
    handler = CVFallbackHandler(logger)
    return handler.with_cv_split_fallback


def with_metrics_fallback(logger: logging.Logger):
    """Decorator factory for metrics calculation with fallback."""
    handler = MetricsFallbackHandler(logger)
    return handler.with_metrics_fallback


def with_hyperopt_fallback(logger: logging.Logger):
    """Decorator factory for hyperparameter optimization with fallback."""
    handler = HyperoptFallbackHandler(logger)
    return handler.with_hyperopt_fallback