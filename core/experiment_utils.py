"""Experiment utilities for consistent patterns and error handling."""

import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


def experiment_handler(experiment_name: str):
    """
    Decorator for consistent experiment error handling.

    Automatically catches exceptions and creates failure results using the
    experiment instance's _create_failure_result method.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment for error logging

    Examples
    --------
    >>> class MyExperiment(ExperimentFramework):
    ...     @experiment_handler("my_experiment")
    ...     def run_experiment(self, data):
    ...         # experiment logic here
    ...         return ExperimentResult(...)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"{experiment_name} failed: {error_msg}")
                return self._create_failure_result(experiment_name, error_msg)
        return wrapper
    return decorator


class SGFARunner:
    """
    Standardized SGFA analysis runner to eliminate repeated patterns.

    Provides consistent interface for running SGFA analysis across experiments.
    """

    @staticmethod
    def run_analysis(X_list: List[np.ndarray], hypers: Dict, args: Dict,
                    experiment_name: str = "sgfa_analysis", **kwargs) -> Dict:
        """
        Run SGFA analysis with consistent error handling and logging.

        Parameters
        ----------
        X_list : List[np.ndarray]
            List of data matrices
        hypers : Dict
            Hyperparameters for SGFA
        args : Dict
            Additional arguments for SGFA
        experiment_name : str
            Name for logging purposes
        **kwargs
            Additional keyword arguments

        Returns
        -------
        Dict
            SGFA analysis results or error information
        """
        try:
            # Import JAX components
            import jax
            import jax.numpy as jnp
            import numpyro
            from numpyro.infer import MCMC, NUTS

            logger.info(f"Starting {experiment_name} with {len(X_list)} datasets")

            # Set up SGFA model and inference
            from models.sparse_gfa import SparseGFAModel as SGFAModel

            # Create model instance
            model = SGFAModel(
                K=hypers.get('K', 5),
                **hypers
            )

            # Run MCMC inference
            nuts_kernel = NUTS(model.model)
            mcmc = MCMC(
                nuts_kernel,
                num_warmup=args.get('num_warmup', 1000),
                num_samples=args.get('num_samples', 2000),
                **kwargs
            )

            # Fit model to data
            rng_key = jax.random.PRNGKey(args.get('seed', 42))
            mcmc.run(rng_key, X_list)

            # Extract results
            samples = mcmc.get_samples()

            logger.info(f"{experiment_name} completed successfully")

            return {
                'success': True,
                'samples': samples,
                'mcmc': mcmc,
                'model': model,
                'experiment_name': experiment_name
            }

        except ImportError as e:
            error_msg = f"Missing required dependency for {experiment_name}: {e}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'error_type': 'ImportError'
            }
        except Exception as e:
            error_msg = f"{experiment_name} failed: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'error_type': type(e).__name__
            }


def get_experiment_logger(experiment_name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a standardized logger for experiments.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment
    level : int
        Logging level

    Returns
    -------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(f"experiments.{experiment_name}")

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'%(asctime)s - {experiment_name} - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


def validate_experiment_inputs(required_keys: List[str], **kwargs) -> Dict[str, Any]:
    """
    Validate experiment inputs and provide defaults.

    Parameters
    ----------
    required_keys : List[str]
        List of required parameter names
    **kwargs
        Parameters to validate

    Returns
    -------
    Dict[str, Any]
        Validated parameters

    Raises
    ------
    ValueError
        If required parameters are missing
    """
    missing_keys = [key for key in required_keys if key not in kwargs or kwargs[key] is None]

    if missing_keys:
        raise ValueError(f"Missing required parameters: {', '.join(missing_keys)}")

    return kwargs


class ExperimentMetrics:
    """Helper class for common experiment metrics calculations."""

    @staticmethod
    def calculate_consensus_score(results_list: List[Dict]) -> float:
        """Calculate consensus score across multiple results."""
        if not results_list:
            return 0.0

        # Extract key metrics and calculate variance
        consensus_scores = []
        for key in ['log_likelihood', 'factor_correlation', 'reconstruction_error']:
            values = [r.get(key, 0) for r in results_list if key in r]
            if values:
                consensus_scores.append(1.0 / (1.0 + np.var(values)))

        return np.mean(consensus_scores) if consensus_scores else 0.0

    @staticmethod
    def calculate_stability_score(samples_list: List[Dict]) -> float:
        """Calculate stability score across multiple sampling runs."""
        if not samples_list:
            return 0.0

        # Calculate stability based on factor loadings consistency
        stability_scores = []
        for key in samples_list[0].keys():
            if key.startswith('factor_loading'):
                values = [samples[key] for samples in samples_list if key in samples]
                if values:
                    # Calculate coefficient of variation as stability measure
                    mean_val = np.mean(values, axis=0)
                    std_val = np.std(values, axis=0)
                    cv = np.mean(std_val / (mean_val + 1e-8))
                    stability_scores.append(1.0 / (1.0 + cv))

        return np.mean(stability_scores) if stability_scores else 0.0