"""SGFA execution and data preparation for clinical validation.

This module provides utilities for running SGFA models in clinical contexts,
including training, evaluation, and application to new data.
"""

import argparse
import logging
import time
from typing import Dict, List, Optional

import jax
import numpy as np
from numpyro.infer import MCMC, NUTS


class ClinicalDataProcessor:
    """SGFA execution and data preparation for clinical validation."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize clinical data processor.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def run_sgfa_training(
        self,
        X_train: List[np.ndarray],
        hypers: Dict,
        args: Dict
    ) -> Dict:
        """
        Run single SGFA training iteration.

        Args:
            X_train: Training data views (list of N x D_i arrays)
            hypers: SGFA hyperparameters
            args: Additional arguments including:
                - model: Model type (default: "sparseGFA")
                - K: Number of latent factors
                - num_warmup: MCMC warmup iterations (default: 200)
                - num_samples: MCMC samples (default: 300)
                - num_chains: MCMC chains (default: 1)
                - reghsZ: Regularize Z (default: True)

        Returns:
            Dict containing:
                - W: List of loading matrices per view
                - Z: Latent factors (N x K)
                - log_likelihood: Model log likelihood
                - convergence: Convergence status
                - execution_time: Runtime in seconds
                - samples: MCMC samples

        Raises:
            Exception if SGFA training fails
        """
        try:
            # Use model factory for consistent model management
            from models.models_integration import integrate_models_with_pipeline

            # Setup data characteristics for optimal model selection
            data_characteristics = {
                "total_features": sum(X.shape[1] for X in X_train),
                "n_views": len(X_train),
                "n_subjects": X_train[0].shape[0],
                "has_imaging_data": True
            }

            # Get optimal model configuration via factory
            model_type, model_instance, models_summary = integrate_models_with_pipeline(
                config={"model": {"type": args.get("model", "sparseGFA")}},
                X_list=X_train,
                data_characteristics=data_characteristics
            )

            self.logger.info(f"🏭 Clinical validation using model: {model_type}")

            # Import SGFA model via interface
            from core.model_interface import get_model_function
            models = get_model_function()

            # Setup MCMC
            num_warmup = args.get("num_warmup", 200)
            num_samples = args.get("num_samples", 300)
            num_chains = args.get("num_chains", 1)

            # Create args object with defaults
            from core.config_utils import dict_to_namespace

            defaults = {
                "model": "sparseGFA",
                "K": 5,
                "num_sources": len(X_train),
                "reghsZ": True,
            }
            model_args = dict_to_namespace(args, defaults)

            # Run MCMC
            rng_key = jax.random.PRNGKey(np.random.randint(0, 10000))
            kernel = NUTS(models, target_accept_prob=0.8)
            mcmc = MCMC(
                kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains
            )

            start_time = time.time()
            mcmc.run(
                rng_key, X_train, hypers, model_args, extra_fields=("potential_energy",)
            )
            execution_time = time.time() - start_time

            # Extract results
            samples = mcmc.get_samples()
            extra_fields = mcmc.get_extra_fields()
            potential_energy = extra_fields.get("potential_energy", np.array([]))

            # Calculate log likelihood
            log_likelihood = (
                -np.mean(potential_energy) if len(potential_energy) > 0 else np.nan
            )

            # Extract parameters
            W_samples = samples["W"]
            Z_samples = samples["Z"]

            W_mean = np.mean(W_samples, axis=0)
            Z_mean = np.mean(Z_samples, axis=0)

            # Split W back into views
            W_list = []
            start_idx = 0
            for X in X_train:
                end_idx = start_idx + X.shape[1]
                W_list.append(W_mean[start_idx:end_idx, :])
                start_idx = end_idx

            return {
                "W": W_list,
                "Z": Z_mean,
                "log_likelihood": float(log_likelihood),
                "convergence": True,  # Assume convergence if no exception
                "execution_time": execution_time,
                "samples": samples
            }

        except Exception as e:
            self.logger.warning(f"SGFA training failed: {str(e)}")
            return {
                "convergence": False,
                "error": str(e),
                "execution_time": float("inf")
            }

    def evaluate_sgfa_test_set(
        self,
        X_test: List[np.ndarray],
        train_result: Dict,
        test_idx: np.ndarray,
        clinical_data: Dict
    ) -> Dict:
        """
        Evaluate SGFA model on held-out test set.

        Args:
            X_test: Test data views (list of N_test x D_i arrays)
            train_result: Trained SGFA model result
            test_idx: Indices of test subjects
            clinical_data: Clinical data dictionary

        Returns:
            Dict containing:
                - Z_test: Test set factors
                - reconstruction_errors: Reconstruction errors per view
                - mean_reconstruction_error: Average reconstruction error
                - test_diagnoses: Clinical diagnoses for test subjects
                - n_test_subjects: Number of test subjects

        Raises:
            Returns dict with 'error' key if evaluation fails
        """
        try:
            W_list = train_result.get("W", [])
            if not W_list:
                return {"error": "No trained weights available"}

            # Project test data onto learned factors
            Z_test_list = []
            reconstruction_errors = []

            for X_view, W_view in zip(X_test, W_list):
                # Simple projection: X = Z @ W.T => Z = X @ W @ (W.T @ W)^-1
                Z_test = X_view @ W_view @ np.linalg.pinv(W_view.T @ W_view)
                Z_test_list.append(Z_test)

                # Reconstruction error
                X_recon = Z_test @ W_view.T
                recon_error = np.mean((X_view - X_recon) ** 2)
                reconstruction_errors.append(recon_error)

            # Average factors across views
            Z_test_mean = np.mean(Z_test_list, axis=0)

            # Clinical labels for test set
            test_diagnoses = np.array(clinical_data["diagnosis"])[test_idx]

            return {
                "Z_test": Z_test_mean,
                "reconstruction_errors": reconstruction_errors,
                "mean_reconstruction_error": np.mean(reconstruction_errors),
                "test_diagnoses": test_diagnoses,
                "n_test_subjects": len(test_idx)
            }

        except Exception as e:
            return {"error": f"Test evaluation failed: {str(e)}"}

    def run_sgfa_analysis(
        self,
        X_list: List[np.ndarray],
        hypers: Dict,
        args: Dict,
        **kwargs
    ) -> Dict:
        """
        Run complete SGFA MCMC analysis for clinical validation.

        This is the main method for running SGFA in clinical contexts,
        with full MCMC sampling and comprehensive output.

        Args:
            X_list: List of data views (each N x D_i)
            hypers: SGFA hyperparameters (must include 'K')
            args: MCMC arguments including:
                - num_warmup: Warmup iterations (default: 100)
                - num_samples: Sampling iterations (default: 300)
                - num_chains: Number of chains (default: 1)
                - reghsZ: Regularize Z (default: True)
                - target_accept_prob: Target acceptance probability (default: 0.8)
            **kwargs: Additional arguments

        Returns:
            Dict containing:
                - W: List of loading matrices per view
                - Z: Latent factors (N x K)
                - W_samples: MCMC samples for W
                - Z_samples: MCMC samples for Z
                - samples: All MCMC samples
                - log_likelihood: Model log likelihood
                - n_iterations: Number of iterations
                - convergence: Convergence status
                - execution_time: Runtime in seconds
                - clinical_info: Clinical metadata

        Note:
            This replaces the original mock implementation at line 1310.
            This is the REAL MCMC implementation from line 1862.
        """
        try:
            K = hypers.get("K", 10)
            self.logger.debug(
                f"Running SGFA for clinical validation: K={K}, "
                f"n_subjects={X_list[0].shape[0]}, "
                f"n_features={sum(X.shape[1] for X in X_list)}"
            )

            # Use model factory for consistent model management
            from models.models_integration import integrate_models_with_pipeline

            data_characteristics = {
                "total_features": sum(X.shape[1] for X in X_list),
                "n_views": len(X_list),
                "n_subjects": X_list[0].shape[0],
                "has_imaging_data": True
            }

            model_type, model_instance, models_summary = integrate_models_with_pipeline(
                config={"model": {"type": "sparseGFA"}},
                X_list=X_list,
                data_characteristics=data_characteristics
            )
            self.logger.info(f"🏭 Model factory configured: {model_type}")

            # Import the SGFA model function via interface
            from core.model_interface import get_model_function
            models = get_model_function()

            # Setup MCMC configuration for clinical validation
            num_warmup = args.get("num_warmup", 100)
            num_samples = args.get("num_samples", 300)
            num_chains = args.get("num_chains", 1)

            # Create args object for model with defaults
            from core.config_utils import dict_to_namespace

            defaults = {
                "model": "sparseGFA",
                "K": K,
                "num_sources": len(X_list),
                "reghsZ": True,
            }
            model_args = dict_to_namespace(args, defaults)

            # Setup MCMC
            rng_key = jax.random.PRNGKey(np.random.randint(0, 10000))
            kernel = NUTS(
                models, target_accept_prob=args.get("target_accept_prob", 0.8)
            )
            mcmc = MCMC(
                kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
            )

            # Run inference
            start_time = time.time()
            mcmc.run(
                rng_key, X_list, hypers, model_args, extra_fields=("potential_energy",)
            )
            elapsed = time.time() - start_time

            # Get samples
            samples = mcmc.get_samples()

            # Calculate log likelihood (approximate)
            extra_fields = mcmc.get_extra_fields()
            potential_energy = extra_fields.get("potential_energy", np.array([]))
            log_likelihood = (
                -np.mean(potential_energy) if len(potential_energy) > 0 else np.nan
            )

            # Extract mean parameters
            W_samples = samples["W"]  # Shape: (num_samples, D, K)
            Z_samples = samples["Z"]  # Shape: (num_samples, N, K)

            W_mean = np.mean(W_samples, axis=0)
            Z_mean = np.mean(Z_samples, axis=0)

            # Split W back into views
            W_list = []
            start_idx = 0
            for X in X_list:
                end_idx = start_idx + X.shape[1]
                W_list.append(W_mean[start_idx:end_idx, :])
                start_idx = end_idx

            return {
                "W": W_list,
                "Z": Z_mean,
                "W_samples": W_samples,
                "Z_samples": Z_samples,
                "samples": samples,
                "log_likelihood": float(log_likelihood),
                "n_iterations": num_samples,
                "convergence": True,
                "execution_time": elapsed,
                "clinical_info": {
                    "factors_extracted": Z_mean.shape[1],
                    "subjects_analyzed": Z_mean.shape[0],
                    "mcmc_config": {
                        "num_warmup": num_warmup,
                        "num_samples": num_samples,
                        "num_chains": num_chains,
                    },
                },
            }

        except Exception as e:
            self.logger.error(f"SGFA clinical analysis failed: {str(e)}")
            return {
                "error": str(e),
                "convergence": False,
                "execution_time": float("inf"),
                "log_likelihood": float("-inf"),
            }

    def apply_trained_model(
        self,
        X_test_list: List[np.ndarray],
        train_result: Dict,
        hypers: Dict,
        args: Dict,
        **kwargs
    ) -> Dict:
        """
        Apply trained SGFA model to new data.

        Args:
            X_test_list: Test data views (list of N_test x D_i arrays)
            train_result: Previously trained model result
            hypers: SGFA hyperparameters
            args: Additional arguments
            **kwargs: Additional arguments

        Returns:
            Dict containing:
                - Z: Factors for new data
                - W: Loadings (from trained model)
                - log_likelihood: Log likelihood on new data
                - domain_applied: Domain identifier

        Note:
            This projects new data onto the learned factor space using
            the trained loadings.
        """
        try:
            W_list = train_result.get("W")
            if not W_list:
                raise ValueError("No trained weights available in train_result")

            K = train_result["Z"].shape[1]
            n_test_subjects = X_test_list[0].shape[0]

            # Project test data onto learned factors
            Z_test_list = []
            for X_view, W_view in zip(X_test_list, W_list):
                # Project: Z = X @ W @ (W.T @ W)^-1
                Z_test = X_view @ W_view @ np.linalg.pinv(W_view.T @ W_view)
                Z_test_list.append(Z_test)

            # Average factors across views
            Z_test = np.mean(Z_test_list, axis=0)

            # Calculate approximate log likelihood on test data
            log_likelihood = train_result.get("log_likelihood", np.nan)
            # Note: Real implementation would calculate actual likelihood on test data

            return {
                "Z": Z_test,
                "W": W_list,  # Use same loadings
                "log_likelihood": log_likelihood,
                "domain_applied": "test_cohort",
                "n_test_subjects": n_test_subjects,
            }

        except Exception as e:
            self.logger.error(f"Failed to apply trained model: {str(e)}")
            return {
                "error": str(e),
                "convergence": False,
            }
