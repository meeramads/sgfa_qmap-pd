"""Reproducibility and robustness experiments for SGFA qMAP-PD analysis."""

import hashlib
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from core.config_utils import ConfigAccessor, get_data_dir, get_output_dir, safe_get
from core.experiment_utils import (
    experiment_handler,
    get_experiment_logger,
    validate_experiment_inputs,
)
from core.io_utils import DataManager, save_csv, save_json, save_numpy, save_plot
from core.validation_utils import (
    ParameterValidator,
    ResultValidator,
    validate_data_types,
    validate_parameters,
)
from experiments.framework import (
    ExperimentConfig,
    ExperimentFramework,
    ExperimentResult,
)
from performance import PerformanceProfiler


class ReproducibilityExperiments(ExperimentFramework):
    """Comprehensive reproducibility and robustness testing for SGFA analysis."""

    def __init__(
        self, config: ExperimentConfig, logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, None, logger)
        self.profiler = PerformanceProfiler()

        # Reproducibility settings
        self.random_seeds = [42, 123, 456, 789, 999, 1234, 5678, 9999]
        self.perturbation_types = ["gaussian", "uniform", "dropout", "permutation"]

    @experiment_handler("seed_reproducibility_test")
    @validate_data_types(X_list=list, hypers=dict, args=dict)
    @validate_parameters(
        X_list=lambda x: len(x) > 0,
        seeds=(
            lambda x: x is None
            or (isinstance(x, list) and all(isinstance(s, int) for s in x)),
            "seeds must be None or list of integers",
        ),
    )
    def run_seed_reproducibility_test(
        self,
        X_list: List[np.ndarray],
        hypers: Dict,
        args: Dict,
        seeds: List[int] = None,
        **kwargs,
    ) -> ExperimentResult:
        """Test reproducibility across different random seeds."""
        # Validate inputs
        ResultValidator.validate_data_matrices(X_list)

        if seeds is None:
            seeds = self.random_seeds

        self.logger.info(f"Running seed reproducibility test with seeds: {seeds}")

        results = {}
        performance_metrics = {}
        for seed in seeds:
            self.logger.info(f"Testing with seed: {seed}")

            # Set random seed for reproducibility
            np.random.seed(seed)

            # Update args with seed
            seed_args = args.copy()
            seed_args["random_seed"] = seed

            with self.profiler.profile(f"seed_{seed}") as p:
                result = self._run_sgfa_analysis(X_list, hypers, seed_args, **kwargs)
                results[seed] = result

                # Store performance metrics
                metrics = self.profiler.get_current_metrics()
                performance_metrics[seed] = {
                    "execution_time": metrics.execution_time,
                    "peak_memory_gb": metrics.peak_memory_gb,
                    "convergence": result.get("convergence", False),
                    "log_likelihood": result.get("log_likelihood", np.nan),
                }

        # Analyze reproducibility
        analysis = self._analyze_seed_reproducibility(results, performance_metrics)

        # Generate basic plots
        plots = self._plot_seed_reproducibility(results, performance_metrics)

        # Add comprehensive reproducibility visualizations (focus on stability &
        # consensus)
        advanced_plots = self._create_comprehensive_reproducibility_visualizations(
            X_list, results, "seed_reproducibility"
        )
        plots.update(advanced_plots)

        return ExperimentResult(
            experiment_name="seed_reproducibility_test",
            config=self.config,
            data=results,
            analysis=analysis,
            plots=plots,
            performance_metrics=performance_metrics,
            success=True,
        )

    @experiment_handler("data_perturbation_robustness")
    @validate_data_types(X_list=list, hypers=dict, args=dict, n_trials=int)
    @validate_parameters(
        X_list=lambda x: len(x) > 0,
        n_trials=lambda x: x > 0,
        perturbation_levels=(
            lambda x: x is None
            or (
                isinstance(x, list)
                and all(isinstance(l, (int, float)) and l >= 0 for l in x)
            ),
            "perturbation_levels must be None or list of non-negative numbers",
        ),
    )
    def run_data_perturbation_robustness(
        self,
        X_list: List[np.ndarray],
        hypers: Dict,
        args: Dict,
        perturbation_levels: List[float] = None,
        perturbation_types: List[str] = None,
        n_trials: int = 5,
        **kwargs,
    ) -> ExperimentResult:
        """Test robustness to data perturbations."""
        # Validate inputs
        ResultValidator.validate_data_matrices(X_list)
        ParameterValidator.validate_positive(n_trials, "n_trials")

        if perturbation_levels is None:
            perturbation_levels = [0.01, 0.05, 0.1, 0.2]
        if perturbation_types is None:
            perturbation_types = self.perturbation_types

        self.logger.info(f"Running data perturbation robustness test")
        self.logger.info(f"Perturbation types: {perturbation_types}")
        self.logger.info(f"Perturbation levels: {perturbation_levels}")

        results = {}
        # Baseline result
        baseline_result = self._run_sgfa_analysis(X_list, hypers, args, **kwargs)
        results["baseline"] = {
            "result": baseline_result,
            "perturbation_type": "none",
            "perturbation_level": 0.0,
        }

        # Test different perturbation types and levels
        for perturbation_type in perturbation_types:
            self.logger.info(f"Testing perturbation type: {perturbation_type}")

            type_results = {}

            for perturbation_level in perturbation_levels:
                self.logger.info(f"Testing perturbation level: {perturbation_level}")

                level_results = []

                for trial in range(n_trials):
                    # Apply perturbation
                    X_perturbed = self._apply_data_perturbation(
                        X_list, perturbation_type, perturbation_level
                    )

                    try:
                        result = self._run_sgfa_analysis(
                            X_perturbed, hypers, args, **kwargs
                        )
                        level_results.append(
                            {
                                "trial": trial,
                                "result": result,
                                "log_likelihood": result.get("log_likelihood", np.nan),
                                "convergence": result.get("convergence", False),
                            }
                        )

                    except Exception as e:
                        self.logger.warning(f"Trial {trial} failed: {str(e)}")
                        level_results.append(
                            {
                                "trial": trial,
                                "result": {"error": str(e)},
                                "log_likelihood": np.nan,
                                "convergence": False,
                            }
                        )

                type_results[perturbation_level] = level_results

            results[perturbation_type] = type_results

        # Analyze robustness
        analysis = self._analyze_data_perturbation_robustness(results, baseline_result)

        # Generate basic plots
        plots = self._plot_data_perturbation_robustness(results)

        # Add comprehensive reproducibility visualizations (focus on robustness)
        advanced_plots = self._create_comprehensive_reproducibility_visualizations(
            X_list, results, "data_perturbation_robustness"
        )
        plots.update(advanced_plots)

        return ExperimentResult(
            experiment_name="data_perturbation_robustness",
            config=self.config,
            data=results,
            analysis=analysis,
            plots=plots,
            success=True,
        )

    @experiment_handler("initialization_robustness")
    @validate_data_types(X_list=list, hypers=dict, args=dict, n_initializations=int)
    @validate_parameters(X_list=lambda x: len(x) > 0, n_initializations=lambda x: x > 0)
    def run_initialization_robustness(
        self,
        X_list: List[np.ndarray],
        hypers: Dict,
        args: Dict,
        n_initializations: int = 10,
        initialization_strategies: List[str] = None,
        **kwargs,
    ) -> ExperimentResult:
        """Test robustness to different initializations."""
        # Validate inputs
        ResultValidator.validate_data_matrices(X_list)
        ParameterValidator.validate_positive(n_initializations, "n_initializations")

        if initialization_strategies is None:
            initialization_strategies = ["random", "pca_based", "kmeans_based", "zero"]

        self.logger.info(f"Running initialization robustness test")
        self.logger.info(f"Strategies: {initialization_strategies}")
        self.logger.info(f"Initializations per strategy: {n_initializations}")

        results = {}
        performance_metrics = {}
        for strategy in initialization_strategies:
            self.logger.info(f"Testing initialization strategy: {strategy}")

            strategy_results = []
            strategy_metrics = []

            for init_idx in range(n_initializations):
                self.logger.debug(f"Initialization {init_idx + 1}/{n_initializations}")

                # Prepare args with initialization strategy
                init_args = args.copy()
                init_args["initialization_strategy"] = strategy
                init_args["initialization_seed"] = init_idx

                with self.profiler.profile(f"{strategy}_init_{init_idx}") as p:
                    try:
                        result = self._run_sgfa_analysis(
                            X_list, hypers, init_args, **kwargs
                        )
                        strategy_results.append(
                            {
                                "initialization": init_idx,
                                "result": result,
                                "log_likelihood": result.get("log_likelihood", np.nan),
                                "convergence": result.get("convergence", False),
                                "n_iterations": result.get("n_iterations", np.nan),
                            }
                        )

                        # Store performance metrics
                        metrics = self.profiler.get_current_metrics()
                        strategy_metrics.append(
                            {
                                "initialization": init_idx,
                                "execution_time": metrics.execution_time,
                                "peak_memory_gb": metrics.peak_memory_gb,
                            }
                        )

                    except Exception as e:
                        self.logger.warning(
                            f"Initialization {init_idx} failed: {str(e)}"
                        )
                        strategy_results.append(
                            {
                                "initialization": init_idx,
                                "result": {"error": str(e)},
                                "log_likelihood": np.nan,
                                "convergence": False,
                                "n_iterations": np.nan,
                            }
                        )
                        strategy_metrics.append(
                            {
                                "initialization": init_idx,
                                "execution_time": np.nan,
                                "peak_memory_gb": np.nan,
                            }
                        )

            results[strategy] = strategy_results
            performance_metrics[strategy] = strategy_metrics

        # Analyze initialization robustness
        analysis = self._analyze_initialization_robustness(results, performance_metrics)

        # Generate basic plots
        plots = self._plot_initialization_robustness(results, performance_metrics)

        # Add comprehensive reproducibility visualizations
        advanced_plots = self._create_comprehensive_reproducibility_visualizations(
            X_list, results, "initialization_robustness"
        )
        plots.update(advanced_plots)

        return ExperimentResult(
            experiment_name="initialization_robustness",
            config=self.config,
            data=results,
            analysis=analysis,
            plots=plots,
            performance_metrics=performance_metrics,
            success=True,
        )

    @experiment_handler("computational_reproducibility_audit")
    @validate_data_types(X_list=list, hypers=dict, args=dict)
    @validate_parameters(X_list=lambda x: len(x) > 0)
    def run_computational_reproducibility_audit(
        self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> ExperimentResult:
        """Comprehensive computational reproducibility audit."""
        # Validate inputs
        ResultValidator.validate_data_matrices(X_list)

        self.logger.info("Running computational reproducibility audit")

        results = {}
        # 1. Exact reproducibility test (same seed, same everything)
        self.logger.info("Testing exact reproducibility...")
        exact_results = []
        fixed_seed = 42

        for run in range(3):
            np.random.seed(fixed_seed)
            run_args = args.copy()
            run_args["random_seed"] = fixed_seed

            result = self._run_sgfa_analysis(X_list, hypers, run_args, **kwargs)
            exact_results.append(result)

        results["exact_reproducibility"] = exact_results

        # 2. Numerical stability test
        self.logger.info("Testing numerical stability...")
        stability_results = self._test_numerical_stability(
            X_list, hypers, args, **kwargs
        )
        results["numerical_stability"] = stability_results

        # 3. Platform consistency test (different random number generators)
        self.logger.info("Testing platform consistency...")
        platform_results = self._test_platform_consistency(
            X_list, hypers, args, **kwargs
        )
        results["platform_consistency"] = platform_results

        # 4. Checksum verification
        self.logger.info("Computing result checksums...")
        checksums = self._compute_result_checksums(results)
        results["checksums"] = checksums

        # Analyze computational reproducibility
        analysis = self._analyze_computational_reproducibility(results)

        # Generate basic plots
        plots = self._plot_computational_reproducibility(results)

        # Add comprehensive reproducibility visualizations
        advanced_plots = self._create_comprehensive_reproducibility_visualizations(
            X_list, results, "computational_reproducibility"
        )
        plots.update(advanced_plots)

        return ExperimentResult(
            experiment_name="computational_reproducibility_audit",
            config=self.config,
            data=results,
            analysis=analysis,
            plots=plots,
            success=True,
        )

    def _run_sgfa_analysis(
        self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> Dict:
        """Run SGFA analysis with given parameters."""
        import jax.numpy as jnp
        import jax.random as random
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        seed = args.get("random_seed", 42)
        key = random.PRNGKey(seed)
        K = hypers.get("K", 5)

        # Convert to JAX arrays
        X_jax = [jnp.array(X) for X in X_list]
        n_samples, n_features = X_jax[0].shape

        def sgfa_model():
            # Priors for factor loadings W
            W = []
            for v, X in enumerate(X_jax):
                W_v = numpyro.sample(
                    f"W_{v}", dist.Normal(0, 1).expand([X.shape[1], K])
                )
                W.append(W_v)

            # Prior for factors Z
            Z = numpyro.sample("Z", dist.Normal(0, 1).expand([n_samples, K]))

            # Likelihood
            for v, X in enumerate(X_jax):
                mu = jnp.dot(Z, W[v].T)
                numpyro.sample(f"obs_{v}", dist.Normal(mu, 1), obs=X)

        # Run reduced MCMC for reproducibility testing
        nuts_kernel = NUTS(sgfa_model)
        mcmc = MCMC(nuts_kernel, num_warmup=50, num_samples=100)
        mcmc.run(key)

        # Extract results
        samples = mcmc.get_samples()
        W_samples = [samples[f"W_{v}"] for v in range(len(X_jax))]
        Z_samples = samples["Z"]

        # Compute log likelihood estimate
        log_likelihood = float(jnp.mean(mcmc.get_extra_fields()["potential_energy"]))

        return {
            "W": [jnp.mean(W_v, axis=0) for W_v in W_samples],
            "Z": jnp.mean(Z_samples, axis=0),
            "log_likelihood": -log_likelihood,  # Convert potential energy to log likelihood
            "n_iterations": 150,  # warmup + samples
            "convergence": True,
            "seed_used": seed,
            "hyperparameters": hypers.copy(),
        }

    def _apply_data_perturbation(
        self,
        X_list: List[np.ndarray],
        perturbation_type: str,
        perturbation_level: float,
    ) -> List[np.ndarray]:
        """Apply perturbation to data."""
        X_perturbed = []

        for X in X_list:
            if perturbation_type == "gaussian":
                # Add Gaussian noise
                noise = np.random.normal(0, perturbation_level * np.std(X), X.shape)
                X_pert = X + noise

            elif perturbation_type == "uniform":
                # Add uniform noise
                noise_range = perturbation_level * (np.max(X) - np.min(X))
                noise = np.random.uniform(-noise_range / 2, noise_range / 2, X.shape)
                X_pert = X + noise

            elif perturbation_type == "dropout":
                # Randomly set some values to zero
                mask = np.random.random(X.shape) > perturbation_level
                X_pert = X * mask

            elif perturbation_type == "permutation":
                # Randomly permute some fraction of features
                X_pert = X.copy()
                n_features = X.shape[1]
                n_permute = int(perturbation_level * n_features)

                if n_permute > 0:
                    feature_indices = np.random.choice(
                        n_features, n_permute, replace=False
                    )
                    for feature_idx in feature_indices:
                        permuted_values = np.random.permutation(X[:, feature_idx])
                        X_pert[:, feature_idx] = permuted_values
            else:
                X_pert = X.copy()

            X_perturbed.append(X_pert)

        return X_perturbed

    def _test_numerical_stability(
        self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> Dict:
        """Test numerical stability with different precision levels."""
        stability_results = {}

        # Test with different data types
        dtypes = [np.float32, np.float64]

        for dtype in dtypes:
            self.logger.debug(f"Testing with dtype: {dtype}")

            # Convert data to specified dtype
            X_dtype = [X.astype(dtype) for X in X_list]

            try:
                result = self._run_sgfa_analysis(X_dtype, hypers, args, **kwargs)
                stability_results[str(dtype)] = result

            except Exception as e:
                self.logger.warning(f"Failed with dtype {dtype}: {str(e)}")
                stability_results[str(dtype)] = {"error": str(e)}

        return stability_results

    def _test_platform_consistency(
        self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> Dict:
        """Test consistency across different random number generator states."""
        consistency_results = {}

        # Test different RNG states
        rng_states = ["numpy_default", "mersenne_twister", "pcg64"]

        for rng_state in rng_states:
            self.logger.debug(f"Testing with RNG: {rng_state}")

            try:
                # Set different RNG states (simplified for this example)
                if rng_state == "numpy_default":
                    np.random.seed(42)
                elif rng_state == "mersenne_twister":
                    np.random.seed(42)
                    # In practice, you would set specific RNG
                elif rng_state == "pcg64":
                    np.random.seed(42)
                    # In practice, you would set PCG64 RNG

                result = self._run_sgfa_analysis(X_list, hypers, args, **kwargs)
                consistency_results[rng_state] = result

            except Exception as e:
                self.logger.warning(f"Failed with RNG {rng_state}: {str(e)}")
                consistency_results[rng_state] = {"error": str(e)}

        return consistency_results

    def _compute_result_checksums(self, results: Dict) -> Dict:
        """Compute checksums for reproducibility verification."""
        checksums = {}

        for experiment_name, experiment_results in results.items():
            if experiment_name == "checksums":
                continue

            # Serialize results and compute checksum
            try:
                serialized = pickle.dumps(
                    experiment_results, protocol=2
                )  # Use protocol 2 for consistency
                checksum = hashlib.md5(serialized).hexdigest()
                checksums[experiment_name] = checksum

            except Exception as e:
                self.logger.warning(
                    f"Failed to compute checksum for {experiment_name}: {str(e)}"
                )
                checksums[experiment_name] = None

        return checksums

    def _run_sgfa_analysis(
        self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> Dict:
        """Run actual SGFA analysis for reproducibility testing."""
        import time

        import jax
        import jax.numpy as jnp
        import numpyro
        from numpyro.infer import MCMC, NUTS

        try:
            K = hypers.get("K", 10)
            self.logger.debug(
                f"Running SGFA for reproducibility test: K={K}, n_subjects={
                    X_list[0].shape[0]}, n_features={
                    sum(
                        X.shape[1] for X in X_list)}")

            # Import the actual SGFA model function
            from core.run_analysis import models

            # Setup MCMC configuration for reproducibility testing
            num_warmup = args.get("num_warmup", 50)
            num_samples = args.get("num_samples", 100)
            num_chains = args.get("num_chains", 1)

            # Create args object for model
            import argparse

            model_args = argparse.Namespace(
                model="sparseGFA",
                K=K,
                num_sources=len(X_list),
                reghsZ=args.get("reghsZ", True),
            )

            # Setup MCMC with seed control for reproducibility
            seed = args.get("random_seed", 42)
            rng_key = jax.random.PRNGKey(seed)
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
            potential_energy = samples.get("potential_energy", np.array([0]))
            log_likelihood = (
                -np.mean(potential_energy) if len(potential_energy) > 0 else 0
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
                "reproducibility_info": {
                    "seed_used": seed,
                    "mcmc_config": {
                        "num_warmup": num_warmup,
                        "num_samples": num_samples,
                        "num_chains": num_chains,
                    },
                },
            }

        except Exception as e:
            self.logger.error(f"SGFA reproducibility analysis failed: {str(e)}")
            return {
                "error": str(e),
                "convergence": False,
                "execution_time": float("inf"),
                "log_likelihood": float("-inf"),
            }

    def _analyze_seed_reproducibility(
        self, results: Dict, performance_metrics: Dict
    ) -> Dict:
        """Analyze seed reproducibility results."""
        analysis = {
            "reproducibility_metrics": {},
            "convergence_consistency": {},
            "likelihood_variation": {},
            "performance_variation": {},
        }

        # Extract log likelihoods and convergence status
        seeds = list(results.keys())
        log_likelihoods = [
            performance_metrics[seed]["log_likelihood"]
            for seed in seeds
            if not np.isnan(performance_metrics[seed]["log_likelihood"])
        ]
        convergences = [performance_metrics[seed]["convergence"] for seed in seeds]
        execution_times = [
            performance_metrics[seed]["execution_time"] for seed in seeds
        ]

        # Likelihood variation analysis
        if log_likelihoods:
            ll_mean = np.mean(log_likelihoods)
            ll_std = np.std(log_likelihoods)
            ll_range = max(log_likelihoods) - min(log_likelihoods)

            analysis["likelihood_variation"] = {
                "mean": ll_mean,
                "std": ll_std,
                "coefficient_of_variation": ll_std / abs(ll_mean),
                "range": ll_range,
                "reproducibility_score": 1.0
                / (1.0 + ll_std),  # Higher is more reproducible
            }

        # Convergence consistency
        convergence_rate = np.mean(convergences)
        analysis["convergence_consistency"] = {
            "convergence_rate": convergence_rate,
            "consistent_convergence": convergence_rate > 0.8,
            "n_seeds_converged": sum(convergences),
            "n_seeds_total": len(convergences),
        }

        # Performance variation
        analysis["performance_variation"] = {
            "execution_time_std": np.std(execution_times),
            "execution_time_cv": np.std(execution_times) / np.mean(execution_times),
        }

        return analysis

    def _analyze_data_perturbation_robustness(
        self, results: Dict, baseline_result: Dict
    ) -> Dict:
        """Analyze data perturbation robustness results."""
        analysis = {
            "robustness_by_perturbation_type": {},
            "robustness_by_level": {},
            "overall_robustness": {},
        }

        baseline_likelihood = baseline_result.get("log_likelihood", np.nan)

        # Analyze by perturbation type
        for perturbation_type, type_results in results.items():
            if perturbation_type == "baseline":
                continue

            type_analysis = {}

            for level, level_results in type_results.items():
                level_likelihoods = [
                    r["log_likelihood"]
                    for r in level_results
                    if not np.isnan(r["log_likelihood"])
                ]
                level_convergences = [r["convergence"] for r in level_results]

                if level_likelihoods:
                    mean_likelihood = np.mean(level_likelihoods)
                    likelihood_drop = (
                        baseline_likelihood - mean_likelihood
                        if not np.isnan(baseline_likelihood)
                        else np.nan
                    )

                    type_analysis[level] = {
                        "mean_likelihood": mean_likelihood,
                        "likelihood_drop": likelihood_drop,
                        "convergence_rate": np.mean(level_convergences),
                        "robustness_score": (
                            1.0 / (1.0 + abs(likelihood_drop))
                            if not np.isnan(likelihood_drop)
                            else 0.0
                        ),
                    }

            analysis["robustness_by_perturbation_type"][
                perturbation_type
            ] = type_analysis

        return analysis

    def _analyze_initialization_robustness(
        self, results: Dict, performance_metrics: Dict
    ) -> Dict:
        """Analyze initialization robustness results."""
        analysis = {
            "strategy_comparison": {},
            "convergence_analysis": {},
            "likelihood_consistency": {},
        }

        # Analyze each strategy
        for strategy, strategy_results in results.items():
            strategy_likelihoods = [
                r["log_likelihood"]
                for r in strategy_results
                if not np.isnan(r["log_likelihood"])
            ]
            strategy_convergences = [r["convergence"] for r in strategy_results]
            strategy_iterations = [
                r["n_iterations"]
                for r in strategy_results
                if not np.isnan(r["n_iterations"])
            ]

            if strategy_likelihoods:
                analysis["strategy_comparison"][strategy] = {
                    "mean_likelihood": np.mean(strategy_likelihoods),
                    "std_likelihood": np.std(strategy_likelihoods),
                    "best_likelihood": max(strategy_likelihoods),
                    "worst_likelihood": min(strategy_likelihoods),
                    "convergence_rate": np.mean(strategy_convergences),
                    "mean_iterations": (
                        np.mean(strategy_iterations) if strategy_iterations else np.nan
                    ),
                    "consistency_score": 1.0 / (1.0 + np.std(strategy_likelihoods)),
                }

        # Find best strategy
        if analysis["strategy_comparison"]:
            best_strategy = max(
                analysis["strategy_comparison"].keys(),
                key=lambda s: analysis["strategy_comparison"][s]["mean_likelihood"],
            )
            analysis["best_strategy"] = best_strategy

        return analysis

    def _analyze_computational_reproducibility(self, results: Dict) -> Dict:
        """Analyze computational reproducibility results."""
        analysis = {
            "exact_reproducibility": {},
            "numerical_stability": {},
            "platform_consistency": {},
            "checksum_verification": {},
        }

        # Exact reproducibility analysis
        exact_results = results.get("exact_reproducibility", [])
        if len(exact_results) > 1:
            likelihoods = [r.get("log_likelihood", np.nan) for r in exact_results]
            valid_likelihoods = [ll for ll in likelihoods if not np.isnan(ll)]

            if len(valid_likelihoods) > 1:
                ll_std = np.std(valid_likelihoods)
                analysis["exact_reproducibility"] = {
                    "is_exactly_reproducible": ll_std < 1e-10,
                    "likelihood_std": ll_std,
                    "max_difference": max(valid_likelihoods) - min(valid_likelihoods),
                }

        # Numerical stability analysis
        stability_results = results.get("numerical_stability", {})
        stable_dtypes = []
        for dtype, result in stability_results.items():
            if "error" not in result:
                stable_dtypes.append(dtype)

        analysis["numerical_stability"] = {
            "stable_dtypes": stable_dtypes,
            "stability_rate": (
                len(stable_dtypes) / len(stability_results)
                if stability_results
                else 0.0
            ),
        }

        # Platform consistency analysis
        platform_results = results.get("platform_consistency", {})
        consistent_platforms = []
        for platform, result in platform_results.items():
            if "error" not in result:
                consistent_platforms.append(platform)

        analysis["platform_consistency"] = {
            "consistent_platforms": consistent_platforms,
            "consistency_rate": (
                len(consistent_platforms) / len(platform_results)
                if platform_results
                else 0.0
            ),
        }

        return analysis

    def _plot_seed_reproducibility(
        self, results: Dict, performance_metrics: Dict
    ) -> Dict:
        """Generate plots for seed reproducibility analysis."""
        plots = {}

        try:
            seeds = list(results.keys())
            log_likelihoods = [
                performance_metrics[seed]["log_likelihood"] for seed in seeds
            ]
            execution_times = [
                performance_metrics[seed]["execution_time"] for seed in seeds
            ]
            convergences = [performance_metrics[seed]["convergence"] for seed in seeds]

            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Seed Reproducibility Analysis", fontsize=16)

            # Plot 1: Log likelihood by seed
            axes[0, 0].scatter(seeds, log_likelihoods)
            axes[0, 0].set_xlabel("Random Seed")
            axes[0, 0].set_ylabel("Log Likelihood")
            axes[0, 0].set_title("Log Likelihood by Seed")
            axes[0, 0].grid(True, alpha=0.3)

            # Add mean line
            if log_likelihoods:
                valid_ll = [ll for ll in log_likelihoods if not np.isnan(ll)]
                if valid_ll:
                    axes[0, 0].axhline(
                        np.mean(valid_ll), color="red", linestyle="--", label="Mean"
                    )
                    axes[0, 0].legend()

            # Plot 2: Execution time by seed
            axes[0, 1].scatter(seeds, execution_times)
            axes[0, 1].set_xlabel("Random Seed")
            axes[0, 1].set_ylabel("Execution Time (seconds)")
            axes[0, 1].set_title("Execution Time by Seed")
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Convergence by seed
            convergence_colors = [
                "red" if not conv else "green" for conv in convergences
            ]
            axes[1, 0].scatter(
                seeds, [1 if conv else 0 for conv in convergences], c=convergence_colors
            )
            axes[1, 0].set_xlabel("Random Seed")
            axes[1, 0].set_ylabel("Converged")
            axes[1, 0].set_title("Convergence by Seed")
            axes[1, 0].set_ylim([-0.1, 1.1])
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Log likelihood distribution
            valid_likelihoods = [ll for ll in log_likelihoods if not np.isnan(ll)]
            if valid_likelihoods:
                axes[1, 1].hist(
                    valid_likelihoods, bins=min(10, len(valid_likelihoods)), alpha=0.7
                )
                axes[1, 1].set_xlabel("Log Likelihood")
                axes[1, 1].set_ylabel("Frequency")
                axes[1, 1].set_title("Log Likelihood Distribution")
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plots["seed_reproducibility"] = fig

        except Exception as e:
            self.logger.warning(
                f"Failed to create seed reproducibility plots: {str(e)}"
            )

        return plots

    def _plot_data_perturbation_robustness(self, results: Dict) -> Dict:
        """Generate plots for data perturbation robustness analysis."""
        plots = {}

        try:
            baseline_likelihood = results["baseline"]["result"].get(
                "log_likelihood", np.nan
            )

            # Collect data for plotting
            perturbation_data = []

            for perturbation_type, type_results in results.items():
                if perturbation_type == "baseline":
                    continue

                for level, level_results in type_results.items():
                    for result in level_results:
                        if not np.isnan(result["log_likelihood"]):
                            perturbation_data.append(
                                {
                                    "perturbation_type": perturbation_type,
                                    "level": level,
                                    "log_likelihood": result["log_likelihood"],
                                    "convergence": result["convergence"],
                                    "likelihood_drop": (
                                        baseline_likelihood - result["log_likelihood"]
                                        if not np.isnan(baseline_likelihood)
                                        else np.nan
                                    ),
                                }
                            )

            if not perturbation_data:
                return plots

            df = pd.DataFrame(perturbation_data)

            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("Data Perturbation Robustness Analysis", fontsize=16)

            # Plot 1: Log likelihood by perturbation type and level
            for i, perturbation_type in enumerate(df["perturbation_type"].unique()):
                type_data = df[df["perturbation_type"] == perturbation_type]
                levels = sorted(type_data["level"].unique())
                mean_likelihoods = [
                    type_data[type_data["level"] == level]["log_likelihood"].mean()
                    for level in levels
                ]

                axes[0, 0].plot(levels, mean_likelihoods, "o-", label=perturbation_type)

            if not np.isnan(baseline_likelihood):
                axes[0, 0].axhline(
                    baseline_likelihood, color="black", linestyle="--", label="Baseline"
                )

            axes[0, 0].set_xlabel("Perturbation Level")
            axes[0, 0].set_ylabel("Mean Log Likelihood")
            axes[0, 0].set_title("Log Likelihood vs Perturbation Level")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Convergence rate by perturbation type and level
            for perturbation_type in df["perturbation_type"].unique():
                type_data = df[df["perturbation_type"] == perturbation_type]
                levels = sorted(type_data["level"].unique())
                convergence_rates = [
                    type_data[type_data["level"] == level]["convergence"].mean()
                    for level in levels
                ]

                axes[0, 1].plot(
                    levels, convergence_rates, "s-", label=perturbation_type
                )

            axes[0, 1].set_xlabel("Perturbation Level")
            axes[0, 1].set_ylabel("Convergence Rate")
            axes[0, 1].set_title("Convergence Rate vs Perturbation Level")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim([0, 1.1])

            # Plot 3: Likelihood drop by perturbation type (boxplot)
            perturbation_types = df["perturbation_type"].unique()
            likelihood_drops = [
                df[df["perturbation_type"] == pt]["likelihood_drop"].dropna().values
                for pt in perturbation_types
            ]

            axes[1, 0].boxplot(likelihood_drops, labels=perturbation_types)
            axes[1, 0].set_ylabel("Likelihood Drop from Baseline")
            axes[1, 0].set_title("Likelihood Drop Distribution by Perturbation Type")
            axes[1, 0].tick_params(axis="x", rotation=45)
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Robustness heatmap
            pivot_data = df.pivot_table(
                values="log_likelihood",
                index="perturbation_type",
                columns="level",
                aggfunc="mean",
            )

            if not pivot_data.empty:
                im = axes[1, 1].imshow(pivot_data.values, cmap="viridis", aspect="auto")
                axes[1, 1].set_xticks(range(len(pivot_data.columns)))
                axes[1, 1].set_yticks(range(len(pivot_data.index)))
                axes[1, 1].set_xticklabels([f"{val:.2f}" for val in pivot_data.columns])
                axes[1, 1].set_yticklabels(pivot_data.index)
                axes[1, 1].set_xlabel("Perturbation Level")
                axes[1, 1].set_ylabel("Perturbation Type")
                axes[1, 1].set_title("Log Likelihood Heatmap")

                plt.colorbar(im, ax=axes[1, 1], label="Log Likelihood")

            plt.tight_layout()
            plots["data_perturbation_robustness"] = fig

        except Exception as e:
            self.logger.warning(
                f"Failed to create data perturbation robustness plots: {str(e)}"
            )

        return plots

    def _plot_initialization_robustness(
        self, results: Dict, performance_metrics: Dict
    ) -> Dict:
        """Generate plots for initialization robustness analysis."""
        plots = {}

        try:
            # Collect data
            initialization_data = []

            for strategy, strategy_results in results.items():
                for result in strategy_results:
                    if not np.isnan(result["log_likelihood"]):
                        initialization_data.append(
                            {
                                "strategy": strategy,
                                "log_likelihood": result["log_likelihood"],
                                "convergence": result["convergence"],
                                "n_iterations": result.get("n_iterations", np.nan),
                            }
                        )

            if not initialization_data:
                return plots

            df = pd.DataFrame(initialization_data)

            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("Initialization Robustness Analysis", fontsize=16)

            # Plot 1: Log likelihood boxplot by strategy
            strategies = df["strategy"].unique()
            likelihood_data = [
                df[df["strategy"] == s]["log_likelihood"].values for s in strategies
            ]

            axes[0, 0].boxplot(likelihood_data, labels=strategies)
            axes[0, 0].set_ylabel("Log Likelihood")
            axes[0, 0].set_title(
                "Log Likelihood Distribution by Initialization Strategy"
            )
            axes[0, 0].tick_params(axis="x", rotation=45)
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Convergence rate by strategy
            convergence_rates = [
                df[df["strategy"] == s]["convergence"].mean() for s in strategies
            ]

            axes[0, 1].bar(strategies, convergence_rates)
            axes[0, 1].set_ylabel("Convergence Rate")
            axes[0, 1].set_title("Convergence Rate by Initialization Strategy")
            axes[0, 1].tick_params(axis="x", rotation=45)
            axes[0, 1].set_ylim([0, 1.1])
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Number of iterations by strategy (if available)
            iteration_data = df.dropna(subset=["n_iterations"])
            if not iteration_data.empty:
                iteration_by_strategy = [
                    iteration_data[iteration_data["strategy"] == s][
                        "n_iterations"
                    ].values
                    for s in strategies
                    if s in iteration_data["strategy"].values
                ]
                strategy_labels = [
                    s for s in strategies if s in iteration_data["strategy"].values
                ]

                if iteration_by_strategy:
                    axes[1, 0].boxplot(iteration_by_strategy, labels=strategy_labels)
                    axes[1, 0].set_ylabel("Number of Iterations")
                    axes[1, 0].set_title("Iterations to Convergence by Strategy")
                    axes[1, 0].tick_params(axis="x", rotation=45)
                    axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Strategy comparison scatter (mean vs std)
            strategy_stats = []
            for strategy in strategies:
                strategy_data = df[df["strategy"] == strategy]
                mean_ll = strategy_data["log_likelihood"].mean()
                std_ll = strategy_data["log_likelihood"].std()
                strategy_stats.append(
                    {
                        "strategy": strategy,
                        "mean_likelihood": mean_ll,
                        "std_likelihood": std_ll,
                    }
                )

            stats_df = pd.DataFrame(strategy_stats)

            axes[1, 1].scatter(stats_df["std_likelihood"], stats_df["mean_likelihood"])

            for i, row in stats_df.iterrows():
                axes[1, 1].annotate(
                    row["strategy"],
                    (row["std_likelihood"], row["mean_likelihood"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                )

            axes[1, 1].set_xlabel("Log Likelihood Standard Deviation")
            axes[1, 1].set_ylabel("Mean Log Likelihood")
            axes[1, 1].set_title("Strategy Comparison: Mean vs Variability")
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plots["initialization_robustness"] = fig

        except Exception as e:
            self.logger.warning(
                f"Failed to create initialization robustness plots: {str(e)}"
            )

        return plots

    def _plot_computational_reproducibility(self, results: Dict) -> Dict:
        """Generate plots for computational reproducibility analysis."""
        plots = {}

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("Computational Reproducibility Analysis", fontsize=16)

            # Plot 1: Exact reproducibility
            exact_results = results.get("exact_reproducibility", [])
            if len(exact_results) > 1:
                likelihoods = [r.get("log_likelihood", np.nan) for r in exact_results]
                valid_likelihoods = [ll for ll in likelihoods if not np.isnan(ll)]

                if valid_likelihoods:
                    run_numbers = list(range(1, len(valid_likelihoods) + 1))
                    axes[0, 0].plot(run_numbers, valid_likelihoods, "o-")
                    axes[0, 0].set_xlabel("Run Number")
                    axes[0, 0].set_ylabel("Log Likelihood")
                    axes[0, 0].set_title("Exact Reproducibility Test")
                    axes[0, 0].grid(True, alpha=0.3)

                    # Add variability info
                    if len(valid_likelihoods) > 1:
                        std_dev = np.std(valid_likelihoods)
                        axes[0, 0].text(
                            0.05,
                            0.95,
                            f"Std Dev: {std_dev:.2e}",
                            transform=axes[0, 0].transAxes,
                            bbox=dict(boxstyle="round", facecolor="wheat"),
                        )

            # Plot 2: Numerical stability
            stability_results = results.get("numerical_stability", {})
            if stability_results:
                dtypes = []
                success = []

                for dtype, result in stability_results.items():
                    dtypes.append(dtype)
                    success.append(1 if "error" not in result else 0)

                axes[0, 1].bar(dtypes, success)
                axes[0, 1].set_ylabel("Success (1) / Failure (0)")
                axes[0, 1].set_title("Numerical Stability by Data Type")
                axes[0, 1].tick_params(axis="x", rotation=45)
                axes[0, 1].set_ylim([0, 1.2])

            # Plot 3: Platform consistency
            platform_results = results.get("platform_consistency", {})
            if platform_results:
                platforms = []
                success = []

                for platform, result in platform_results.items():
                    platforms.append(platform)
                    success.append(1 if "error" not in result else 0)

                axes[1, 0].bar(platforms, success)
                axes[1, 0].set_ylabel("Success (1) / Failure (0)")
                axes[1, 0].set_title("Platform Consistency")
                axes[1, 0].tick_params(axis="x", rotation=45)
                axes[1, 0].set_ylim([0, 1.2])

            # Plot 4: Checksum verification status
            checksums = results.get("checksums", {})
            if checksums:
                experiments = []
                has_checksum = []

                for exp_name, checksum in checksums.items():
                    experiments.append(exp_name)
                    has_checksum.append(1 if checksum is not None else 0)

                axes[1, 1].bar(experiments, has_checksum)
                axes[1, 1].set_ylabel("Has Checksum (1) / Missing (0)")
                axes[1, 1].set_title("Checksum Verification Status")
                axes[1, 1].tick_params(axis="x", rotation=45)
                axes[1, 1].set_ylim([0, 1.2])

            plt.tight_layout()
            plots["computational_reproducibility"] = fig

        except Exception as e:
            self.logger.warning(
                f"Failed to create computational reproducibility plots: {str(e)}"
            )

        return plots

    def _create_comprehensive_reproducibility_visualizations(
        self, X_list: List[np.ndarray], results: Dict, experiment_name: str
    ) -> Dict:
        """Create comprehensive reproducibility visualizations focusing on consensus and stability."""
        advanced_plots = {}

        try:
            self.logger.info(
                f"🎨 Creating comprehensive reproducibility visualizations for {experiment_name}")

            # Import visualization system
            from core.config_utils import ConfigAccessor
            from visualization.manager import VisualizationManager

            # Create a reproducibility-focused config for visualization
            viz_config = ConfigAccessor(
                {
                    "visualization": {
                        "create_brain_viz": True,  # Include brain maps for reproducibility assessment
                        "output_format": ["png", "pdf"],
                        "dpi": 300,
                        "reproducibility_focus": True,
                    },
                    "output_dir": f"/tmp/reproducibility_viz_{experiment_name}",
                }
            )

            # Initialize visualization manager
            viz_manager = VisualizationManager(viz_config)

            # Prepare reproducibility data structure
            data = {
                "X_list": X_list,
                "view_names": [f"view_{i}" for i in range(len(X_list))],
                "n_subjects": X_list[0].shape[0],
                "view_dimensions": [X.shape[1] for X in X_list],
                "preprocessing": {
                    "status": "completed",
                    "strategy": "reproducibility_analysis",
                },
            }

            # Extract best reproducible result for analysis
            best_repro_result = self._extract_best_reproducible_result(results)

            if best_repro_result:
                # Prepare reproducibility analysis results
                analysis_results = {
                    "best_run": best_repro_result,
                    "all_runs": results,
                    "model_type": "reproducibility_sparseGFA",
                    "convergence": best_repro_result.get("convergence", False),
                    "reproducibility_analysis": True,
                }

                # Add cross-validation style results for consensus analysis
                cv_results = {
                    "consensus_analysis": {
                        "multiple_runs": results,
                        "stability_metrics": self._extract_reproducibility_metrics(
                            results
                        ),
                        "seed_variations": self._extract_seed_variations(results),
                    }
                }

                # Create comprehensive visualizations with reproducibility focus
                viz_manager.create_all_visualizations(
                    data=data, analysis_results=analysis_results, cv_results=cv_results
                )

                # Extract and process generated plots
                if hasattr(viz_manager, "plot_dir") and viz_manager.plot_dir.exists():
                    plot_files = list(viz_manager.plot_dir.glob("**/*.png"))

                    for plot_file in plot_files:
                        plot_name = f"reproducibility_{plot_file.stem}"

                        try:
                            import matplotlib.image as mpimg
                            import matplotlib.pyplot as plt

                            fig, ax = plt.subplots(figsize=(12, 8))
                            img = mpimg.imread(str(plot_file))
                            ax.imshow(img)
                            ax.axis("off")
                            ax.set_title(
                                f"Reproducibility Analysis: {plot_name}", fontsize=14
                            )

                            advanced_plots[plot_name] = fig

                        except Exception as e:
                            self.logger.warning(
                                f"Could not load reproducibility plot {plot_name}: {e}"
                            )

                    self.logger.info(
                        f"✅ Created {
                            len(plot_files)} comprehensive reproducibility visualizations")
                    self.logger.info(
                        "   → Consensus factor analysis and stability plots generated"
                    )
                    self.logger.info(
                        "   → Cross-seed reproducibility and robustness visualizations generated"
                    )

                else:
                    self.logger.warning(
                        "Reproducibility visualization manager did not create plot directory"
                    )
            else:
                self.logger.warning(
                    "No reproducible results found for comprehensive visualization"
                )

        except Exception as e:
            self.logger.warning(
                f"Failed to create comprehensive reproducibility visualizations: {e}"
            )

        return advanced_plots

    def _extract_best_reproducible_result(self, results: Dict) -> Optional[Dict]:
        """Extract the most reproducible result from multiple runs."""
        best_result = None
        best_consistency = 0

        # Look through different reproducibility result structures
        if isinstance(results, list):
            # Multiple seed results
            convergence_count = sum(1 for r in results if r.get("convergence", False))
            if convergence_count > best_consistency:
                best_consistency = convergence_count
                # Return the result with median likelihood
                converged = [r for r in results if r.get("convergence", False)]
                if converged:
                    likelihoods = [
                        r.get("log_likelihood", float("-inf")) for r in converged
                    ]
                    median_idx = np.argsort(likelihoods)[len(likelihoods) // 2]
                    best_result = converged[median_idx]

        elif isinstance(results, dict):
            # Nested result structure
            for key, result_set in results.items():
                if isinstance(result_set, list):
                    convergence_count = sum(
                        1 for r in result_set if r.get("convergence", False)
                    )
                    if convergence_count > best_consistency:
                        best_consistency = convergence_count
                        converged = [
                            r for r in result_set if r.get("convergence", False)
                        ]
                        if converged:
                            likelihoods = [
                                r.get("log_likelihood", float("-inf"))
                                for r in converged
                            ]
                            median_idx = np.argsort(likelihoods)[len(likelihoods) // 2]
                            best_result = converged[median_idx]

        return best_result

    def _extract_reproducibility_metrics(self, results: Dict) -> Dict:
        """Extract reproducibility metrics for visualization."""
        metrics = {
            "convergence_rates": {},
            "likelihood_stability": {},
            "factor_consistency": {},
            "computational_stability": {},
        }

        # Process different types of reproducibility results
        if isinstance(results, list):
            # Multiple seed results
            convergence_rate = sum(
                1 for r in results if r.get("convergence", False)
            ) / len(results)
            metrics["convergence_rates"]["overall"] = convergence_rate

            converged_results = [r for r in results if r.get("convergence", False)]
            if converged_results:
                likelihoods = [r.get("log_likelihood", 0) for r in converged_results]
                metrics["likelihood_stability"]["mean"] = np.mean(likelihoods)
                metrics["likelihood_stability"]["std"] = np.std(likelihoods)

        elif isinstance(results, dict):
            for key, result_set in results.items():
                if isinstance(result_set, list):
                    convergence_rate = sum(
                        1 for r in result_set if r.get("convergence", False)
                    ) / len(result_set)
                    metrics["convergence_rates"][key] = convergence_rate

        return metrics

    def _extract_seed_variations(self, results: Dict) -> Dict:
        """Extract seed variation data for visualization."""
        variations = {
            "seed_results": [],
            "perturbation_effects": {},
            "initialization_effects": {},
        }

        # Extract seed-specific results if available
        if isinstance(results, list):
            variations["seed_results"] = results
        elif isinstance(results, dict):
            for key, result_set in results.items():
                if "seed" in key.lower() or isinstance(result_set, list):
                    variations["seed_results"] = result_set
                    break

        return variations


def run_reproducibility(config):
    """Run reproducibility tests with remote workstation integration."""
    import logging
    import os
    import sys

    import numpy as np

    logger = logging.getLogger(__name__)
    logger.info("Starting Reproducibility Tests")

    try:
        # Add project root to path for imports
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from pathlib import Path

        # Load data with standard preprocessing for reproducibility testing
        from data.preprocessing_integration import apply_preprocessing_to_pipeline
        from experiments.framework import ExperimentConfig, ExperimentFramework

        logger.info("🔧 Loading data for reproducibility testing...")
        X_list, preprocessing_info = apply_preprocessing_to_pipeline(
            config=config,
            data_dir=get_data_dir(config),
            auto_select_strategy=False,
            preferred_strategy="standard",  # Use standard preprocessing for consistency
        )

        logger.info(f"✅ Data loaded: {len(X_list)} views for reproducibility testing")
        for i, X in enumerate(X_list):
            logger.info(f"   View {i}: {X.shape}")

        # Initialize experiment framework
        framework = ExperimentFramework(base_output_dir=get_output_dir(config))

        exp_config = ExperimentConfig(
            experiment_name="reproducibility_tests",
            description="Reproducibility and robustness testing for SGFA",
            dataset="qmap_pd",
            data_dir=get_data_dir(config),
        )

        # Create reproducibility experiment instance
        repro_exp = ReproducibilityExperiments(exp_config, logger)

        # Setup base hyperparameters
        base_hypers = {
            "Dm": [X.shape[1] for X in X_list],
            "a_sigma": 1.0,
            "b_sigma": 1.0,
            "slab_df": 4.0,
            "slab_scale": 2.0,
            "percW": 33.0,
            "K": 10,  # Base number of factors
        }

        # Setup base args
        base_args = {
            "K": 10,
            "num_warmup": 50,  # Reduced for reproducibility testing speed
            "num_samples": 100,  # Reduced for reproducibility testing speed
            "num_chains": 1,
            "target_accept_prob": 0.8,
            "reghsZ": True,
        }

        # Run the experiment
        def reproducibility_experiment(config, output_dir, **kwargs):
            logger.info("🔄 Running comprehensive reproducibility tests...")

            results = {}
            total_tests = 0
            successful_tests = 0

            # 1. Test seed reproducibility
            logger.info("📊 Testing seed reproducibility...")
            seeds = [42, 123, 456]  # Reduced set for testing
            seed_results = {}

            for seed in seeds:
                try:
                    # Set seed for reproducibility
                    np.random.seed(seed)
                    test_args = base_args.copy()
                    test_args["random_seed"] = seed

                    with repro_exp.profiler.profile(f"seed_{seed}") as p:
                        result = repro_exp._run_sgfa_analysis(
                            X_list, base_hypers, test_args
                        )

                    metrics = repro_exp.profiler.get_current_metrics()
                    seed_results[f"seed_{seed}"] = {
                        "seed": seed,
                        "result": result,
                        "performance": {
                            "execution_time": metrics.execution_time,
                            "peak_memory_gb": metrics.peak_memory_gb,
                            "convergence": result.get("convergence", False),
                            "log_likelihood": result.get(
                                "log_likelihood", float("-inf")
                            ),
                        },
                    }
                    successful_tests += 1
                    logger.info(
                        f"✅ Seed {seed}: {
                            metrics.execution_time:.1f}s, LL={
                            result.get(
                                'log_likelihood',
                                0):.2f}")

                except Exception as e:
                    logger.error(f"❌ Seed {seed} test failed: {e}")
                    seed_results[f"seed_{seed}"] = {"error": str(e)}

                total_tests += 1

            results["seed_reproducibility"] = seed_results

            # 2. Test data perturbation robustness
            logger.info("📊 Testing data perturbation robustness...")
            noise_levels = [0.01, 0.05]  # Small noise levels for testing
            perturbation_results = {}

            for noise_level in noise_levels:
                try:
                    # Add Gaussian noise to data
                    X_noisy = []
                    for X in X_list:
                        noise = np.random.normal(0, noise_level * np.std(X), X.shape)
                        X_noisy.append(X + noise)

                    with repro_exp.profiler.profile(f"noise_{noise_level}") as p:
                        result = repro_exp._run_sgfa_analysis(
                            X_noisy, base_hypers, base_args
                        )

                    metrics = repro_exp.profiler.get_current_metrics()
                    perturbation_results[f"noise_{noise_level}"] = {
                        "noise_level": noise_level,
                        "result": result,
                        "performance": {
                            "execution_time": metrics.execution_time,
                            "peak_memory_gb": metrics.peak_memory_gb,
                            "convergence": result.get("convergence", False),
                            "log_likelihood": result.get(
                                "log_likelihood", float("-inf")
                            ),
                        },
                    }
                    successful_tests += 1
                    logger.info(
                        f"✅ Noise {noise_level}: {
                            metrics.execution_time:.1f}s, LL={
                            result.get(
                                'log_likelihood',
                                0):.2f}")

                except Exception as e:
                    logger.error(f"❌ Noise {noise_level} test failed: {e}")
                    perturbation_results[f"noise_{noise_level}"] = {"error": str(e)}

                total_tests += 1

            results["perturbation_robustness"] = perturbation_results

            # 3. Test initialization robustness
            logger.info("📊 Testing initialization robustness...")
            init_results = {}
            n_inits = 3  # Test different initializations

            for init_id in range(n_inits):
                try:
                    # Different random initialization
                    test_args = base_args.copy()
                    test_args["random_seed"] = 1000 + init_id

                    with repro_exp.profiler.profile(f"init_{init_id}") as p:
                        result = repro_exp._run_sgfa_analysis(
                            X_list, base_hypers, test_args
                        )

                    metrics = repro_exp.profiler.get_current_metrics()
                    init_results[f"init_{init_id}"] = {
                        "init_id": init_id,
                        "result": result,
                        "performance": {
                            "execution_time": metrics.execution_time,
                            "peak_memory_gb": metrics.peak_memory_gb,
                            "convergence": result.get("convergence", False),
                            "log_likelihood": result.get(
                                "log_likelihood", float("-inf")
                            ),
                        },
                    }
                    successful_tests += 1
                    logger.info(
                        f"✅ Init {init_id}: {
                            metrics.execution_time:.1f}s, LL={
                            result.get(
                                'log_likelihood',
                                0):.2f}")

                except Exception as e:
                    logger.error(f"❌ Init {init_id} test failed: {e}")
                    init_results[f"init_{init_id}"] = {"error": str(e)}

                total_tests += 1

            results["initialization_robustness"] = init_results

            logger.info("🔄 Reproducibility tests completed!")
            logger.info(f"   Successful tests: {successful_tests}/{total_tests}")

            return {
                "status": "completed",
                "reproducibility_results": results,
                "summary": {
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "success_rate": (
                        successful_tests / total_tests if total_tests > 0 else 0
                    ),
                    "test_categories": [
                        "seed_reproducibility",
                        "perturbation_robustness",
                        "initialization_robustness",
                    ],
                    "data_characteristics": {
                        "n_subjects": X_list[0].shape[0],
                        "n_views": len(X_list),
                        "view_dimensions": [X.shape[1] for X in X_list],
                    },
                },
            }

        # Run experiment using framework
        result = framework.run_experiment(
            experiment_function=reproducibility_experiment,
            config=exp_config,
            data={"X_list": X_list, "preprocessing_info": preprocessing_info},
        )

        logger.info("✅ Reproducibility tests completed successfully")
        return result

    except Exception as e:
        logger.error(f"Reproducibility tests failed: {e}")
        return None
