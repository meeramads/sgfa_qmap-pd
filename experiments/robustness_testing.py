"""Robustness experiments for SGFA qMAP-PD analysis."""

import hashlib
import logging
import pickle
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpyro.diagnostics import split_gelman_rubin, effective_sample_size

from core.config_utils import get_data_dir, get_output_dir
from core.experiment_utils import (
    experiment_handler,
)
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
from optimization import PerformanceProfiler
from optimization.experiment_mixins import performance_optimized_experiment


@performance_optimized_experiment()
class RobustnessExperiments(ExperimentFramework):
    """Comprehensive Robustness testing for SGFA analysis."""

    def __init__(
        self, config: ExperimentConfig, logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, None, logger)
        self.profiler = PerformanceProfiler()

        # Robustness settings from config
        from core.config_utils import ConfigHelper
        config_dict = ConfigHelper.to_dict(config)
        repro_config = config_dict.get("robustness", {})

        self.random_seeds = repro_config.get("seed_values", [42, 123, 456, 789, 999, 1234, 5678, 9999])
        perturbation_config = repro_config.get("perturbation", {})
        self.perturbation_types = perturbation_config.get("types", ["gaussian", "uniform", "dropout", "permutation"])

    @experiment_handler("seed_robustness_test")
    @validate_data_types(X_list=list, hypers=dict, args=dict)
    @validate_parameters(
        X_list=lambda x: len(x) > 0,
        seeds=(
            lambda x: x is None
            or (isinstance(x, list) and all(isinstance(s, int) for s in x)),
            "seeds must be None or list of integers",
        ),
    )
    def run_seed_robustness_test(
        self,
        X_list: List[np.ndarray],
        hypers: Dict,
        args: Dict,
        seeds: List[int] = None,
        **kwargs,
    ) -> ExperimentResult:
        """Test robustness across different random seeds."""
        # Validate inputs
        ResultValidator.validate_data_matrices(X_list)

        if seeds is None:
            seeds = self.random_seeds

        self.logger.info(f"Running seed robustness test with seeds: {seeds}")

        results = {}
        performance_metrics = {}
        for idx, seed in enumerate(seeds):
            self.logger.info(f"Testing with seed: {seed}")

            # Set random seed for robustness
            np.random.seed(seed)

            # Update args with seed
            seed_args = args.copy()
            seed_args["random_seed"] = seed

            with self.profiler.profile(f"seed_{seed}") as p:
                # Verbose logging only for first seed
                result = self._run_sgfa_analysis(X_list, hypers, seed_args, verbose=(idx == 0), **kwargs)

                # CRITICAL: Remove large sample arrays to prevent memory leak
                if "W_samples" in result:
                    del result["W_samples"]
                if "Z_samples" in result:
                    del result["Z_samples"]
                if "samples" in result:
                    del result["samples"]

                results[seed] = result

                # Store performance metrics
                metrics = self.profiler.get_current_metrics()
                performance_metrics[seed] = {
                    "execution_time": metrics.execution_time,
                    "peak_memory_gb": metrics.peak_memory_gb,
                    "convergence": result.get("convergence", False),
                    "log_likelihood": result.get("log_likelihood", np.nan),
                }

            # Cleanup memory after each seed run to prevent accumulation
            import jax
            import gc
            jax.clear_caches()
            gc.collect()

        # Analyze robustness
        analysis = self._analyze_seed_robustness(results, performance_metrics)

        # Generate basic plots
        plots = self._plot_seed_robustness(results, performance_metrics)

        # Add comprehensive robustness visualizations (focus on stability &
        # consensus)
        advanced_plots = self._create_comprehensive_robustness_visualizations(
            X_list, results, "seed_robustness"
        )
        plots.update(advanced_plots)

        return ExperimentResult(
            experiment_id="seed_robustness_test",
            config=self.config,
            model_results=results,
            diagnostics=analysis,
            plots=plots,
            performance_metrics=performance_metrics,
            status="completed",
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
        # Baseline result (verbose=True for first call)
        baseline_result = self._run_sgfa_analysis(X_list, hypers, args, verbose=True, **kwargs)
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
                            X_perturbed, hypers, args, verbose=False, **kwargs
                        )

                        # CRITICAL: Remove large sample arrays to prevent memory leak
                        if "W_samples" in result:
                            del result["W_samples"]
                        if "Z_samples" in result:
                            del result["Z_samples"]
                        if "samples" in result:
                            del result["samples"]

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

        # Add comprehensive robustness visualizations (focus on robustness)
        advanced_plots = self._create_comprehensive_robustness_visualizations(
            X_list, results, "data_perturbation_robustness"
        )
        plots.update(advanced_plots)

        return ExperimentResult(
            experiment_id="data_perturbation_robustness",
            config=self.config,
            model_results=results,
            diagnostics=analysis,
            plots=plots,
            status="completed",
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
        for strategy_idx, strategy in enumerate(initialization_strategies):
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
                        # Only verbose for first init of first strategy
                        result = self._run_sgfa_analysis(
                            X_list, hypers, init_args, verbose=(strategy_idx == 0 and init_idx == 0), **kwargs
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

        # Add comprehensive robustness visualizations
        advanced_plots = self._create_comprehensive_robustness_visualizations(
            X_list, results, "initialization_robustness"
        )
        plots.update(advanced_plots)

        return ExperimentResult(
            experiment_id="initialization_robustness",
            config=self.config,
            model_results=results,
            diagnostics=analysis,
            plots=plots,
            performance_metrics=performance_metrics,
            status="completed",
        )

    @experiment_handler("computational_robustness_audit")
    @validate_data_types(X_list=list, hypers=dict, args=dict)
    @validate_parameters(X_list=lambda x: len(x) > 0)
    def run_computational_robustness_audit(
        self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> ExperimentResult:
        """Comprehensive computational robustness audit."""
        # Validate inputs
        ResultValidator.validate_data_matrices(X_list)

        self.logger.info("Running computational robustness audit")

        results = {}
        # 1. Exact robustness test (same seed, same everything)
        self.logger.info("Testing exact robustness...")
        exact_results = []
        fixed_seed = 42

        for run in range(3):
            np.random.seed(fixed_seed)
            run_args = args.copy()
            run_args["random_seed"] = fixed_seed

            # Only verbose for first run
            result = self._run_sgfa_analysis(X_list, hypers, run_args, verbose=(run == 0), **kwargs)

            # CRITICAL: Remove large sample arrays to prevent memory leak
            if "W_samples" in result:
                del result["W_samples"]
            if "Z_samples" in result:
                del result["Z_samples"]
            if "samples" in result:
                del result["samples"]

            exact_results.append(result)

            # Cleanup memory after each exact robustness run
            import jax
            import gc
            jax.clear_caches()
            gc.collect()

        results["exact_robustness"] = exact_results

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

        # Analyze computational robustness
        analysis = self._analyze_computational_robustness(results)

        # Generate basic plots
        plots = self._plot_computational_robustness(results)

        # Add comprehensive robustness visualizations
        advanced_plots = self._create_comprehensive_robustness_visualizations(
            X_list, results, "computational_robustness"
        )
        plots.update(advanced_plots)

        return ExperimentResult(
            experiment_id="computational_robustness_audit",
            config=self.config,
            model_results=results,
            diagnostics=analysis,
            plots=plots,
            status="completed",
        )

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

        for dtype_idx, dtype in enumerate(dtypes):
            self.logger.debug(f"Testing with dtype: {dtype}")

            # Convert data to specified dtype
            X_dtype = [X.astype(dtype) for X in X_list]

            try:
                # Only verbose for first dtype
                result = self._run_sgfa_analysis(X_dtype, hypers, args, verbose=(dtype_idx == 0), **kwargs)

                # CRITICAL: Remove large sample arrays to prevent memory leak
                if "W_samples" in result:
                    del result["W_samples"]
                if "Z_samples" in result:
                    del result["Z_samples"]
                if "samples" in result:
                    del result["samples"]

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

        for rng_idx, rng_state in enumerate(rng_states):
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

                # Only verbose for first RNG test
                result = self._run_sgfa_analysis(X_list, hypers, args, verbose=(rng_idx == 0), **kwargs)

                # CRITICAL: Remove large sample arrays to prevent memory leak
                if "W_samples" in result:
                    del result["W_samples"]
                if "Z_samples" in result:
                    del result["Z_samples"]
                if "samples" in result:
                    del result["samples"]

                consistency_results[rng_state] = result

            except Exception as e:
                self.logger.warning(f"Failed with RNG {rng_state}: {str(e)}")
                consistency_results[rng_state] = {"error": str(e)}

        return consistency_results

    def _compute_result_checksums(self, results: Dict) -> Dict:
        """Compute checksums for robustness verification."""
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
        self, X_list: List[np.ndarray], hypers: Dict, args: Dict, verbose: bool = True, **kwargs
    ) -> Dict:
        """Run actual SGFA analysis for robustness testing.

        Parameters
        ----------
        verbose : bool
            If True, log full details. If False, only log seed and key metrics (for repeated calls).
        """
        import time
        import os

        # Configure JAX memory settings BEFORE importing JAX
        # This prevents JAX from pre-allocating 90% of GPU memory
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

        import jax
        from numpyro.infer import MCMC, NUTS

        # Condensed logging for repeated calls
        seed = args.get("random_seed", 42)
        if not verbose:
            self.logger.info(f"🔄 Seed {seed}: Starting MCMC sampling...")
        else:
            self.logger.info("=" * 80)
            self.logger.info("_RUN_SGFA_ANALYSIS - STARTING")
            self.logger.info("=" * 80)
            self.logger.info(f"Input data: {len(X_list)} views")
            for i, X in enumerate(X_list):
                self.logger.info(f"  View {i}: shape {X.shape}, dtype {X.dtype}, has_nan {np.isnan(X).any()}")
            self.logger.info(f"Hyperparameters: {hypers}")
            self.logger.info(f"MCMC args: {args}")
            self.logger.info(f"Additional kwargs keys: {list(kwargs.keys())}")

        # Check cache first
        from core.config_utils import ConfigHelper
        config_dict = ConfigHelper.to_dict(self.config)
        results_cache = config_dict.get("experiments", {}).get("results_cache")
        if results_cache:
            cached_result = results_cache.get(X_list, hypers, args)
            if cached_result is not None:
                self.logger.info("♻️  Using cached SGFA results")
                return cached_result

        try:
            K = hypers.get("K", 10)
            if verbose:
                self.logger.info(
                    f"Running SGFA: K={K}, n_subjects={X_list[0].shape[0]}, n_features={sum(X.shape[1] for X in X_list)}"
                )

            # Use model factory for consistent model management
            from models.models_integration import integrate_models_with_pipeline

            # Setup data characteristics for optimal model selection
            data_characteristics = {
                "total_features": sum(X.shape[1] for X in X_list),
                "n_views": len(X_list),
                "n_subjects": X_list[0].shape[0],
                "has_imaging_data": True
            }

            # Get optimal model configuration via factory
            if verbose:
                self.logger.info("Setting up model via factory...")

            # Debug: Log what's in config_dict BEFORE structuring
            if verbose:
                self.logger.info(f"🔍 DEBUG: config_dict.get('model_type') = {config_dict.get('model_type', 'NOT FOUND')}")
                self.logger.info(f"🔍 DEBUG: 'model' in config_dict = {'model' in config_dict}")
                if "model_type" in args:
                    self.logger.info(f"🔍 DEBUG: args['model_type'] = {args['model_type']}")

            # Ensure model configuration is structured correctly for integration
            if "model" not in config_dict:
                config_dict["model"] = {}

            # Use model_type from config (config_dict already extracted above)
            # Priority: args > ExperimentConfig.model_type > default
            if "model_type" in args:
                # Allow args to override if explicitly provided
                config_dict["model"]["model_type"] = args["model_type"]
            elif "model_type" not in config_dict["model"]:
                # Use model_type from ExperimentConfig if available
                config_dict["model"]["model_type"] = config_dict.get("model_type", "sparseGFA")

            # Debug: Log what model_type is in the config
            if verbose:
                model_type_in_config = config_dict.get("model", {}).get("model_type", "NOT SET")
                self.logger.info(f"🔍 Config model_type: {model_type_in_config}")

            model_type, model_instance, models_summary = integrate_models_with_pipeline(
                config=config_dict,
                X_list=X_list,
                data_characteristics=data_characteristics,
                hypers=hypers,  # Pass hypers to ensure correct percW, slab_df, slab_scale
                verbose=verbose  # Pass verbose to suppress repetitive logging
            )

            if verbose:
                self.logger.info(f"✅ Model setup complete: {model_type}")
                # Log only key hyperparameters (not full models summary with available_models list)
                hypers_log = models_summary.get('hyperparameters', {})
                self.logger.info(f"   Hyperparameters: Dm={hypers_log.get('Dm')}, percW={hypers_log.get('percW')}, slab_df={hypers_log.get('slab_df')}, slab_scale={hypers_log.get('slab_scale')}")

            # Import the SGFA model function via interface
            from core.model_interface import get_model_function
            models = get_model_function()
            if verbose:
                self.logger.info(f"✅ Model function loaded: SparseGFA")

            # Setup MCMC configuration for robustness testing
            num_warmup = args.get("num_warmup", 50)
            num_samples = args.get("num_samples", 100)
            num_chains = args.get("num_chains", 1)
            if verbose:
                self.logger.info(f"MCMC configuration: warmup={num_warmup}, samples={num_samples}, chains={num_chains}")

            # Create args object for model
            import argparse

            model_args = argparse.Namespace(
                model="sparseGFA",
                K=K,
                num_sources=len(X_list),
                reghsZ=args.get("reghsZ", True),
            )

            # Setup MCMC with seed control for robustness
            seed = args.get("random_seed", 42)
            if verbose:
                self.logger.info(f"Setting up MCMC with seed: {seed}")
            rng_key = jax.random.PRNGKey(seed)

            # Store kernel parameters for reuse (create fresh kernel per chain to avoid state accumulation)
            target_accept_prob = args.get("target_accept_prob", 0.8)
            # Ensure max_tree_depth is explicitly an int (JAX requires proper type for internal operations)
            max_tree_depth_raw = args.get("max_tree_depth", 13)
            max_tree_depth = int(max_tree_depth_raw) if max_tree_depth_raw is not None else 13
            dense_mass = args.get("dense_mass", False)  # Use diagonal mass matrix for memory efficiency
            if verbose:
                self.logger.info(f"NUTS kernel parameters: target_accept_prob={target_accept_prob}, max_tree_depth={max_tree_depth}, dense_mass={dense_mass}")
                if not dense_mass:
                    self.logger.info("   Using diagonal mass matrix for ~5GB memory savings per chain")

            # Get chain method from args (parallel, sequential, or vectorized)
            chain_method = args.get("chain_method", "sequential")
            if verbose:
                self.logger.info(f"Creating MCMC sampler with chain_method={chain_method}...")

            # For multiple chains with memory constraints, run chains individually with cache clearing
            # NOTE: Even with chain_method='sequential', NumPyro may not clear JAX caches between chains,
            # leading to OOM errors. We explicitly run chains one-by-one with cache clearing.
            if num_chains > 1:
                self.logger.warning("⚠️  Running chains in explicit sequential mode with JAX cache clearing to prevent OOM")
                self.logger.warning("    (NumPyro's default chain execution can cause GPU memory accumulation)")
                self.logger.info("    (JAX memory preallocation disabled at function entry)")

                all_samples_W = []
                all_samples_Z = []
                total_elapsed = 0

                import jax
                from jax import random as jax_random
                import gc

                for chain_idx in range(num_chains):
                    self.logger.info("=" * 80)
                    self.logger.info(f"STARTING MCMC SAMPLING - CHAIN {chain_idx + 1}/{num_chains}")
                    self.logger.info("=" * 80)
                    self.logger.info(f"This will run {num_warmup} warmup + {num_samples} sampling iterations")

                    # Extra cleanup before starting new chain
                    if chain_idx > 0:
                        import time
                        self.logger.info("⏳ Performing aggressive memory cleanup before next chain...")

                        # Multiple rounds of cache clearing and garbage collection
                        for cleanup_round in range(3):
                            jax.clear_caches()
                            gc.collect()
                            time.sleep(2)  # Give GPU driver time to process deallocations

                        self.logger.info("🧹 Aggressive cleanup complete, waiting 5 seconds for GPU...")
                        time.sleep(5)

                    # Monitor GPU memory before starting chain (only for GPU devices)
                    try:
                        device = jax.devices()[0]
                        device_kind = device.device_kind.lower()

                        if device_kind == 'gpu':
                            mem_stats = device.memory_stats()
                            total_memory_gb = mem_stats.get('bytes_limit', 0) / (1024**3)
                            allocated_memory_gb = mem_stats.get('bytes_in_use', 0) / (1024**3)
                            available_memory_gb = total_memory_gb - allocated_memory_gb

                            self.logger.info(f"📊 GPU Memory Stats BEFORE Chain {chain_idx + 1}:")
                            self.logger.info(f"   Total: {total_memory_gb:.2f} GB")
                            self.logger.info(f"   Allocated: {allocated_memory_gb:.2f} GB")
                            self.logger.info(f"   Available: {available_memory_gb:.2f} GB")

                            if chain_idx > 0:
                                # Check if memory leaked from previous chains
                                if allocated_memory_gb > 0.5:  # More than 500MB still allocated
                                    self.logger.warning(f"⚠️  WARNING: {allocated_memory_gb:.2f} GB still allocated after cleanup!")
                                    self.logger.warning("   Memory may be leaking between chains")
                    except Exception as mem_error:
                        pass  # Silently skip if memory stats unavailable

                    # Create fresh NUTS kernel for this chain (avoid state accumulation)
                    self.logger.info(f"Creating fresh NUTS kernel for chain {chain_idx + 1}...")
                    # Workaround for JAX 0.7.x + NumPyro 0.19.0 compatibility: only pass max_tree_depth if not default
                    if max_tree_depth != 10:
                        kernel = NUTS(models, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth, dense_mass=dense_mass)
                    else:
                        kernel = NUTS(models, target_accept_prob=target_accept_prob, dense_mass=dense_mass)

                    # Create individual MCMC sampler for this chain
                    mcmc_single = MCMC(
                        kernel,
                        num_warmup=num_warmup,
                        num_samples=num_samples,
                        num_chains=1,  # Single chain
                        chain_method="sequential",
                    )

                    # Generate unique RNG key for this chain
                    chain_rng_key = jax_random.fold_in(rng_key, chain_idx)

                    # Check if PCA initialization should be used
                    init_params = None
                    if args.get("use_pca_initialization", False):
                        try:
                            from core.pca_initialization import create_numpyro_init_params
                            # Use model_args.model (not args.model_type) since that's what the actual model uses
                            # model_args.model is set to "sparseGFA" at line 731
                            pca_model_type = "sparseGFA"  # Always use sparseGFA for this model
                            self.logger.info(f"   🔧 Creating PCA initialization for chain {chain_idx + 1}...")
                            init_params = create_numpyro_init_params(X_list, K, pca_model_type)
                            if chain_idx == 0:  # Log detailed info only for first chain
                                self.logger.info(f"   ✓ PCA initialization created with params: {list(init_params.keys())}")
                            else:
                                self.logger.info(f"   ✓ PCA initialization created")
                        except Exception as e:
                            self.logger.warning(f"   ⚠️ PCA initialization failed, using default: {e}")
                            import traceback
                            self.logger.warning(f"   Traceback: {traceback.format_exc()}")
                            init_params = None

                    start_time = time.time()
                    self.logger.info(f"   🚀 Starting MCMC sampling for chain {chain_idx + 1}...")
                    self.logger.info(f"      Model: {models.__class__.__name__}")
                    self.logger.info(f"      Warmup: {num_warmup}, Samples: {num_samples}")
                    self.logger.info(f"      Using init_params: {init_params is not None}")
                    try:
                        mcmc_single.run(
                            chain_rng_key, X_list, hypers, model_args,
                            init_params=init_params,
                            extra_fields=("potential_energy",)
                        )
                        elapsed = time.time() - start_time
                        total_elapsed += elapsed
                        self.logger.info(f"✅ Chain {chain_idx + 1} COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f} min)")

                        # Get samples from this chain and convert to numpy to break JAX references
                        chain_samples = mcmc_single.get_samples()

                        # Monitor GPU memory right after sampling (before cleanup, only for GPU devices)
                        allocated_after_sampling_gb = 0.0  # Initialize to avoid scope issues
                        try:
                            device = jax.devices()[0]
                            device_kind = device.device_kind.lower()

                            if device_kind == 'gpu':
                                mem_stats = device.memory_stats()
                                allocated_after_sampling_gb = mem_stats.get('bytes_in_use', 0) / (1024**3)
                                self.logger.info(f"📊 GPU Memory AFTER sampling: {allocated_after_sampling_gb:.2f} GB allocated")
                        except Exception:
                            pass

                        # Convert JAX arrays to numpy arrays to free device memory
                        # Extract W and Z
                        W_np = np.array(chain_samples["W"])
                        Z_np = np.array(chain_samples["Z"])
                        all_samples_W.append(W_np)
                        all_samples_Z.append(Z_np)

                        # Also extract hyperparameters for plotting
                        # Store full samples dict for this chain (for MCMC diagnostics)
                        chain_samples_dict = {}
                        for key, value in chain_samples.items():
                            # Convert to numpy and store
                            chain_samples_dict[key] = np.array(value)

                        # Store this chain's full samples
                        if not hasattr(self, '_all_chain_samples'):
                            self._all_chain_samples = []
                        self._all_chain_samples.append(chain_samples_dict)

                        # Delete ALL references to free memory - be very explicit
                        del chain_samples, W_np, Z_np, chain_samples_dict
                        del mcmc_single  # Delete MCMC object
                        del kernel  # Delete NUTS kernel object

                        # Aggressive multi-stage cleanup
                        self.logger.info(f"🧹 Starting aggressive cleanup after chain {chain_idx + 1}...")

                        # Stage 1: Clear caches and collect garbage multiple times
                        for cleanup_round in range(3):
                            jax.clear_caches()
                            gc.collect()
                            import time
                            time.sleep(1)  # Brief pause between rounds

                        # Stage 2: Final garbage collection
                        gc.collect()

                        # Monitor GPU memory after cleanup (only for GPU devices)
                        try:
                            device = jax.devices()[0]
                            device_kind = device.device_kind.lower()

                            if device_kind == 'gpu':
                                mem_stats = device.memory_stats()
                                allocated_after_cleanup_gb = mem_stats.get('bytes_in_use', 0) / (1024**3)
                                freed_memory_gb = allocated_after_sampling_gb - allocated_after_cleanup_gb

                                self.logger.info(f"📊 GPU Memory AFTER cleanup:")
                                self.logger.info(f"   Allocated: {allocated_after_cleanup_gb:.2f} GB")
                                self.logger.info(f"   Freed: {freed_memory_gb:.2f} GB")

                                if freed_memory_gb < 1.0 and allocated_after_sampling_gb > 1.0:
                                    self.logger.warning(f"⚠️  WARNING: Only {freed_memory_gb:.2f} GB freed - expected ~2-3 GB")
                                    self.logger.warning("   Memory leak detected!")
                        except Exception:
                            pass

                        self.logger.info(f"✅ Chain {chain_idx + 1} cleanup complete")

                    except Exception as e:
                        elapsed = time.time() - start_time
                        total_elapsed += elapsed
                        self.logger.error(f"❌ Chain {chain_idx + 1} FAILED after {elapsed:.1f}s")
                        self.logger.error(f"Error type: {type(e).__name__}")
                        self.logger.error(f"Error message: {str(e)}")

                        # Log full traceback for debugging
                        import traceback
                        self.logger.error("Full traceback:")
                        self.logger.error(traceback.format_exc())

                        # Log additional context
                        self.logger.error(f"Context: Model={models.__class__.__name__}, K={K}, N={X_list[0].shape[0]}")
                        self.logger.error(f"Init params used: {init_params is not None}")
                        if init_params is not None:
                            self.logger.error(f"Init param keys: {list(init_params.keys())}")

                        # Clear JAX cache even on failure
                        jax.clear_caches()
                        gc.collect()
                        self.logger.info(f"🧹 Cleared JAX caches after chain {chain_idx + 1} failure")
                        raise

                # Stack all chain samples together using numpy (already converted above)
                W_samples = np.stack(all_samples_W, axis=0)  # Shape: (num_chains, num_samples, D, K)
                Z_samples = np.stack(all_samples_Z, axis=0)  # Shape: (num_chains, num_samples, N, K)

                # Delete intermediate lists to free memory immediately
                del all_samples_W, all_samples_Z

                # Create samples dict with W and Z for compatibility
                samples = {"W": W_samples, "Z": Z_samples}

                # Store the full chain samples for MCMC diagnostics (accessible via self._all_chain_samples)
                elapsed = total_elapsed
                log_likelihood = 0.0  # Approximate, can compute if needed

                self.logger.info(f"✅ ALL {num_chains} CHAINS COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f} min)")
                self.logger.info(f"Got samples grouped by chain: {num_chains} chains")
                self.logger.info(f"Stored full samples for {len(self._all_chain_samples)} chains for MCMC diagnostics")

            else:
                # Standard execution for single chain or non-parallel methods
                # Create NUTS kernel for standard execution
                self.logger.info("Creating NUTS kernel for standard MCMC execution...")
                # Workaround for JAX 0.7.x + NumPyro 0.19.0 compatibility: only pass max_tree_depth if not default
                if max_tree_depth != 10:
                    kernel = NUTS(models, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth, dense_mass=dense_mass)
                else:
                    kernel = NUTS(models, target_accept_prob=target_accept_prob, dense_mass=dense_mass)

                mcmc = MCMC(
                    kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    chain_method=chain_method,
                )

                # Check if PCA initialization should be used
                init_params = None
                if args.get("use_pca_initialization", False):
                    try:
                        from core.pca_initialization import create_numpyro_init_params
                        # Use "sparseGFA" since that's what model_args.model is set to (line 731)
                        pca_model_type = "sparseGFA"
                        init_params = create_numpyro_init_params(X_list, K, pca_model_type)
                        self.logger.info(f"   Using PCA initialization for MCMC")
                    except Exception as e:
                        self.logger.warning(f"   PCA initialization failed, using default: {e}")
                        init_params = None

                # Run inference
                self.logger.info("=" * 80)
                self.logger.info("STARTING MCMC SAMPLING")
                self.logger.info("=" * 80)
                self.logger.info(f"This will run {num_warmup} warmup + {num_samples} sampling iterations")
                start_time = time.time()

                try:
                    mcmc.run(
                        rng_key, X_list, hypers, model_args, init_params=init_params, extra_fields=("potential_energy",)
                    )
                    elapsed = time.time() - start_time
                    self.logger.info(f"✅ MCMC SAMPLING COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f} min)")

                    # Clear JAX cache immediately after MCMC to free GPU memory
                    import jax
                    jax.clear_caches()
                    self.logger.info("🧹 Cleared JAX caches after MCMC completion")

                except Exception as e:
                    elapsed = time.time() - start_time
                    self.logger.error(f"❌ MCMC SAMPLING FAILED after {elapsed:.1f}s")
                    self.logger.error(f"Error: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())

                    # Clear JAX cache even on failure to prevent memory leaks
                    import jax
                    jax.clear_caches()
                    self.logger.info("🧹 Cleared JAX caches after MCMC failure")
                    raise

                # Get samples - group by chain if running multiple chains
                if num_chains > 1:
                    samples = mcmc.get_samples(group_by_chain=True)
                    self.logger.info(f"Got samples grouped by chain: {num_chains} chains")
                else:
                    samples = mcmc.get_samples()
                    self.logger.info(f"Got samples from single chain")

                # Calculate log likelihood
                extra_fields = mcmc.get_extra_fields()
                potential_energy = extra_fields.get("potential_energy", np.array([]))
                log_likelihood = (
                    -np.mean(potential_energy) if len(potential_energy) > 0 else np.nan
                )

            # Extract mean parameters
            W_samples = samples["W"]  # Shape: (num_chains, num_samples, D, K) or (num_samples, D, K)
            Z_samples = samples["Z"]  # Shape: (num_chains, num_samples, N, K) or (num_samples, N, K)

            # For parallel chains, we return per-chain results
            # For single chain, we return the mean as before
            if num_chains > 1:
                # Return all chains for factor stability analysis
                self.logger.info(f"W_samples shape: {W_samples.shape}")
                self.logger.info(f"Z_samples shape: {Z_samples.shape}")

                result = {
                    "W_samples": W_samples,
                    "Z_samples": Z_samples,
                    "samples": samples,
                    "log_likelihood": float(log_likelihood),
                    "n_iterations": num_samples,
                    "num_chains": num_chains,
                    "convergence": True,
                    "execution_time": elapsed,
                    "robustness_info": {
                        "seed_used": seed,
                        "mcmc_config": {
                            "num_warmup": num_warmup,
                            "num_samples": num_samples,
                            "num_chains": num_chains,
                            "chain_method": chain_method,
                        },
                    },
                }
            else:
                # Single chain: compute mean and split as before
                W_mean = np.mean(W_samples, axis=0)
                Z_mean = np.mean(Z_samples, axis=0)

                # Split W back into views
                W_list = []
                start_idx = 0
                for X in X_list:
                    end_idx = start_idx + X.shape[1]
                    W_list.append(W_mean[start_idx:end_idx, :])
                    start_idx = end_idx

                result = {
                    "W": W_list,
                    "Z": Z_mean,
                    "W_samples": W_samples,
                    "Z_samples": Z_samples,
                    "samples": samples,
                    "log_likelihood": float(log_likelihood),
                    "n_iterations": num_samples,
                    "convergence": True,
                    "execution_time": elapsed,
                    "robustness_info": {
                        "seed_used": seed,
                        "mcmc_config": {
                            "num_warmup": num_warmup,
                            "num_samples": num_samples,
                            "num_chains": num_chains,
                        },
                    },
                }

            # Cache the result for reuse by later experiments
            if results_cache:
                results_cache.put(X_list, hypers, args, result)

            return result

        except Exception as e:
            self.logger.error(f"SGFA robustness analysis failed: {str(e)}")
            return {
                "error": str(e),
                "convergence": False,
                "execution_time": float("inf"),
                "log_likelihood": float("-inf"),
            }

    def _analyze_seed_robustness(
        self, results: Dict, performance_metrics: Dict
    ) -> Dict:
        """Analyze seed robustness results."""
        analysis = {
            "robustness_metrics": {},
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
                "robustness_score": 1.0
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

    def _analyze_computational_robustness(self, results: Dict) -> Dict:
        """Analyze computational robustness results."""
        analysis = {
            "exact_robustness": {},
            "numerical_stability": {},
            "platform_consistency": {},
            "checksum_verification": {},
        }

        # Exact robustness analysis
        exact_results = results.get("exact_robustness", [])
        if len(exact_results) > 1:
            likelihoods = [r.get("log_likelihood", np.nan) for r in exact_results]
            valid_likelihoods = [ll for ll in likelihoods if not np.isnan(ll)]

            if len(valid_likelihoods) > 1:
                ll_std = np.std(valid_likelihoods)
                analysis["exact_robustness"] = {
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

    def _plot_seed_robustness(
        self, results: Dict, performance_metrics: Dict
    ) -> Dict:
        """Generate plots for seed robustness analysis."""
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
            fig.suptitle("Seed Robustness Analysis", fontsize=16)

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
            plots["seed_robustness"] = fig

        except Exception as e:
            self.logger.warning(
                f"Failed to create seed robustness plots: {str(e)}"
            )

        return plots

    def _plot_data_perturbation_robustness(self, results: Dict) -> Dict:
        """Generate plots for data perturbation robustness analysis."""
        self.logger.info("📊 Generating data perturbation robustness plots...")
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

            self.logger.info(f"   Creating 4-panel perturbation robustness plot for {len(df['perturbation_type'].unique())} perturbation types...")
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
            self.logger.info("   ✅ Data perturbation robustness plot created")

        except Exception as e:
            self.logger.warning(
                f"Failed to create data perturbation robustness plots: {str(e)}"
            )

        self.logger.info(f"📊 Data perturbation robustness plots completed: {len(plots)} plots generated")
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

    def _plot_computational_robustness(self, results: Dict) -> Dict:
        """Generate plots for computational robustness analysis."""
        self.logger.info("📊 Generating computational robustness plots...")
        plots = {}

        try:
            self.logger.info("   Creating 4-panel computational robustness plot...")
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("Computational Robustness Analysis", fontsize=16)

            # Plot 1: Exact robustness
            exact_results = results.get("exact_robustness", [])
            if len(exact_results) > 1:
                likelihoods = [r.get("log_likelihood", np.nan) for r in exact_results]
                valid_likelihoods = [ll for ll in likelihoods if not np.isnan(ll)]

                if valid_likelihoods:
                    run_numbers = list(range(1, len(valid_likelihoods) + 1))
                    axes[0, 0].plot(run_numbers, valid_likelihoods, "o-")
                    axes[0, 0].set_xlabel("Run Number")
                    axes[0, 0].set_ylabel("Log Likelihood")
                    axes[0, 0].set_title("Exact Robustness Test")
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
            plots["computational_robustness"] = fig
            self.logger.info("   ✅ Computational robustness plot created")

        except Exception as e:
            self.logger.warning(
                f"Failed to create computational robustness plots: {str(e)}"
            )

        self.logger.info(f"📊 Computational robustness plots completed: {len(plots)} plots generated")
        return plots

    def _create_comprehensive_robustness_visualizations(
        self, X_list: List[np.ndarray], results: Dict, experiment_name: str
    ) -> Dict:
        """Create comprehensive robustness visualizations focusing on consensus and stability."""
        advanced_plots = {}

        try:
            self.logger.info(
                f"🎨 Creating comprehensive robustness visualizations for {experiment_name}"
            )

            # Import visualization system
            from core.config_utils import ConfigAccessor
            from visualization.manager import VisualizationManager

            # Create a robustness-focused config for visualization
            viz_config = ConfigAccessor(
                {
                    "visualization": {
                        "create_brain_viz": True,  # Include brain maps for robustness assessment
                        "output_format": ["png", "pdf"],
                        "dpi": 300,
                        "robustness_focus": True,
                    },
                    "output_dir": f"/tmp/robustness_viz_{experiment_name}",
                }
            )

            # Initialize visualization manager
            viz_manager = VisualizationManager(viz_config)

            # Prepare robustness data structure
            data = {
                "X_list": X_list,
                "view_names": [f"view_{i}" for i in range(len(X_list))],
                "n_subjects": X_list[0].shape[0],
                "view_dimensions": [X.shape[1] for X in X_list],
                "preprocessing": {
                    "status": "completed",
                    "strategy": "robustness_analysis",
                },
            }

            # Extract best reproducible result for analysis
            best_repro_result = self._extract_best_reproducible_result(results)

            if best_repro_result:
                # Prepare robustness analysis results
                analysis_results = {
                    "best_run": best_repro_result,
                    "all_runs": results,
                    "model_type": "robustness_sparseGFA",
                    "convergence": best_repro_result.get("convergence", False),
                    "robustness_analysis": True,
                }

                # Add cross-validation style results for consensus analysis
                cv_results = {
                    "consensus_analysis": {
                        "multiple_runs": results,
                        "stability_metrics": self._extract_robustness_metrics(
                            results
                        ),
                        "seed_variations": self._extract_seed_variations(results),
                    }
                }

                # Create comprehensive visualizations with robustness focus
                viz_manager.create_all_visualizations(
                    data=data, analysis_results=analysis_results, cv_results=cv_results
                )

                # Extract and process generated plots
                if hasattr(viz_manager, "plot_dir") and viz_manager.plot_dir.exists():
                    plot_files = list(viz_manager.plot_dir.glob("**/*.png"))

                    for plot_file in plot_files:
                        plot_name = f"robustness_{plot_file.stem}"

                        try:
                            import matplotlib.image as mpimg
                            import matplotlib.pyplot as plt

                            fig, ax = plt.subplots(figsize=(12, 8))
                            img = mpimg.imread(str(plot_file))
                            ax.imshow(img)
                            ax.axis("off")
                            ax.set_title(
                                f"Robustness Analysis: {plot_name}", fontsize=14
                            )

                            advanced_plots[plot_name] = fig

                        except Exception as e:
                            self.logger.warning(
                                f"Could not load robustness plot {plot_name}: {e}"
                            )

                    self.logger.info(
                        f"✅ Created { len(plot_files)} comprehensive robustness visualizations"
                    )
                    self.logger.info(
                        "   → Consensus factor analysis and stability plots generated"
                    )
                    self.logger.info(
                        "   → Cross-seed robustness visualizations generated"
                    )

                else:
                    self.logger.warning(
                        "Robustness visualization manager did not create plot directory"
                    )
            else:
                self.logger.warning(
                    "No reproducible results found for comprehensive visualization"
                )

        except Exception as e:
            self.logger.warning(
                f"Failed to create comprehensive robustness visualizations: {e}"
            )

        return advanced_plots

    def _extract_best_reproducible_result(self, results: Dict) -> Optional[Dict]:
        """Extract the most reproducible result from multiple runs."""
        best_result = None
        best_consistency = 0

        # Look through different robustness result structures
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

    def _extract_robustness_metrics(self, results: Dict) -> Dict:
        """Extract robustness metrics for visualization."""
        metrics = {
            "convergence_rates": {},
            "likelihood_stability": {},
            "factor_consistency": {},
            "computational_stability": {},
        }

        # Process different types of robustness results
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

    @experiment_handler("factor_stability_analysis")
    @validate_data_types(X_list=list, hypers=dict, args=dict)
    @validate_parameters(
        X_list=lambda x: len(x) > 0,
        n_chains=lambda x: x > 1,
    )
    def run_factor_stability_analysis(
        self,
        X_list: List[np.ndarray],
        hypers: Dict,
        args: Dict,
        n_chains: int = 4,
        cosine_threshold: float = 0.8,
        min_match_rate: float = 0.5,
        **kwargs,
    ) -> ExperimentResult:
        """Run factor stability analysis with multiple independent chains.

        Implements the methodology from Ferreira et al. 2024 Section 2.7:
        - Run N independent MCMC chains sequentially
        - Extract factor loadings (W) and scores (Z) from each chain
        - Assess factor stability using cosine similarity matching
        - Count effective factors (non-shrunk by priors)

        Parameters
        ----------
        X_list : List[np.ndarray]
            List of data matrices (multi-view data)
        hypers : Dict
            Hyperparameters for SGFA model
        args : Dict
            MCMC arguments (num_warmup, num_samples, etc.)
        n_chains : int, default=4
            Number of independent chains to run
        cosine_threshold : float, default=0.8
            Minimum cosine similarity for factor matching
        min_match_rate : float, default=0.5
            Minimum fraction of chains for robust factor (>50%)
        **kwargs : dict
            Additional arguments

        Returns
        -------
        ExperimentResult
            Contains:
            - chain_results: Results from each chain
            - stability_analysis: Factor stability metrics
            - effective_factors: Effective factor counts per chain
            - plots: Stability visualization plots
        """
        # Validate inputs
        self.logger.info("=" * 80)
        self.logger.info("FACTOR STABILITY ANALYSIS - STARTING")
        self.logger.info("=" * 80)
        self.logger.info(f"Input data: {len(X_list)} views")
        for i, X in enumerate(X_list):
            self.logger.info(f"  View {i}: shape {X.shape}, dtype {X.dtype}")
        self.logger.info(f"Hyperparameters: K={hypers.get('K')}, percW={hypers.get('percW')}, Dm={hypers.get('Dm')}")
        self.logger.info(f"MCMC args: num_samples={args.get('num_samples')}, num_warmup={args.get('num_warmup')}")

        ResultValidator.validate_data_matrices(X_list)
        ParameterValidator.validate_positive(n_chains, "n_chains")

        if n_chains < 2:
            raise ValueError("n_chains must be at least 2 for stability analysis")

        self.logger.info(f"Running factor stability analysis with {n_chains} independent chains")
        self.logger.info(f"Cosine similarity threshold: {cosine_threshold}")
        self.logger.info(f"Minimum match rate: {min_match_rate}")

        # Log data characteristics concisely
        view_names = kwargs.get('view_names')
        feature_names = kwargs.get('feature_names')
        if view_names:
            self.logger.info(f"View names: {view_names}")
        if feature_names:
            feature_counts = {view: len(names) for view, names in feature_names.items()}
            self.logger.info(f"Feature counts per view: {feature_counts}")

        # Import factor stability utilities
        from analysis.factor_stability import (
            assess_factor_stability_cosine,
            count_effective_factors,
            save_stability_results,
        )

        # Get chain method from experiment config
        from core.config_utils import ConfigHelper
        config_dict = ConfigHelper.to_dict(self.config)
        factor_stability_config = config_dict.get("factor_stability", {})
        chain_method = factor_stability_config.get("chain_method", "parallel")

        self.logger.info("=" * 80)
        self.logger.info(f"RUNNING {n_chains} CHAINS WITH METHOD: {chain_method}")
        self.logger.info("=" * 80)

        # Prepare args for multi-chain execution
        base_seed = args.get("random_seed", 42)
        multi_chain_args = args.copy()
        multi_chain_args["num_chains"] = n_chains
        multi_chain_args["chain_method"] = chain_method
        multi_chain_args["random_seed"] = base_seed

        self.logger.info(f"Base seed: {base_seed}")
        self.logger.info(f"MCMC config: samples={multi_chain_args.get('num_samples')}, "
                        f"warmup={multi_chain_args.get('num_warmup')}, chains={n_chains}")
        self.logger.info(f"JAX will automatically split PRNG key for independent chains")

        # Run all chains in one MCMC call (NumPyro handles parallelization)
        chain_results = []
        performance_metrics = {}

        with self.profiler.profile("all_chains") as p:
            self.logger.info(f"Starting SGFA analysis with {n_chains} chains...")
            try:
                # Verbose=True to see the configuration (this is typically called once)
                result = self._run_sgfa_analysis(X_list, hypers, multi_chain_args, verbose=True, **kwargs)
                self.logger.info(f"   Result keys: {list(result.keys())}")

                # Check if execution succeeded
                if "error" in result:
                    self.logger.error(f"❌ SGFA analysis returned with error")
                    raise RuntimeError(result["error"])

                self.logger.info(f"✅ All {n_chains} chains completed successfully")
            except Exception as e:
                self.logger.error(f"❌ Multi-chain SGFA analysis FAILED: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise

        # Extract per-chain results from grouped samples
        W_samples = result.get("W_samples")  # Shape: (num_chains, num_samples, D, K)
        Z_samples = result.get("Z_samples")  # Shape: (num_chains, num_samples, N, K)

        if W_samples is not None:
            self.logger.info(f"W_samples shape: {W_samples.shape}")
        if Z_samples is not None:
            self.logger.info(f"Z_samples shape: {Z_samples.shape}")

        # Process each chain's results
        for chain_id in range(n_chains):
            self.logger.info(f"Processing chain {chain_id + 1}/{n_chains}...")

            # Extract this chain's samples: (num_samples, D, K)
            W_chain = W_samples[chain_id]  # (num_samples, D, K)
            Z_chain = Z_samples[chain_id]  # (num_samples, N, K)

            # Compute mean over samples for this chain
            W_mean = np.mean(W_chain, axis=0)  # (D, K)
            Z_mean = np.mean(Z_chain, axis=0)  # (N, K)

            # Split W back into views
            W_list = []
            start_idx = 0
            for X in X_list:
                end_idx = start_idx + X.shape[1]
                W_list.append(W_mean[start_idx:end_idx, :])
                start_idx = end_idx

            chain_result = {
                "chain_id": chain_id,
                "seed": base_seed + chain_id * 1000,  # Report equivalent seed
                "W": W_list,
                "Z": Z_mean,
                "log_likelihood": result.get("log_likelihood", np.nan),
                "convergence": result.get("convergence", False),
                "execution_time": result.get("execution_time", 0) / n_chains,  # Approximate per-chain time
                "samples": {
                    "W": W_chain,
                    "Z": Z_chain,
                }
            }

            chain_results.append(chain_result)

            self.logger.info(
                f"Chain {chain_id} processed: "
                f"W shape: {W_list[0].shape if W_list else 'N/A'}, "
                f"Z shape: {Z_mean.shape}"
            )

        # Store overall performance metrics
        metrics = self.profiler.get_current_metrics()
        if metrics is not None:
            performance_metrics["all_chains"] = {
                "total_execution_time": metrics.execution_time,
                "peak_memory_gb": metrics.peak_memory_gb,
                "convergence": result.get("convergence", False),
                "log_likelihood": result.get("log_likelihood", np.nan),
                "num_chains": n_chains,
                "chain_method": chain_method,
            }
            # Estimate per-chain metrics
            for chain_id in range(n_chains):
                performance_metrics[f"chain_{chain_id}"] = {
                    "execution_time": metrics.execution_time / n_chains,
                    "peak_memory_gb": metrics.peak_memory_gb,
                    "convergence": result.get("convergence", False),
                    "log_likelihood": result.get("log_likelihood", np.nan),
                }

        self.logger.info(f"✅ All chains completed in {metrics.execution_time:.1f}s ({metrics.execution_time/60:.1f} min)")

        # Assess factor stability using cosine similarity
        self.logger.info("=" * 80)
        self.logger.info("ASSESSING FACTOR STABILITY ACROSS CHAINS")
        self.logger.info("=" * 80)
        self.logger.info(f"Number of chains to analyze: {len(chain_results)}")
        self.logger.info(f"Cosine threshold: {cosine_threshold}")
        self.logger.info(f"Min match rate: {min_match_rate}")

        try:
            stability_results = assess_factor_stability_cosine(
                chain_results,
                threshold=cosine_threshold,
                min_match_rate=min_match_rate,
            )
            self.logger.info(f"✅ Stability analysis completed")
            self.logger.info(f"   Stable factors: {stability_results.get('n_stable_factors', 0)}/{stability_results.get('total_factors', 0)}")
            self.logger.info(f"   Stability rate: {stability_results.get('stability_rate', 0):.1%}")
        except Exception as e:
            self.logger.error(f"❌ Stability analysis FAILED: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

        # Count effective factors for each chain
        self.logger.info("=" * 80)
        self.logger.info("COUNTING EFFECTIVE FACTORS PER CHAIN")
        self.logger.info("=" * 80)
        effective_factors_per_chain = []
        for i, chain_result in enumerate(chain_results):
            W = chain_result["W"]
            effective = count_effective_factors(
                W,
                sparsity_threshold=0.01,
                min_nonzero_pct=0.05,
            )
            effective["chain_id"] = i
            effective_factors_per_chain.append(effective)

            self.logger.info(
                f"Chain {i}: {effective['n_effective']}/{effective['total_factors']} "
                f"effective factors"
            )

        # Compute R-hat (Gelman-Rubin) convergence diagnostics
        self.logger.info("=" * 80)
        self.logger.info("COMPUTING R-HAT CONVERGENCE DIAGNOSTICS")
        self.logger.info("=" * 80)

        rhat_diagnostics = {}
        try:
            # Get W and Z samples from all chains
            # W_samples shape: (num_chains, num_samples, D, K)
            # Z_samples shape: (num_chains, num_samples, N, K)
            if W_samples is not None and len(W_samples.shape) == 4:
                # Compute R-hat for W (factor loadings)
                # For each factor k, compute R-hat across all features
                K = W_samples.shape[3]
                D = W_samples.shape[2]

                rhat_W = split_gelman_rubin(W_samples)  # Shape: (D, K)

                # Summary statistics for W
                rhat_W_max_per_factor = np.max(rhat_W, axis=0)  # Max R-hat per factor
                rhat_W_mean_per_factor = np.mean(rhat_W, axis=0)  # Mean R-hat per factor
                rhat_W_max_overall = np.max(rhat_W)
                rhat_W_mean_overall = np.mean(rhat_W)

                rhat_diagnostics["W"] = {
                    "max_rhat_per_factor": rhat_W_max_per_factor.tolist(),
                    "mean_rhat_per_factor": rhat_W_mean_per_factor.tolist(),
                    "max_rhat_overall": float(rhat_W_max_overall),
                    "mean_rhat_overall": float(rhat_W_mean_overall),
                }

                self.logger.info(f"R-hat for W (factor loadings):")
                self.logger.info(f"  Overall: max={rhat_W_max_overall:.4f}, mean={rhat_W_mean_overall:.4f}")
                for k in range(K):
                    max_rhat_k = rhat_W_max_per_factor[k]
                    mean_rhat_k = rhat_W_mean_per_factor[k]
                    status = "✓" if max_rhat_k < 1.1 else "⚠️"
                    self.logger.info(f"  Factor {k}: max={max_rhat_k:.4f}, mean={mean_rhat_k:.4f} {status}")

                # Check for convergence issues
                if rhat_W_max_overall > 1.1:
                    self.logger.warning(f"⚠️  High R-hat detected for W: {rhat_W_max_overall:.4f} > 1.1")
                    self.logger.warning(f"     Chains may not have converged. Consider more MCMC samples.")
                else:
                    self.logger.info(f"✓ W converged: all R-hat < 1.1")

            if Z_samples is not None and len(Z_samples.shape) == 4:
                # Compute R-hat for Z (factor scores)
                rhat_Z = split_gelman_rubin(Z_samples)  # Shape: (N, K)

                # Summary statistics for Z
                rhat_Z_max_per_factor = np.max(rhat_Z, axis=0)  # Max R-hat per factor
                rhat_Z_mean_per_factor = np.mean(rhat_Z, axis=0)  # Mean R-hat per factor
                rhat_Z_max_overall = np.max(rhat_Z)
                rhat_Z_mean_overall = np.mean(rhat_Z)

                rhat_diagnostics["Z"] = {
                    "max_rhat_per_factor": rhat_Z_max_per_factor.tolist(),
                    "mean_rhat_per_factor": rhat_Z_mean_per_factor.tolist(),
                    "max_rhat_overall": float(rhat_Z_max_overall),
                    "mean_rhat_overall": float(rhat_Z_mean_overall),
                }

                self.logger.info(f"R-hat for Z (factor scores):")
                self.logger.info(f"  Overall: max={rhat_Z_max_overall:.4f}, mean={rhat_Z_mean_overall:.4f}")
                for k in range(K):
                    max_rhat_k = rhat_Z_max_per_factor[k]
                    mean_rhat_k = rhat_Z_mean_per_factor[k]
                    status = "✓" if max_rhat_k < 1.1 else "⚠️"
                    self.logger.info(f"  Factor {k}: max={max_rhat_k:.4f}, mean={mean_rhat_k:.4f} {status}")

                # Check for convergence issues
                if rhat_Z_max_overall > 1.1:
                    self.logger.warning(f"⚠️  High R-hat detected for Z: {rhat_Z_max_overall:.4f} > 1.1")
                    self.logger.warning(f"     Chains may not have converged. Consider more MCMC samples.")
                else:
                    self.logger.info(f"✓ Z converged: all R-hat < 1.1")

        except Exception as e:
            self.logger.warning(f"Failed to compute R-hat diagnostics: {e}")
            import traceback
            self.logger.warning(traceback.format_exc())

        # Compile diagnostics
        diagnostics = {
            "stability_summary": {
                "n_chains": n_chains,
                "n_stable_factors": stability_results["n_stable_factors"],
                "total_factors": stability_results["total_factors"],
                "stability_rate": stability_results["stability_rate"],
                "stable_factor_indices": stability_results["stable_factor_indices"],
            },
            "effective_factors_summary": {
                "per_chain": [
                    {
                        "chain_id": ef["chain_id"],
                        "n_effective": ef["n_effective"],
                        "shrinkage_rate": ef["shrinkage_rate"],
                    }
                    for ef in effective_factors_per_chain
                ],
                "mean_effective": np.mean([ef["n_effective"] for ef in effective_factors_per_chain]),
                "std_effective": np.std([ef["n_effective"] for ef in effective_factors_per_chain]),
            },
            "convergence_summary": {
                "n_converged": sum(
                    1 for pm in performance_metrics.values() if pm["convergence"]
                ),
                "convergence_rate": sum(
                    1 for pm in performance_metrics.values() if pm["convergence"]
                ) / n_chains,
                "rhat_diagnostics": rhat_diagnostics,  # Add R-hat diagnostics
            },
        }

        # Generate plots
        self.logger.info("=" * 80)
        self.logger.info("GENERATING FACTOR STABILITY PLOTS")
        self.logger.info("=" * 80)
        try:
            plots = self._plot_factor_stability(
                chain_results,
                stability_results,
                effective_factors_per_chain,
                performance_metrics,
                X_list=X_list,
                data=kwargs,  # Pass any additional data (view_names, feature_names, etc.)
            )
            self.logger.info(f"✅ Generated {len(plots)} plots")
            for plot_name in plots.keys():
                self.logger.info(f"   - {plot_name}")
        except Exception as e:
            self.logger.error(f"❌ Plot generation FAILED: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            plots = {}

        # Create final result
        self.logger.info("=" * 80)
        self.logger.info("FACTOR STABILITY ANALYSIS - COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"Summary:")
        self.logger.info(f"  - {len(chain_results)} chains completed")
        self.logger.info(f"  - {stability_results.get('n_stable_factors', 0)}/{stability_results.get('total_factors', 0)} stable factors")
        self.logger.info(f"  - Stability rate: {stability_results.get('stability_rate', 0):.1%}")
        self.logger.info(f"  - Mean effective factors: {np.mean([ef['n_effective'] for ef in effective_factors_per_chain]):.1f}")

        # Add R-hat summary to final output
        if rhat_diagnostics:
            if "W" in rhat_diagnostics:
                rhat_W_max = rhat_diagnostics["W"]["max_rhat_overall"]
                rhat_W_mean = rhat_diagnostics["W"]["mean_rhat_overall"]
                W_converged = "✓" if rhat_W_max < 1.1 else "⚠️"
                self.logger.info(f"  - R-hat (W): max={rhat_W_max:.4f}, mean={rhat_W_mean:.4f} {W_converged}")
            if "Z" in rhat_diagnostics:
                rhat_Z_max = rhat_diagnostics["Z"]["max_rhat_overall"]
                rhat_Z_mean = rhat_diagnostics["Z"]["mean_rhat_overall"]
                Z_converged = "✓" if rhat_Z_max < 1.1 else "⚠️"
                self.logger.info(f"  - R-hat (Z): max={rhat_Z_max:.4f}, mean={rhat_Z_mean:.4f} {Z_converged}")

        self.logger.info(f"  - {len(plots)} plots generated")

        result = ExperimentResult(
            experiment_id="factor_stability_analysis",
            config=self.config,
            model_results={
                "chain_results": chain_results,
                "stability_results": stability_results,
                "effective_factors": effective_factors_per_chain,
            },
            diagnostics=diagnostics,
            plots=plots,
            performance_metrics=performance_metrics,
            status="completed",
        )

        self.logger.info("Returning ExperimentResult object")
        return result

    def _plot_factor_stability(
        self,
        chain_results: List[Dict],
        stability_results: Dict,
        effective_factors_per_chain: List[Dict],
        performance_metrics: Dict,
        X_list: Optional[List[np.ndarray]] = None,
        data: Optional[Dict] = None,
    ) -> Dict:
        """Generate plots for factor stability analysis."""
        plots = {}

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("Factor Stability Analysis", fontsize=16, fontweight='bold')

            # Plot 1: Stable vs Unstable factors
            per_factor = stability_results["per_factor_details"]
            stable_indices = [f["factor_index"] for f in per_factor if f["is_robust"]]
            unstable_indices = [f["factor_index"] for f in per_factor if not f["is_robust"]]
            match_rates = [f["match_rate"] for f in per_factor]

            colors = ['green' if f["is_robust"] else 'red' for f in per_factor]
            axes[0, 0].bar(range(len(match_rates)), match_rates, color=colors, alpha=0.7)
            axes[0, 0].axhline(
                y=stability_results["min_match_rate"],
                color='black',
                linestyle='--',
                label=f'Threshold ({stability_results["min_match_rate"]:.1%})',
            )
            axes[0, 0].set_xlabel("Factor Index")
            axes[0, 0].set_ylabel("Match Rate")
            axes[0, 0].set_title(
                f"Factor Stability: {len(stable_indices)} Robust, "
                f"{len(unstable_indices)} Unstable"
            )
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Effective factors per chain
            chain_ids = [ef["chain_id"] for ef in effective_factors_per_chain]
            n_effective = [ef["n_effective"] for ef in effective_factors_per_chain]
            total_factors = effective_factors_per_chain[0]["total_factors"]

            axes[0, 1].bar(chain_ids, n_effective, alpha=0.7)
            axes[0, 1].axhline(
                y=total_factors,
                color='red',
                linestyle='--',
                label=f'K={total_factors}',
                alpha=0.5,
            )
            axes[0, 1].set_xlabel("Chain ID")
            axes[0, 1].set_ylabel("Number of Effective Factors")
            axes[0, 1].set_title(
                f"Effective Factors Per Chain (Mean: {np.mean(n_effective):.1f})"
            )
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Log likelihood by chain
            lls = [
                pm["log_likelihood"]
                for pm in performance_metrics.values()
                if not np.isnan(pm["log_likelihood"])
            ]
            chain_labels = [f"Chain {i}" for i in range(len(lls))]

            axes[1, 0].plot(range(len(lls)), lls, 'o-', markersize=8, linewidth=2)
            axes[1, 0].set_xlabel("Chain ID")
            axes[1, 0].set_ylabel("Log Likelihood")
            axes[1, 0].set_title("Log Likelihood Across Chains")
            axes[1, 0].grid(True, alpha=0.3)

            # Add mean line
            if lls:
                axes[1, 0].axhline(
                    np.mean(lls), color='red', linestyle='--', label='Mean', alpha=0.7
                )
                axes[1, 0].legend()

            # Plot 4: Stability summary
            summary_data = [
                len(stable_indices),
                len(unstable_indices),
                np.mean(n_effective),
                total_factors - np.mean(n_effective),
            ]
            summary_labels = [
                'Robust\nFactors',
                'Unstable\nFactors',
                'Mean Effective\nFactors',
                'Mean Shrunk\nFactors',
            ]
            colors = ['green', 'red', 'blue', 'gray']

            axes[1, 1].bar(range(len(summary_data)), summary_data, color=colors, alpha=0.7)
            axes[1, 1].set_xticks(range(len(summary_labels)))
            axes[1, 1].set_xticklabels(summary_labels, rotation=0, ha='center')
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].set_title("Stability and Effectiveness Summary")
            axes[1, 1].grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plots["factor_stability_summary"] = fig

        except Exception as e:
            self.logger.warning(f"Failed to create factor stability plots: {str(e)}")

        # Add enhanced factor loading distributions visualization
        if X_list is not None and len(chain_results) > 0:
            try:
                from visualization.factor_plots import FactorVisualizer

                # Use consensus W from the first chain (or average if available)
                consensus_W = stability_results.get("consensus_W")
                if consensus_W is None:
                    # Fall back to first chain's W
                    consensus_W = chain_results[0].get("W")

                if consensus_W is not None:
                    # Prepare data dict for visualizer
                    viz_data = {
                        "X_list": X_list,
                        "view_names": data.get("view_names", [f"View_{i}" for i in range(len(X_list))]) if data else [f"View_{i}" for i in range(len(X_list))],
                        "feature_names": data.get("feature_names", {}) if data else {},
                    }

                    # Create visualizer
                    visualizer = FactorVisualizer(self.config)

                    # Create enhanced loading distributions plot
                    self.logger.info("Creating enhanced factor loading distribution plot...")
                    fig_loadings = visualizer.plot_enhanced_factor_loading_distributions(
                        consensus_W, viz_data, save_path=None
                    )
                    plots["enhanced_loading_distributions"] = fig_loadings

            except Exception as e:
                self.logger.warning(f"Failed to create enhanced loading distributions: {str(e)}")

        # Add brain visualization summary
        if X_list is not None and len(chain_results) > 0:
            try:
                from visualization.brain_plots import BrainVisualizer

                # Use consensus W from stability results
                consensus_W = stability_results.get("consensus_W")
                consensus_Z = chain_results[0].get("Z")  # Use first chain's Z

                if consensus_W is not None and consensus_Z is not None:
                    # Check if data has brain regions
                    view_names = data.get("view_names", []) if data else []
                    brain_regions = ['lentiform', 'sn', 'putamen', 'caudate', 'thalamus',
                                    'hippocampus', 'amygdala', 'cortical', 'subcortical',
                                    'frontal', 'parietal', 'temporal', 'occipital',
                                    'cerebellum', 'brainstem', 'roi', 'region', 'voxel', 'vertex']

                    has_brain_data = any(
                        any(region in vn.lower() for region in brain_regions)
                        for vn in view_names
                    )

                    if has_brain_data:
                        self.logger.info("Creating brain visualization summary...")

                        # Prepare analysis results dict
                        analysis_results = {
                            "W": [consensus_W] if not isinstance(consensus_W, list) else consensus_W,
                            "Z": consensus_Z,
                        }

                        # Prepare data dict
                        viz_data = {
                            "X_list": X_list,
                            "view_names": view_names,
                            "feature_names": data.get("feature_names", {}) if data else {},
                        }

                        # Create visualizer
                        brain_viz = BrainVisualizer(self.config)

                        # Create brain visualization summary
                        fig_brain = brain_viz.create_brain_visualization_summary(
                            analysis_results, viz_data, plot_dir=None
                        )

                        if fig_brain is not None:
                            plots["brain_visualization_summary"] = fig_brain

            except Exception as e:
                self.logger.warning(f"Failed to create brain visualization summary: {str(e)}")

        # Add factor stability heatmap (chain-to-chain matching matrix)
        try:
            similarity_matrix = stability_results.get("similarity_matrix")

            if similarity_matrix is not None and len(chain_results) > 0:
                self.logger.info("Creating factor stability heatmap (chain-to-chain matching)...")

                n_chains = similarity_matrix.shape[0]
                K = similarity_matrix.shape[2]  # Number of factors

                # Create a composite heatmap showing factor matching across chains
                # For each factor, show how well it matches across all chain pairs
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle("Factor Stability: Chain-to-Chain Matching", fontsize=16, fontweight='bold')

                # Plot 1: Average similarity per factor across all chain comparisons
                # Compute mean similarity for each factor across chain 0 to other chains
                avg_similarity_per_factor = np.zeros(K)
                for k in range(K):
                    # Get similarities from chain 0 to all other chains for factor k
                    chain_similarities = similarity_matrix[0, 1:, k]
                    # Handle case where all similarities might be 0 or filtered out
                    valid_sims = chain_similarities[chain_similarities > 0]
                    if len(valid_sims) > 0:
                        avg_similarity_per_factor[k] = np.mean(valid_sims)
                    else:
                        # If no valid similarities, use the mean of all (including 0s)
                        avg_similarity_per_factor[k] = np.mean(chain_similarities) if len(chain_similarities) > 0 else 0.0

                # Bar plot of average similarity per factor
                factor_indices = np.arange(K)
                colors_factors = ['green' if not np.isnan(sim) and sim >= stability_results["threshold"] else 'red'
                                 for sim in avg_similarity_per_factor]

                axes[0].bar(factor_indices, avg_similarity_per_factor, color=colors_factors, alpha=0.7)
                axes[0].axhline(y=stability_results["threshold"], color='black',
                               linestyle='--', label=f'Threshold ({stability_results["threshold"]})')
                axes[0].set_xlabel('Factor Index')
                axes[0].set_ylabel('Average Cosine Similarity')
                axes[0].set_title('Average Factor Matching Across Chains (vs Chain 0)')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                axes[0].set_ylim([0, 1])

                # Plot 2: Heatmap showing per-factor similarity across chains
                # Show similarity of each factor to its best match in other chains
                # similarity_matrix[0, j, k] = similarity of factor k in chain 0 to best match in chain j
                factor_chain_matrix = similarity_matrix[0, :, :]  # Shape: (n_chains, K)

                # Transpose to show factors on Y-axis, chains on X-axis
                factor_chain_matrix = factor_chain_matrix.T  # Shape: (K, n_chains)

                im = axes[1].imshow(factor_chain_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
                axes[1].set_xticks(range(n_chains))
                axes[1].set_yticks(range(K))
                axes[1].set_xticklabels([f'Chain {i}' for i in range(n_chains)])
                axes[1].set_yticklabels([f'F{k}' for k in range(K)])
                axes[1].set_xlabel('Chain')
                axes[1].set_ylabel('Factor')
                axes[1].set_title('Factor Similarity Matrix (Reference: Chain 0)')

                # Add text annotations
                for k in range(K):
                    for j in range(n_chains):
                        val = factor_chain_matrix[k, j]
                        if val > 0:  # Only annotate non-zero values
                            text = axes[1].text(j, k, f'{val:.2f}',
                                              ha="center", va="center", color="black", fontsize=8)

                plt.colorbar(im, ax=axes[1], label='Cosine Similarity')
                plt.tight_layout()

                plots["factor_stability_heatmap"] = fig
                self.logger.info("   ✅ Factor stability heatmap created")

        except Exception as e:
            self.logger.warning(f"Failed to create factor stability heatmap: {str(e)}")
            import traceback
            self.logger.warning(f"Traceback: {traceback.format_exc()}")

        # Add MCMC trace diagnostic plots
        if len(chain_results) > 0:
            try:
                from analysis.mcmc_diagnostics import (
                    plot_trace_diagnostics,
                    plot_parameter_distributions,
                    plot_hyperparameter_posteriors,
                    plot_hyperparameter_traces,
                )

                self.logger.info("Creating MCMC trace diagnostic plots...")

                # Extract W and Z samples from chain results
                # Expected format: chain_results contains samples with shape (n_chains, n_samples, ...)
                W_samples_list = []
                Z_samples_list = []
                samples_by_chain = []

                # Check if we have stored full chain samples with hyperparameters
                if hasattr(self, '_all_chain_samples') and len(self._all_chain_samples) > 0:
                    self.logger.info(f"  Using stored full chain samples ({len(self._all_chain_samples)} chains) with hyperparameters")
                    samples_by_chain = self._all_chain_samples
                    # Extract W and Z from full samples for compatibility
                    for chain_samples in self._all_chain_samples:
                        if "W" in chain_samples and "Z" in chain_samples:
                            W_samples_list.append(chain_samples["W"])
                            Z_samples_list.append(chain_samples["Z"])
                else:
                    # Fallback to result samples (may not have hyperparameters)
                    self.logger.warning("  No stored chain samples found, using result samples (may lack hyperparameters)")
                    for result in chain_results:
                        samples = result.get("samples", {})
                        if "W" in samples and "Z" in samples:
                            W_samples_list.append(samples["W"])
                            Z_samples_list.append(samples["Z"])
                            samples_by_chain.append(samples)

                if len(W_samples_list) > 0:
                    # Stack samples from different chains
                    # W_samples: (n_chains, n_samples, D, K)
                    # Z_samples: (n_chains, n_samples, N, K)
                    W_samples = np.stack(W_samples_list, axis=0)
                    Z_samples = np.stack(Z_samples_list, axis=0)

                    self.logger.info(f"  W_samples shape: {W_samples.shape}")
                    self.logger.info(f"  Z_samples shape: {Z_samples.shape}")

                    # Create trace diagnostic plots
                    fig_trace = plot_trace_diagnostics(
                        W_samples=W_samples,
                        Z_samples=Z_samples,
                        save_path=None,
                        max_factors=min(4, W_samples.shape[3]),
                        thin=max(1, W_samples.shape[1] // 1000),  # Thin for readability
                    )
                    plots["mcmc_trace_diagnostics"] = fig_trace
                    self.logger.info("   ✅ MCMC trace diagnostics created")

                    # Create parameter distribution plots
                    # Set up individual plots directory
                    from pathlib import Path
                    experiment_output_dir = Path(self.base_output_dir) if hasattr(self, 'base_output_dir') else Path(self.output_dir)
                    wz_plots_dir = experiment_output_dir / "individual_plots" / "wz_distributions"
                    wz_plots_dir.mkdir(parents=True, exist_ok=True)

                    fig_dist = plot_parameter_distributions(
                        W_samples=W_samples,
                        Z_samples=Z_samples,
                        save_path=None,
                        max_factors=min(4, W_samples.shape[3]),
                        save_individual=True,
                        output_dir=str(wz_plots_dir),
                    )
                    plots["mcmc_parameter_distributions"] = fig_dist
                    self.logger.info("   ✅ Parameter distribution plots created")

                    # Create hyperparameter posterior plots (tauW, tauZ)
                    if len(samples_by_chain) > 0:
                        self.logger.info("Creating hyperparameter posterior plots...")

                        # Infer number of sources from X_list
                        num_sources = len(X_list) if X_list is not None else None

                        # Determine output directory for individual plots
                        # Use experiment-specific output directory, not global results dir
                        from pathlib import Path
                        experiment_output_dir = Path(self.base_output_dir) if hasattr(self, 'base_output_dir') else Path(self.output_dir)
                        individual_plots_dir = experiment_output_dir / "individual_plots" / "hyperparameters"
                        individual_plots_dir.mkdir(parents=True, exist_ok=True)

                        fig_hyper_post = plot_hyperparameter_posteriors(
                            samples_by_chain=samples_by_chain,
                            save_path=None,
                            num_sources=num_sources,
                            save_individual=True,
                            output_dir=str(individual_plots_dir),
                            view_names=view_names,
                        )
                        plots["hyperparameter_posteriors"] = fig_hyper_post
                        self.logger.info("   ✅ Hyperparameter posterior plots created")

                        # Create hyperparameter trace plots
                        fig_hyper_trace = plot_hyperparameter_traces(
                            samples_by_chain=samples_by_chain,
                            save_path=None,
                            num_sources=num_sources,
                            thin=max(1, W_samples.shape[1] // 1000),
                            view_names=view_names,
                            save_individual=True,
                            output_dir=str(individual_plots_dir),
                        )
                        plots["hyperparameter_traces"] = fig_hyper_trace
                        self.logger.info("   ✅ Hyperparameter trace plots created")

                    # Create factor variance profile analysis (ARD shrinkage assessment)
                    try:
                        from analysis.mcmc_diagnostics import analyze_factor_variance_profile

                        self.logger.info("Creating factor variance profile analysis (ARD shrinkage)...")

                        # Analyze variance profile to assess effective dimensionality
                        variance_results = analyze_factor_variance_profile(
                            Z_samples=Z_samples,
                            variance_threshold=0.1,  # Factors below this are shrunk away
                            save_path=None,
                        )

                        # Create the plot figure that was generated
                        # (analyze_factor_variance_profile returns data dict and generates figure)
                        plots["factor_variance_profile"] = plt.gcf()

                        # Log key insights
                        K = Z_samples.shape[3]
                        n_active = variance_results['n_active_factors']
                        effective_dim = variance_results['effective_dimensionality']

                        self.logger.info(f"   📊 Variance Profile Summary:")
                        self.logger.info(f"      Total factors (K): {K}")
                        self.logger.info(f"      Active factors (var > 0.1): {n_active}")
                        self.logger.info(f"      Effective dimensionality (90% var): {effective_dim}")

                        if K >= 10:
                            sorted_vars = variance_results['sorted_variances']
                            top_5_mean = sorted_vars[:5].mean()
                            next_10_mean = sorted_vars[5:15].mean() if K >= 15 else sorted_vars[5:].mean()
                            shrinkage_ratio = top_5_mean / (next_10_mean + 1e-10)

                            if shrinkage_ratio > 10:
                                self.logger.info(f"      ✅ HEALTHY ARD shrinkage (ratio={shrinkage_ratio:.1f})")
                            elif shrinkage_ratio > 3:
                                self.logger.info(f"      ⚠️  MODERATE shrinkage (ratio={shrinkage_ratio:.1f})")
                            else:
                                self.logger.info(f"      ❌ POOR shrinkage - check convergence (ratio={shrinkage_ratio:.1f})")

                        # Compare to stability results if available
                        n_stable_factors = stability_results.get("n_stable_factors", 0)
                        if n_stable_factors > 0:
                            self.logger.info(f"   🔍 Cross-check: {n_stable_factors} stable factors vs {n_active} active factors")
                            if abs(n_stable_factors - n_active) > 2:
                                self.logger.warning(f"      ⚠️  Mismatch between stability ({n_stable_factors}) and variance ({n_active})")
                                self.logger.warning(f"         → May indicate measurement artifact or convergence issues")

                        self.logger.info("   ✅ Factor variance profile analysis created")

                    except Exception as e:
                        self.logger.warning(f"Failed to create variance profile: {str(e)}")
                        import traceback
                        self.logger.warning(f"Traceback: {traceback.format_exc()}")

                else:
                    self.logger.warning("  ⚠️  No samples found in chain_results for trace plots")

            except Exception as e:
                self.logger.warning(f"Failed to create MCMC trace plots: {str(e)}")
                import traceback
                self.logger.warning(f"Traceback: {traceback.format_exc()}")

        # Save all plots as individual files
        try:
            from core.io_utils import save_all_plots_individually
            from core.config_utils import ConfigHelper
            config_dict = ConfigHelper.to_dict(self.config)
            output_dir = get_output_dir(config_dict) / "factor_stability" / "individual_plots"
            save_all_plots_individually(plots, output_dir, dpi=300)
            self.logger.info(f"✅ Saved {len(plots)} individual plots to {output_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to save individual plots: {e}")

        return plots


def run_robustness_testing(config):
    """Run robustness tests with remote workstation integration."""
    import logging
    import os
    import sys

    import numpy as np

    logger = logging.getLogger(__name__)
    logger.info("Starting Robustness Tests")

    try:
        # Add project root to path for imports
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Check if shared data is available from previous experiments
        from data.preprocessing_integration import apply_preprocessing_to_pipeline
        from experiments.framework import ExperimentConfig, ExperimentFramework
        from core.config_utils import ConfigHelper

        config_dict = ConfigHelper.to_dict(config)

        # Check for shared data from data_validation
        if "_shared_data" in config_dict and config_dict["_shared_data"].get("X_list") is not None:
            logger.info("🔗 Using shared preprocessed data from data_validation...")
            X_list = config_dict["_shared_data"]["X_list"]
            preprocessing_info = config_dict["_shared_data"].get("preprocessing_info", {})
            logger.info(f"✅ Shared data: {len(X_list)} views for robustness testing")
            for i, X in enumerate(X_list):
                logger.info(f"   View {i}: {X.shape}")
        else:
            # Load data with standard preprocessing for robustness testing
            logger.info("🔧 Loading data for robustness testing...")
            # Get preprocessing strategy from config
            preprocessing_config = config_dict.get("preprocessing", {})
            strategy = preprocessing_config.get("strategy", "standard")

            X_list, preprocessing_info = apply_preprocessing_to_pipeline(
                config=config_dict,
                data_dir=get_data_dir(config_dict),
                auto_select_strategy=False,
                preferred_strategy=strategy,  # Use strategy from config
                output_dir=get_output_dir(config_dict),
            )

            logger.info(f"✅ Data loaded: {len(X_list)} views for robustness testing")
            for i, X in enumerate(X_list):
                logger.info(f"   View {i}: {X.shape}")

        # Initialize experiment framework
        framework = ExperimentFramework(get_output_dir(config))

        # Check for command-line K override and read hyperparameters first
        model_config = config_dict.get("model", {})
        K_override = model_config.get("K", None)
        K_value = K_override if K_override is not None else 10
        if K_override is not None:
            logger.info(f"   Using command-line K override: {K_value} (default is 10)")

        # Read hyperparameters from global model section for consistency across pipeline
        percW_value = model_config.get("percW", 33)  # Default 33
        slab_df_value = model_config.get("slab_df", 4)
        slab_scale_value = model_config.get("slab_scale", 2)
        logger.info(f"   Using global model hyperparameters: K={K_value}, percW={percW_value}, slab_df={slab_df_value}, slab_scale={slab_scale_value}")

        # Get QC outlier threshold (MAD) from preprocessing config for semantic naming
        preprocessing_config = config_dict.get("preprocessing", {})
        qc_outlier_threshold = preprocessing_config.get("qc_outlier_threshold", None)
        if qc_outlier_threshold:
            logger.info(f"   QC outlier threshold (MAD): {qc_outlier_threshold}")

        # Get model type from config
        model_type = model_config.get("model_type", "sparseGFA")
        logger.info(f"   Using model type: {model_type}")

        # Create ExperimentConfig with model parameters for semantic naming
        exp_config = ExperimentConfig(
            experiment_name="robustness_tests",
            description="Robustness testing for SGFA",
            dataset="qmap_pd",
            data_dir=get_data_dir(config),
            model_type=model_type,  # Use model_type from config
            K=K_value,
            percW=percW_value,
            slab_df=slab_df_value,
            slab_scale=slab_scale_value,
            qc_outlier_threshold=qc_outlier_threshold,
            max_tree_depth=config_dict.get("mcmc", {}).get("max_tree_depth"),  # For semantic naming
        )

        # Create robustness experiment instance
        repro_exp = RobustnessExperiments(exp_config, logger)

        # Setup base hyperparameters
        base_hypers = {
            "Dm": [X.shape[1] for X in X_list],
            "a_sigma": 1.0,
            "b_sigma": 1.0,
            "slab_df": slab_df_value,
            "slab_scale": slab_scale_value,
            "percW": percW_value,
            "K": K_value,  # Use K_value which respects command-line override
        }

        # Setup base args
        base_args = {
            "K": K_value,  # Use K_value which respects command-line override
            "num_warmup": 50,  # Reduced for robustness testing speed
            "num_samples": 100,  # Reduced for robustness testing speed
            "num_chains": 1,
            "target_accept_prob": config_dict.get("mcmc", {}).get("target_accept_prob", 0.8),  # Read from mcmc config (respects command-line override)
            "reghsZ": True,
            "max_tree_depth": config_dict.get("mcmc", {}).get("max_tree_depth"),  # NUTS max tree depth
            "use_pca_initialization": model_config.get("use_pca_initialization", False),  # PCA initialization
            "model_type": model_type,  # Pass model type for PCA init
        }

        # Run the experiment
        def robustness_experiment(config, output_dir, **kwargs):
            logger.info("🔄 Running comprehensive robustness tests...")

            # Normalize config input using standard ConfigHelper
            from core.config_utils import ConfigHelper
            config_dict = ConfigHelper.to_dict(config)

            # Get robustness testing configuration
            repro_config = config_dict.get("robustness_testing", {})
            seed_values = repro_config.get("seed_values", [42, 123, 456])
            test_scenarios = repro_config.get("test_scenarios", ["seed_robustness", "data_perturbation", "initialization_robustness"])
            perturbation_config = repro_config.get("perturbation", {
                "types": ["gaussian", "uniform", "dropout"],
                "levels": [0.01, 0.05, 0.1]
            })

            results = {}
            total_tests = 0
            successful_tests = 0

            # 1. Test seed robustness (if configured)
            if "seed_robustness" in test_scenarios:
                logger.info("📊 Testing seed robustness...")
                seeds = seed_values
                seed_results = {}

                for seed_idx, seed in enumerate(seeds):
                    try:
                        # Set seed for robustness
                        np.random.seed(seed)
                        test_args = base_args.copy()
                        test_args["random_seed"] = seed

                        with repro_exp.profiler.profile(f"seed_{seed}") as p:
                            # Only verbose for first seed
                            result = repro_exp._run_sgfa_analysis(
                                X_list, base_hypers, test_args, verbose=(seed_idx == 0)
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
                            f"✅ Seed {seed}: { metrics.execution_time:.1f}s, LL={ result.get( 'log_likelihood', 0):.2f}"
                        )

                    except Exception as e:
                        logger.error(f"❌ Seed {seed} test failed: {e}")
                        seed_results[f"seed_{seed}"] = {"error": str(e)}

                    # Cleanup memory after each seed test
                    import jax
                    import gc
                    jax.clear_caches()
                    gc.collect()

                    total_tests += 1

                results["seed_robustness"] = seed_results

            # 2. Test data perturbation robustness (if configured)
            if "data_perturbation" in test_scenarios:
                logger.info("📊 Testing data perturbation robustness...")
                noise_levels = perturbation_config.get("levels", [0.01, 0.05])
                perturbation_results = {}

                for noise_idx, noise_level in enumerate(noise_levels):
                    try:
                        # Add Gaussian noise to data
                        X_noisy = []
                        for X in X_list:
                            noise = np.random.normal(0, noise_level * np.std(X), X.shape)
                            X_noisy.append(X + noise)

                        with repro_exp.profiler.profile(f"noise_{noise_level}") as p:
                            # Only verbose for first noise level
                            result = repro_exp._run_sgfa_analysis(
                                X_noisy, base_hypers, base_args, verbose=(noise_idx == 0)
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
                            f"✅ Noise {noise_level}: { metrics.execution_time:.1f}s, LL={ result.get( 'log_likelihood', 0):.2f}"
                        )

                    except Exception as e:
                        logger.error(f"❌ Noise {noise_level} test failed: {e}")
                        perturbation_results[f"noise_{noise_level}"] = {"error": str(e)}

                    total_tests += 1

                results["perturbation_robustness"] = perturbation_results

            # 3. Test initialization robustness (if configured)
            if "initialization_robustness" in test_scenarios:
                logger.info("📊 Testing initialization robustness...")
                init_results = {}
                n_inits = repro_config.get("n_initializations", 3)  # Test different initializations

                for init_id in range(n_inits):
                    try:
                        # Different random initialization
                        test_args = base_args.copy()
                        test_args["random_seed"] = 1000 + init_id

                        with repro_exp.profiler.profile(f"init_{init_id}") as p:
                            # Only verbose for first initialization
                            result = repro_exp._run_sgfa_analysis(
                                X_list, base_hypers, test_args, verbose=(init_id == 0)
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
                            f"✅ Init {init_id}: { metrics.execution_time:.1f}s, LL={ result.get( 'log_likelihood', 0):.2f}"
                        )

                    except Exception as e:
                        logger.error(f"❌ Init {init_id} test failed: {e}")
                        init_results[f"init_{init_id}"] = {"error": str(e)}

                    total_tests += 1

                results["initialization_robustness"] = init_results

            logger.info("🔄 Robustness tests completed!")
            logger.info(f"   Successful tests: {successful_tests}/{total_tests}")

            # Generate robustness plots
            import matplotlib.pyplot as plt
            plots = {}

            try:
                # Determine which tests were run to create appropriate grid
                has_seed = "seed_robustness" in results and any("error" not in results["seed_robustness"][k] for k in results["seed_robustness"].keys())
                has_perturbation = "data_perturbation" in results and any("error" not in results["data_perturbation"][k] for k in results["data_perturbation"].keys())
                has_init = "initialization_robustness" in results and any("error" not in results["initialization_robustness"][k] for k in results["initialization_robustness"].keys())

                # Count how many test plots we have (excluding summary)
                test_plots = sum([has_seed, has_perturbation, has_init])

                # Create grid: if all 3 tests ran, use 2x2; otherwise use 1xN
                if test_plots == 3 and has_perturbation:
                    # 2x2 grid with all tests
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    axes = axes.flatten()
                else:
                    # 1xN grid (N = number of plots including summary)
                    n_plots = test_plots + 1  # +1 for summary
                    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
                    if n_plots == 1:
                        axes = [axes]

                fig.suptitle("Robustness Analysis", fontsize=16)

                plot_idx = 0

                # Plot 1: Seed robustness (if run)
                if has_seed:
                    seed_data = results["seed_robustness"]
                    seeds = [seed_data[k]["seed"] for k in sorted(seed_data.keys()) if "error" not in seed_data[k]]
                    lls = [seed_data[k]["performance"]["log_likelihood"] for k in sorted(seed_data.keys()) if "error" not in seed_data[k]]
                    axes[plot_idx].plot(seeds, lls, 'o-', linewidth=2, markersize=8)
                    axes[plot_idx].set_xlabel("Random Seed")
                    axes[plot_idx].set_ylabel("Log Likelihood")
                    axes[plot_idx].set_title("Seed Robustness")
                    axes[plot_idx].grid(True, alpha=0.3)
                    plot_idx += 1

                # Plot 2: Data perturbation (if run)
                if has_perturbation:
                    perturb_data = results["data_perturbation"]
                    noise_levels = [perturb_data[k]["noise_level"] for k in sorted(perturb_data.keys()) if "error" not in perturb_data[k]]
                    lls = [perturb_data[k]["performance"]["log_likelihood"] for k in sorted(perturb_data.keys()) if "error" not in perturb_data[k]]
                    axes[plot_idx].plot(noise_levels, lls, 's-', linewidth=2, markersize=8)
                    axes[plot_idx].set_xlabel("Noise Level")
                    axes[plot_idx].set_ylabel("Log Likelihood")
                    axes[plot_idx].set_title("Data Perturbation Robustness")
                    axes[plot_idx].grid(True, alpha=0.3)
                    plot_idx += 1

                # Plot 3: Initialization robustness (if run)
                if has_init:
                    init_data = results["initialization_robustness"]
                    init_ids = list(range(len([k for k in init_data.keys() if "error" not in init_data[k]])))
                    lls = [init_data[k]["performance"]["log_likelihood"] for k in sorted(init_data.keys()) if "error" not in init_data[k]]
                    axes[plot_idx].plot(init_ids, lls, '^-', linewidth=2, markersize=8)
                    axes[plot_idx].set_xlabel("Initialization ID")
                    axes[plot_idx].set_ylabel("Log Likelihood")
                    axes[plot_idx].set_title("Initialization Robustness")
                    axes[plot_idx].grid(True, alpha=0.3)
                    plot_idx += 1

                # Plot N: Test success summary (always included)
                test_names = []
                test_success = []
                if has_seed:
                    test_names.append("Seed\nrobustness")
                    test_success.append(len([k for k in results["seed_robustness"].keys() if "error" not in results["seed_robustness"][k]]))
                if has_perturbation:
                    test_names.append("Data\nPerturbation")
                    test_success.append(len([k for k in results["data_perturbation"].keys() if "error" not in results["data_perturbation"][k]]))
                if has_init:
                    test_names.append("Initialization\nRobustness")
                    test_success.append(len([k for k in results["initialization_robustness"].keys() if "error" not in results["initialization_robustness"][k]]))

                if test_names:
                    axes[plot_idx].bar(test_names, test_success)
                    axes[plot_idx].set_ylabel("Successful Tests")
                    axes[plot_idx].set_title(f"Test Summary ({successful_tests}/{total_tests} total)")
                    axes[plot_idx].grid(True, alpha=0.3, axis='y')

                plt.tight_layout()
                plots["robustness_summary"] = fig

            except Exception as e:
                logger.warning(f"Failed to create robustness plots: {e}")

            # Save all plots as individual files
            try:
                from core.io_utils import save_all_plots_individually
                output_dir = get_output_dir(config_dict) / "robustness_testing" / "individual_plots"
                save_all_plots_individually(plots, output_dir, dpi=300)
                logger.info(f"✅ Saved {len(plots)} individual plots to {output_dir}")
            except Exception as e:
                logger.warning(f"Failed to save individual plots: {e}")

            return {
                "status": "completed",
                "robustness_results": results,
                "plots": plots,
                "summary": {
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "success_rate": (
                        successful_tests / total_tests if total_tests > 0 else 0
                    ),
                    "test_categories": [
                        "seed_robustness",
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
            experiment_function=robustness_experiment,
            config=exp_config,
            model_results={"X_list": X_list, "preprocessing_info": preprocessing_info},
        )

        logger.info("✅ Robustness tests completed successfully")
        return result

    except Exception as e:
        logger.error(f"Robustness tests failed: {e}")
        return None
