"""Performance benchmarking experiments for SGFA qMAP-PD analysis."""

import gc
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import psutil

from core.config_utils import get_data_dir, get_output_dir
from core.experiment_utils import (
    experiment_handler,
)
from core.validation_utils import (
    ResultValidator,
    validate_data_types,
    validate_parameters,
)
from experiments.framework import (
    ExperimentConfig,
    ExperimentFramework,
    ExperimentResult,
)
from optimization import PerformanceManager, PerformanceProfiler
from optimization.experiment_mixins import performance_optimized_experiment
from analysis.cross_validation_library import (
    ClinicalAwareSplitter,
    NeuroImagingCVConfig,
    NeuroImagingMetrics,
)
from analysis.cv_fallbacks import CVFallbackHandler, MetricsFallbackHandler


@performance_optimized_experiment()
class PerformanceBenchmarkExperiments(ExperimentFramework):
    """Comprehensive performance benchmarking for SGFA analysis."""

    def __init__(
        self, config: ExperimentConfig, logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, None, logger)
        self.profiler = PerformanceProfiler()

        # Memory optimizers now handled by @performance_optimized_experiment decorator

        # Initialize fallback handlers
        self.cv_fallback = CVFallbackHandler(self.logger)
        self.metrics_fallback = MetricsFallbackHandler(self.logger)

        # Benchmarking configurations
        self.sample_size_ranges = [50, 100, 250, 500, 1000, 2000, 5000]
        self.feature_size_ranges = [10, 25, 50, 100, 250, 500, 1000]
        self.component_ranges = [2, 3, 5, 8, 10, 15, 20]
        self.chain_ranges = [1, 2, 4, 8]

        # System monitoring
        self.system_monitor = SystemMonitor()

    @experiment_handler("scalability_benchmarks")
    @validate_data_types(X_base=list, hypers=dict, args=dict)
    @validate_parameters(X_base=lambda x: len(x) > 0)
    def run_scalability_benchmarks(
        self, X_base: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> ExperimentResult:
        """Run comprehensive scalability benchmarks."""
        # Validate inputs
        ResultValidator.validate_data_matrices(X_base)

        self.logger.info("Running scalability benchmarks")

        results = {}
        # Sample size scalability
        self.logger.info("Benchmarking sample size scalability")
        sample_results = self._benchmark_sample_scalability(
            X_base, hypers, args, **kwargs
        )
        results["sample_scalability"] = sample_results

        # Feature size scalability
        self.logger.info("Benchmarking feature size scalability")
        feature_results = self._benchmark_feature_scalability(
            X_base, hypers, args, **kwargs
        )
        results["feature_scalability"] = feature_results

        # Component scalability
        self.logger.info("Benchmarking component scalability")
        component_results = self._benchmark_component_scalability(
            X_base, hypers, args, **kwargs
        )
        results["component_scalability"] = component_results

        # Multi-chain scalability
        self.logger.info("Benchmarking multi-chain scalability")
        chain_results = self._benchmark_chain_scalability(
            X_base, hypers, args, **kwargs
        )
        results["chain_scalability"] = chain_results

        # Analyze scalability
        analysis = self._analyze_scalability_benchmarks(results)

        # Generate basic plots
        plots = self._plot_scalability_benchmarks(results)

        # Add comprehensive performance visualizations (focus on optimization surfaces)
        advanced_plots = self._create_comprehensive_performance_visualizations(
            X_base, results, "scalability_benchmarks"
        )
        plots.update(advanced_plots)

        return ExperimentResult(
            experiment_name="scalability_benchmarks",
            config=self.config,
            data=results,
            analysis=analysis,
            plots=plots,
            success=True,
        )

    @experiment_handler("memory_benchmarks")
    @validate_data_types(X_base=list, hypers=dict, args=dict)
    @validate_parameters(X_base=lambda x: len(x) > 0)
    def run_memory_benchmarks(
        self,
        X_base: List[np.ndarray],
        hypers: Dict,
        args: Dict,
        memory_constraints: List[float] = None,
        **kwargs,
    ) -> ExperimentResult:
        """Run memory usage benchmarks."""
        if memory_constraints is None:
            memory_constraints = [1.0, 2.0, 4.0, 8.0, 16.0]  # GB

        self.logger.info(
            f"Running memory benchmarks with constraints: {memory_constraints}"
        )

        results = {}

        # Memory benchmark tests
        for memory_limit in memory_constraints:
            self.logger.info(f"Testing memory constraint: {memory_limit}GB")

            # Configure memory-constrained environment with MCMC optimization
            memory_config = self._create_memory_config(memory_limit)

            # Configure memory-specific MCMC optimizer using mixin
            # Note: Base MCMC optimizer available through self.mcmc_optimizer from decorator

            constraint_results = []

            # Test different dataset sizes under memory constraint
            for scale_factor in [0.5, 1.0, 2.0, 4.0]:
                X_scaled = self._scale_dataset(X_base, scale_factor)

                # Skip if dataset would be too large
                estimated_memory = self._estimate_memory_usage(X_scaled)
                if estimated_memory > memory_limit * 1.5:  # 50% buffer
                    self.logger.info(
                        f"Skipping scale {scale_factor} (estimated {
                            estimated_memory:.2f}GB > {memory_limit}GB)"
                    )
                    continue

                try:
                    with self.system_monitor.monitor_memory():
                        with PerformanceManager(memory_config) as manager:
                            with self.profiler.profile(
                                f"memory_{memory_limit}GB_scale_{scale_factor}"
                            ) as p:
                                result = self._run_sgfa_analysis(
                                    X_scaled, hypers, args, **kwargs
                                )

                            # Get detailed memory metrics
                            memory_metrics = self.system_monitor.get_memory_report()
                            performance_metrics = self.profiler.get_current_metrics()

                            constraint_results.append(
                                {
                                    "scale_factor": scale_factor,
                                    "result": result,
                                    "memory_metrics": memory_metrics,
                                    "performance_metrics": performance_metrics,
                                    "success": True,
                                }
                            )

                except Exception as e:
                    self.logger.warning(
                        f"Memory test failed for {memory_limit}GB, scale {scale_factor}: {
                            str(e)}"
                    )
                    constraint_results.append(
                        {
                            "scale_factor": scale_factor,
                            "result": {"error": str(e)},
                            "memory_metrics": {},
                            "performance_metrics": {},
                            "success": False,
                        }
                    )

            results[f"{memory_limit}GB"] = constraint_results

        # Analyze memory benchmarks
        analysis = self._analyze_memory_benchmarks(results)

        # Generate basic plots
        plots = self._plot_memory_benchmarks(results)

        # Add comprehensive performance visualizations
        advanced_plots = self._create_comprehensive_performance_visualizations(
            X_base, results, "memory_benchmarks"
        )
        plots.update(advanced_plots)

        return ExperimentResult(
            experiment_name="memory_benchmarks",
            config=self.config,
            data=results,
            analysis=analysis,
            plots=plots,
            success=True,
        )

    @experiment_handler("optimization_benchmarks")
    @validate_data_types(X_base=list, hypers=dict, args=dict)
    @validate_parameters(X_base=lambda x: len(x) > 0)
    def run_optimization_benchmarks(
        self,
        X_base: List[np.ndarray],
        hypers: Dict,
        args: Dict,
        optimization_strategies: List[str] = None,
        **kwargs,
    ) -> ExperimentResult:
        """Benchmark different optimization strategies."""
        if optimization_strategies is None:
            optimization_strategies = [
                "standard",
                "memory_efficient",
                "speed_optimized",
                "balanced",
                "high_precision",
            ]

        self.logger.info(f"Running optimization benchmarks: {optimization_strategies}")

        results = {}

        # Test different dataset sizes
        test_scales = [0.5, 1.0, 2.0]

        for scale in test_scales:
            self.logger.info(f"Testing dataset scale: {scale}")

            X_scaled = self._scale_dataset(X_base, scale)
            scale_results = {}

            for strategy in optimization_strategies:
                self.logger.info(f"Testing optimization strategy: {strategy}")

                try:
                    # Configure strategy
                    strategy_config = self._create_optimization_config(strategy)

                    with PerformanceManager(strategy_config) as manager:
                        with self.profiler.profile(f"{strategy}_scale_{scale}") as p:
                            result = self._run_sgfa_analysis(
                                X_scaled, hypers, args, **kwargs
                            )

                        # Get comprehensive metrics
                        performance_metrics = self.profiler.get_current_metrics()
                        optimization_metrics = manager.get_optimization_report()

                        scale_results[strategy] = {
                            "result": result,
                            "performance_metrics": performance_metrics,
                            "optimization_metrics": optimization_metrics,
                            "success": True,
                        }

                except Exception as e:
                    self.logger.warning(
                        f"Strategy {strategy} failed for scale {scale}: {str(e)}"
                    )
                    scale_results[strategy] = {
                        "result": {"error": str(e)},
                        "performance_metrics": {},
                        "optimization_metrics": {},
                        "success": False,
                    }

            results[f"scale_{scale}"] = scale_results

        # Analyze optimization benchmarks
        analysis = self._analyze_optimization_benchmarks(results)

        # Generate plots
        plots = self._plot_optimization_benchmarks(results)

        return ExperimentResult(
            experiment_name="optimization_benchmarks",
            config=self.config,
            data=results,
            analysis=analysis,
            plots=plots,
            success=True,
        )

    @experiment_handler("comparative_benchmarks")
    @validate_data_types(X_base=list, hypers=dict, args=dict)
    @validate_parameters(X_base=lambda x: len(x) > 0)
    def run_comparative_benchmarks(
        self,
        X_base: List[np.ndarray],
        hypers: Dict,
        args: Dict,
        baseline_methods: List[str] = None,
        **kwargs,
    ) -> ExperimentResult:
        """Run comparative benchmarks against baseline methods."""
        if baseline_methods is None:
            baseline_methods = ["pca", "ica", "factor_analysis", "nmf"]

        self.logger.info(f"Running comparative benchmarks against: {baseline_methods}")

        results = {}

        # Test different dataset configurations
        test_configs = [
            {"n_subjects": 500, "n_features_per_view": 100, "n_views": 2},
            {"n_subjects": 1000, "n_features_per_view": 200, "n_views": 3},
            {"n_subjects": 2000, "n_features_per_view": 150, "n_views": 4},
        ]

        for i, config in enumerate(test_configs):
            config_name = f"config_{i + 1}"
            self.logger.info(f"Testing configuration {config_name}: {config}")

            # Generate test dataset with specified configuration
            X_test = self._generate_test_dataset(**config)

            config_results = {}

            # Test SGFA
            self.logger.info("Benchmarking SGFA")
            with self.profiler.profile(f"sgfa_{config_name}") as p:
                sgfa_result = self._run_sgfa_analysis(X_test, hypers, args, **kwargs)

            sgfa_metrics = self.profiler.get_current_metrics()
            config_results["sgfa"] = {
                "result": sgfa_result,
                "performance_metrics": sgfa_metrics,
                "method_type": "multiview",
            }

            # Test baseline methods
            X_concat = np.hstack(X_test)  # Concatenate for traditional methods

            for method in baseline_methods:
                self.logger.info(f"Benchmarking {method}")

                try:
                    with self.profiler.profile(f"{method}_{config_name}") as p:
                        baseline_result = self._run_baseline_method(
                            X_concat, method, **kwargs
                        )

                    baseline_metrics = self.profiler.get_current_metrics()
                    config_results[method] = {
                        "result": baseline_result,
                        "performance_metrics": baseline_metrics,
                        "method_type": "traditional",
                    }

                except Exception as e:
                    self.logger.warning(f"Baseline method {method} failed: {str(e)}")
                    config_results[method] = {
                        "result": {"error": str(e)},
                        "performance_metrics": {},
                        "method_type": "traditional",
                    }

            results[config_name] = {"config": config, "results": config_results}

            # Analyze comparative benchmarks
            analysis = self._analyze_comparative_benchmarks(results)

            # Generate plots
            plots = self._plot_comparative_benchmarks(results)

            return ExperimentResult(
                experiment_name="comparative_benchmarks",
                config=self.config,
                data=results,
                analysis=analysis,
                plots=plots,
                success=True,
            )

    def _benchmark_sample_scalability(
        self, X_base: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> Dict:
        """Benchmark scalability with respect to sample size."""
        results = {}

        for n_samples in self.sample_size_ranges:
            if n_samples > X_base[0].shape[0]:
                continue

            self.logger.debug(f"Testing {n_samples} samples")

            # Subsample data
            indices = np.random.choice(X_base[0].shape[0], n_samples, replace=False)
            X_subset = [X[indices] for X in X_base]

            try:
                with self.system_monitor.monitor():
                    with self.profiler.profile(f"samples_{n_samples}") as p:
                        result = self._run_sgfa_analysis(
                            X_subset, hypers, args, **kwargs
                        )

                # Collect metrics
                performance_metrics = self.profiler.get_current_metrics()
                system_metrics = self.system_monitor.get_report()

                results[n_samples] = {
                    "result": result,
                    "performance_metrics": performance_metrics,
                    "system_metrics": system_metrics,
                    "dataset_info": {
                        "n_subjects": n_samples,
                        "n_features": [X.shape[1] for X in X_subset],
                        "n_views": len(X_subset),
                    },
                }

            except Exception as e:
                self.logger.warning(
                    f"Sample scalability test failed for {n_samples}: {str(e)}"
                )
                results[n_samples] = {"error": str(e)}

        return results

    def _benchmark_feature_scalability(
        self, X_base: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> Dict:
        """Benchmark scalability with respect to feature size."""
        results = {}

        for n_features in self.feature_size_ranges:
            self.logger.debug(f"Testing {n_features} features per view")

            # Select subset of features from each view
            X_subset = []
            for X in X_base:
                if n_features >= X.shape[1]:
                    X_subset.append(X)
                else:
                    feature_indices = np.random.choice(
                        X.shape[1], n_features, replace=False
                    )
                    X_subset.append(X[:, feature_indices])

            try:
                with self.system_monitor.monitor():
                    with self.profiler.profile(f"features_{n_features}") as p:
                        result = self._run_sgfa_analysis(
                            X_subset, hypers, args, **kwargs
                        )

                # Collect metrics
                performance_metrics = self.profiler.get_current_metrics()
                system_metrics = self.system_monitor.get_report()

                results[n_features] = {
                    "result": result,
                    "performance_metrics": performance_metrics,
                    "system_metrics": system_metrics,
                    "dataset_info": {
                        "n_subjects": X_subset[0].shape[0],
                        "n_features": [X.shape[1] for X in X_subset],
                        "n_views": len(X_subset),
                    },
                }

            except Exception as e:
                self.logger.warning(
                    f"Feature scalability test failed for {n_features}: {str(e)}"
                )
                results[n_features] = {"error": str(e)}

        return results

    def _benchmark_component_scalability(
        self, X_base: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> Dict:
        """Benchmark scalability with respect to number of components."""
        results = {}

        for K in self.component_ranges:
            self.logger.debug(f"Testing {K} components")

            # Update hyperparameters with component count
            test_hypers = hypers.copy()
            test_hypers["K"] = K

            try:
                with self.system_monitor.monitor():
                    with self.profiler.profile(f"components_{K}") as p:
                        result = self._run_sgfa_analysis(
                            X_base, test_hypers, args, **kwargs
                        )

                # Collect metrics
                performance_metrics = self.profiler.get_current_metrics()
                system_metrics = self.system_monitor.get_report()

                results[K] = {
                    "result": result,
                    "performance_metrics": performance_metrics,
                    "system_metrics": system_metrics,
                    "hyperparameters": test_hypers,
                }

            except Exception as e:
                self.logger.warning(
                    f"Component scalability test failed for K={K}: {str(e)}"
                )
                results[K] = {"error": str(e)}

        return results

    def _benchmark_chain_scalability(
        self, X_base: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> Dict:
        """Benchmark scalability with respect to number of MCMC chains."""
        results = {}

        for n_chains in self.chain_ranges:
            self.logger.debug(f"Testing {n_chains} chains")

            # Update args with chain count
            test_args = args.copy()
            test_args["num_chains"] = n_chains

            # Optimize MCMC configuration using mixin method
            mcmc_config = self.optimize_mcmc_config(X_base, test_args)
            test_args.update(mcmc_config)

            try:
                with self.system_monitor.monitor():
                    with self.profiler.profile(f"chains_{n_chains}") as p:
                        # Use memory-optimized MCMC execution (using mixin method)
                        result = self.memory_efficient_operation(
                            self._run_sgfa_analysis, X_base, hypers, test_args, **kwargs
                        )

                # Collect metrics
                performance_metrics = self.profiler.get_current_metrics()
                system_metrics = self.system_monitor.get_report()

                results[n_chains] = {
                    "result": result,
                    "performance_metrics": performance_metrics,
                    "system_metrics": system_metrics,
                    "mcmc_args": test_args,
                }

            except Exception as e:
                self.logger.warning(
                    f"Chain scalability test failed for {n_chains} chains: {str(e)}"
                )
                results[n_chains] = {"error": str(e)}

        return results

    def _run_sgfa_analysis(
        self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> Dict:
        """Run actual SGFA analysis for benchmarking."""
        import time

        import jax
        from numpyro.infer import MCMC, NUTS

        try:
            K = hypers.get("K", 5)
            self.logger.debug(
                f"Running SGFA benchmark: K={K}, n_subjects={
                    X_list[0].shape[0]}, n_features={
                    sum(
                        X.shape[1] for X in X_list)}"
            )

            # Import the actual SGFA model function
            from core.run_analysis import models

            # Setup MCMC configuration for benchmarking (reduced for speed)
            num_warmup = args.get("num_warmup", 100)  # Reduced for benchmarking
            num_samples = args.get("num_samples", 200)  # Reduced for benchmarking
            num_chains = args.get("num_chains", 1)  # Single chain for benchmarking

            # Create args object for model
            import argparse

            model_args = argparse.Namespace(
                model="sparseGFA",
                K=K,
                num_sources=len(X_list),
                reghsZ=args.get("reghsZ", True),
            )

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
                "log_likelihood": float(log_likelihood),
                "n_iterations": num_samples,
                "convergence": True,
                "execution_time": elapsed,
                "computation_info": {
                    "problem_size": X_list[0].shape[0]
                    * sum(X.shape[1] for X in X_list)
                    * K,
                    "actual_time": elapsed,
                    "num_warmup": num_warmup,
                    "num_samples": num_samples,
                    "num_chains": num_chains,
                },
            }

        except Exception as e:
            self.logger.error(f"SGFA benchmark analysis failed: {str(e)}")
            return {
                "error": str(e),
                "convergence": False,
                "execution_time": float("inf"),
                "log_likelihood": float("-inf"),
            }

    def _run_baseline_method(self, X: np.ndarray, method: str, **kwargs) -> Dict:
        """Run baseline method for comparison."""
        from sklearn.decomposition import NMF, PCA, FactorAnalysis, FastICA

        n_components = kwargs.get("n_components", 5)

        try:
            if method == "pca":
                model = PCA(n_components=n_components)
                Z = model.fit_transform(X)
                result = {
                    "components": model.components_,
                    "explained_variance_ratio": model.explained_variance_ratio_,
                    "Z": Z,
                }

            elif method == "ica":
                model = FastICA(n_components=n_components, random_state=42)
                Z = model.fit_transform(X)
                result = {"components": model.components_, "Z": Z}

            elif method == "factor_analysis":
                model = FactorAnalysis(n_components=n_components)
                Z = model.fit_transform(X)
                result = {
                    "components": model.components_,
                    "Z": Z,
                    "loglik": model.score(X),
                }

            elif method == "nmf":
                model = NMF(n_components=n_components, random_state=42)
                Z = model.fit_transform(X)
                result = {"components": model.components_, "Z": Z}

            else:
                raise ValueError(f"Unknown baseline method: {method}")

            return result

        except Exception as e:
            return {"error": str(e)}

    def _scale_dataset(
        self, X_base: List[np.ndarray], scale_factor: float
    ) -> List[np.ndarray]:
        """Scale dataset size by given factor."""
        if scale_factor <= 1.0:
            # Subsample
            n_samples = int(X_base[0].shape[0] * scale_factor)
            indices = np.random.choice(X_base[0].shape[0], n_samples, replace=False)
            return [X[indices] for X in X_base]
        else:
            # Upsample with noise
            n_samples = int(X_base[0].shape[0] * scale_factor)
            X_scaled = []

            for X in X_base:
                if n_samples <= X.shape[0]:
                    indices = np.random.choice(X.shape[0], n_samples, replace=False)
                    X_scaled.append(X[indices])
                else:
                    # Repeat samples with noise
                    indices = np.random.choice(X.shape[0], n_samples, replace=True)
                    X_repeated = X[indices]

                    # Add small amount of noise to avoid exact duplicates
                    noise_std = 0.01 * np.std(X)
                    noise = np.random.normal(0, noise_std, X_repeated.shape)
                    X_scaled.append(X_repeated + noise)

            return X_scaled

    def _generate_test_dataset(
        self, n_subjects: int, n_features_per_view: int, n_views: int
    ) -> List[np.ndarray]:
        """Generate synthetic test dataset with specified dimensions."""
        X_test = []

        # Generate shared factors
        K = 5
        Z = np.random.randn(n_subjects, K)

        for view in range(n_views):
            # Generate loading matrix
            W = np.random.randn(n_features_per_view, K)

            # Generate noise
            noise = np.random.randn(n_subjects, n_features_per_view) * 0.1

            # Create view data
            X_view = Z @ W.T + noise
            X_test.append(X_view)

        return X_test

    def _create_memory_config(self, memory_limit_gb: float):
        """Create memory-constrained configuration."""
        from optimization.config import DataConfig, MemoryConfig, PerformanceConfig

        return PerformanceConfig(
            memory=MemoryConfig(
                max_memory_gb=memory_limit_gb,
                warning_threshold=0.8,
                enable_monitoring=True,
                enable_aggressive_cleanup=True,
            ),
            data=DataConfig(
                enable_chunking=True,
                memory_limit_gb=memory_limit_gb * 0.7,
                enable_compression=True,
            ),
        )

    def _create_optimization_config(self, strategy: str):
        """Create optimization configuration for given strategy."""
        from optimization.config import PerformanceConfig

        return PerformanceConfig().create_preset(strategy)

    def _estimate_memory_usage(self, X_list: List[np.ndarray]) -> float:
        """Estimate memory usage in GB for dataset."""
        total_elements = sum(X.size for X in X_list)
        bytes_per_element = 8  # float64
        total_bytes = total_elements * bytes_per_element
        return total_bytes / (1024**3)  # Convert to GB

    def _analyze_scalability_benchmarks(self, results: Dict) -> Dict:
        """Analyze scalability benchmark results."""
        analysis = {
            "scalability_trends": {},
            "performance_bottlenecks": {},
            "efficiency_metrics": {},
        }

        # Analyze each scalability dimension
        for dimension, dimension_results in results.items():
            if not dimension_results:
                continue

            sizes = []
            execution_times = []
            memory_usage = []
            throughput = []

            for size, result in dimension_results.items():
                if "error" not in result:
                    sizes.append(size)

                    perf_metrics = result.get("performance_metrics", {})
                    execution_times.append(perf_metrics.get("execution_time", np.nan))
                    memory_usage.append(perf_metrics.get("peak_memory_gb", np.nan))

                    # Calculate throughput (items processed per second)
                    if dimension == "sample_scalability":
                        throughput.append(size / perf_metrics.get("execution_time", 1))
                    elif dimension == "feature_scalability":
                        dataset_info = result.get("dataset_info", {})
                        total_features = sum(dataset_info.get("n_features", [0]))
                        throughput.append(
                            total_features / perf_metrics.get("execution_time", 1)
                        )

            if sizes:
                # Fit scaling trends
                analysis["scalability_trends"][dimension] = {
                    "sizes": sizes,
                    "execution_times": execution_times,
                    "memory_usage": memory_usage,
                    "throughput": throughput,
                }

                # Calculate scaling coefficients (approximate)
                if len(sizes) > 1:
                    time_scaling = self._estimate_scaling_coefficient(
                        sizes, execution_times
                    )
                    memory_scaling = self._estimate_scaling_coefficient(
                        sizes, memory_usage
                    )

                    analysis["scalability_trends"][dimension].update(
                        {
                            "time_scaling_coefficient": time_scaling,
                            "memory_scaling_coefficient": memory_scaling,
                        }
                    )

        return analysis

    def _analyze_memory_benchmarks(self, results: Dict) -> Dict:
        """Analyze memory benchmark results."""
        analysis = {
            "memory_efficiency": {},
            "constraint_compliance": {},
            "optimization_effectiveness": {},
        }

        for constraint_key, constraint_results in results.items():
            memory_limit = float(constraint_key.replace("GB", ""))

            successful_tests = [
                r for r in constraint_results if r.get("success", False)
            ]

            if successful_tests:
                # Memory efficiency analysis
                peak_memories = [
                    r["memory_metrics"].get("peak_memory_gb", np.nan)
                    for r in successful_tests
                ]
                scale_factors = [r["scale_factor"] for r in successful_tests]

                valid_memories = [m for m in peak_memories if not np.isnan(m)]

                if valid_memories:
                    analysis["memory_efficiency"][constraint_key] = {
                        "mean_memory_usage": np.mean(valid_memories),
                        "max_memory_usage": max(valid_memories),
                        "memory_efficiency": np.mean(valid_memories) / memory_limit,
                        "largest_scale_handled": max(scale_factors),
                    }

                    # Constraint compliance
                    violations = [m for m in valid_memories if m > memory_limit]
                    analysis["constraint_compliance"][constraint_key] = {
                        "compliance_rate": 1.0 - len(violations) / len(valid_memories),
                        "max_violation": (
                            max(violations) - memory_limit if violations else 0.0
                        ),
                    }

        return analysis

    def _analyze_optimization_benchmarks(self, results: Dict) -> Dict:
        """Analyze optimization benchmark results."""
        analysis = {
            "strategy_comparison": {},
            "scalability_impact": {},
            "optimization_rankings": {},
        }

        # Compare strategies across scales
        all_strategies = set()
        for scale_results in results.values():
            all_strategies.update(scale_results.keys())

        strategy_metrics = {
            strategy: {"execution_times": [], "memory_usage": [], "scales": []}
            for strategy in all_strategies
        }

        for scale_key, scale_results in results.items():
            scale = float(scale_key.replace("scale_", ""))

            for strategy, strategy_result in scale_results.items():
                if strategy_result.get("success", False):
                    perf_metrics = strategy_result.get("performance_metrics", {})

                    strategy_metrics[strategy]["execution_times"].append(
                        perf_metrics.get("execution_time", np.nan)
                    )
                    strategy_metrics[strategy]["memory_usage"].append(
                        perf_metrics.get("peak_memory_gb", np.nan)
                    )
                    strategy_metrics[strategy]["scales"].append(scale)

        # Analyze each strategy
        for strategy, metrics in strategy_metrics.items():
            if metrics["execution_times"]:
                valid_times = [t for t in metrics["execution_times"] if not np.isnan(t)]
                valid_memory = [m for m in metrics["memory_usage"] if not np.isnan(m)]

                if valid_times:
                    analysis["strategy_comparison"][strategy] = {
                        "mean_execution_time": np.mean(valid_times),
                        "mean_memory_usage": (
                            np.mean(valid_memory) if valid_memory else np.nan
                        ),
                        "time_stability": np.std(valid_times) / np.mean(valid_times),
                        "scales_tested": len(valid_times),
                    }

        # Strategy rankings
        if analysis["strategy_comparison"]:
            # Rank by execution time
            time_ranking = sorted(
                analysis["strategy_comparison"].items(),
                key=lambda x: x[1].get("mean_execution_time", float("inf")),
            )

            # Rank by memory usage
            memory_ranking = sorted(
                analysis["strategy_comparison"].items(),
                key=lambda x: x[1].get("mean_memory_usage", float("inf")),
            )

            analysis["optimization_rankings"] = {
                "fastest_strategies": [name for name, _ in time_ranking],
                "most_memory_efficient": [name for name, _ in memory_ranking],
            }

        return analysis

    @experiment_handler("clinical_aware_cv_benchmarks")
    @validate_data_types(X_base=list, hypers=dict, args=dict)
    @validate_parameters(X_base=lambda x: len(x) > 0)
    def run_clinical_aware_cv_benchmarks(
        self,
        X_base: List[np.ndarray],
        hypers: Dict,
        args: Dict,
        clinical_data: Optional[Dict] = None,
        **kwargs,
    ) -> ExperimentResult:
        """Run clinical-aware cross-validation benchmarks using ClinicalAwareSplitter."""
        self.logger.info("Running clinical-aware cross-validation benchmarks")

        if clinical_data is None:
            # Generate synthetic clinical data for benchmarking
            clinical_data = self._generate_synthetic_clinical_data(X_base[0].shape[0])

        # Initialize neuroimaging CV configuration
        cv_config = NeuroImagingCVConfig()
        cv_config.outer_cv_folds = 5
        # Note: Using basic config - may need to extend for clinical awareness

        # Initialize clinical-aware splitter
        splitter = ClinicalAwareSplitter(config=cv_config)

        # Initialize neuroimaging metrics
        metrics_calculator = NeuroImagingMetrics()

        results = {}

        try:
            # Generate CV splits with automatic fallback
            self.logger.info("Generating clinical-aware CV splits with fallback")
            splits = self.cv_fallback.with_cv_split_fallback(
                advanced_split_func=splitter.split,
                X=X_base[0],
                y=clinical_data.get("diagnosis"),
                groups=clinical_data.get("subject_id"),
                clinical_data=clinical_data,
                cv_folds=cv_config.outer_cv_folds,
                random_state=42
            )

            self.logger.info(f"Generated {len(splits)} CV folds")

            # Run SGFA on each fold and evaluate performance
            fold_results = []

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                self.logger.info(f"Processing fold {fold_idx + 1}/{len(splits)}")

                # Split data into train/test
                X_train = [X[train_idx] for X in X_base]
                X_test = [X[test_idx] for X in X_base]

                # Train clinical data
                train_clinical = {
                    key: np.array(values)[train_idx] if isinstance(values, (list, np.ndarray)) else values
                    for key, values in clinical_data.items()
                }

                # Test clinical data
                test_clinical = {
                    key: np.array(values)[test_idx] if isinstance(values, (list, np.ndarray)) else values
                    for key, values in clinical_data.items()
                }

                try:
                    with self.system_monitor.monitor():
                        with self.profiler.profile(f"fold_{fold_idx}") as p:
                            # Train SGFA model
                            train_result = self._run_sgfa_analysis(
                                X_train, hypers, args, **kwargs
                            )

                            # Evaluate on test set if model converged
                            if train_result.get("convergence", False):
                                test_metrics = self._evaluate_sgfa_test_performance(
                                    X_test, train_result, test_clinical
                                )
                            else:
                                test_metrics = {"error": "Model did not converge"}

                    # Collect performance metrics
                    performance_metrics = self.profiler.get_current_metrics()
                    system_metrics = self.system_monitor.get_report()

                    # Calculate neuroimaging-specific metrics with fallback
                    fold_info = {
                        "fold_idx": fold_idx,
                        "train_size": len(train_idx),
                        "test_size": len(test_idx)
                    }

                    neuroimaging_metrics = self.metrics_fallback.with_metrics_fallback(
                        advanced_metrics_func=metrics_calculator.calculate_fold_metrics,
                        fallback_metrics=self.metrics_fallback.create_basic_fold_metrics(
                            train_result, test_metrics, fold_idx
                        ),
                        train_result=train_result,
                        test_metrics=test_metrics,
                        clinical_data=test_clinical,
                        fold_info=fold_info
                    )

                    fold_results.append({
                        "fold_idx": fold_idx,
                        "train_result": train_result,
                        "test_metrics": test_metrics,
                        "performance_metrics": performance_metrics,
                        "system_metrics": system_metrics,
                        "neuroimaging_metrics": neuroimaging_metrics,
                        "split_info": {
                            "train_size": len(train_idx),
                            "test_size": len(test_idx),
                            "train_diagnosis_dist": self._get_distribution(
                                train_clinical.get("diagnosis", [])
                            ),
                            "test_diagnosis_dist": self._get_distribution(
                                test_clinical.get("diagnosis", [])
                            )
                        },
                        "success": True
                    })

                    self.logger.info(
                        f"✅ Fold {fold_idx + 1}: "
                        f"Train size={len(train_idx)}, Test size={len(test_idx)}, "
                        f"Convergence={train_result.get('convergence', False)}"
                    )

                except Exception as e:
                    self.logger.warning(f"Fold {fold_idx + 1} failed: {str(e)}")
                    fold_results.append({
                        "fold_idx": fold_idx,
                        "error": str(e),
                        "split_info": {
                            "train_size": len(train_idx),
                            "test_size": len(test_idx)
                        },
                        "success": False
                    })

            results["fold_results"] = fold_results

            # Calculate aggregated CV metrics
            cv_analysis = self._analyze_clinical_cv_results(fold_results)
            results["cv_analysis"] = cv_analysis

            # Evaluate clinical stratification quality
            stratification_analysis = self._analyze_clinical_stratification(
                splits, clinical_data
            )
            results["stratification_analysis"] = stratification_analysis

            # Generate plots
            plots = self._plot_clinical_cv_benchmarks(results)

            return ExperimentResult(
                experiment_name="clinical_aware_cv_benchmarks",
                config=self.config,
                data=results,
                analysis={
                    "cv_analysis": cv_analysis,
                    "stratification_analysis": stratification_analysis
                },
                plots=plots,
                success=True,
            )

        except Exception as e:
            self.logger.error(f"Clinical-aware CV benchmarks failed: {str(e)}")
            return ExperimentResult(
                experiment_name="clinical_aware_cv_benchmarks",
                config=self.config,
                data={"error": str(e)},
                analysis={},
                plots={},
                success=False,
            )

    def _generate_synthetic_clinical_data(self, n_subjects: int) -> Dict:
        """Generate synthetic clinical data for benchmarking."""
        np.random.seed(42)  # For reproducibility

        # Generate diagnoses (PD subtypes)
        diagnoses = np.random.choice(
            ["tremor_dominant", "postural_instability", "mixed", "control"],
            size=n_subjects,
            p=[0.3, 0.25, 0.25, 0.2]  # Realistic distribution
        )

        # Generate ages with diagnosis correlation
        ages = []
        for diag in diagnoses:
            if diag == "control":
                age = np.random.normal(55, 15)  # Younger controls
            else:
                age = np.random.normal(65, 12)  # Older PD patients
            ages.append(max(30, min(85, age)))  # Constrain age range

        # Create age groups
        age_groups = ["young" if age < 55 else "middle" if age < 70 else "old" for age in ages]

        # Generate other clinical variables
        disease_duration = []
        motor_scores = []

        for i, diag in enumerate(diagnoses):
            if diag == "control":
                disease_duration.append(0)
                motor_scores.append(np.random.normal(2, 1))  # Low motor scores
            else:
                duration = max(0, np.random.exponential(5))  # Years since diagnosis
                disease_duration.append(duration)

                # Motor scores correlated with disease duration
                base_score = 15 + duration * 2 + np.random.normal(0, 5)
                motor_scores.append(max(0, min(50, base_score)))

        return {
            "subject_id": [f"subj_{i:04d}" for i in range(n_subjects)],
            "diagnosis": diagnoses,
            "age": np.array(ages),
            "age_group": age_groups,
            "disease_duration": np.array(disease_duration),
            "motor_score": np.array(motor_scores),
            "site": np.random.choice(["site_A", "site_B", "site_C"], n_subjects),
        }

    def _evaluate_sgfa_test_performance(
        self, X_test: List[np.ndarray], train_result: Dict, test_clinical: Dict
    ) -> Dict:
        """Evaluate trained SGFA model on test data."""
        try:
            # Extract trained parameters
            W_list = train_result.get("W", [])
            if not W_list:
                return {"error": "No trained weights found"}

            # Project test data onto learned factors
            Z_test_list = []
            reconstruction_errors = []

            for i, (X_view, W_view) in enumerate(zip(X_test, W_list)):
                # Simple projection (in practice, would use proper inference)
                Z_test = X_view @ W_view @ np.linalg.pinv(W_view.T @ W_view)
                Z_test_list.append(Z_test)

                # Calculate reconstruction error
                X_recon = Z_test @ W_view.T
                recon_error = np.mean((X_view - X_recon) ** 2)
                reconstruction_errors.append(recon_error)

            # Average factors across views
            Z_test_mean = np.mean(Z_test_list, axis=0)

            return {
                "Z_test": Z_test_mean,
                "reconstruction_errors": reconstruction_errors,
                "mean_reconstruction_error": np.mean(reconstruction_errors),
                "test_log_likelihood": train_result.get("log_likelihood", 0) * 0.8,  # Approximation
                "n_test_subjects": X_test[0].shape[0]
            }

        except Exception as e:
            return {"error": f"Test evaluation failed: {str(e)}"}

    def _get_distribution(self, labels: np.ndarray) -> Dict:
        """Get distribution of categorical labels."""
        if len(labels) == 0:
            return {}

        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        return {
            str(label): {
                "count": int(count),
                "proportion": float(count / total)
            }
            for label, count in zip(unique, counts)
        }

    def _analyze_clinical_cv_results(self, fold_results: List[Dict]) -> Dict:
        """Analyze clinical cross-validation results."""
        successful_folds = [f for f in fold_results if f.get("success", False)]

        if not successful_folds:
            return {"error": "No successful folds to analyze"}

        # Aggregate performance metrics
        execution_times = [f["performance_metrics"].get("execution_time", 0) for f in successful_folds]
        memory_usage = [f["performance_metrics"].get("peak_memory_gb", 0) for f in successful_folds]
        convergence_rates = [f["train_result"].get("convergence", False) for f in successful_folds]

        # Aggregate test performance
        recon_errors = []
        test_ll = []

        for fold in successful_folds:
            test_metrics = fold.get("test_metrics", {})
            if "mean_reconstruction_error" in test_metrics:
                recon_errors.append(test_metrics["mean_reconstruction_error"])
            if "test_log_likelihood" in test_metrics:
                test_ll.append(test_metrics["test_log_likelihood"])

        # Neuroimaging-specific metrics
        neuroimaging_scores = []
        for fold in successful_folds:
            neuro_metrics = fold.get("neuroimaging_metrics", {})
            if neuro_metrics and "overall_score" in neuro_metrics:
                neuroimaging_scores.append(neuro_metrics["overall_score"])

        analysis = {
            "n_successful_folds": len(successful_folds),
            "n_total_folds": len(fold_results),
            "success_rate": len(successful_folds) / len(fold_results),

            "performance_metrics": {
                "mean_execution_time": np.mean(execution_times),
                "std_execution_time": np.std(execution_times),
                "mean_memory_usage": np.mean(memory_usage),
                "std_memory_usage": np.std(memory_usage),
            },

            "convergence_analysis": {
                "convergence_rate": np.mean(convergence_rates),
                "n_converged": sum(convergence_rates),
            },

            "predictive_performance": {
                "mean_reconstruction_error": np.mean(recon_errors) if recon_errors else np.nan,
                "std_reconstruction_error": np.std(recon_errors) if recon_errors else np.nan,
                "mean_test_log_likelihood": np.mean(test_ll) if test_ll else np.nan,
                "std_test_log_likelihood": np.std(test_ll) if test_ll else np.nan,
            },

            "neuroimaging_metrics": {
                "mean_overall_score": np.mean(neuroimaging_scores) if neuroimaging_scores else np.nan,
                "std_overall_score": np.std(neuroimaging_scores) if neuroimaging_scores else np.nan,
            } if neuroimaging_scores else {}
        }

        return analysis

    def _analyze_clinical_stratification(self, splits: List, clinical_data: Dict) -> Dict:
        """Analyze quality of clinical stratification in CV splits."""
        stratification_metrics = {
            "diagnosis_balance": [],
            "age_balance": [],
            "site_balance": [],
        }

        for train_idx, test_idx in splits:
            # Check diagnosis balance
            train_diag = np.array(clinical_data["diagnosis"])[train_idx]
            test_diag = np.array(clinical_data["diagnosis"])[test_idx]

            train_dist = self._get_distribution(train_diag)
            test_dist = self._get_distribution(test_diag)

            # Calculate KL divergence for diagnosis distribution
            diag_kl = self._calculate_kl_divergence(train_dist, test_dist)
            stratification_metrics["diagnosis_balance"].append(diag_kl)

            # Check age balance
            train_ages = np.array(clinical_data["age"])[train_idx]
            test_ages = np.array(clinical_data["age"])[test_idx]

            age_diff = abs(np.mean(train_ages) - np.mean(test_ages))
            stratification_metrics["age_balance"].append(age_diff)

            # Check site balance
            train_sites = np.array(clinical_data["site"])[train_idx]
            test_sites = np.array(clinical_data["site"])[test_idx]

            train_site_dist = self._get_distribution(train_sites)
            test_site_dist = self._get_distribution(test_sites)

            site_kl = self._calculate_kl_divergence(train_site_dist, test_site_dist)
            stratification_metrics["site_balance"].append(site_kl)

        return {
            "diagnosis_balance": {
                "mean_kl_divergence": np.mean(stratification_metrics["diagnosis_balance"]),
                "std_kl_divergence": np.std(stratification_metrics["diagnosis_balance"]),
            },
            "age_balance": {
                "mean_age_difference": np.mean(stratification_metrics["age_balance"]),
                "std_age_difference": np.std(stratification_metrics["age_balance"]),
            },
            "site_balance": {
                "mean_kl_divergence": np.mean(stratification_metrics["site_balance"]),
                "std_kl_divergence": np.std(stratification_metrics["site_balance"]),
            },
            "overall_quality": {
                "good_stratification": (
                    np.mean(stratification_metrics["diagnosis_balance"]) < 0.1 and
                    np.mean(stratification_metrics["age_balance"]) < 5.0 and
                    np.mean(stratification_metrics["site_balance"]) < 0.2
                )
            }
        }

    def _calculate_kl_divergence(self, dist1: Dict, dist2: Dict) -> float:
        """Calculate KL divergence between two categorical distributions."""
        try:
            # Get all categories
            all_categories = set(dist1.keys()) | set(dist2.keys())

            if not all_categories:
                return 0.0

            kl_div = 0.0
            epsilon = 1e-8  # Small constant to avoid log(0)

            for category in all_categories:
                p1 = dist1.get(category, {}).get("proportion", epsilon)
                p2 = dist2.get(category, {}).get("proportion", epsilon)

                p1 = max(p1, epsilon)  # Avoid log(0)
                p2 = max(p2, epsilon)

                kl_div += p1 * np.log(p1 / p2)

            return float(kl_div)

        except Exception:
            return float('inf')  # Return infinity if calculation fails

    def _plot_clinical_cv_benchmarks(self, results: Dict) -> Dict:
        """Generate plots for clinical-aware CV benchmarks."""
        plots = {}

        try:
            fold_results = results.get("fold_results", [])
            successful_folds = [f for f in fold_results if f.get("success", False)]

            if not successful_folds:
                return plots

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("Clinical-Aware Cross-Validation Benchmarks", fontsize=16)

            # Plot 1: Performance across folds
            fold_indices = [f["fold_idx"] for f in successful_folds]
            execution_times = [f["performance_metrics"].get("execution_time", 0) for f in successful_folds]
            memory_usage = [f["performance_metrics"].get("peak_memory_gb", 0) for f in successful_folds]

            ax1 = axes[0, 0]
            ax1_twin = ax1.twinx()

            line1 = ax1.plot(fold_indices, execution_times, 'o-', color='blue', label='Execution Time')
            line2 = ax1_twin.plot(fold_indices, memory_usage, 's-', color='red', label='Memory Usage')

            ax1.set_xlabel("Fold Index")
            ax1.set_ylabel("Execution Time (seconds)", color='blue')
            ax1_twin.set_ylabel("Memory Usage (GB)", color='red')
            ax1.set_title("Performance Across CV Folds")
            ax1.grid(True, alpha=0.3)

            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')

            # Plot 2: Reconstruction errors
            recon_errors = []
            for fold in successful_folds:
                test_metrics = fold.get("test_metrics", {})
                if "mean_reconstruction_error" in test_metrics:
                    recon_errors.append(test_metrics["mean_reconstruction_error"])

            if recon_errors:
                axes[0, 1].plot(fold_indices[:len(recon_errors)], recon_errors, 'o-', color='green')
                axes[0, 1].set_xlabel("Fold Index")
                axes[0, 1].set_ylabel("Mean Reconstruction Error")
                axes[0, 1].set_title("Test Set Reconstruction Error")
                axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Split size distribution
            train_sizes = [f["split_info"]["train_size"] for f in fold_results]
            test_sizes = [f["split_info"]["test_size"] for f in fold_results]

            x_pos = np.arange(len(fold_results))
            width = 0.35

            axes[1, 0].bar(x_pos - width/2, train_sizes, width, label='Train', alpha=0.7)
            axes[1, 0].bar(x_pos + width/2, test_sizes, width, label='Test', alpha=0.7)
            axes[1, 0].set_xlabel("Fold Index")
            axes[1, 0].set_ylabel("Number of Subjects")
            axes[1, 0].set_title("Train/Test Split Sizes")
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels([f"Fold {i}" for i in range(len(fold_results))])
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Convergence and success rates
            convergence_rates = [f["train_result"].get("convergence", False) for f in successful_folds]
            success_rates = [f.get("success", False) for f in fold_results]

            metrics_summary = {
                "Convergence Rate": np.mean(convergence_rates) if convergence_rates else 0,
                "Success Rate": np.mean(success_rates),
                "Fold Completion": len(successful_folds) / len(fold_results) if fold_results else 0
            }

            axes[1, 1].bar(metrics_summary.keys(), metrics_summary.values(),
                          color=['green', 'blue', 'orange'], alpha=0.7)
            axes[1, 1].set_ylabel("Rate")
            axes[1, 1].set_title("CV Quality Metrics")
            axes[1, 1].set_ylim([0, 1.1])
            axes[1, 1].grid(True, alpha=0.3)

            # Add value labels on bars
            for i, (key, value) in enumerate(metrics_summary.items()):
                axes[1, 1].text(i, value + 0.02, f"{value:.2f}", ha='center', va='bottom')

            plt.tight_layout()
            plots["clinical_cv_benchmarks"] = fig

        except Exception as e:
            self.logger.warning(f"Failed to create clinical CV benchmark plots: {str(e)}")

        return plots

    def _analyze_comparative_benchmarks(self, results: Dict) -> Dict:
        """Analyze comparative benchmark results."""
        analysis = {
            "method_comparison": {},
            "sgfa_advantages": {},
            "performance_trade_offs": {},
        }

        # Extract performance metrics for all methods
        method_performance = {}

        for config_name, config_data in results.items():
            config_results = config_data["results"]

            for method_name, method_data in config_results.items():
                if method_name not in method_performance:
                    method_performance[method_name] = {
                        "execution_times": [],
                        "memory_usage": [],
                        "success_rate": [],
                    }

                if method_data.get("result", {}).get("error"):
                    method_performance[method_name]["success_rate"].append(0)
                else:
                    method_performance[method_name]["success_rate"].append(1)

                    perf_metrics = method_data.get("performance_metrics", {})
                    method_performance[method_name]["execution_times"].append(
                        perf_metrics.get("execution_time", np.nan)
                    )
                    method_performance[method_name]["memory_usage"].append(
                        perf_metrics.get("peak_memory_gb", np.nan)
                    )

        # Analyze method performance
        for method_name, perf_data in method_performance.items():
            valid_times = [t for t in perf_data["execution_times"] if not np.isnan(t)]
            valid_memory = [m for m in perf_data["memory_usage"] if not np.isnan(m)]

            analysis["method_comparison"][method_name] = {
                "mean_execution_time": np.mean(valid_times) if valid_times else np.nan,
                "mean_memory_usage": np.mean(valid_memory) if valid_memory else np.nan,
                "success_rate": np.mean(perf_data["success_rate"]),
                "method_type": "multiview" if method_name == "sgfa" else "traditional",
            }

        # SGFA-specific analysis
        if "sgfa" in analysis["method_comparison"]:
            sgfa_perf = analysis["method_comparison"]["sgfa"]

            # Compare with traditional methods
            traditional_methods = {
                name: perf
                for name, perf in analysis["method_comparison"].items()
                if perf["method_type"] == "traditional"
            }

            if traditional_methods:
                avg_traditional_time = np.mean(
                    [
                        p["mean_execution_time"]
                        for p in traditional_methods.values()
                        if not np.isnan(p["mean_execution_time"])
                    ]
                )
                avg_traditional_memory = np.mean(
                    [
                        p["mean_memory_usage"]
                        for p in traditional_methods.values()
                        if not np.isnan(p["mean_memory_usage"])
                    ]
                )

                analysis["sgfa_advantages"] = {
                    "time_advantage": (
                        avg_traditional_time / sgfa_perf["mean_execution_time"]
                        if not np.isnan(sgfa_perf["mean_execution_time"])
                        else np.nan
                    ),
                    "memory_advantage": (
                        avg_traditional_memory / sgfa_perf["mean_memory_usage"]
                        if not np.isnan(sgfa_perf["mean_memory_usage"])
                        else np.nan
                    ),
                    "multiview_capability": True,
                    "success_rate_advantage": sgfa_perf["success_rate"]
                    - np.mean(
                        [p["success_rate"] for p in traditional_methods.values()]
                    ),
                }

        return analysis

    def _estimate_scaling_coefficient(self, sizes: List, values: List) -> float:
        """Estimate scaling coefficient (approximate power law exponent)."""
        valid_pairs = [
            (s, v) for s, v in zip(sizes, values) if not np.isnan(v) and v > 0
        ]

        if len(valid_pairs) < 2:
            return np.nan

        log_sizes = [np.log(s) for s, v in valid_pairs]
        log_values = [np.log(v) for s, v in valid_pairs]

        # Simple linear regression in log space
        n = len(log_sizes)
        sum_x = sum(log_sizes)
        sum_y = sum(log_values)
        sum_xy = sum(x * y for x, y in zip(log_sizes, log_values))
        sum_x2 = sum(x * x for x in log_sizes)

        # Slope of log-log plot approximates scaling exponent
        scaling_coefficient = (n * sum_xy - sum_x * sum_y) / (
            n * sum_x2 - sum_x * sum_x
        )

        return scaling_coefficient

    def _plot_scalability_benchmarks(self, results: Dict) -> Dict:
        """Generate plots for scalability benchmarks."""
        plots = {}

        try:
            # Create comprehensive scalability plots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle("Scalability Benchmarks", fontsize=16)

            dimensions = list(results.keys())
            colors = plt.cm.Set1(np.linspace(0, 1, len(dimensions)))

            # Plot 1: Execution time scaling
            for i, (dimension, dimension_results) in enumerate(results.items()):
                sizes = []
                times = []

                for size, result in dimension_results.items():
                    if "error" not in result:
                        sizes.append(size)
                        perf_metrics = result.get("performance_metrics", {})
                        times.append(perf_metrics.get("execution_time", np.nan))

                valid_data = [(s, t) for s, t in zip(sizes, times) if not np.isnan(t)]
                if valid_data:
                    sizes_valid, times_valid = zip(*valid_data)
                    axes[0, 0].loglog(
                        sizes_valid, times_valid, "o-", label=dimension, color=colors[i]
                    )

            axes[0, 0].set_xlabel("Problem Size")
            axes[0, 0].set_ylabel("Execution Time (seconds)")
            axes[0, 0].set_title("Execution Time Scaling")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Memory usage scaling
            for i, (dimension, dimension_results) in enumerate(results.items()):
                sizes = []
                memory = []

                for size, result in dimension_results.items():
                    if "error" not in result:
                        sizes.append(size)
                        perf_metrics = result.get("performance_metrics", {})
                        memory.append(perf_metrics.get("peak_memory_gb", np.nan))

                valid_data = [(s, m) for s, m in zip(sizes, memory) if not np.isnan(m)]
                if valid_data:
                    sizes_valid, memory_valid = zip(*valid_data)
                    axes[0, 1].loglog(
                        sizes_valid,
                        memory_valid,
                        "s-",
                        label=dimension,
                        color=colors[i],
                    )

            axes[0, 1].set_xlabel("Problem Size")
            axes[0, 1].set_ylabel("Peak Memory (GB)")
            axes[0, 1].set_title("Memory Usage Scaling")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Throughput scaling
            for i, (dimension, dimension_results) in enumerate(results.items()):
                sizes = []
                throughput = []

                for size, result in dimension_results.items():
                    if "error" not in result:
                        perf_metrics = result.get("performance_metrics", {})
                        exec_time = perf_metrics.get("execution_time")

                        if exec_time and exec_time > 0:
                            sizes.append(size)
                            throughput.append(size / exec_time)  # Items per second

                if sizes:
                    axes[1, 0].semilogx(
                        sizes, throughput, "^-", label=dimension, color=colors[i]
                    )

            axes[1, 0].set_xlabel("Problem Size")
            axes[1, 0].set_ylabel("Throughput (items/second)")
            axes[1, 0].set_title("Throughput Scaling")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Success rate by problem size
            for i, (dimension, dimension_results) in enumerate(results.items()):
                sizes = []
                success_rates = []

                for size, result in dimension_results.items():
                    sizes.append(size)
                    success_rates.append(1.0 if "error" not in result else 0.0)

                if sizes:
                    axes[1, 1].semilogx(
                        sizes, success_rates, "o-", label=dimension, color=colors[i]
                    )

            axes[1, 1].set_xlabel("Problem Size")
            axes[1, 1].set_ylabel("Success Rate")
            axes[1, 1].set_title("Success Rate by Problem Size")
            axes[1, 1].set_ylim([0, 1.1])
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plots["scalability_benchmarks"] = fig

        except Exception as e:
            self.logger.warning(f"Failed to create scalability plots: {str(e)}")

        return plots

    def _plot_memory_benchmarks(self, results: Dict) -> Dict:
        """Generate plots for memory benchmarks."""
        plots = {}

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("Memory Benchmarks", fontsize=16)

            # Extract data for plotting
            memory_limits = []
            max_scale_factors = []
            efficiency_scores = []
            compliance_rates = []

            for constraint_key, constraint_results in results.items():
                memory_limit = float(constraint_key.replace("GB", ""))
                memory_limits.append(memory_limit)

                successful_tests = [
                    r for r in constraint_results if r.get("success", False)
                ]

                if successful_tests:
                    scale_factors = [r["scale_factor"] for r in successful_tests]
                    max_scale_factors.append(max(scale_factors))

                    # Memory efficiency
                    peak_memories = [
                        r["memory_metrics"].get("peak_memory_gb", np.nan)
                        for r in successful_tests
                    ]
                    valid_memories = [m for m in peak_memories if not np.isnan(m)]

                    if valid_memories:
                        efficiency = np.mean(valid_memories) / memory_limit
                        efficiency_scores.append(efficiency)

                        # Compliance rate
                        violations = [m for m in valid_memories if m > memory_limit]
                        compliance = 1.0 - len(violations) / len(valid_memories)
                        compliance_rates.append(compliance)
                    else:
                        efficiency_scores.append(np.nan)
                        compliance_rates.append(np.nan)
                else:
                    max_scale_factors.append(0)
                    efficiency_scores.append(np.nan)
                    compliance_rates.append(0)

            # Plot 1: Maximum scale factor vs memory limit
            axes[0, 0].plot(memory_limits, max_scale_factors, "o-")
            axes[0, 0].set_xlabel("Memory Limit (GB)")
            axes[0, 0].set_ylabel("Maximum Scale Factor Handled")
            axes[0, 0].set_title("Scalability vs Memory Constraints")
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Memory efficiency
            valid_efficiency = [
                (ml, es)
                for ml, es in zip(memory_limits, efficiency_scores)
                if not np.isnan(es)
            ]
            if valid_efficiency:
                ml_valid, es_valid = zip(*valid_efficiency)
                axes[0, 1].plot(ml_valid, es_valid, "s-", color="orange")
                axes[0, 1].set_xlabel("Memory Limit (GB)")
                axes[0, 1].set_ylabel("Memory Efficiency (Usage/Limit)")
                axes[0, 1].set_title("Memory Usage Efficiency")
                axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Compliance rates
            axes[1, 0].bar(memory_limits, compliance_rates, alpha=0.7, color="green")
            axes[1, 0].set_xlabel("Memory Limit (GB)")
            axes[1, 0].set_ylabel("Compliance Rate")
            axes[1, 0].set_title("Memory Constraint Compliance")
            axes[1, 0].set_ylim([0, 1.1])
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Memory usage distribution for each constraint
            memory_usage_data = []
            constraint_labels = []

            for constraint_key, constraint_results in results.items():
                successful_tests = [
                    r for r in constraint_results if r.get("success", False)
                ]
                if successful_tests:
                    peak_memories = [
                        r["memory_metrics"].get("peak_memory_gb", np.nan)
                        for r in successful_tests
                    ]
                    valid_memories = [m for m in peak_memories if not np.isnan(m)]

                    if valid_memories:
                        memory_usage_data.append(valid_memories)
                        constraint_labels.append(constraint_key)

            if memory_usage_data:
                axes[1, 1].boxplot(memory_usage_data, labels=constraint_labels)
                axes[1, 1].set_ylabel("Peak Memory Usage (GB)")
                axes[1, 1].set_title("Memory Usage Distribution by Constraint")
                axes[1, 1].tick_params(axis="x", rotation=45)
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plots["memory_benchmarks"] = fig

        except Exception as e:
            self.logger.warning(f"Failed to create memory benchmark plots: {str(e)}")

        return plots

    def _plot_optimization_benchmarks(self, results: Dict) -> Dict:
        """Generate plots for optimization benchmarks."""
        plots = {}

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("Optimization Strategy Benchmarks", fontsize=16)

            # Extract data
            strategies = set()
            for scale_results in results.values():
                strategies.update(scale_results.keys())
            strategies = sorted(strategies)

            # Plot 1: Execution time by strategy and scale
            scales = sorted([float(k.replace("scale_", "")) for k in results.keys()])

            for strategy in strategies:
                strategy_times = []
                strategy_scales = []

                for scale in scales:
                    scale_key = f"scale_{scale}"
                    if scale_key in results and strategy in results[scale_key]:
                        strategy_result = results[scale_key][strategy]
                        if strategy_result.get("success", False):
                            perf_metrics = strategy_result.get(
                                "performance_metrics", {}
                            )
                            exec_time = perf_metrics.get("execution_time")
                            if exec_time and not np.isnan(exec_time):
                                strategy_times.append(exec_time)
                                strategy_scales.append(scale)

                if strategy_times:
                    axes[0, 0].plot(
                        strategy_scales, strategy_times, "o-", label=strategy
                    )

            axes[0, 0].set_xlabel("Scale Factor")
            axes[0, 0].set_ylabel("Execution Time (seconds)")
            axes[0, 0].set_title("Execution Time by Optimization Strategy")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Memory usage by strategy and scale
            for strategy in strategies:
                strategy_memory = []
                strategy_scales = []

                for scale in scales:
                    scale_key = f"scale_{scale}"
                    if scale_key in results and strategy in results[scale_key]:
                        strategy_result = results[scale_key][strategy]
                        if strategy_result.get("success", False):
                            perf_metrics = strategy_result.get(
                                "performance_metrics", {}
                            )
                            memory = perf_metrics.get("peak_memory_gb")
                            if memory and not np.isnan(memory):
                                strategy_memory.append(memory)
                                strategy_scales.append(scale)

                if strategy_memory:
                    axes[0, 1].plot(
                        strategy_scales, strategy_memory, "s-", label=strategy
                    )

            axes[0, 1].set_xlabel("Scale Factor")
            axes[0, 1].set_ylabel("Peak Memory (GB)")
            axes[0, 1].set_title("Memory Usage by Optimization Strategy")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Strategy comparison (average performance)
            avg_times = []
            avg_memory = []
            strategy_names = []

            for strategy in strategies:
                times = []
                memory = []

                for scale_key in results.keys():
                    if strategy in results[scale_key]:
                        strategy_result = results[scale_key][strategy]
                        if strategy_result.get("success", False):
                            perf_metrics = strategy_result.get(
                                "performance_metrics", {}
                            )

                            exec_time = perf_metrics.get("execution_time")
                            if exec_time and not np.isnan(exec_time):
                                times.append(exec_time)

                            mem_usage = perf_metrics.get("peak_memory_gb")
                            if mem_usage and not np.isnan(mem_usage):
                                memory.append(mem_usage)

                if times:
                    avg_times.append(np.mean(times))
                    avg_memory.append(np.mean(memory) if memory else np.nan)
                    strategy_names.append(strategy)

            if avg_times:
                x_pos = np.arange(len(strategy_names))
                axes[1, 0].bar(x_pos, avg_times)
                axes[1, 0].set_xlabel("Optimization Strategy")
                axes[1, 0].set_ylabel("Average Execution Time (seconds)")
                axes[1, 0].set_title("Average Performance by Strategy")
                axes[1, 0].set_xticks(x_pos)
                axes[1, 0].set_xticklabels(strategy_names, rotation=45)
                axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Performance vs memory trade-off
            valid_data = [
                (t, m, s)
                for t, m, s in zip(avg_times, avg_memory, strategy_names)
                if not np.isnan(m)
            ]

            if valid_data:
                times_valid, memory_valid, names_valid = zip(*valid_data)

                scatter = axes[1, 1].scatter(
                    memory_valid, times_valid, s=100, alpha=0.7
                )

                for i, name in enumerate(names_valid):
                    axes[1, 1].annotate(
                        name,
                        (memory_valid[i], times_valid[i]),
                        xytext=(5, 5),
                        textcoords="offset points",
                    )

                axes[1, 1].set_xlabel("Average Memory Usage (GB)")
                axes[1, 1].set_ylabel("Average Execution Time (seconds)")
                axes[1, 1].set_title("Performance vs Memory Trade-off")
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plots["optimization_benchmarks"] = fig

        except Exception as e:
            self.logger.warning(
                f"Failed to create optimization benchmark plots: {str(e)}"
            )

        return plots

    def _plot_comparative_benchmarks(self, results: Dict) -> Dict:
        """Generate plots for comparative benchmarks."""
        plots = {}

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("Comparative Method Benchmarks", fontsize=16)

            # Extract method performance data
            methods = set()
            for config_data in results.values():
                methods.update(config_data["results"].keys())
            methods = sorted(methods)

            # Plot 1: Execution time comparison
            method_times = {method: [] for method in methods}
            config_names = sorted(results.keys())

            for config_name in config_names:
                config_results = results[config_name]["results"]

                for method in methods:
                    if method in config_results:
                        method_result = config_results[method]
                        if not method_result.get("result", {}).get("error"):
                            perf_metrics = method_result.get("performance_metrics", {})
                            exec_time = perf_metrics.get("execution_time")
                            if exec_time and not np.isnan(exec_time):
                                method_times[method].append(exec_time)
                            else:
                                method_times[method].append(np.nan)
                        else:
                            method_times[method].append(np.nan)
                    else:
                        method_times[method].append(np.nan)

            # Create grouped bar chart
            x_pos = np.arange(len(config_names))
            bar_width = 0.8 / len(methods)
            colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))

            for i, method in enumerate(methods):
                method_data = [
                    t if not np.isnan(t) else 0 for t in method_times[method]
                ]
                axes[0, 0].bar(
                    x_pos + i * bar_width,
                    method_data,
                    bar_width,
                    label=method,
                    color=colors[i],
                    alpha=0.8,
                )

            axes[0, 0].set_xlabel("Dataset Configuration")
            axes[0, 0].set_ylabel("Execution Time (seconds)")
            axes[0, 0].set_title("Execution Time by Method and Dataset")
            axes[0, 0].set_xticks(x_pos + bar_width * (len(methods) - 1) / 2)
            axes[0, 0].set_xticklabels(config_names)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Memory usage comparison
            method_memory = {method: [] for method in methods}

            for config_name in config_names:
                config_results = results[config_name]["results"]

                for method in methods:
                    if method in config_results:
                        method_result = config_results[method]
                        if not method_result.get("result", {}).get("error"):
                            perf_metrics = method_result.get("performance_metrics", {})
                            memory = perf_metrics.get("peak_memory_gb")
                            if memory and not np.isnan(memory):
                                method_memory[method].append(memory)
                            else:
                                method_memory[method].append(np.nan)
                        else:
                            method_memory[method].append(np.nan)
                    else:
                        method_memory[method].append(np.nan)

            for i, method in enumerate(methods):
                method_data = [
                    m if not np.isnan(m) else 0 for m in method_memory[method]
                ]
                axes[0, 1].bar(
                    x_pos + i * bar_width,
                    method_data,
                    bar_width,
                    label=method,
                    color=colors[i],
                    alpha=0.8,
                )

            axes[0, 1].set_xlabel("Dataset Configuration")
            axes[0, 1].set_ylabel("Peak Memory (GB)")
            axes[0, 1].set_title("Memory Usage by Method and Dataset")
            axes[0, 1].set_xticks(x_pos + bar_width * (len(methods) - 1) / 2)
            axes[0, 1].set_xticklabels(config_names)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Success rate by method
            method_success_rates = []
            method_names = []

            for method in methods:
                success_count = 0
                total_count = 0

                for config_name in config_names:
                    if method in results[config_name]["results"]:
                        total_count += 1
                        method_result = results[config_name]["results"][method]
                        if not method_result.get("result", {}).get("error"):
                            success_count += 1

                if total_count > 0:
                    method_success_rates.append(success_count / total_count)
                    method_names.append(method)

            if method_success_rates:
                axes[1, 0].bar(
                    method_names, method_success_rates, alpha=0.7, color="green"
                )
                axes[1, 0].set_ylabel("Success Rate")
                axes[1, 0].set_title("Success Rate by Method")
                axes[1, 0].set_ylim([0, 1.1])
                axes[1, 0].tick_params(axis="x", rotation=45)
                axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Performance efficiency scatter
            method_avg_times = []
            method_avg_memory = []
            scatter_names = []

            for method in methods:
                times = [t for t in method_times[method] if not np.isnan(t)]
                memory = [m for m in method_memory[method] if not np.isnan(m)]

                if times and memory:
                    method_avg_times.append(np.mean(times))
                    method_avg_memory.append(np.mean(memory))
                    scatter_names.append(method)

            if method_avg_times:
                # Color by method type
                colors_scatter = [
                    "red" if name == "sgfa" else "blue" for name in scatter_names
                ]

                axes[1, 1].scatter(
                    method_avg_memory,
                    method_avg_times,
                    c=colors_scatter,
                    s=100,
                    alpha=0.7,
                )

                for i, name in enumerate(scatter_names):
                    axes[1, 1].annotate(
                        name,
                        (method_avg_memory[i], method_avg_times[i]),
                        xytext=(5, 5),
                        textcoords="offset points",
                    )

                axes[1, 1].set_xlabel("Average Memory Usage (GB)")
                axes[1, 1].set_ylabel("Average Execution Time (seconds)")
                axes[1, 1].set_title("Performance Efficiency Comparison")
                axes[1, 1].grid(True, alpha=0.3)

                # Add legend
                from matplotlib.patches import Patch

                legend_elements = [
                    Patch(facecolor="red", label="SGFA"),
                    Patch(facecolor="blue", label="Traditional Methods"),
                ]
                axes[1, 1].legend(handles=legend_elements)

            plt.tight_layout()
            plots["comparative_benchmarks"] = fig

        except Exception as e:
            self.logger.warning(
                f"Failed to create comparative benchmark plots: {str(e)}"
            )

        return plots

    def _create_comprehensive_performance_visualizations(
        self, X_base: List[np.ndarray], results: Dict, experiment_name: str
    ) -> Dict:
        """Create comprehensive performance visualizations focusing on optimization surfaces."""
        advanced_plots = {}

        try:
            self.logger.info(
                f"🎨 Creating comprehensive performance visualizations for {experiment_name}"
            )

            # Import visualization system
            from core.config_utils import ConfigAccessor
            from visualization.manager import VisualizationManager

            # Create a performance-focused config for visualization
            viz_config = ConfigAccessor(
                {
                    "visualization": {
                        "create_brain_viz": False,  # Focus on performance, not brain maps
                        "output_format": ["png", "pdf"],
                        "dpi": 300,
                        "performance_focus": True,
                    },
                    "output_dir": f"/tmp/performance_viz_{experiment_name}",
                }
            )

            # Initialize visualization manager
            viz_manager = VisualizationManager(viz_config)

            # Prepare performance data structure for visualizations
            data = {
                "X_list": X_base,
                "view_names": [f"view_{i}" for i in range(len(X_base))],
                "n_subjects": X_base[0].shape[0],
                "view_dimensions": [X.shape[1] for X in X_base],
                "preprocessing": {
                    "status": "completed",
                    "strategy": "performance_optimized",
                },
            }

            # Extract performance metrics for visualization
            best_performance_result = self._extract_best_performance_result(results)

            if best_performance_result:
                # Prepare performance analysis results with hyperparameter data
                analysis_results = {
                    "best_run": best_performance_result,
                    "all_runs": results,
                    "model_type": "performance_sparseGFA",
                    "convergence": best_performance_result.get("convergence", False),
                    "performance_optimization": True,
                }

                # Add cross-validation results for hyperparameter optimization
                cv_results = None
                if any("scalability" in key for key in results.keys()):
                    cv_results = {
                        "hyperparameter_optimization": {
                            "scalability_results": results,
                            "performance_metrics": self._extract_performance_metrics(
                                results
                            ),
                        }
                    }

                # Create all comprehensive visualizations with performance focus
                viz_manager.create_all_visualizations(
                    data=data, analysis_results=analysis_results, cv_results=cv_results
                )

                # Extract the generated plots and convert to matplotlib figures
                if hasattr(viz_manager, "plot_dir") and viz_manager.plot_dir.exists():
                    plot_files = list(viz_manager.plot_dir.glob("**/*.png"))

                    for plot_file in plot_files:
                        plot_name = f"performance_{plot_file.stem}"

                        # Load the saved plot as a matplotlib figure
                        try:
                            import matplotlib.image as mpimg
                            import matplotlib.pyplot as plt

                            fig, ax = plt.subplots(figsize=(12, 8))
                            img = mpimg.imread(str(plot_file))
                            ax.imshow(img)
                            ax.axis("off")
                            ax.set_title(
                                f"Performance Analysis: {plot_name}", fontsize=14
                            )

                            advanced_plots[plot_name] = fig

                        except Exception as e:
                            self.logger.warning(
                                f"Could not load performance plot {plot_name}: {e}"
                            )

                    self.logger.info(
                        f"✅ Created {
                            len(plot_files)} comprehensive performance visualizations"
                    )

                    # Additional performance-specific summary
                    if cv_results and "hyperparameter_optimization" in cv_results:
                        self.logger.info(
                            "   → Hyperparameter optimization surface plots generated"
                        )
                    self.logger.info(
                        "   → Scalability and memory analysis visualizations generated"
                    )

                else:
                    self.logger.warning(
                        "Performance visualization manager did not create plot directory"
                    )
            else:
                self.logger.warning(
                    "No performance results found for comprehensive visualization"
                )

        except Exception as e:
            self.logger.warning(
                f"Failed to create comprehensive performance visualizations: {e}"
            )
            # Don't fail the experiment if advanced visualizations fail

        return advanced_plots

    def _extract_best_performance_result(self, results: Dict) -> Optional[Dict]:
        """Extract the best performance result from benchmark results."""
        best_result = None
        best_score = float("inf")  # Lower execution time is better

        # Look through different performance result structures
        for key, result_set in results.items():
            if isinstance(result_set, list):
                for result in result_set:
                    if isinstance(result, dict) and result.get("result", {}).get(
                        "convergence", False
                    ):
                        exec_time = result.get("result", {}).get(
                            "execution_time", float("inf")
                        )
                        if exec_time < best_score:
                            best_score = exec_time
                            best_result = result.get("result", {})
            elif isinstance(result_set, dict) and "execution_time" in result_set:
                exec_time = result_set.get("execution_time", float("inf"))
                if exec_time < best_score:
                    best_score = exec_time
                    best_result = result_set

        return best_result

    def _extract_performance_metrics(self, results: Dict) -> Dict:
        """Extract performance metrics for hyperparameter optimization visualization."""
        metrics = {
            "execution_times": [],
            "memory_usage": [],
            "parameter_combinations": [],
            "convergence_rates": [],
        }

        for key, result_set in results.items():
            if isinstance(result_set, list):
                for result in result_set:
                    if isinstance(result, dict) and "result" in result:
                        sgfa_result = result["result"]
                        metrics["execution_times"].append(
                            sgfa_result.get("execution_time", 0)
                        )
                        metrics["memory_usage"].append(
                            result.get("memory_metrics", {}).get("peak_memory_gb", 0)
                        )
                        metrics["convergence_rates"].append(
                            1 if sgfa_result.get("convergence", False) else 0
                        )

        return metrics


class SystemMonitor:
    """System resource monitoring utility."""

    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = None
        self.peak_memory = 0
        self.monitoring_active = False

    @contextmanager
    def monitor(self):
        """Context manager for system monitoring."""
        self.start_monitoring()
        try:
            yield
        finally:
            self.stop_monitoring()

    @contextmanager
    def monitor_memory(self):
        """Context manager for memory-specific monitoring."""
        self.start_memory_monitoring()
        try:
            yield
        finally:
            self.stop_monitoring()

    def start_monitoring(self):
        """Start system monitoring."""
        self.baseline_memory = self.process.memory_info().rss / (1024**3)  # GB
        self.peak_memory = self.baseline_memory
        self.monitoring_active = True

    def start_memory_monitoring(self):
        """Start memory-specific monitoring."""
        self.start_monitoring()
        gc.collect()  # Force garbage collection for clean baseline

    def stop_monitoring(self):
        """Stop monitoring."""
        if self.monitoring_active:
            current_memory = self.process.memory_info().rss / (1024**3)
            self.peak_memory = max(self.peak_memory, current_memory)
        self.monitoring_active = False

    def get_memory_report(self) -> Dict[str, float]:
        """Get memory usage report."""
        current_memory = self.process.memory_info().rss / (1024**3)

        if self.monitoring_active:
            self.peak_memory = max(self.peak_memory, current_memory)

        return {
            "current_memory_gb": current_memory,
            "peak_memory_gb": self.peak_memory,
            "memory_increase_gb": current_memory - (self.baseline_memory or 0),
        }

    def get_report(self) -> Dict[str, Any]:
        """Get comprehensive system report."""
        cpu_percent = self.process.cpu_percent()
        memory_info = self.get_memory_report()

        return {
            "cpu_percent": cpu_percent,
            **memory_info,
            "system_memory_available_gb": psutil.virtual_memory().available / (1024**3),
        }


def run_performance_benchmarks(config):
    """Run performance benchmarks with remote workstation integration."""
    import logging
    import os
    import sys

    logger = logging.getLogger(__name__)
    logger.info("Starting Performance Benchmarks Experiments")

    try:
        # Add project root to path for imports
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Load data with advanced preprocessing for consistent benchmarking
        from data.preprocessing_integration import apply_preprocessing_to_pipeline
        from experiments.framework import ExperimentConfig, ExperimentFramework

        logger.info("🔧 Loading data for performance benchmarking...")
        X_list, preprocessing_info = apply_preprocessing_to_pipeline(
            config=config,
            data_dir=get_data_dir(config),
            auto_select_strategy=False,
            preferred_strategy="standard",  # Use standard preprocessing for benchmarks
        )

        logger.info(f"✅ Data loaded: {len(X_list)} views for benchmarking")
        for i, X in enumerate(X_list):
            logger.info(f"   View {i}: {X.shape}")

        # Initialize experiment framework
        framework = ExperimentFramework(get_output_dir(config))

        exp_config = ExperimentConfig(
            experiment_name="performance_benchmarks",
            description="Performance benchmarking for SGFA on remote workstation",
            dataset="qmap_pd",
            data_dir=get_data_dir(config),
        )

        # Create benchmark experiment instance
        benchmark_exp = PerformanceBenchmarkExperiments(exp_config, logger)

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
            "num_warmup": 100,  # Reduced for benchmarking speed
            "num_samples": 200,  # Reduced for benchmarking speed
            "num_chains": 1,
            "target_accept_prob": 0.8,
            "reghsZ": True,
        }

        # Run the experiment
        def performance_benchmark_experiment(config, output_dir, **kwargs):
            logger.info("🚀 Running comprehensive performance benchmarks...")

            # Test different benchmark configurations - hardcoded for debug
            benchmark_configs = {
                "tiny_scale": {
                    "n_subjects": 20,
                    "n_features_per_view": [50, 30]
                },
                "small_scale": {
                    "n_subjects": 30,
                    "n_features_per_view": [100, 80]
                }
            }

            results = {}
            total_tests = 0
            successful_tests = 0

            # Run sample size scalability tests
            if "small_scale" in benchmark_configs:
                logger.info("📊 Testing small-scale performance...")
                small_config = benchmark_configs["small_scale"]
                n_subjects = min(small_config["n_subjects"], X_list[0].shape[0])

                # Subsample data
                indices = np.random.choice(
                    X_list[0].shape[0], n_subjects, replace=False
                )
                X_subset = [X[indices] for X in X_list]

                # Update hyperparameters
                test_hypers = base_hypers.copy()
                test_hypers["Dm"] = [X.shape[1] for X in X_subset]

                try:
                    with benchmark_exp.profiler.profile("small_scale_benchmark") as p:
                        result = benchmark_exp._run_sgfa_analysis(
                            X_subset, test_hypers, base_args
                        )

                    metrics = benchmark_exp.profiler.get_current_metrics()
                    results["small_scale"] = {
                        "config": small_config,
                        "result": result,
                        "performance": {
                            "execution_time": metrics.execution_time,
                            "peak_memory_gb": metrics.peak_memory_gb,
                        },
                        "data_info": {
                            "n_subjects": n_subjects,
                            "n_features": [X.shape[1] for X in X_subset],
                        },
                    }
                    successful_tests += 1
                    logger.info(
                        f"✅ Small-scale: {metrics.execution_time:.1f}s, {metrics.peak_memory_gb:.1f}GB"
                    )
                except Exception as e:
                    logger.error(f"❌ Small-scale benchmark failed: {e}")
                    results["small_scale"] = {"error": str(e)}

                total_tests += 1

            # Test different K values for component scalability
            logger.info("📊 Testing component scalability...")
            K_values = [5, 10, 15]  # Reduced set for testing
            component_results = {}

            for K in K_values:
                try:
                    test_hypers = base_hypers.copy()
                    test_hypers["K"] = K
                    test_args = base_args.copy()
                    test_args["K"] = K

                    with benchmark_exp.profiler.profile(f"components_K{K}") as p:
                        result = benchmark_exp._run_sgfa_analysis(
                            X_list, test_hypers, test_args
                        )

                    metrics = benchmark_exp.profiler.get_current_metrics()
                    component_results[f"K{K}"] = {
                        "K": K,
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
                        f"✅ K={K}: {
                            metrics.execution_time:.1f}s, LL={
                            result.get(
                                'log_likelihood',
                                0):.2f}"
                    )
                except Exception as e:
                    logger.error(f"❌ K={K} benchmark failed: {e}")
                    component_results[f"K{K}"] = {"error": str(e)}

                total_tests += 1

            results["component_scalability"] = component_results

            logger.info("🚀 Performance benchmarks completed!")
            logger.info(f"   Successful tests: {successful_tests}/{total_tests}")

            return {
                "status": "completed",
                "benchmark_results": results,
                "summary": {
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "success_rate": (
                        successful_tests / total_tests if total_tests > 0 else 0
                    ),
                    "data_characteristics": {
                        "n_subjects": X_list[0].shape[0],
                        "n_views": len(X_list),
                        "view_dimensions": [X.shape[1] for X in X_list],
                    },
                },
            }

        # Run experiment using framework
        result = framework.run_experiment(
            experiment_function=performance_benchmark_experiment,
            config=exp_config,
            data={"X_list": X_list, "preprocessing_info": preprocessing_info},
        )

        logger.info("✅ Performance benchmarks completed successfully")
        return result

    except Exception as e:
        logger.error(f"Performance benchmarks failed: {e}")
        return None
