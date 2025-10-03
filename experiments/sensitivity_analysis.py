"""Sensitivity analysis experiments for SGFA hyperparameters."""

import logging
from itertools import product
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from core.config_utils import get_data_dir, get_output_dir
from core.experiment_utils import experiment_handler
from core.validation_utils import validate_data_types, validate_parameters
from experiments.framework import (
    ExperimentConfig,
    ExperimentFramework,
    ExperimentResult,
)
from optimization import PerformanceProfiler
from optimization.experiment_mixins import performance_optimized_experiment


@performance_optimized_experiment()
class SensitivityAnalysisExperiments(ExperimentFramework):
    """Comprehensive sensitivity analysis for SGFA hyperparameters."""

    def __init__(
        self, config: ExperimentConfig, logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, None, logger)
        self.profiler = PerformanceProfiler()

        # Default hyperparameter ranges for sensitivity analysis
        self.hyperparameter_ranges = {
            "alpha_w": [0.1, 0.5, 1.0, 2.0, 5.0],
            "alpha_z": [0.1, 0.5, 1.0, 2.0, 5.0],
            "tau_w": [0.01, 0.1, 1.0, 10.0],
            "tau_z": [0.01, 0.1, 1.0, 10.0],
            "gamma": [0.1, 0.5, 1.0, 2.0, 5.0],
            "K": [2, 3, 5, 8, 10, 15],
            "sparsity_level": [0.1, 0.3, 0.5, 0.7, 0.9],
        }

        # Core hyperparameters that are most critical
        self.core_hyperparameters = ["alpha_w", "alpha_z", "K", "sparsity_level"]

    @experiment_handler("univariate_sensitivity_analysis")
    @validate_data_types(X_list=list, base_hypers=dict, args=dict)
    @validate_parameters(X_list=lambda x: len(x) > 0)
    def run_univariate_sensitivity_analysis(
        self,
        X_list: List[np.ndarray],
        base_hypers: Dict,
        args: Dict,
        hyperparameters: List[str] = None,
        **kwargs,
    ) -> ExperimentResult:
        """Run univariate sensitivity analysis for individual hyperparameters."""
        if hyperparameters is None:
            hyperparameters = self.core_hyperparameters

        self.logger.info(
            f"Running univariate sensitivity analysis for: {hyperparameters}"
        )

        results = {}
        performance_metrics = {}

        for hyperparam in hyperparameters:
            if hyperparam not in self.hyperparameter_ranges:
                self.logger.warning(
                    f"No range defined for hyperparameter: {hyperparam}"
                )
                continue

            self.logger.info(f"Analyzing sensitivity for {hyperparam}")

            hyperparam_results = {}
            hyperparam_metrics = {}

            param_range = self.hyperparameter_ranges[hyperparam]

            for param_value in param_range:
                self.logger.debug(f"Testing {hyperparam}={param_value}")

                # Create hyperparameter configuration
                test_hypers = base_hypers.copy()
                test_hypers[hyperparam] = param_value

                # Run analysis
                with self.profiler.profile(f"{hyperparam}_{param_value}") as p:
                    try:
                        result = self._run_sgfa_analysis(
                            X_list, test_hypers, args, **kwargs
                        )
                        hyperparam_results[param_value] = result

                        # Store performance metrics
                        metrics = self.profiler.get_current_metrics()
                        hyperparam_metrics[param_value] = {
                            "execution_time": metrics.execution_time,
                            "peak_memory_gb": metrics.peak_memory_gb,
                            "convergence": result.get("convergence", False),
                            "log_likelihood": result.get("log_likelihood", np.nan),
                        }

                    except Exception as e:
                        self.logger.warning(
                            f"Failed for {hyperparam}={param_value}: {str(e)}"
                        )
                        hyperparam_results[param_value] = {"error": str(e)}
                        hyperparam_metrics[param_value] = {
                            "execution_time": np.nan,
                            "peak_memory_gb": np.nan,
                            "convergence": False,
                            "log_likelihood": np.nan,
                        }

            results[hyperparam] = hyperparam_results
            performance_metrics[hyperparam] = hyperparam_metrics

        # Analyze sensitivity
        analysis = self._analyze_univariate_sensitivity(results, performance_metrics)

        # Generate basic plots
        plots = self._plot_univariate_sensitivity(results, performance_metrics)

        # Add comprehensive sensitivity visualizations (focus on factor stability)
        advanced_plots = self._create_comprehensive_sensitivity_visualizations(
            X_list, results, "univariate_sensitivity"
        )
        plots.update(advanced_plots)

        return ExperimentResult(
            experiment_id="univariate_sensitivity_analysis",
            config=self.config,
            model_results=results,
            diagnostics=analysis,
            plots=plots,
            performance_metrics=performance_metrics,
            status="completed",
        )

    @experiment_handler("multivariate_sensitivity_analysis")
    @validate_data_types(X_list=list, base_hypers=dict, args=dict)
    @validate_parameters(X_list=lambda x: len(x) > 0)
    def run_multivariate_sensitivity_analysis(
        self,
        X_list: List[np.ndarray],
        base_hypers: Dict,
        args: Dict,
        hyperparameter_pairs: List[Tuple[str, str]] = None,
        **kwargs,
    ) -> ExperimentResult:
        """Run multivariate sensitivity analysis for hyperparameter interactions."""
        if hyperparameter_pairs is None:
            hyperparameter_pairs = [
                ("alpha_w", "alpha_z"),
                ("alpha_w", "K"),
                ("tau_w", "tau_z"),
                ("K", "sparsity_level"),
            ]

        self.logger.info(
            f"Running multivariate sensitivity analysis for: {hyperparameter_pairs}"
        )

        results = {}
        performance_metrics = {}

        for param1, param2 in hyperparameter_pairs:
            if (
                param1 not in self.hyperparameter_ranges
                or param2 not in self.hyperparameter_ranges
            ):
                self.logger.warning(
                    f"Missing range for parameter pair: ({param1}, {param2})"
                )
                continue

            self.logger.info(f"Analyzing interaction between {param1} and {param2}")

            pair_results = {}
            pair_metrics = {}

            # Use smaller ranges for multivariate analysis to keep computation tractable
            range1 = self._get_reduced_range(param1)
            range2 = self._get_reduced_range(param2)

            for val1, val2 in product(range1, range2):
                param_key = f"{param1}={val1}_{param2}={val2}"
                self.logger.debug(f"Testing {param_key}")

                # Create hyperparameter configuration
                test_hypers = base_hypers.copy()
                test_hypers[param1] = val1
                test_hypers[param2] = val2

                # Run analysis
                with self.profiler.profile(param_key) as p:
                    try:
                        result = self._run_sgfa_analysis(
                            X_list, test_hypers, args, **kwargs
                        )
                        pair_results[(val1, val2)] = result

                        # Store performance metrics
                        metrics = self.profiler.get_current_metrics()
                        pair_metrics[(val1, val2)] = {
                            "execution_time": metrics.execution_time,
                            "peak_memory_gb": metrics.peak_memory_gb,
                            "convergence": result.get("convergence", False),
                            "log_likelihood": result.get("log_likelihood", np.nan),
                        }

                    except Exception as e:
                        self.logger.warning(f"Failed for {param_key}: {str(e)}")
                        pair_results[(val1, val2)] = {"error": str(e)}
                        pair_metrics[(val1, val2)] = {
                            "execution_time": np.nan,
                            "peak_memory_gb": np.nan,
                            "convergence": False,
                            "log_likelihood": np.nan,
                        }

            results[f"{param1}_vs_{param2}"] = {
                "results": pair_results,
                "param1": param1,
                "param2": param2,
                "range1": range1,
                "range2": range2,
            }
            performance_metrics[f"{param1}_vs_{param2}"] = pair_metrics

        # Analyze multivariate sensitivity
        analysis = self._analyze_multivariate_sensitivity(results, performance_metrics)

        # Generate plots
        plots = self._plot_multivariate_sensitivity(results, performance_metrics)

        return ExperimentResult(
            experiment_id="multivariate_sensitivity_analysis",
            config=self.config,
            model_results=results,
            diagnostics=analysis,
            plots=plots,
            performance_metrics=performance_metrics,
            status="completed",
        )

    @experiment_handler("gradient_based_sensitivity")
    @validate_data_types(X_list=list, base_hypers=dict, args=dict)
    @validate_parameters(X_list=lambda x: len(x) > 0, epsilon=lambda x: x > 0)
    def run_gradient_based_sensitivity(
        self,
        X_list: List[np.ndarray],
        base_hypers: Dict,
        args: Dict,
        epsilon: float = 0.01,
        **kwargs,
    ) -> ExperimentResult:
        """Run gradient-based sensitivity analysis using finite differences."""
        self.logger.info("Running gradient-based sensitivity analysis")

        results = {}
        gradients = {}

        # Get baseline result
        baseline_result = self._run_sgfa_analysis(X_list, base_hypers, args, **kwargs)
        baseline_likelihood = baseline_result.get("log_likelihood", np.nan)

        if np.isnan(baseline_likelihood):
            raise ValueError("Baseline analysis failed to produce valid log likelihood")

        # Calculate gradients for each hyperparameter
        for param_name in self.core_hyperparameters:
            if param_name not in base_hypers:
                continue

            self.logger.info(f"Calculating gradient for {param_name}")

            base_value = base_hypers[param_name]

            # Forward difference
            forward_hypers = base_hypers.copy()
            forward_hypers[param_name] = base_value * (1 + epsilon)

            try:
                forward_result = self._run_sgfa_analysis(
                    X_list, forward_hypers, args, **kwargs
                )
                forward_likelihood = forward_result.get("log_likelihood", np.nan)

                # Backward difference
                backward_hypers = base_hypers.copy()
                backward_hypers[param_name] = base_value * (1 - epsilon)

                backward_result = self._run_sgfa_analysis(
                    X_list, backward_hypers, args, **kwargs
                )
                backward_likelihood = backward_result.get("log_likelihood", np.nan)

                # Calculate gradient
                if not (np.isnan(forward_likelihood) or np.isnan(backward_likelihood)):
                    gradient = (forward_likelihood - backward_likelihood) / (
                        2 * epsilon * base_value
                    )

                    gradients[param_name] = {
                        "gradient": gradient,
                        "forward_likelihood": forward_likelihood,
                        "backward_likelihood": backward_likelihood,
                        "baseline_likelihood": baseline_likelihood,
                        "relative_sensitivity": abs(
                            gradient * base_value / baseline_likelihood
                        ),
                    }
                else:
                    gradients[param_name] = {
                        "gradient": np.nan,
                        "error": "Failed to compute differences",
                    }

            except Exception as e:
                self.logger.warning(
                    f"Failed to compute gradient for {param_name}: {str(e)}"
                )
                gradients[param_name] = {"gradient": np.nan, "error": str(e)}

        results["baseline"] = baseline_result
        results["gradients"] = gradients

        # Analyze gradient-based sensitivity
        analysis = self._analyze_gradient_sensitivity(gradients)

        # Generate plots
        plots = self._plot_gradient_sensitivity(gradients)

        return ExperimentResult(
            experiment_id="gradient_based_sensitivity",
            config=self.config,
            model_results=results,
            diagnostics=analysis,
            plots=plots,
            status="completed",
        )

    @experiment_handler("robustness_analysis")
    @validate_data_types(X_list=list, base_hypers=dict, args=dict)
    @validate_parameters(X_list=lambda x: len(x) > 0, n_trials=lambda x: x > 0)
    def run_robustness_analysis(
        self,
        X_list: List[np.ndarray],
        base_hypers: Dict,
        args: Dict,
        noise_levels: List[float] = None,
        n_trials: int = 10,
        **kwargs,
    ) -> ExperimentResult:
        """Run robustness analysis with hyperparameter perturbations."""
        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]

        self.logger.info(
            f"Running robustness analysis with noise levels: {noise_levels}"
        )

        results = {}

        # Baseline result
        baseline_result = self._run_sgfa_analysis(X_list, base_hypers, args, **kwargs)
        results["baseline"] = baseline_result

        # Robustness testing
        for noise_level in noise_levels:
            self.logger.info(f"Testing robustness with noise level: {noise_level}")

            noise_results = []

            for trial in range(n_trials):
                # Add noise to hyperparameters
                noisy_hypers = self._add_hyperparameter_noise(base_hypers, noise_level)

                try:
                    result = self._run_sgfa_analysis(
                        X_list, noisy_hypers, args, **kwargs
                    )
                    noise_results.append(
                        {
                            "trial": trial,
                            "noisy_hypers": noisy_hypers,
                            "result": result,
                            "log_likelihood": result.get("log_likelihood", np.nan),
                            "convergence": result.get("convergence", False),
                        }
                    )

                except Exception as e:
                    noise_results.append(
                        {
                            "trial": trial,
                            "noisy_hypers": noisy_hypers,
                            "result": {"error": str(e)},
                            "log_likelihood": np.nan,
                            "convergence": False,
                        }
                    )

            results[f"noise_{noise_level}"] = noise_results

        # Analyze robustness
        analysis = self._analyze_robustness(results, baseline_result)

        # Generate plots
        plots = self._plot_robustness_analysis(results)

        return ExperimentResult(
            experiment_id="robustness_analysis",
            config=self.config,
            model_results=results,
            diagnostics=analysis,
            plots=plots,
            status="completed",
        )

    def _get_reduced_range(self, param_name: str, n_values: int = 3) -> List:
        """Get a reduced range for multivariate analysis."""
        full_range = self.hyperparameter_ranges[param_name]
        if len(full_range) <= n_values:
            return full_range

        # Select evenly spaced values
        indices = np.linspace(0, len(full_range) - 1, n_values, dtype=int)
        return [full_range[i] for i in indices]

    def _add_hyperparameter_noise(self, base_hypers: Dict, noise_level: float) -> Dict:
        """Add noise to hyperparameters for robustness testing."""
        noisy_hypers = {}

        for param_name, param_value in base_hypers.items():
            if isinstance(param_value, (int, float)):
                # Add multiplicative noise
                noise_factor = 1.0 + np.random.normal(0, noise_level)
                noisy_value = param_value * noise_factor

                # Ensure positive values
                noisy_value = max(noisy_value, 0.001)

                # For integer parameters, round appropriately
                if param_name == "K":
                    noisy_value = max(1, int(round(noisy_value)))

                noisy_hypers[param_name] = noisy_value
            else:
                noisy_hypers[param_name] = param_value

        return noisy_hypers

    def _run_sgfa_analysis(
        self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> Dict:
        """Run actual SGFA analysis for sensitivity testing."""
        import time

        import jax
        from numpyro.infer import MCMC, NUTS

        try:
            K = hypers.get("K", 5)
            self.logger.debug(
                f"Running SGFA sensitivity test: K={K}, n_subjects={ X_list[0].shape[0]}, n_features={ sum( X.shape[1] for X in X_list)}"
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
            model_type, model_instance, models_summary = integrate_models_with_pipeline(
                config={"model": {"type": "sparseGFA"}},
                X_list=X_list,
                data_characteristics=data_characteristics
            )

            self.logger.info(f"🏭 Sensitivity analysis using model: {model_type}")

            # Import the actual SGFA model function for execution
            from core.run_analysis import models

            # Setup MCMC configuration for sensitivity analysis (reduced for speed)
            num_warmup = args.get("num_warmup", 50)  # Reduced for sensitivity analysis
            num_samples = args.get(
                "num_samples", 100
            )  # Reduced for sensitivity analysis
            num_chains = args.get(
                "num_chains", 1
            )  # Single chain for sensitivity analysis

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
                "log_likelihood": float(log_likelihood),
                "n_iterations": num_samples,
                "convergence": True,
                "execution_time": elapsed,
                "sensitivity_info": {
                    "parameter_tested": hypers,
                    "mcmc_config": {
                        "num_warmup": num_warmup,
                        "num_samples": num_samples,
                        "num_chains": num_chains,
                    },
                },
            }

        except Exception as e:
            self.logger.error(f"SGFA sensitivity analysis failed: {str(e)}")
            return {
                "error": str(e),
                "convergence": False,
                "execution_time": float("inf"),
                "log_likelihood": float("-inf"),
            }

    def _analyze_univariate_sensitivity(
        self, results: Dict, performance_metrics: Dict
    ) -> Dict:
        """Analyze univariate sensitivity results."""
        analysis = {
            "sensitivity_ranking": {},
            "optimal_values": {},
            "stability_assessment": {},
            "performance_impact": {},
            "roi_specificity_trends": {},
        }

        sensitivity_scores = {}

        for param_name, param_results in results.items():
            if not param_results:
                continue

            param_metrics = performance_metrics[param_name]

            # Extract log likelihoods and ROI specificity
            likelihoods = []
            param_values = []
            roi_specificities = []

            for param_value, result in param_results.items():
                if "error" not in result:
                    likelihood = param_metrics[param_value]["log_likelihood"]
                    if not np.isnan(likelihood):
                        likelihoods.append(likelihood)
                        param_values.append(param_value)

                        # Calculate ROI specificity if W_list available
                        if "W" in result and isinstance(result["W"], list):
                            try:
                                from analysis.cross_validation_library import NeuroImagingMetrics
                                W_list = result["W"]
                                view_names = result.get("view_names", [f"View_{i}" for i in range(len(W_list))])

                                roi_spec = NeuroImagingMetrics.roi_specificity_score(
                                    W_list, view_names=view_names
                                )
                                roi_specificities.append({
                                    "param_value": param_value,
                                    "mean_specificity": roi_spec["mean_specificity"],
                                    "specificity_rate": roi_spec["specificity_rate"],
                                    "n_specific_factors": roi_spec["n_specific_factors"]
                                })
                            except Exception as e:
                                self.logger.debug(f"Could not calculate ROI specificity for {param_name}={param_value}: {e}")

            if len(likelihoods) > 1:
                # Calculate sensitivity as range of log likelihoods
                likelihood_range = max(likelihoods) - min(likelihoods)
                sensitivity_scores[param_name] = likelihood_range

                # Find optimal value
                best_idx = np.argmax(likelihoods)
                optimal_value = param_values[best_idx]

                analysis["optimal_values"][param_name] = {
                    "value": optimal_value,
                    "log_likelihood": likelihoods[best_idx],
                }

                # Assess stability (coefficient of variation)
                cv = np.std(likelihoods) / np.abs(np.mean(likelihoods))
                analysis["stability_assessment"][param_name] = {
                    "coefficient_of_variation": cv,
                    "stability_level": (
                        "high" if cv < 0.1 else "medium" if cv < 0.3 else "low"
                    ),
                }

                # Performance impact
                execution_times = [
                    param_metrics[pv]["execution_time"] for pv in param_values
                ]
                memory_usages = [
                    param_metrics[pv]["peak_memory_gb"] for pv in param_values
                ]

                analysis["performance_impact"][param_name] = {
                    "time_range": max(execution_times) - min(execution_times),
                    "memory_range": max(memory_usages) - min(memory_usages),
                }

                # Store ROI specificity trends if available
                if roi_specificities:
                    analysis["roi_specificity_trends"][param_name] = roi_specificities
                    self.logger.info(f"ROI Specificity trend for {param_name}:")
                    for spec in roi_specificities:
                        self.logger.info(
                            f"  {param_name}={spec['param_value']}: "
                            f"{spec['n_specific_factors']} specific factors, "
                            f"mean={spec['mean_specificity']:.2f}"
                        )

        # Rank parameters by sensitivity
        sorted_sensitivity = sorted(
            sensitivity_scores.items(), key=lambda x: x[1], reverse=True
        )
        analysis["sensitivity_ranking"] = {
            "most_sensitive": [name for name, score in sorted_sensitivity],
            "sensitivity_scores": sensitivity_scores,
        }

        return analysis

    def _analyze_multivariate_sensitivity(
        self, results: Dict, performance_metrics: Dict
    ) -> Dict:
        """Analyze multivariate sensitivity results."""
        analysis = {
            "interaction_effects": {},
            "optimal_combinations": {},
            "interaction_strength": {},
        }

        for pair_name, pair_data in results.items():
            param1 = pair_data["param1"]
            param2 = pair_data["param2"]
            pair_data["results"]
            pair_metrics = performance_metrics[pair_name]

            # Extract log likelihoods in matrix form
            range1 = pair_data["range1"]
            range2 = pair_data["range2"]

            likelihood_matrix = np.full((len(range1), len(range2)), np.nan)

            for i, val1 in enumerate(range1):
                for j, val2 in enumerate(range2):
                    if (val1, val2) in pair_metrics:
                        likelihood = pair_metrics[(val1, val2)]["log_likelihood"]
                        if not np.isnan(likelihood):
                            likelihood_matrix[i, j] = likelihood

            # Find optimal combination
            if not np.all(np.isnan(likelihood_matrix)):
                best_idx = np.unravel_index(
                    np.nanargmax(likelihood_matrix), likelihood_matrix.shape
                )
                optimal_val1 = range1[best_idx[0]]
                optimal_val2 = range2[best_idx[1]]
                optimal_likelihood = likelihood_matrix[best_idx]

                analysis["optimal_combinations"][pair_name] = {
                    param1: optimal_val1,
                    param2: optimal_val2,
                    "log_likelihood": optimal_likelihood,
                }

                # Assess interaction strength
                # Compare to sum of individual effects (approximation)
                main_effect_strength = np.nanstd(
                    np.nanmean(likelihood_matrix, axis=1)
                ) + np.nanstd(np.nanmean(likelihood_matrix, axis=0))
                total_variation = np.nanstd(likelihood_matrix.flatten())

                interaction_ratio = total_variation / (main_effect_strength + 1e-10)

                analysis["interaction_strength"][pair_name] = {
                    "interaction_ratio": interaction_ratio,
                    "interaction_level": (
                        "strong"
                        if interaction_ratio > 1.5
                        else "moderate" if interaction_ratio > 1.1 else "weak"
                    ),
                }

        return analysis

    def _analyze_gradient_sensitivity(self, gradients: Dict) -> Dict:
        """Analyze gradient-based sensitivity results."""
        analysis = {
            "gradient_magnitudes": {},
            "sensitivity_ranking": {},
            "relative_importance": {},
        }

        valid_gradients = {}

        for param_name, gradient_data in gradients.items():
            if "gradient" in gradient_data and not np.isnan(gradient_data["gradient"]):
                gradient = gradient_data["gradient"]
                relative_sensitivity = gradient_data.get(
                    "relative_sensitivity", abs(gradient)
                )

                valid_gradients[param_name] = gradient

                analysis["gradient_magnitudes"][param_name] = {
                    "gradient": gradient,
                    "absolute_gradient": abs(gradient),
                    "relative_sensitivity": relative_sensitivity,
                }

        if valid_gradients:
            # Rank by absolute gradient
            sorted_gradients = sorted(
                valid_gradients.items(), key=lambda x: abs(x[1]), reverse=True
            )
            analysis["sensitivity_ranking"]["most_sensitive"] = [
                name for name, grad in sorted_gradients
            ]

            # Relative importance
            total_abs_gradient = sum(abs(grad) for grad in valid_gradients.values())
            for param_name, gradient in valid_gradients.items():
                analysis["relative_importance"][param_name] = (
                    abs(gradient) / total_abs_gradient
                )

        return analysis

    def _analyze_robustness(self, results: Dict, baseline_result: Dict) -> Dict:
        """Analyze robustness results."""
        analysis = {
            "robustness_metrics": {},
            "convergence_rates": {},
            "likelihood_stability": {},
        }

        baseline_likelihood = baseline_result.get("log_likelihood", np.nan)

        for noise_key, noise_results in results.items():
            if noise_key == "baseline":
                continue

            noise_level = float(noise_key.split("_")[1])

            # Extract metrics
            likelihoods = [
                r["log_likelihood"]
                for r in noise_results
                if not np.isnan(r["log_likelihood"])
            ]
            convergence_count = sum(1 for r in noise_results if r["convergence"])

            if likelihoods:
                likelihood_mean = np.mean(likelihoods)
                likelihood_std = np.std(likelihoods)

                # Robustness metrics
                analysis["robustness_metrics"][noise_level] = {
                    "mean_likelihood": likelihood_mean,
                    "std_likelihood": likelihood_std,
                    "coefficient_of_variation": likelihood_std / abs(likelihood_mean),
                    "likelihood_drop": (
                        baseline_likelihood - likelihood_mean
                        if not np.isnan(baseline_likelihood)
                        else np.nan
                    ),
                }

                # Likelihood stability
                analysis["likelihood_stability"][noise_level] = {
                    "stability_score": 1.0 / (1.0 + likelihood_std),
                    "relative_stability": (
                        likelihood_std / abs(baseline_likelihood)
                        if not np.isnan(baseline_likelihood)
                        else np.nan
                    ),
                }

            # Convergence rates
            convergence_rate = convergence_count / len(noise_results)
            analysis["convergence_rates"][noise_level] = convergence_rate

        return analysis

    def _plot_univariate_sensitivity(
        self, results: Dict, performance_metrics: Dict
    ) -> Dict:
        """Generate plots for univariate sensitivity analysis."""
        self.logger.info("📊 Generating univariate sensitivity analysis plots...")
        plots = {}

        try:
            n_params = len(results)
            if n_params == 0:
                return plots

            self.logger.info(f"   Creating 4-panel sensitivity plot for {n_params} parameters...")
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("Univariate Sensitivity Analysis", fontsize=16)

            # Flatten axes for easier indexing
            axes_flat = axes.flatten()

            param_names = list(results.keys())
            colors = plt.cm.Set1(np.linspace(0, 1, len(param_names)))

            # Plot 1: Log likelihood vs parameter values
            ax1 = axes_flat[0]
            for i, param_name in enumerate(param_names):
                param_results = results[param_name]
                param_metrics = performance_metrics[param_name]

                param_values = []
                likelihoods = []

                for param_value, result in param_results.items():
                    if "error" not in result:
                        likelihood = param_metrics[param_value]["log_likelihood"]
                        if not np.isnan(likelihood):
                            param_values.append(param_value)
                            likelihoods.append(likelihood)

                if param_values:
                    ax1.plot(
                        param_values,
                        likelihoods,
                        "o-",
                        label=param_name,
                        color=colors[i],
                    )

            ax1.set_xlabel("Parameter Value")
            ax1.set_ylabel("Log Likelihood")
            ax1.set_title("Log Likelihood vs Parameter Values")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Execution time vs parameter values
            ax2 = axes_flat[1]
            for i, param_name in enumerate(param_names):
                param_metrics = performance_metrics[param_name]

                param_values = []
                times = []

                for param_value, metrics in param_metrics.items():
                    if not np.isnan(metrics["execution_time"]):
                        param_values.append(param_value)
                        times.append(metrics["execution_time"])

                if param_values:
                    ax2.plot(
                        param_values, times, "s-", label=param_name, color=colors[i]
                    )

            ax2.set_xlabel("Parameter Value")
            ax2.set_ylabel("Execution Time (seconds)")
            ax2.set_title("Execution Time vs Parameter Values")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Sensitivity ranking (bar plot)
            ax3 = axes_flat[2]
            sensitivity_scores = {}

            for param_name, param_results in results.items():
                param_metrics = performance_metrics[param_name]
                likelihoods = [
                    param_metrics[pv]["log_likelihood"]
                    for pv in param_results.keys()
                    if "error" not in param_results[pv]
                    and not np.isnan(param_metrics[pv]["log_likelihood"])
                ]

                if len(likelihoods) > 1:
                    sensitivity_scores[param_name] = max(likelihoods) - min(likelihoods)

            if sensitivity_scores:
                param_names_sorted = sorted(
                    sensitivity_scores.keys(),
                    key=lambda x: sensitivity_scores[x],
                    reverse=True,
                )
                scores_sorted = [
                    sensitivity_scores[name] for name in param_names_sorted
                ]

                ax3.bar(param_names_sorted, scores_sorted)
                ax3.set_ylabel("Likelihood Range")
                ax3.set_title("Parameter Sensitivity Ranking")
                ax3.tick_params(axis="x", rotation=45)

            # Plot 4: Memory usage vs parameter values
            ax4 = axes_flat[3]
            for i, param_name in enumerate(param_names):
                param_metrics = performance_metrics[param_name]

                param_values = []
                memory = []

                for param_value, metrics in param_metrics.items():
                    if not np.isnan(metrics["peak_memory_gb"]):
                        param_values.append(param_value)
                        memory.append(metrics["peak_memory_gb"])

                if param_values:
                    ax4.plot(
                        param_values, memory, "^-", label=param_name, color=colors[i]
                    )

            ax4.set_xlabel("Parameter Value")
            ax4.set_ylabel("Peak Memory (GB)")
            ax4.set_title("Memory Usage vs Parameter Values")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plots["univariate_sensitivity"] = fig
            self.logger.info("   ✅ Univariate sensitivity plot created")

            # Create ROI specificity trend plot if data available
            self.logger.info("   Creating ROI specificity trend plot...")
            roi_spec_trends = {}
            for param_name, param_results in results.items():
                param_specs = []
                param_vals = []
                for param_value, result in param_results.items():
                    if "W" in result and isinstance(result["W"], list):
                        try:
                            from analysis.cross_validation_library import NeuroImagingMetrics
                            W_list = result["W"]
                            view_names = result.get("view_names", [f"View_{i}" for i in range(len(W_list))])
                            roi_spec = NeuroImagingMetrics.roi_specificity_score(W_list, view_names=view_names)
                            param_vals.append(param_value)
                            param_specs.append(roi_spec["mean_specificity"])
                        except:
                            pass
                if param_vals:
                    roi_spec_trends[param_name] = (param_vals, param_specs)

            if roi_spec_trends:
                fig_roi, ax_roi = plt.subplots(1, 1, figsize=(10, 6))
                for i, (param_name, (vals, specs)) in enumerate(roi_spec_trends.items()):
                    ax_roi.plot(vals, specs, 'o-', label=param_name, color=colors[i % len(colors)])

                ax_roi.set_xlabel("Parameter Value")
                ax_roi.set_ylabel("Mean ROI Specificity")
                ax_roi.set_title("ROI Specificity vs Parameter Values")
                ax_roi.legend()
                ax_roi.grid(True, alpha=0.3)
                ax_roi.set_ylim([0, 1])
                plt.tight_layout()
                plots["roi_specificity_trends"] = fig_roi
                self.logger.info(f"   ✅ ROI specificity trend plot created ({len(roi_spec_trends)} parameters)")

        except Exception as e:
            self.logger.warning(
                f"Failed to create univariate sensitivity plots: {str(e)}"
            )

        self.logger.info(f"📊 Univariate sensitivity plots completed: {len(plots)} plots generated")
        return plots

    def _plot_multivariate_sensitivity(
        self, results: Dict, performance_metrics: Dict
    ) -> Dict:
        """Generate plots for multivariate sensitivity analysis."""
        self.logger.info("📊 Generating multivariate sensitivity analysis plots...")
        plots = {}

        try:
            n_pairs = len(results)
            if n_pairs == 0:
                return plots

            self.logger.info(f"   Creating heatmaps for {n_pairs} parameter pairs...")
            # Create heatmaps for each parameter pair
            for pair_name, pair_data in results.items():
                param1 = pair_data["param1"]
                param2 = pair_data["param2"]
                range1 = pair_data["range1"]
                range2 = pair_data["range2"]
                pair_metrics = performance_metrics[pair_name]

                # Create likelihood matrix
                likelihood_matrix = np.full((len(range1), len(range2)), np.nan)

                for i, val1 in enumerate(range1):
                    for j, val2 in enumerate(range2):
                        if (val1, val2) in pair_metrics:
                            likelihood = pair_metrics[(val1, val2)]["log_likelihood"]
                            if not np.isnan(likelihood):
                                likelihood_matrix[i, j] = likelihood

                # Create heatmap
                fig, ax = plt.subplots(figsize=(8, 6))

                im = ax.imshow(likelihood_matrix, cmap="viridis", aspect="auto")

                # Set ticks and labels
                ax.set_xticks(range(len(range2)))
                ax.set_yticks(range(len(range1)))
                ax.set_xticklabels([f"{val:.3f}" for val in range2])
                ax.set_yticklabels([f"{val:.3f}" for val in range1])

                ax.set_xlabel(param2)
                ax.set_ylabel(param1)
                ax.set_title(f"Log Likelihood Heatmap: {param1} vs {param2}")

                # Add colorbar
                plt.colorbar(im, ax=ax, label="Log Likelihood")

                # Mark optimal point
                if not np.all(np.isnan(likelihood_matrix)):
                    best_idx = np.unravel_index(
                        np.nanargmax(likelihood_matrix), likelihood_matrix.shape
                    )
                    ax.plot(
                        best_idx[1], best_idx[0], "r*", markersize=15, label="Optimal"
                    )
                    ax.legend()

                plt.tight_layout()
                plots[f"heatmap_{pair_name}"] = fig

            self.logger.info(f"   ✅ Created {n_pairs} heatmaps")

        except Exception as e:
            self.logger.warning(
                f"Failed to create multivariate sensitivity plots: {str(e)}"
            )

        self.logger.info(f"📊 Multivariate sensitivity plots completed: {len(plots)} plots generated")
        return plots

    def _plot_gradient_sensitivity(self, gradients: Dict) -> Dict:
        """Generate plots for gradient-based sensitivity analysis."""
        self.logger.info("📊 Generating gradient sensitivity plots...")
        plots = {}

        try:
            # Filter valid gradients
            valid_gradients = {
                name: data
                for name, data in gradients.items()
                if "gradient" in data and not np.isnan(data["gradient"])
            }

            if not valid_gradients:
                return plots

            self.logger.info(f"   Creating gradient plot for {len(valid_gradients)} parameters...")
            # Create gradient plots
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            param_names = list(valid_gradients.keys())
            gradient_values = [
                valid_gradients[name]["gradient"] for name in param_names
            ]
            absolute_gradients = [abs(g) for g in gradient_values]

            # Plot 1: Gradient values
            axes[0].bar(param_names, gradient_values)
            axes[0].set_ylabel("Gradient (∂LL/∂θ)")
            axes[0].set_title("Gradient-Based Sensitivity")
            axes[0].tick_params(axis="x", rotation=45)
            axes[0].axhline(y=0, color="k", linestyle="--", alpha=0.5)

            # Plot 2: Absolute gradients (sensitivity ranking)
            sorted_indices = np.argsort(absolute_gradients)[::-1]
            sorted_names = [param_names[i] for i in sorted_indices]
            sorted_abs_gradients = [absolute_gradients[i] for i in sorted_indices]

            axes[1].bar(sorted_names, sorted_abs_gradients)
            axes[1].set_ylabel("|Gradient|")
            axes[1].set_title("Sensitivity Ranking")
            axes[1].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plots["gradient_sensitivity"] = fig
            self.logger.info("   ✅ Gradient sensitivity plot created")

        except Exception as e:
            self.logger.warning(
                f"Failed to create gradient sensitivity plots: {str(e)}"
            )

        self.logger.info(f"📊 Gradient sensitivity plots completed: {len(plots)} plots generated")
        return plots

    def _plot_robustness_analysis(self, results: Dict) -> Dict:
        """Generate plots for robustness analysis."""
        self.logger.info("📊 Generating robustness analysis plots...")
        plots = {}

        try:
            # Extract noise levels and metrics
            noise_levels = []
            mean_likelihoods = []
            std_likelihoods = []
            convergence_rates = []

            for noise_key, noise_results in results.items():
                if noise_key == "baseline":
                    continue

                noise_level = float(noise_key.split("_")[1])
                noise_levels.append(noise_level)

                likelihoods = [
                    r["log_likelihood"]
                    for r in noise_results
                    if not np.isnan(r["log_likelihood"])
                ]
                convergences = [r["convergence"] for r in noise_results]

                mean_likelihoods.append(np.mean(likelihoods) if likelihoods else np.nan)
                std_likelihoods.append(np.std(likelihoods) if likelihoods else np.nan)
                convergence_rates.append(np.mean(convergences))

            if not noise_levels:
                return plots

            self.logger.info(f"   Creating 4-panel robustness plot for {len(noise_levels)} noise levels...")
            # Create robustness plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Robustness Analysis", fontsize=16)

            # Plot 1: Mean likelihood vs noise level
            axes[0, 0].plot(noise_levels, mean_likelihoods, "o-")
            axes[0, 0].set_xlabel("Noise Level")
            axes[0, 0].set_ylabel("Mean Log Likelihood")
            axes[0, 0].set_title("Likelihood vs Noise Level")
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Likelihood standard deviation vs noise level
            axes[0, 1].plot(noise_levels, std_likelihoods, "s-", color="orange")
            axes[0, 1].set_xlabel("Noise Level")
            axes[0, 1].set_ylabel("Likelihood Standard Deviation")
            axes[0, 1].set_title("Likelihood Variability vs Noise Level")
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Convergence rate vs noise level
            axes[1, 0].plot(noise_levels, convergence_rates, "^-", color="green")
            axes[1, 0].set_xlabel("Noise Level")
            axes[1, 0].set_ylabel("Convergence Rate")
            axes[1, 0].set_title("Convergence Rate vs Noise Level")
            axes[1, 0].set_ylim([0, 1.1])
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Likelihood distribution for highest noise level
            if noise_levels:
                highest_noise_key = f"noise_{max(noise_levels)}"
                highest_noise_results = results[highest_noise_key]
                highest_noise_likelihoods = [
                    r["log_likelihood"]
                    for r in highest_noise_results
                    if not np.isnan(r["log_likelihood"])
                ]

                if highest_noise_likelihoods:
                    axes[1, 1].hist(
                        highest_noise_likelihoods, bins=10, alpha=0.7, color="red"
                    )
                    axes[1, 1].set_xlabel("Log Likelihood")
                    axes[1, 1].set_ylabel("Frequency")
                    axes[1, 1].set_title(
                        f"Likelihood Distribution (Noise={max(noise_levels)})"
                    )

                    # Add baseline line if available
                    baseline_likelihood = results["baseline"].get("log_likelihood")
                    if not np.isnan(baseline_likelihood):
                        axes[1, 1].axvline(
                            baseline_likelihood,
                            color="black",
                            linestyle="--",
                            label="Baseline",
                        )
                        axes[1, 1].legend()

            plt.tight_layout()
            plots["robustness_analysis"] = fig
            self.logger.info("   ✅ Robustness analysis plot created")

        except Exception as e:
            self.logger.warning(f"Failed to create robustness analysis plots: {str(e)}")

        self.logger.info(f"📊 Robustness analysis plots completed: {len(plots)} plots generated")
        return plots

    def _create_comprehensive_sensitivity_visualizations(
        self, X_list: List[np.ndarray], results: Dict, experiment_name: str
    ) -> Dict:
        """Create comprehensive sensitivity visualizations focusing on factor stability."""
        advanced_plots = {}

        try:
            self.logger.info(
                f"🎨 Creating comprehensive sensitivity visualizations for {experiment_name}"
            )

            # Import visualization system
            from core.config_utils import ConfigAccessor
            from visualization.manager import VisualizationManager

            # Create a sensitivity-focused config for visualization
            viz_config = ConfigAccessor(
                {
                    "visualization": {
                        "create_brain_viz": True,  # Include brain maps for sensitivity
                        "output_format": ["png", "pdf"],
                        "dpi": 300,
                        "sensitivity_focus": True,
                    },
                    "output_dir": f"/tmp/sensitivity_viz_{experiment_name}",
                }
            )

            # Initialize visualization manager
            viz_manager = VisualizationManager(viz_config)

            # Prepare sensitivity data structure
            data = {
                "X_list": X_list,
                "view_names": [f"view_{i}" for i in range(len(X_list))],
                "n_subjects": X_list[0].shape[0],
                "view_dimensions": [X.shape[1] for X in X_list],
                "preprocessing": {
                    "status": "completed",
                    "strategy": "sensitivity_analysis",
                },
            }

            # Extract best sensitivity result for analysis
            best_sensitivity_result = self._extract_best_sensitivity_result(results)

            if best_sensitivity_result:
                # Prepare sensitivity analysis results
                analysis_results = {
                    "best_run": best_sensitivity_result,
                    "all_runs": results,
                    "model_type": "sensitivity_sparseGFA",
                    "convergence": best_sensitivity_result.get("convergence", False),
                    "sensitivity_analysis": True,
                }

                # Add cross-validation style results for factor stability analysis
                cv_results = {
                    "factor_stability": {
                        "parameter_variations": results,
                        "stability_metrics": self._extract_stability_metrics(results),
                    }
                }

                # Create comprehensive visualizations with sensitivity focus
                viz_manager.create_all_visualizations(
                    model_results=data, analysis_results=analysis_results, cv_results=cv_results
                )

                # Extract and process generated plots
                if hasattr(viz_manager, "plot_dir") and viz_manager.plot_dir.exists():
                    plot_files = list(viz_manager.plot_dir.glob("**/*.png"))

                    for plot_file in plot_files:
                        plot_name = f"sensitivity_{plot_file.stem}"

                        try:
                            import matplotlib.image as mpimg
                            import matplotlib.pyplot as plt

                            fig, ax = plt.subplots(figsize=(12, 8))
                            img = mpimg.imread(str(plot_file))
                            ax.imshow(img)
                            ax.axis("off")
                            ax.set_title(
                                f"Sensitivity Analysis: {plot_name}", fontsize=14
                            )

                            advanced_plots[plot_name] = fig

                        except Exception as e:
                            self.logger.warning(
                                f"Could not load sensitivity plot {plot_name}: {e}"
                            )

                    self.logger.info(
                        f"✅ Created { len(plot_files)} comprehensive sensitivity visualizations"
                    )
                    self.logger.info(
                        "   → Factor stability and robustness plots generated"
                    )

                else:
                    self.logger.warning(
                        "Sensitivity visualization manager did not create plot directory"
                    )
            else:
                self.logger.warning(
                    "No sensitivity results found for comprehensive visualization"
                )

        except Exception as e:
            self.logger.warning(
                f"Failed to create comprehensive sensitivity visualizations: {e}"
            )

        return advanced_plots

    def _extract_best_sensitivity_result(self, results: Dict) -> Optional[Dict]:
        """Extract the best sensitivity result from analysis results."""
        best_result = None
        best_likelihood = float("-inf")

        # Look through sensitivity parameter results
        for param_name, param_results in results.items():
            if isinstance(param_results, dict):
                for param_value, result in param_results.items():
                    if isinstance(result, dict) and result.get("convergence", False):
                        likelihood = result.get("log_likelihood", float("-inf"))
                        if likelihood > best_likelihood:
                            best_likelihood = likelihood
                            best_result = result

        return best_result

    def _extract_stability_metrics(self, results: Dict) -> Dict:
        """Extract stability metrics for visualization."""
        metrics = {
            "parameter_sensitivity": {},
            "factor_stability": {},
            "convergence_rates": {},
        }

        for param_name, param_results in results.items():
            if isinstance(param_results, dict):
                param_metrics = []
                convergence_count = 0
                total_count = 0

                for param_value, result in param_results.items():
                    if isinstance(result, dict):
                        total_count += 1
                        if result.get("convergence", False):
                            convergence_count += 1
                            param_metrics.append(
                                {
                                    "value": param_value,
                                    "log_likelihood": result.get("log_likelihood", 0),
                                    "execution_time": result.get("execution_time", 0),
                                }
                            )

                metrics["parameter_sensitivity"][param_name] = param_metrics
                metrics["convergence_rates"][param_name] = convergence_count / max(
                    total_count, 1
                )

        return metrics


def run_sensitivity_analysis(config):
    """Run sensitivity analysis with remote workstation integration."""
    import logging
    import os
    import sys

    logger = logging.getLogger(__name__)
    logger.info("Starting Sensitivity Analysis Experiments")

    try:
        # Add project root to path for imports
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Load data with advanced preprocessing for consistent analysis
        from data.preprocessing_integration import apply_preprocessing_to_pipeline
        from experiments.framework import ExperimentConfig, ExperimentFramework

        logger.info("🔧 Loading data for sensitivity analysis...")
        # Get preprocessing strategy from config
        from core.config_utils import ConfigHelper
        config_dict = ConfigHelper.to_dict(config)
        preprocessing_config = config_dict.get("preprocessing", {})
        strategy = preprocessing_config.get("strategy", "standard")

        X_list, preprocessing_info = apply_preprocessing_to_pipeline(
            config=config,
            data_dir=get_data_dir(config),
            auto_select_strategy=False,
            preferred_strategy=strategy,  # Use strategy from config
        )

        logger.info(f"✅ Data loaded: {len(X_list)} views for sensitivity analysis")
        for i, X in enumerate(X_list):
            logger.info(f"   View {i}: {X.shape}")

        # Initialize experiment framework
        framework = ExperimentFramework(get_output_dir(config))

        exp_config = ExperimentConfig(
            experiment_name="sensitivity_analysis",
            description="Hyperparameter sensitivity analysis for SGFA",
            dataset="qmap_pd",
            data_dir=get_data_dir(config),
        )

        # Create sensitivity experiment instance
        sensitivity_exp = SensitivityAnalysisExperiments(exp_config, logger)

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
            "num_warmup": 50,  # Reduced for sensitivity analysis speed
            "num_samples": 100,  # Reduced for sensitivity analysis speed
            "num_chains": 1,
            "target_accept_prob": 0.8,
            "reghsZ": True,
        }

        # Run the experiment
        def sensitivity_analysis_experiment(config, output_dir, **kwargs):
            logger.info("🔬 Running comprehensive sensitivity analysis...")

            # Normalize config input using standard ConfigHelper
            from core.config_utils import ConfigHelper
            config_dict = ConfigHelper.to_dict(config)

            # Get sensitivity analysis configuration
            sensitivity_config = config_dict.get("sensitivity_analysis", {})
            parameter_ranges = sensitivity_config.get("parameter_ranges", {})

            results = {}
            total_tests = 0
            successful_tests = 0

            # Test K sensitivity (number of factors)
            logger.info("📊 Testing K (number of factors) sensitivity...")
            K_values = parameter_ranges.get("n_factors", [5, 10, 15, 20])[
                :3
            ]  # Limit for testing
            K_results = {}

            for K in K_values:
                try:
                    test_hypers = base_hypers.copy()
                    test_hypers["K"] = K
                    test_args = base_args.copy()
                    test_args["K"] = K

                    with sensitivity_exp.profiler.profile(f"K_sensitivity_{K}") as p:
                        result = sensitivity_exp._run_sgfa_analysis(
                            X_list, test_hypers, test_args
                        )

                    metrics = sensitivity_exp.profiler.get_current_metrics()
                    K_results[f"K{K}"] = {
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
                        f"✅ K={K}: { metrics.execution_time:.1f}s, LL={ result.get( 'log_likelihood', 0):.2f}"
                    )
                except Exception as e:
                    logger.error(f"❌ K={K} sensitivity test failed: {e}")
                    K_results[f"K{K}"] = {"error": str(e)}

                # Cleanup memory after each iteration
                import jax
                import gc
                jax.clear_caches()
                gc.collect()

                total_tests += 1

            results["K_sensitivity"] = K_results

            # Test sparsity sensitivity (percW)
            logger.info("📊 Testing sparsity (percW) sensitivity...")
            sparsity_values = [20, 33, 50]  # Different sparsity levels
            sparsity_results = {}

            for percW in sparsity_values:
                try:
                    test_hypers = base_hypers.copy()
                    test_hypers["percW"] = percW

                    with sensitivity_exp.profiler.profile(
                        f"percW_sensitivity_{percW}"
                    ) as p:
                        result = sensitivity_exp._run_sgfa_analysis(
                            X_list, test_hypers, base_args
                        )

                    metrics = sensitivity_exp.profiler.get_current_metrics()
                    sparsity_results[f"percW{percW}"] = {
                        "percW": percW,
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
                        f"✅ percW={percW}: { metrics.execution_time:.1f}s, LL={ result.get( 'log_likelihood', 0):.2f}"
                    )
                except Exception as e:
                    logger.error(f"❌ percW={percW} sensitivity test failed: {e}")
                    sparsity_results[f"percW{percW}"] = {"error": str(e)}

                # Cleanup memory after each iteration
                import jax
                import gc
                jax.clear_caches()
                gc.collect()

                total_tests += 1

            results["sparsity_sensitivity"] = sparsity_results

            # Test MCMC parameter sensitivity
            logger.info("📊 Testing MCMC parameter sensitivity...")
            mcmc_configs = [
                {"num_samples": 50, "num_warmup": 25, "label": "fast"},
                {"num_samples": 100, "num_warmup": 50, "label": "standard"},
                {"num_samples": 200, "num_warmup": 100, "label": "thorough"},
            ]
            mcmc_results = {}

            for mcmc_config in mcmc_configs[:2]:  # Test first 2 for speed
                try:
                    test_args = base_args.copy()
                    test_args["num_samples"] = mcmc_config["num_samples"]
                    test_args["num_warmup"] = mcmc_config["num_warmup"]
                    label = mcmc_config["label"]

                    with sensitivity_exp.profiler.profile(
                        f"mcmc_sensitivity_{label}"
                    ) as p:
                        result = sensitivity_exp._run_sgfa_analysis(
                            X_list, base_hypers, test_args
                        )

                    metrics = sensitivity_exp.profiler.get_current_metrics()
                    mcmc_results[label] = {
                        "config": mcmc_config,
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
                        f"✅ MCMC {label}: { metrics.execution_time:.1f}s, LL={ result.get( 'log_likelihood', 0):.2f}"
                    )
                except Exception as e:
                    logger.error(f"❌ MCMC {label} sensitivity test failed: {e}")
                    mcmc_results[label] = {"error": str(e)}

                # Cleanup memory after each iteration
                import jax
                import gc
                jax.clear_caches()
                gc.collect()

                total_tests += 1

            results["mcmc_sensitivity"] = mcmc_results

            logger.info("🔬 Sensitivity analysis completed!")
            logger.info(f"   Successful tests: {successful_tests}/{total_tests}")

            # Generate plots for sensitivity analysis
            import matplotlib.pyplot as plt
            plots = {}

            try:
                # Create parameter sensitivity plot
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle("SGFA Parameter Sensitivity Analysis", fontsize=16)

                # Plot 1: K sensitivity
                K_data = results.get("K_sensitivity", {})
                if K_data:
                    K_vals = [K_data[k]["K"] for k in sorted(K_data.keys())]
                    K_lls = [K_data[k]["performance"]["log_likelihood"] for k in sorted(K_data.keys())]
                    axes[0, 0].plot(K_vals, K_lls, 'o-', linewidth=2, markersize=8)
                    axes[0, 0].set_xlabel("Number of Factors (K)")
                    axes[0, 0].set_ylabel("Log Likelihood")
                    axes[0, 0].set_title("K (Number of Factors) Sensitivity")
                    axes[0, 0].grid(True, alpha=0.3)

                # Plot 2: percW sensitivity
                percW_data = results.get("sparsity_sensitivity", {})
                if percW_data:
                    percW_vals = [percW_data[p]["percW"] for p in sorted(percW_data.keys())]
                    percW_lls = [percW_data[p]["performance"]["log_likelihood"] for p in sorted(percW_data.keys())]
                    axes[0, 1].plot(percW_vals, percW_lls, 's-', linewidth=2, markersize=8)
                    axes[0, 1].set_xlabel("Sparsity (percW %)")
                    axes[0, 1].set_ylabel("Log Likelihood")
                    axes[0, 1].set_title("Sparsity (percW) Sensitivity")
                    axes[0, 1].grid(True, alpha=0.3)

                # Plot 3: MCMC config comparison
                mcmc_data = results.get("mcmc_sensitivity", {})
                if mcmc_data:
                    mcmc_labels = list(mcmc_data.keys())
                    mcmc_lls = [mcmc_data[m]["performance"]["log_likelihood"] for m in mcmc_labels]
                    axes[1, 0].bar(mcmc_labels, mcmc_lls)
                    axes[1, 0].set_xlabel("MCMC Configuration")
                    axes[1, 0].set_ylabel("Log Likelihood")
                    axes[1, 0].set_title("MCMC Parameter Sensitivity")
                    axes[1, 0].grid(True, alpha=0.3, axis='y')

                # Plot 4: Success summary
                params_tested = ["K", "percW", "MCMC"]
                success_counts = [
                    len([k for k in K_data.values() if not k.get("error")]),
                    len([p for p in percW_data.values() if not p.get("error")]),
                    len([m for m in mcmc_data.values() if not m.get("error")])
                ]
                axes[1, 1].bar(params_tested, success_counts)
                axes[1, 1].set_ylabel("Successful Tests")
                axes[1, 1].set_title(f"Test Success Summary ({successful_tests}/{total_tests} total)")
                axes[1, 1].grid(True, alpha=0.3, axis='y')

                plt.tight_layout()
                plots["parameter_sensitivity_summary"] = fig

            except Exception as e:
                logger.warning(f"Failed to create sensitivity plots: {e}")

            return {
                "status": "completed",
                "sensitivity_results": results,
                "plots": plots,
                "summary": {
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "success_rate": (
                        successful_tests / total_tests if total_tests > 0 else 0
                    ),
                    "parameters_tested": ["K", "percW", "mcmc_config"],
                    "data_characteristics": {
                        "n_subjects": X_list[0].shape[0],
                        "n_views": len(X_list),
                        "view_dimensions": [X.shape[1] for X in X_list],
                    },
                },
            }

        # Run experiment using framework
        result = framework.run_experiment(
            experiment_function=sensitivity_analysis_experiment,
            config=exp_config,
            model_results={"X_list": X_list, "preprocessing_info": preprocessing_info},
        )

        logger.info("✅ Sensitivity analysis completed successfully")
        return result

    except Exception as e:
        logger.error(f"Sensitivity analysis failed: {e}")
        return None
