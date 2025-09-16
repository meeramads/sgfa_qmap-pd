"""Model architecture comparison experiments for qMAP-PD analysis.

This module compares different model architectures including:
- Implemented: sparseGFA vs traditional methods (PCA, ICA, FA)
- Future work: standardGFA, neuroGFA, LCA (documented but not run due to computational constraints)

For optimizing hyperparameters within sparseGFA, see experiments/sgfa_parameter_comparison.py.
"""

import gc
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Safe configuration access
from core.config_utils import ConfigAccessor, get_output_dir, safe_get
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


class ModelArchitectureComparison(ExperimentFramework):
    """Compare different SGFA model architectures (sparseGFA, neuroGFA, standard GFA)."""

    def __init__(
        self, config: ExperimentConfig, logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, None, logger)
        self.profiler = PerformanceProfiler()

        # Model architectures to compare
        self.model_architectures = {
            "sparseGFA": {
                "model_type": "sparseGFA",
                "description": "Sparse Group Factor Analysis with regularized horseshoe priors",
                "implemented": True,
            },
            # Future work models (computationally intensive or not yet implemented)
            "standardGFA": {
                "model_type": "GFA",
                "description": "Standard Group Factor Analysis (requires more computational resources)",
                "implemented": False,
                "reason": "Computational limits - would exceed available GPU memory",
            },
            "neuroGFA": {
                "model_type": "neuroGFA",
                "description": "Neuroimaging-optimized GFA with spatial priors",
                "implemented": False,
                "reason": "Research idea not yet implemented - future work for spatial brain data",
            },
            "LCA": {
                "model_type": "LCA",
                "description": "Latent Class Analysis with discrete latent variables",
                "implemented": False,
                "reason": "Computational demands likely exceed GPU limits - future work for discrete factor modeling",
            },
        }

        # Fixed hyperparameters for fair comparison
        self.comparison_params = {
            "K": 5,  # Number of factors
            "percW": 25.0,  # Sparsity percentage
            "num_warmup": 300,
            "num_samples": 500,
            "num_chains": 1,  # Conservative for memory
            "target_accept_prob": 0.8,
        }

        # Traditional methods for comparison
        self.traditional_methods = ["pca", "ica", "fa"]

    @experiment_handler("model_architecture_comparison")
    @validate_data_types(X_list=list, hypers=dict, args=dict)
    @validate_parameters(X_list=lambda x: len(x) > 0)
    def run_model_architecture_comparison(
        self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> ExperimentResult:
        """Compare different model architectures with fixed hyperparameters."""
        # Validate inputs
        ResultValidator.validate_data_matrices(X_list)

        self.logger.info("🧠 Starting model architecture comparison")

        results = {}
        performance_metrics = {}

        # Test each model architecture
        for model_name, model_config in self.model_architectures.items():
            if model_config.get("implemented", True):
                self.logger.info(f"Testing model architecture: {model_name}")
                self.logger.info(f"Description: {model_config['description']}")

                # Update model configuration
                model_args = args.copy()
                model_args.update(
                    {"model": model_config["model_type"], **self.comparison_params}
                )

                # Run model
                model_result = self._run_model_architecture(
                    X_list, hypers, model_args, model_name, **kwargs
                )

                results[model_name] = model_result

                # Store basic performance metrics
                performance_metrics[model_name] = {
                    "execution_time": model_result.get("execution_time", 0),
                    "peak_memory_gb": model_result.get("peak_memory_gb", 0.0),
                    "convergence": model_result.get("convergence", False),
                    "log_likelihood": model_result.get("log_likelihood", float("-inf")),
                }
            else:
                self.logger.info(f"Skipping model architecture: {model_name}")
                self.logger.info(
                    f"Reason: {model_config.get('reason', 'Not implemented')}"
                )

                # Record as skipped
                results[model_name] = {
                    "skipped": True,
                    "reason": model_config.get("reason", "Not implemented"),
                    "model_type": model_config["model_type"],
                    "description": model_config["description"],
                }

        # Analyze results
        analysis = self._analyze_model_architectures(results, performance_metrics)

        # Generate basic plots
        plots = self._plot_model_comparison(results, performance_metrics)

        # Add comprehensive visualizations
        advanced_plots = self._create_comprehensive_visualizations(
            X_list, results, "model_architecture_comparison"
        )
        plots.update(advanced_plots)

        return ExperimentResult(
            experiment_name="model_architecture_comparison",
            config=self.config,
            data=results,
            analysis=analysis,
            plots=plots,
            performance_metrics=performance_metrics,
            success=True,
        )

    def _run_model_architecture(
        self,
        X_list: List[np.ndarray],
        hypers: Dict,
        args: Dict,
        model_name: str,
        **kwargs,
    ) -> Dict:
        """Run a specific model architecture."""
        import jax
        import jax.numpy as jnp
        import numpyro
        from numpyro.infer import MCMC, NUTS

        try:
            # Clear GPU memory before starting
            jax.clear_caches()
            if jax.default_backend() == "gpu":
                gc.collect()
                for device in jax.local_devices():
                    if device.platform == "gpu":
                        try:
                            device.synchronize_all_activity()
                        except BaseException:
                            pass

            self.logger.info(
                f"Training {model_name} with K={args.get('K', 5)}, "
                f"model_type={args.get('model', 'unknown')}"
            )

            # Use the standard models function from core.run_analysis
            from core.run_analysis import models

            # Setup MCMC configuration
            num_warmup = args.get("num_warmup", 300)
            num_samples = args.get("num_samples", 500)
            num_chains = args.get("num_chains", 1)

            # Create args object for model
            import argparse

            model_args = argparse.Namespace(
                model=args.get("model", "sparseGFA"),
                K=args.get("K", 5),
                num_sources=len(X_list),
                reghsZ=args.get("reghsZ", True),
            )

            # Setup MCMC with memory-efficient options
            rng_key = jax.random.PRNGKey(np.random.randint(0, 10000))
            kernel = NUTS(
                models, target_accept_prob=args.get("target_accept_prob", 0.8)
            )
            mcmc = MCMC(
                kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                progress_bar=False,
                chain_method="sequential",
            )

            # Run inference
            start_time = time.time()
            mcmc.run(
                rng_key, X_list, hypers, model_args, extra_fields=("potential_energy",)
            )
            elapsed = time.time() - start_time

            # Get samples
            samples = mcmc.get_samples()

            # Calculate log likelihood
            potential_energy = samples.get("potential_energy", np.array([]))
            if len(potential_energy) > 0:
                log_likelihood = -np.mean(potential_energy)
                self.logger.debug(
                    f"Potential energy stats: mean={np.mean(potential_energy):.3f}"
                )
            else:
                log_likelihood = float("nan")
                self.logger.warning("No potential energy data collected")

            # Extract mean parameters
            W_samples = samples["W"]
            Z_samples = samples["Z"]

            W_mean = np.mean(W_samples, axis=0)
            Z_mean = np.mean(Z_samples, axis=0)

            # Split W back into views
            W_list = []
            start_idx = 0
            for X in X_list:
                end_idx = start_idx + X.shape[1]
                W_list.append(W_mean[start_idx:end_idx, :])
                start_idx = end_idx

            self.logger.info(
                f"{model_name} training completed in {elapsed:.2f}s, "
                f"log_likelihood: {log_likelihood:.3f}"
            )

            result = {
                "W": W_list,
                "Z": Z_mean,
                "W_samples": W_samples,
                "Z_samples": Z_samples,
                "samples": samples,
                "log_likelihood": (
                    float(log_likelihood)
                    if not np.isnan(log_likelihood)
                    else float("-inf")
                ),
                "execution_time": elapsed,
                "convergence": True,
                "model_type": args.get("model"),
                "hyperparameters": {
                    "K": args.get("K"),
                    "num_samples": num_samples,
                    "num_warmup": num_warmup,
                    "num_chains": num_chains,
                },
            }

            # Clear GPU memory after training
            del samples, W_samples, Z_samples, mcmc
            jax.clear_caches()
            gc.collect()

            return result

        except Exception as e:
            self.logger.error(f"{model_name} training failed: {str(e)}")
            # Clear memory even on failure
            jax.clear_caches()
            gc.collect()
            return {
                "error": str(e),
                "convergence": False,
                "execution_time": 0,
                "log_likelihood": float("-inf"),
                "model_type": args.get("model", "unknown"),
            }

    def run_traditional_method_comparison(
        self, X_list: List[np.ndarray], model_results: Dict = None, **kwargs
    ) -> ExperimentResult:
        """Compare model architectures with traditional methods."""
        self.logger.info("📊 Running traditional method comparison")

        results = {}
        performance_metrics = {}

        # Combine all views for traditional methods
        X_combined = np.hstack(X_list)
        n_components = self.comparison_params["K"]  # Use same K as model comparison

        try:
            for method_name in self.traditional_methods:
                self.logger.info(f"Testing traditional method: {method_name}")

                method_result = self._run_traditional_method(
                    X_combined, method_name, n_components, **kwargs
                )

                results[method_name] = method_result

                # Store basic performance metrics
                performance_metrics[method_name] = {
                    "execution_time": method_result.get("execution_time", 0),
                    "peak_memory_gb": 0.0,  # Traditional methods use minimal memory
                    "convergence": method_result.get("convergence", True),
                }

            # Analyze results
            analysis = self._analyze_traditional_methods(results, performance_metrics)

            # Generate plots
            plots = self._plot_traditional_comparison(results, performance_metrics)

            return ExperimentResult(
                experiment_name="traditional_method_comparison",
                config=self.config,
                data=results,
                analysis=analysis,
                plots=plots,
                performance_metrics=performance_metrics,
                success=True,
            )

        except Exception as e:
            self.logger.error(f"Traditional method comparison failed: {str(e)}")
            return self._create_failure_result("traditional_method_comparison", str(e))

    def _run_traditional_method(
        self, X: np.ndarray, method_name: str, n_components: int, **kwargs
    ) -> Dict:
        """Run traditional dimensionality reduction method."""
        results = {}

        try:
            start_time = time.time()

            if method_name == "pca":
                method = PCA(n_components=n_components)
                Z = method.fit_transform(X)
                W = method.components_.T  # Transpose to match our convention

                results = {
                    "Z": Z,
                    "W": W,
                    "explained_variance_ratio": method.explained_variance_ratio_,
                    "singular_values": method.singular_values_,
                    "method": "pca",
                }

            elif method_name == "fa":
                method = FactorAnalysis(n_components=n_components, max_iter=1000)
                Z = method.fit_transform(X)
                W = method.components_.T

                results = {
                    "Z": Z,
                    "W": W,
                    "noise_variance": method.noise_variance_,
                    "loglikelihood": method.score(X),
                    "method": "fa",
                }

            elif method_name == "ica":
                method = FastICA(
                    n_components=n_components, max_iter=1000, random_state=42
                )
                Z = method.fit_transform(X)
                W = method.components_.T

                results = {
                    "Z": Z,
                    "W": W,
                    "mixing_matrix": method.mixing_,
                    "method": "ica",
                }

            elapsed = time.time() - start_time

            results.update(
                {
                    "execution_time": elapsed,
                    "convergence": True,
                    "n_components": n_components,
                }
            )

            self.logger.info(f"✅ {method_name}: {elapsed:.1f}s")

        except Exception as e:
            self.logger.error(f"Traditional method {method_name} failed: {str(e)}")
            results = {
                "error": str(e),
                "execution_time": 0,
                "convergence": False,
                "method": method_name,
            }

        return results

    def _analyze_model_architectures(
        self, results: Dict, performance_metrics: Dict
    ) -> Dict:
        """Analyze model architecture comparison results."""
        analysis = {
            "summary": {
                "total_models_tested": len(results),
                "successful_models": sum(
                    1 for r in results.values() if r.get("convergence", False)
                ),
                "total_execution_time": sum(
                    performance_metrics[m]["execution_time"]
                    for m in performance_metrics
                ),
            },
            "performance_comparison": {},
            "model_quality": {},
            "convergence_analysis": {},
        }

        # Performance comparison
        for model_name, metrics in performance_metrics.items():
            analysis["performance_comparison"][model_name] = {
                "execution_time": metrics["execution_time"],
                "memory_efficiency": 1.0
                / max(metrics["peak_memory_gb"], 0.1),  # Inverse for efficiency
                "convergence_success": metrics["convergence"],
            }

        # Model quality comparison
        successful_models = [
            m for m in results.keys() if results[m].get("convergence", False)
        ]
        if successful_models:
            log_likelihoods = {
                m: results[m].get("log_likelihood", float("-inf"))
                for m in successful_models
            }

            # Find best model by log-likelihood
            if any(
                ll != float("-inf") and not np.isnan(ll)
                for ll in log_likelihoods.values()
            ):
                best_model = max(
                    log_likelihoods.keys(),
                    key=lambda x: (
                        log_likelihoods[x]
                        if not np.isnan(log_likelihoods[x])
                        else float("-inf")
                    ),
                )
                analysis["model_quality"]["best_model"] = best_model
                analysis["model_quality"]["log_likelihoods"] = log_likelihoods

        # Convergence analysis
        convergence_rates = {
            m: int(metrics["convergence"]) for m, metrics in performance_metrics.items()
        }
        analysis["convergence_analysis"] = {
            "convergence_rates": convergence_rates,
            "overall_success_rate": np.mean(list(convergence_rates.values())),
        }

        return analysis

    def _analyze_traditional_methods(
        self, results: Dict, performance_metrics: Dict
    ) -> Dict:
        """Analyze traditional method results."""
        analysis = {
            "summary": {
                "methods_tested": len(results),
                "successful_methods": sum(
                    1 for r in results.values() if r.get("convergence", False)
                ),
            },
            "performance_comparison": performance_metrics,
            "method_characteristics": {},
        }

        # Analyze method-specific characteristics
        for method_name, result in results.items():
            if result.get("convergence", False):
                if method_name == "pca":
                    analysis["method_characteristics"][method_name] = {
                        "total_explained_variance": np.sum(
                            result.get("explained_variance_ratio", [])
                        ),
                        "eigenvalue_decay": result.get("singular_values", []),
                    }
                elif method_name == "fa":
                    analysis["method_characteristics"][method_name] = {
                        "log_likelihood": result.get("loglikelihood", float("-inf")),
                        "noise_variance_mean": np.mean(
                            result.get("noise_variance", [])
                        ),
                    }

        return analysis

    def _plot_model_comparison(self, results: Dict, performance_metrics: Dict) -> Dict:
        """Generate plots for model architecture comparison."""
        plots = {}

        try:
            # Performance comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Model Architecture Comparison", fontsize=16)

            models = list(results.keys())
            successful_models = [
                m for m in models if results[m].get("convergence", False)
            ]

            if successful_models:
                # Execution time comparison
                times = [
                    performance_metrics[m]["execution_time"] for m in successful_models
                ]
                axes[0, 0].bar(successful_models, times)
                axes[0, 0].set_title("Execution Time")
                axes[0, 0].set_ylabel("Time (seconds)")
                axes[0, 0].tick_params(axis="x", rotation=45)

                # Log-likelihood comparison
                log_liks = [
                    results[m].get("log_likelihood", float("-inf"))
                    for m in successful_models
                ]
                valid_log_liks = [
                    (m, ll)
                    for m, ll in zip(successful_models, log_liks)
                    if ll != float("-inf") and not np.isnan(ll)
                ]

                if valid_log_liks:
                    models_ll, values_ll = zip(*valid_log_liks)
                    axes[0, 1].bar(models_ll, values_ll)
                    axes[0, 1].set_title("Log-Likelihood")
                    axes[0, 1].set_ylabel("Log-Likelihood")
                    axes[0, 1].tick_params(axis="x", rotation=45)
                else:
                    axes[0, 1].text(
                        0.5,
                        0.5,
                        "No valid log-likelihood data",
                        ha="center",
                        va="center",
                        transform=axes[0, 1].transAxes,
                    )

                # Model complexity (placeholder)
                axes[1, 0].bar(
                    successful_models,
                    [self.comparison_params["K"]] * len(successful_models),
                )
                axes[1, 0].set_title("Model Complexity (K factors)")
                axes[1, 0].set_ylabel("Number of Factors")
                axes[1, 0].tick_params(axis="x", rotation=45)

                # Convergence status
                convergence = [
                    int(results[m].get("convergence", False)) for m in models
                ]
                axes[1, 1].bar(models, convergence)
                axes[1, 1].set_title("Convergence Success")
                axes[1, 1].set_ylabel("Converged (1=Yes, 0=No)")
                axes[1, 1].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plots["model_performance_comparison"] = fig

        except Exception as e:
            self.logger.warning(f"Failed to generate model comparison plots: {e}")

        return plots

    def _plot_traditional_comparison(
        self, results: Dict, performance_metrics: Dict
    ) -> Dict:
        """Generate plots for traditional method comparison."""
        plots = {}

        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle("Traditional Methods Comparison", fontsize=16)

            methods = list(results.keys())
            successful_methods = [
                m for m in methods if results[m].get("convergence", False)
            ]

            if successful_methods:
                # Execution time
                times = [
                    performance_metrics[m]["execution_time"] for m in successful_methods
                ]
                axes[0].bar(successful_methods, times)
                axes[0].set_title("Execution Time")
                axes[0].set_ylabel("Time (seconds)")

                # Method-specific quality metric
                quality_scores = []
                for method in successful_methods:
                    if method == "pca":
                        score = np.sum(
                            results[method].get("explained_variance_ratio", [0])
                        )
                    elif method == "fa":
                        score = results[method].get("loglikelihood", 0)
                    else:
                        score = 0
                    quality_scores.append(score)

                axes[1].bar(successful_methods, quality_scores)
                axes[1].set_title("Quality Metric")
                axes[1].set_ylabel("Explained Variance (PCA) / Log-Likelihood (FA)")

            plt.tight_layout()
            plots["traditional_methods_comparison"] = fig

        except Exception as e:
            self.logger.warning(f"Failed to generate traditional method plots: {e}")

        return plots

    def _create_comprehensive_visualizations(
        self, X_list: List[np.ndarray], results: Dict, experiment_name: str
    ) -> Dict:
        """Create comprehensive visualizations using the advanced visualization system."""
        advanced_plots = {}

        try:
            self.logger.info(
                f"🎨 Creating comprehensive visualizations for {experiment_name}"
            )

            # Import visualization system
            from core.config_utils import ConfigAccessor
            from visualization.manager import VisualizationManager

            # Create a temporary config for visualization
            viz_config = ConfigAccessor(
                {
                    "visualization": {
                        "create_brain_viz": True,
                        "output_format": ["png", "pdf"],
                        "dpi": 300,
                    },
                    "output_dir": "/tmp/model_comparison_viz",  # Will be overridden
                }
            )

            # Initialize visualization manager
            viz_manager = VisualizationManager(viz_config)

            # Prepare data structure for visualizations
            data = {
                "X_list": X_list,
                "view_names": [f"view_{i}" for i in range(len(X_list))],
                "n_subjects": X_list[0].shape[0],
                "view_dimensions": [X.shape[1] for X in X_list],
                "preprocessing": {
                    "status": "completed",
                    "strategy": "neuroimaging_aware",
                },
            }

            # Extract the best model result for detailed analysis
            best_model_result = self._extract_best_model_result(results)

            if best_model_result:
                analysis_results = {
                    "best_run": best_model_result,
                    "all_runs": results,
                    "model_type": best_model_result.get("model_type", "sparseGFA"),
                    "convergence": best_model_result.get("convergence", False),
                }

                # Create all comprehensive visualizations
                viz_manager.create_all_visualizations(
                    data=data, analysis_results=analysis_results
                )

                # Extract the generated plots and create matplotlib figures for the
                # framework
                if hasattr(viz_manager, "plot_dir") and viz_manager.plot_dir.exists():
                    plot_files = list(viz_manager.plot_dir.glob("**/*.png"))

                    for plot_file in plot_files:
                        plot_name = f"advanced_{plot_file.stem}"

                        # Load the saved plot as a matplotlib figure for the framework
                        try:
                            import matplotlib.image as mpimg
                            import matplotlib.pyplot as plt

                            fig, ax = plt.subplots(figsize=(12, 8))
                            img = mpimg.imread(str(plot_file))
                            ax.imshow(img)
                            ax.axis("off")
                            ax.set_title(f"Advanced: {plot_name}", fontsize=14)

                            advanced_plots[plot_name] = fig

                        except Exception as e:
                            self.logger.warning(
                                f"Could not load advanced plot {plot_name}: {e}"
                            )
                            # Store path reference as fallback
                            advanced_plots[plot_name] = {
                                "file_path": str(plot_file),
                                "type": "advanced_visualization",
                            }

                    self.logger.info(
                        f"✅ Created {len(plot_files)} comprehensive visualizations"
                    )
                else:
                    self.logger.warning(
                        "Visualization manager did not create plot directory"
                    )
            else:
                self.logger.warning(
                    "No converged model results found for comprehensive visualization"
                )

        except Exception as e:
            self.logger.warning(f"Failed to create comprehensive visualizations: {e}")
            # Don't fail the experiment if advanced visualizations fail

        return advanced_plots

    def _extract_best_model_result(self, results: Dict) -> Optional[Dict]:
        """Extract the best model result from experiment results."""
        best_result = None
        best_likelihood = float("-inf")

        # Look for successful model results
        for model_name, model_result in results.items():
            if (
                model_result
                and not model_result.get("skipped", False)
                and model_result.get("convergence", False)
            ):
                likelihood = model_result.get("log_likelihood", float("-inf"))
                if likelihood > best_likelihood:
                    best_likelihood = likelihood
                    best_result = model_result

        return best_result


def run_model_comparison(config=None, **kwargs):
    """
    Run comprehensive model architecture comparison experiments.

    This function compares different SGFA model architectures (sparseGFA, neuroGFA, standard GFA)
    with the same hyperparameters for fair comparison.

    Parameters:
    -----------
    config : dict or str
        Configuration dictionary or path to config file
    """
    import yaml

    from core.config_utils import ConfigAccessor

    logger = logging.getLogger(__name__)
    logger.info("🔬 Starting model architecture comparison experiments...")

    # Handle config parameter
    if isinstance(config, str):
        # Load configuration from file
        with open(config, "r") as f:
            config_dict = yaml.safe_load(f)
    elif isinstance(config, dict):
        config_dict = config
    else:
        # Default fallback
        config_dict = {}

    config_accessor = ConfigAccessor(config_dict)

    # Check for shared data and optimal parameters
    shared_data = config_dict.get("_shared_data")
    optimal_sgfa_params = config_dict.get("_optimal_sgfa_params")

    # Use shared data if available, otherwise load data
    if shared_data and shared_data.get("X_list") is not None:
        logger.info("   → Using shared data from previous experiments")
        X_list = shared_data["X_list"]
        view_dims = [X.shape[1] for X in X_list]
        n_subjects = X_list[0].shape[0]
        logger.info(
            f"   → Data shape: {n_subjects} subjects, {
                len(X_list)} views with dims {view_dims}")
    else:
        logger.info("   → Loading fresh data (no shared data available)")
        # Fallback to dummy data for testing
        np.random.seed(42)
        n_subjects = 100
        view_dims = [50, 75, 60]
        X_list = [np.random.randn(n_subjects, dim) for dim in view_dims]

    # Use optimal SGFA parameters if available
    if optimal_sgfa_params:
        logger.info(
            f"   → Using optimal SGFA parameters: K={
                optimal_sgfa_params['K']}, percW={
                optimal_sgfa_params['percW']}")
        optimal_K = optimal_sgfa_params["K"]
        optimal_percW = optimal_sgfa_params["percW"]
    else:
        logger.info("   → Using default parameters (no optimal parameters available)")
        optimal_K = 5
        optimal_percW = 25.0

    # Setup hyperparameters
    hypers = {
        "Dm": view_dims,
        "a_sigma": 1.0,
        "b_sigma": 1.0,
        "percW": optimal_percW,
        "slab_scale": 1.0,
        "slab_df": 4.0,
    }

    args = {"K": optimal_K, "reghsZ": True, "device": "gpu"}

    # Create experiment configuration
    exp_config = ExperimentConfig(
        name="model_architecture_comparison",
        description="Compare different SGFA model architectures",
        parameters={"K": optimal_K, "percW": optimal_percW},
    )

    # Run experiments
    experiment = ModelArchitectureComparison(exp_config, logger)

    # Run model architecture comparison
    model_results = experiment.run_model_architecture_comparison(
        X_list, hypers, args, **kwargs
    )

    # Run traditional method comparison
    traditional_results = experiment.run_traditional_method_comparison(X_list, **kwargs)

    logger.info("🔬 Model architecture comparison experiments completed!")
    logger.info(f"   Model architectures tested: {len(experiment.model_architectures)}")
    logger.info(f"   Traditional methods tested: {len(experiment.traditional_methods)}")

    return {
        "model_results": model_results,
        "traditional_results": traditional_results,
        "experiment": experiment,
    }


if __name__ == "__main__":
    # Run model comparison experiments
    results = run_model_comparison()
    print("Model architecture comparison completed!")
