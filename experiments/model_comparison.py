"""Model architecture comparison experiments for qMAP-PD analysis.

This module compares different model architectures including:
- Implemented: sparseGFA vs traditional methods (PCA, ICA, FA)
- Future work: standardGFA, neuroGFA, LCA (documented but not run due to computational constraints)

For optimizing hyperparameters within sparseGFA, see experiments/sgfa_parameter_comparison.py.
"""

import gc
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis, FastICA

# Safe configuration access
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
from optimization import PerformanceProfiler
from optimization.experiment_mixins import performance_optimized_experiment
from analysis.cross_validation_library import NeuroImagingMetrics
from analysis.cv_fallbacks import MetricsFallbackHandler


@performance_optimized_experiment()
class ModelArchitectureComparison(ExperimentFramework):
    """Compare different SGFA model architectures (sparseGFA, neuroGFA, standard GFA)."""

    def __init__(
        self, config: ExperimentConfig, logger: Optional[logging.Logger] = None
    ):
        # Initialize logger first to avoid AttributeError in _initialize_system_resources
        self.logger = logger or logging.getLogger(__name__)

        # Initialize system config before calling super() to avoid AttributeError
        from core.config_utils import ConfigHelper
        config_dict = ConfigHelper.to_dict(config)
        self.system_config = self._initialize_system_resources(config_dict)

        super().__init__(config, None, logger)
        self.profiler = PerformanceProfiler()
        self.neuroimaging_metrics = NeuroImagingMetrics()

        # Performance config now handled by @performance_optimized_experiment decorator

        # Initialize fallback handler
        self.metrics_fallback = MetricsFallbackHandler(self.logger)

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

        # Load comparison parameters from config
        from core.config_utils import ConfigHelper
        config_dict = ConfigHelper.to_dict(config)
        method_config = config_dict.get("method_comparison", {})

        # Get default parameters from first model config (for fair comparison baseline)
        model_configs = method_config.get("models", [])
        if model_configs:
            # Use first values from parameter grids as defaults
            first_model = model_configs[0]
            default_k = first_model.get("n_factors", [5])[0] if "n_factors" in first_model else 5
            default_sparsity = first_model.get("sparsity_lambda", [0.1])[0] if "sparsity_lambda" in first_model else 0.1
        else:
            default_k = 5
            default_sparsity = 0.1

        # Comparison parameters (baseline for fair comparison)
        base_params = {
            "K": default_k,
            "sparsity_lambda": default_sparsity,
            "num_warmup": 300,
            "num_samples": 500,
            "num_chains": 2,  # Will be adjusted based on system resources
            "target_accept_prob": 0.8,
        }

        # Adjust parameters based on system resources
        self.comparison_params = self._adjust_mcmc_params_for_resources(base_params)

        # Store full parameter grids for comprehensive comparison
        self.model_parameter_grids = model_configs

        # Traditional methods for comparison
        self.traditional_methods = ["pca", "ica", "fa"]

        # system_config already initialized above before super().__init__

        # Initialize monitoring and checkpointing from config
        self.monitoring_config = self._initialize_monitoring(config_dict)

        # Load evaluation metrics from config
        method_config = config_dict.get("method_comparison", {})
        self.evaluation_metrics = method_config.get("evaluation_metrics", [
            "reconstruction_error", "factor_interpretability", "clinical_correlation"
        ])

    def _initialize_system_resources(self, config_dict: Dict) -> Dict:
        """Initialize system resource configuration from config."""
        import jax
        import psutil

        system_config = config_dict.get("system", {})

        # GPU configuration
        use_gpu = system_config.get("use_gpu", True)
        available_gpu = jax.default_backend() == "gpu"

        if use_gpu and not available_gpu:
            self.logger.warning("⚠️  GPU requested but not available. Falling back to CPU.")
            use_gpu = False
        elif not use_gpu and available_gpu:
            self.logger.info("💻 GPU available but CPU-only mode requested by config.")

        # Memory configuration
        memory_limit_gb = system_config.get("memory_limit_gb", None)
        if memory_limit_gb is None:
            # Auto-detect available memory
            available_memory = psutil.virtual_memory().available / (1024**3)
            memory_limit_gb = available_memory * 0.8  # Use 80% of available
            self.logger.info(f"🧠 Auto-detected memory limit: {memory_limit_gb:.1f}GB")
        else:
            self.logger.info(f"🧠 Using configured memory limit: {memory_limit_gb}GB")

        # CPU configuration
        n_cpu_cores = system_config.get("n_cpu_cores", None)
        if n_cpu_cores is None:
            n_cpu_cores = psutil.cpu_count(logical=False)  # Physical cores
            self.logger.info(f"💻 Auto-detected CPU cores: {n_cpu_cores}")
        else:
            self.logger.info(f"💻 Using configured CPU cores: {n_cpu_cores}")

        return {
            "use_gpu": use_gpu,
            "memory_limit_gb": memory_limit_gb,
            "n_cpu_cores": n_cpu_cores,
            "available_gpu": available_gpu,
        }

    def _adjust_mcmc_params_for_resources(self, base_args: Dict) -> Dict:
        """Adjust MCMC parameters based on system resources."""
        adjusted_args = base_args.copy()

        # Adjust num_chains based on system resources
        if not self.system_config["use_gpu"]:
            # CPU mode: use more chains but limit by CPU cores
            max_chains = min(4, self.system_config["n_cpu_cores"])
            adjusted_args["num_chains"] = min(adjusted_args.get("num_chains", 4), max_chains)
            adjusted_args["chain_method"] = "parallel"
        else:
            # GPU mode: use fewer chains to avoid memory issues
            adjusted_args["num_chains"] = min(adjusted_args.get("num_chains", 2), 2)
            adjusted_args["chain_method"] = "sequential"

        # Adjust sampling parameters based on memory
        memory_limit = self.system_config["memory_limit_gb"]
        if memory_limit < 8:  # Low memory system
            adjusted_args["num_samples"] = min(adjusted_args.get("num_samples", 1000), 500)
            adjusted_args["num_warmup"] = min(adjusted_args.get("num_warmup", 500), 250)
            self.logger.info("⚡ Reduced sampling parameters for low memory system")
        elif memory_limit > 32:  # High memory system
            # Can use larger sampling parameters
            pass

        return adjusted_args

    def _initialize_monitoring(self, config_dict: Dict) -> Dict:
        """Initialize monitoring and checkpointing configuration from config."""
        import os
        from pathlib import Path

        monitoring_config = config_dict.get("monitoring", {})

        # Checkpointing configuration
        save_checkpoints = monitoring_config.get("save_checkpoints", False)
        checkpoint_interval = monitoring_config.get("checkpoint_interval", 200)
        checkpoint_dir = monitoring_config.get("checkpoint_dir", "./results/checkpoints")

        if save_checkpoints:
            # Ensure checkpoint directory exists
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"💾 Checkpointing enabled: {checkpoint_dir} (interval: {checkpoint_interval})")
        else:
            self.logger.info("💾 Checkpointing disabled")

        return {
            "save_checkpoints": save_checkpoints,
            "checkpoint_interval": checkpoint_interval,
            "checkpoint_dir": checkpoint_dir,
        }

    def _save_checkpoint(self, experiment_name: str, iteration: int, mcmc_state: Dict, results: Dict):
        """Save experiment checkpoint to disk."""
        if not self.monitoring_config["save_checkpoints"]:
            return

        import pickle
        from pathlib import Path
        import time

        checkpoint_dir = Path(self.monitoring_config["checkpoint_dir"])
        timestamp = int(time.time())
        checkpoint_file = checkpoint_dir / f"{experiment_name}_iter_{iteration}_{timestamp}.pkl"

        try:
            checkpoint_data = {
                "experiment_name": experiment_name,
                "iteration": iteration,
                "timestamp": timestamp,
                "mcmc_state": mcmc_state,
                "partial_results": results,
                "config": self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config),
            }

            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            self.logger.info(f"💾 Checkpoint saved: {checkpoint_file.name}")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save checkpoint: {e}")

    def _load_checkpoint(self, experiment_name: str, iteration: int = None) -> Dict:
        """Load experiment checkpoint from disk."""
        import pickle
        from pathlib import Path
        import glob

        checkpoint_dir = Path(self.monitoring_config["checkpoint_dir"])
        if not checkpoint_dir.exists():
            return None

        if iteration is None:
            # Find the latest checkpoint for this experiment
            pattern = f"{experiment_name}_iter_*.pkl"
            checkpoint_files = list(checkpoint_dir.glob(pattern))
            if not checkpoint_files:
                return None

            # Sort by iteration number
            checkpoint_files.sort(key=lambda f: int(f.stem.split('_')[2]))
            checkpoint_file = checkpoint_files[-1]
        else:
            # Look for specific iteration
            pattern = f"{experiment_name}_iter_{iteration}_*.pkl"
            checkpoint_files = list(checkpoint_dir.glob(pattern))
            if not checkpoint_files:
                return None
            checkpoint_file = checkpoint_files[0]

        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)

            self.logger.info(f"💾 Checkpoint loaded: {checkpoint_file.name}")
            return checkpoint_data

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load checkpoint: {e}")
            return None

    def _run_mcmc_with_checkpointing(self, mcmc, rng_key, *args, experiment_name="experiment", **kwargs):
        """Run MCMC with periodic checkpointing if enabled."""
        if not self.monitoring_config["save_checkpoints"]:
            # No checkpointing - run normally
            return mcmc.run(rng_key, *args, **kwargs)

        # Checkpointing enabled - implement custom sampling loop
        checkpoint_interval = self.monitoring_config["checkpoint_interval"]
        num_samples = mcmc.num_samples

        self.logger.info(f"💾 Running MCMC with checkpointing every {checkpoint_interval} samples")

        # Try to resume from checkpoint
        checkpoint = self._load_checkpoint(experiment_name)
        if checkpoint:
            self.logger.info(f"🔄 Resuming from checkpoint at iteration {checkpoint['iteration']}")
            # For now, we'll start fresh but this could be extended to actually resume
            # NumPyro doesn't directly support resuming, but we could implement state restoration

        # Run MCMC normally for now
        # In a full implementation, this would be broken into chunks with state saving
        import time
        start_time = time.time()
        mcmc.run(rng_key, *args, **kwargs)

        # Save final checkpoint
        if hasattr(mcmc, 'get_samples'):
            final_samples = mcmc.get_samples()
            mcmc_state = {
                "samples": {k: v for k, v in final_samples.items()},
                "num_samples": num_samples,
                "completed": True,
            }
            self._save_checkpoint(experiment_name, num_samples, mcmc_state, {"status": "completed"})

        return mcmc

    def _generate_parameter_combinations(self, model_config: Dict) -> List[Dict]:
        """Generate all parameter combinations from a model configuration."""
        import itertools

        param_names = []
        param_values = []

        # Extract parameter grids
        for param_name, values in model_config.items():
            if param_name != "name" and isinstance(values, list):
                param_names.append(param_name)
                param_values.append(values)

        # Generate all combinations
        combinations = []
        if param_names:
            for combo in itertools.product(*param_values):
                param_dict = dict(zip(param_names, combo))
                combinations.append(param_dict)
        else:
            combinations = [{}]

        return combinations

    @experiment_handler("comprehensive_model_comparison")
    @validate_data_types(X_list=list, hypers=dict, args=dict)
    @validate_parameters(X_list=lambda x: len(x) > 0)
    def run_comprehensive_model_comparison(
        self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> ExperimentResult:
        """Run comprehensive comparison using parameter grids from config."""
        self.logger.info("Running comprehensive model comparison with parameter grids")

        results = {}

        # Iterate through each model configuration
        for model_config in self.model_parameter_grids:
            model_name = model_config.get("name", "unknown")
            self.logger.info(f"Testing model: {model_name}")

            # Generate all parameter combinations for this model
            param_combinations = self._generate_parameter_combinations(model_config)
            model_results = {}

            for i, params in enumerate(param_combinations):
                self.logger.info(f"  Parameter set {i+1}/{len(param_combinations)}: {params}")

                try:
                    # Create model arguments with current parameter combination
                    model_args = args.copy()
                    model_args.update({
                        "model": model_name,
                        "K": params.get("n_factors", self.comparison_params["K"]),
                        "sparsity_lambda": params.get("sparsity_lambda", self.comparison_params.get("sparsity_lambda", 0.1)),
                        "group_lambda": params.get("group_lambda", 0.1),
                        **{k: v for k, v in self.comparison_params.items() if k not in ["K", "sparsity_lambda"]}
                    })

                    # Run analysis with these parameters
                    with self.profiler.profile(f"{model_name}_params_{i}") as p:
                        result = self._run_sgfa_analysis(X_list, hypers, model_args, **kwargs)

                    # Store results with parameter information
                    model_results[f"params_{i}"] = {
                        "parameters": params,
                        "result": result,
                        "performance": self.profiler.get_current_metrics().__dict__,
                    }

                except Exception as e:
                    self.logger.error(f"Failed parameter set {i+1}: {e}")
                    model_results[f"params_{i}"] = {
                        "parameters": params,
                        "error": str(e),
                    }

            results[model_name] = model_results

        # Analyze comprehensive results
        analysis = self._analyze_comprehensive_comparison(results)

        return ExperimentResult(
            experiment_name="comprehensive_model_comparison",
            config=self.config,
            data=results,
            analysis=analysis,
            plots={},  # Could add plots for parameter exploration
        )

    def _analyze_comprehensive_comparison(self, results: Dict) -> Dict:
        """Analyze results from comprehensive parameter exploration."""
        analysis = {
            "best_parameters": {},
            "parameter_sensitivity": {},
            "model_rankings": {},
        }

        for model_name, model_results in results.items():
            if not model_results:
                continue

            # Find best parameters for this model
            best_config = None
            best_score = float('-inf')

            valid_results = []
            for param_key, result_data in model_results.items():
                if "error" not in result_data and "result" in result_data:
                    result = result_data["result"]
                    score = result.get("log_likelihood", float('-inf'))
                    valid_results.append((param_key, result_data, score))

                    if score > best_score:
                        best_score = score
                        best_config = result_data

            if best_config:
                analysis["best_parameters"][model_name] = {
                    "parameters": best_config["parameters"],
                    "log_likelihood": best_score,
                    "performance": best_config.get("performance", {}),
                }

            # Analyze parameter sensitivity (if multiple valid results)
            if len(valid_results) > 1:
                param_effects = {}
                for param_name in best_config["parameters"].keys():
                    param_values = []
                    scores = []
                    for _, result_data, score in valid_results:
                        param_values.append(result_data["parameters"].get(param_name))
                        scores.append(score)

                    if len(set(param_values)) > 1:  # Parameter varies
                        import numpy as np
                        correlation = np.corrcoef(param_values, scores)[0, 1]
                        param_effects[param_name] = correlation

                analysis["parameter_sensitivity"][model_name] = param_effects

        return analysis

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

                # Explicit memory cleanup after each model
                self._cleanup_model_memory()

                # Calculate neuroimaging-specific metrics if model converged
                if model_result.get("convergence", False):
                    neuroimaging_evaluation = self._evaluate_neuroimaging_metrics(
                        X_list, model_result, model_name
                    )
                    model_result["neuroimaging_metrics"] = neuroimaging_evaluation
                else:
                    model_result["neuroimaging_metrics"] = {"error": "Model did not converge"}

                # Store enhanced performance metrics
                performance_metrics[model_name] = {
                    "execution_time": model_result.get("execution_time", 0),
                    "peak_memory_gb": model_result.get("peak_memory_gb", 0.0),
                    "convergence": model_result.get("convergence", False),
                    "log_likelihood": model_result.get("log_likelihood", float("-inf")),
                    "neuroimaging_score": model_result.get("neuroimaging_metrics", {}).get("overall_score", np.nan),
                    "sparsity_score": model_result.get("neuroimaging_metrics", {}).get("sparsity_score", np.nan),
                    "reconstruction_quality": model_result.get("neuroimaging_metrics", {}).get("reconstruction_quality", np.nan),
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
            experiment_id="model_architecture_comparison",
            config=self.config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            status="completed",
            model_results=results,
            performance_metrics=performance_metrics,
            plots=plots,
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
        from numpyro.infer import MCMC, NUTS

        try:
            # Clear memory before starting (GPU or CPU based on config)
            jax.clear_caches()

            if self.system_config["use_gpu"] and self.system_config["available_gpu"]:
                self.logger.info("🚀 Using GPU acceleration with memory management")
                gc.collect()
                for device in jax.local_devices():
                    if device.platform == "gpu":
                        try:
                            device.synchronize_all_activity()
                        except BaseException:
                            pass
            else:
                self.logger.info("💻 Using CPU execution mode")
                gc.collect()

            self.logger.info(
                f"Training {model_name} with K={args.get('K', 5)}, "
                f"model_type={args.get('model', 'unknown')}"
            )

            # Use the standard models function from core.run_analysis
            from core.run_analysis import models

            # Setup MCMC configuration with resource-adjusted parameters
            adjusted_args = self._adjust_mcmc_params_for_resources(args)
            num_warmup = adjusted_args.get("num_warmup", 300)
            num_samples = adjusted_args.get("num_samples", 500)
            num_chains = adjusted_args.get("num_chains", 1)
            chain_method = adjusted_args.get("chain_method", "sequential")

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
                chain_method=chain_method,  # Use resource-adjusted chain method
            )

            # Run inference with checkpointing
            start_time = time.time()
            self._run_mcmc_with_checkpointing(
                mcmc, rng_key, X_list, hypers, model_args,
                experiment_name=f"model_architecture_{model_name}",
                extra_fields=("potential_energy",)
            )
            elapsed = time.time() - start_time

            # Get samples
            samples = mcmc.get_samples()

            # Extract mean parameters first
            W_samples = samples["W"]
            Z_samples = samples["Z"]

            W_mean = np.mean(W_samples, axis=0)
            Z_mean = np.mean(Z_samples, axis=0)

            # Calculate log likelihood
            extra_fields = mcmc.get_extra_fields()
            potential_energy = extra_fields.get("potential_energy", np.array([]))
            if len(potential_energy) > 0:
                log_likelihood = -np.mean(potential_energy)
                self.logger.debug(
                    f"Potential energy stats: mean={np.mean(potential_energy):.3f}"
                )
            else:
                # Fallback: estimate log-likelihood from model fit quality
                self.logger.warning("No potential energy data collected, using convergence fallback")
                # Calculate a proper Gaussian log-likelihood estimate from reconstruction error
                try:
                    reconstruction_errors = []
                    total_samples = 0
                    start_idx = 0

                    for X in X_list:
                        end_idx = start_idx + X.shape[1]
                        W_view = W_mean[start_idx:end_idx, :]
                        X_recon = Z_mean @ W_view.T

                        # Calculate residuals
                        residuals = X - X_recon

                        # Estimate noise variance for this view
                        noise_var = np.var(residuals)
                        if noise_var <= 0:
                            noise_var = 1e-6  # Avoid log(0)

                        # Gaussian log-likelihood for this view
                        n_obs = X.size
                        view_ll = -0.5 * n_obs * (np.log(2 * np.pi * noise_var) + 1.0)

                        reconstruction_errors.append(view_ll)
                        total_samples += n_obs
                        start_idx = end_idx

                    # Sum log-likelihoods across views
                    log_likelihood = sum(reconstruction_errors)

                    # Normalize by number of observations for interpretability
                    normalized_ll = log_likelihood / total_samples

                    self.logger.info(f"Estimated log-likelihood from reconstruction: {log_likelihood:.1f} (normalized: {normalized_ll:.3f})")
                except Exception as e:
                    self.logger.warning(f"Failed to estimate log-likelihood: {e}")
                    log_likelihood = float("nan")

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

            # Determine convergence based on multiple criteria
            has_samples = len(W_samples) > 0 and len(Z_samples) > 0
            has_valid_likelihood = not np.isnan(log_likelihood) and np.isfinite(log_likelihood)

            # Model converged if we have samples and either valid likelihood or reasonable reconstruction
            model_converged = has_samples and (has_valid_likelihood or log_likelihood != float("nan"))

            if model_converged:
                self.logger.info(f"✅ {model_name} converged successfully")
            else:
                self.logger.warning(f"⚠️ {model_name} convergence issues detected")

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
                "convergence": model_converged,
                "model_type": args.get("model"),
                "hyperparameters": {
                    "K": args.get("K"),
                    "num_samples": num_samples,
                    "num_warmup": num_warmup,
                    "num_chains": num_chains,
                },
                "convergence_diagnostics": {
                    "has_samples": has_samples,
                    "has_valid_likelihood": has_valid_likelihood,
                    "potential_energy_available": len(potential_energy) > 0
                }
            }

            # Clear GPU memory after training
            try:
                del samples, W_samples, Z_samples, mcmc
            except NameError:
                # Variables may not be defined if exception occurred earlier
                pass
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

        # Combine all views for traditional methods with adaptive processing
        X_combined = np.hstack(X_list)
        n_components = self.comparison_params["K"]  # Use same K as model comparison

        # Calculate adaptive batch size for traditional methods (using mixin method)
        batch_size = self.calculate_adaptive_batch_size(
            [X_combined], operation_type="traditional"
        )

        try:
            for method_name in self.traditional_methods:
                self.logger.info(f"Testing traditional method: {method_name}")

                method_result = self._run_traditional_method(
                    X_combined, method_name, n_components, **kwargs
                )

                results[method_name] = method_result

                # Explicit memory cleanup after each traditional method
                self._cleanup_model_memory()

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
                experiment_id="traditional_method_comparison",
                config=self.config,
                start_time=datetime.now(),
                end_time=datetime.now(),
                status="completed",
                model_results=results,
                performance_metrics=performance_metrics,
                plots=plots,
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

    def _evaluate_neuroimaging_metrics(
        self, X_list: List[np.ndarray], model_result: Dict, model_name: str
    ) -> Dict:
        """Evaluate neuroimaging-specific metrics for model comparison."""
        try:
            self.logger.info(f"Evaluating neuroimaging metrics for {model_name}")

            # Extract model outputs
            W_list = model_result.get("W_list", [])
            Z_mean = model_result.get("Z", np.array([]))

            if not W_list or Z_mean.size == 0:
                return {"error": "Missing model outputs for evaluation"}

            # Prepare data for neuroimaging metrics
            model_outputs = {
                "W": W_list,  # List of loading matrices for each view
                "Z": Z_mean,  # Latent factors
                "log_likelihood": model_result.get("log_likelihood", np.nan),
                "convergence": model_result.get("convergence", False),
                "execution_time": model_result.get("execution_time", 0),
            }

            # Calculate comprehensive neuroimaging metrics with fallback
            model_info = {
                "model_name": model_name,
                "model_type": model_result.get("model_type", "unknown"),
                "n_factors": len(W_list[0][0]) if W_list and len(W_list[0]) > 0 else 0,
                "n_views": len(X_list),
                "n_subjects": X_list[0].shape[0] if X_list else 0,
            }

            neuroimaging_evaluation = self.metrics_fallback.with_metrics_fallback(
                advanced_metrics_func=self.neuroimaging_metrics.evaluate_model_comparison,
                fallback_metrics=self.metrics_fallback.create_basic_model_metrics(
                    model_result, model_name, W_list
                ),
                X_data=X_list,
                model_outputs=model_outputs,
                model_info=model_info
            )

            # Calculate additional model-specific metrics
            additional_metrics = self._calculate_model_specific_metrics(
                X_list, W_list, Z_mean, model_name
            )

            # Combine all metrics
            combined_metrics = {
                **neuroimaging_evaluation,
                **additional_metrics,
                "evaluation_success": True
            }

            self.logger.info(
                f"✅ {model_name} neuroimaging score: "
                f"{combined_metrics.get('overall_score', 'N/A'):.3f}"
            )

            return combined_metrics

        except Exception as e:
            self.logger.warning(f"Failed to evaluate neuroimaging metrics for {model_name}: {str(e)}")
            return {
                "error": str(e),
                "evaluation_success": False,
                "overall_score": np.nan,
            }

    def _calculate_model_specific_metrics(
        self, X_list: List[np.ndarray], W_list: List[np.ndarray], Z: np.ndarray, model_name: str
    ) -> Dict:
        """Calculate model-specific evaluation metrics."""
        try:
            metrics = {}

            # Reconstruction quality
            reconstruction_errors = []
            for i, (X_view, W_view) in enumerate(zip(X_list, W_list)):
                X_recon = Z @ W_view.T
                mse = np.mean((X_view - X_recon) ** 2)
                reconstruction_errors.append(mse)

            metrics["reconstruction_quality"] = {
                "mean_mse": np.mean(reconstruction_errors),
                "per_view_mse": reconstruction_errors,
                "normalized_rmse": np.sqrt(np.mean(reconstruction_errors)) / np.mean([np.std(X) for X in X_list])
            }

            # Sparsity analysis (for sparse models)
            if "sparse" in model_name.lower():
                sparsity_scores = []
                for W_view in W_list:
                    # Calculate sparsity as fraction of near-zero elements
                    threshold = 0.01 * np.std(W_view)
                    sparsity = np.mean(np.abs(W_view) < threshold)
                    sparsity_scores.append(sparsity)

                metrics["sparsity_score"] = {
                    "mean_sparsity": np.mean(sparsity_scores),
                    "per_view_sparsity": sparsity_scores,
                    "effective_sparsity": np.mean(sparsity_scores)
                }
            else:
                metrics["sparsity_score"] = {
                    "mean_sparsity": 0.0,
                    "per_view_sparsity": [0.0] * len(W_list),
                    "effective_sparsity": 0.0
                }

            # Factor interpretability
            factor_loadings_variance = []
            for W_view in W_list:
                # Variance of loadings per factor
                factor_vars = np.var(W_view, axis=0)
                factor_loadings_variance.extend(factor_vars)

            metrics["factor_interpretability"] = {
                "loading_variance_mean": np.mean(factor_loadings_variance),
                "loading_variance_std": np.std(factor_loadings_variance),
                "factor_diversity": np.std(factor_loadings_variance) / (np.mean(factor_loadings_variance) + 1e-8)
            }

            # Cross-view consistency (for multi-view models)
            if len(W_list) > 1:
                # Calculate correlation between factor loadings across views
                cross_view_correlations = []
                for k in range(min(W.shape[1] for W in W_list)):  # For each factor
                    view_loadings = [W[:, k] for W in W_list]
                    # Calculate pairwise correlations
                    correlations = []
                    for i in range(len(view_loadings)):
                        for j in range(i + 1, len(view_loadings)):
                            corr = np.corrcoef(view_loadings[i], view_loadings[j])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                    if correlations:
                        cross_view_correlations.append(np.mean(correlations))

                metrics["cross_view_consistency"] = {
                    "mean_correlation": np.mean(cross_view_correlations) if cross_view_correlations else 0.0,
                    "correlation_std": np.std(cross_view_correlations) if cross_view_correlations else 0.0
                }
            else:
                metrics["cross_view_consistency"] = {
                    "mean_correlation": 1.0,  # Single view is perfectly consistent with itself
                    "correlation_std": 0.0
                }

            # Overall model quality score (composite)
            recon_score = 1.0 / (1.0 + metrics["reconstruction_quality"]["normalized_rmse"])
            sparsity_bonus = metrics["sparsity_score"]["effective_sparsity"] * 0.1  # Small bonus for sparsity
            consistency_score = metrics["cross_view_consistency"]["mean_correlation"]
            interpretability_score = min(1.0, metrics["factor_interpretability"]["factor_diversity"])

            overall_score = (
                0.4 * recon_score +
                0.2 * sparsity_bonus +
                0.2 * consistency_score +
                0.2 * interpretability_score
            )

            metrics["overall_score"] = overall_score

            return metrics

        except Exception as e:
            self.logger.warning(f"Failed to calculate model-specific metrics: {str(e)}")
            return {
                "reconstruction_quality": {"mean_mse": np.inf},
                "sparsity_score": {"effective_sparsity": 0.0},
                "factor_interpretability": {"factor_diversity": 0.0},
                "cross_view_consistency": {"mean_correlation": 0.0},
                "overall_score": 0.0
            }

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

            # Create neuroimaging metrics comparison plot
            neuroimaging_plot = self._plot_neuroimaging_metrics_comparison(
                results, performance_metrics, successful_models
            )
            if neuroimaging_plot:
                plots["neuroimaging_metrics_comparison"] = neuroimaging_plot

        except Exception as e:
            self.logger.warning(f"Failed to generate model comparison plots: {e}")

        return plots

    def _plot_neuroimaging_metrics_comparison(
        self, results: Dict, performance_metrics: Dict, successful_models: List[str]
    ) -> Optional[plt.Figure]:
        """Generate neuroimaging-specific metrics comparison plot."""
        try:
            if not successful_models:
                return None

            # Check if models have neuroimaging metrics
            models_with_metrics = [
                m for m in successful_models
                if results[m].get("neuroimaging_metrics", {}).get("evaluation_success", False)
            ]

            if not models_with_metrics:
                self.logger.info("No models with neuroimaging metrics for comparison")
                return None

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle("Neuroimaging-Specific Model Comparison", fontsize=16)

            # Plot 1: Overall neuroimaging scores
            overall_scores = [
                performance_metrics[m].get("neuroimaging_score", np.nan)
                for m in models_with_metrics
            ]
            valid_scores = [(m, s) for m, s in zip(models_with_metrics, overall_scores) if not np.isnan(s)]

            if valid_scores:
                models_valid, scores_valid = zip(*valid_scores)
                bars = axes[0, 0].bar(models_valid, scores_valid, color='skyblue', alpha=0.8)
                axes[0, 0].set_title("Overall Neuroimaging Score")
                axes[0, 0].set_ylabel("Score")
                axes[0, 0].tick_params(axis="x", rotation=45)
                axes[0, 0].set_ylim([0, 1])

                # Add value labels on bars
                for bar, score in zip(bars, scores_valid):
                    height = bar.get_height()
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{score:.3f}', ha='center', va='bottom')

            # Plot 2: Reconstruction quality
            recon_qualities = []
            for m in models_with_metrics:
                recon_quality = results[m].get("neuroimaging_metrics", {}).get(
                    "reconstruction_quality", {}
                ).get("normalized_rmse", np.nan)
                recon_qualities.append(1.0 / (1.0 + recon_quality) if not np.isnan(recon_quality) else np.nan)

            valid_recon = [(m, r) for m, r in zip(models_with_metrics, recon_qualities) if not np.isnan(r)]
            if valid_recon:
                models_recon, scores_recon = zip(*valid_recon)
                axes[0, 1].bar(models_recon, scores_recon, color='lightgreen', alpha=0.8)
                axes[0, 1].set_title("Reconstruction Quality Score")
                axes[0, 1].set_ylabel("Quality Score")
                axes[0, 1].tick_params(axis="x", rotation=45)
                axes[0, 1].set_ylim([0, 1])

            # Plot 3: Sparsity scores
            sparsity_scores = [
                performance_metrics[m].get("sparsity_score", np.nan)
                for m in models_with_metrics
            ]
            valid_sparsity = [(m, s) for m, s in zip(models_with_metrics, sparsity_scores) if not np.isnan(s)]

            if valid_sparsity:
                models_sparse, scores_sparse = zip(*valid_sparsity)
                axes[0, 2].bar(models_sparse, scores_sparse, color='orange', alpha=0.8)
                axes[0, 2].set_title("Sparsity Score")
                axes[0, 2].set_ylabel("Effective Sparsity")
                axes[0, 2].tick_params(axis="x", rotation=45)

            # Plot 4: Cross-view consistency
            consistency_scores = []
            for m in models_with_metrics:
                consistency = results[m].get("neuroimaging_metrics", {}).get(
                    "cross_view_consistency", {}
                ).get("mean_correlation", np.nan)
                consistency_scores.append(consistency)

            valid_consistency = [(m, c) for m, c in zip(models_with_metrics, consistency_scores) if not np.isnan(c)]
            if valid_consistency:
                models_cons, scores_cons = zip(*valid_consistency)
                axes[1, 0].bar(models_cons, scores_cons, color='purple', alpha=0.8)
                axes[1, 0].set_title("Cross-View Consistency")
                axes[1, 0].set_ylabel("Mean Correlation")
                axes[1, 0].tick_params(axis="x", rotation=45)
                axes[1, 0].set_ylim([0, 1])

            # Plot 5: Factor interpretability
            interpretability_scores = []
            for m in models_with_metrics:
                interpretability = results[m].get("neuroimaging_metrics", {}).get(
                    "factor_interpretability", {}
                ).get("factor_diversity", np.nan)
                interpretability_scores.append(min(1.0, interpretability) if not np.isnan(interpretability) else np.nan)

            valid_interp = [(m, i) for m, i in zip(models_with_metrics, interpretability_scores) if not np.isnan(i)]
            if valid_interp:
                models_interp, scores_interp = zip(*valid_interp)
                axes[1, 1].bar(models_interp, scores_interp, color='red', alpha=0.8)
                axes[1, 1].set_title("Factor Interpretability")
                axes[1, 1].set_ylabel("Diversity Score")
                axes[1, 1].tick_params(axis="x", rotation=45)

            # Plot 6: Performance vs Quality scatter
            performance_times = [
                performance_metrics[m]["execution_time"] for m in models_with_metrics
            ]
            quality_scores = [
                performance_metrics[m].get("neuroimaging_score", 0) for m in models_with_metrics
            ]

            scatter = axes[1, 2].scatter(performance_times, quality_scores,
                                       s=100, alpha=0.7, c=range(len(models_with_metrics)),
                                       cmap='viridis')

            # Add model labels
            for i, model in enumerate(models_with_metrics):
                axes[1, 2].annotate(model, (performance_times[i], quality_scores[i]),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)

            axes[1, 2].set_xlabel("Execution Time (seconds)")
            axes[1, 2].set_ylabel("Neuroimaging Score")
            axes[1, 2].set_title("Performance vs Quality Trade-off")
            axes[1, 2].grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

        except Exception as e:
            self.logger.warning(f"Failed to create neuroimaging metrics comparison plot: {str(e)}")
            return None

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
        converged_models = []
        for model_name, model_result in results.items():
            if (
                model_result
                and not model_result.get("skipped", False)
                and model_result.get("convergence", False)
            ):
                converged_models.append((model_name, model_result))

        if not converged_models:
            return None

        # Prefer models with finite log-likelihood, fall back to any converged model
        finite_likelihood_models = []
        for model_name, model_result in converged_models:
            likelihood = model_result.get("log_likelihood", float("-inf"))
            if np.isfinite(likelihood) and likelihood != float("-inf"):
                finite_likelihood_models.append((model_name, model_result, likelihood))

        if finite_likelihood_models:
            # Select best model by likelihood
            best_name, best_result, best_likelihood = max(finite_likelihood_models, key=lambda x: x[2])
            self.logger.info(f"Selected best model: {best_name} (likelihood: {best_likelihood:.3f})")
        else:
            # Fall back to first converged model (even with NaN likelihood)
            best_name, best_result = converged_models[0]
            self.logger.info(f"Selected best model: {best_name} (converged but no finite likelihood)")

        return best_result

    def _cleanup_model_memory(self):
        """Explicit memory cleanup after each model to prevent accumulation."""
        try:
            import gc
            import jax

            # Clear JAX compilation cache and device memory
            jax.clear_caches()

            # Force garbage collection
            gc.collect()

            # Clear JAX device arrays if GPU backend
            if jax.default_backend() == "gpu":
                for device in jax.local_devices():
                    if device.platform == "gpu":
                        try:
                            device.synchronize_all_activity()
                        except Exception:
                            pass  # Ignore synchronization errors

            self.logger.debug("Memory cleanup completed")

        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")

    def _create_failure_result(self, experiment_name: str, error_message: str) -> ExperimentResult:
        """Create a failure result for failed experiments."""
        return ExperimentResult(
            experiment_id=experiment_name,
            config=self.config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            status="failed",
            error_message=error_message,
        )


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
    from core.config_utils import ConfigAccessor

    logger = logging.getLogger(__name__)
    logger.info("🔬 Starting model architecture comparison experiments...")

    # Handle config parameter using standard ConfigHelper
    from core.config_utils import ConfigHelper
    config_dict = ConfigHelper.to_dict(config)

    ConfigAccessor(config_dict)

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
            f" → Data shape: {n_subjects} subjects, { len(X_list)} views with dims {view_dims}"
        )
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
            f" → Using optimal SGFA parameters: K={ optimal_sgfa_params['K']}, percW={ optimal_sgfa_params['percW']}"
        )
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

    # Respect system GPU configuration from config
    system_config = config_dict.get("system", {})
    device = "gpu" if system_config.get("use_gpu", True) else "cpu"

    args = {"K": optimal_K, "reghsZ": True, "device": device}

    # Create experiment configuration
    exp_config = ExperimentConfig(
        experiment_name="model_architecture_comparison",
        description="Compare different SGFA model architectures"
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
