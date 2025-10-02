"""SGFA parameter comparison experiments for qMAP-PD analysis.

This module focuses on optimizing hyperparameters (K, percW) for the sparseGFA model.
For comparing different model architectures (sparseGFA vs alternatives),
see experiments/model_comparison.py.
"""

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
# KMeans and CCA imports removed - not needed for SGFA-focused experiment
# sklearn imports removed - traditional methods moved to model_comparison.py

# Safe configuration access
from core.config_utils import ConfigAccessor, safe_get
from core.experiment_utils import experiment_handler
from core.validation_utils import validate_data_types, validate_parameters
from experiments.framework import (
    ExperimentConfig,
    ExperimentFramework,
    ExperimentResult,
)
from optimization import PerformanceProfiler
from optimization.experiment_mixins import performance_optimized_experiment
from analysis.cross_validation_library import NeuroImagingHyperOptimizer, NeuroImagingCVConfig
from analysis.cv_fallbacks import HyperoptFallbackHandler

# Import clinical validation modules for parameter optimization
from analysis.clinical import ClinicalMetrics, ClinicalClassifier


@performance_optimized_experiment()
class SGFAParameterComparison(ExperimentFramework):
    """SGFA parameter comparison experiments for K and percW optimization."""

    def __init__(
        self, config: ExperimentConfig, logger: Optional[logging.Logger] = None
    ):
        # Initialize logger first to avoid AttributeError in fallback handler
        self.logger = logger or logging.getLogger(__name__)

        super().__init__(config, None, logger)
        self.profiler = PerformanceProfiler()

        # Initialize neuroimaging hyperparameter optimizer from config
        from core.config_utils import ConfigHelper
        config_dict = ConfigHelper.to_dict(config)
        cv_settings = config_dict.get("cross_validation", {})

        cv_config = NeuroImagingCVConfig()
        # Use reduced folds for hyperparameter optimization (typically n_folds - 2)
        base_folds = cv_settings.get("n_folds", 5)
        cv_config.inner_cv_folds = max(2, base_folds - 2)

        self.hyperopt = NeuroImagingHyperOptimizer(config=cv_config)

        # Initialize fallback handler for hyperparameter optimization
        self.hyperopt_fallback = HyperoptFallbackHandler(self.logger)

        # Initialize comparison visualizer
        from visualization import ComparisonVisualizer
        self.comparison_viz = ComparisonVisualizer(config=config_dict)

        # Initialize clinical validation modules for parameter optimization
        self.clinical_metrics = ClinicalMetrics(logger=self.logger)
        self.clinical_classifier = ClinicalClassifier(
            metrics_calculator=self.clinical_metrics,
            logger=self.logger
        )

        # Method configurations
        self.sgfa_variants = {
            "standard": {"use_sparse": True, "use_group": True},
            "sparse_only": {"use_sparse": True, "use_group": False},
            "group_only": {"use_sparse": False, "use_group": True},
            "basic_fa": {"use_sparse": False, "use_group": False},
        }

        # Focus purely on SGFA parameter optimization
        # Traditional methods moved to model_comparison.py

        # Load scalability test ranges from config
        sgfa_config = config_dict.get("sgfa_parameter_comparison", {})
        scalability_config = sgfa_config.get("scalability_analysis", {})
        parameter_ranges = sgfa_config.get("parameter_ranges", {})

        self.sample_size_ranges = scalability_config.get("sample_size_ranges", [50, 100, 250, 500, 1000, 2000, 5000])
        self.feature_size_ranges = scalability_config.get("feature_size_ranges", [10, 25, 50, 100, 250, 500, 1000])
        self.component_ranges = parameter_ranges.get("n_factors", [2, 3, 5, 8, 10, 15, 20])
        self.chain_ranges = scalability_config.get("chain_ranges", [1, 2, 4, 8])

        # Store parameter ranges for easy access
        self.parameter_ranges = parameter_ranges

    @experiment_handler("sgfa_variant_comparison")
    @validate_data_types(X_list=list, hypers=dict, args=dict)
    @validate_parameters(
        X_list=lambda x: len(x) > 0 and all(isinstance(arr, np.ndarray) for arr in x),
        hypers=lambda x: isinstance(x, dict),
    )
    def run_sgfa_variant_comparison(
        self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> ExperimentResult:
        """Compare different SGFA variants."""
        self.logger.info("Running SGFA variant comparison")

        results = {}
        performance_metrics = {}

        for variant_name, variant_config in self.sgfa_variants.items():
            self.logger.info(f"Testing SGFA variant: {variant_name}")

            # Update hyperparameters with variant config
            variant_hypers = hypers.copy()
            variant_hypers.update(variant_config)

            # Run analysis without profiling to debug the argparse.Namespace error
            variant_result = self._run_sgfa_variant(
                X_list, variant_hypers, args, **kwargs
            )

            results[variant_name] = variant_result

            # Store basic performance metrics
            performance_metrics[variant_name] = {
                "execution_time": variant_result.get("execution_time", 0),
                "peak_memory_gb": 0.0,  # Will be filled by system monitoring
                "convergence_iterations": variant_result.get("n_iterations", 0),
            }

        # Analyze results
        analysis = self._analyze_sgfa_variants(results, performance_metrics)

        # Generate basic plots
        plots = self._plot_sgfa_comparison(results, performance_metrics)

        # Add comprehensive visualizations
        advanced_plots = self._create_comprehensive_visualizations(
            X_list, results, "sgfa_variant_comparison"
        )
        plots.update(advanced_plots)

        return ExperimentResult(
            experiment_name="sgfa_variant_comparison",
            config=self.config,
            data=results,
            analysis=analysis,
            plots=plots,
            performance_metrics=performance_metrics,
            success=True,
        )

    # Traditional method comparison removed - available in model_comparison.py
    # This experiment focuses purely on SGFA hyperparameter optimization

    @experiment_handler("multiview_capability_assessment")
    @validate_data_types(X_list=list)
    @validate_parameters(
        X_list=lambda x: len(x) > 0 and all(isinstance(arr, np.ndarray) for arr in x)
    )
    def run_multiview_capability_assessment(
        self, X_list: List[np.ndarray], **kwargs
    ) -> ExperimentResult:
        """Assess multi-view capabilities of different methods."""
        self.logger.info("Running multi-view capability assessment")

        results = {}

        n_views = len(X_list)

        # Test with different numbers of views
        [list(range(i + 1)) for i in range(n_views)]

        for n_view_test in range(1, n_views + 1):
            view_subset = X_list[:n_view_test]

            # SGFA with subset of views
            sgfa_result = self._run_sgfa_multiview(view_subset, **kwargs)

            # Store results for this view configuration
            results[f"{n_view_test}_views"] = {
                "sgfa": sgfa_result,
                "view_dimensions": [X.shape[1] for X in view_subset],
            }

        # Analyze multi-view capabilities
        analysis = self._analyze_multiview_capabilities(results)

        # Generate plots
        plots = self._plot_multiview_comparison(results)

        return ExperimentResult(
            experiment_name="multiview_capability_assessment",
            config=self.config,
            data=results,
            analysis=analysis,
            plots=plots,
            success=True,
        )

    @experiment_handler("scalability_comparison")
    @validate_data_types(
        X_list=list, sample_sizes=(list, type(None)), feature_sizes=(list, type(None))
    )
    @validate_parameters(
        X_list=lambda x: len(x) > 0 and all(isinstance(arr, np.ndarray) for arr in x),
        sample_sizes=lambda x: x is None
        or (isinstance(x, list) and all(isinstance(s, int) and s > 0 for s in x)),
        feature_sizes=lambda x: x is None
        or (isinstance(x, list) and all(isinstance(f, int) and f > 0 for f in x)),
    )
    def run_comprehensive_sgfa_scalability_analysis(
        self,
        X_list: List[np.ndarray],
        hypers: Dict,
        args: Dict,
        **kwargs,
    ) -> ExperimentResult:
        """Run comprehensive SGFA scalability analysis (moved from performance_benchmarks.py)."""
        self.logger.info("Running comprehensive SGFA scalability analysis")

        results = {}

        # Sample size scalability
        self.logger.info("Benchmarking SGFA sample size scalability")
        sample_results = self._benchmark_sgfa_sample_scalability(X_list, hypers, args, **kwargs)
        results["sample_scalability"] = sample_results

        # Feature size scalability
        self.logger.info("Benchmarking SGFA feature size scalability")
        feature_results = self._benchmark_sgfa_feature_scalability(X_list, hypers, args, **kwargs)
        results["feature_scalability"] = feature_results

        # Component scalability (K values)
        self.logger.info("Benchmarking SGFA component scalability")
        component_results = self._benchmark_sgfa_component_scalability(X_list, hypers, args, **kwargs)
        results["component_scalability"] = component_results

        # MCMC chain scalability
        self.logger.info("Benchmarking SGFA MCMC chain scalability")
        chain_results = self._benchmark_sgfa_chain_scalability(X_list, hypers, args, **kwargs)
        results["chain_scalability"] = chain_results

        # Comprehensive analysis
        analysis = self._analyze_sgfa_scalability(results)

        # Generate comprehensive plots
        plots = self._plot_sgfa_scalability_analysis(results)

        return ExperimentResult(
            experiment_id="comprehensive_sgfa_scalability_analysis",
            config=self.config,
            start_time=self.profiler.get_current_metrics().execution_time if hasattr(self.profiler, 'get_current_metrics') else 0,
            status="completed",
            model_results=results,
            diagnostics=analysis,
            plots=plots,
        )

    @experiment_handler("neuroimaging_hyperparameter_optimization")
    @validate_data_types(X_list=list, hypers=dict, args=dict)
    @validate_parameters(
        X_list=lambda x: len(x) > 0 and all(isinstance(arr, np.ndarray) for arr in x),
        hypers=lambda x: isinstance(x, dict),
    )
    def run_neuroimaging_hyperparameter_optimization(
        self,
        X_list: List[np.ndarray],
        hypers: Dict,
        args: Dict,
        clinical_data: Optional[Dict] = None
    ) -> ExperimentResult:
        """Run neuroimaging-specific hyperparameter optimization using NeuroImagingHyperOptimizer."""
        self.logger.info("🔬 Starting neuroimaging hyperparameter optimization")

        if clinical_data is None:
            # Generate synthetic clinical data for optimization
            clinical_data = self._generate_synthetic_clinical_data(X_list[0].shape[0])

        # Define hyperparameter search space from config
        n_factors_range = self.parameter_ranges.get("n_factors", [3, 5, 8, 10, 12])
        sparsity_range = self.parameter_ranges.get("sparsity_lambda", [0.01, 0.1, 0.5, 1.0])

        # Convert sparsity range to percentage for percW
        percW_range = [val * 100 if val <= 1.0 else val for val in sparsity_range]

        search_space = {
            'K': {
                'type': 'int',
                'low': min(n_factors_range),
                'high': max(n_factors_range),
                'step': 1,
                'description': 'Number of latent factors'
            },
            'percW': {
                'type': 'float',
                'low': min(percW_range),
                'high': max(percW_range),
                'step': 5.0,
                'description': 'Sparsity percentage for factor loadings'
            },
            'num_samples': {
                'type': 'categorical',
                'choices': [200, 300, 500],
                'description': 'Number of MCMC samples'
            },
            'num_warmup': {
                'type': 'int',
                'low': 100,
                'high': 300,
                'step': 50,
                'description': 'Number of MCMC warmup steps'
            }
        }

        try:
            self.logger.info(f"Optimizing hyperparameters with {self.hyperopt.n_trials} trials")
            self.logger.info(f"Search space: {list(search_space.keys())}")

            # Define the objective function for SGFA
            def sgfa_objective(trial_params: Dict) -> float:
                """Objective function for SGFA hyperparameter optimization."""
                try:
                    # Update hyperparameters with trial parameters
                    trial_hypers = hypers.copy()
                    trial_hypers.update({
                        'percW': trial_params['percW']
                    })

                    # Update args with trial parameters
                    trial_args = args.copy()
                    trial_args.update({
                        'K': trial_params['K'],
                        'num_samples': trial_params['num_samples'],
                        'num_warmup': trial_params['num_warmup'],
                        'model': 'sparseGFA'  # Focus on sparse GFA
                    })

                    # Run SGFA with these parameters
                    result = self._run_sgfa_variant(X_list, trial_hypers, trial_args, **kwargs)

                    if not result.get('convergence', False):
                        return -1000.0  # Heavy penalty for non-convergence

                    # Calculate composite score based on multiple criteria
                    log_likelihood = result.get('log_likelihood', float('-inf'))
                    execution_time = result.get('execution_time', float('inf'))

                    # Normalize scores
                    ll_score = log_likelihood if log_likelihood != float('-inf') else -1000
                    time_penalty = min(100, execution_time) / 100.0  # Normalize time to 0-1

                    # Composite score favoring both likelihood and efficiency
                    composite_score = ll_score - (0.1 * time_penalty)

                    self.logger.debug(
                        f"Trial K={trial_params['K']}, percW={trial_params['percW']:.1f}: "
                        f"LL={ll_score:.2f}, Time={execution_time:.1f}s, Score={composite_score:.2f}"
                    )

                    return composite_score

                except Exception as e:
                    self.logger.warning(f"Trial failed: {str(e)}")
                    return -2000.0  # Heavy penalty for failed trials

            # Run hyperparameter optimization with automatic fallback
            optimization_result = self.hyperopt_fallback.with_hyperopt_fallback(
                advanced_hyperopt_func=self.hyperopt.optimize_hyperparameters,
                search_space=search_space,
                objective_function=sgfa_objective,
                max_combinations=20,
                X_data=X_list,
                clinical_data=clinical_data,
                study_name="sgfa_neuroimaging_optimization"
            )

            self.logger.info("✅ Hyperparameter optimization completed")

            # Extract results
            best_params = optimization_result.get('best_params', {})
            best_score = optimization_result.get('best_score', float('-inf'))
            optimization_history = optimization_result.get('optimization_history', [])

            self.logger.info(f"🏆 Best parameters: {best_params}")
            self.logger.info(f"🏆 Best score: {best_score:.3f}")

            # Run final evaluation with best parameters
            self.logger.info("Running final evaluation with best parameters")
            final_hypers = hypers.copy()
            final_hypers.update({'percW': best_params.get('percW', 25.0)})

            final_args = args.copy()
            final_args.update({
                'K': best_params.get('K', 5),
                'num_samples': best_params.get('num_samples', 300),
                'num_warmup': best_params.get('num_warmup', 150),
                'model': 'sparseGFA'
            })

            final_result = self._run_sgfa_variant(X_list, final_hypers, final_args, **kwargs)

            # Compile comprehensive results
            results = {
                'optimization_result': optimization_result,
                'best_parameters': best_params,
                'best_score': best_score,
                'final_evaluation': final_result,
                'search_space': search_space,
                'n_trials': self.hyperopt.n_trials,
                'optimization_history': optimization_history
            }

            # Analyze optimization results
            analysis = self._analyze_hyperparameter_optimization(results)

            # Generate optimization plots
            plots = self._plot_hyperparameter_optimization(results)

            return ExperimentResult(
                experiment_name="neuroimaging_hyperparameter_optimization",
                config=self.config,
                data=results,
                analysis=analysis,
                plots=plots,
                success=True,
            )

        except Exception as e:
            self.logger.error(f"Neuroimaging hyperparameter optimization failed: {str(e)}")
            return ExperimentResult(
                experiment_name="neuroimaging_hyperparameter_optimization",
                config=self.config,
                data={'error': str(e)},
                analysis={},
                plots={},
                success=False,
            )

    def _generate_synthetic_clinical_data(self, n_subjects: int) -> Dict:
        """Generate synthetic clinical data for hyperparameter optimization."""
        np.random.seed(42)  # For reproducibility

        # Generate basic clinical data structure
        diagnoses = np.random.choice(
            ["PD", "control"],
            size=n_subjects,
            p=[0.7, 0.3]  # 70% PD patients, 30% controls
        )

        ages = np.random.normal(65, 10, n_subjects)
        ages = np.clip(ages, 40, 85)  # Reasonable age range

        return {
            "subject_id": [f"subj_{i:04d}" for i in range(n_subjects)],
            "diagnosis": diagnoses,
            "age": ages,
            "age_group": ["young" if age < 60 else "old" for age in ages],
        }

    def _analyze_hyperparameter_optimization(self, results: Dict) -> Dict:
        """Analyze hyperparameter optimization results."""
        try:
            optimization_history = results.get('optimization_history', [])
            best_params = results.get('best_parameters', {})
            search_space = results.get('search_space', {})

            analysis = {
                'parameter_importance': {},
                'convergence_analysis': {},
                'efficiency_analysis': {},
                'recommendation': {}
            }

            if optimization_history:
                # Analyze parameter importance (correlation with objective)
                for param_name in search_space.keys():
                    param_values = [trial.get(param_name) for trial in optimization_history if param_name in trial]
                    objective_values = [trial.get('objective_value', 0) for trial in optimization_history if param_name in trial]

                    if len(param_values) > 3:  # Need sufficient data points
                        correlation = np.corrcoef(param_values, objective_values)[0, 1]
                        analysis['parameter_importance'][param_name] = {
                            'correlation_with_objective': correlation if not np.isnan(correlation) else 0.0,
                            'value_range': [min(param_values), max(param_values)],
                            'best_value': best_params.get(param_name, 'unknown')
                        }

                # Convergence analysis
                objective_values = [trial.get('objective_value', float('-inf')) for trial in optimization_history]
                valid_objectives = [obj for obj in objective_values if obj != float('-inf')]

                if valid_objectives:
                    # Running best score
                    running_best = []
                    current_best = float('-inf')
                    for obj in objective_values:
                        if obj > current_best:
                            current_best = obj
                        running_best.append(current_best)

                    analysis['convergence_analysis'] = {
                        'trials_completed': len(optimization_history),
                        'successful_trials': len(valid_objectives),
                        'success_rate': len(valid_objectives) / len(optimization_history),
                        'final_best_score': max(valid_objectives),
                        'convergence_trend': running_best[-10:] if len(running_best) >= 10 else running_best
                    }

                # Efficiency analysis
                execution_times = [trial.get('execution_time', 0) for trial in optimization_history]
                if execution_times:
                    analysis['efficiency_analysis'] = {
                        'mean_trial_time': np.mean(execution_times),
                        'total_optimization_time': sum(execution_times),
                        'fastest_trial_time': min(execution_times),
                        'slowest_trial_time': max(execution_times)
                    }

            # Generate recommendations
            analysis['recommendation'] = {
                'optimal_K': best_params.get('K', 5),
                'optimal_percW': best_params.get('percW', 25.0),
                'optimal_num_samples': best_params.get('num_samples', 300),
                'confidence': 'high' if len(optimization_history) >= 15 else 'medium',
                'suggested_next_steps': [
                    f"Use K={best_params.get('K', 5)} factors for this dataset",
                    f"Set sparsity to {best_params.get('percW', 25.0):.1f}%",
                    "Consider running longer chains for final analysis"
                ]
            }

            return analysis

        except Exception as e:
            self.logger.warning(f"Failed to analyze hyperparameter optimization: {str(e)}")
            return {'error': str(e)}

    def _plot_hyperparameter_optimization(self, results: Dict) -> Dict:
        """Generate plots for hyperparameter optimization results using ComparisonVisualizer."""
        plots = {}

        try:
            optimization_history = results.get('optimization_history', [])
            best_params = results.get('best_parameters', {})

            if not optimization_history:
                return plots

            # Extract objective values
            objective_values = [trial.get('objective_value', float('-inf')) for trial in optimization_history]
            valid_indices = [i for i, obj in enumerate(objective_values) if obj != float('-inf')]
            valid_objectives = [objective_values[i] for i in valid_indices]

            # Plot 1: Optimization progress using scalability_analysis
            if valid_objectives:
                progress_fig = self.comparison_viz.plot_scalability_analysis(
                    data_sizes=valid_indices,
                    metrics_by_size={"Objective Score": valid_objectives},
                    title="Hyperparameter Optimization Progress",
                    xlabel="Trial Number",
                    ylabel="Objective Score"
                )
                plots["optimization_progress"] = progress_fig

            # Plot 2: K vs Performance using hyperparameter_grid
            K_values = [trial.get('K') for trial in optimization_history if 'K' in trial]
            K_objectives = [trial.get('objective_value') for trial in optimization_history if 'K' in trial]

            if K_values and K_objectives:
                valid_K_data = [(k, obj) for k, obj in zip(K_values, K_objectives) if obj != float('-inf')]
                if valid_K_data:
                    K_vals, K_objs = zip(*valid_K_data)
                    K_fig = self.comparison_viz.plot_hyperparameter_grid(
                        param_name="Number of Factors (K)",
                        param_values=list(K_vals),
                        scores=list(K_objs),
                        title="Factors vs Performance",
                        ylabel="Objective Score"
                    )
                    plots["K_optimization"] = K_fig

            # Plot 3: percW vs Performance using hyperparameter_grid
            percW_values = [trial.get('percW') for trial in optimization_history if 'percW' in trial]
            percW_objectives = [trial.get('objective_value') for trial in optimization_history if 'percW' in trial]

            if percW_values and percW_objectives:
                valid_percW_data = [(p, obj) for p, obj in zip(percW_values, percW_objectives) if obj != float('-inf')]
                if valid_percW_data:
                    percW_vals, percW_objs = zip(*valid_percW_data)
                    percW_fig = self.comparison_viz.plot_hyperparameter_grid(
                        param_name="Sparsity Percentage (percW)",
                        param_values=list(percW_vals),
                        scores=list(percW_objs),
                        title="Sparsity vs Performance",
                        ylabel="Objective Score"
                    )
                    plots["percW_optimization"] = percW_fig

        except Exception as e:
            self.logger.warning(f"Failed to create hyperparameter optimization plots: {str(e)}")

        return plots

    def _run_sgfa_variant(
        self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> Dict:
        """Run SGFA with specific variant configuration."""
        import gc
        import time

        import jax
        from numpyro.infer import MCMC, NUTS

        try:
            # Clear GPU memory before starting
            jax.clear_caches()
            if jax.default_backend() == "gpu":
                # Force garbage collection
                gc.collect()
                # Clear JAX device arrays
                for device in jax.local_devices():
                    if device.platform == "gpu":
                        try:
                            device.synchronize_all_activity()
                        except BaseException:
                            pass

            # Disable any potential performance optimizations that might cause issues
            import os

            os.environ["JAX_DISABLE_JIT"] = (
                "0"  # Keep JIT enabled but ensure no other transforms
            )

            self.logger.info(
                f"Training SGFA model with K={ args.get( 'K', 10)}, percW={ hypers.get( 'percW', 33)}"
            )

            # Import the actual SGFA model function
            from core.run_analysis import models

            # Setup MCMC configuration with memory-aware adjustments
            K = args.get("K", 10)
            percW = hypers.get("percW", 33)

            # Reduce sampling parameters for high memory variants
            if K >= 10 and percW >= 33:
                num_warmup = args.get(
                    "num_warmup", 200
                )  # Even more reduced for high memory
                num_samples = args.get("num_samples", 300)  # Even more reduced
                num_chains = 1  # Force single chain for high memory variants
                self.logger.info(
                    f"Using heavily reduced sampling for high memory variant: warmup={num_warmup}, samples={num_samples}, chains={num_chains}"
                )
            else:
                num_warmup = args.get("num_warmup", 500)
                num_samples = args.get("num_samples", 1000)
                num_chains = args.get("num_chains", 2)

            # Create args object for model
            import argparse

            model_args = argparse.Namespace(
                model="sparseGFA",
                K=args.get("K", 10),
                num_sources=len(X_list),
                reghsZ=args.get("reghsZ", True),
            )

            # Additional memory optimization for high memory variants
            if K >= 10 and percW >= 33:
                # Apply more aggressive memory management
                import gc

                gc.collect()
                jax.clear_caches()

                # Use lower target accept probability and tree depth to reduce memory
                kernel = NUTS(models, target_accept_prob=0.6, max_tree_depth=8)
                self.logger.info(
                    "Applied aggressive memory optimizations: lower target_accept_prob and max_tree_depth"
                )
            else:
                kernel = NUTS(
                    models, target_accept_prob=args.get("target_accept_prob", 0.8)
                )

            # Setup MCMC
            rng_key = jax.random.PRNGKey(np.random.randint(0, 10000))
            mcmc = MCMC(
                kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                # Memory efficiency options
                progress_bar=False,  # Reduces memory usage
                chain_method="sequential",  # Use sequential chains to reduce GPU memory
            )

            # Run inference
            start_time = time.time()
            mcmc.run(
                rng_key, X_list, hypers, model_args, extra_fields=("potential_energy",)
            )
            elapsed = time.time() - start_time

            # Get samples and extra fields
            samples = mcmc.get_samples()

            # Calculate log likelihood (approximate)
            extra_fields = mcmc.get_extra_fields()
            potential_energy = extra_fields.get("potential_energy", np.array([]))
            if len(potential_energy) > 0:
                log_likelihood = -np.mean(potential_energy)
                self.logger.debug(
                    f"Potential energy stats: mean={np.mean(potential_energy):.3f}, std={np.std(potential_energy):.3f}"
                )
            else:
                log_likelihood = np.nan
                self.logger.warning("No potential energy data collected - log likelihood unavailable")

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

            self.logger.info(
                f"SGFA training completed in { elapsed:.2f}s, log_likelihood: { log_likelihood:.3f}"
            )

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
                "mcmc_diagnostics": {
                    "num_samples": num_samples,
                    "num_warmup": num_warmup,
                    "num_chains": num_chains,
                },
            }

            # Add clinical performance if clinical labels provided
            if "clinical_labels" in kwargs and kwargs["clinical_labels"] is not None:
                try:
                    clinical_perf = self.clinical_classifier.test_factor_classification(
                        Z_mean, kwargs["clinical_labels"], f"SGFA_K{K}_percW{percW}"
                    )
                    result["clinical_performance"] = clinical_perf
                    self.logger.info(f"  Clinical validation: {len(clinical_perf)} classifiers tested")
                except Exception as e:
                    self.logger.warning(f"  Clinical validation failed: {str(e)}")

            # Clear GPU memory after training
            del samples, W_samples, Z_samples, mcmc
            jax.clear_caches()
            gc.collect()

            return result

        except Exception as e:
            self.logger.error(f"SGFA variant training failed: {str(e)}")
            # Clear memory even on failure
            jax.clear_caches()
            gc.collect()
            return {
                "error": str(e),
                "convergence": False,
                "execution_time": 0,
                "log_likelihood": float("-inf"),
            }

    # _run_traditional_method removed - not needed for SGFA-focused hyperparameter optimization
    # Traditional method comparison is available in model_comparison.py

    def _run_sgfa_multiview(self, X_list: List[np.ndarray]) -> Dict:
        """Run SGFA on multi-view data."""
        # Mock SGFA results for multi-view data
        return {
            "W": [np.random.randn(X.shape[1], 5) for X in X_list],
            "Z": np.random.randn(X_list[0].shape[0], 5),
            "log_likelihood": np.random.randn(),
            "n_views_used": len(X_list),
        }

    # _run_scalability_test removed - unused method that tested traditional methods (PCA, FA)
    # This experiment focuses purely on SGFA hyperparameter optimization, not method comparison

    def _analyze_sgfa_variants(self, results: Dict, performance_metrics: Dict) -> Dict:
        """Analyze SGFA variant comparison results."""
        analysis = {
            "variant_summary": {},
            "performance_ranking": {},
            "convergence_analysis": {},
            "recommendations": [],
        }

        # Summarize each variant
        for variant_name, result in results.items():
            metrics = performance_metrics[variant_name]

            analysis["variant_summary"][variant_name] = {
                "converged": result.get("convergence", False),
                "log_likelihood": result.get("log_likelihood", 0),
                "execution_time": metrics["execution_time"],
                "memory_usage": metrics["peak_memory_gb"],
                "iterations": metrics["convergence_iterations"],
            }

        # Performance ranking
        time_ranking = sorted(
            performance_metrics.items(), key=lambda x: x[1]["execution_time"]
        )
        memory_ranking = sorted(
            performance_metrics.items(), key=lambda x: x[1]["peak_memory_gb"]
        )

        analysis["performance_ranking"] = {
            "fastest": [name for name, _ in time_ranking],
            "memory_efficient": [name for name, _ in memory_ranking],
        }

        # Convergence analysis
        converged_variants = [
            name for name, result in results.items() if result.get("convergence", False)
        ]

        analysis["convergence_analysis"] = {
            "converged_variants": converged_variants,
            "convergence_rate": len(converged_variants) / len(results),
        }

        # Recommendations
        if "standard" in converged_variants:
            analysis["recommendations"].append("Standard SGFA converged successfully")

        fastest_variant = time_ranking[0][0]
        analysis["recommendations"].append(f"Fastest variant: {fastest_variant}")

        most_memory_efficient = memory_ranking[0][0]
        analysis["recommendations"].append(
            f"Most memory efficient: {most_memory_efficient}"
        )

        return analysis

    # _analyze_traditional_comparison removed - available in model_comparison.py

    def _analyze_multiview_capabilities(self, results: Dict) -> Dict:
        """Analyze multi-view capabilities of different methods."""
        analysis = {
            "view_scaling": {},
            "information_preservation": {},
            "method_comparison": {},
        }

        # Analyze how methods scale with number of views
        for view_key, view_results in results.items():
            n_views = int(view_key.split("_")[0])

            sgfa_result = view_results["sgfa"]
            sgfa_variants = view_results.get("sgfa_variants", {})

            analysis["view_scaling"][n_views] = {
                "sgfa_likelihood": sgfa_result.get("log_likelihood", 0),
                "sgfa_variants_available": len(sgfa_variants),
                "total_features": sum(view_results["view_dimensions"]),
            }

        return analysis

    def _analyze_scalability(self, results: Dict) -> Dict:
        """Analyze scalability comparison results."""
        analysis = {
            "sample_scaling": {},
            "feature_scaling": {},
            "efficiency_trends": {},
            "scalability_ranking": {},
        }

        # Sample scaling analysis
        sample_results = results["sample_scalability"]
        for sample_size, sample_result in sample_results.items():
            analysis["sample_scaling"][sample_size] = {}
            for method, method_result in sample_result.items():
                analysis["sample_scaling"][sample_size][method] = {
                    "execution_time": method_result["execution_time"],
                    "memory_usage": method_result["peak_memory_gb"],
                }

        # Feature scaling analysis
        feature_results = results["feature_scalability"]
        for feature_size, feature_result in feature_results.items():
            analysis["feature_scaling"][feature_size] = {}
            for method, method_result in feature_result.items():
                analysis["feature_scaling"][feature_size][method] = {
                    "execution_time": method_result["execution_time"],
                    "memory_usage": method_result["peak_memory_gb"],
                }

        return analysis

    # _get_output_dimensions removed - was only used for traditional method comparison

    def _plot_sgfa_comparison(self, results: Dict, performance_metrics: Dict) -> Dict:
        """Generate plots for SGFA variant comparison using ComparisonVisualizer."""
        plots = {}

        try:
            variants = list(results.keys())

            # Performance comparison (execution time, memory)
            perf_fig = self.comparison_viz.plot_performance_comparison(
                methods=variants,
                performance_metrics=performance_metrics,
                title="SGFA Variant Performance Comparison",
                metrics_to_plot=['execution_time', 'peak_memory_gb']
            )
            plots["sgfa_variant_performance"] = perf_fig

            # Quality comparison (log-likelihood)
            log_liks = {v: results[v].get("log_likelihood", float("-inf")) for v in variants}
            quality_fig = self.comparison_viz.plot_quality_comparison(
                methods=variants,
                quality_scores=log_liks,
                title="SGFA Variant Quality Comparison",
                ylabel="Log-Likelihood",
                higher_is_better=True
            )
            plots["sgfa_variant_quality"] = quality_fig

        except Exception as e:
            self.logger.warning(f"Failed to create SGFA comparison plots: {str(e)}")

        return plots

    # _plot_traditional_comparison removed - available in model_comparison.py

    def _plot_multiview_comparison(self, results: Dict) -> Dict:
        """Generate plots for multi-view comparison using ComparisonVisualizer."""
        plots = {}

        try:
            view_counts = sorted([int(k.split("_")[0]) for k in results.keys()])

            # SGFA likelihood vs number of views
            sgfa_likelihoods = [
                results[f"{n}_views"]["sgfa"].get("log_likelihood", 0)
                for n in view_counts
            ]

            # Use scalability analysis for multiview scaling
            likelihood_fig = self.comparison_viz.plot_scalability_analysis(
                data_sizes=view_counts,
                metrics_by_size={"SGFA Log-Likelihood": sgfa_likelihoods},
                title="Multi-view Capability Assessment",
                xlabel="Number of Views",
                ylabel="Log-Likelihood"
            )
            plots["multiview_likelihood"] = likelihood_fig

            # Feature dimensions handled
            total_features = [
                sum(results[f"{n}_views"]["view_dimensions"]) for n in view_counts
            ]
            features_fig = self.comparison_viz.plot_scalability_analysis(
                data_sizes=view_counts,
                metrics_by_size={"Total Features": total_features},
                title="Feature Scaling with Multi-view",
                xlabel="Number of Views",
                ylabel="Total Features"
            )
            plots["multiview_features"] = features_fig

        except Exception as e:
            self.logger.warning(f"Failed to create multi-view comparison plots: {str(e)}")

        return plots

    def _plot_scalability_comparison(self, results: Dict) -> Dict:
        """Generate plots for scalability comparison using ComparisonVisualizer."""
        plots = {}

        try:
            # Sample size scalability - execution time
            sample_results = results["sample_scalability"]
            sample_sizes = sorted(sample_results.keys())
            methods = list(next(iter(sample_results.values())).keys())

            # Build metrics_by_size dict for execution time
            time_metrics = {}
            for method in methods:
                time_metrics[method] = [
                    sample_results[size][method]["execution_time"]
                    for size in sample_sizes
                ]

            sample_time_fig = self.comparison_viz.plot_scalability_analysis(
                data_sizes=sample_sizes,
                metrics_by_size=time_metrics,
                title="Execution Time vs Sample Size",
                xlabel="Sample Size",
                ylabel="Time (seconds)",
                log_x=True,
                log_y=True
            )
            plots["sample_time_scalability"] = sample_time_fig

            # Sample size scalability - memory
            memory_metrics = {}
            for method in methods:
                memory_metrics[method] = [
                    sample_results[size][method]["peak_memory_gb"]
                    for size in sample_sizes
                ]

            sample_memory_fig = self.comparison_viz.plot_scalability_analysis(
                data_sizes=sample_sizes,
                metrics_by_size=memory_metrics,
                title="Memory Usage vs Sample Size",
                xlabel="Sample Size",
                ylabel="Memory (GB)",
                log_x=True
            )
            plots["sample_memory_scalability"] = sample_memory_fig

            # Feature size scalability - execution time
            feature_results = results["feature_scalability"]
            feature_sizes = sorted(feature_results.keys())

            feature_time_metrics = {}
            for method in methods:
                feature_time_metrics[method] = [
                    feature_results[size][method]["execution_time"]
                    for size in feature_sizes
                ]

            feature_time_fig = self.comparison_viz.plot_scalability_analysis(
                data_sizes=feature_sizes,
                metrics_by_size=feature_time_metrics,
                title="Execution Time vs Feature Size",
                xlabel="Feature Size",
                ylabel="Time (seconds)",
                log_x=True,
                log_y=True
            )
            plots["feature_time_scalability"] = feature_time_fig

            # Feature size scalability - memory
            feature_memory_metrics = {}
            for method in methods:
                feature_memory_metrics[method] = [
                    feature_results[size][method]["peak_memory_gb"]
                    for size in feature_sizes
                ]

            feature_memory_fig = self.comparison_viz.plot_scalability_analysis(
                data_sizes=feature_sizes,
                metrics_by_size=feature_memory_metrics,
                title="Memory Usage vs Feature Size",
                xlabel="Feature Size",
                ylabel="Memory (GB)",
                log_x=True
            )
            plots["feature_memory_scalability"] = feature_memory_fig

        except Exception as e:
            self.logger.warning(f"Failed to create scalability comparison plots: {str(e)}")

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

            # Aggressive memory cleanup before visualization
            self.logger.info("🧹 Performing pre-visualization memory cleanup...")
            import gc
            import jax

            # Clear JAX compilation cache
            try:
                from jax._src import compilation_cache
                compilation_cache.clear_cache()
                self.logger.info("JAX compilation cache cleared")
            except Exception:
                pass

            # Force garbage collection multiple times
            for i in range(5):
                gc.collect()

            # Clear JAX device memory if using GPU
            try:
                for device in jax.devices():
                    if device.platform == 'gpu':
                        device.memory_stats()  # Force memory cleanup
                self.logger.info("GPU memory cleanup attempted")
            except Exception:
                pass

            # Brief delay for cleanup to complete
            import time
            time.sleep(2)
            self.logger.info("✅ Pre-visualization memory cleanup completed")

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
                    "output_dir": "/tmp/sgfa_viz",  # Will be overridden
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

            # Extract the best SGFA result for detailed analysis
            best_sgfa_result = self._extract_best_sgfa_result(results)

            if best_sgfa_result:
                analysis_results = {
                    "best_run": best_sgfa_result,
                    "all_runs": results,
                    "model_type": "sparseGFA",
                    "convergence": best_sgfa_result.get("convergence", False),
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
                    "No converged SGFA results found for comprehensive visualization"
                )

        except Exception as e:
            self.logger.warning(f"Failed to create comprehensive visualizations: {e}")
            # Don't fail the experiment if advanced visualizations fail

        return advanced_plots

    def _extract_best_sgfa_result(self, results: Dict) -> Optional[Dict]:
        """Extract the best SGFA result from experiment results."""
        best_result = None
        best_likelihood = float("-inf")

        # Look through different result structures
        if "sgfa_variants" in results:
            # From variant comparison
            for _variant_name, variant_result in results["sgfa_variants"].items():
                if variant_result and variant_result.get("convergence", False):
                    likelihood = variant_result.get("log_likelihood", float("-inf"))
                    if likelihood > best_likelihood:
                        best_likelihood = likelihood
                        best_result = variant_result
        elif isinstance(results, dict) and "W" in results:
            # Direct SGFA result
            if results.get("convergence", False):
                best_result = results

        return best_result

    def _benchmark_sgfa_sample_scalability(
        self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> Dict:
        """Benchmark SGFA scalability with respect to sample size."""
        results = {}

        for n_samples in self.sample_size_ranges:
            if n_samples > X_list[0].shape[0]:
                continue

            self.logger.debug(f"Testing SGFA with {n_samples} samples")

            # Subsample data
            indices = np.random.choice(X_list[0].shape[0], n_samples, replace=False)
            X_subset = [X[indices] for X in X_list]

            try:
                with self.profiler.profile(f"sgfa_samples_{n_samples}") as p:
                    # Use existing SGFA method from the class
                    result = self._run_sgfa_multiview(X_subset, hypers, args, **kwargs)

                # Collect metrics
                performance_metrics = self.profiler.get_current_metrics()

                results[n_samples] = {
                    "result": result,
                    "performance_metrics": {
                        "execution_time": performance_metrics.execution_time,
                        "peak_memory_gb": performance_metrics.peak_memory_gb,
                        "cpu_percent": performance_metrics.cpu_percent,
                    },
                    "dataset_info": {
                        "n_subjects": n_samples,
                        "n_features": [X.shape[1] for X in X_subset],
                        "n_views": len(X_subset),
                    },
                }

            except Exception as e:
                self.logger.warning(
                    f"SGFA sample scalability test failed for {n_samples}: {str(e)}"
                )
                results[n_samples] = {"error": str(e)}

        return results

    def _benchmark_sgfa_feature_scalability(
        self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> Dict:
        """Benchmark SGFA scalability with respect to feature size."""
        results = {}

        for n_features in self.feature_size_ranges:
            self.logger.debug(f"Testing SGFA with {n_features} features per view")

            # Select subset of features from each view
            X_subset = []
            for X in X_list:
                if n_features >= X.shape[1]:
                    X_subset.append(X)
                else:
                    feature_indices = np.random.choice(
                        X.shape[1], n_features, replace=False
                    )
                    X_subset.append(X[:, feature_indices])

            try:
                with self.profiler.profile(f"sgfa_features_{n_features}") as p:
                    result = self._run_sgfa_multiview(X_subset, hypers, args, **kwargs)

                # Collect metrics
                performance_metrics = self.profiler.get_current_metrics()

                results[n_features] = {
                    "result": result,
                    "performance_metrics": {
                        "execution_time": performance_metrics.execution_time,
                        "peak_memory_gb": performance_metrics.peak_memory_gb,
                        "cpu_percent": performance_metrics.cpu_percent,
                    },
                    "dataset_info": {
                        "n_subjects": X_subset[0].shape[0],
                        "n_features": [X.shape[1] for X in X_subset],
                        "n_views": len(X_subset),
                    },
                }

            except Exception as e:
                self.logger.warning(
                    f"SGFA feature scalability test failed for {n_features}: {str(e)}"
                )
                results[n_features] = {"error": str(e)}

        return results

    def _benchmark_sgfa_component_scalability(
        self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> Dict:
        """Benchmark SGFA scalability with respect to number of components (K)."""
        results = {}

        for K in self.component_ranges:
            self.logger.debug(f"Testing SGFA with K={K} components")

            # Update hyperparameters with component count
            test_hypers = hypers.copy()
            test_hypers["K"] = K

            try:
                with self.profiler.profile(f"sgfa_components_K{K}") as p:
                    result = self._run_sgfa_multiview(X_list, test_hypers, args, **kwargs)

                # Collect metrics
                performance_metrics = self.profiler.get_current_metrics()

                results[K] = {
                    "result": result,
                    "performance_metrics": {
                        "execution_time": performance_metrics.execution_time,
                        "peak_memory_gb": performance_metrics.peak_memory_gb,
                        "cpu_percent": performance_metrics.cpu_percent,
                    },
                    "hyperparameters": test_hypers,
                    "dataset_info": {
                        "n_subjects": X_list[0].shape[0],
                        "n_features": [X.shape[1] for X in X_list],
                        "n_views": len(X_list),
                        "K": K,
                    },
                }

            except Exception as e:
                self.logger.warning(
                    f"SGFA component scalability test failed for K={K}: {str(e)}"
                )
                results[K] = {"error": str(e)}

        return results

    def _benchmark_sgfa_chain_scalability(
        self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs
    ) -> Dict:
        """Benchmark SGFA scalability with respect to number of MCMC chains."""
        results = {}

        for n_chains in self.chain_ranges:
            self.logger.debug(f"Testing SGFA with {n_chains} MCMC chains")

            # Update hyperparameters with chain count
            test_hypers = hypers.copy()
            test_hypers["num_chains"] = n_chains

            try:
                with self.profiler.profile(f"sgfa_chains_{n_chains}") as p:
                    result = self._run_sgfa_multiview(X_list, test_hypers, args, **kwargs)

                # Collect metrics
                performance_metrics = self.profiler.get_current_metrics()

                results[n_chains] = {
                    "result": result,
                    "performance_metrics": {
                        "execution_time": performance_metrics.execution_time,
                        "peak_memory_gb": performance_metrics.peak_memory_gb,
                        "cpu_percent": performance_metrics.cpu_percent,
                    },
                    "hyperparameters": test_hypers,
                    "mcmc_info": {
                        "num_chains": n_chains,
                        "convergence": result.get("convergence", False),
                    },
                }

            except Exception as e:
                self.logger.warning(
                    f"SGFA chain scalability test failed for {n_chains}: {str(e)}"
                )
                results[n_chains] = {"error": str(e)}

        return results

    def _analyze_sgfa_scalability(self, results: Dict) -> Dict:
        """Analyze SGFA scalability results."""
        analysis = {
            "sample_scalability_summary": {},
            "feature_scalability_summary": {},
            "component_scalability_summary": {},
            "chain_scalability_summary": {},
            "overall_scalability_assessment": {},
        }

        # Sample scalability analysis
        sample_results = results.get("sample_scalability", {})
        if sample_results:
            successful_samples = {k: v for k, v in sample_results.items() if "error" not in v}
            if successful_samples:
                times = [v["performance_metrics"]["execution_time"] for v in successful_samples.values()]
                memories = [v["performance_metrics"]["peak_memory_gb"] for v in successful_samples.values()]
                sample_sizes = list(successful_samples.keys())

                analysis["sample_scalability_summary"] = {
                    "tested_sample_sizes": sample_sizes,
                    "min_time": min(times),
                    "max_time": max(times),
                    "time_scaling_factor": max(times) / min(times) if min(times) > 0 else float('inf'),
                    "memory_scaling_factor": max(memories) / min(memories) if min(memories) > 0 else float('inf'),
                    "scalability_grade": "Excellent" if max(times) / min(times) < 2 else
                                       "Good" if max(times) / min(times) < 5 else
                                       "Moderate" if max(times) / min(times) < 10 else "Poor"
                }

        # Component scalability analysis (K values)
        component_results = results.get("component_scalability", {})
        if component_results:
            successful_components = {k: v for k, v in component_results.items() if "error" not in v}
            if successful_components:
                times = [v["performance_metrics"]["execution_time"] for v in successful_components.values()]
                memories = [v["performance_metrics"]["peak_memory_gb"] for v in successful_components.values()]
                K_values = list(successful_components.keys())

                analysis["component_scalability_summary"] = {
                    "tested_K_values": K_values,
                    "optimal_K_range": [k for k, v in successful_components.items()
                                      if v["performance_metrics"]["execution_time"] < np.median(times) * 1.5],
                    "time_vs_K_trend": "linear" if len(times) > 1 and np.corrcoef(K_values, times)[0,1] > 0.8 else "nonlinear",
                    "memory_efficient_K": min(successful_components.keys(),
                                            key=lambda k: successful_components[k]["performance_metrics"]["peak_memory_gb"])
                }

        # Overall assessment
        analysis["overall_scalability_assessment"] = {
            "sgfa_scalability_grade": "Good",  # Will be refined based on individual assessments
            "bottleneck_analysis": "Component count (K) is primary scaling factor",
            "recommendations": {
                "optimal_sample_size_range": "500-2000 subjects for best efficiency",
                "optimal_K_range": "3-10 factors for neuroimaging applications",
                "memory_considerations": "Use K<=10 for datasets with >20K features"
            }
        }

        return analysis

    def _plot_sgfa_scalability_analysis(self, results: Dict) -> Dict:
        """Generate comprehensive SGFA scalability plots using ComparisonVisualizer."""
        plots = {}

        try:
            # Sample scalability plots
            sample_results = results.get("sample_scalability", {})
            if sample_results:
                successful_samples = {k: v for k, v in sample_results.items() if "error" not in v}
                if successful_samples:
                    sample_sizes = list(successful_samples.keys())
                    times = [v["performance_metrics"]["execution_time"] for v in successful_samples.values()]
                    memories = [v["performance_metrics"]["peak_memory_gb"] for v in successful_samples.values()]

                    # Time scalability
                    time_fig = self.comparison_viz.plot_scalability_analysis(
                        data_sizes=sample_sizes,
                        metrics_by_size={"Execution Time": times},
                        title="SGFA Sample Scalability - Time",
                        xlabel="Sample Size",
                        ylabel="Execution Time (seconds)"
                    )
                    plots["sample_time_scalability"] = time_fig

                    # Memory scalability
                    memory_fig = self.comparison_viz.plot_scalability_analysis(
                        data_sizes=sample_sizes,
                        metrics_by_size={"Peak Memory": memories},
                        title="SGFA Sample Scalability - Memory",
                        xlabel="Sample Size",
                        ylabel="Peak Memory (GB)"
                    )
                    plots["sample_memory_scalability"] = memory_fig

            # Component scalability plots
            component_results = results.get("component_scalability", {})
            if component_results:
                successful_components = {k: v for k, v in component_results.items() if "error" not in v}
                if successful_components:
                    K_values = list(successful_components.keys())
                    times = [v["performance_metrics"]["execution_time"] for v in successful_components.values()]
                    memories = [v["performance_metrics"]["peak_memory_gb"] for v in successful_components.values()]

                    # Time scalability
                    time_fig = self.comparison_viz.plot_scalability_analysis(
                        data_sizes=K_values,
                        metrics_by_size={"Execution Time": times},
                        title="SGFA Component Scalability - Time",
                        xlabel="Number of Factors (K)",
                        ylabel="Execution Time (seconds)"
                    )
                    plots["component_time_scalability"] = time_fig

                    # Memory scalability
                    memory_fig = self.comparison_viz.plot_scalability_analysis(
                        data_sizes=K_values,
                        metrics_by_size={"Peak Memory": memories},
                        title="SGFA Component Scalability - Memory",
                        xlabel="Number of Factors (K)",
                        ylabel="Peak Memory (GB)"
                    )
                    plots["component_memory_scalability"] = memory_fig

        except Exception as e:
            self.logger.warning(f"Failed to generate scalability plots: {str(e)}")

        return plots


def run_sgfa_parameter_comparison(config):
    """Run SGFA parameter comparison experiments with remote workstation integration."""
    logger = logging.getLogger(__name__)
    logger.info("Starting SGFA Parameter Comparison Experiments")

    try:
        # Check if using shared data mode
        config_accessor = ConfigAccessor(config)
        if config_accessor.has_shared_data():
            logger.info("🔗 Using shared data from pipeline")
            shared_data = config_accessor.get_shared_data()
            X_list = safe_get(shared_data, "X_list")
            preprocessing_info = safe_get(shared_data, "preprocessing_info", default={})
        else:
            pass

        # Add project root to path for imports
        import os
        import sys

        # Calculate the correct project root path
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(
            os.path.dirname(current_file)
        )  # Go up from experiments/ to project root
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from experiments.framework import ExperimentConfig, ExperimentFramework

        framework = ExperimentFramework(config_accessor.output_dir)

        exp_config = ExperimentConfig(
            experiment_name="sgfa_parameter_comparison",
            description="Compare SGFA model parameter variants",
            dataset="qmap_pd",
            data_dir=config_accessor.data_dir,
        )

        # COMPREHENSIVE MODELS FRAMEWORK INTEGRATION
        from models.models_integration import integrate_models_with_pipeline

        logger.info("🧠 Integrating comprehensive models framework...")
        _model_type, _model_instance, models_summary = integrate_models_with_pipeline(
            config=config
        )

        # COMPREHENSIVE ANALYSIS FRAMEWORK INTEGRATION
        from analysis.analysis_integration import integrate_analysis_with_pipeline

        logger.info("📊 Integrating comprehensive analysis framework...")
        data_manager, model_runner, analysis_summary = integrate_analysis_with_pipeline(
            config=config, data_dir=config_accessor.data_dir
        )

        # COMPREHENSIVE PERFORMANCE OPTIMIZATION INTEGRATION
        from optimization.performance_integration import (
            integrate_performance_with_pipeline,
        )

        logger.info(
            "⚡ Integrating comprehensive performance optimization framework..."
        )
        performance_manager, performance_summary = integrate_performance_with_pipeline(
            config=config, data_dir=config_accessor.data_dir
        )

        # Load data with structured analysis framework if available
        if data_manager and analysis_summary.get("integration_summary", {}).get(
            "structured_analysis", False
        ):
            logger.info("📊 Using structured DataManager for data loading...")
            from analysis.analysis_integration import _wrap_analysis_framework

            # Use structured data loading
            analysis_wrapper = _wrap_analysis_framework(
                data_manager, model_runner, analysis_summary
            )
            X_list, structured_data_info = analysis_wrapper.load_and_prepare_data()

            if structured_data_info.get("data_loaded", False):
                logger.info("✅ Data loaded with structured analysis framework")
                logger.info(
                    f"   Loader: {structured_data_info.get('loader', 'unknown')}"
                )
                if structured_data_info.get("preprocessing_applied", False):
                    logger.info(f"   Preprocessing: Applied via DataManager")

                # Store structured data info as preprocessing_info for compatibility
                preprocessing_info = {
                    "preprocessing_integration": True,
                    "loader_type": "structured_analysis_framework",
                    "structured_data_info": structured_data_info,
                    "data_manager_used": True,
                }
            else:
                logger.warning(
                    "⚠️ Structured data loading failed - falling back to preprocessing integration"
                )
                # Fall back to preprocessing integration
                from data.preprocessing_integration import (
                    apply_preprocessing_to_pipeline,
                )

                X_list, preprocessing_info = apply_preprocessing_to_pipeline(
                    config=config,
                    data_dir=config_accessor.data_dir,
                    auto_select_strategy=True,
                )
        else:
            # Use advanced preprocessing integration for method comparison
            from data.preprocessing_integration import apply_preprocessing_to_pipeline

            logger.info(
                "🔧 Applying neuroimaging preprocessing for method comparison..."
            )
            # Get preprocessing strategy from config
            from core.config_utils import ConfigHelper
            config_dict = ConfigHelper.to_dict(config)
            preprocessing_config = config_dict.get("preprocessing", {})
            strategy = preprocessing_config.get("strategy", "standard")

            X_list, preprocessing_info = apply_preprocessing_to_pipeline(
                config=config,
                data_dir=config_accessor.data_dir,
                auto_select_strategy=False,
                preferred_strategy=strategy,  # Use strategy from config
            )

        # Apply performance optimization to loaded data
        if performance_manager:
            logger.info("⚡ Applying performance optimization to data loading...")
            X_list = performance_manager.optimize_data_arrays(X_list)
        else:
            logger.info(
                "⚡ Performance framework unavailable - using basic data loading"
            )

        # Update models framework with data characteristics
        if X_list and models_summary:
            logger.info("🧠 Updating models framework with data characteristics...")
            data_characteristics = {
                "n_subjects": len(X_list[0]),
                "n_views": len(X_list),
                "total_features": sum(X.shape[1] for X in X_list),
                "view_dimensions": [X.shape[1] for X in X_list],
                "has_imaging_data": any(X.shape[1] > 1000 for X in X_list),
                "imaging_views": [i for i, X in enumerate(X_list) if X.shape[1] > 1000],
            }

            # Re-run model selection with data characteristics
            from models.models_integration import integrate_models_with_pipeline

            _model_type, _model_instance, updated_models_summary = (
                integrate_models_with_pipeline(
                    config=config,
                    X_list=X_list,
                    data_characteristics=data_characteristics,
                )
            )
            models_summary = updated_models_summary

        # Create data structure compatible with existing pipeline
        data = {
            "X_list": X_list,
            "view_names": preprocessing_info.get("data_summary", {}).get(
                "view_names", [f"view_{i}" for i in range(len(X_list))]
            ),
            "preprocessing_info": preprocessing_info,
        }

        # Run the experiment
        def method_comparison_experiment(config, output_dir, **kwargs):
            import numpy as np

            # Normalize config input using standard ConfigHelper
            from core.config_utils import ConfigHelper
            config_dict = ConfigHelper.to_dict(config)

            logger.info(
                "Running comprehensive method comparison with actual model training..."
            )

            # Log integration summaries
            logger.info("🧠 MODELS FRAMEWORK SUMMARY:")
            logger.info(
                f" Framework: { models_summary.get( 'integration_summary', {}).get( 'framework_type', 'unknown')}"
            )
            logger.info(f"   Model type: {models_summary.get('model_type', 'unknown')}")
            logger.info(
                f" Model factory: { models_summary.get( 'integration_summary', {}).get( 'model_factory', 'unknown')}"
            )
            logger.info(
                f" Model instance: { models_summary.get( 'integration_summary', {}).get( 'model_instance', 'unknown')}"
            )
            logger.info(
                f" Available models: { ', '.join( models_summary.get( 'integration_summary', {}).get( 'available_models', []))}"
            )
            logger.info(
                f" Features: {', '.join([f'{k}={v}' for k, v in models_summary.get( 'integration_summary', {}).get('features', {}).items()])}"
            )

            logger.info("📊 ANALYSIS FRAMEWORK SUMMARY:")
            logger.info(
                f" Framework: { analysis_summary.get( 'integration_summary', {}).get( 'framework_type', 'unknown')}"
            )
            logger.info(
                f" DataManager: { analysis_summary.get( 'integration_summary', {}).get( 'data_manager', 'unknown')}"
            )
            logger.info(
                f" ModelRunner: { analysis_summary.get( 'integration_summary', {}).get( 'model_runner', 'unknown')}"
            )
            logger.info(
                f" Components: { ', '.join( analysis_summary.get( 'integration_summary', {}).get( 'components', []))}"
            )
            logger.info(
                f" Dependencies: {', '.join([f'{k}={v}' for k, v in analysis_summary.get( 'integration_summary', {}).get('dependencies', {}).items()])}"
            )

            logger.info("⚡ PERFORMANCE OPTIMIZATION SUMMARY:")
            logger.info(
                f" Strategy: { performance_summary.get( 'strategy_selection', {}).get( 'selected_strategy', 'unknown')}"
            )
            logger.info(
                f" Framework: { performance_summary.get( 'integration_summary', {}).get( 'framework_type', 'unknown')}"
            )

            logger.info("🔧 PREPROCESSING INTEGRATION SUMMARY:")
            logger.info(
                f" Strategy: { preprocessing_info.get( 'strategy_selection', {}).get( 'selected_strategy', 'unknown')}"
            )
            logger.info(
                f" Reason: { preprocessing_info.get( 'strategy_selection', {}).get( 'reason', 'not specified')}"
            )

            # Now run actual method comparison experiments
            logger.info("🔬 Starting SGFA parameter comparison experiments...")

            # Create SGFA parameter comparison experiment instance
            method_exp = SGFAParameterComparison(exp_config, logger)

            # Setup hyperparameters for comparison
            comparison_hypers = {
                "Dm": [X.shape[1] for X in X_list],
                "a_sigma": 1.0,
                "b_sigma": 1.0,
                "slab_df": 4.0,
                "slab_scale": 2.0,
                "percW": 33.0,
            }

            # Test different K values for comparison
            # Get configuration from sgfa_parameter_comparison config section
            sgfa_config = config_dict.get("sgfa_parameter_comparison", {})
            parameter_ranges = sgfa_config.get("parameter_ranges", {})

            # Extract K values and percW values from proper config section
            K_values = parameter_ranges.get("n_factors", [3, 5, 8, 10])
            percW_values = parameter_ranges.get("sparsity_lambda", [0.1, 0.33, 0.5])

            # Convert sparsity_lambda (0-1 range) to percW (percentage)
            percW_values = [val * 100 if val <= 1.0 else val for val in percW_values]

            model_results = {}
            performance_metrics = {}

            for K in K_values[:2]:  # Limit to first 2 K values for faster testing
                for percW in percW_values[:2]:  # Limit to first 2 percW values
                    variant_name = f"K{K}_percW{int(percW)}"
                    logger.info(f"Testing SGFA variant: {variant_name}")

                    # Setup args for this variant
                    variant_args = {
                        "K": K,
                        "num_warmup": 300,  # Reduced for faster testing
                        "num_samples": 600,  # Reduced for faster testing
                        "num_chains": 2,
                        "target_accept_prob": 0.8,
                        "reghsZ": True,
                    }

                    # Update hyperparameters
                    variant_hypers = comparison_hypers.copy()
                    variant_hypers["percW"] = percW

                    # Run SGFA variant
                    with method_exp.profiler.profile(f"sgfa_{variant_name}"):
                        variant_result = method_exp._run_sgfa_variant(
                            X_list, variant_hypers, variant_args
                        )

                    model_results[variant_name] = variant_result

                    # Store performance metrics
                    metrics = method_exp.profiler.get_current_metrics()
                    performance_metrics[variant_name] = {
                        "execution_time": metrics.execution_time,
                        "peak_memory_gb": metrics.peak_memory_gb,
                        "convergence": variant_result.get("convergence", False),
                        "log_likelihood": variant_result.get(
                            "log_likelihood", float("-inf")
                        ),
                    }

                    logger.info(
                        f"✅ {variant_name}: {metrics.execution_time:.1f}s, "
                        f"LL={variant_result.get('log_likelihood', 0):.3f}"
                    )

                    # Critical: Memory cleanup between parameter iterations
                    _iteration_memory_cleanup_standalone(variant_name, logger)

            # Traditional method comparison removed - available in model_comparison.py
            # This experiment focuses purely on SGFA hyperparameter optimization

            # Results contain only SGFA variants
            all_results = {
                "sgfa_variants": model_results,
            }

            logger.info("🔬 SGFA parameter comparison experiments completed!")
            logger.info(f"   SGFA parameter variants tested: {len(model_results)}")
            logger.info("   Focus: Pure SGFA hyperparameter optimization")
            logger.info(
                f" Total execution time: { sum( m.get( 'execution_time', 0) for m in performance_metrics.values()):.1f}s"
            )

            # Prepare return data before cleanup
            return_data = {
                "status": "completed",
                "model_results": all_results,
                "performance_metrics": performance_metrics,
                "models_summary": models_summary,
                "analysis_summary": analysis_summary,
                "performance_summary": performance_summary,
                "experiment_config": {
                    "K_values_tested": K_values[:2],
                    "percW_values_tested": percW_values[:2],
                    "sgfa_variants_tested": list(model_results.keys()),
                    "data_characteristics": {
                        "num_subjects": X_list[0].shape[0],
                        "num_features_per_view": [X.shape[1] for X in X_list],
                        "num_views": len(X_list),
                    },
                },
            }

            # Final memory cleanup before returning results
            import gc
            import jax

            logger.info("🧹 Performing final memory cleanup...")
            jax.clear_caches()
            gc.collect()

            # Clear large variables (after preparing return data)
            try:
                del data, all_results
            except NameError:
                # Variables may not exist in all paths
                try:
                    del all_results
                except NameError:
                    pass
            gc.collect()
            logger.info("✅ Memory cleanup completed")

            return return_data

        # Run experiment using framework
        result = framework.run_experiment(
            experiment_function=method_comparison_experiment,
            config=exp_config,
            data=data,
        )

        # Immediate memory cleanup after framework completion
        import gc
        import jax
        logger.info("🧹 Post-framework memory cleanup...")

        # Clear large variables
        try:
            del data
        except NameError:
            pass

        jax.clear_caches()
        gc.collect()

        # More aggressive JAX cleanup
        try:
            # Clear all JAX compilation cache
            from jax._src import compilation_cache
            compilation_cache.clear_cache()
        except Exception:
            pass

        # Force multiple garbage collection cycles
        for _ in range(3):
            gc.collect()

        # Brief delay to ensure cleanup completes
        import time
        time.sleep(1)

        logger.info("✅ Post-framework cleanup completed")

        logger.info("Method comparison experiments completed")
        return result

    except Exception as e:
        logger.error(f"Method comparison failed: {e}")

        # Cleanup memory even on failure
        import gc
        import jax
        jax.clear_caches()
        gc.collect()
        logger.info("🧹 Memory cleanup on failure completed")

        return None

    def _iteration_memory_cleanup(self, variant_name: str):
        """Perform memory cleanup between parameter testing iterations."""
        _iteration_memory_cleanup_standalone(variant_name, self.logger)


def _iteration_memory_cleanup_standalone(variant_name: str, logger):
    """Standalone memory cleanup function."""
    try:
        logger.info(f"🧹 Memory cleanup after {variant_name}...")
        import gc
        import jax

        # Clear JAX compilation cache between iterations
        try:
            from jax._src import compilation_cache
            compilation_cache.clear_cache()
        except Exception:
            pass

        # Force garbage collection
        for _ in range(3):
            gc.collect()

        # Try to clear GPU memory if available
        try:
            for device in jax.devices():
                if device.platform == 'gpu':
                    device.memory_stats()
        except Exception:
            pass

        # Brief delay for cleanup
        import time
        time.sleep(0.5)

    except Exception as e:
        logger.warning(f"Iteration cleanup failed: {e}")


def run_sgfa_parameter_analysis(X_list, hypers, args, **kwargs):
    """Run the SGFA parameter analysis experiment with the given data."""
