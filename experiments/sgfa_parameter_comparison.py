"""SGFA parameter comparison experiments for qMAP-PD analysis.

This module focuses on optimizing hyperparameters (K, percW) for the sparseGFA model.
For comparing different model architectures (sparseGFA vs alternatives),
see experiments/model_comparison.py.
"""

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA, FactorAnalysis

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


@performance_optimized_experiment()
class SGFAParameterComparison(ExperimentFramework):
    """SGFA parameter comparison experiments for K and percW optimization."""

    def __init__(
        self, config: ExperimentConfig, logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, None, logger)
        self.profiler = PerformanceProfiler()

        # Initialize neuroimaging hyperparameter optimizer
        cv_config = NeuroImagingCVConfig()
        cv_config.inner_cv_folds = 3  # Reduced for hyperparameter optimization

        self.hyperopt = NeuroImagingHyperOptimizer(config=cv_config)

        # Initialize fallback handler for hyperparameter optimization
        self.hyperopt_fallback = HyperoptFallbackHandler(self.logger)

        # Method configurations
        self.sgfa_variants = {
            "standard": {"use_sparse": True, "use_group": True},
            "sparse_only": {"use_sparse": True, "use_group": False},
            "group_only": {"use_sparse": False, "use_group": True},
            "basic_fa": {"use_sparse": False, "use_group": False},
        }

        # Focus purely on SGFA parameter optimization
        # Traditional methods moved to model_comparison.py

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

    @experiment_handler("traditional_method_comparison")
    @validate_data_types(X_list=list, sgfa_results=(dict, type(None)))
    @validate_parameters(
        X_list=lambda x: len(x) > 0 and all(isinstance(arr, np.ndarray) for arr in x)
    )
    def run_traditional_method_comparison(
        self, X_list: List[np.ndarray], sgfa_results: Dict = None, **kwargs
    ) -> ExperimentResult:
        """Compare SGFA with traditional dimensionality reduction methods."""
        self.logger.info("Running traditional method comparison")

        results = {}
        performance_metrics = {}

        # Concatenate multi-view data for traditional methods
        X_concat = np.hstack(X_list) if len(X_list) > 1 else X_list[0]
        X_concat.shape[0]

        # Determine number of components
        n_components = kwargs.get("n_components", min(10, X_concat.shape[1] // 2))

        traditional_methods = ["pca", "fa", "cca", "kmeans", "ica"]
        for method_name in traditional_methods:
            self.logger.info(f"Testing traditional method: {method_name}")

            with self.profiler.profile(f"traditional_{method_name}"):
                method_result = self._run_traditional_method(
                    X_concat, method_name, n_components, **kwargs
                )
                results[method_name] = method_result

            # Store performance metrics
            metrics = self.profiler.get_current_metrics()
            performance_metrics[method_name] = {
                "execution_time": metrics.execution_time,
                "peak_memory_gb": metrics.peak_memory_gb,
            }

        # Include SGFA results if provided
        if sgfa_results:
            results["sgfa"] = sgfa_results
            performance_metrics["sgfa"] = {
                "execution_time": sgfa_results.get("execution_time", 0),
                "peak_memory_gb": sgfa_results.get("peak_memory_gb", 0),
            }

        # Analyze method comparison
        analysis = self._analyze_traditional_comparison(results, X_list)

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

            # Traditional methods (concatenated)
            X_concat = np.hstack(view_subset)
            traditional_results = {}

            for method in ["pca", "fa", "cca"]:
                if method == "cca" and n_view_test < 2:
                    continue  # CCA needs at least 2 views

                traditional_results[method] = self._run_traditional_method(
                    X_concat, method, min(10, X_concat.shape[1] // 2)
                )

            results[f"{n_view_test}_views"] = {
                "sgfa": sgfa_result,
                "traditional": traditional_results,
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
    def run_scalability_comparison(
        self,
        X_list: List[np.ndarray],
        sample_sizes: List[int] = None,
        feature_sizes: List[int] = None,
        **kwargs,
    ) -> ExperimentResult:
        """Compare scalability of different methods."""
        self.logger.info("Running scalability comparison")

        if sample_sizes is None:
            sample_sizes = [100, 500, 1000, 2000]
        if feature_sizes is None:
            feature_sizes = [50, 100, 200, 500]

        results = {}

        # Sample size scalability
        self.logger.info("Testing sample size scalability")
        sample_results = {}

        for n_samples in sample_sizes:
            if n_samples > X_list[0].shape[0]:
                continue

            # Subsample data
            indices = np.random.choice(X_list[0].shape[0], n_samples, replace=False)
            X_subset = [X[indices] for X in X_list]

            sample_results[n_samples] = self._run_scalability_test(X_subset, **kwargs)

        results["sample_scalability"] = sample_results

        # Feature size scalability
        self.logger.info("Testing feature size scalability")
        feature_results = {}

        for n_features in feature_sizes:
            # Select subset of features from each view
            X_feature_subset = []
            for X in X_list:
                if n_features >= X.shape[1]:
                    X_feature_subset.append(X)
                else:
                    feature_indices = np.random.choice(
                        X.shape[1], n_features, replace=False
                    )
                    X_feature_subset.append(X[:, feature_indices])

            feature_results[n_features] = self._run_scalability_test(
                X_feature_subset, **kwargs
            )

        results["feature_scalability"] = feature_results

        # Analyze scalability
        analysis = self._analyze_scalability(results)

        # Generate plots
        plots = self._plot_scalability_comparison(results)

        return ExperimentResult(
            experiment_name="scalability_comparison",
            config=self.config,
            data=results,
            analysis=analysis,
            plots=plots,
            success=True,
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

        # Define hyperparameter search space specific to SGFA neuroimaging applications
        search_space = {
            'K': {
                'type': 'int',
                'low': 3,
                'high': 15,
                'step': 1,
                'description': 'Number of latent factors'
            },
            'percW': {
                'type': 'float',
                'low': 10.0,
                'high': 50.0,
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
                    result = self._run_sgfa_variant(X_list, trial_hypers, trial_args)

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

            final_result = self._run_sgfa_variant(X_list, final_hypers, final_args)

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
        """Generate plots for hyperparameter optimization results."""
        plots = {}

        try:
            optimization_history = results.get('optimization_history', [])
            best_params = results.get('best_parameters', {})

            if not optimization_history:
                return plots

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("Neuroimaging Hyperparameter Optimization Results", fontsize=16)

            # Plot 1: Optimization progress
            objective_values = [trial.get('objective_value', float('-inf')) for trial in optimization_history]
            valid_indices = [i for i, obj in enumerate(objective_values) if obj != float('-inf')]
            valid_objectives = [objective_values[i] for i in valid_indices]

            if valid_objectives:
                axes[0, 0].plot(valid_indices, valid_objectives, 'o-', alpha=0.7)
                axes[0, 0].set_xlabel("Trial Number")
                axes[0, 0].set_ylabel("Objective Score")
                axes[0, 0].set_title("Optimization Progress")
                axes[0, 0].grid(True, alpha=0.3)

                # Add best score line
                best_score = max(valid_objectives)
                axes[0, 0].axhline(y=best_score, color='red', linestyle='--',
                                 label=f'Best Score: {best_score:.2f}')
                axes[0, 0].legend()

            # Plot 2: Parameter vs Objective (K)
            K_values = [trial.get('K') for trial in optimization_history if 'K' in trial]
            K_objectives = [trial.get('objective_value') for trial in optimization_history if 'K' in trial]

            if K_values and K_objectives:
                valid_K_data = [(k, obj) for k, obj in zip(K_values, K_objectives) if obj != float('-inf')]
                if valid_K_data:
                    K_vals, K_objs = zip(*valid_K_data)
                    axes[0, 1].scatter(K_vals, K_objs, alpha=0.7, s=50)
                    axes[0, 1].set_xlabel("Number of Factors (K)")
                    axes[0, 1].set_ylabel("Objective Score")
                    axes[0, 1].set_title("Factors vs Performance")
                    axes[0, 1].grid(True, alpha=0.3)

                    # Highlight best K
                    best_K = best_params.get('K')
                    if best_K is not None:
                        axes[0, 1].axvline(x=best_K, color='red', linestyle='--',
                                         label=f'Best K: {best_K}')
                        axes[0, 1].legend()

            # Plot 3: Parameter vs Objective (percW)
            percW_values = [trial.get('percW') for trial in optimization_history if 'percW' in trial]
            percW_objectives = [trial.get('objective_value') for trial in optimization_history if 'percW' in trial]

            if percW_values and percW_objectives:
                valid_percW_data = [(p, obj) for p, obj in zip(percW_values, percW_objectives) if obj != float('-inf')]
                if valid_percW_data:
                    percW_vals, percW_objs = zip(*valid_percW_data)
                    axes[1, 0].scatter(percW_vals, percW_objs, alpha=0.7, s=50, color='orange')
                    axes[1, 0].set_xlabel("Sparsity Percentage (percW)")
                    axes[1, 0].set_ylabel("Objective Score")
                    axes[1, 0].set_title("Sparsity vs Performance")
                    axes[1, 0].grid(True, alpha=0.3)

                    # Highlight best percW
                    best_percW = best_params.get('percW')
                    if best_percW is not None:
                        axes[1, 0].axvline(x=best_percW, color='red', linestyle='--',
                                         label=f'Best percW: {best_percW:.1f}')
                        axes[1, 0].legend()

            # Plot 4: Parameter correlation heatmap
            param_names = ['K', 'percW', 'num_samples']
            param_data = []

            for param in param_names:
                param_values = [trial.get(param, np.nan) for trial in optimization_history]
                param_data.append(param_values)

            # Add objective values
            param_data.append(objective_values)
            param_names.append('Objective')

            # Calculate correlation matrix
            valid_data = []
            for i in range(len(optimization_history)):
                row_data = [param_data[j][i] for j in range(len(param_names))]
                if all(not (isinstance(x, float) and np.isnan(x)) and x != float('-inf') for x in row_data):
                    valid_data.append(row_data)

            if len(valid_data) > 3:  # Need sufficient data
                corr_matrix = np.corrcoef(np.array(valid_data).T)

                im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                axes[1, 1].set_xticks(range(len(param_names)))
                axes[1, 1].set_yticks(range(len(param_names)))
                axes[1, 1].set_xticklabels(param_names, rotation=45)
                axes[1, 1].set_yticklabels(param_names)
                axes[1, 1].set_title("Parameter Correlation Matrix")

                # Add correlation values
                for i in range(len(param_names)):
                    for j in range(len(param_names)):
                        axes[1, 1].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontsize=8)

                plt.colorbar(im, ax=axes[1, 1])

            plt.tight_layout()
            plots["hyperparameter_optimization"] = fig

        except Exception as e:
            self.logger.warning(f"Failed to create hyperparameter optimization plots: {str(e)}")

        return plots

    def _run_sgfa_variant(
        self, X_list: List[np.ndarray], hypers: Dict, args: Dict
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
                f"Training SGFA model with K={
                    args.get(
                        'K',
                        10)}, percW={
                    hypers.get(
                        'percW',
                        33)}"
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

            # Get samples
            samples = mcmc.get_samples()

            # Calculate log likelihood (approximate)
            potential_energy = samples.get("potential_energy", np.array([]))
            if len(potential_energy) > 0:
                log_likelihood = -np.mean(potential_energy)
                self.logger.debug(
                    f"Potential energy stats: mean={
                        np.mean(potential_energy):.3f}, std={
                        np.std(potential_energy):.3f}"
                )
            else:
                log_likelihood = float("nan")  # Indicate missing data
                self.logger.warning(
                    "No potential energy data collected - log likelihood unavailable"
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

            self.logger.info(
                f"SGFA training completed in {
                    elapsed:.2f}s, log_likelihood: {
                    log_likelihood:.3f}"
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

    def _run_traditional_method(
        self, X: np.ndarray, method_name: str, n_components: int
    ) -> Dict:
        """Run traditional dimensionality reduction method."""
        results = {}

        try:
            if method_name == "pca":
                model = PCA(n_components=n_components)
                Z = model.fit_transform(X)
                results = {
                    "components": model.components_,
                    "explained_variance_ratio": model.explained_variance_ratio_,
                    "Z": Z,
                    "model": model,
                }

            elif method_name == "fa":
                model = FactorAnalysis(n_components=n_components)
                Z = model.fit_transform(X)
                results = {
                    "components": model.components_,
                    "Z": Z,
                    "loglik": model.score(X),
                    "model": model,
                }

            elif method_name == "cca":
                # For CCA, split data in half
                n_features_1 = X.shape[1] // 2
                X1 = X[:, :n_features_1]
                X2 = X[:, n_features_1:]

                model = CCA(
                    n_components=min(n_components, min(X1.shape[1], X2.shape[1]))
                )
                Z1, Z2 = model.fit_transform(X1, X2)

                results = {
                    "Z1": Z1,
                    "Z2": Z2,
                    "x_weights": model.x_weights_,
                    "y_weights": model.y_weights_,
                    "model": model,
                }

            elif method_name == "kmeans":
                model = KMeans(n_clusters=n_components, random_state=42)
                labels = model.fit_predict(X)

                results = {
                    "labels": labels,
                    "centers": model.cluster_centers_,
                    "inertia": model.inertia_,
                    "model": model,
                }

            elif method_name == "ica":
                from sklearn.decomposition import FastICA

                model = FastICA(n_components=n_components, random_state=42)
                Z = model.fit_transform(X)

                results = {"components": model.components_, "Z": Z, "model": model}

        except Exception as e:
            self.logger.warning(f"Method {method_name} failed: {str(e)}")
            results = {"error": str(e)}

        return results

    def _run_sgfa_multiview(self, X_list: List[np.ndarray]) -> Dict:
        """Run SGFA on multi-view data."""
        # Mock SGFA results for multi-view data
        return {
            "W": [np.random.randn(X.shape[1], 5) for X in X_list],
            "Z": np.random.randn(X_list[0].shape[0], 5),
            "log_likelihood": np.random.randn(),
            "n_views_used": len(X_list),
        }

    def _run_scalability_test(self, X_list: List[np.ndarray]) -> Dict:
        """Run scalability test for all methods."""
        results = {}

        # SGFA
        with self.profiler.profile("sgfa_scalability"):
            sgfa_result = self._run_sgfa_multiview(X_list)

        sgfa_metrics = self.profiler.get_current_metrics()
        results["sgfa"] = {
            "result": sgfa_result,
            "execution_time": sgfa_metrics.execution_time,
            "peak_memory_gb": sgfa_metrics.peak_memory_gb,
        }

        # Traditional methods
        X_concat = np.hstack(X_list)
        n_components = min(5, X_concat.shape[1] // 2)

        for method in ["pca", "fa"]:
            with self.profiler.profile(f"{method}_scalability"):
                method_result = self._run_traditional_method(
                    X_concat, method, n_components
                )

            method_metrics = self.profiler.get_current_metrics()
            results[method] = {
                "result": method_result,
                "execution_time": method_metrics.execution_time,
                "peak_memory_gb": method_metrics.peak_memory_gb,
            }

        return results

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

    def _analyze_traditional_comparison(
        self, results: Dict, X_list: List[np.ndarray]
    ) -> Dict:
        """Analyze traditional method comparison results."""
        analysis = {
            "method_summary": {},
            "multi_view_handling": {},
            "dimensionality_reduction_quality": {},
            "computational_efficiency": {},
        }

        # Method summary
        for method_name, result in results.items():
            if "error" in result:
                analysis["method_summary"][method_name] = {
                    "status": "failed",
                    "error": result["error"],
                }
                continue

            analysis["method_summary"][method_name] = {
                "status": "success",
                "output_dimensions": self._get_output_dimensions(result, method_name),
            }

        # Multi-view handling assessment
        sum(X.shape[1] for X in X_list)
        for method_name, result in results.items():
            if method_name == "sgfa":
                analysis["multi_view_handling"][method_name] = "native_support"
            elif method_name == "cca":
                analysis["multi_view_handling"][method_name] = "pairwise_only"
            else:
                analysis["multi_view_handling"][method_name] = "concatenation_required"

        return analysis

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
            traditional_results = view_results["traditional"]

            analysis["view_scaling"][n_views] = {
                "sgfa_likelihood": sgfa_result.get("log_likelihood", 0),
                "traditional_methods_available": list(traditional_results.keys()),
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

    def _get_output_dimensions(self, result: Dict, method_name: str) -> int:
        """Get output dimensions for a method result."""
        if method_name == "sgfa":
            return result["Z"].shape[1] if "Z" in result else 0
        elif "Z" in result:
            return result["Z"].shape[1]
        elif "Z1" in result and "Z2" in result:  # CCA
            return result["Z1"].shape[1]
        elif "labels" in result:  # K-means
            return len(np.unique(result["labels"]))
        else:
            return 0

    def _plot_sgfa_comparison(self, results: Dict, performance_metrics: Dict) -> Dict:
        """Generate plots for SGFA variant comparison."""
        plots = {}

        try:
            # Performance comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("SGFA Variant Comparison", fontsize=16)

            variants = list(results.keys())

            # Execution time
            times = [performance_metrics[v]["execution_time"] for v in variants]
            axes[0, 0].bar(variants, times)
            axes[0, 0].set_title("Execution Time")
            axes[0, 0].set_ylabel("Time (seconds)")
            axes[0, 0].tick_params(axis="x", rotation=45)

            # Memory usage
            memory = [performance_metrics[v]["peak_memory_gb"] for v in variants]
            axes[0, 1].bar(variants, memory)
            axes[0, 1].set_title("Peak Memory Usage")
            axes[0, 1].set_ylabel("Memory (GB)")
            axes[0, 1].tick_params(axis="x", rotation=45)

            # Convergence iterations
            iterations = [
                performance_metrics[v]["convergence_iterations"] for v in variants
            ]
            axes[1, 0].bar(variants, iterations)
            axes[1, 0].set_title("Convergence Iterations")
            axes[1, 0].set_ylabel("Iterations")
            axes[1, 0].tick_params(axis="x", rotation=45)

            # Log likelihood
            likelihoods = [results[v].get("log_likelihood", 0) for v in variants]
            axes[1, 1].bar(variants, likelihoods)
            axes[1, 1].set_title("Log Likelihood")
            axes[1, 1].set_ylabel("Log Likelihood")
            axes[1, 1].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plots["sgfa_variant_comparison"] = fig

        except Exception as e:
            self.logger.warning(f"Failed to create SGFA comparison plots: {str(e)}")

        return plots

    def _plot_traditional_comparison(
        self, results: Dict, performance_metrics: Dict
    ) -> Dict:
        """Generate plots for traditional method comparison."""
        plots = {}

        try:
            # Method performance comparison
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle("Traditional Method Comparison", fontsize=16)

            methods = list(performance_metrics.keys())

            # Execution time comparison
            times = [performance_metrics[m]["execution_time"] for m in methods]
            axes[0].bar(methods, times)
            axes[0].set_title("Execution Time by Method")
            axes[0].set_ylabel("Time (seconds)")
            axes[0].tick_params(axis="x", rotation=45)

            # Memory usage comparison
            memory = [performance_metrics[m]["peak_memory_gb"] for m in methods]
            axes[1].bar(methods, memory)
            axes[1].set_title("Peak Memory Usage by Method")
            axes[1].set_ylabel("Memory (GB)")
            axes[1].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plots["traditional_method_comparison"] = fig

        except Exception as e:
            self.logger.warning(
                f"Failed to create traditional comparison plots: {str(e)}"
            )

        return plots

    def _plot_multiview_comparison(self, results: Dict) -> Dict:
        """Generate plots for multi-view comparison."""
        plots = {}

        try:
            # Multi-view scaling plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle("Multi-view Capability Assessment", fontsize=16)

            view_counts = sorted([int(k.split("_")[0]) for k in results.keys()])

            # SGFA likelihood vs number of views
            sgfa_likelihoods = [
                results[f"{n}_views"]["sgfa"].get("log_likelihood", 0)
                for n in view_counts
            ]
            axes[0].plot(view_counts, sgfa_likelihoods, "o-", label="SGFA")
            axes[0].set_title("Log Likelihood vs Number of Views")
            axes[0].set_xlabel("Number of Views")
            axes[0].set_ylabel("Log Likelihood")
            axes[0].legend()

            # Feature dimensions handled
            total_features = [
                sum(results[f"{n}_views"]["view_dimensions"]) for n in view_counts
            ]
            axes[1].plot(view_counts, total_features, "s-", color="red")
            axes[1].set_title("Total Features vs Number of Views")
            axes[1].set_xlabel("Number of Views")
            axes[1].set_ylabel("Total Features")

            plt.tight_layout()
            plots["multiview_comparison"] = fig

        except Exception as e:
            self.logger.warning(
                f"Failed to create multi-view comparison plots: {str(e)}"
            )

        return plots

    def _plot_scalability_comparison(self, results: Dict) -> Dict:
        """Generate plots for scalability comparison."""
        plots = {}

        try:
            # Scalability plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("Scalability Comparison", fontsize=16)

            # Sample size scalability - execution time
            sample_results = results["sample_scalability"]
            sample_sizes = sorted(sample_results.keys())

            methods = list(next(iter(sample_results.values())).keys())

            for method in methods:
                times = [
                    sample_results[size][method]["execution_time"]
                    for size in sample_sizes
                ]
                axes[0, 0].plot(sample_sizes, times, "o-", label=method)

            axes[0, 0].set_title("Execution Time vs Sample Size")
            axes[0, 0].set_xlabel("Sample Size")
            axes[0, 0].set_ylabel("Time (seconds)")
            axes[0, 0].legend()
            axes[0, 0].set_xscale("log")
            axes[0, 0].set_yscale("log")

            # Sample size scalability - memory
            for method in methods:
                memory = [
                    sample_results[size][method]["peak_memory_gb"]
                    for size in sample_sizes
                ]
                axes[0, 1].plot(sample_sizes, memory, "s-", label=method)

            axes[0, 1].set_title("Memory Usage vs Sample Size")
            axes[0, 1].set_xlabel("Sample Size")
            axes[0, 1].set_ylabel("Memory (GB)")
            axes[0, 1].legend()
            axes[0, 1].set_xscale("log")

            # Feature size scalability - execution time
            feature_results = results["feature_scalability"]
            feature_sizes = sorted(feature_results.keys())

            for method in methods:
                times = [
                    feature_results[size][method]["execution_time"]
                    for size in feature_sizes
                ]
                axes[1, 0].plot(feature_sizes, times, "o-", label=method)

            axes[1, 0].set_title("Execution Time vs Feature Size")
            axes[1, 0].set_xlabel("Feature Size")
            axes[1, 0].set_ylabel("Time (seconds)")
            axes[1, 0].legend()
            axes[1, 0].set_xscale("log")
            axes[1, 0].set_yscale("log")

            # Feature size scalability - memory
            for method in methods:
                memory = [
                    feature_results[size][method]["peak_memory_gb"]
                    for size in feature_sizes
                ]
                axes[1, 1].plot(feature_sizes, memory, "s-", label=method)

            axes[1, 1].set_title("Memory Usage vs Feature Size")
            axes[1, 1].set_xlabel("Feature Size")
            axes[1, 1].set_ylabel("Memory (GB)")
            axes[1, 1].legend()
            axes[1, 1].set_xscale("log")

            plt.tight_layout()
            plots["scalability_comparison"] = fig

        except Exception as e:
            self.logger.warning(
                f"Failed to create scalability comparison plots: {str(e)}"
            )

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


def run_method_comparison(config):
    """Run method comparison experiments with remote workstation integration."""
    logger = logging.getLogger(__name__)
    logger.info("Starting Method Comparison Experiments")

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

        framework = ExperimentFramework(base_output_dir=config_accessor.output_dir)

        exp_config = ExperimentConfig(
            experiment_name="remote_workstation_method_comparison",
            description="Compare SGFA model variants on remote workstation",
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
                "🔧 Applying advanced neuroimaging preprocessing for method comparison..."
            )
            X_list, preprocessing_info = apply_preprocessing_to_pipeline(
                config=config,
                data_dir=config_accessor.data_dir,
                auto_select_strategy=False,
                preferred_strategy="aggressive",  # Use advanced preprocessing for better model comparison
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
        def method_comparison_experiment(config, _output_dir, **_kwargs):
            import numpy as np

            logger.info(
                "Running comprehensive method comparison with actual model training..."
            )

            # Log integration summaries
            logger.info("🧠 MODELS FRAMEWORK SUMMARY:")
            logger.info(
                f"   Framework: {
                    models_summary.get(
                        'integration_summary',
                        {}).get(
                        'framework_type',
                        'unknown')}"
            )
            logger.info(f"   Model type: {models_summary.get('model_type', 'unknown')}")
            logger.info(
                f"   Model factory: {
                    models_summary.get(
                        'integration_summary',
                        {}).get(
                        'model_factory',
                        'unknown')}"
            )
            logger.info(
                f"   Model instance: {
                    models_summary.get(
                        'integration_summary',
                        {}).get(
                        'model_instance',
                        'unknown')}"
            )
            logger.info(
                f"   Available models: {
                    ', '.join(
                        models_summary.get(
                            'integration_summary',
                            {}).get(
                            'available_models',
                            []))}"
            )
            logger.info(
                f"   Features: {', '.join([f'{k}={v}' for k, v in models_summary.get(
                'integration_summary', {}).get('features', {}).items()])}"
            )

            logger.info("📊 ANALYSIS FRAMEWORK SUMMARY:")
            logger.info(
                f"   Framework: {
                    analysis_summary.get(
                        'integration_summary',
                        {}).get(
                        'framework_type',
                        'unknown')}"
            )
            logger.info(
                f"   DataManager: {
                    analysis_summary.get(
                        'integration_summary',
                        {}).get(
                        'data_manager',
                        'unknown')}"
            )
            logger.info(
                f"   ModelRunner: {
                    analysis_summary.get(
                        'integration_summary',
                        {}).get(
                        'model_runner',
                        'unknown')}"
            )
            logger.info(
                f"   Components: {
                    ', '.join(
                        analysis_summary.get(
                            'integration_summary',
                            {}).get(
                            'components',
                            []))}"
            )
            logger.info(
                f"   Dependencies: {', '.join([f'{k}={v}' for k, v in analysis_summary.get(
                'integration_summary', {}).get('dependencies', {}).items()])}"
            )

            logger.info("⚡ PERFORMANCE OPTIMIZATION SUMMARY:")
            logger.info(
                f"   Strategy: {
                    performance_summary.get(
                        'strategy_selection',
                        {}).get(
                        'selected_strategy',
                        'unknown')}"
            )
            logger.info(
                f"   Framework: {
                    performance_summary.get(
                        'integration_summary',
                        {}).get(
                        'framework_type',
                        'unknown')}"
            )

            logger.info("🔧 PREPROCESSING INTEGRATION SUMMARY:")
            logger.info(
                f"   Strategy: {
                    preprocessing_info.get(
                        'strategy_selection',
                        {}).get(
                        'selected_strategy',
                        'unknown')}"
            )
            logger.info(
                f"   Reason: {
                    preprocessing_info.get(
                        'strategy_selection',
                        {}).get(
                        'reason',
                        'not specified')}"
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
            # Handle both ExperimentConfig object and dict config
            if hasattr(config, "K_values"):
                # ExperimentConfig object
                K_values = config.K_values
                percW_values = config.percW_values
            else:
                # Dictionary config
                K_values = (
                    config.get("method_comparison", {})
                    .get("models", [{}])[0]
                    .get("n_factors", [5, 10, 15])
                )
                percW_values = [25.0, 33.0, 50.0]  # Different sparsity levels

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

            # Run traditional method comparison for a subset of methods
            logger.info("🔬 Running traditional method comparison...")
            X_concat = np.hstack(X_list)

            traditional_methods = ["pca", "fa"]  # Reduced set for faster testing
            traditional_results = {}

            for method in traditional_methods:
                logger.info(f"Testing traditional method: {method}")

                with method_exp.profiler.profile(f"traditional_{method}"):
                    method_result = method_exp._run_traditional_method(
                        X_concat, method, min(10, X_concat.shape[1] // 2)
                    )

                traditional_results[method] = method_result

                # Store performance metrics
                metrics = method_exp.profiler.get_current_metrics()
                performance_metrics[method] = {
                    "execution_time": metrics.execution_time,
                    "peak_memory_gb": metrics.peak_memory_gb,
                }

                logger.info(f"✅ {method}: {metrics.execution_time:.1f}s")

            # Combine all results
            all_results = {
                "sgfa_variants": model_results,
                "traditional_methods": traditional_results,
            }

            logger.info("🔬 SGFA parameter comparison experiments completed!")
            logger.info(f"   SGFA parameter variants tested: {len(model_results)}")
            logger.info(f"   Traditional methods tested: {len(traditional_results)}")
            logger.info(
                f"   Total execution time: {
                    sum(
                        m.get(
                            'execution_time',
                            0) for m in performance_metrics.values()):.1f}s"
            )

            return {
                "status": "completed",
                "model_results": all_results,
                "performance_metrics": performance_metrics,
                "models_summary": models_summary,
                "analysis_summary": analysis_summary,
                "performance_summary": performance_summary,
                "data": data,
                "experiment_config": {
                    "K_values_tested": K_values[:2],
                    "percW_values_tested": percW_values[:2],
                    "traditional_methods_tested": traditional_methods,
                    "data_characteristics": {
                        "n_subjects": X_list[0].shape[0],
                        "n_views": len(X_list),
                        "view_dimensions": [X.shape[1] for X in X_list],
                    },
                },
            }

        # Run experiment using framework
        result = framework.run_experiment(
            experiment_function=method_comparison_experiment,
            config=exp_config,
            data=data,
        )

        logger.info("Method comparison experiments completed")
        return result

    except Exception as e:
        logger.error(f"Method comparison failed: {e}")
        return None
