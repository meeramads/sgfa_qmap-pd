"""Sensitivity analysis experiments for SGFA hyperparameters."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
from pathlib import Path
import logging
from scipy import stats
from itertools import product
import warnings

from .framework import ExperimentFramework, ExperimentConfig, ExperimentResult
from performance import PerformanceProfiler

class SensitivityAnalysisExperiments(ExperimentFramework):
    """Comprehensive sensitivity analysis for SGFA hyperparameters."""
    
    def __init__(self, config: ExperimentConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.profiler = PerformanceProfiler()
        
        # Default hyperparameter ranges for sensitivity analysis
        self.hyperparameter_ranges = {
            'alpha_w': [0.1, 0.5, 1.0, 2.0, 5.0],
            'alpha_z': [0.1, 0.5, 1.0, 2.0, 5.0], 
            'tau_w': [0.01, 0.1, 1.0, 10.0],
            'tau_z': [0.01, 0.1, 1.0, 10.0],
            'gamma': [0.1, 0.5, 1.0, 2.0, 5.0],
            'K': [2, 3, 5, 8, 10, 15],
            'sparsity_level': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        
        # Core hyperparameters that are most critical
        self.core_hyperparameters = ['alpha_w', 'alpha_z', 'K', 'sparsity_level']
        
    def run_univariate_sensitivity_analysis(self, X_list: List[np.ndarray],
                                          base_hypers: Dict, args: Dict,
                                          hyperparameters: List[str] = None,
                                          **kwargs) -> ExperimentResult:
        """Run univariate sensitivity analysis for individual hyperparameters."""
        if hyperparameters is None:
            hyperparameters = self.core_hyperparameters
            
        self.logger.info(f"Running univariate sensitivity analysis for: {hyperparameters}")
        
        results = {}
        performance_metrics = {}
        
        try:
            for hyperparam in hyperparameters:
                if hyperparam not in self.hyperparameter_ranges:
                    self.logger.warning(f"No range defined for hyperparameter: {hyperparam}")
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
                    with self.profiler.profile(f'{hyperparam}_{param_value}') as p:
                        try:
                            result = self._run_sgfa_analysis(X_list, test_hypers, args, **kwargs)
                            hyperparam_results[param_value] = result
                            
                            # Store performance metrics
                            metrics = self.profiler.get_current_metrics()
                            hyperparam_metrics[param_value] = {
                                'execution_time': metrics.execution_time,
                                'peak_memory_gb': metrics.peak_memory_gb,
                                'convergence': result.get('convergence', False),
                                'log_likelihood': result.get('log_likelihood', np.nan)
                            }
                            
                        except Exception as e:
                            self.logger.warning(f"Failed for {hyperparam}={param_value}: {str(e)}")
                            hyperparam_results[param_value] = {'error': str(e)}
                            hyperparam_metrics[param_value] = {
                                'execution_time': np.nan,
                                'peak_memory_gb': np.nan,
                                'convergence': False,
                                'log_likelihood': np.nan
                            }
                
                results[hyperparam] = hyperparam_results
                performance_metrics[hyperparam] = hyperparam_metrics
            
            # Analyze sensitivity
            analysis = self._analyze_univariate_sensitivity(results, performance_metrics)
            
            # Generate plots
            plots = self._plot_univariate_sensitivity(results, performance_metrics)
            
            return ExperimentResult(
                experiment_name="univariate_sensitivity_analysis",
                config=self.config,
                data=results,
                analysis=analysis,
                plots=plots,
                performance_metrics=performance_metrics,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Univariate sensitivity analysis failed: {str(e)}")
            return self._create_failure_result("univariate_sensitivity_analysis", str(e))
    
    def run_multivariate_sensitivity_analysis(self, X_list: List[np.ndarray],
                                            base_hypers: Dict, args: Dict,
                                            hyperparameter_pairs: List[Tuple[str, str]] = None,
                                            **kwargs) -> ExperimentResult:
        """Run multivariate sensitivity analysis for hyperparameter interactions."""
        if hyperparameter_pairs is None:
            hyperparameter_pairs = [
                ('alpha_w', 'alpha_z'),
                ('alpha_w', 'K'),
                ('tau_w', 'tau_z'),
                ('K', 'sparsity_level')
            ]
            
        self.logger.info(f"Running multivariate sensitivity analysis for: {hyperparameter_pairs}")
        
        results = {}
        performance_metrics = {}
        
        try:
            for param1, param2 in hyperparameter_pairs:
                if param1 not in self.hyperparameter_ranges or param2 not in self.hyperparameter_ranges:
                    self.logger.warning(f"Missing range for parameter pair: ({param1}, {param2})")
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
                            result = self._run_sgfa_analysis(X_list, test_hypers, args, **kwargs)
                            pair_results[(val1, val2)] = result
                            
                            # Store performance metrics
                            metrics = self.profiler.get_current_metrics()
                            pair_metrics[(val1, val2)] = {
                                'execution_time': metrics.execution_time,
                                'peak_memory_gb': metrics.peak_memory_gb,
                                'convergence': result.get('convergence', False),
                                'log_likelihood': result.get('log_likelihood', np.nan)
                            }
                            
                        except Exception as e:
                            self.logger.warning(f"Failed for {param_key}: {str(e)}")
                            pair_results[(val1, val2)] = {'error': str(e)}
                            pair_metrics[(val1, val2)] = {
                                'execution_time': np.nan,
                                'peak_memory_gb': np.nan,
                                'convergence': False,
                                'log_likelihood': np.nan
                            }
                
                results[f"{param1}_vs_{param2}"] = {
                    'results': pair_results,
                    'param1': param1,
                    'param2': param2,
                    'range1': range1,
                    'range2': range2
                }
                performance_metrics[f"{param1}_vs_{param2}"] = pair_metrics
            
            # Analyze multivariate sensitivity
            analysis = self._analyze_multivariate_sensitivity(results, performance_metrics)
            
            # Generate plots
            plots = self._plot_multivariate_sensitivity(results, performance_metrics)
            
            return ExperimentResult(
                experiment_name="multivariate_sensitivity_analysis",
                config=self.config,
                data=results,
                analysis=analysis,
                plots=plots,
                performance_metrics=performance_metrics,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Multivariate sensitivity analysis failed: {str(e)}")
            return self._create_failure_result("multivariate_sensitivity_analysis", str(e))
    
    def run_gradient_based_sensitivity(self, X_list: List[np.ndarray],
                                     base_hypers: Dict, args: Dict,
                                     epsilon: float = 0.01,
                                     **kwargs) -> ExperimentResult:
        """Run gradient-based sensitivity analysis using finite differences."""
        self.logger.info("Running gradient-based sensitivity analysis")
        
        results = {}
        gradients = {}
        
        try:
            # Get baseline result
            baseline_result = self._run_sgfa_analysis(X_list, base_hypers, args, **kwargs)
            baseline_likelihood = baseline_result.get('log_likelihood', np.nan)
            
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
                    forward_result = self._run_sgfa_analysis(X_list, forward_hypers, args, **kwargs)
                    forward_likelihood = forward_result.get('log_likelihood', np.nan)
                    
                    # Backward difference  
                    backward_hypers = base_hypers.copy()
                    backward_hypers[param_name] = base_value * (1 - epsilon)
                    
                    backward_result = self._run_sgfa_analysis(X_list, backward_hypers, args, **kwargs)
                    backward_likelihood = backward_result.get('log_likelihood', np.nan)
                    
                    # Calculate gradient
                    if not (np.isnan(forward_likelihood) or np.isnan(backward_likelihood)):
                        gradient = (forward_likelihood - backward_likelihood) / (2 * epsilon * base_value)
                        
                        gradients[param_name] = {
                            'gradient': gradient,
                            'forward_likelihood': forward_likelihood,
                            'backward_likelihood': backward_likelihood,
                            'baseline_likelihood': baseline_likelihood,
                            'relative_sensitivity': abs(gradient * base_value / baseline_likelihood)
                        }
                    else:
                        gradients[param_name] = {'gradient': np.nan, 'error': 'Failed to compute differences'}
                        
                except Exception as e:
                    self.logger.warning(f"Failed to compute gradient for {param_name}: {str(e)}")
                    gradients[param_name] = {'gradient': np.nan, 'error': str(e)}
            
            results['baseline'] = baseline_result
            results['gradients'] = gradients
            
            # Analyze gradient-based sensitivity
            analysis = self._analyze_gradient_sensitivity(gradients)
            
            # Generate plots
            plots = self._plot_gradient_sensitivity(gradients)
            
            return ExperimentResult(
                experiment_name="gradient_based_sensitivity",
                config=self.config,
                data=results,
                analysis=analysis,
                plots=plots,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Gradient-based sensitivity analysis failed: {str(e)}")
            return self._create_failure_result("gradient_based_sensitivity", str(e))
    
    def run_robustness_analysis(self, X_list: List[np.ndarray],
                              base_hypers: Dict, args: Dict,
                              noise_levels: List[float] = None,
                              n_trials: int = 10,
                              **kwargs) -> ExperimentResult:
        """Run robustness analysis with hyperparameter perturbations."""
        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
            
        self.logger.info(f"Running robustness analysis with noise levels: {noise_levels}")
        
        results = {}
        
        try:
            # Baseline result
            baseline_result = self._run_sgfa_analysis(X_list, base_hypers, args, **kwargs)
            results['baseline'] = baseline_result
            
            # Robustness testing
            for noise_level in noise_levels:
                self.logger.info(f"Testing robustness with noise level: {noise_level}")
                
                noise_results = []
                
                for trial in range(n_trials):
                    # Add noise to hyperparameters
                    noisy_hypers = self._add_hyperparameter_noise(base_hypers, noise_level)
                    
                    try:
                        result = self._run_sgfa_analysis(X_list, noisy_hypers, args, **kwargs)
                        noise_results.append({
                            'trial': trial,
                            'noisy_hypers': noisy_hypers,
                            'result': result,
                            'log_likelihood': result.get('log_likelihood', np.nan),
                            'convergence': result.get('convergence', False)
                        })
                        
                    except Exception as e:
                        noise_results.append({
                            'trial': trial,
                            'noisy_hypers': noisy_hypers,
                            'result': {'error': str(e)},
                            'log_likelihood': np.nan,
                            'convergence': False
                        })
                
                results[f'noise_{noise_level}'] = noise_results
            
            # Analyze robustness
            analysis = self._analyze_robustness(results, baseline_result)
            
            # Generate plots
            plots = self._plot_robustness_analysis(results)
            
            return ExperimentResult(
                experiment_name="robustness_analysis",
                config=self.config,
                data=results,
                analysis=analysis,
                plots=plots,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Robustness analysis failed: {str(e)}")
            return self._create_failure_result("robustness_analysis", str(e))
    
    def _run_sgfa_analysis(self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs) -> Dict:
        """Run SGFA analysis with given hyperparameters."""
        # This would call your actual SGFA implementation
        # For now, return mock results based on hyperparameters
        
        # Simulate some realistic behavior
        K = hypers.get('K', 5)
        alpha_w = hypers.get('alpha_w', 1.0)
        alpha_z = hypers.get('alpha_z', 1.0)
        
        # Mock log likelihood that depends on hyperparameters
        base_likelihood = -1000.0
        likelihood_adjustment = (
            -np.log(alpha_w) * 10 - 
            np.log(alpha_z) * 5 - 
            K * 2 +
            np.random.randn() * 50  # Add some noise
        )
        
        return {
            'W': [np.random.randn(X.shape[1], K) for X in X_list],
            'Z': np.random.randn(X_list[0].shape[0], K),
            'log_likelihood': base_likelihood + likelihood_adjustment,
            'n_iterations': np.random.randint(50, 500),
            'convergence': np.random.random() > 0.1,  # 90% convergence rate
            'hyperparameters': hypers.copy()
        }
    
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
                if param_name == 'K':
                    noisy_value = max(1, int(round(noisy_value)))
                    
                noisy_hypers[param_name] = noisy_value
            else:
                noisy_hypers[param_name] = param_value
        
        return noisy_hypers
    
    def _analyze_univariate_sensitivity(self, results: Dict, performance_metrics: Dict) -> Dict:
        """Analyze univariate sensitivity results."""
        analysis = {
            'sensitivity_ranking': {},
            'optimal_values': {},
            'stability_assessment': {},
            'performance_impact': {}
        }
        
        sensitivity_scores = {}
        
        for param_name, param_results in results.items():
            if not param_results:
                continue
                
            param_metrics = performance_metrics[param_name]
            
            # Extract log likelihoods
            likelihoods = []
            param_values = []
            
            for param_value, result in param_results.items():
                if 'error' not in result:
                    likelihood = param_metrics[param_value]['log_likelihood']
                    if not np.isnan(likelihood):
                        likelihoods.append(likelihood)
                        param_values.append(param_value)
            
            if len(likelihoods) > 1:
                # Calculate sensitivity as range of log likelihoods
                likelihood_range = max(likelihoods) - min(likelihoods)
                sensitivity_scores[param_name] = likelihood_range
                
                # Find optimal value
                best_idx = np.argmax(likelihoods)
                optimal_value = param_values[best_idx]
                
                analysis['optimal_values'][param_name] = {
                    'value': optimal_value,
                    'log_likelihood': likelihoods[best_idx]
                }
                
                # Assess stability (coefficient of variation)
                cv = np.std(likelihoods) / np.abs(np.mean(likelihoods))
                analysis['stability_assessment'][param_name] = {
                    'coefficient_of_variation': cv,
                    'stability_level': 'high' if cv < 0.1 else 'medium' if cv < 0.3 else 'low'
                }
                
                # Performance impact
                execution_times = [param_metrics[pv]['execution_time'] for pv in param_values]
                memory_usages = [param_metrics[pv]['peak_memory_gb'] for pv in param_values]
                
                analysis['performance_impact'][param_name] = {
                    'time_range': max(execution_times) - min(execution_times),
                    'memory_range': max(memory_usages) - min(memory_usages)
                }
        
        # Rank parameters by sensitivity
        sorted_sensitivity = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
        analysis['sensitivity_ranking'] = {
            'most_sensitive': [name for name, score in sorted_sensitivity],
            'sensitivity_scores': sensitivity_scores
        }
        
        return analysis
    
    def _analyze_multivariate_sensitivity(self, results: Dict, performance_metrics: Dict) -> Dict:
        """Analyze multivariate sensitivity results."""
        analysis = {
            'interaction_effects': {},
            'optimal_combinations': {},
            'interaction_strength': {}
        }
        
        for pair_name, pair_data in results.items():
            param1 = pair_data['param1']
            param2 = pair_data['param2']
            pair_results = pair_data['results']
            pair_metrics = performance_metrics[pair_name]
            
            # Extract log likelihoods in matrix form
            range1 = pair_data['range1']
            range2 = pair_data['range2']
            
            likelihood_matrix = np.full((len(range1), len(range2)), np.nan)
            
            for i, val1 in enumerate(range1):
                for j, val2 in enumerate(range2):
                    if (val1, val2) in pair_metrics:
                        likelihood = pair_metrics[(val1, val2)]['log_likelihood']
                        if not np.isnan(likelihood):
                            likelihood_matrix[i, j] = likelihood
            
            # Find optimal combination
            if not np.all(np.isnan(likelihood_matrix)):
                best_idx = np.unravel_index(np.nanargmax(likelihood_matrix), likelihood_matrix.shape)
                optimal_val1 = range1[best_idx[0]]
                optimal_val2 = range2[best_idx[1]]
                optimal_likelihood = likelihood_matrix[best_idx]
                
                analysis['optimal_combinations'][pair_name] = {
                    param1: optimal_val1,
                    param2: optimal_val2,
                    'log_likelihood': optimal_likelihood
                }
                
                # Assess interaction strength
                # Compare to sum of individual effects (approximation)
                main_effect_strength = np.nanstd(np.nanmean(likelihood_matrix, axis=1)) + \
                                     np.nanstd(np.nanmean(likelihood_matrix, axis=0))
                total_variation = np.nanstd(likelihood_matrix.flatten())
                
                interaction_ratio = total_variation / (main_effect_strength + 1e-10)
                
                analysis['interaction_strength'][pair_name] = {
                    'interaction_ratio': interaction_ratio,
                    'interaction_level': 'strong' if interaction_ratio > 1.5 else 'moderate' if interaction_ratio > 1.1 else 'weak'
                }
        
        return analysis
    
    def _analyze_gradient_sensitivity(self, gradients: Dict) -> Dict:
        """Analyze gradient-based sensitivity results."""
        analysis = {
            'gradient_magnitudes': {},
            'sensitivity_ranking': {},
            'relative_importance': {}
        }
        
        valid_gradients = {}
        
        for param_name, gradient_data in gradients.items():
            if 'gradient' in gradient_data and not np.isnan(gradient_data['gradient']):
                gradient = gradient_data['gradient']
                relative_sensitivity = gradient_data.get('relative_sensitivity', abs(gradient))
                
                valid_gradients[param_name] = gradient
                
                analysis['gradient_magnitudes'][param_name] = {
                    'gradient': gradient,
                    'absolute_gradient': abs(gradient),
                    'relative_sensitivity': relative_sensitivity
                }
        
        if valid_gradients:
            # Rank by absolute gradient
            sorted_gradients = sorted(valid_gradients.items(), key=lambda x: abs(x[1]), reverse=True)
            analysis['sensitivity_ranking']['most_sensitive'] = [name for name, grad in sorted_gradients]
            
            # Relative importance
            total_abs_gradient = sum(abs(grad) for grad in valid_gradients.values())
            for param_name, gradient in valid_gradients.items():
                analysis['relative_importance'][param_name] = abs(gradient) / total_abs_gradient
        
        return analysis
    
    def _analyze_robustness(self, results: Dict, baseline_result: Dict) -> Dict:
        """Analyze robustness results."""
        analysis = {
            'robustness_metrics': {},
            'convergence_rates': {},
            'likelihood_stability': {}
        }
        
        baseline_likelihood = baseline_result.get('log_likelihood', np.nan)
        
        for noise_key, noise_results in results.items():
            if noise_key == 'baseline':
                continue
                
            noise_level = float(noise_key.split('_')[1])
            
            # Extract metrics
            likelihoods = [r['log_likelihood'] for r in noise_results if not np.isnan(r['log_likelihood'])]
            convergence_count = sum(1 for r in noise_results if r['convergence'])
            
            if likelihoods:
                likelihood_mean = np.mean(likelihoods)
                likelihood_std = np.std(likelihoods)
                
                # Robustness metrics
                analysis['robustness_metrics'][noise_level] = {
                    'mean_likelihood': likelihood_mean,
                    'std_likelihood': likelihood_std,
                    'coefficient_of_variation': likelihood_std / abs(likelihood_mean),
                    'likelihood_drop': baseline_likelihood - likelihood_mean if not np.isnan(baseline_likelihood) else np.nan
                }
                
                # Likelihood stability
                analysis['likelihood_stability'][noise_level] = {
                    'stability_score': 1.0 / (1.0 + likelihood_std),
                    'relative_stability': likelihood_std / abs(baseline_likelihood) if not np.isnan(baseline_likelihood) else np.nan
                }
            
            # Convergence rates
            convergence_rate = convergence_count / len(noise_results)
            analysis['convergence_rates'][noise_level] = convergence_rate
        
        return analysis
    
    def _plot_univariate_sensitivity(self, results: Dict, performance_metrics: Dict) -> Dict:
        """Generate plots for univariate sensitivity analysis."""
        plots = {}
        
        try:
            n_params = len(results)
            if n_params == 0:
                return plots
                
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Univariate Sensitivity Analysis', fontsize=16)
            
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
                    if 'error' not in result:
                        likelihood = param_metrics[param_value]['log_likelihood']
                        if not np.isnan(likelihood):
                            param_values.append(param_value)
                            likelihoods.append(likelihood)
                
                if param_values:
                    ax1.plot(param_values, likelihoods, 'o-', label=param_name, color=colors[i])
            
            ax1.set_xlabel('Parameter Value')
            ax1.set_ylabel('Log Likelihood')
            ax1.set_title('Log Likelihood vs Parameter Values')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Execution time vs parameter values
            ax2 = axes_flat[1]
            for i, param_name in enumerate(param_names):
                param_metrics = performance_metrics[param_name]
                
                param_values = []
                times = []
                
                for param_value, metrics in param_metrics.items():
                    if not np.isnan(metrics['execution_time']):
                        param_values.append(param_value)
                        times.append(metrics['execution_time'])
                
                if param_values:
                    ax2.plot(param_values, times, 's-', label=param_name, color=colors[i])
            
            ax2.set_xlabel('Parameter Value')
            ax2.set_ylabel('Execution Time (seconds)')
            ax2.set_title('Execution Time vs Parameter Values')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Sensitivity ranking (bar plot)
            ax3 = axes_flat[2]
            sensitivity_scores = {}
            
            for param_name, param_results in results.items():
                param_metrics = performance_metrics[param_name]
                likelihoods = [param_metrics[pv]['log_likelihood'] for pv in param_results.keys()
                             if 'error' not in param_results[pv] and not np.isnan(param_metrics[pv]['log_likelihood'])]
                
                if len(likelihoods) > 1:
                    sensitivity_scores[param_name] = max(likelihoods) - min(likelihoods)
            
            if sensitivity_scores:
                param_names_sorted = sorted(sensitivity_scores.keys(), key=lambda x: sensitivity_scores[x], reverse=True)
                scores_sorted = [sensitivity_scores[name] for name in param_names_sorted]
                
                ax3.bar(param_names_sorted, scores_sorted)
                ax3.set_ylabel('Likelihood Range')
                ax3.set_title('Parameter Sensitivity Ranking')
                ax3.tick_params(axis='x', rotation=45)
            
            # Plot 4: Memory usage vs parameter values
            ax4 = axes_flat[3]
            for i, param_name in enumerate(param_names):
                param_metrics = performance_metrics[param_name]
                
                param_values = []
                memory = []
                
                for param_value, metrics in param_metrics.items():
                    if not np.isnan(metrics['peak_memory_gb']):
                        param_values.append(param_value)
                        memory.append(metrics['peak_memory_gb'])
                
                if param_values:
                    ax4.plot(param_values, memory, '^-', label=param_name, color=colors[i])
            
            ax4.set_xlabel('Parameter Value')
            ax4.set_ylabel('Peak Memory (GB)')
            ax4.set_title('Memory Usage vs Parameter Values')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plots['univariate_sensitivity'] = fig
            
        except Exception as e:
            self.logger.warning(f"Failed to create univariate sensitivity plots: {str(e)}")
            
        return plots
    
    def _plot_multivariate_sensitivity(self, results: Dict, performance_metrics: Dict) -> Dict:
        """Generate plots for multivariate sensitivity analysis."""
        plots = {}
        
        try:
            n_pairs = len(results)
            if n_pairs == 0:
                return plots
            
            # Create heatmaps for each parameter pair
            for pair_name, pair_data in results.items():
                param1 = pair_data['param1']
                param2 = pair_data['param2']
                range1 = pair_data['range1']
                range2 = pair_data['range2']
                pair_metrics = performance_metrics[pair_name]
                
                # Create likelihood matrix
                likelihood_matrix = np.full((len(range1), len(range2)), np.nan)
                
                for i, val1 in enumerate(range1):
                    for j, val2 in enumerate(range2):
                        if (val1, val2) in pair_metrics:
                            likelihood = pair_metrics[(val1, val2)]['log_likelihood']
                            if not np.isnan(likelihood):
                                likelihood_matrix[i, j] = likelihood
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(8, 6))
                
                im = ax.imshow(likelihood_matrix, cmap='viridis', aspect='auto')
                
                # Set ticks and labels
                ax.set_xticks(range(len(range2)))
                ax.set_yticks(range(len(range1)))
                ax.set_xticklabels([f"{val:.3f}" for val in range2])
                ax.set_yticklabels([f"{val:.3f}" for val in range1])
                
                ax.set_xlabel(param2)
                ax.set_ylabel(param1)
                ax.set_title(f'Log Likelihood Heatmap: {param1} vs {param2}')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, label='Log Likelihood')
                
                # Mark optimal point
                if not np.all(np.isnan(likelihood_matrix)):
                    best_idx = np.unravel_index(np.nanargmax(likelihood_matrix), likelihood_matrix.shape)
                    ax.plot(best_idx[1], best_idx[0], 'r*', markersize=15, label='Optimal')
                    ax.legend()
                
                plt.tight_layout()
                plots[f'heatmap_{pair_name}'] = fig
                
        except Exception as e:
            self.logger.warning(f"Failed to create multivariate sensitivity plots: {str(e)}")
            
        return plots
    
    def _plot_gradient_sensitivity(self, gradients: Dict) -> Dict:
        """Generate plots for gradient-based sensitivity analysis."""
        plots = {}
        
        try:
            # Filter valid gradients
            valid_gradients = {name: data for name, data in gradients.items()
                             if 'gradient' in data and not np.isnan(data['gradient'])}
            
            if not valid_gradients:
                return plots
            
            # Create gradient plots
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            param_names = list(valid_gradients.keys())
            gradient_values = [valid_gradients[name]['gradient'] for name in param_names]
            absolute_gradients = [abs(g) for g in gradient_values]
            
            # Plot 1: Gradient values
            axes[0].bar(param_names, gradient_values)
            axes[0].set_ylabel('Gradient (∂LL/∂θ)')
            axes[0].set_title('Gradient-Based Sensitivity')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            # Plot 2: Absolute gradients (sensitivity ranking)
            sorted_indices = np.argsort(absolute_gradients)[::-1]
            sorted_names = [param_names[i] for i in sorted_indices]
            sorted_abs_gradients = [absolute_gradients[i] for i in sorted_indices]
            
            axes[1].bar(sorted_names, sorted_abs_gradients)
            axes[1].set_ylabel('|Gradient|')
            axes[1].set_title('Sensitivity Ranking')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plots['gradient_sensitivity'] = fig
            
        except Exception as e:
            self.logger.warning(f"Failed to create gradient sensitivity plots: {str(e)}")
            
        return plots
    
    def _plot_robustness_analysis(self, results: Dict) -> Dict:
        """Generate plots for robustness analysis."""
        plots = {}
        
        try:
            # Extract noise levels and metrics
            noise_levels = []
            mean_likelihoods = []
            std_likelihoods = []
            convergence_rates = []
            
            for noise_key, noise_results in results.items():
                if noise_key == 'baseline':
                    continue
                    
                noise_level = float(noise_key.split('_')[1])
                noise_levels.append(noise_level)
                
                likelihoods = [r['log_likelihood'] for r in noise_results if not np.isnan(r['log_likelihood'])]
                convergences = [r['convergence'] for r in noise_results]
                
                mean_likelihoods.append(np.mean(likelihoods) if likelihoods else np.nan)
                std_likelihoods.append(np.std(likelihoods) if likelihoods else np.nan)
                convergence_rates.append(np.mean(convergences))
            
            if not noise_levels:
                return plots
            
            # Create robustness plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Robustness Analysis', fontsize=16)
            
            # Plot 1: Mean likelihood vs noise level
            axes[0, 0].plot(noise_levels, mean_likelihoods, 'o-')
            axes[0, 0].set_xlabel('Noise Level')
            axes[0, 0].set_ylabel('Mean Log Likelihood')
            axes[0, 0].set_title('Likelihood vs Noise Level')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Likelihood standard deviation vs noise level
            axes[0, 1].plot(noise_levels, std_likelihoods, 's-', color='orange')
            axes[0, 1].set_xlabel('Noise Level')
            axes[0, 1].set_ylabel('Likelihood Standard Deviation')
            axes[0, 1].set_title('Likelihood Variability vs Noise Level')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Convergence rate vs noise level
            axes[1, 0].plot(noise_levels, convergence_rates, '^-', color='green')
            axes[1, 0].set_xlabel('Noise Level')
            axes[1, 0].set_ylabel('Convergence Rate')
            axes[1, 0].set_title('Convergence Rate vs Noise Level')
            axes[1, 0].set_ylim([0, 1.1])
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Likelihood distribution for highest noise level
            if noise_levels:
                highest_noise_key = f"noise_{max(noise_levels)}"
                highest_noise_results = results[highest_noise_key]
                highest_noise_likelihoods = [r['log_likelihood'] for r in highest_noise_results 
                                           if not np.isnan(r['log_likelihood'])]
                
                if highest_noise_likelihoods:
                    axes[1, 1].hist(highest_noise_likelihoods, bins=10, alpha=0.7, color='red')
                    axes[1, 1].set_xlabel('Log Likelihood')
                    axes[1, 1].set_ylabel('Frequency')
                    axes[1, 1].set_title(f'Likelihood Distribution (Noise={max(noise_levels)})')
                    
                    # Add baseline line if available
                    baseline_likelihood = results['baseline'].get('log_likelihood')
                    if not np.isnan(baseline_likelihood):
                        axes[1, 1].axvline(baseline_likelihood, color='black', linestyle='--', 
                                         label='Baseline')
                        axes[1, 1].legend()
            
            plt.tight_layout()
            plots['robustness_analysis'] = fig
            
        except Exception as e:
            self.logger.warning(f"Failed to create robustness analysis plots: {str(e)}")
            
        return plots